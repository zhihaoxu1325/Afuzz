from __future__ import annotations

import os
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from asfuzz.config import ASFuzzConfig
from asfuzz.fuzzer.coverage import CoverageTracker, complexity_score, work_items
from asfuzz.fuzzer.grammar import GrammarFuzzer
from asfuzz.mr.registry import make_mr
from asfuzz.oracle.numeric import numeric_equal
from asfuzz.reducer.repro import write_repro
from asfuzz.reporter.bug_db import BugDB
from asfuzz.schedulers.registry import make_backend
from asfuzz.spec.opspec import OpSpec
from asfuzz.spec.reference import sample_inputs
from asfuzz.spec.validate import validate
from .worker import WorkerTimeout, run_backend_once


def run_campaign(cfg: ASFuzzConfig, resume: bool = False) -> dict:
    out_dir = Path(cfg.out_dir)
    if out_dir.exists() and not resume:
        shutil.rmtree(out_dir)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    db = BugDB(cfg.db_path)
    fuzzer = GrammarFuzzer(cfg.seed, cfg.fuzzer.op_weights, cfg.fuzzer.dtypes, cfg.fuzzer.complexity)
    backends = [make_backend(name) for name in cfg.backends]
    mrs = [make_mr(name) for name in cfg.mrs]
    coverage = CoverageTracker()
    max_workers = _resolve_max_workers(cfg)
    summary = {
        "seed": cfg.seed,
        "iterations": cfg.budget.iterations,
        "max_workers": max_workers,
        "cases": [],
        "failures": [],
        "skipped": [],
    }
    try:
        pending = {}
        results_by_index = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx in range(cfg.budget.iterations):
                completed = _drain_completed(pending, results_by_index, summary, coverage)
                for done_idx, result in completed:
                    print(f"[asfuzz] {done_idx + 1}/{cfg.budget.iterations} completed status={result['status']}", flush=True)

                while len(pending) >= max_workers:
                    future = next(as_completed(pending))
                    done_idx = pending.pop(future)
                    result = future.result()
                    results_by_index[done_idx] = result
                    _merge_result(summary, result)
                    print(f"[asfuzz] {done_idx + 1}/{cfg.budget.iterations} completed status={result['status']}", flush=True)

                case_dir = cases_dir / f"case_{idx:06d}"
                existing = case_dir / "result.json"
                if resume and existing.exists():
                    result = json.loads(existing.read_text())
                    spec_path = case_dir / "spec.json"
                    spec = OpSpec.load_json(spec_path) if spec_path.exists() else None
                    if spec is not None:
                        fuzzer.accept(spec)
                        coverage.add(spec)
                    results_by_index[idx] = result
                    _merge_result(summary, result)
                    print(f"[asfuzz] {idx + 1}/{cfg.budget.iterations} resume status={result['status']}", flush=True)
                    continue
                if resume and case_dir.exists() and not existing.exists():
                    print(f"[asfuzz] {idx + 1}/{cfg.budget.iterations} rerun incomplete {case_dir}", flush=True)
                    spec_path = case_dir / "spec.json"
                    if spec_path.exists():
                        spec = OpSpec.load_json(spec_path)
                        fuzzer.accept(spec)
                    else:
                        spec = fuzzer.sample_diverse(
                            coverage,
                            cfg.fuzzer.max_work_items,
                            candidates=cfg.fuzzer.diversity_candidates,
                            min_complexity=cfg.fuzzer.min_complexity_score,
                            complexity_weight=cfg.fuzzer.complexity_weight,
                            novelty_weight=cfg.fuzzer.novelty_weight,
                        )
                else:
                    spec = fuzzer.sample_diverse(
                        coverage,
                        cfg.fuzzer.max_work_items,
                        candidates=cfg.fuzzer.diversity_candidates,
                        min_complexity=cfg.fuzzer.min_complexity_score,
                        complexity_weight=cfg.fuzzer.complexity_weight,
                        novelty_weight=cfg.fuzzer.novelty_weight,
                    )
                coverage.add(spec)
                future = executor.submit(run_one_spec, cfg, spec, idx, cases_dir, backends, mrs, db)
                pending[future] = idx
                print(f"[asfuzz] {idx + 1}/{cfg.budget.iterations} submitted {spec.op_kind} {spec.signature()} workers={len(pending)}/{max_workers}", flush=True)

            for future in as_completed(pending):
                idx = pending[future]
                result = future.result()
                results_by_index[idx] = result
                _merge_result(summary, result)
                print(f"[asfuzz] {idx + 1}/{cfg.budget.iterations} completed status={result['status']}", flush=True)
        summary["cases"] = [results_by_index[idx] for idx in sorted(results_by_index)]
    finally:
        db.close()
    summary["coverage"] = coverage.to_dict()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _resolve_max_workers(cfg: ASFuzzConfig) -> int:
    configured = int(cfg.budget.max_workers)
    if configured > 0:
        return configured
    cpu_count = os.cpu_count() or 1
    utilization = max(0.05, min(1.0, float(cfg.budget.cpu_utilization)))
    return max(1, int(cpu_count * utilization))


def _drain_completed(pending, results_by_index, summary, coverage):
    completed = []
    for future, idx in list(pending.items()):
        if not future.done():
            continue
        pending.pop(future)
        result = future.result()
        results_by_index[idx] = result
        _merge_result(summary, result)
        completed.append((idx, result))
    return completed


def _merge_result(summary: dict, result: dict) -> None:
    if result.get("failures"):
        summary["failures"].extend(result["failures"])
    if result.get("skipped"):
        summary["skipped"].extend(result["skipped"])


def run_one_spec(cfg: ASFuzzConfig, spec: OpSpec, index: int, cases_dir: Path, backends, mrs, db: BugDB) -> dict:
    case_dir = cases_dir / f"case_{index:06d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    spec.save_json(case_dir / "spec.json")
    inputs = sample_inputs(spec, cfg.seed + index)
    np.savez(case_dir / "inputs.npz", **inputs)

    valid = validate(spec, cfg.fuzzer.max_work_items)
    if not valid.ok:
        return {
            "case": str(case_dir),
            "spec_hash": spec.signature(),
            "op_kind": spec.op_kind,
            "work_items": work_items(spec),
            "complexity_score": complexity_score(spec, cfg.fuzzer.max_work_items),
            "status": "skipped",
            "skipped": valid.reasons,
        }

    failures = []
    skipped = []
    abort_case = False
    if cfg.oracle.cross_backend:
        abort_case = _run_cross_backend_oracle(cfg, spec, inputs, case_dir, backends, db, failures, skipped)
    for backend in backends:
        if abort_case:
            skipped.append({"backend": backend.name, "reason": "case_aborted_after_timeout"})
            continue
        if not backend.supports(spec):
            skipped.append({"backend": backend.name, "reason": "unsupported"})
            continue
        for mr in mrs:
            if abort_case:
                break
            if not mr.applicable(spec):
                continue
            print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr={mr.name} start", flush=True)
            start = time.time()
            try:
                timeout_sec = cfg.budget.compile_timeout_sec + cfg.budget.run_timeout_sec
                base_outputs, base_ms = run_backend_once(backend, spec, inputs, cfg.target, 0, cfg.seed, timeout_sec)
                base_outputs = _copy_outputs(base_outputs)
                for variant in mr.variants(spec, inputs, cfg.seed):
                    print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr={mr.name} variant={variant.tag} start", flush=True)
                    out, elapsed_ms = run_backend_once(backend, variant.spec, variant.inputs, cfg.target, variant.trials, variant.seed, timeout_sec)
                    expected = variant.expected_recover_fn(base_outputs)
                    recovered = variant.recover_fn(out)
                    cmp = _compare_output_dict(expected, recovered, variant.spec, cfg)
                    result_status = "ok" if cmp["ok"] else "mismatch"
                    db.record_iteration(spec, backend.name, mr.name, result_status, elapsed_ms + base_ms)
                    if not cmp["ok"]:
                        repro_path = write_repro(case_dir, backend.name, mr.name, variant.tag)
                        detail = {"variant": variant.tag, "compare": cmp, "case": str(case_dir)}
                        failure = {
                            "case": str(case_dir),
                            "repro": str(repro_path),
                            "backend": backend.name,
                            "mr": mr.name,
                            "variant": variant.tag,
                            "status": "mismatch",
                            "detail": cmp,
                        }
                        failures.append(failure)
                        db.record_bug(spec, backend.name, mr.name, "mismatch", str(repro_path), detail)
                print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr={mr.name} done", flush=True)
            except WorkerTimeout as exc:
                elapsed_ms = (time.time() - start) * 1000.0
                db.record_iteration(spec, backend.name, mr.name, "timeout", elapsed_ms)
                repro_path = write_repro(case_dir, backend.name, mr.name, "timeout")
                detail = {
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "case": str(case_dir),
                    "triage": "case_aborted_after_first_timeout",
                }
                failures.append({"case": str(case_dir), "repro": str(repro_path), "backend": backend.name, "mr": mr.name, "status": "timeout", "detail": detail})
                db.record_bug(spec, backend.name, mr.name, "timeout", str(repro_path), detail)
                print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr={mr.name} timeout={exc}; aborting remaining checks for this case", flush=True)
                abort_case = True
            except Exception as exc:
                elapsed_ms = (time.time() - start) * 1000.0
                db.record_iteration(spec, backend.name, mr.name, "error", elapsed_ms)
                repro_path = write_repro(case_dir, backend.name, mr.name, "error")
                detail = {"exception": type(exc).__name__, "message": str(exc), "case": str(case_dir)}
                failures.append({"case": str(case_dir), "repro": str(repro_path), "backend": backend.name, "mr": mr.name, "status": "error", "detail": detail})
                db.record_bug(spec, backend.name, mr.name, "error", str(repro_path), detail)
                print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr={mr.name} error={type(exc).__name__}: {exc}", flush=True)
    result = {
        "case": str(case_dir),
        "spec_hash": spec.signature(),
        "op_kind": spec.op_kind,
        "work_items": work_items(spec),
        "complexity_score": complexity_score(spec, cfg.fuzzer.max_work_items),
        "status": "failed" if failures else "ok",
        "failures": failures,
        "skipped": skipped,
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _run_cross_backend_oracle(cfg: ASFuzzConfig, spec: OpSpec, inputs, case_dir: Path, backends, db: BugDB, failures: list, skipped: list) -> bool:
    reference_backend = ""
    reference_outputs = None
    timeout_sec = cfg.budget.compile_timeout_sec + cfg.budget.run_timeout_sec
    ordered_backends = _cross_backend_order(spec, backends)
    for backend in ordered_backends:
        if not backend.supports(spec):
            continue
        print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr=cross_backend start", flush=True)
        start = time.time()
        try:
            outputs, elapsed_ms = run_backend_once(backend, spec, inputs, cfg.target, cfg.budget.trials_smoke, cfg.seed, timeout_sec)
            outputs = _copy_outputs(outputs)
            db.record_iteration(spec, backend.name, "cross_backend", "ok", elapsed_ms)
            if reference_outputs is None:
                reference_backend = backend.name
                reference_outputs = outputs
                print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr=cross_backend reference", flush=True)
                continue
            cmp = _compare_output_dict(reference_outputs, outputs, spec, cfg)
            if not cmp["ok"]:
                repro_path = write_repro(case_dir, backend.name, "cross_backend", f"vs_{reference_backend}")
                detail = {
                    "reference_backend": reference_backend,
                    "backend": backend.name,
                    "compare": cmp,
                    "case": str(case_dir),
                }
                failures.append(
                    {
                        "case": str(case_dir),
                        "repro": str(repro_path),
                        "backend": backend.name,
                        "mr": "cross_backend",
                        "variant": f"vs_{reference_backend}",
                        "status": "mismatch",
                        "detail": detail,
                    }
                )
                db.record_bug(spec, backend.name, "cross_backend", "mismatch", str(repro_path), detail)
            print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr=cross_backend done", flush=True)
        except WorkerTimeout as exc:
            elapsed_ms = (time.time() - start) * 1000.0
            db.record_iteration(spec, backend.name, "cross_backend", "timeout", elapsed_ms)
            repro_path = write_repro(case_dir, backend.name, "cross_backend", "timeout")
            detail = {
                "exception": type(exc).__name__,
                "message": str(exc),
                "case": str(case_dir),
                "triage": "case_aborted_after_first_timeout",
            }
            failures.append({"case": str(case_dir), "repro": str(repro_path), "backend": backend.name, "mr": "cross_backend", "status": "timeout", "detail": detail})
            db.record_bug(spec, backend.name, "cross_backend", "timeout", str(repro_path), detail)
            print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr=cross_backend timeout={exc}; aborting remaining checks for this case", flush=True)
            return True
        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000.0
            db.record_iteration(spec, backend.name, "cross_backend", "error", elapsed_ms)
            repro_path = write_repro(case_dir, backend.name, "cross_backend", "error")
            detail = {"exception": type(exc).__name__, "message": str(exc), "case": str(case_dir)}
            failures.append({"case": str(case_dir), "repro": str(repro_path), "backend": backend.name, "mr": "cross_backend", "status": "error", "detail": detail})
            db.record_bug(spec, backend.name, "cross_backend", "error", str(repro_path), detail)
            print(f"[asfuzz] case={case_dir.name} backend={backend.name} mr=cross_backend error={type(exc).__name__}: {exc}", flush=True)
    if reference_outputs is None:
        skipped.append({"backend": "all", "reason": "cross_backend_no_supported_backend"})
    return False


def _cross_backend_order(spec: OpSpec, backends) -> list:
    available = {backend.name: backend for backend in backends}
    if spec.dtype() == "float16":
        # NumPy reference accumulates many ops in float64 then casts to float16,
        # while compiler backends may legally accumulate in float16 or use a
        # different reduction order. Use TVM as the fp16 compiler reference and
        # compare other compiler backends against it.
        preferred = ["tvm", "metaschedule", "halide"]
    else:
        preferred = ["numpy", "tvm", "metaschedule", "halide"]
    return [available[name] for name in preferred if name in available]


def replay_case(cfg: ASFuzzConfig, spec_path: str | Path) -> dict:
    spec = OpSpec.load_json(Path(spec_path))
    out_dir = Path(cfg.out_dir) / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    db = BugDB(cfg.db_path)
    try:
        return run_one_spec(cfg, spec, 0, out_dir, [make_backend(name) for name in cfg.backends], [make_mr(name) for name in cfg.mrs], db)
    finally:
        db.close()


def _copy_outputs(outputs):
    return {name: np.array(arr, copy=True) for name, arr in outputs.items()}


def _compare_output_dict(expected, actual, spec: OpSpec, cfg: ASFuzzConfig) -> dict:
    for output in spec.tensors_by_role("output"):
        name = output.name
        if name not in expected or name not in actual:
            return {"ok": False, "reason": f"missing output {name}"}
        eq = numeric_equal(
            expected[name],
            actual[name],
            output.dtype,
            cfg.oracle.rtol.get(output.dtype),
            cfg.oracle.atol.get(output.dtype),
        )
        if not eq.ok:
            return {
                "ok": False,
                "output": name,
                "reason": eq.reason,
                "max_abs": eq.max_abs,
                "max_rel": eq.max_rel,
                "index": eq.index,
            }
    return {"ok": True}
