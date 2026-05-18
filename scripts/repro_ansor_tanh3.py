#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tvm
from tvm import auto_scheduler, te


@auto_scheduler.register_workload
def tanh3_workload():
    a = te.placeholder((3,), name="A", dtype="float32")
    c = te.compute((3,), lambda i: te.tanh(a[i]), name="C")
    return [a, c]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure TVM/Ansor reproducer for tanh(shape=[3]) crash.")
    parser.add_argument("--target", default="llvm")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", default="runs/repros/ansor_tanh3")
    parser.add_argument("--cost-model", choices=("xgb", "random"), default="xgb")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = work_dir / "records.json"
    if log_file.exists():
        log_file.unlink()

    print(f"[repro] TVM version: {tvm.__version__}", flush=True)
    print("[repro] workload: C[i] = tanh(A[i]), A/C shape=[3], dtype=float32", flush=True)
    print(f"[repro] target={args.target} trials={args.trials} seed={args.seed}", flush=True)
    print(f"[repro] log_file={log_file}", flush=True)

    task = auto_scheduler.SearchTask(func=tanh3_workload, args=(), target=args.target)
    print(f"[repro] workload_key={task.workload_key}", flush=True)

    tune_options = auto_scheduler.TuningOptions(
        num_measure_trials=args.trials,
        num_measures_per_round=max(1, min(16, args.trials)),
        verbose=1,
        runner=auto_scheduler.LocalRunner(number=1, repeat=1, min_repeat_ms=0, timeout=10),
        measure_callbacks=[auto_scheduler.RecordToFile(str(log_file))],
    )
    if args.cost_model == "xgb":
        cost_model = auto_scheduler.XGBModel(seed=args.seed, adaptive_training=False)
    else:
        cost_model = auto_scheduler.RandomModel()

    print(f"[repro] cost_model={args.cost_model}", flush=True)
    policy = auto_scheduler.SketchPolicy(
        task,
        program_cost_model=cost_model,
        seed=args.seed,
        verbose=1,
    )

    print("[repro] start task.tune(...)", flush=True)
    task.tune(tune_options, search_policy=policy)
    print("[repro] tune finished", flush=True)

    print("[repro] start task.apply_best(...)", flush=True)
    schedule, tensors = task.apply_best(str(log_file))
    print("[repro] apply_best finished", flush=True)

    print("[repro] start tvm.build(...)", flush=True)
    module = tvm.build(schedule, tensors, target=args.target, name="repro_ansor_tanh3")
    print("[repro] build finished", flush=True)

    if not args.skip_run:
        dev = tvm.cpu(0)
        inp = tvm.nd.array(np.array([-1.0, 0.0, 1.0], dtype="float32"), dev)
        out = tvm.nd.empty((3,), dtype="float32", device=dev)
        module(inp, out)
        expected = np.tanh(inp.numpy())
        actual = out.numpy()
        print(f"[repro] actual={actual}", flush=True)
        print(f"[repro] expected={expected}", flush=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        print("[repro] output check passed", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
