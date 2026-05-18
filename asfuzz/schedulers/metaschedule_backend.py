from __future__ import annotations

from pathlib import Path

import tvm

from asfuzz.lowering.to_tir import lower_to_tir
from asfuzz.schedulers.base import CompiledArtifact
from asfuzz.spec.opspec import OpSpec
from .tvm_backend import TVMBackend


try:
    import tvm.s_tir.meta_schedule as ms

    _MS_FLAVOR = "s_tir"
except ImportError:
    import tvm.meta_schedule as ms

    _MS_FLAVOR = "tir"


class MetaScheduleBackend(TVMBackend):
    name = "metaschedule"
    schedule_policy = "metaschedule"

    def schedule_and_build(self, spec: OpSpec, target: str, trials: int, seed: int) -> CompiledArtifact:
        if trials <= 0:
            return super().schedule_and_build(spec, target, trials, seed)

        prim = lower_to_tir(spec)
        mod = tvm.IRModule({"main": prim})
        work_dir = Path("runs") / "tuning" / "metaschedule" / spec.signature() / f"seed_{seed}_trials_{trials}"
        work_dir.mkdir(parents=True, exist_ok=True)
        tune_target = _normalize_cpu_target(target)
        try:
            database = ms.tune_tir(
                mod=mod,
                target=tune_target,
                work_dir=str(work_dir),
                max_trials_global=int(trials),
                max_trials_per_task=int(trials),
                num_trials_per_iter=max(1, min(16, int(trials))),
                runner="local",
                builder="local",
                database="json",
                cost_model="random",
                strategy="replay-trace",
                seed=seed,
                num_tuning_cores=1,
            )
            sch = ms.tir_integration.compile_tir(database, mod, tune_target)
        except Exception:
            return super().schedule_and_build(spec, target, 0, seed)
        if sch is None:
            return super().schedule_and_build(spec, target, 0, seed)
        build_mod = _schedule_module(sch)
        try:
            module = tvm.build(build_mod, target=tune_target, name=f"asfuzz_ms_{spec.signature()}")
        except TypeError:
            module = tvm.build(build_mod, target=tune_target)
        return CompiledArtifact(self.name, spec, target, trials, seed, handle=(module, []))


def _normalize_cpu_target(target: str) -> str:
    if target.strip() == "llvm":
        return "llvm -num-cores 1"
    if target.startswith("llvm") and "num-cores" not in target:
        return f"{target} -num-cores 1"
    return target


def _schedule_module(schedule):
    if _MS_FLAVOR == "s_tir":
        return schedule.mod
    return schedule.mod
