from __future__ import annotations

import json
from pathlib import Path

import tvm
from tvm import auto_scheduler

from asfuzz.lowering.to_te import lower_to_te
from asfuzz.schedulers.base import CompiledArtifact
from asfuzz.spec.opspec import OpSpec
from .tvm_backend import TVMBackend


@auto_scheduler.register_workload
def asfuzz_ansor_workload(spec_json: str):
    spec = OpSpec.from_dict(json.loads(spec_json))
    inputs, output = lower_to_te(spec)
    return [*inputs, output]


class AnsorBackend(TVMBackend):
    name = "ansor"
    schedule_policy = "ansor"

    def schedule_and_build(self, spec: OpSpec, target: str, trials: int, seed: int) -> CompiledArtifact:
        if trials <= 0:
            return super().schedule_and_build(spec, target, trials, seed)

        spec_json = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
        task = auto_scheduler.SearchTask(func=asfuzz_ansor_workload, args=(spec_json,), target=target)
        work_dir = Path("runs") / "tuning" / "ansor" / spec.signature() / f"seed_{seed}_trials_{trials}"
        work_dir.mkdir(parents=True, exist_ok=True)
        log_file = work_dir / "records.json"
        tune_options = auto_scheduler.TuningOptions(
            num_measure_trials=int(trials),
            num_measures_per_round=max(1, min(16, int(trials))),
            verbose=0,
            runner=auto_scheduler.LocalRunner(number=1, repeat=1, min_repeat_ms=0, timeout=10),
            measure_callbacks=[auto_scheduler.RecordToFile(str(log_file))],
        )
        cost_model = _make_cost_model(seed)
        policy = auto_scheduler.SketchPolicy(
            task,
            program_cost_model=cost_model,
            seed=seed,
            verbose=0,
        )
        task.tune(tune_options, search_policy=policy)
        try:
            schedule, args = task.apply_best(str(log_file))
        except Exception:
            return super().schedule_and_build(spec, target, 0, seed)
        module = tvm.build(schedule, args, target=target, name=f"asfuzz_ansor_{spec.signature()}")
        return CompiledArtifact(self.name, spec, target, trials, seed, handle=(module, args))


def _make_cost_model(seed: int):
    try:
        return auto_scheduler.XGBModel(seed=seed, adaptive_training=False)
    except Exception:
        return auto_scheduler.RandomModel()
