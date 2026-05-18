from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from asfuzz.lowering.to_halide import lower_to_halide
from asfuzz.schedulers.base import CompiledArtifact, SchedulerBackend
from asfuzz.spec.opspec import OpSpec


class HalideBackend(SchedulerBackend):
    name = "halide"

    def supports(self, spec: OpSpec) -> bool:
        try:
            import halide  # noqa: F401
        except Exception:
            return False
        if spec.dtype() != "float32":
            return False
        if spec.op_kind in {"softmax", "softmax_decomposed"}:
            return int(spec.extra.get("axis", len(spec.shape_of("A")) - 1)) == len(spec.shape_of("A")) - 1
        return spec.op_kind in {"matmul", "elementwise", "unary", "reduce"}

    def schedule_and_build(self, spec: OpSpec, target: str, trials: int, seed: int) -> CompiledArtifact:
        import halide as hl

        lowered = lower_to_halide(spec)
        scheduler = os.environ.get("ASFUZZ_HALIDE_AUTOSCHEDULER", "Mullapudi2016")
        plugin = _find_autoscheduler_plugin(scheduler)
        if plugin is not None:
            hl.load_plugin(str(plugin))
        params = hl.AutoschedulerParams()
        params.name = scheduler
        # Keep stochastic autoschedulers reproducible when they expose seed knobs.
        params.extra["random_dropout_seed"] = str(seed)
        params.extra["parallelism"] = os.environ.get("ASFUZZ_HALIDE_PARALLELISM", "1")
        result = lowered.pipeline.apply_autoscheduler(hl.get_host_target(), params)
        return CompiledArtifact(
            self.name,
            spec,
            target,
            trials,
            seed,
            handle={"lowered": lowered, "schedule_source": getattr(result, "schedule_source", "")},
        )

    def run(self, artifact: CompiledArtifact, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import halide as hl

        lowered = artifact.handle["lowered"]
        live_arrays = []
        live_buffers = []
        for name, param in lowered.inputs.items():
            arr = np.ascontiguousarray(inputs[name].astype("float32", copy=False))
            buf = hl.Buffer(arr)
            live_arrays.append(arr)
            live_buffers.append(buf)
            param.set(buf)
        output = lowered.pipeline.realize(lowered.realize_shape)
        arr = np.array(output, copy=True).reshape(lowered.output_shape)
        return {lowered.output_name: arr.astype(artifact.spec.dtype(), copy=False)}


def _find_autoscheduler_plugin(name: str) -> Path | None:
    import halide as hl

    root = Path(hl.__file__).resolve().parent / "lib64"
    mapping = {
        "Mullapudi2016": "libautoschedule_mullapudi2016.so",
        "Adams2019": "libautoschedule_adams2019.so",
        "Li2018": "libautoschedule_li2018.so",
        "Anderson2021": "libautoschedule_anderson2021.so",
    }
    filename = mapping.get(name)
    if filename is None:
        return None
    path = root / filename
    return path if path.exists() else None
