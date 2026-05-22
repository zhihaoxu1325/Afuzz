from __future__ import annotations

import numpy as np

from asfuzz.schedulers.base import CompiledArtifact, SchedulerBackend
from asfuzz.spec.ops_catalog import SUPPORTED_OP_KINDS
from asfuzz.spec.opspec import OpSpec
from asfuzz.spec.reference import run_reference


class NumpyBackend(SchedulerBackend):
    name = "numpy"

    def supports(self, spec: OpSpec) -> bool:
        return spec.op_kind in (SUPPORTED_OP_KINDS | {"softmax_decomposed", "reduce_split"})

    def schedule_and_build(self, spec: OpSpec, target: str, trials: int, seed: int) -> CompiledArtifact:
        return CompiledArtifact(self.name, spec, target, trials, seed, handle=None)

    def run(self, artifact: CompiledArtifact, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return run_reference(artifact.spec, inputs)
