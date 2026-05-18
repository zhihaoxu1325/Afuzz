from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from asfuzz.spec.opspec import OpSpec


@dataclass
class CompiledArtifact:
    backend: str
    spec: OpSpec
    target: str
    trials: int
    seed: int
    handle: Any


class SchedulerBackend(ABC):
    name: str

    @abstractmethod
    def supports(self, spec: OpSpec) -> bool:
        raise NotImplementedError

    @abstractmethod
    def schedule_and_build(self, spec: OpSpec, target: str, trials: int, seed: int) -> CompiledArtifact:
        raise NotImplementedError

    @abstractmethod
    def run(self, artifact: CompiledArtifact, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError

