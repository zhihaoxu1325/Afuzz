from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from asfuzz.spec.opspec import OpSpec


RecoverFn = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def identity(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return outputs


@dataclass
class MRCase:
    spec: OpSpec
    inputs: dict[str, np.ndarray]
    recover_fn: RecoverFn
    seed: int
    trials: int
    tag: str
    expected_recover_fn: RecoverFn = identity


class MetamorphicRelation(ABC):
    name: str

    @abstractmethod
    def applicable(self, spec: OpSpec) -> bool:
        raise NotImplementedError

    @abstractmethod
    def variants(self, spec: OpSpec, inputs: dict[str, np.ndarray], seed: int) -> list[MRCase]:
        raise NotImplementedError
