from __future__ import annotations

import copy
import random

from asfuzz.spec.opspec import OpSpec


class SpecMutator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def mutate_shape(self, spec: OpSpec) -> OpSpec:
        mutated = copy.deepcopy(spec)
        axis = self.rng.choice(list(mutated.axes.values()))
        axis.size = max(1, self.rng.choice([axis.size + 1, axis.size * 2, 1, 7, 31]))
        mutated.name = f"{spec.name}_mut"
        return mutated

