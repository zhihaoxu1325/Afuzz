from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation, identity
from asfuzz.spec.opspec import OpSpec


class DecompositionMR(MetamorphicRelation):
    name = "decomposition"

    def applicable(self, spec: OpSpec) -> bool:
        return spec.op_kind == "softmax"

    def variants(self, spec: OpSpec, inputs, seed: int):
        rng = np.random.default_rng(seed)
        axis = int(spec.extra.get("axis", len(inputs["A"].shape) - 1))
        shift_shape = list(inputs["A"].shape)
        shift_shape[axis] = 1
        shifted = {k: np.array(v, copy=True) for k, v in inputs.items()}
        shifted["A"] = shifted["A"] + rng.normal(0, 3, size=shift_shape).astype("float32")
        decomposed = copy.deepcopy(spec)
        decomposed.op_kind = "softmax_decomposed"
        decomposed.name = f"{spec.name}_decomposed"
        return [
            MRCase(copy.deepcopy(spec), shifted, identity, seed, 0, "softmax_shift_invariance"),
            MRCase(decomposed, {k: np.array(v, copy=True) for k, v in inputs.items()}, identity, seed, 0, "softmax_explicit_decomposition"),
        ]
