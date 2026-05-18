from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation, identity
from asfuzz.spec.opspec import OpSpec


class AlgebraicMR(MetamorphicRelation):
    name = "algebraic"

    def applicable(self, spec: OpSpec) -> bool:
        return spec.op_kind == "unary" and spec.extra.get("op") == "relu"

    def variants(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        new_inputs = {k: np.array(v, copy=True) for k, v in inputs.items()}
        new_inputs["A"] = np.maximum(new_inputs["A"], 0)
        return [MRCase(copy.deepcopy(spec), new_inputs, identity, seed, 0, "relu_relu")]

