from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation, identity
from asfuzz.spec.opspec import OpSpec


class InputScaleMR(MetamorphicRelation):
    name = "input_scale"

    def applicable(self, spec: OpSpec) -> bool:
        if spec.op_kind == "matmul":
            if spec.extra.get("with_bias"):
                return False
            return all(op == "relu" for op in spec.epilogue)
        if spec.op_kind == "reduce":
            return spec.extra.get("op") in {"sum", "mean", "max"}
        return False

    def variants(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        scale = np.array(0.5, dtype=inputs["A"].dtype)
        new_inputs = {k: np.array(v, copy=True) for k, v in inputs.items()}
        new_inputs["A"] = (new_inputs["A"] * scale).astype(inputs["A"].dtype, copy=False)

        def expected(outputs):
            return {
                name: (arr * scale).astype(arr.dtype, copy=False)
                for name, arr in outputs.items()
            }

        return [MRCase(copy.deepcopy(spec), new_inputs, identity, seed, 0, "scale_A_by_half", expected)]
