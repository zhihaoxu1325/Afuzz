from __future__ import annotations

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation
from asfuzz.spec.ops_catalog import make_matmul
from asfuzz.spec.opspec import OpSpec


class MatmulTransposeMR(MetamorphicRelation):
    name = "matmul_transpose"

    def applicable(self, spec: OpSpec) -> bool:
        if spec.op_kind != "matmul":
            return False
        if spec.extra.get("with_bias"):
            return False
        return all(op in {"relu", "tanh"} for op in spec.epilogue)

    def variants(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        m = spec.axes["M"].size
        k = spec.axes["K"].size
        n = spec.axes["N"].size
        act = spec.epilogue[0] if spec.epilogue else None
        new_spec = make_matmul(n, k, m, dtype=spec.dtype(), with_bias=False, act=act)
        new_spec.name = f"{spec.name}_transpose_dual"
        new_inputs = {
            "A": np.ascontiguousarray(inputs["B"].T),
            "B": np.ascontiguousarray(inputs["A"].T),
        }

        def recover(outputs):
            return {name: np.ascontiguousarray(arr.T) for name, arr in outputs.items()}

        return [MRCase(new_spec, new_inputs, recover, seed, 0, "transpose_dual")]
