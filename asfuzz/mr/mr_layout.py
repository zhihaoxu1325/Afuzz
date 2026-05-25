from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation
from asfuzz.spec.opspec import OpSpec


class LayoutMR(MetamorphicRelation):
    name = "layout"

    def applicable(self, spec: OpSpec) -> bool:
        if spec.op_kind != "conv2d" or spec.layout != "NHWC":
            return False
        # Swapping two non-trivial kernel axes also swaps floating reduction
        # order.  Keep this relation exact by requiring one singleton axis.
        return spec.axes["KH"].size == 1 or spec.axes["KW"].size == 1

    def variants(self, spec: OpSpec, inputs, seed: int):
        new_spec = copy.deepcopy(spec)
        # Equivalent layout perturbation: swap image H/W and filter KH/KW.
        for a, b in [("H", "W"), ("OH", "OW"), ("KH", "KW")]:
            new_spec.axes[a].size, new_spec.axes[b].size = new_spec.axes[b].size, new_spec.axes[a].size
        new_spec.name = f"{spec.name}_hw_swapped"
        new_spec.extra["layout_mr"] = "swap_hw"
        new_inputs = {
            "A": np.ascontiguousarray(np.transpose(inputs["A"], (0, 2, 1, 3))),
            "B": np.ascontiguousarray(np.transpose(inputs["B"], (1, 0, 2, 3))),
        }

        def recover(outputs):
            return {name: np.ascontiguousarray(np.transpose(arr, (0, 2, 1, 3))) for name, arr in outputs.items()}

        return [MRCase(new_spec, new_inputs, recover, seed, 0, "conv2d_swap_hw")]
