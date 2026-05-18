from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation
from asfuzz.spec.opspec import OpSpec


class PadSliceMR(MetamorphicRelation):
    name = "pad_slice"

    def applicable(self, spec: OpSpec) -> bool:
        return spec.op_kind in {"elementwise", "unary"} and len(spec.shape_of(spec.tensors_by_role("output")[0])) >= 1

    def variants(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        new_spec = copy.deepcopy(spec)
        output = new_spec.tensors_by_role("output")[0]
        axis_name = output.axes[-1]
        old_size = new_spec.axes[axis_name].size
        new_spec.axes[axis_name].size = old_size + 1
        pads = [(0, 0)] * len(output.axes)
        pads[-1] = (0, 1)
        new_inputs = {name: np.pad(arr, pads, mode="constant") for name, arr in inputs.items()}

        def recover(outputs):
            slices = tuple(slice(0, old_size) if i == len(output.axes) - 1 else slice(None) for i in range(len(output.axes)))
            return {name: arr[slices] for name, arr in outputs.items()}

        new_spec.name = f"{spec.name}_pad_slice"
        return [MRCase(new_spec, new_inputs, recover, seed, 0, "pad_last_dim")]

