from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation
from asfuzz.spec.opspec import OpSpec


class AxisPermuteMR(MetamorphicRelation):
    name = "axis_permute"

    def applicable(self, spec: OpSpec) -> bool:
        return spec.op_kind in {"elementwise", "unary"} and len(spec.shape_of(spec.tensors_by_role("output")[0])) >= 2

    def variants(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        new_spec = copy.deepcopy(spec)
        output = new_spec.tensors_by_role("output")[0]
        perm = list(reversed(range(len(output.axes))))
        old_axes = list(output.axes)
        new_axes = [old_axes[i] for i in perm]
        for tensor in new_spec.tensors:
            if tensor.axes == old_axes:
                tensor.axes = list(new_axes)
        new_inputs = {}
        for name, arr in inputs.items():
            new_inputs[name] = np.transpose(arr, perm)

        inv = np.argsort(perm)

        def recover(outputs):
            return {name: np.transpose(arr, inv) for name, arr in outputs.items()}

        new_spec.name = f"{spec.name}_axis_permute"
        return [MRCase(new_spec, new_inputs, recover, seed, 0, "reverse_axes")]

