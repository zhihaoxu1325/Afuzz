from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation, identity
from asfuzz.spec.opspec import OpSpec


class BatchSplitMR(MetamorphicRelation):
    name = "batch_split"

    def applicable(self, spec: OpSpec) -> bool:
        outputs = spec.tensors_by_role("output")
        if not outputs or not outputs[0].axes:
            return False
        batch_axis = outputs[0].axes[0]
        if spec.axes[batch_axis].size < 2:
            return False
        if spec.op_kind == "reduce" and int(spec.extra.get("axis", -1)) == 0:
            return False
        if spec.op_kind in {"softmax", "softmax_decomposed"} and int(spec.extra.get("axis", -1)) == 0:
            return False
        return spec.op_kind in {"matmul", "elementwise", "unary", "reduce", "softmax", "conv2d"}

    def variants(self, spec: OpSpec, inputs, seed: int):
        output = spec.tensors_by_role("output")[0]
        batch_axis = output.axes[0]
        old_size = spec.axes[batch_axis].size
        split_size = old_size // 2
        variants = []
        for tag, start, end in [
            ("batch_prefix", 0, split_size),
            ("batch_suffix", split_size, old_size),
        ]:
            part_size = end - start
            if part_size <= 0:
                continue
            new_spec = copy.deepcopy(spec)
            new_spec.axes[batch_axis].size = part_size

            new_inputs = {}
            for tensor in new_spec.tensors:
                if tensor.role not in {"input", "weight", "bias"}:
                    continue
                arr = inputs[tensor.name]
                if tensor.axes and tensor.axes[0] == batch_axis:
                    new_inputs[tensor.name] = np.array(arr[start:end], copy=True)
                else:
                    new_inputs[tensor.name] = np.array(arr, copy=True)

            def expected_recover(outputs, start=start, end=end):
                return {
                    name: np.array(arr[start:end], copy=True)
                    for name, arr in outputs.items()
                }

            new_spec.name = f"{spec.name}_{tag}_{part_size}"
            variants.append(MRCase(new_spec, new_inputs, identity, seed, 0, tag, expected_recover))
        return variants
