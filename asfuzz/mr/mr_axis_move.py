from __future__ import annotations

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation
from asfuzz.spec.ops_catalog import make_reduce, make_softmax
from asfuzz.spec.opspec import OpSpec


class AxisMoveMR(MetamorphicRelation):
    name = "axis_move"

    def applicable(self, spec: OpSpec) -> bool:
        if spec.op_kind not in {"reduce", "softmax"}:
            return False
        rank = len(spec.shape_of("A"))
        if rank < 2:
            return False
        axis = int(spec.extra.get("axis", -1))
        return 0 <= axis < rank

    def variants(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        shape = list(spec.shape_of("A"))
        rank = len(shape)
        axis = int(spec.extra["axis"])
        perm = _move_axis_perm(rank, axis)
        new_axis = perm.index(axis)
        new_shape = [shape[i] for i in perm]
        new_inputs = {"A": np.ascontiguousarray(np.transpose(inputs["A"], perm))}
        if spec.op_kind == "softmax":
            new_spec = make_softmax(new_shape, axis=new_axis, dtype=spec.dtype())
            new_spec.name = f"{spec.name}_axis_moved"
            inv = np.argsort(perm)

            def recover(outputs):
                return {name: np.ascontiguousarray(np.transpose(arr, inv)) for name, arr in outputs.items()}

            return [MRCase(new_spec, new_inputs, recover, seed, 0, "move_softmax_axis")]

        keepdims = bool(spec.extra.get("keepdims", False))
        new_spec = make_reduce(
            new_shape,
            axis=new_axis,
            op=spec.extra.get("op", "sum"),
            dtype=spec.dtype(),
            keepdims=keepdims,
        )
        new_spec.name = f"{spec.name}_axis_moved"

        if keepdims:
            inv = np.argsort(perm)

            def recover(outputs):
                return {name: np.ascontiguousarray(np.transpose(arr, inv)) for name, arr in outputs.items()}

        else:
            original_order = [i for i in range(rank) if i != axis]
            variant_order = [old_axis for old_axis in perm if old_axis != axis]
            out_perm = [variant_order.index(old_axis) for old_axis in original_order]

            def recover(outputs):
                return {name: np.ascontiguousarray(np.transpose(arr, out_perm)) for name, arr in outputs.items()}

        return [MRCase(new_spec, new_inputs, recover, seed, 0, "move_reduce_axis")]


def _move_axis_perm(rank: int, axis: int) -> list[int]:
    axes = list(range(rank))
    if axis == rank - 1:
        return [axis] + axes[:axis]
    return axes[:axis] + axes[axis + 1 :] + [axis]
