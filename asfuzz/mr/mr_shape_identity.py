from __future__ import annotations

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation
from asfuzz.spec.ops_catalog import make_reshape, make_slice, make_transpose
from asfuzz.spec.opspec import OpSpec


class ShapeIdentityMR(MetamorphicRelation):
    name = "shape_identity"

    def applicable(self, spec: OpSpec) -> bool:
        return spec.op_kind in {"transpose", "reshape", "pad", "concat"}

    def variants(self, spec: OpSpec, inputs: dict[str, np.ndarray], seed: int) -> list[MRCase]:
        if spec.op_kind == "transpose":
            return self._transpose_roundtrip(spec, inputs, seed)
        if spec.op_kind == "reshape":
            return self._reshape_roundtrip(spec, inputs, seed)
        if spec.op_kind == "pad":
            return self._pad_slice(spec, inputs, seed)
        if spec.op_kind == "concat":
            return self._concat_split(spec, inputs, seed)
        return []

    def _transpose_roundtrip(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        perm = [int(v) for v in spec.extra["perm"]]
        inv = list(np.argsort(perm))
        transposed = np.transpose(inputs["A"], perm)
        new_spec = make_transpose(list(transposed.shape), perm=inv, dtype=spec.dtype())

        def expected(outputs):
            return {"C": np.transpose(outputs["C"], inv)}

        return [MRCase(new_spec, {"A": transposed}, lambda out: out, seed, 0, "transpose_roundtrip", expected)]

    def _reshape_roundtrip(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        in_shape = list(inputs["A"].shape)
        out_shape = list(spec.shape_of("C"))
        reshaped = np.reshape(inputs["A"], out_shape)
        new_spec = make_reshape(out_shape, in_shape, dtype=spec.dtype())

        def expected(outputs):
            return {"C": np.reshape(outputs["C"], in_shape)}

        return [MRCase(new_spec, {"A": reshaped}, lambda out: out, seed, 0, "reshape_roundtrip", expected)]

    def _pad_slice(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        before = [int(v) for v in spec.extra["before"]]
        after = [int(v) for v in spec.extra["after"]]
        padded = np.pad(inputs["A"], tuple(zip(before, after)), mode="constant")
        new_spec = make_slice(list(padded.shape), before, list(inputs["A"].shape), dtype=spec.dtype())

        def expected(outputs):
            slices = tuple(slice(b, b + s) for b, s in zip(before, inputs["A"].shape))
            return {"C": outputs["C"][slices]}

        return [MRCase(new_spec, {"A": padded}, lambda out: out, seed, 0, "pad_then_slice_original", expected)]

    def _concat_split(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        axis = int(spec.extra["axis"])
        concat = np.concatenate([inputs["A"], inputs["B"]], axis=axis)
        variants = []
        begin_a = [0] * concat.ndim
        size_a = list(inputs["A"].shape)
        spec_a = make_slice(list(concat.shape), begin_a, size_a, dtype=spec.dtype())

        def expected_a(outputs):
            slices = tuple(slice(0, dim) for dim in inputs["A"].shape)
            return {"C": outputs["C"][slices]}

        variants.append(MRCase(spec_a, {"A": concat}, lambda out: out, seed, 0, "concat_split_a", expected_a))
        begin_b = [0] * concat.ndim
        begin_b[axis] = inputs["A"].shape[axis]
        size_b = list(inputs["B"].shape)
        spec_b = make_slice(list(concat.shape), begin_b, size_b, dtype=spec.dtype())

        def expected_b(outputs):
            slices = []
            for dim_i, dim in enumerate(inputs["B"].shape):
                if dim_i == axis:
                    start = inputs["A"].shape[axis]
                    slices.append(slice(start, start + dim))
                else:
                    slices.append(slice(0, dim))
            return {"C": outputs["C"][tuple(slices)]}

        variants.append(MRCase(spec_b, {"A": concat}, lambda out: out, seed, 0, "concat_split_b", expected_b))
        return variants
