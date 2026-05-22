from __future__ import annotations

import random

from asfuzz.fuzzer.coverage import CoverageTracker, complexity_score
from asfuzz.spec.ops_catalog import (
    make_batch_matmul,
    make_broadcast,
    make_concat,
    make_conv2d,
    make_elem_reduce,
    make_elementwise,
    make_layer_norm,
    make_matmul,
    make_matmul_chain,
    make_matmul_softmax,
    make_pad,
    make_pool2d,
    make_reduce,
    make_reshape,
    make_slice,
    make_softmax,
    make_transpose,
    make_unary,
)
from asfuzz.spec.opspec import OpSpec
from asfuzz.spec.validate import validate


class GrammarFuzzer:
    def __init__(self, rng_seed: int, op_weights: dict[str, float] | None = None, dtypes: list[str] | None = None, complexity: str = "stress"):
        self.rng = random.Random(rng_seed)
        self.op_weights = op_weights or {
            "matmul": 1.0,
            "elementwise": 1.0,
            "unary": 1.0,
            "reduce": 1.0,
            "softmax": 1.0,
            "conv2d": 1.0,
            "shape": 1.0,
            "batch_matmul": 1.0,
            "pool2d": 1.0,
            "layer_norm": 1.0,
            "composite": 1.0,
        }
        self.dtypes = dtypes or ["float32"]
        self.complexity = complexity
        self.seen: set[str] = set()

    def sample(self) -> OpSpec:
        for _ in range(100):
            spec = self._sample_once()
            sig = spec.signature()
            if sig not in self.seen:
                self.seen.add(sig)
                return spec
        return self._sample_once()

    def sample_diverse(
        self,
        coverage: CoverageTracker,
        max_work_items: int,
        candidates: int = 96,
        min_complexity: float = 0.0,
        complexity_weight: float = 0.35,
        novelty_weight: float = 1.0,
    ) -> OpSpec:
        best: tuple[float, OpSpec] | None = None
        attempts = max(1, candidates)
        for _ in range(attempts):
            spec = self._sample_once()
            sig = spec.signature()
            if sig in self.seen:
                continue
            if not validate(spec, max_work_items).ok:
                continue
            cscore = complexity_score(spec, max_work_items)
            if cscore < min_complexity:
                continue
            score = novelty_weight * coverage.diversity_score(spec) + complexity_weight * cscore
            if best is None or score > best[0]:
                best = (score, spec)
        if best is not None:
            self.accept(best[1])
            return best[1]
        return self.sample()

    def accept(self, spec: OpSpec) -> None:
        self.seen.add(spec.signature())

    def _sample_once(self) -> OpSpec:
        kind = self._choice_weighted(self.op_weights)
        dtype = self.rng.choice(self.dtypes)
        if kind == "matmul":
            sizes = self._dims("matmul")
            return make_matmul(
                M=self.rng.choice(sizes),
                K=self.rng.choice(sizes),
                N=self.rng.choice(sizes),
                dtype=dtype,
                with_bias=self.rng.random() < self._prob("matmul_bias"),
                act=self.rng.choice([None, None, "relu", "tanh"]),
            )
        if kind == "elementwise":
            shape = self._shape(rank=self.rng.choice(self._ranks()))
            op = self.rng.choice(["add", "sub", "mul", "mul", "div", "max", "min"])
            return make_elementwise(shape, op=op, dtype=dtype)
        if kind == "unary":
            shape = self._shape(rank=self.rng.choice(self._ranks()))
            op = self.rng.choice(["relu", "relu", "abs", "neg", "tanh", "sigmoid"])
            return make_unary(shape, op=op, dtype=dtype)
        if kind == "reduce":
            rank = self.rng.choice(self._ranks())
            shape = self._shape(rank=rank)
            if self.complexity == "stress" and self.rng.random() < 0.65:
                heavy_axis = self.rng.randrange(rank)
                shape[heavy_axis] = self.rng.choice([63, 64, 65, 96, 127, 128, 129, 191, 192, 255, 256])
                axis = heavy_axis
            else:
                axis = self.rng.randrange(rank)
            op = self.rng.choice(["sum", "mean", "max"])
            return make_reduce(shape, axis=axis, op=op, dtype=dtype, keepdims=self.rng.random() < self._prob("keepdims"))
        if kind == "softmax":
            rank = self.rng.choice([r for r in self._ranks() if r >= 2])
            shape = self._shape(rank=rank)
            if self.complexity == "stress" and self.rng.random() < 0.75:
                axis = self.rng.randrange(rank)
                shape[axis] = self.rng.choice([31, 63, 64, 65, 96, 127, 128, 129, 191, 192])
            else:
                axis = rank - 1
            return make_softmax(shape, axis=axis, dtype=dtype)
        if kind == "conv2d":
            n = self.rng.choice([1, 2, 3, 4] if self.complexity in {"large", "stress"} else [1, 2])
            h = self.rng.choice(self._spatial_dims())
            w = self.rng.choice(self._spatial_dims())
            c = self.rng.choice([1, 3, 4, 8, 16, 31, 32] if self.complexity == "stress" else [1, 3, 4, 8, 16])
            f = self.rng.choice([1, 4, 8, 16, 31, 32, 64] if self.complexity == "stress" else [1, 4, 8, 16, 32])
            kernels = [1, 2, 3, 5, 7] if self.complexity == "stress" else [1, 2, 3, 5]
            kh = self.rng.choice(kernels)
            kw = self.rng.choice(kernels)
            stride = self.rng.choice([1, 2, 3] if self.complexity == "stress" else [1, 2])
            dilation = self.rng.choice([1, 1, 2, 3] if self.complexity == "stress" else [1, 1, 2])
            pad_y = max(0, dilation * (kh - 1))
            pad = self.rng.choice([0, kh // 2, max(0, pad_y // 2), pad_y])
            return make_conv2d(n, h, w, c, f, kh, kw, stride=stride, pad=pad, dilation=dilation, dtype=dtype, act=self.rng.choice([None, "relu", "tanh"]))
        if kind == "shape":
            return self._sample_shape_op(dtype)
        if kind == "batch_matmul":
            sizes = self._dims("matmul")
            return make_batch_matmul(
                batch=self.rng.choice([1, 2, 3, 4, 7, 8]),
                M=self.rng.choice(sizes),
                K=self.rng.choice(sizes),
                N=self.rng.choice(sizes),
                dtype=dtype,
                act=self.rng.choice([None, None, "relu", "tanh"]),
            )
        if kind == "pool2d":
            n = self.rng.choice([1, 2, 3, 4])
            h = self.rng.choice(self._spatial_dims())
            w = self.rng.choice(self._spatial_dims())
            c = self.rng.choice([1, 3, 4, 8, 16, 31, 32])
            kh = self.rng.choice([2, 3, 5, 7])
            kw = self.rng.choice([2, 3, 5, 7])
            stride = self.rng.choice([1, 2, 3])
            pad = self.rng.choice([0, kh // 2, kw // 2])
            return make_pool2d(n, h, w, c, kh, kw, stride=stride, pad=pad, op=self.rng.choice(["max", "avg"]), dtype=dtype)
        if kind == "layer_norm":
            rank = self.rng.choice([r for r in self._ranks() if r >= 2])
            shape = self._shape(rank)
            axis = self.rng.choice([rank - 1, self.rng.randrange(rank)])
            shape[axis] = self.rng.choice([7, 8, 16, 31, 32, 63, 64, 65, 96, 127, 128])
            return make_layer_norm(shape, axis=axis, dtype=dtype)
        if kind == "composite":
            return self._sample_composite(dtype)
        raise ValueError(kind)

    def _sample_shape_op(self, dtype: str) -> OpSpec:
        op = self.rng.choice(["transpose", "broadcast", "reshape", "slice", "pad", "concat"])
        rank = self.rng.choice([r for r in self._ranks() if r >= 2])
        shape = self._shape(rank)
        if op == "transpose":
            perm = list(range(rank))
            self.rng.shuffle(perm)
            if perm == list(range(rank)):
                perm = list(reversed(perm))
            return make_transpose(shape, perm=perm, dtype=dtype)
        if op == "broadcast":
            out_shape = list(shape)
            in_shape = list(shape)
            for i in range(rank):
                if self.rng.random() < 0.45:
                    in_shape[i] = 1
                    out_shape[i] = self.rng.choice(self._dims("shape"))
            if self.rng.random() < 0.35:
                out_shape = [self.rng.choice([2, 3, 7, 8])] + out_shape
            return make_broadcast(in_shape, out_shape, dtype=dtype)
        if op == "reshape":
            elems = 1
            for dim in shape:
                elems *= dim
            factors = [dim for dim in self._dims("shape") if dim > 1 and elems % dim == 0]
            if not factors:
                return make_reshape(shape, [elems], dtype=dtype)
            first = self.rng.choice(factors)
            rest = elems // first
            if rest > 1 and self.rng.random() < 0.5:
                rest_factors = [dim for dim in self._dims("shape") if dim > 1 and rest % dim == 0]
                if rest_factors:
                    second = self.rng.choice(rest_factors)
                    return make_reshape(shape, [first, second, rest // second], dtype=dtype)
            return make_reshape(shape, [first, rest], dtype=dtype)
        if op == "slice":
            begin = []
            size = []
            for dim in shape:
                if dim <= 2:
                    b = 0
                    s = dim
                else:
                    b = self.rng.randrange(0, max(1, dim // 3))
                    s = self.rng.randrange(1, dim - b + 1)
                begin.append(b)
                size.append(s)
            return make_slice(shape, begin, size, dtype=dtype)
        if op == "pad":
            before = [self.rng.choice([0, 0, 1, 2, 3]) for _ in shape]
            after = [self.rng.choice([0, 1, 2, 3, 5]) for _ in shape]
            return make_pad(shape, before, after, dtype=dtype)
        axis = self.rng.randrange(rank)
        shape_b = list(shape)
        shape_b[axis] = self.rng.choice(self._dims("shape"))
        return make_concat(shape, shape_b, axis=axis, dtype=dtype)

    def _sample_composite(self, dtype: str) -> OpSpec:
        op = self.rng.choice(["elem_reduce", "matmul_chain", "matmul_softmax"])
        sizes = self._dims("matmul")
        if op == "elem_reduce":
            rank = self.rng.choice([r for r in self._ranks() if r >= 2])
            shape = self._shape(rank)
            axis = self.rng.randrange(rank)
            shape[axis] = self.rng.choice([31, 32, 63, 64, 65, 96, 127, 128, 129])
            return make_elem_reduce(
                shape,
                axis=axis,
                elem_op=self.rng.choice(["add", "sub", "mul", "mul", "max"]),
                reduce_op=self.rng.choice(["sum", "mean", "max"]),
                dtype=dtype,
            )
        if op == "matmul_chain":
            return make_matmul_chain(
                self.rng.choice(sizes),
                self.rng.choice(sizes),
                self.rng.choice(sizes),
                self.rng.choice(sizes),
                dtype=dtype,
                order=self.rng.choice(["left", "right"]),
            )
        return make_matmul_softmax(self.rng.choice(sizes), self.rng.choice(sizes), self.rng.choice(sizes), dtype=dtype)

    def _shape(self, rank: int) -> list[int]:
        dims = self._dims("shape")
        return [self.rng.choice(dims) for _ in range(rank)]

    def _ranks(self) -> list[int]:
        if self.complexity == "small":
            return [1, 2, 3, 4]
        if self.complexity == "medium":
            return [1, 2, 3, 4, 5]
        if self.complexity == "stress":
            return [2, 3, 4, 5, 6, 7]
        return [1, 2, 3, 4, 5, 6]

    def _dims(self, family: str) -> list[int]:
        if self.complexity == "small":
            return [1, 2, 3, 4, 7, 8, 13, 16, 31, 32, 64]
        if self.complexity == "medium":
            return [1, 2, 3, 4, 7, 8, 13, 16, 31, 32, 63, 64, 65, 96]
        if family == "matmul":
            return [
                1, 2, 3, 4, 7, 8, 13, 15, 16, 17, 29, 31, 32, 33,
                63, 64, 65, 96, 97, 113, 127, 128, 129, 191, 192,
                193, 255, 256, 257, 384,
            ]
        return [
            1, 2, 3, 4, 7, 8, 13, 16, 17, 29, 31, 32, 33, 63, 64,
            65, 67, 96, 97, 113, 127, 128, 129, 191, 192, 193, 257,
        ]

    def _spatial_dims(self) -> list[int]:
        if self.complexity == "small":
            return [8, 13, 16, 31]
        if self.complexity == "medium":
            return [8, 13, 16, 31, 32, 47, 64]
        return [8, 13, 16, 17, 31, 32, 33, 47, 63, 64, 65, 67, 96, 97, 127, 128, 129]

    def _prob(self, feature: str) -> float:
        if feature == "matmul_bias":
            return 0.55 if self.complexity in {"large", "stress"} else 0.35
        if feature == "keepdims":
            return 0.55 if self.complexity in {"large", "stress"} else 0.3
        return 0.5

    def _choice_weighted(self, weights: dict[str, float]) -> str:
        keys = list(weights)
        vals = [float(weights[k]) for k in keys]
        total = sum(vals)
        pick = self.rng.random() * total
        acc = 0.0
        for key, val in zip(keys, vals):
            acc += val
            if pick <= acc:
                return key
        return keys[-1]
