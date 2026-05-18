from __future__ import annotations

import numpy as np

from .opspec import OpSpec


def _apply_epilogue(out: np.ndarray, epilogue: list[str]) -> np.ndarray:
    for op in epilogue:
        if op == "relu":
            out = np.maximum(out, 0)
        elif op == "tanh":
            out = np.tanh(out)
        elif op == "sigmoid":
            out = 1 / (1 + np.exp(-out))
        else:
            raise NotImplementedError(f"unsupported epilogue {op}")
    return out


def run_reference(spec: OpSpec, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if spec.op_kind == "matmul":
        out = inputs["A"].astype("float64") @ inputs["B"].astype("float64")
        if spec.extra.get("with_bias"):
            out = out + inputs["bias"].astype("float64")
        out = _apply_epilogue(out, spec.epilogue).astype(spec.dtype())
    elif spec.op_kind == "elementwise":
        op = spec.extra.get("op", "add")
        a = inputs["A"]
        b = inputs["B"]
        if op == "add":
            out = a + b
        elif op == "sub":
            out = a - b
        elif op == "mul":
            out = a * b
        elif op == "div":
            out = a / np.where(np.abs(b) < 1e-6, np.where(b < 0, -1e-6, 1e-6), b)
        elif op == "max":
            out = np.maximum(a, b)
        elif op == "min":
            out = np.minimum(a, b)
        else:
            raise NotImplementedError(f"unsupported elementwise op {op}")
        out = out.astype(spec.dtype())
    elif spec.op_kind == "unary":
        op = spec.extra.get("op", "relu")
        a = inputs["A"]
        if op == "relu":
            out = np.maximum(a, 0)
        elif op == "abs":
            out = np.abs(a)
        elif op == "neg":
            out = -a
        elif op == "tanh":
            out = np.tanh(a)
        elif op == "sigmoid":
            out = 1 / (1 + np.exp(-a))
        else:
            raise NotImplementedError(f"unsupported unary op {op}")
        out = out.astype(spec.dtype())
    elif spec.op_kind == "reduce":
        op = spec.extra.get("op", "sum")
        axis = int(spec.extra["axis"])
        keepdims = bool(spec.extra.get("keepdims", False))
        a64 = inputs["A"].astype("float64")
        if op == "sum":
            out = np.sum(a64, axis=axis, keepdims=keepdims)
        elif op == "mean":
            out = np.mean(a64, axis=axis, keepdims=keepdims)
        elif op == "max":
            out = np.max(a64, axis=axis, keepdims=keepdims)
        else:
            raise NotImplementedError(f"unsupported reduce op {op}")
        if out.shape == ():
            out = out.reshape((1,))
        out = out.astype(spec.dtype())
    elif spec.op_kind in {"softmax", "softmax_decomposed"}:
        axis = int(spec.extra.get("axis", len(inputs["A"].shape) - 1))
        a64 = inputs["A"].astype("float64")
        shifted = a64 - np.max(a64, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        out = (exp / np.sum(exp, axis=axis, keepdims=True)).astype(spec.dtype())
    elif spec.op_kind == "conv2d":
        a = inputs["A"].astype("float64")
        b = inputs["B"].astype("float64")
        stride = int(spec.extra.get("stride", 1))
        pad = int(spec.extra.get("pad", 0))
        dilation = int(spec.extra.get("dilation", 1))
        n, h, w, c = a.shape
        kh, kw, _, f = b.shape
        oh = spec.axes["OH"].size
        ow = spec.axes["OW"].size
        padded = np.pad(a, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        out = np.zeros((n, oh, ow, f), dtype="float64")
        for yy in range(oh):
            for xx in range(ow):
                for ky in range(kh):
                    iy = yy * stride + ky * dilation
                    for kx in range(kw):
                        ix = xx * stride + kx * dilation
                        out[:, yy, xx, :] += padded[:, iy, ix, :] @ b[ky, kx, :, :]
        out = _apply_epilogue(out, spec.epilogue).astype(spec.dtype())
    else:
        raise NotImplementedError(spec.op_kind)
    return {spec.tensors_by_role("output")[0].name: out}


def sample_inputs(spec: OpSpec, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for tensor in spec.tensors:
        if tensor.role not in {"input", "weight", "bias"}:
            continue
        shape = spec.shape_of(tensor)
        if tensor.dtype in {"float32", "float16", "bfloat16"}:
            arr = rng.normal(0, 1, size=shape).astype("float32")
            if spec.op_kind == "elementwise" and spec.extra.get("op") == "div" and tensor.name == "B":
                arr = np.where(np.abs(arr) < 0.1, arr + 0.25, arr)
            inputs[tensor.name] = arr.astype("float16" if tensor.dtype == "float16" else "float32")
        elif tensor.dtype in {"int8", "int32"}:
            inputs[tensor.name] = rng.integers(-8, 8, size=shape, dtype=np.int32).astype(tensor.dtype)
        else:
            raise NotImplementedError(tensor.dtype)
    return inputs
