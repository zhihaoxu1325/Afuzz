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
    elif spec.op_kind in {"reduce", "reduce_split"}:
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
    elif spec.op_kind == "transpose":
        out = np.transpose(inputs["A"], spec.extra["perm"]).astype(spec.dtype())
    elif spec.op_kind == "broadcast":
        out = np.broadcast_to(inputs["A"], tuple(spec.extra["out_shape"])).astype(spec.dtype())
    elif spec.op_kind == "reshape":
        out = np.reshape(inputs["A"], tuple(spec.extra["out_shape"])).astype(spec.dtype())
    elif spec.op_kind == "slice":
        begin = [int(v) for v in spec.extra["begin"]]
        size = [int(v) for v in spec.extra["size"]]
        slices = tuple(slice(b, b + s) for b, s in zip(begin, size))
        out = inputs["A"][slices].astype(spec.dtype())
    elif spec.op_kind == "pad":
        before = [int(v) for v in spec.extra["before"]]
        after = [int(v) for v in spec.extra["after"]]
        out = np.pad(inputs["A"], tuple(zip(before, after)), mode="constant").astype(spec.dtype())
    elif spec.op_kind == "concat":
        axis = int(spec.extra["axis"])
        out = np.concatenate([inputs["A"], inputs["B"]], axis=axis).astype(spec.dtype())
    elif spec.op_kind == "batch_matmul":
        out = np.matmul(inputs["A"].astype("float64"), inputs["B"].astype("float64"))
        out = _apply_epilogue(out, spec.epilogue).astype(spec.dtype())
    elif spec.op_kind == "pool2d":
        a = inputs["A"].astype("float64")
        stride = int(spec.extra.get("stride", 1))
        pad = int(spec.extra.get("pad", 0))
        op = spec.extra.get("op", "max")
        n, h, w, c = a.shape
        kh = spec.axes["KH"].size
        kw = spec.axes["KW"].size
        oh = spec.axes["OH"].size
        ow = spec.axes["OW"].size
        padded = np.pad(a, ((0, 0), (pad, pad), (pad, pad), (0, 0)), constant_values=-np.inf if op == "max" else 0)
        out = np.zeros((n, oh, ow, c), dtype="float64")
        for yy in range(oh):
            for xx in range(ow):
                window = padded[:, yy * stride : yy * stride + kh, xx * stride : xx * stride + kw, :]
                if op == "avg":
                    out[:, yy, xx, :] = np.mean(window, axis=(1, 2))
                elif op == "max":
                    out[:, yy, xx, :] = np.max(window, axis=(1, 2))
                else:
                    raise NotImplementedError(f"unsupported pool op {op}")
        out = out.astype(spec.dtype())
    elif spec.op_kind == "layer_norm":
        axis = int(spec.extra.get("axis", len(inputs["A"].shape) - 1))
        eps = float(spec.extra.get("eps", 1e-5))
        a = inputs["A"].astype("float64")
        mean = np.mean(a, axis=axis, keepdims=True)
        var = np.mean((a - mean) * (a - mean), axis=axis, keepdims=True)
        shape = [1] * a.ndim
        shape[axis] = a.shape[axis]
        gamma = inputs["gamma"].astype("float64").reshape(shape)
        beta = inputs["beta"].astype("float64").reshape(shape)
        out = ((a - mean) / np.sqrt(var + eps) * gamma + beta).astype(spec.dtype())
    elif spec.op_kind == "elem_reduce":
        elem_op = spec.extra.get("elem_op", "mul")
        reduce_op = spec.extra.get("reduce_op", "sum")
        axis = int(spec.extra["axis"])
        a = inputs["A"].astype("float64")
        b = inputs["B"].astype("float64")
        if elem_op == "add":
            tmp = a + b
        elif elem_op == "sub":
            tmp = a - b
        elif elem_op == "mul":
            tmp = a * b
        elif elem_op == "max":
            tmp = np.maximum(a, b)
        else:
            raise NotImplementedError(f"unsupported elem_reduce elem op {elem_op}")
        if reduce_op == "sum":
            out = np.sum(tmp, axis=axis)
        elif reduce_op == "max":
            out = np.max(tmp, axis=axis)
        elif reduce_op == "mean":
            out = np.mean(tmp, axis=axis)
        else:
            raise NotImplementedError(f"unsupported elem_reduce reduce op {reduce_op}")
        if out.shape == ():
            out = out.reshape((1,))
        out = out.astype(spec.dtype())
    elif spec.op_kind == "matmul_chain":
        a = inputs["A"].astype("float64")
        b = inputs["B"].astype("float64")
        d = inputs["D"].astype("float64")
        if spec.extra.get("order", "left") == "right":
            out = a @ (b @ d)
        else:
            out = (a @ b) @ d
        out = out.astype(spec.dtype())
    elif spec.op_kind == "matmul_softmax":
        logits = inputs["A"].astype("float64") @ inputs["B"].astype("float64")
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        out = (exp / np.sum(exp, axis=1, keepdims=True)).astype(spec.dtype())
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
            mode = rng.choice(["normal", "normal", "small", "large", "sparse", "pattern"])
            if mode == "small":
                arr = rng.normal(0, 1e-3, size=shape).astype("float32")
            elif mode == "large":
                arr = rng.normal(0, 8, size=shape).astype("float32")
            elif mode == "sparse":
                arr = rng.normal(0, 1, size=shape).astype("float32")
                arr = np.where(rng.random(size=shape) < 0.7, 0, arr).astype("float32")
            elif mode == "pattern":
                base = np.arange(max(1, int(np.prod(shape))), dtype="float32").reshape(shape)
                arr = ((base % 17) - 8) / 4.0
            else:
                arr = rng.normal(0, 1, size=shape).astype("float32")
            if spec.op_kind == "elementwise" and spec.extra.get("op") == "div" and tensor.name == "B":
                arr = np.where(np.abs(arr) < 0.1, arr + 0.25, arr)
            if tensor.name == "gamma":
                arr = np.where(np.abs(arr) < 0.1, arr + 1.0, arr)
            inputs[tensor.name] = arr.astype("float16" if tensor.dtype == "float16" else "float32")
        elif tensor.dtype in {"int8", "int32"}:
            inputs[tensor.name] = rng.integers(-8, 8, size=shape, dtype=np.int32).astype(tensor.dtype)
        else:
            raise NotImplementedError(tensor.dtype)
    return inputs
