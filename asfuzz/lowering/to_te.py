from __future__ import annotations

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

import tvm
from tvm import te

_tir = getattr(tvm, "tir", getattr(tvm, "tirx", None))

from asfuzz.spec.opspec import OpSpec


def lower_to_te(spec: OpSpec):
    if spec.op_kind == "matmul":
        return _lower_matmul(spec)
    if spec.op_kind == "elementwise":
        return _lower_elementwise(spec)
    if spec.op_kind == "unary":
        return _lower_unary(spec)
    if spec.op_kind == "reduce":
        return _lower_reduce(spec)
    if spec.op_kind in {"softmax", "softmax_decomposed"}:
        return _lower_softmax(spec)
    if spec.op_kind == "conv2d":
        return _lower_conv2d(spec)
    raise NotImplementedError(spec.op_kind)


def lower_to_schedule(spec: OpSpec):
    args, out = lower_to_te(spec)
    if not hasattr(te, "create_schedule"):
        raise RuntimeError("this TVM build no longer exposes te.create_schedule")
    schedule = te.create_schedule(out.op)
    return schedule, args + [out]


def _apply_epilogue(expr, epilogue: list[str]):
    for op in epilogue:
        if op == "relu":
            expr = te.max(expr, _tir.const(0, expr.dtype))
        elif op == "tanh":
            expr = te.tanh(expr)
        elif op == "sigmoid":
            one = _tir.const(1, expr.dtype)
            expr = one / (one + te.exp(-expr))
        else:
            raise NotImplementedError(f"unsupported epilogue {op}")
    return expr


def _lower_matmul(spec: OpSpec):
    dtype = spec.dtype()
    m = spec.axes["M"].size
    k = spec.axes["K"].size
    n = spec.axes["N"].size
    A = te.placeholder((m, k), name="A", dtype=dtype)
    B = te.placeholder((k, n), name="B", dtype=dtype)
    rk = te.reduce_axis((0, k), name="k")
    Acc = te.compute((m, n), lambda i, j: te.sum(A[i, rk] * B[rk, j], axis=rk), name="Acc")
    if spec.extra.get("with_bias"):
        bias = te.placeholder((n,), name="bias", dtype=dtype)
        C = te.compute((m, n), lambda i, j: _apply_epilogue(Acc[i, j] + bias[j], spec.epilogue), name="C")
        return [A, B, bias], C
    if spec.epilogue:
        C = te.compute((m, n), lambda i, j: _apply_epilogue(Acc[i, j], spec.epilogue), name="C")
    else:
        C = te.compute((m, n), lambda i, j: Acc[i, j], name="C")
    return [A, B], C


def _lower_elementwise(spec: OpSpec):
    dtype = spec.dtype()
    shape = spec.shape_of("C")
    A = te.placeholder(shape, name="A", dtype=dtype)
    B = te.placeholder(shape, name="B", dtype=dtype)
    op = spec.extra.get("op", "add")

    def body(*idx):
        if op == "add":
            return A[idx] + B[idx]
        if op == "sub":
            return A[idx] - B[idx]
        if op == "mul":
            return A[idx] * B[idx]
        if op == "div":
            return A[idx] / B[idx]
        if op == "max":
            return te.max(A[idx], B[idx])
        if op == "min":
            return te.min(A[idx], B[idx])
        raise NotImplementedError(op)

    C = te.compute(shape, body, name="C")
    return [A, B], C


def _lower_unary(spec: OpSpec):
    dtype = spec.dtype()
    shape = spec.shape_of("C")
    A = te.placeholder(shape, name="A", dtype=dtype)
    op = spec.extra.get("op", "relu")

    def body(*idx):
        val = A[idx]
        if op == "relu":
            return te.max(val, _tir.const(0, dtype))
        if op == "abs":
            return te.if_then_else(val >= _tir.const(0, dtype), val, -val)
        if op == "neg":
            return -val
        if op == "tanh":
            return te.tanh(val)
        if op == "sigmoid":
            one = _tir.const(1, dtype)
            return one / (one + te.exp(-val))
        raise NotImplementedError(op)

    C = te.compute(shape, body, name="C")
    return [A], C


def _lower_reduce(spec: OpSpec):
    dtype = spec.dtype()
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    axis = int(spec.extra["axis"])
    op = spec.extra.get("op", "sum")
    keepdims = bool(spec.extra.get("keepdims", False))
    A = te.placeholder(in_shape, name="A", dtype=dtype)
    r = te.reduce_axis((0, in_shape[axis]), name=f"r{axis}")

    def map_input(out_idx):
        in_idx = []
        out_pos = 0
        for dim_i in range(len(in_shape)):
            if dim_i == axis:
                in_idx.append(r)
            elif keepdims:
                in_idx.append(out_idx[dim_i])
            else:
                in_idx.append(out_idx[out_pos])
                out_pos += 1
        return tuple(in_idx)

    def body(*idx):
        val = A[map_input(idx)]
        if op == "sum":
            return te.sum(val, axis=r)
        if op == "mean":
            return te.sum(val, axis=r)
        if op == "max":
            return te.max(val, axis=r)
        raise NotImplementedError(op)

    Acc = te.compute(out_shape, body, name="Acc")
    if op == "mean":
        C = te.compute(out_shape, lambda *idx: Acc[idx] / _tir.const(in_shape[axis], dtype), name="C")
    else:
        C = te.compute(out_shape, lambda *idx: Acc[idx], name="C")
    return [A], C


def _lower_softmax(spec: OpSpec):
    dtype = spec.dtype()
    shape = spec.shape_of("A")
    axis = int(spec.extra.get("axis", len(shape) - 1))
    rank = len(shape)
    A = te.placeholder(shape, name="A", dtype=dtype)
    outer_shape = shape[:axis] + shape[axis + 1 :] or (1,)
    reduce_extent = shape[axis]
    rmax = te.reduce_axis((0, reduce_extent), name="rmax")
    rsum = te.reduce_axis((0, reduce_extent), name="rsum")

    def remove_axis(idx):
        if rank == 1:
            return (0,)
        return idx[:axis] + idx[axis + 1 :]

    def insert_axis(outer_idx, red):
        if rank == 1:
            return (red,)
        return outer_idx[:axis] + (red,) + outer_idx[axis:]

    def a_at(outer_idx, red):
        if rank == 1:
            return A[red]
        return A[insert_axis(outer_idx, red)]

    Max = te.compute(outer_shape, lambda *idx: te.max(a_at(idx, rmax), axis=rmax), name="Max")
    Exp = te.compute(shape, lambda *idx: te.exp(A[idx] - Max[remove_axis(idx)]), name="Exp")

    def exp_at(outer_idx, red):
        if rank == 1:
            return Exp[red]
        return Exp[insert_axis(outer_idx, red)]

    Sum = te.compute(outer_shape, lambda *idx: te.sum(exp_at(idx, rsum), axis=rsum), name="Sum")
    C = te.compute(shape, lambda *idx: Exp[idx] / Sum[remove_axis(idx)], name="C")
    return [A], C


def _lower_conv2d(spec: OpSpec):
    dtype = spec.dtype()
    n = spec.axes["N"].size
    h = spec.axes["H"].size
    w = spec.axes["W"].size
    c = spec.axes["C"].size
    f = spec.axes["F"].size
    kh = spec.axes["KH"].size
    kw = spec.axes["KW"].size
    oh = spec.axes["OH"].size
    ow = spec.axes["OW"].size
    stride = int(spec.extra.get("stride", 1))
    pad = int(spec.extra.get("pad", 0))
    dilation = int(spec.extra.get("dilation", 1))
    A = te.placeholder((n, h, w, c), name="A", dtype=dtype)
    B = te.placeholder((kh, kw, c, f), name="B", dtype=dtype)
    rkh = te.reduce_axis((0, kh), name="rkh")
    rkw = te.reduce_axis((0, kw), name="rkw")
    rc = te.reduce_axis((0, c), name="rc")

    def load(nn, yy, xx, cc):
        in_y = yy * stride + rkh * dilation - pad
        in_x = xx * stride + rkw * dilation - pad
        valid = (in_y >= 0) & (in_y < h) & (in_x >= 0) & (in_x < w)
        return te.if_then_else(valid, A[nn, in_y, in_x, cc], _tir.const(0, dtype))

    Acc = te.compute(
        (n, oh, ow, f),
        lambda nn, yy, xx, ff: te.sum(load(nn, yy, xx, rc) * B[rkh, rkw, rc, ff], axis=[rkh, rkw, rc]),
        name="Acc",
    )
    if spec.epilogue:
        C = te.compute((n, oh, ow, f), lambda nn, yy, xx, ff: _apply_epilogue(Acc[nn, yy, xx, ff], spec.epilogue), name="C")
    else:
        C = te.compute((n, oh, ow, f), lambda nn, yy, xx, ff: Acc[nn, yy, xx, ff], name="C")
    return [A, B], C
