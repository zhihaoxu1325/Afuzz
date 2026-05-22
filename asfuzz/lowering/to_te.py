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
    if spec.op_kind == "reduce_split":
        return _lower_reduce_split(spec)
    if spec.op_kind in {"softmax", "softmax_decomposed"}:
        return _lower_softmax(spec)
    if spec.op_kind == "conv2d":
        return _lower_conv2d(spec)
    if spec.op_kind == "transpose":
        return _lower_transpose(spec)
    if spec.op_kind == "broadcast":
        return _lower_broadcast(spec)
    if spec.op_kind == "reshape":
        return _lower_reshape(spec)
    if spec.op_kind == "slice":
        return _lower_slice(spec)
    if spec.op_kind == "pad":
        return _lower_pad(spec)
    if spec.op_kind == "concat":
        return _lower_concat(spec)
    if spec.op_kind == "batch_matmul":
        return _lower_batch_matmul(spec)
    if spec.op_kind == "pool2d":
        return _lower_pool2d(spec)
    if spec.op_kind == "layer_norm":
        return _lower_layer_norm(spec)
    if spec.op_kind == "elem_reduce":
        return _lower_elem_reduce(spec)
    if spec.op_kind == "matmul_chain":
        return _lower_matmul_chain(spec)
    if spec.op_kind == "matmul_softmax":
        return _lower_matmul_softmax(spec)
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


def _lower_reduce_split(spec: OpSpec):
    dtype = spec.dtype()
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    axis = int(spec.extra["axis"])
    op = spec.extra.get("op", "sum")
    keepdims = bool(spec.extra.get("keepdims", False))
    factor = int(spec.extra.get("split_factor", 2))
    if op not in {"sum", "mean"} or in_shape[axis] % factor != 0:
        return _lower_reduce(spec)
    chunks = in_shape[axis] // factor
    A = te.placeholder(in_shape, name="A", dtype=dtype)
    r_inner = te.reduce_axis((0, factor), name=f"r{axis}_inner")
    r_outer = te.reduce_axis((0, chunks), name=f"r{axis}_outer")
    temp_shape = in_shape[:axis] + (chunks,) + in_shape[axis + 1 :]

    def temp_input_idx(temp_idx):
        idx = list(temp_idx)
        idx[axis] = temp_idx[axis] * factor + r_inner
        return tuple(idx)

    Temp = te.compute(temp_shape, lambda *idx: te.sum(A[temp_input_idx(idx)], axis=r_inner), name="SplitReduceTemp")

    def temp_at_output(out_idx):
        temp_idx = []
        out_pos = 0
        for dim_i in range(len(in_shape)):
            if dim_i == axis:
                temp_idx.append(r_outer)
            elif keepdims:
                temp_idx.append(out_idx[dim_i])
            else:
                temp_idx.append(out_idx[out_pos])
                out_pos += 1
        return tuple(temp_idx)

    Acc = te.compute(out_shape, lambda *idx: te.sum(Temp[temp_at_output(idx)], axis=r_outer), name="Acc")
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


def _lower_transpose(spec: OpSpec):
    dtype = spec.dtype()
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    perm = [int(v) for v in spec.extra["perm"]]
    inv = [perm.index(i) for i in range(len(perm))]
    A = te.placeholder(in_shape, name="A", dtype=dtype)
    C = te.compute(out_shape, lambda *idx: A[tuple(idx[inv_i] for inv_i in inv)], name="C")
    return [A], C


def _lower_broadcast(spec: OpSpec):
    dtype = spec.dtype()
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    offset = len(out_shape) - len(in_shape)
    A = te.placeholder(in_shape, name="A", dtype=dtype)

    def body(*idx):
        in_idx = []
        for i, dim in enumerate(in_shape):
            in_idx.append(0 if dim == 1 else idx[offset + i])
        return A[tuple(in_idx)]

    C = te.compute(out_shape, body, name="C")
    return [A], C


def _lower_reshape(spec: OpSpec):
    dtype = spec.dtype()
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    A = te.placeholder(in_shape, name="A", dtype=dtype)

    def strides(shape):
        result = []
        stride = 1
        for dim in reversed(shape):
            result.append(stride)
            stride *= dim
        return list(reversed(result))

    in_strides = strides(in_shape)
    out_strides = strides(out_shape)

    def body(*idx):
        flat = _tir.const(0, "int64")
        for i, stride in enumerate(out_strides):
            flat = flat + idx[i] * stride
        in_idx = []
        for dim, stride in zip(in_shape, in_strides):
            in_idx.append((flat // stride) % dim)
        return A[tuple(in_idx)]

    C = te.compute(out_shape, body, name="C")
    return [A], C


def _lower_slice(spec: OpSpec):
    dtype = spec.dtype()
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    begin = [int(v) for v in spec.extra["begin"]]
    A = te.placeholder(in_shape, name="A", dtype=dtype)
    C = te.compute(out_shape, lambda *idx: A[tuple(idx[i] + begin[i] for i in range(len(out_shape)))], name="C")
    return [A], C


def _lower_pad(spec: OpSpec):
    dtype = spec.dtype()
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    before = [int(v) for v in spec.extra["before"]]
    A = te.placeholder(in_shape, name="A", dtype=dtype)

    def body(*idx):
        valid = None
        in_idx = []
        for i, dim in enumerate(in_shape):
            lo = before[i]
            expr = (idx[i] >= lo) & (idx[i] < lo + dim)
            valid = expr if valid is None else (valid & expr)
            in_idx.append(idx[i] - lo)
        return te.if_then_else(valid, A[tuple(in_idx)], _tir.const(0, dtype))

    C = te.compute(out_shape, body, name="C")
    return [A], C


def _lower_concat(spec: OpSpec):
    dtype = spec.dtype()
    shape_a = spec.shape_of("A")
    shape_b = spec.shape_of("B")
    out_shape = spec.shape_of("C")
    axis = int(spec.extra["axis"])
    A = te.placeholder(shape_a, name="A", dtype=dtype)
    B = te.placeholder(shape_b, name="B", dtype=dtype)

    def body(*idx):
        a_idx = list(idx)
        b_idx = list(idx)
        b_idx[axis] = idx[axis] - shape_a[axis]
        return te.if_then_else(idx[axis] < shape_a[axis], A[tuple(a_idx)], B[tuple(b_idx)])

    C = te.compute(out_shape, body, name="C")
    return [A, B], C


def _lower_batch_matmul(spec: OpSpec):
    dtype = spec.dtype()
    batch = spec.axes["BATCH"].size
    m = spec.axes["M"].size
    k = spec.axes["K"].size
    n = spec.axes["N"].size
    A = te.placeholder((batch, m, k), name="A", dtype=dtype)
    B = te.placeholder((batch, k, n), name="B", dtype=dtype)
    rk = te.reduce_axis((0, k), name="k")
    Acc = te.compute((batch, m, n), lambda b, i, j: te.sum(A[b, i, rk] * B[b, rk, j], axis=rk), name="Acc")
    if spec.epilogue:
        C = te.compute((batch, m, n), lambda b, i, j: _apply_epilogue(Acc[b, i, j], spec.epilogue), name="C")
    else:
        C = te.compute((batch, m, n), lambda b, i, j: Acc[b, i, j], name="C")
    return [A, B], C


def _lower_pool2d(spec: OpSpec):
    dtype = spec.dtype()
    n = spec.axes["N"].size
    h = spec.axes["H"].size
    w = spec.axes["W"].size
    c = spec.axes["C"].size
    kh = spec.axes["KH"].size
    kw = spec.axes["KW"].size
    oh = spec.axes["OH"].size
    ow = spec.axes["OW"].size
    stride = int(spec.extra.get("stride", 1))
    pad = int(spec.extra.get("pad", 0))
    op = spec.extra.get("op", "max")
    A = te.placeholder((n, h, w, c), name="A", dtype=dtype)
    rkh = te.reduce_axis((0, kh), name="rkh")
    rkw = te.reduce_axis((0, kw), name="rkw")

    def load(nn, yy, xx, cc):
        in_y = yy * stride + rkh - pad
        in_x = xx * stride + rkw - pad
        valid = (in_y >= 0) & (in_y < h) & (in_x >= 0) & (in_x < w)
        fill = _tir.const(-3.4028234663852886e38, dtype) if op == "max" else _tir.const(0, dtype)
        return te.if_then_else(valid, A[nn, in_y, in_x, cc], fill)

    if op == "avg":
        Acc = te.compute((n, oh, ow, c), lambda nn, yy, xx, cc: te.sum(load(nn, yy, xx, cc), axis=[rkh, rkw]), name="Acc")
        C = te.compute((n, oh, ow, c), lambda nn, yy, xx, cc: Acc[nn, yy, xx, cc] / _tir.const(kh * kw, dtype), name="C")
    elif op == "max":
        C = te.compute((n, oh, ow, c), lambda nn, yy, xx, cc: te.max(load(nn, yy, xx, cc), axis=[rkh, rkw]), name="C")
    else:
        raise NotImplementedError(op)
    return [A], C


def _lower_layer_norm(spec: OpSpec):
    dtype = spec.dtype()
    shape = spec.shape_of("A")
    axis = int(spec.extra.get("axis", len(shape) - 1))
    eps = float(spec.extra.get("eps", 1e-5))
    reduce_extent = shape[axis]
    A = te.placeholder(shape, name="A", dtype=dtype)
    gamma = te.placeholder((reduce_extent,), name="gamma", dtype=dtype)
    beta = te.placeholder((reduce_extent,), name="beta", dtype=dtype)
    outer_shape = shape[:axis] + shape[axis + 1 :] or (1,)
    rmean = te.reduce_axis((0, reduce_extent), name="rmean")
    rvar = te.reduce_axis((0, reduce_extent), name="rvar")

    def insert_axis(outer_idx, red):
        if len(shape) == 1:
            return (red,)
        return outer_idx[:axis] + (red,) + outer_idx[axis:]

    def remove_axis(idx):
        if len(shape) == 1:
            return (0,)
        return idx[:axis] + idx[axis + 1 :]

    MeanSum = te.compute(outer_shape, lambda *idx: te.sum(A[insert_axis(idx, rmean)], axis=rmean), name="MeanSum")
    Mean = te.compute(outer_shape, lambda *idx: MeanSum[idx] / _tir.const(reduce_extent, dtype), name="Mean")
    VarSum = te.compute(
        outer_shape,
        lambda *idx: te.sum((A[insert_axis(idx, rvar)] - Mean[idx]) * (A[insert_axis(idx, rvar)] - Mean[idx]), axis=rvar),
        name="VarSum",
    )
    Var = te.compute(outer_shape, lambda *idx: VarSum[idx] / _tir.const(reduce_extent, dtype), name="Var")
    C = te.compute(
        shape,
        lambda *idx: (A[idx] - Mean[remove_axis(idx)]) / te.sqrt(Var[remove_axis(idx)] + _tir.const(eps, dtype)) * gamma[idx[axis]] + beta[idx[axis]],
        name="C",
    )
    return [A, gamma, beta], C


def _lower_elem_reduce(spec: OpSpec):
    dtype = spec.dtype()
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    axis = int(spec.extra["axis"])
    elem_op = spec.extra.get("elem_op", "mul")
    reduce_op = spec.extra.get("reduce_op", "sum")
    A = te.placeholder(in_shape, name="A", dtype=dtype)
    B = te.placeholder(in_shape, name="B", dtype=dtype)
    r = te.reduce_axis((0, in_shape[axis]), name=f"r{axis}")

    def input_idx(out_idx):
        idx = []
        out_pos = 0
        for dim_i in range(len(in_shape)):
            if dim_i == axis:
                idx.append(r)
            else:
                idx.append(out_idx[out_pos])
                out_pos += 1
        return tuple(idx)

    def elem(idx):
        a = A[idx]
        b = B[idx]
        if elem_op == "add":
            return a + b
        if elem_op == "sub":
            return a - b
        if elem_op == "mul":
            return a * b
        if elem_op == "max":
            return te.max(a, b)
        raise NotImplementedError(elem_op)

    def body(*idx):
        val = elem(input_idx(idx))
        if reduce_op == "sum":
            return te.sum(val, axis=r)
        if reduce_op == "mean":
            return te.sum(val, axis=r)
        if reduce_op == "max":
            return te.max(val, axis=r)
        raise NotImplementedError(reduce_op)

    Acc = te.compute(out_shape, body, name="Acc")
    if reduce_op == "mean":
        C = te.compute(out_shape, lambda *idx: Acc[idx] / _tir.const(in_shape[axis], dtype), name="C")
    else:
        C = te.compute(out_shape, lambda *idx: Acc[idx], name="C")
    return [A, B], C


def _lower_matmul_chain(spec: OpSpec):
    dtype = spec.dtype()
    m = spec.axes["M"].size
    k = spec.axes["K"].size
    n = spec.axes["N"].size
    p = spec.axes["P"].size
    A = te.placeholder((m, k), name="A", dtype=dtype)
    B = te.placeholder((k, n), name="B", dtype=dtype)
    D = te.placeholder((n, p), name="D", dtype=dtype)
    if spec.extra.get("order", "left") == "right":
        rn = te.reduce_axis((0, n), name="rn")
        rk = te.reduce_axis((0, k), name="rk")
        BD = te.compute((k, p), lambda kk, pp: te.sum(B[kk, rn] * D[rn, pp], axis=rn), name="BD")
        C = te.compute((m, p), lambda mm, pp: te.sum(A[mm, rk] * BD[rk, pp], axis=rk), name="C")
    else:
        rk = te.reduce_axis((0, k), name="rk")
        rn = te.reduce_axis((0, n), name="rn")
        AB = te.compute((m, n), lambda mm, nn: te.sum(A[mm, rk] * B[rk, nn], axis=rk), name="AB")
        C = te.compute((m, p), lambda mm, pp: te.sum(AB[mm, rn] * D[rn, pp], axis=rn), name="C")
    return [A, B, D], C


def _lower_matmul_softmax(spec: OpSpec):
    dtype = spec.dtype()
    m = spec.axes["M"].size
    k = spec.axes["K"].size
    n = spec.axes["N"].size
    A = te.placeholder((m, k), name="A", dtype=dtype)
    B = te.placeholder((k, n), name="B", dtype=dtype)
    rk = te.reduce_axis((0, k), name="rk")
    rn_max = te.reduce_axis((0, n), name="rn_max")
    rn_sum = te.reduce_axis((0, n), name="rn_sum")
    Logits = te.compute((m, n), lambda i, j: te.sum(A[i, rk] * B[rk, j], axis=rk), name="Logits")
    Max = te.compute((m,), lambda i: te.max(Logits[i, rn_max], axis=rn_max), name="Max")
    Exp = te.compute((m, n), lambda i, j: te.exp(Logits[i, j] - Max[i]), name="Exp")
    Sum = te.compute((m,), lambda i: te.sum(Exp[i, rn_sum], axis=rn_sum), name="Sum")
    C = te.compute((m, n), lambda i, j: Exp[i, j] / Sum[i], name="C")
    return [A, B], C
