from __future__ import annotations

from dataclasses import dataclass

from asfuzz.spec.opspec import OpSpec


@dataclass
class HalideLowered:
    pipeline: object
    inputs: dict[str, object]
    output_name: str
    realize_shape: list[int]
    output_shape: tuple[int, ...]


def lower_to_halide(spec: OpSpec) -> HalideLowered:
    import halide as hl

    if spec.dtype() != "float32":
        raise NotImplementedError("Halide MVP backend currently supports float32 only")
    if spec.op_kind == "matmul":
        return _lower_matmul(spec, hl)
    if spec.op_kind == "elementwise":
        return _lower_elementwise(spec, hl)
    if spec.op_kind == "unary":
        return _lower_unary(spec, hl)
    if spec.op_kind == "reduce":
        return _lower_reduce(spec, hl)
    if spec.op_kind in {"softmax", "softmax_decomposed"}:
        return _lower_softmax(spec, hl)
    raise NotImplementedError(spec.op_kind)


def _vars(hl, rank: int):
    return [hl.Var(f"x{i}") for i in range(rank)]


def _image_param(hl, name: str, rank: int):
    return hl.ImageParam(hl.Float(32), rank, name)


def _buffer_index(vars_, rank: int):
    return tuple(vars_[i] for i in range(rank))


def _set_estimates(obj, shape: tuple[int, ...]) -> None:
    # Halide dimensions are x-major; numpy arrays arrive as row-major, so the
    # Halide extent order is the reverse of the NumPy shape.
    obj.set_estimates([(0, int(dim)) for dim in reversed(shape)])


def _finish(spec: OpSpec, hl, func, inputs: dict[str, object], output_shape: tuple[int, ...]) -> HalideLowered:
    for tensor in spec.tensors:
        if tensor.name in inputs:
            _set_estimates(inputs[tensor.name], spec.shape_of(tensor))
    _set_estimates(func, output_shape)
    return HalideLowered(
        pipeline=hl.Pipeline(func),
        inputs=inputs,
        output_name=spec.tensors_by_role("output")[0].name,
        realize_shape=[int(dim) for dim in reversed(output_shape)],
        output_shape=output_shape,
    )


def _lower_matmul(spec: OpSpec, hl) -> HalideLowered:
    m, k, n = spec.axes["M"].size, spec.axes["K"].size, spec.axes["N"].size
    x, y = hl.Var("x"), hl.Var("y")
    r = hl.RDom([(0, k)])
    A = _image_param(hl, "A", 2)
    B = _image_param(hl, "B", 2)
    expr = hl.sum(A[r[0], y] * B[x, r[0]])
    inputs = {"A": A, "B": B}
    if spec.extra.get("with_bias"):
        bias = _image_param(hl, "bias", 1)
        expr = expr + bias[x]
        inputs["bias"] = bias
    expr = _apply_epilogue(expr, spec.epilogue, hl)
    C = hl.Func("C")
    C[x, y] = expr
    return _finish(spec, hl, C, inputs, (m, n))


def _lower_elementwise(spec: OpSpec, hl) -> HalideLowered:
    shape = spec.shape_of("C")
    rank = len(shape)
    vars_ = _vars(hl, rank)
    A = _image_param(hl, "A", rank)
    B = _image_param(hl, "B", rank)
    idx = _buffer_index(vars_, rank)
    op = spec.extra.get("op", "add")
    if op == "add":
        expr = A[idx] + B[idx]
    elif op == "sub":
        expr = A[idx] - B[idx]
    elif op == "mul":
        expr = A[idx] * B[idx]
    elif op == "div":
        expr = A[idx] / B[idx]
    elif op == "max":
        expr = hl.max(A[idx], B[idx])
    elif op == "min":
        expr = hl.min(A[idx], B[idx])
    else:
        raise NotImplementedError(op)
    C = hl.Func("C")
    C[tuple(vars_)] = expr
    return _finish(spec, hl, C, {"A": A, "B": B}, shape)


def _lower_unary(spec: OpSpec, hl) -> HalideLowered:
    shape = spec.shape_of("C")
    rank = len(shape)
    vars_ = _vars(hl, rank)
    A = _image_param(hl, "A", rank)
    val = A[_buffer_index(vars_, rank)]
    op = spec.extra.get("op", "relu")
    if op == "relu":
        expr = hl.max(val, 0.0)
    elif op == "abs":
        expr = hl.abs(val)
    elif op == "neg":
        expr = -val
    elif op == "tanh":
        expr = hl.tanh(val)
    elif op == "sigmoid":
        expr = 1.0 / (1.0 + hl.exp(-val))
    else:
        raise NotImplementedError(op)
    C = hl.Func("C")
    C[tuple(vars_)] = expr
    return _finish(spec, hl, C, {"A": A}, shape)


def _lower_reduce(spec: OpSpec, hl) -> HalideLowered:
    in_shape = spec.shape_of("A")
    out_shape = spec.shape_of("C")
    axis = int(spec.extra["axis"])
    op = spec.extra.get("op", "sum")
    keepdims = bool(spec.extra.get("keepdims", False))
    vars_ = _vars(hl, len(out_shape))
    A = _image_param(hl, "A", len(in_shape))
    r = hl.RDom([(0, in_shape[axis])])
    in_idx = []
    out_pos = 0
    for dim_i in range(len(in_shape)):
        halide_dim_i = len(in_shape) - 1 - dim_i
        if dim_i == axis:
            in_idx.append(r[0])
        elif keepdims:
            in_idx.append(vars_[len(out_shape) - 1 - dim_i])
        else:
            numpy_out_pos = out_pos
            out_pos += 1
            in_idx.append(vars_[len(out_shape) - 1 - numpy_out_pos])
    in_idx = tuple(reversed(in_idx))
    val = A[in_idx]
    if op == "sum":
        expr = hl.sum(val)
    elif op == "mean":
        expr = hl.sum(val) / float(in_shape[axis])
    elif op == "max":
        expr = hl.maximum(val)
    else:
        raise NotImplementedError(op)
    C = hl.Func("C")
    C[tuple(vars_)] = expr
    return _finish(spec, hl, C, {"A": A}, out_shape)


def _lower_softmax(spec: OpSpec, hl) -> HalideLowered:
    shape = spec.shape_of("A")
    axis = int(spec.extra.get("axis", len(shape) - 1))
    if axis != len(shape) - 1:
        raise NotImplementedError("Halide softmax lowering currently supports last axis only")
    rank = len(shape)
    vars_ = _vars(hl, rank)
    A = _image_param(hl, "A", rank)
    r = hl.RDom([(0, shape[axis])])
    current = A[tuple(vars_)]
    reduced_idx = (r[0], *vars_[1:])
    max_val = hl.maximum(A[reduced_idx])
    exp_val = hl.exp(current - max_val)
    denom = hl.sum(hl.exp(A[reduced_idx] - max_val))
    C = hl.Func("C")
    C[tuple(vars_)] = exp_val / denom
    return _finish(spec, hl, C, {"A": A}, shape)


def _apply_epilogue(expr, epilogue: list[str], hl):
    for op in epilogue:
        if op == "relu":
            expr = hl.max(expr, 0.0)
        elif op == "tanh":
            expr = hl.tanh(expr)
        elif op == "sigmoid":
            expr = 1.0 / (1.0 + hl.exp(-expr))
        else:
            raise NotImplementedError(f"unsupported epilogue {op}")
    return expr
