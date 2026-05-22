from __future__ import annotations

from collections.abc import Callable

from .opspec import AxisSpec, OpSpec, TensorSpec


def _axes(names_sizes: list[tuple[str, int]], reduce: set[str] | None = None) -> dict[str, AxisSpec]:
    reduce = reduce or set()
    return {
        name: AxisSpec(name=name, size=int(size), is_reduce=name in reduce)
        for name, size in names_sizes
    }


def make_matmul(M: int, K: int, N: int, dtype: str = "float32", with_bias: bool = False, act: str | None = None) -> OpSpec:
    tensors = [
        TensorSpec(name="A", axes=["M", "K"], dtype=dtype, role="input"),
        TensorSpec(name="B", axes=["K", "N"], dtype=dtype, role="weight"),
    ]
    expr = "C[m,n] = sum_k A[m,k] * B[k,n]"
    if with_bias:
        tensors.append(TensorSpec(name="bias", axes=["N"], dtype=dtype, role="bias"))
        expr += " + bias[n]"
    tensors.append(TensorSpec(name="C", axes=["M", "N"], dtype=dtype, role="output"))
    epilogue = [act] if act else []
    return OpSpec(
        name=f"matmul_m{M}_k{K}_n{N}",
        op_kind="matmul",
        axes=_axes([("M", M), ("K", K), ("N", N)], {"K"}),
        tensors=tensors,
        einsum_expr=expr,
        epilogue=epilogue,
        extra={"with_bias": with_bias},
    )


def make_elementwise(shape: list[int], op: str = "add", dtype: str = "float32") -> OpSpec:
    axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    names = [name for name, _ in axes]
    return OpSpec(
        name=f"{op}_{'x'.join(map(str, shape))}",
        op_kind="elementwise",
        axes=_axes(axes),
        tensors=[
            TensorSpec(name="A", axes=names, dtype=dtype, role="input"),
            TensorSpec(name="B", axes=names, dtype=dtype, role="input"),
            TensorSpec(name="C", axes=names, dtype=dtype, role="output"),
        ],
        einsum_expr=f"C[...] = A[...] {op} B[...]",
        extra={"op": op},
    )


def make_unary(shape: list[int], op: str = "relu", dtype: str = "float32") -> OpSpec:
    axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    names = [name for name, _ in axes]
    return OpSpec(
        name=f"{op}_{'x'.join(map(str, shape))}",
        op_kind="unary",
        axes=_axes(axes),
        tensors=[
            TensorSpec(name="A", axes=names, dtype=dtype, role="input"),
            TensorSpec(name="C", axes=names, dtype=dtype, role="output"),
        ],
        einsum_expr=f"C[...] = {op}(A[...])",
        extra={"op": op},
    )


def make_reduce(shape: list[int], axis: int, op: str = "sum", dtype: str = "float32", keepdims: bool = False) -> OpSpec:
    in_axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    reduce_axis = f"d{axis}"
    if keepdims:
        out_axes = [f"d{i}_keep" if i == axis else name for i, (name, _) in enumerate(in_axes)]
        all_axes = in_axes + [(f"d{axis}_keep", 1)]
    else:
        out_axes = [name for i, (name, _) in enumerate(in_axes) if i != axis]
        if not out_axes:
            out_axes = ["scalar"]
            all_axes = in_axes + [("scalar", 1)]
        else:
            all_axes = in_axes
    return OpSpec(
        name=f"reduce_{op}_{'x'.join(map(str, shape))}_axis{axis}",
        op_kind="reduce",
        axes=_axes(all_axes, {reduce_axis}),
        tensors=[
            TensorSpec(name="A", axes=[name for name, _ in in_axes], dtype=dtype, role="input"),
            TensorSpec(name="C", axes=out_axes, dtype=dtype, role="output"),
        ],
        einsum_expr=f"C[...] = {op}_{reduce_axis} A[...]",
        extra={"op": op, "axis": axis, "keepdims": keepdims},
    )


def make_softmax(shape: list[int], axis: int | None = None, dtype: str = "float32") -> OpSpec:
    axis = len(shape) - 1 if axis is None else axis
    axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    names = [name for name, _ in axes]
    return OpSpec(
        name=f"softmax_{'x'.join(map(str, shape))}_axis{axis}",
        op_kind="softmax",
        axes=_axes(axes, {f"d{axis}"}),
        tensors=[
            TensorSpec(name="A", axes=names, dtype=dtype, role="input"),
            TensorSpec(name="C", axes=names, dtype=dtype, role="output"),
        ],
        einsum_expr="C[...] = exp(A[...] - max_axis(A)) / sum_axis(exp(A[...] - max_axis(A)))",
        extra={"axis": axis},
    )


def make_conv2d(
    N: int,
    H: int,
    W: int,
    C: int,
    F: int,
    KH: int,
    KW: int,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
    dtype: str = "float32",
    act: str | None = None,
) -> OpSpec:
    oh = (H + 2 * pad - dilation * (KH - 1) - 1) // stride + 1
    ow = (W + 2 * pad - dilation * (KW - 1) - 1) // stride + 1
    epilogue = [act] if act else []
    return OpSpec(
        name=f"conv2d_n{N}_h{H}_w{W}_c{C}_f{F}_kh{KH}_kw{KW}_s{stride}_p{pad}_d{dilation}",
        op_kind="conv2d",
        axes=_axes(
            [
                ("N", N),
                ("H", H),
                ("W", W),
                ("C", C),
                ("F", F),
                ("KH", KH),
                ("KW", KW),
                ("OH", oh),
                ("OW", ow),
            ],
            {"C", "KH", "KW"},
        ),
        tensors=[
            TensorSpec(name="A", axes=["N", "H", "W", "C"], dtype=dtype, role="input"),
            TensorSpec(name="B", axes=["KH", "KW", "C", "F"], dtype=dtype, role="weight"),
            TensorSpec(name="C", axes=["N", "OH", "OW", "F"], dtype=dtype, role="output"),
        ],
        einsum_expr="C[n,oh,ow,f] = sum_{kh,kw,c} A[n,oh*s+kh*d-p,ow*s+kw*d-p,c] * B[kh,kw,c,f]",
        epilogue=epilogue,
        layout="NHWC",
        extra={"stride": stride, "pad": pad, "dilation": dilation},
    )


def make_transpose(shape: list[int], perm: list[int] | None = None, dtype: str = "float32") -> OpSpec:
    rank = len(shape)
    perm = list(reversed(range(rank))) if perm is None else perm
    axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    in_names = [name for name, _ in axes]
    out_names = [in_names[i] for i in perm]
    return OpSpec(
        name=f"transpose_{'x'.join(map(str, shape))}_{'_'.join(map(str, perm))}",
        op_kind="transpose",
        axes=_axes(axes),
        tensors=[
            TensorSpec(name="A", axes=in_names, dtype=dtype, role="input"),
            TensorSpec(name="C", axes=out_names, dtype=dtype, role="output"),
        ],
        einsum_expr="C[perm(...)] = A[...]",
        extra={"perm": perm},
    )


def make_broadcast(shape: list[int], out_shape: list[int], dtype: str = "float32") -> OpSpec:
    axes = [(f"d{i}", dim) for i, dim in enumerate(out_shape)]
    out_names = [name for name, _ in axes]
    offset = len(out_shape) - len(shape)
    in_names = []
    all_axes = list(axes)
    for i, dim in enumerate(shape):
        out_axis = out_names[offset + i]
        if dim == out_shape[offset + i]:
            in_names.append(out_axis)
        elif dim == 1:
            unit_axis = f"{out_axis}_unit"
            all_axes.append((unit_axis, 1))
            in_names.append(unit_axis)
        else:
            raise ValueError(f"cannot broadcast {shape} to {out_shape}")
    return OpSpec(
        name=f"broadcast_{'x'.join(map(str, shape))}_to_{'x'.join(map(str, out_shape))}",
        op_kind="broadcast",
        axes=_axes(all_axes),
        tensors=[
            TensorSpec(name="A", axes=in_names, dtype=dtype, role="input"),
            TensorSpec(name="C", axes=out_names, dtype=dtype, role="output"),
        ],
        einsum_expr="C[...] = broadcast(A)",
        extra={"in_shape": shape, "out_shape": out_shape},
    )


def make_reshape(shape: list[int], out_shape: list[int], dtype: str = "float32") -> OpSpec:
    elems = 1
    for dim in shape:
        elems *= dim
    out_elems = 1
    for dim in out_shape:
        out_elems *= dim
    if elems != out_elems:
        raise ValueError(f"cannot reshape {shape} to {out_shape}")
    in_axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    out_axes = [(f"o{i}", dim) for i, dim in enumerate(out_shape)]
    return OpSpec(
        name=f"reshape_{'x'.join(map(str, shape))}_to_{'x'.join(map(str, out_shape))}",
        op_kind="reshape",
        axes=_axes(in_axes + out_axes),
        tensors=[
            TensorSpec(name="A", axes=[name for name, _ in in_axes], dtype=dtype, role="input"),
            TensorSpec(name="C", axes=[name for name, _ in out_axes], dtype=dtype, role="output"),
        ],
        einsum_expr="C[...] = reshape(A)",
        extra={"in_shape": shape, "out_shape": out_shape},
    )


def make_slice(shape: list[int], begin: list[int], size: list[int], dtype: str = "float32") -> OpSpec:
    in_axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    out_axes = [(f"o{i}", dim) for i, dim in enumerate(size)]
    return OpSpec(
        name=f"slice_{'x'.join(map(str, shape))}_{'_'.join(map(str, begin))}_{'x'.join(map(str, size))}",
        op_kind="slice",
        axes=_axes(in_axes + out_axes),
        tensors=[
            TensorSpec(name="A", axes=[name for name, _ in in_axes], dtype=dtype, role="input"),
            TensorSpec(name="C", axes=[name for name, _ in out_axes], dtype=dtype, role="output"),
        ],
        einsum_expr="C[...] = A[begin + ...]",
        extra={"begin": begin, "size": size},
    )


def make_pad(shape: list[int], before: list[int], after: list[int], dtype: str = "float32") -> OpSpec:
    out_shape = [before[i] + shape[i] + after[i] for i in range(len(shape))]
    in_axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    out_axes = [(f"o{i}", dim) for i, dim in enumerate(out_shape)]
    return OpSpec(
        name=f"pad_{'x'.join(map(str, shape))}_b{'_'.join(map(str, before))}_a{'_'.join(map(str, after))}",
        op_kind="pad",
        axes=_axes(in_axes + out_axes),
        tensors=[
            TensorSpec(name="A", axes=[name for name, _ in in_axes], dtype=dtype, role="input"),
            TensorSpec(name="C", axes=[name for name, _ in out_axes], dtype=dtype, role="output"),
        ],
        einsum_expr="C[...] = pad(A)",
        extra={"before": before, "after": after},
    )


def make_concat(shape_a: list[int], shape_b: list[int], axis: int, dtype: str = "float32") -> OpSpec:
    if len(shape_a) != len(shape_b):
        raise ValueError("concat ranks must match")
    for i, (a, b) in enumerate(zip(shape_a, shape_b)):
        if i != axis and a != b:
            raise ValueError("concat non-axis dimensions must match")
    out_shape = list(shape_a)
    out_shape[axis] += shape_b[axis]
    axes = [(f"d{i}", dim) for i, dim in enumerate(out_shape)]
    out_names = [name for name, _ in axes]
    a_axes = [f"a{i}" if i == axis else out_names[i] for i in range(len(shape_a))]
    b_axes = [f"b{i}" if i == axis else out_names[i] for i in range(len(shape_b))]
    extra_axes = [(a_axes[axis], shape_a[axis]), (b_axes[axis], shape_b[axis])]
    return OpSpec(
        name=f"concat_{'x'.join(map(str, shape_a))}_{'x'.join(map(str, shape_b))}_axis{axis}",
        op_kind="concat",
        axes=_axes(axes + extra_axes),
        tensors=[
            TensorSpec(name="A", axes=a_axes, dtype=dtype, role="input"),
            TensorSpec(name="B", axes=b_axes, dtype=dtype, role="input"),
            TensorSpec(name="C", axes=out_names, dtype=dtype, role="output"),
        ],
        einsum_expr="C[...] = concat(A, B)",
        extra={"axis": axis, "shape_a": shape_a, "shape_b": shape_b},
    )


def make_batch_matmul(batch: int, M: int, K: int, N: int, dtype: str = "float32", act: str | None = None) -> OpSpec:
    epilogue = [act] if act else []
    return OpSpec(
        name=f"batch_matmul_b{batch}_m{M}_k{K}_n{N}",
        op_kind="batch_matmul",
        axes=_axes([("BATCH", batch), ("M", M), ("K", K), ("N", N)], {"K"}),
        tensors=[
            TensorSpec(name="A", axes=["BATCH", "M", "K"], dtype=dtype, role="input"),
            TensorSpec(name="B", axes=["BATCH", "K", "N"], dtype=dtype, role="weight"),
            TensorSpec(name="C", axes=["BATCH", "M", "N"], dtype=dtype, role="output"),
        ],
        einsum_expr="C[b,m,n] = sum_k A[b,m,k] * B[b,k,n]",
        epilogue=epilogue,
    )


def make_pool2d(
    N: int,
    H: int,
    W: int,
    C: int,
    KH: int,
    KW: int,
    stride: int = 1,
    pad: int = 0,
    op: str = "max",
    dtype: str = "float32",
) -> OpSpec:
    oh = (H + 2 * pad - KH) // stride + 1
    ow = (W + 2 * pad - KW) // stride + 1
    return OpSpec(
        name=f"pool2d_{op}_n{N}_h{H}_w{W}_c{C}_kh{KH}_kw{KW}_s{stride}_p{pad}",
        op_kind="pool2d",
        axes=_axes(
            [("N", N), ("H", H), ("W", W), ("C", C), ("KH", KH), ("KW", KW), ("OH", oh), ("OW", ow)],
            {"KH", "KW"},
        ),
        tensors=[
            TensorSpec(name="A", axes=["N", "H", "W", "C"], dtype=dtype, role="input"),
            TensorSpec(name="C", axes=["N", "OH", "OW", "C"], dtype=dtype, role="output"),
        ],
        einsum_expr="C[n,oh,ow,c] = pool_{kh,kw} A[n,oh*s+kh-p,ow*s+kw-p,c]",
        layout="NHWC",
        extra={"stride": stride, "pad": pad, "op": op},
    )


def make_layer_norm(shape: list[int], axis: int | None = None, dtype: str = "float32", eps: float = 1e-5) -> OpSpec:
    axis = len(shape) - 1 if axis is None else axis
    axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    names = [name for name, _ in axes]
    return OpSpec(
        name=f"layer_norm_{'x'.join(map(str, shape))}_axis{axis}",
        op_kind="layer_norm",
        axes=_axes(axes, {f"d{axis}"}),
        tensors=[
            TensorSpec(name="A", axes=names, dtype=dtype, role="input"),
            TensorSpec(name="gamma", axes=[names[axis]], dtype=dtype, role="weight"),
            TensorSpec(name="beta", axes=[names[axis]], dtype=dtype, role="bias"),
            TensorSpec(name="C", axes=names, dtype=dtype, role="output"),
        ],
        einsum_expr="C[...] = (A - mean_axis(A)) / sqrt(var_axis(A) + eps) * gamma + beta",
        extra={"axis": axis, "eps": eps},
    )


def make_elem_reduce(shape: list[int], axis: int, elem_op: str = "mul", reduce_op: str = "sum", dtype: str = "float32") -> OpSpec:
    in_axes = [(f"d{i}", dim) for i, dim in enumerate(shape)]
    out_axes = [name for i, (name, _) in enumerate(in_axes) if i != axis]
    if not out_axes:
        out_axes = ["scalar"]
        all_axes = in_axes + [("scalar", 1)]
    else:
        all_axes = in_axes
    return OpSpec(
        name=f"elem_reduce_{elem_op}_{reduce_op}_{'x'.join(map(str, shape))}_axis{axis}",
        op_kind="elem_reduce",
        axes=_axes(all_axes, {f"d{axis}"}),
        tensors=[
            TensorSpec(name="A", axes=[name for name, _ in in_axes], dtype=dtype, role="input"),
            TensorSpec(name="B", axes=[name for name, _ in in_axes], dtype=dtype, role="input"),
            TensorSpec(name="C", axes=out_axes, dtype=dtype, role="output"),
        ],
        einsum_expr="C[...] = reduce_axis(A elem_op B)",
        extra={"axis": axis, "elem_op": elem_op, "reduce_op": reduce_op},
    )


def make_matmul_chain(M: int, K: int, N: int, P: int, dtype: str = "float32", order: str = "left") -> OpSpec:
    return OpSpec(
        name=f"matmul_chain_m{M}_k{K}_n{N}_p{P}_{order}",
        op_kind="matmul_chain",
        axes=_axes([("M", M), ("K", K), ("N", N), ("P", P)], {"K", "N"}),
        tensors=[
            TensorSpec(name="A", axes=["M", "K"], dtype=dtype, role="input"),
            TensorSpec(name="B", axes=["K", "N"], dtype=dtype, role="weight"),
            TensorSpec(name="D", axes=["N", "P"], dtype=dtype, role="weight"),
            TensorSpec(name="C", axes=["M", "P"], dtype=dtype, role="output"),
        ],
        einsum_expr="C = (A @ B) @ D or A @ (B @ D)",
        extra={"order": order},
    )


def make_matmul_softmax(M: int, K: int, N: int, dtype: str = "float32") -> OpSpec:
    return OpSpec(
        name=f"matmul_softmax_m{M}_k{K}_n{N}",
        op_kind="matmul_softmax",
        axes=_axes([("M", M), ("K", K), ("N", N)], {"K", "N"}),
        tensors=[
            TensorSpec(name="A", axes=["M", "K"], dtype=dtype, role="input"),
            TensorSpec(name="B", axes=["K", "N"], dtype=dtype, role="weight"),
            TensorSpec(name="C", axes=["M", "N"], dtype=dtype, role="output"),
        ],
        einsum_expr="C[m,n] = softmax_n(sum_k A[m,k] * B[k,n])",
        extra={"axis": 1},
    )


OP_REGISTRY: dict[str, Callable] = {
    "matmul": make_matmul,
    "elementwise": make_elementwise,
    "unary": make_unary,
    "reduce": make_reduce,
    "softmax": make_softmax,
    "conv2d": make_conv2d,
    "transpose": make_transpose,
    "broadcast": make_broadcast,
    "reshape": make_reshape,
    "slice": make_slice,
    "pad": make_pad,
    "concat": make_concat,
    "batch_matmul": make_batch_matmul,
    "pool2d": make_pool2d,
    "layer_norm": make_layer_norm,
    "elem_reduce": make_elem_reduce,
    "matmul_chain": make_matmul_chain,
    "matmul_softmax": make_matmul_softmax,
}

SUPPORTED_OP_KINDS = frozenset(OP_REGISTRY)
