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


OP_REGISTRY: dict[str, Callable] = {
    "matmul": make_matmul,
    "elementwise": make_elementwise,
    "unary": make_unary,
    "reduce": make_reduce,
    "softmax": make_softmax,
    "conv2d": make_conv2d,
}
