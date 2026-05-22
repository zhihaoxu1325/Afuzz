from __future__ import annotations

from dataclasses import dataclass, field

from .ops_catalog import SUPPORTED_OP_KINDS
from .opspec import OpSpec


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    supported_backends: list[str] = field(default_factory=list)


def validate(spec: OpSpec, max_work_items: int = 4_000_000) -> ValidationResult:
    reasons: list[str] = []
    if not spec.tensors_by_role("output"):
        reasons.append("missing output tensor")
    for axis in spec.axes.values():
        if axis.size <= 0:
            reasons.append(f"axis {axis.name} has non-positive size {axis.size}")
    tensor_names = set()
    for tensor in spec.tensors:
        if tensor.name in tensor_names:
            reasons.append(f"duplicate tensor {tensor.name}")
        tensor_names.add(tensor.name)
        for axis in tensor.axes:
            if axis not in spec.axes:
                reasons.append(f"tensor {tensor.name} references missing axis {axis}")
    work_items = 1
    for output in spec.tensors_by_role("output"):
        for dim in spec.shape_of(output):
            work_items *= dim
    for axis in spec.axes.values():
        if axis.is_reduce:
            work_items *= axis.size
    if work_items > max_work_items:
        reasons.append(f"work items {work_items} exceeds limit {max_work_items}")
    if spec.op_kind not in SUPPORTED_OP_KINDS and spec.op_kind not in {"softmax_decomposed", "reduce_split"}:
        reasons.append(f"unsupported op_kind {spec.op_kind}")
    if spec.op_kind in {"softmax", "softmax_decomposed"}:
        axis = int(spec.extra.get("axis", -1))
        rank = len(spec.shape_of("A"))
        if axis < 0 or axis >= rank:
            reasons.append(f"softmax axis {axis} outside rank {rank}")
    if spec.op_kind == "conv2d":
        if spec.layout != "NHWC":
            reasons.append("conv2d MVP expects NHWC layout")
        if spec.axes["OH"].size <= 0 or spec.axes["OW"].size <= 0:
            reasons.append("conv2d output spatial extent must be positive")
    if spec.op_kind == "pool2d":
        if spec.layout != "NHWC":
            reasons.append("pool2d expects NHWC layout")
        if spec.axes["OH"].size <= 0 or spec.axes["OW"].size <= 0:
            reasons.append("pool2d output spatial extent must be positive")
    if spec.op_kind in {"transpose", "broadcast", "reshape", "slice", "pad", "concat"}:
        try:
            _validate_shape_op(spec, reasons)
        except Exception as exc:
            reasons.append(f"invalid shape op metadata: {exc}")
    if spec.op_kind in {"layer_norm", "matmul_softmax"}:
        axis = int(spec.extra.get("axis", -1))
        rank = len(spec.shape_of("A"))
        if axis < 0 or axis >= rank:
            reasons.append(f"{spec.op_kind} axis {axis} outside rank {rank}")
    if spec.dtype() not in {"float32", "float16"}:
        reasons.append(f"unsupported dtype {spec.dtype()}")
    return ValidationResult(ok=not reasons, reasons=reasons, supported_backends=["numpy", "tvm"] if not reasons else [])


def _validate_shape_op(spec: OpSpec, reasons: list[str]) -> None:
    if spec.op_kind == "transpose":
        rank = len(spec.shape_of("A"))
        perm = [int(v) for v in spec.extra["perm"]]
        if sorted(perm) != list(range(rank)):
            reasons.append(f"invalid transpose perm {perm} for rank {rank}")
    elif spec.op_kind == "broadcast":
        in_shape = spec.shape_of("A")
        out_shape = spec.shape_of("C")
        if len(in_shape) > len(out_shape):
            reasons.append("broadcast input rank exceeds output rank")
        offset = len(out_shape) - len(in_shape)
        for i, dim in enumerate(in_shape):
            out_dim = out_shape[offset + i]
            if dim not in {1, out_dim}:
                reasons.append(f"cannot broadcast input dim {dim} to {out_dim}")
    elif spec.op_kind == "reshape":
        in_elems = 1
        for dim in spec.shape_of("A"):
            in_elems *= dim
        out_elems = 1
        for dim in spec.shape_of("C"):
            out_elems *= dim
        if in_elems != out_elems:
            reasons.append(f"reshape element count mismatch {in_elems} != {out_elems}")
    elif spec.op_kind == "slice":
        in_shape = spec.shape_of("A")
        begin = [int(v) for v in spec.extra["begin"]]
        size = [int(v) for v in spec.extra["size"]]
        if len(begin) != len(in_shape) or len(size) != len(in_shape):
            reasons.append("slice begin/size rank mismatch")
        for dim, b, s in zip(in_shape, begin, size):
            if b < 0 or s <= 0 or b + s > dim:
                reasons.append(f"invalid slice begin={b} size={s} dim={dim}")
    elif spec.op_kind == "pad":
        before = [int(v) for v in spec.extra["before"]]
        after = [int(v) for v in spec.extra["after"]]
        if len(before) != len(spec.shape_of("A")) or len(after) != len(spec.shape_of("A")):
            reasons.append("pad rank mismatch")
        if any(v < 0 for v in before + after):
            reasons.append("pad values must be non-negative")
    elif spec.op_kind == "concat":
        a = spec.shape_of("A")
        b = spec.shape_of("B")
        c = spec.shape_of("C")
        axis = int(spec.extra["axis"])
        if len(a) != len(b) or len(a) != len(c):
            reasons.append("concat rank mismatch")
        if axis < 0 or axis >= len(a):
            reasons.append(f"concat axis {axis} outside rank {len(a)}")
        for i, (da, db, dc) in enumerate(zip(a, b, c)):
            expected = da + db if i == axis else da
            if i != axis and da != db:
                reasons.append("concat non-axis dims differ")
            if dc != expected:
                reasons.append(f"concat output dim {dc} != expected {expected}")
