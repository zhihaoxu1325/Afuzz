from __future__ import annotations

from dataclasses import dataclass, field

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
    if spec.op_kind not in {"matmul", "elementwise", "unary", "reduce", "softmax", "softmax_decomposed", "conv2d"}:
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
    if spec.dtype() not in {"float32", "float16"}:
        reasons.append(f"unsupported dtype {spec.dtype()}")
    return ValidationResult(ok=not reasons, reasons=reasons, supported_backends=["numpy", "tvm"] if not reasons else [])
