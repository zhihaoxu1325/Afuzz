from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

DType = Literal["float32", "float16", "bfloat16", "int8", "int32"]
TensorRole = Literal["input", "weight", "output", "bias", "temp"]


class AxisSpec(BaseModel):
    name: str
    size: int
    is_reduce: bool = False
    is_spatial: bool = False
    stride: int = 1
    dilation: int = 1
    padding: int = 0


class TensorSpec(BaseModel):
    name: str
    axes: list[str]
    dtype: DType = "float32"
    role: TensorRole = "input"


class OpSpec(BaseModel):
    name: str
    op_kind: str
    axes: dict[str, AxisSpec]
    tensors: list[TensorSpec]
    einsum_expr: str = ""
    epilogue: list[str] = Field(default_factory=list)
    layout: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def tensor(self, name: str) -> TensorSpec:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def tensors_by_role(self, role: TensorRole) -> list[TensorSpec]:
        return [tensor for tensor in self.tensors if tensor.role == role]

    def shape_of(self, tensor: TensorSpec | str) -> tuple[int, ...]:
        ts = self.tensor(tensor) if isinstance(tensor, str) else tensor
        return tuple(self.axes[axis].size for axis in ts.axes)

    def dtype(self) -> DType:
        outputs = self.tensors_by_role("output")
        if outputs:
            return outputs[0].dtype
        return self.tensors[0].dtype

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpSpec":
        return cls.model_validate(data)

    def signature(self) -> str:
        stable = self.model_dump(mode="json", exclude={"name"})
        payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def save_json(self, path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load_json(cls, path) -> "OpSpec":
        return cls.from_dict(json.loads(path.read_text()))

