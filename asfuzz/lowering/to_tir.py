from __future__ import annotations

from tvm import te

from asfuzz.spec.opspec import OpSpec
from .to_te import lower_to_te


def lower_to_tir(spec: OpSpec):
    inputs, output = lower_to_te(spec)
    return te.create_prim_func([*inputs, output])

