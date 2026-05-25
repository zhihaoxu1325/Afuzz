from __future__ import annotations

from asfuzz.runner.pipeline import _cross_backend_order
from asfuzz.spec.ops_catalog import make_matmul, make_transpose


class Backend:
    def __init__(self, name: str):
        self.name = name


def test_float16_cross_backend_uses_compiler_reference():
    backends = [Backend("numpy"), Backend("tvm"), Backend("metaschedule"), Backend("halide")]
    ordered = _cross_backend_order(make_matmul(2, 3, 4, dtype="float16"), backends)
    assert [backend.name for backend in ordered] == ["tvm", "metaschedule", "halide"]


def test_float32_reduction_uses_compiler_reference():
    backends = [Backend("numpy"), Backend("tvm"), Backend("metaschedule")]
    ordered = _cross_backend_order(make_matmul(2, 3, 4, dtype="float32"), backends)
    assert [backend.name for backend in ordered] == ["tvm", "metaschedule"]


def test_float32_exact_op_keeps_numpy_reference():
    backends = [Backend("numpy"), Backend("tvm"), Backend("metaschedule")]
    ordered = _cross_backend_order(make_transpose([2, 3], [1, 0], dtype="float32"), backends)
    assert [backend.name for backend in ordered] == ["numpy", "tvm", "metaschedule"]
