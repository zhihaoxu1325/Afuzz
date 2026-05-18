from asfuzz.oracle.numeric import numeric_equal
from asfuzz.schedulers.halide_backend import HalideBackend
from asfuzz.spec.ops_catalog import make_elementwise, make_matmul, make_reduce, make_softmax, make_unary
from asfuzz.spec.reference import run_reference, sample_inputs


def _check(spec):
    inputs = sample_inputs(spec, 0)
    backend = HalideBackend()
    artifact = backend.schedule_and_build(spec, "llvm", 0, 0)
    actual = backend.run(artifact, inputs)["C"]
    expected = run_reference(spec, inputs)["C"]
    assert numeric_equal(expected, actual, spec.dtype()).ok


def test_halide_basic_ops():
    _check(make_matmul(2, 3, 4, with_bias=True, act="relu"))
    _check(make_elementwise([2, 3], "mul"))
    _check(make_unary([2, 3], "sigmoid"))
    _check(make_reduce([2, 3], axis=1, op="mean"))
    _check(make_softmax([2, 3]))
