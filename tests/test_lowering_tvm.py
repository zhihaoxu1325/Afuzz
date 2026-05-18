from asfuzz.schedulers.tvm_backend import TVMBackend
from asfuzz.oracle.numeric import numeric_equal
from asfuzz.spec.ops_catalog import make_conv2d, make_elementwise, make_matmul, make_reduce, make_softmax, make_unary
from asfuzz.spec.reference import run_reference, sample_inputs


def _run(spec):
    be = TVMBackend()
    art = be.schedule_and_build(spec, "llvm", 0, 0)
    return be.run(art, sample_inputs(spec, 0))["C"]


def test_tvm_matmul():
    assert _run(make_matmul(2, 3, 4)).shape == (2, 4)


def test_tvm_elementwise_unary_reduce():
    assert _run(make_elementwise([2, 3], "add")).shape == (2, 3)
    assert _run(make_unary([2, 3], "relu")).shape == (2, 3)
    assert _run(make_reduce([2, 3], axis=1, op="sum")).shape == (2,)
    assert _run(make_softmax([2, 3])).shape == (2, 3)
    assert _run(make_softmax([2, 3, 4], axis=1)).shape == (2, 3, 4)
    assert _run(make_conv2d(1, 5, 5, 2, 3, 3, 3, pad=1)).shape == (1, 5, 5, 3)


def test_tvm_non_last_axis_softmax_matches_reference():
    spec = make_softmax([2, 3, 4], axis=1)
    inputs = sample_inputs(spec, 17)
    backend = TVMBackend()
    actual = backend.run(backend.schedule_and_build(spec, "llvm", 0, 0), inputs)["C"]
    expected = run_reference(spec, inputs)["C"]
    assert numeric_equal(expected, actual, "float32", rtol=1e-4, atol=1e-4).ok


def test_tvm_conv2d_matches_reference():
    spec = make_conv2d(1, 5, 5, 2, 3, 3, 3, pad=1)
    inputs = sample_inputs(spec, 11)
    backend = TVMBackend()
    actual = backend.run(backend.schedule_and_build(spec, "llvm", 0, 0), inputs)["C"]
    expected = run_reference(spec, inputs)["C"]
    assert numeric_equal(expected, actual, "float32", rtol=1e-4, atol=1e-4).ok
