from asfuzz.schedulers.tvm_backend import TVMBackend
from asfuzz.oracle.numeric import numeric_equal
from asfuzz.spec.ops_catalog import (
    make_batch_matmul,
    make_broadcast,
    make_concat,
    make_conv2d,
    make_elem_reduce,
    make_elementwise,
    make_layer_norm,
    make_matmul,
    make_matmul_chain,
    make_matmul_softmax,
    make_pad,
    make_pool2d,
    make_reduce,
    make_reshape,
    make_slice,
    make_softmax,
    make_transpose,
    make_unary,
)
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


def test_tvm_shape_model_and_composite_ops_match_reference():
    specs = [
        make_transpose([2, 3, 4], [1, 2, 0]),
        make_broadcast([1, 3, 1], [2, 3, 5]),
        make_reshape([2, 3, 4], [4, 6]),
        make_slice([5, 7, 9], [1, 2, 3], [3, 4, 5]),
        make_pad([2, 3, 4], [1, 0, 2], [0, 3, 1]),
        make_concat([2, 3, 4], [2, 5, 4], 1),
        make_batch_matmul(2, 3, 4, 5),
        make_pool2d(1, 8, 8, 3, 3, 3, stride=2, pad=1, op="avg"),
        make_pool2d(1, 8, 8, 3, 3, 3, stride=2, pad=1, op="max"),
        make_layer_norm([2, 3, 4], axis=2),
        make_elem_reduce([2, 3, 4], axis=1, elem_op="mul", reduce_op="sum"),
        make_matmul_chain(2, 3, 4, 5),
        make_matmul_chain(2, 3, 4, 5, order="right"),
        make_matmul_softmax(2, 3, 4),
    ]
    backend = TVMBackend()
    for seed, spec in enumerate(specs, 100):
        inputs = sample_inputs(spec, seed)
        actual = backend.run(backend.schedule_and_build(spec, "llvm", 0, seed), inputs)["C"]
        expected = run_reference(spec, inputs)["C"]
        assert numeric_equal(expected, actual, spec.dtype(), rtol=1e-3, atol=1e-4).ok


def test_tvm_max_pool_padding_matches_reference_for_float16_and_excess_padding():
    backend = TVMBackend()
    for dtype in ["float32", "float16"]:
        spec = make_pool2d(1, 1, 1, 1, 2, 2, stride=1, pad=3, op="max", dtype=dtype)
        inputs = sample_inputs(spec, 201)
        actual = backend.run(backend.schedule_and_build(spec, "llvm", 0, 201), inputs)["C"]
        expected = run_reference(spec, inputs)["C"]
        assert numeric_equal(expected, actual, dtype).ok


def test_tvm_reduce_split_max_matches_reference():
    spec = make_reduce([2, 8, 3], axis=1, op="max")
    spec.op_kind = "reduce_split"
    spec.extra["split_factor"] = 2
    inputs = sample_inputs(spec, 202)
    actual = TVMBackend().run(TVMBackend().schedule_and_build(spec, "llvm", 0, 202), inputs)["C"]
    expected = run_reference(spec, inputs)["C"]
    assert numeric_equal(expected, actual, spec.dtype()).ok
