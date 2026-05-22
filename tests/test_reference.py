from asfuzz.spec.ops_catalog import (
    make_batch_matmul,
    make_broadcast,
    make_concat,
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
    make_conv2d,
)
from asfuzz.spec.reference import run_reference, sample_inputs


def test_reference_matmul():
    spec = make_matmul(2, 3, 4, with_bias=True, act="relu")
    out = run_reference(spec, sample_inputs(spec, 0))["C"]
    assert out.shape == (2, 4)


def test_reference_elementwise_reduce():
    spec = make_elementwise([2, 3], op="mul")
    out = run_reference(spec, sample_inputs(spec, 1))["C"]
    assert out.shape == (2, 3)
    red = make_reduce([2, 3], axis=1, op="sum")
    assert run_reference(red, sample_inputs(red, 2))["C"].shape == (2,)
    red_keep = make_reduce([2, 3, 4], axis=1, op="mean", keepdims=True)
    assert red_keep.shape_of("C") == (2, 1, 4)
    assert run_reference(red_keep, sample_inputs(red_keep, 22))["C"].shape == (2, 1, 4)
    sm = make_softmax([2, 3])
    assert run_reference(sm, sample_inputs(sm, 3))["C"].shape == (2, 3)
    conv = make_conv2d(1, 5, 5, 2, 3, 3, 3, pad=1)
    assert run_reference(conv, sample_inputs(conv, 4))["C"].shape == (1, 5, 5, 3)


def test_reference_shape_ops():
    specs = [
        (make_transpose([2, 3, 4], [1, 2, 0]), (3, 4, 2)),
        (make_broadcast([1, 3, 1], [2, 3, 5]), (2, 3, 5)),
        (make_reshape([2, 3, 4], [4, 6]), (4, 6)),
        (make_slice([5, 7, 9], [1, 2, 3], [3, 4, 5]), (3, 4, 5)),
        (make_pad([2, 3], [1, 2], [3, 4]), (6, 9)),
        (make_concat([2, 3, 4], [2, 5, 4], axis=1), (2, 8, 4)),
    ]
    for seed, (spec, shape) in enumerate(specs, 10):
        assert run_reference(spec, sample_inputs(spec, seed))["C"].shape == shape


def test_reference_model_and_composite_ops():
    specs = [
        (make_batch_matmul(2, 3, 4, 5), (2, 3, 5)),
        (make_pool2d(1, 8, 8, 3, 3, 3, stride=2, pad=1, op="avg"), (1, 4, 4, 3)),
        (make_layer_norm([2, 3, 4], axis=2), (2, 3, 4)),
        (make_elem_reduce([2, 3, 4], axis=1, elem_op="mul", reduce_op="sum"), (2, 4)),
        (make_matmul_chain(2, 3, 4, 5), (2, 5)),
        (make_matmul_softmax(2, 3, 4), (2, 4)),
    ]
    for seed, (spec, shape) in enumerate(specs, 20):
        assert run_reference(spec, sample_inputs(spec, seed))["C"].shape == shape
