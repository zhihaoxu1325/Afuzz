from asfuzz.spec.ops_catalog import make_conv2d, make_elementwise, make_matmul, make_reduce, make_softmax
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
    sm = make_softmax([2, 3])
    assert run_reference(sm, sample_inputs(sm, 3))["C"].shape == (2, 3)
    conv = make_conv2d(1, 5, 5, 2, 3, 3, 3, pad=1)
    assert run_reference(conv, sample_inputs(conv, 4))["C"].shape == (1, 5, 5, 3)
