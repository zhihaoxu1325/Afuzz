from asfuzz.mr.mr_batch_split import BatchSplitMR
from asfuzz.mr.mr_decomposition import DecompositionMR
from asfuzz.mr.mr_layout import LayoutMR
from asfuzz.oracle.numeric import numeric_equal
from asfuzz.spec.ops_catalog import make_conv2d, make_elementwise, make_softmax
from asfuzz.spec.reference import run_reference, sample_inputs


def _assert_variant_matches_reference(mr, spec):
    inputs = sample_inputs(spec, 123)
    base = run_reference(spec, inputs)
    assert mr.applicable(spec)
    for variant in mr.variants(spec, inputs, 123):
        expected = variant.expected_recover_fn(base)
        actual = variant.recover_fn(run_reference(variant.spec, variant.inputs))
        assert numeric_equal(expected["C"], actual["C"], spec.dtype()).ok


def test_batch_split_prefix_suffix():
    _assert_variant_matches_reference(BatchSplitMR(), make_elementwise([5, 3], "add"))


def test_batch_split_rejects_softmax_on_split_axis():
    assert not BatchSplitMR().applicable(make_softmax([5, 3], axis=0))
    assert BatchSplitMR().applicable(make_softmax([5, 3], axis=1))


def test_layout_conv2d_swap_hw():
    _assert_variant_matches_reference(LayoutMR(), make_conv2d(1, 5, 7, 2, 3, 3, 1, pad=1))


def test_softmax_decomposition():
    _assert_variant_matches_reference(DecompositionMR(), make_softmax([2, 5]))


def test_softmax_decomposition_preserves_float16_inputs():
    spec = make_softmax([2, 5], dtype="float16")
    inputs = sample_inputs(spec, 123)
    variants = DecompositionMR().variants(spec, inputs, 123)
    assert variants[0].inputs["A"].dtype == inputs["A"].dtype
