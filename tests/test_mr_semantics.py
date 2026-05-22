from asfuzz.mr.mr_axis_move import AxisMoveMR
from asfuzz.mr.mr_batch_split import BatchSplitMR
from asfuzz.mr.mr_decomposition import DecompositionMR
from asfuzz.mr.mr_input_scale import InputScaleMR
from asfuzz.mr.mr_layout import LayoutMR
from asfuzz.mr.mr_matmul_chain_assoc import MatmulChainAssociativityMR
from asfuzz.mr.mr_matmul_transpose import MatmulTransposeMR
from asfuzz.mr.mr_reduce_split import ReduceSplitMR
from asfuzz.mr.mr_shape_identity import ShapeIdentityMR
from asfuzz.oracle.numeric import numeric_equal
from asfuzz.spec.ops_catalog import make_concat, make_conv2d, make_elementwise, make_matmul, make_matmul_chain, make_pad, make_reduce, make_reshape, make_softmax, make_transpose
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


def test_matmul_transpose_dual():
    _assert_variant_matches_reference(MatmulTransposeMR(), make_matmul(5, 7, 3, act="relu"))


def test_axis_move_reduce_non_keepdims():
    _assert_variant_matches_reference(AxisMoveMR(), make_reduce([3, 5, 7], axis=0, op="sum"))


def test_axis_move_reduce_keepdims():
    _assert_variant_matches_reference(AxisMoveMR(), make_reduce([3, 5, 7], axis=2, op="mean", keepdims=True))


def test_axis_move_softmax():
    _assert_variant_matches_reference(AxisMoveMR(), make_softmax([3, 5, 7], axis=0))


def test_input_scale_reduce_and_matmul():
    _assert_variant_matches_reference(InputScaleMR(), make_reduce([3, 5, 7], axis=1, op="max"))
    _assert_variant_matches_reference(InputScaleMR(), make_matmul(5, 7, 3, act="relu"))


def test_reduce_split():
    _assert_variant_matches_reference(ReduceSplitMR(), make_reduce([3, 8, 5], axis=1, op="sum"))
    _assert_variant_matches_reference(ReduceSplitMR(), make_reduce([3, 16, 5], axis=1, op="mean", keepdims=True))


def test_matmul_chain_associativity():
    _assert_variant_matches_reference(MatmulChainAssociativityMR(), make_matmul_chain(3, 4, 5, 6))


def test_shape_identity_roundtrips():
    _assert_variant_matches_reference(ShapeIdentityMR(), make_transpose([2, 3, 4], [1, 2, 0]))
    _assert_variant_matches_reference(ShapeIdentityMR(), make_reshape([2, 3, 4], [4, 6]))
    _assert_variant_matches_reference(ShapeIdentityMR(), make_pad([2, 3, 4], [1, 0, 2], [0, 3, 1]))
    _assert_variant_matches_reference(ShapeIdentityMR(), make_concat([2, 3, 4], [2, 5, 4], axis=1))
