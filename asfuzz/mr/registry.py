from __future__ import annotations

from .mr_algebraic import AlgebraicMR
from .mr_axis_permute import AxisPermuteMR
from .mr_batch_split import BatchSplitMR
from .mr_budget_monotonic import BudgetMonotonicMR
from .mr_decomposition import DecompositionMR
from .mr_axis_move import AxisMoveMR
from .mr_input_scale import InputScaleMR
from .mr_layout import LayoutMR
from .mr_matmul_transpose import MatmulTransposeMR
from .mr_matmul_chain_assoc import MatmulChainAssociativityMR
from .mr_pad_slice import PadSliceMR
from .mr_reduce_split import ReduceSplitMR
from .mr_seed_invariance import SeedInvarianceMR
from .mr_shape_identity import ShapeIdentityMR


def make_mr(name: str):
    mapping = {
        "seed_invariance": SeedInvarianceMR,
        "budget_monotonic": BudgetMonotonicMR,
        "axis_permute": AxisPermuteMR,
        "pad_slice": PadSliceMR,
        "batch_split": BatchSplitMR,
        "layout": LayoutMR,
        "matmul_transpose": MatmulTransposeMR,
        "matmul_chain_assoc": MatmulChainAssociativityMR,
        "axis_move": AxisMoveMR,
        "input_scale": InputScaleMR,
        "reduce_split": ReduceSplitMR,
        "shape_identity": ShapeIdentityMR,
        "algebraic": AlgebraicMR,
        "decomposition": DecompositionMR,
    }
    if name not in mapping:
        raise KeyError(f"unknown MR {name}")
    return mapping[name]()
