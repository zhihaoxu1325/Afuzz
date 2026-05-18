from __future__ import annotations

from .mr_algebraic import AlgebraicMR
from .mr_axis_permute import AxisPermuteMR
from .mr_batch_split import BatchSplitMR
from .mr_budget_monotonic import BudgetMonotonicMR
from .mr_decomposition import DecompositionMR
from .mr_layout import LayoutMR
from .mr_pad_slice import PadSliceMR
from .mr_seed_invariance import SeedInvarianceMR


def make_mr(name: str):
    mapping = {
        "seed_invariance": SeedInvarianceMR,
        "budget_monotonic": BudgetMonotonicMR,
        "axis_permute": AxisPermuteMR,
        "pad_slice": PadSliceMR,
        "batch_split": BatchSplitMR,
        "layout": LayoutMR,
        "algebraic": AlgebraicMR,
        "decomposition": DecompositionMR,
    }
    if name not in mapping:
        raise KeyError(f"unknown MR {name}")
    return mapping[name]()

