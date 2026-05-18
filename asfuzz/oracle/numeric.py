from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EqualResult:
    ok: bool
    max_abs: float = 0.0
    max_rel: float = 0.0
    index: tuple[int, ...] | None = None
    reason: str = ""


def numeric_equal(a: np.ndarray, b: np.ndarray, dtype: str, rtol: float | None = None, atol: float | None = None) -> EqualResult:
    if a.shape != b.shape:
        return EqualResult(False, reason=f"shape mismatch {a.shape} vs {b.shape}")
    default_rtol = {"float32": 1e-3, "float16": 1e-2, "bfloat16": 5e-2}.get(dtype, 1e-3)
    default_atol = {"float32": 1e-4, "float16": 1e-3, "bfloat16": 5e-3}.get(dtype, 1e-4)
    rtol = default_rtol if rtol is None else rtol
    atol = default_atol if atol is None else atol

    a = np.asarray(a)
    b = np.asarray(b)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return EqualResult(False, reason="NaN pattern mismatch")
    if not np.array_equal(np.isposinf(a), np.isposinf(b)) or not np.array_equal(np.isneginf(a), np.isneginf(b)):
        return EqualResult(False, reason="Inf pattern mismatch")

    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return EqualResult(True)
    diff = np.abs(a[finite].astype("float64") - b[finite].astype("float64"))
    denom = np.maximum(np.abs(a[finite].astype("float64")), atol)
    rel = diff / denom
    ok_mask = diff <= (atol + rtol * np.abs(a[finite].astype("float64")))
    if bool(np.all(ok_mask)):
        return EqualResult(True, float(np.max(diff)), float(np.max(rel)))
    flat_bad = int(np.argmax(~ok_mask))
    finite_indices = np.argwhere(finite)
    idx = tuple(int(x) for x in finite_indices[flat_bad])
    return EqualResult(False, float(np.max(diff)), float(np.max(rel)), idx, "value mismatch")

