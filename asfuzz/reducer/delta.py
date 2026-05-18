from __future__ import annotations

from asfuzz.spec.opspec import OpSpec


def shrink_shape_once(spec: OpSpec) -> OpSpec:
    """Return a conservative one-step smaller copy for manual triage.

    Full delta debugging depends on repeatedly replaying failures; this helper
    keeps the reducer module useful without hiding expensive policy decisions
    inside the campaign runner.
    """
    clone = spec.model_copy(deep=True)
    for axis in clone.axes.values():
        if axis.size > 1:
            axis.size = max(1, axis.size // 2)
            clone.name = f"{spec.name}_shrunk"
            break
    return clone

