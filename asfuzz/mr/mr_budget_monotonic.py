from __future__ import annotations

from asfuzz.mr.base import MRCase, MetamorphicRelation, identity
from asfuzz.spec.opspec import OpSpec


class BudgetMonotonicMR(MetamorphicRelation):
    name = "budget_monotonic"

    def applicable(self, spec: OpSpec) -> bool:
        return True

    def variants(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        return [
            MRCase(spec, inputs, identity, seed, 0, "trials=0"),
            MRCase(spec, inputs, identity, seed, 64, "trials=64"),
        ]
