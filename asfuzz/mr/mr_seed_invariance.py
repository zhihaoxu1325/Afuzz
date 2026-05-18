from __future__ import annotations

from asfuzz.mr.base import MRCase, MetamorphicRelation, identity
from asfuzz.spec.opspec import OpSpec


class SeedInvarianceMR(MetamorphicRelation):
    name = "seed_invariance"

    def applicable(self, spec: OpSpec) -> bool:
        return True

    def variants(self, spec: OpSpec, inputs, seed: int) -> list[MRCase]:
        return [
            MRCase(spec, inputs, identity, seed + 1, 0, "seed+1"),
            MRCase(spec, inputs, identity, seed + 2, 0, "seed+2"),
        ]

