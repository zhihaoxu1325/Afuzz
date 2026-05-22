from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation, identity
from asfuzz.spec.opspec import OpSpec


class ReduceSplitMR(MetamorphicRelation):
    name = "reduce_split"

    def applicable(self, spec: OpSpec) -> bool:
        if spec.op_kind != "reduce":
            return False
        if spec.extra.get("op") not in {"sum", "mean"}:
            return False
        axis = int(spec.extra["axis"])
        extent = spec.shape_of("A")[axis]
        return extent >= 4 and any(extent % factor == 0 for factor in (2, 4, 8))

    def variants(self, spec: OpSpec, inputs: dict[str, np.ndarray], seed: int) -> list[MRCase]:
        axis = int(spec.extra["axis"])
        extent = spec.shape_of("A")[axis]
        variants = []
        for factor in (2, 4, 8):
            if extent % factor != 0:
                continue
            new_spec = copy.deepcopy(spec)
            new_spec.op_kind = "reduce_split"
            new_spec.name = f"{spec.name}_split{factor}"
            new_spec.extra["split_factor"] = factor
            variants.append(MRCase(new_spec, {k: np.array(v, copy=True) for k, v in inputs.items()}, identity, seed, 0, f"split_factor_{factor}"))
        return variants[:2]
