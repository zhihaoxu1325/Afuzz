from __future__ import annotations

import copy

import numpy as np

from asfuzz.mr.base import MRCase, MetamorphicRelation, identity
from asfuzz.spec.opspec import OpSpec


class MatmulChainAssociativityMR(MetamorphicRelation):
    name = "matmul_chain_assoc"

    def applicable(self, spec: OpSpec) -> bool:
        return spec.op_kind == "matmul_chain"

    def variants(self, spec: OpSpec, inputs: dict[str, np.ndarray], seed: int) -> list[MRCase]:
        new_spec = copy.deepcopy(spec)
        old_order = spec.extra.get("order", "left")
        new_order = "right" if old_order == "left" else "left"
        new_spec.extra["order"] = new_order
        new_spec.name = f"{spec.name}_{new_order}"
        new_inputs = {name: np.array(arr, copy=True) for name, arr in inputs.items()}
        return [MRCase(new_spec, new_inputs, identity, seed, 0, f"{old_order}_vs_{new_order}")]
