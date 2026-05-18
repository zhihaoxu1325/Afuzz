from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EinsumExpr:
    text: str

    def free_indices(self) -> set[str]:
        if "[" not in self.text or "]" not in self.text:
            return set()
        lhs = self.text.split("=", 1)[0]
        inside = lhs[lhs.find("[") + 1 : lhs.find("]")]
        return {part.strip() for part in inside.split(",") if part.strip()}

    def reduce_indices(self) -> set[str]:
        out = set()
        for token in self.text.replace("{", "_").replace("}", " ").split():
            if token.startswith("sum_") or token.startswith("max_"):
                out.add(token.split("_", 1)[1])
        return out


def parse(text: str) -> EinsumExpr:
    return EinsumExpr(text=text.strip())

