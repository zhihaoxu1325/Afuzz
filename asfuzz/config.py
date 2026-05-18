from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BudgetConfig(BaseModel):
    iterations: int = 100
    trials_smoke: int = 0
    trials_full: int = 0
    compile_timeout_sec: int = 60
    run_timeout_sec: int = 30
    max_workers: int = 1
    cpu_utilization: float = 0.8


class FuzzerConfig(BaseModel):
    mode: str = "grammar"
    max_work_items: int = 4_000_000
    op_weights: dict[str, float] = Field(default_factory=lambda: {"matmul": 1.0, "elementwise": 1.0, "unary": 1.0, "reduce": 1.0, "softmax": 1.0, "conv2d": 1.0})
    dtypes: list[str] = Field(default_factory=lambda: ["float32"])
    complexity: str = "stress"
    diversity_candidates: int = 96
    min_complexity_score: float = 0.0
    complexity_weight: float = 0.35
    novelty_weight: float = 1.0


class OracleConfig(BaseModel):
    rtol: dict[str, float] = Field(default_factory=lambda: {"float32": 1e-3})
    atol: dict[str, float] = Field(default_factory=lambda: {"float32": 1e-4})


class ASFuzzConfig(BaseModel):
    seed: int = 42
    target: str = "llvm"
    out_dir: str = "runs/latest"
    db_path: str = "runs/latest/asfuzz_bugs.sqlite"
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    backends: list[str] = Field(default_factory=lambda: ["numpy", "tvm"])
    fuzzer: FuzzerConfig = Field(default_factory=FuzzerConfig)
    mrs: list[str] = Field(default_factory=lambda: ["seed_invariance"])
    oracle: OracleConfig = Field(default_factory=OracleConfig)


def load_config(path: str | Path) -> ASFuzzConfig:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    return ASFuzzConfig.model_validate(data)
