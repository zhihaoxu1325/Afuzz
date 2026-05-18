from __future__ import annotations

from collections import Counter

from asfuzz.fuzzer.coverage import CoverageTracker, work_items
from asfuzz.fuzzer.grammar import GrammarFuzzer
from asfuzz.spec.validate import validate


def test_diverse_sampler_prefers_unique_valid_complex_cases():
    max_work_items = 16_000_000
    fuzzer = GrammarFuzzer(
        123,
        {
            "matmul": 2.5,
            "elementwise": 1.5,
            "unary": 1.0,
            "reduce": 2.0,
            "softmax": 2.0,
            "conv2d": 2.5,
        },
        ["float32"],
        "stress",
    )
    coverage = CoverageTracker()
    specs = []

    for _ in range(40):
        spec = fuzzer.sample_diverse(
            coverage,
            max_work_items,
            candidates=64,
            min_complexity=0.08,
            complexity_weight=0.55,
            novelty_weight=1.0,
        )
        assert validate(spec, max_work_items).ok
        coverage.add(spec)
        specs.append(spec)

    signatures = [spec.signature() for spec in specs]
    assert len(signatures) == len(set(signatures))
    assert max(work_items(spec) for spec in specs) > 1_000_000
    assert len(Counter(spec.op_kind for spec in specs)) >= 4
