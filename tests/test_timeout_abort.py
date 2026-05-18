from __future__ import annotations

from asfuzz.config import ASFuzzConfig, BudgetConfig, FuzzerConfig
from asfuzz.runner import pipeline
from asfuzz.runner.worker import WorkerTimeout
from asfuzz.spec.ops_catalog import make_unary


class FakeBackend:
    def __init__(self, name: str):
        self.name = name

    def supports(self, spec):
        return True


class FakeMR:
    def __init__(self, name: str):
        self.name = name

    def applicable(self, spec):
        return True

    def variants(self, spec, inputs, seed):
        raise AssertionError("variants should not run after base timeout")


class FakeDB:
    def __init__(self):
        self.iterations = []
        self.bugs = []

    def record_iteration(self, spec, backend, mr, status, elapsed_ms):
        self.iterations.append((backend, mr, status))

    def record_bug(self, spec, backend, mr, status, repro_path, detail):
        self.bugs.append((backend, mr, status, detail))


def test_first_timeout_aborts_remaining_mrs_and_backends(tmp_path, monkeypatch):
    calls = []

    def timeout_backend_once(backend, spec, inputs, target, trials, seed, timeout_sec):
        calls.append((backend.name, trials))
        raise WorkerTimeout(f"{backend.name} timed out after {timeout_sec}s")

    monkeypatch.setattr(pipeline, "run_backend_once", timeout_backend_once)

    cfg = ASFuzzConfig(
        out_dir=str(tmp_path),
        db_path=str(tmp_path / "bugs.sqlite"),
        budget=BudgetConfig(compile_timeout_sec=1, run_timeout_sec=1),
        fuzzer=FuzzerConfig(max_work_items=1024),
    )
    spec = make_unary([3], op="tanh")
    db = FakeDB()
    result = pipeline.run_one_spec(
        cfg,
        spec,
        0,
        tmp_path / "cases",
        [FakeBackend("first"), FakeBackend("second")],
        [FakeMR("mr_a"), FakeMR("mr_b")],
        db,
    )

    assert calls == [("first", 0)]
    assert result["status"] == "failed"
    assert result["failures"][0]["status"] == "timeout"
    assert result["failures"][0]["backend"] == "first"
    assert result["failures"][0]["mr"] == "mr_a"
    assert result["skipped"] == [{"backend": "second", "reason": "case_aborted_after_timeout"}]
    assert db.iterations == [("first", "mr_a", "timeout")]
