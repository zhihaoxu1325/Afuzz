from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time

from asfuzz.schedulers.base import SchedulerBackend
from asfuzz.schedulers.registry import make_backend
from asfuzz.spec.opspec import OpSpec


class WorkerError(RuntimeError):
    pass


class WorkerTimeout(TimeoutError):
    pass


def _run_child(queue, backend_name: str, spec: OpSpec, inputs, target: str, trials: int, seed: int):
    _limit_native_threads()
    start = time.time()
    try:
        backend = make_backend(backend_name)
        artifact = backend.schedule_and_build(spec, target=target, trials=trials, seed=seed)
        outputs = backend.run(artifact, inputs)
        queue.put(("ok", outputs, (time.time() - start) * 1000.0))
    except Exception as exc:
        queue.put(("error", {"type": type(exc).__name__, "message": str(exc)}, (time.time() - start) * 1000.0))


def _limit_native_threads() -> None:
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TVM_NUM_THREADS",
    ]:
        os.environ.setdefault(name, "1")


def run_backend_once(backend: SchedulerBackend, spec: OpSpec, inputs, target: str, trials: int, seed: int, timeout_sec: int | None = None):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_child, args=(queue, backend.name, spec, inputs, target, trials, seed))
    proc.start()
    ready = wait([queue._reader, proc.sentinel], timeout=timeout_sec)
    if queue._reader in ready:
        status, payload, elapsed_ms = queue.get()
        proc.join(3)
        if proc.is_alive():
            proc.terminate()
            proc.join(3)
        if proc.exitcode not in (0, None):
            raise WorkerError(f"{backend.name} worker exited with code {proc.exitcode}")
        if status == "error":
            raise WorkerError(f"{payload['type']}: {payload['message']}")
        return payload, elapsed_ms
    proc.join(0)
    if proc.is_alive():
        proc.terminate()
        proc.join(3)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise WorkerTimeout(f"{backend.name} timed out after {timeout_sec}s")
    if proc.exitcode != 0:
        raise WorkerError(f"{backend.name} worker exited with code {proc.exitcode}")
    raise WorkerError(f"{backend.name} worker produced no result")
