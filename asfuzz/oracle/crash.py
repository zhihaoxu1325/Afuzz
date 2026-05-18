from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrashInfo:
    signal: int | None = None
    stderr_tail: str = ""
    stack_trace: str = ""


class CrashError(RuntimeError):
    def __init__(self, info: CrashInfo):
        super().__init__(info.stderr_tail or "backend crash")
        self.info = info

