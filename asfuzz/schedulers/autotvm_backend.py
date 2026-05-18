from __future__ import annotations

from .tvm_backend import TVMBackend


class AutoTVMBackend(TVMBackend):
    name = "autotvm"
    schedule_policy = "autotvm"
