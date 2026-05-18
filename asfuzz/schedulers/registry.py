from __future__ import annotations


def make_backend(name: str):
    if name == "numpy":
        from .numpy_backend import NumpyBackend

        return NumpyBackend()
    if name == "tvm":
        from .tvm_backend import TVMBackend

        return TVMBackend()
    if name == "metaschedule":
        from .metaschedule_backend import MetaScheduleBackend

        return MetaScheduleBackend()
    if name == "autotvm":
        from .autotvm_backend import AutoTVMBackend

        return AutoTVMBackend()
    if name == "halide":
        from .halide_backend import HalideBackend

        return HalideBackend()
    if name == "ansor":
        try:
            from .ansor_backend import AnsorBackend
        except Exception as exc:
            raise KeyError("backend ansor is unavailable in this TVM build") from exc
        return AnsorBackend()
    raise KeyError(f"unknown backend {name}")
