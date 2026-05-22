from __future__ import annotations

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

import tvm

from asfuzz.lowering.to_te import lower_to_schedule
from asfuzz.lowering.to_tir import lower_to_tir
from asfuzz.schedulers.base import CompiledArtifact, SchedulerBackend
from asfuzz.spec.ops_catalog import SUPPORTED_OP_KINDS
from asfuzz.spec.opspec import OpSpec


class TVMBackend(SchedulerBackend):
    name = "tvm"
    schedule_policy = "default"

    def supports(self, spec: OpSpec) -> bool:
        return spec.op_kind in (SUPPORTED_OP_KINDS | {"softmax_decomposed", "reduce_split"}) and spec.dtype() in {"float32", "float16"}

    def schedule_and_build(self, spec: OpSpec, target: str, trials: int, seed: int) -> CompiledArtifact:
        _require_target_codegen(target)
        if _has_te_schedule():
            schedule, args = lower_to_schedule(spec)
            _apply_schedule_variation(schedule, args[-1], spec, seed, trials, self.schedule_policy)
            module = tvm.build(schedule, args, target=target, name=f"asfuzz_{spec.signature()}")
            return CompiledArtifact(self.name, spec, target, trials, seed, handle=(module, args))
        prim = lower_to_tir(spec)
        module = tvm.build(tvm.IRModule({"main": prim}), target=target)
        return CompiledArtifact(self.name, spec, target, trials, seed, handle=(module, None))

    def run(self, artifact: CompiledArtifact, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        module, _args = artifact.handle
        dev = tvm.cpu(0)
        tvm_args = []
        for tensor in artifact.spec.tensors:
            if tensor.role in {"input", "weight", "bias"}:
                tvm_args.append(_tvm_tensor(np.ascontiguousarray(inputs[tensor.name]), dev))
        outputs = artifact.spec.tensors_by_role("output")
        out_arrays = []
        for output in outputs:
            out = _tvm_empty(artifact.spec.shape_of(output), output.dtype, dev)
            out_arrays.append(out)
            tvm_args.append(out)
        module(*tvm_args)
        return {output.name: out.numpy() for output, out in zip(outputs, out_arrays)}


def _apply_schedule_variation(schedule, output, spec: OpSpec, seed: int, trials: int, policy: str) -> None:
    if policy == "default" and trials == 0:
        return
    axes = list(output.op.axis)
    if not axes:
        return
    last = axes[-1]
    extent = int(spec.shape_of(spec.tensors_by_role("output")[0])[-1])
    factor = 4 if (seed + trials) % 2 == 0 else 8
    factor = max(1, min(factor, extent))
    try:
        outer, inner = schedule[output].split(last, factor=factor)
        if len(axes) >= 2:
            schedule[output].parallel(axes[0])
        schedule[output].vectorize(inner)
        if policy in {"ansor", "metaschedule", "autotvm"} and len(axes) >= 2:
            schedule[output].reorder(*axes[:-1], outer, inner)
    except Exception:
        # Schedule fuzzing must never turn an otherwise valid semantic case
        # into a harness error; unsupported transforms fall back to default.
        return


def _has_te_schedule() -> bool:
    try:
        from tvm import te

        return hasattr(te, "create_schedule")
    except Exception:
        return False


def _tvm_tensor(arr: np.ndarray, dev):
    if hasattr(tvm, "nd"):
        return tvm.nd.array(arr, dev)
    return tvm.runtime.tensor(arr, dev)


def _tvm_empty(shape, dtype: str, dev):
    if hasattr(tvm, "nd"):
        return tvm.nd.empty(shape, dtype=dtype, device=dev)
    return tvm.runtime.empty(shape, dtype=dtype, device=dev)


def _require_target_codegen(target: str) -> None:
    kind = target.split()[0]
    if kind == "llvm" and tvm.get_global_func("target.build.llvm", True) is None:
        raise RuntimeError(
            "current TVM is built without LLVM codegen; rebuild latest TVM with USE_LLVM=ON "
            "or set target to a runnable backend"
        )
