# ASFuzz

ASFuzz is an auto-scheduler correctness fuzzer for DL compilers.

The runnable slice implements the main architecture from `Code.md`:

- unified `OpSpec`
- grammar-based generation for `matmul`, `elementwise`, `unary`, `reduce`, `softmax`, and NHWC `conv2d`
- NumPy reference oracle
- TVM TE/TIR lowering plus real tuning backends:
  - `metaschedule`: `tune_tir` JSON database + `compile_tir`, supporting both older `tvm.meta_schedule` and current `tvm.s_tir.meta_schedule`
  - `tvm`/`autotvm`: fast TE schedule variants
- real Halide lowering and `Pipeline.apply_autoscheduler` backend for the supported scalar/reduction ops
- metamorphic checks for seed invariance, budget invariance, algebraic identity, axis permutation, pad/slice, batch prefix/suffix split, conv2d H/W layout swap, and softmax decomposition/shift invariance
- SQLite bug database and JSON campaign summaries
- per-failure `repro.py` generation
- diversity-guided generation: resume restores existing signatures, candidates are scored by op/rank/work-bucket/shape-feature novelty plus complexity, and long runs avoid repeating equivalent specs

Run a smoke campaign:

```bash
PYTHONPATH=. /home/tsmc193/GraphCAD/miniconda3/envs/spike/bin/python -m asfuzz.cli run --config configs/smoke.yaml
```

Replay a saved case:

```bash
PYTHONPATH=. /home/tsmc193/GraphCAD/miniconda3/envs/spike/bin/python -m asfuzz.cli replay --case runs/latest/cases/case_000000/spec.json
```

Switch Halide autoscheduler:

```bash
ASFUZZ_HALIDE_AUTOSCHEDULER=Adams2019 ./scripts/run_smoke.sh
```

Use the locally built latest TVM:

```bash
export LD_LIBRARY_PATH=/home/tsmc193/zhzh/libtest/tvm/build/lib:$LD_LIBRARY_PATH
export TVM_LIBRARY_PATH=/home/tsmc193/zhzh/libtest/tvm/build/lib
PYTHONPATH=. /home/tsmc193/GraphCAD/miniconda3/envs/spike/bin/python -m asfuzz.cli run --config configs/smoke.yaml
```

The current `/home/tsmc193/zhzh/libtest/tvm` import is wired into the `spike` environment through
`asfuzz_local_tvm.pth`. This build reports `tvm 0.25.dev0`, exposes
`tvm.s_tir.meta_schedule`, and has LLVM codegen enabled for configs whose `target` is `llvm`.

Useful outputs:

- `runs/*/summary.json`
- `runs/*/asfuzz_bugs.sqlite`
- `runs/*/cases/case_*/spec.json`
- `runs/*/cases/case_*/repro.py` for failing cases

Current scope:

- Halide supports `matmul`, `elementwise`, `unary`, `reduce`, and last-axis `softmax`.
- TVM supports those plus NHWC `conv2d`.
- MetaSchedule JSON databases are written under `runs/tuning/metaschedule/...`.
- Ansor is no longer in the default configs because current TVM main has removed the old `tvm.auto_scheduler` Python API.
- Tuning is used when an MR passes `trials > 0` such as `budget_monotonic`; `trials = 0` keeps the fast compile path for smoke and replay.

Fuzzer complexity controls:

- `fuzzer.complexity`: `small`, `medium`, `large`, or `stress`; nightly uses `stress`.
- `fuzzer.max_work_items`: upper bound for output elements multiplied by reduction extents.
- `fuzzer.diversity_candidates`: number of sampled candidates scored before accepting one case.
- `fuzzer.min_complexity_score`: rejects candidates below the requested complexity floor.
- `fuzzer.complexity_weight` and `fuzzer.novelty_weight`: trade off large/heavy cases against rare op/rank/work-bucket/shape features.

With the current nightly config, the generator targets irregular high-stress shapes up to
`16_000_000` work items and records `work_items` plus `complexity_score` in each
`result.json`. On `--resume`, existing `spec.json` files are added back to the
seen set, so completed cases are not regenerated, and incomplete cases reuse their
old spec instead of being overwritten.

Parallel execution:

- `budget.max_workers > 0` uses that many concurrent cases.
- `budget.max_workers: 0` auto-selects `floor(os.cpu_count() * budget.cpu_utilization)`.
- Nightly uses `max_workers: 0` and `cpu_utilization: 0.8`; on the current 28-thread
  server that means 22 concurrent cases.
- Worker subprocesses set common native thread env vars to `1` by default, avoiding
  oversubscription when many cases run at once.
