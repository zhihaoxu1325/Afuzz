#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-/doc2/zhzh/conda/envs/afuzz/bin/python}"
export TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH:-/doc2/zhzh/tvm/build/lib}"
export LD_LIBRARY_PATH="/doc2/zhzh/tvm/build/lib:${LD_LIBRARY_PATH:-}"
PYTHONPATH=. "$PY" -m asfuzz.cli run --config configs/nightly.yaml --resume
