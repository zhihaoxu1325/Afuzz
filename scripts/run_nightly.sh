#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-/home/tsmc193/GraphCAD/miniconda3/envs/spike/bin/python}"
PYTHONPATH=. "$PY" -m asfuzz.cli run --config configs/nightly.yaml --resume
