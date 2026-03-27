#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-0}"
shift 1 || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

"$PYTHON_BIN" scripts/prepare_bopo_jsp_data.py

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=scheduling/mgl-jssp-bopo-paper"
  "trainer.accelerator=gpu"
  "+trainer.devices=[0]"
  "logger=csv"
  "logger.csv.name=bopo"
)

CMD+=("$@")

echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Running: ${CMD[*]}"

env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
