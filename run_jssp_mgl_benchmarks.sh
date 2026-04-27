#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-0}"
shift 1 || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CMD=(
  "$PYTHON_BIN" -u scripts/eval_jssp_benchmarks.py
  --device cuda:0
  --B 128
  --greedy 0
)

CMD+=("$@")

echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Running: ${CMD[*]}"

env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
