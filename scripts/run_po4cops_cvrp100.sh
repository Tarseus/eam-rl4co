#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${ROOT_DIR}/PO4COPs/POMO/CVRP/POMO"

MODE="${1:-train}"
GPU_ID="${2:-5}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$MODE" in
  train)
    TARGET_SCRIPT="train_n100_po.py"
    ;;
  test)
    TARGET_SCRIPT="test_n100.py"
    ;;
  rl-train)
    TARGET_SCRIPT="train_n100_rl.py"
    ;;
  *)
    echo "Usage: $0 [train|test|rl-train] [physical_gpu_id]" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "work_dir=${WORK_DIR}"
echo "mode=${MODE}"
echo "physical_gpu_id=${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "process_device=cuda:0"

cd "${WORK_DIR}"
exec "${PYTHON_BIN}" "${TARGET_SCRIPT}"
