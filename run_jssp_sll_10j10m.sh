#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-0}"
shift 1 || true

EXP_NAME="scheduling/mgl-jssp-sll-paper"
MAX_EPOCHS=20

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

mkdir -p "${ROOT_DIR}/logs"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/train/runs/mgl-jssp-sll_10x10_${TS}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_PATH="${ROOT_DIR}/logs/mgl-jssp-sll_10x10_${TS}.out"

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=${EXP_NAME}"
  "hydra.run.dir=${RUN_DIR}"
  # Prevent mixed-shape training: restrict dataset to 10x10 only.
  "model.allowed_shapes=[[10,10]]"
  "model.required_allowed_shapes=[[10,10]]"
  "~callbacks.learning_rate_monitor"
  "~callbacks.rich_progress_bar"
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"
  "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
  "callbacks.model_checkpoint.auto_insert_metric_name=False"
  "callbacks.model_checkpoint.save_top_k=-1"
  "callbacks.model_checkpoint.save_last=True"
  "callbacks.model_checkpoint.every_n_epochs=1"
  "trainer.accelerator=gpu"
  "+trainer.devices=[0]"
  "trainer.max_epochs=${MAX_EPOCHS}"
  "+trainer.enable_progress_bar=false"
  "logger=csv"
  "logger.csv.name=mgl-jssp-sll_10x10"
)

CMD+=("$@")

echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Running: ${CMD[*]}"
echo "Log: ${LOG_PATH}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
TRAIN_PID=$!

echo "Started PID: ${TRAIN_PID}"
tail -f "${LOG_PATH}"
