#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_ID="${1:-0}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/train/runs/cvrp100_baseline_${TS}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_PATH="${LOG_DIR}/cvrp100_baseline_${TS}.out"

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=routing/pomo-po4cops-cvrp100-po"
  "hydra.run.dir=${RUN_DIR}"
  "~callbacks.learning_rate_monitor"
  "~callbacks.rich_progress_bar"
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"
  "callbacks.model_checkpoint.filename=epoch_{epoch:03d}"
  "callbacks.model_checkpoint.auto_insert_metric_name=False"
  "callbacks.model_checkpoint.save_top_k=-1"
  "callbacks.model_checkpoint.save_last=True"
  "callbacks.model_checkpoint.every_n_epochs=100"
  "trainer.devices=[0]"
  "logger=csv"
  "logger.csv.name=cvrp100_baseline"
)

echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Running: ${CMD[*]}"
echo "Log: ${LOG_PATH}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
TRAIN_PID=$!

(
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    for alias_epoch in 100 200; do
      src_epoch="$(printf "%03d" $((alias_epoch - 1)))"
      dst_epoch="${alias_epoch}"
      if [[ -f "${CKPT_DIR}/epoch_${src_epoch}.ckpt" && ! -f "${CKPT_DIR}/epoch_${dst_epoch}.ckpt" ]]; then
        cp -f "${CKPT_DIR}/epoch_${src_epoch}.ckpt" "${CKPT_DIR}/epoch_${dst_epoch}.ckpt"
      fi
    done
    sleep 15
  done

  for alias_epoch in 100 200; do
    src_epoch="$(printf "%03d" $((alias_epoch - 1)))"
    dst_epoch="${alias_epoch}"
    if [[ -f "${CKPT_DIR}/epoch_${src_epoch}.ckpt" ]]; then
      cp -f "${CKPT_DIR}/epoch_${src_epoch}.ckpt" "${CKPT_DIR}/epoch_${dst_epoch}.ckpt"
    elif [[ "${alias_epoch}" == "200" && -f "${CKPT_DIR}/last.ckpt" ]]; then
      cp -f "${CKPT_DIR}/last.ckpt" "${CKPT_DIR}/epoch_${dst_epoch}.ckpt"
    fi
  done
  if [[ -f "${CKPT_DIR}/epoch_100.ckpt" ]]; then
    cp -f "${CKPT_DIR}/epoch_100.ckpt" "${ROOT_DIR}/baseline/cvrp100_epoch_100.ckpt"
  fi
  if [[ -f "${CKPT_DIR}/epoch_200.ckpt" ]]; then
    cp -f "${CKPT_DIR}/epoch_200.ckpt" "${ROOT_DIR}/baseline/cvrp100_epoch_200.ckpt"
  fi
) >/dev/null 2>&1 &

echo "Started PID: ${TRAIN_PID}"
tail -f "${LOG_PATH}"
