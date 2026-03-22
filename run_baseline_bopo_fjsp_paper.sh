#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-0}"
shift || true

MAX_EPOCHS=20
MID_EPOCH=10
MID_SRC="epoch_009.ckpt"
MID_ALIAS="epoch_010.ckpt"
FINAL_SRC="epoch_019.ckpt"
FINAL_ALIAS="epoch_020.ckpt"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

mkdir -p "${ROOT_DIR}/baseline" "${ROOT_DIR}/logs"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/train/runs/bopo_fjsp_${TS}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_PATH="${ROOT_DIR}/logs/bopo_fjsp_${TS}.out"

if ! "$PYTHON_BIN" -c "import torch_geometric" >/dev/null 2>&1; then
  echo "Missing dependency: torch_geometric" >&2
  echo "Install it first, for example:" >&2
  echo "  pip install torch_geometric" >&2
  echo "or from the repo root:" >&2
  echo "  pip install -e '.[graph]'" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=scheduling/bopo-fjsp-paper-10j5m"
  "hydra.run.dir=${RUN_DIR}"
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
  "logger.csv.name=bopo_fjsp"
)

CMD+=("$@")

echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Running: ${CMD[*]}"
echo "Log: ${LOG_PATH}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
TRAIN_PID=$!

(
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    if [[ -f "${CKPT_DIR}/${MID_SRC}" && ! -f "${CKPT_DIR}/${MID_ALIAS}" ]]; then
      cp -f "${CKPT_DIR}/${MID_SRC}" "${CKPT_DIR}/${MID_ALIAS}"
    fi
    if [[ -f "${CKPT_DIR}/${FINAL_SRC}" && ! -f "${CKPT_DIR}/${FINAL_ALIAS}" ]]; then
      cp -f "${CKPT_DIR}/${FINAL_SRC}" "${CKPT_DIR}/${FINAL_ALIAS}"
    fi
    if [[ -f "${CKPT_DIR}/${MID_ALIAS}" ]]; then
      cp -f "${CKPT_DIR}/${MID_ALIAS}" "${ROOT_DIR}/baseline/bopo_fjsp_paper_10j5m_epoch_${MID_EPOCH}.ckpt"
    fi
    if [[ -f "${CKPT_DIR}/${FINAL_ALIAS}" ]]; then
      cp -f "${CKPT_DIR}/${FINAL_ALIAS}" "${ROOT_DIR}/baseline/bopo_fjsp_paper_10j5m_epoch_${MAX_EPOCHS}.ckpt"
    fi
    if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
      cp -f "${CKPT_DIR}/last.ckpt" "${ROOT_DIR}/baseline/bopo_fjsp_paper_10j5m_last.ckpt"
    fi
    sleep 15
  done

  if [[ -f "${CKPT_DIR}/${MID_SRC}" ]]; then
    cp -f "${CKPT_DIR}/${MID_SRC}" "${CKPT_DIR}/${MID_ALIAS}"
  fi
  if [[ -f "${CKPT_DIR}/${FINAL_SRC}" ]]; then
    cp -f "${CKPT_DIR}/${FINAL_SRC}" "${CKPT_DIR}/${FINAL_ALIAS}"
  elif [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
    cp -f "${CKPT_DIR}/last.ckpt" "${CKPT_DIR}/${FINAL_ALIAS}"
  fi
  if [[ -f "${CKPT_DIR}/${MID_ALIAS}" ]]; then
    cp -f "${CKPT_DIR}/${MID_ALIAS}" "${ROOT_DIR}/baseline/bopo_fjsp_paper_10j5m_epoch_${MID_EPOCH}.ckpt"
  fi
  if [[ -f "${CKPT_DIR}/${FINAL_ALIAS}" ]]; then
    cp -f "${CKPT_DIR}/${FINAL_ALIAS}" "${ROOT_DIR}/baseline/bopo_fjsp_paper_10j5m_epoch_${MAX_EPOCHS}.ckpt"
  fi
  if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
    cp -f "${CKPT_DIR}/last.ckpt" "${ROOT_DIR}/baseline/bopo_fjsp_paper_10j5m_last.ckpt"
  fi
) >/dev/null 2>&1 &

echo "Started PID: ${TRAIN_PID}"
tail -f "${LOG_PATH}"
