#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-0}"
SIZE="${2:-20}"
shift 2 || true

case "$SIZE" in
  20)
    EXP_NAME="scheduling/ffsp-matnet-po-paper20"
    MAX_EPOCHS=100
    MID_EPOCH=50
    MID_SRC_INDEX=49
    FINAL_SRC_INDEX=99
    ;;
  50)
    EXP_NAME="scheduling/ffsp-matnet-po-paper50"
    MAX_EPOCHS=150
    MID_EPOCH=75
    MID_SRC_INDEX=74
    FINAL_SRC_INDEX=149
    ;;
  100)
    EXP_NAME="scheduling/ffsp-matnet-po-paper100"
    MAX_EPOCHS=200
    MID_EPOCH=100
    MID_SRC_INDEX=99
    FINAL_SRC_INDEX=199
    ;;
  *)
    echo "Unsupported FFSP size: ${SIZE}. Use 20, 50, or 100." >&2
    exit 1
    ;;
esac

MID_SRC="$(printf "epoch_%03d.ckpt" "${MID_SRC_INDEX}")"
MID_ALIAS="$(printf "epoch_%03d.ckpt" "${MID_EPOCH}")"
FINAL_SRC="$(printf "epoch_%03d.ckpt" "${FINAL_SRC_INDEX}")"
FINAL_ALIAS="$(printf "epoch_%03d.ckpt" "${MAX_EPOCHS}")"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

mkdir -p "${ROOT_DIR}/baseline" "${ROOT_DIR}/logs"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/train/runs/ffsp_matnet_po_${SIZE}_${TS}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_PATH="${ROOT_DIR}/logs/ffsp_matnet_po_${SIZE}_${TS}.out"

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=${EXP_NAME}"
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
  "logger.csv.name=ffsp_matnet_po_${SIZE}"
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
      cp -f "${CKPT_DIR}/${MID_ALIAS}" "${ROOT_DIR}/baseline/ffsp_matnet_po_paper${SIZE}_epoch_${MID_EPOCH}.ckpt"
    fi
    if [[ -f "${CKPT_DIR}/${FINAL_ALIAS}" ]]; then
      cp -f "${CKPT_DIR}/${FINAL_ALIAS}" "${ROOT_DIR}/baseline/ffsp_matnet_po_paper${SIZE}_epoch_${MAX_EPOCHS}.ckpt"
    fi
    if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
      cp -f "${CKPT_DIR}/last.ckpt" "${ROOT_DIR}/baseline/ffsp_matnet_po_paper${SIZE}_last.ckpt"
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
    cp -f "${CKPT_DIR}/${MID_ALIAS}" "${ROOT_DIR}/baseline/ffsp_matnet_po_paper${SIZE}_epoch_${MID_EPOCH}.ckpt"
  fi
  if [[ -f "${CKPT_DIR}/${FINAL_ALIAS}" ]]; then
    cp -f "${CKPT_DIR}/${FINAL_ALIAS}" "${ROOT_DIR}/baseline/ffsp_matnet_po_paper${SIZE}_epoch_${MAX_EPOCHS}.ckpt"
  fi
  if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
    cp -f "${CKPT_DIR}/last.ckpt" "${ROOT_DIR}/baseline/ffsp_matnet_po_paper${SIZE}_last.ckpt"
  fi
) >/dev/null 2>&1 &

echo "Started PID: ${TRAIN_PID}"
tail -f "${LOG_PATH}"
