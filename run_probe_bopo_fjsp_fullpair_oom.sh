#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-0}"
K_VALUE="${2:-16}"
shift 2 || true

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

mkdir -p "${ROOT_DIR}/logs"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/train/runs/bopo_fjsp_fullpair_probe_${TS}"
LOG_PATH="${ROOT_DIR}/logs/bopo_fjsp_fullpair_probe_${TS}.out"

if ! "$PYTHON_BIN" -c "import torch_geometric" >/dev/null 2>&1; then
  echo "Missing dependency: torch_geometric" >&2
  echo "Install it first, for example:" >&2
  echo "  pip install -e '.[graph]'" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=scheduling/bopo-fjsp-paper-10j5m"
  "hydra.run.dir=${RUN_DIR}"
  "test=false"
  "~callbacks.learning_rate_monitor"
  "~callbacks.rich_progress_bar"
  "~callbacks.model_checkpoint"
  "trainer.accelerator=gpu"
  "+trainer.devices=[0]"
  "trainer.max_epochs=1"
  "++trainer.limit_train_batches=1"
  "++trainer.limit_val_batches=0"
  "++trainer.num_sanity_val_steps=0"
  "++trainer.enable_progress_bar=false"
  "++trainer.log_every_n_steps=1"
  "logger=csv"
  "logger.csv.name=bopo_fjsp_fullpair_probe"
  "model.K=${K_VALUE}"
  "++model.pair_mode=all_pairs"
  "model.metrics.train=[loss,reward,pair_count]"
)

CMD+=("$@")

echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Running: ${CMD[*]}"
echo "Log: ${LOG_PATH}"
echo "Expected all-pairs count per instance: $(( K_VALUE * (K_VALUE - 1) / 2 ))"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
TRAIN_PID=$!

echo "Started PID: ${TRAIN_PID}"
tail -f "${LOG_PATH}"
