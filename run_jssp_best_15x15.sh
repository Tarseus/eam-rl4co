#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-0}"
shift 1 || true

# Usage:
#   ./run_jssp_best_15x15.sh [GPU_ID] [RUN_DIR|BEST_PAIR_JSON] [hydra overrides...]
# Examples:
#   ./run_jssp_best_15x15.sh 4
#   ./run_jssp_best_15x15.sh 4 20260417-123033
#   ./run_jssp_best_15x15.sh 4 runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033/best_pair.json
#   ./run_jssp_best_15x15.sh 4 trainer.max_epochs=30

RUNS_ROOT="runs/pref_builder_weight_search_jssp10x10_from_best_loss"
EXPERIMENT="scheduling/mgl-jssp-bopo-bucketed-multishape"
SHAPE_LITERAL="[[15,15]]"
EXPECTED_TRAIN_SIZE=5000
EXPECTED_VAL_SIZE=100
MAX_EPOCHS="${MAX_EPOCHS:-20}"

if [[ ! -d "$RUNS_ROOT" ]]; then
  echo "ERROR: runs_root does not exist: $RUNS_ROOT" >&2
  exit 1
fi

best_pair_path=""
if [[ $# -gt 0 && "${1:-}" != *=* && "${1:-}" != -* ]]; then
  if [[ -f "$1" && "$(basename "$1")" == "best_pair.json" ]]; then
    best_pair_path="$1"
    shift
  elif [[ -f "${RUNS_ROOT}/$1/best_pair.json" ]]; then
    best_pair_path="${RUNS_ROOT}/$1/best_pair.json"
    shift
  fi
fi

latest="$(
  "$PYTHON_BIN" - "$RUNS_ROOT" <<'PY'
import os
import sys

runs_root = sys.argv[1]
cands = []
for name in os.listdir(runs_root):
    path = os.path.join(runs_root, name)
    if not os.path.isdir(path):
        continue
    best_pair = os.path.join(path, "best_pair.json")
    if not os.path.isfile(best_pair):
        continue
    try:
        mtime = os.path.getmtime(best_pair)
    except OSError:
        mtime = 0.0
    cands.append((mtime, name))

if not cands:
    sys.exit(2)

cands.sort()
print(cands[-1][1])
PY
)" || true

if [[ -z "${latest:-}" ]]; then
  echo "ERROR: no run dirs with best_pair.json found under: $RUNS_ROOT" >&2
  exit 1
fi

if [[ -z "${best_pair_path:-}" ]]; then
  best_pair_path="${RUNS_ROOT}/${latest}/best_pair.json"
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL
: "${HYDRA_FULL_ERROR:=1}"
export HYDRA_FULL_ERROR

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/logs/train/runs"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/train/runs/mgl-jssp-bopo-best_15x15_${TS}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_PATH="${ROOT_DIR}/logs/mgl-jssp-bopo-best_15x15_${TS}.out"

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=${EXPERIMENT}"
  "hydra.run.dir=${RUN_DIR}"
  "~callbacks.learning_rate_monitor"
  "~callbacks.rich_progress_bar"
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"
  "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
  "callbacks.model_checkpoint.auto_insert_metric_name=False"
  "callbacks.model_checkpoint.save_top_k=1"
  "callbacks.model_checkpoint.save_last=True"
  "callbacks.model_checkpoint.every_n_epochs=1"
  "trainer.accelerator=gpu"
  "+trainer.devices=[0]"
  "trainer.max_epochs=${MAX_EPOCHS}"
  "+trainer.enable_progress_bar=false"
  "logger=csv"
  "logger.csv.name=mgl_jssp_bopo_best_15x15"
  "+model.pref_pair_json_path=${best_pair_path}"
  "+model.allowed_shapes=${SHAPE_LITERAL}"
  "+model.required_allowed_shapes=${SHAPE_LITERAL}"
  "+model.expected_train_dataset_size=${EXPECTED_TRAIN_SIZE}"
  "+model.expected_val_dataset_size=${EXPECTED_VAL_SIZE}"
)

CMD+=("$@")

echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Latest search run: ${RUNS_ROOT}/${latest}"
echo "Using best_pair: ${best_pair_path}"
echo "Run dir: ${RUN_DIR}"
echo "Running: ${CMD[*]}"
echo "Log: ${LOG_PATH}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
TRAIN_PID=$!

echo "Started PID: ${TRAIN_PID}"
tail -f "${LOG_PATH}"
