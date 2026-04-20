#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Usage:
#   ./run_full_train_latest_pair_jssp10x10.sh [EXPERIMENT] [RUN_DIR|BEST_PAIR_JSON] [hydra overrides...]
# Examples:
#   ./run_full_train_latest_pair_jssp10x10.sh
#   ./run_full_train_latest_pair_jssp10x10.sh scheduling/mgl-jssp-po-paper
#   ./run_full_train_latest_pair_jssp10x10.sh scheduling/mgl-jssp-rl-paper 20260417-123033 trainer.devices=[0]
#   ./run_full_train_latest_pair_jssp10x10.sh 20260417-123033 trainer.devices=[0]
#   ./run_full_train_latest_pair_jssp10x10.sh runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033/best_pair.json trainer.devices=[0]

RUNS_ROOT="runs/pref_builder_weight_search_jssp10x10_from_best_loss"
EXPERIMENT_DEFAULT="scheduling/mgl-jssp-bopo-paper"

if [[ ! -d "$RUNS_ROOT" ]]; then
  echo "ERROR: runs_root does not exist: $RUNS_ROOT" >&2
  exit 1
fi

experiment="$EXPERIMENT_DEFAULT"
if [[ $# -gt 0 && "${1:-}" == scheduling/* ]]; then
  experiment="$1"
  shift
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
export LOG_TZ="${LOG_TZ:-Asia/Shanghai}"
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL
: "${HYDRA_FULL_ERROR:=1}"
export HYDRA_FULL_ERROR

"$PYTHON_BIN" scripts/prepare_bopo_jsp_data.py

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/logs/train/runs"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/train/runs/mgl-jssp-bopo-pref_10x10_${TS}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_PATH="${ROOT_DIR}/logs/mgl-jssp-bopo-pref_10x10_${TS}.out"

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=${experiment}"
  "hydra.run.dir=${RUN_DIR}"
  "~callbacks.learning_rate_monitor"
  "~callbacks.rich_progress_bar"
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"
  "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
  "callbacks.model_checkpoint.auto_insert_metric_name=False"
  "callbacks.model_checkpoint.save_top_k=1"
  "callbacks.model_checkpoint.save_last=True"
  "callbacks.model_checkpoint.every_n_epochs=1"
  "logger=csv"
  "logger.csv.name=bopo_pref"
  "+model.pref_pair_json_path=${best_pair_path}"
)

has_devices_override=false
has_accelerator_override=false
has_progress_override=false
has_allowed_shapes_override=false
for arg in "$@"; do
  case "$arg" in
    trainer.devices=*)
      has_devices_override=true
      ;;
    trainer.accelerator=*)
      has_accelerator_override=true
      ;;
    trainer.enable_progress_bar=*)
      has_progress_override=true
      ;;
    model.allowed_shapes=*|+model.allowed_shapes=*)
      has_allowed_shapes_override=true
      ;;
  esac
done

if [[ "$has_accelerator_override" == "false" ]]; then
  CMD+=("trainer.accelerator=gpu")
fi
if [[ "$has_devices_override" == "false" ]]; then
  CMD+=("+trainer.devices=[0]")
fi
if [[ "$has_progress_override" == "false" ]]; then
  CMD+=("+trainer.enable_progress_bar=false")
fi
if [[ "$has_allowed_shapes_override" == "false" ]]; then
  # Keep the 10x10 full-train run on the same shape-restricted data regime as the baseline.
  CMD+=("+model.allowed_shapes=[[10,10]]")
fi

CMD+=("$@")

echo "Latest search run: ${RUNS_ROOT}/${latest}"
echo "Experiment: ${experiment}"
echo "Using best_pair: ${best_pair_path}"
echo "Run dir: ${RUN_DIR}"
echo "Running: ${CMD[*]}"
echo "Log: ${LOG_PATH}"

nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
echo "Started PID: $!"
