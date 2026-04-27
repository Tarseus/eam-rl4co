#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

FFSP50_LOSS_GPU="${FFSP50_LOSS_GPU:-0}"
FFSP50_WEIGHT_GPU="${FFSP50_WEIGHT_GPU:-1}"
JSSP10_LOSS_GPU="${JSSP10_LOSS_GPU:-3}"
JSSP15_LOSS_GPU="${JSSP15_LOSS_GPU:-4}"

FFSP50_LOSS_PAIR_PATH="${FFSP50_LOSS_PAIR_PATH:-runs/pref_loss_ffsp100_discovery/20260403-142801/best_pair.json}"
FFSP50_WEIGHT_PAIR_PATH="${FFSP50_WEIGHT_PAIR_PATH:-runs/pref_builder_weight_search_ffsp100/20260416-111514/best_pair.json}"
JSSP10_LOSS_PAIR_PATH="${JSSP10_LOSS_PAIR_PATH:-runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409/best_pair.json}"
JSSP15_LOSS_PAIR_PATH="${JSSP15_LOSS_PAIR_PATH:-runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409/best_pair.json}"

DRY_RUN="${DRY_RUN:-0}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}" "${ROOT_DIR}/logs/train/runs"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="${LOG_TZ:-Asia/Shanghai}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export HYDRA_FULL_ERROR=1

usage() {
  cat <<'EOF'
Usage:
  bash run_missing_pref_full_trains.sh

This script starts four full-train jobs only:
  1. FFSP50 loss-only pref full train
  2. FFSP50 weighting-only pref full train
  3. JSSP10x10 loss-only pref full train
  4. JSSP15x15 loss-only pref full train

Default GPUs:
  FFSP50_LOSS_GPU=1
  FFSP50_WEIGHT_GPU=2
  JSSP10_LOSS_GPU=3
  JSSP15_LOSS_GPU=4

Default best_pair.json paths:
  FFSP50_LOSS_PAIR_PATH=runs/pref_loss_ffsp100_discovery/20260403-142801/best_pair.json
  FFSP50_WEIGHT_PAIR_PATH=runs/pref_builder_weight_search_ffsp100/20260416-111514/best_pair.json
  JSSP10_LOSS_PAIR_PATH=runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409/best_pair.json
  JSSP15_LOSS_PAIR_PATH=runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409/best_pair.json

Notes:
  - No discovery/search is launched here.
  - JSSP15x15 defaults to reusing the existing JSSP10x10 loss-only best_pair.
EOF
}

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+=" $(printf '%q' "${arg}")"
  done
  printf '%s\n' "${out# }"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

prepare_bopo_jssp_data_if_needed() {
  if [[ "${PREPARED_BOPO_JSP_DATA:-0}" == "1" ]]; then
    return 0
  fi
  echo "[prep] scripts/prepare_bopo_jsp_data.py"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "${PYTHON_BIN}" scripts/prepare_bopo_jsp_data.py >/dev/null
  fi
  PREPARED_BOPO_JSP_DATA=1
}

run_fg() {
  echo "cmd=$(quote_cmd "$@")"
  if [[ "${DRY_RUN}" != "0" ]]; then
    return 0
  fi
  "$@"
}

worker_ffsp50_loss_only() {
  local run_dir="${ROOT_DIR}/logs/train/runs/ffsp50_loss_only_pref_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"

  require_file "${FFSP50_LOSS_PAIR_PATH}" "FFSP50 loss-only best_pair.json"

  run_fg \
    "${PYTHON_BIN}" -u run.py \
    "experiment=scheduling/ffsp-matnet-po-paper50" \
    "hydra.run.dir=${run_dir}" \
    "model.loss_type=free_loss" \
    "+model.pref_pair_json_path=${FFSP50_LOSS_PAIR_PATH}" \
    "~callbacks.learning_rate_monitor" \
    "~callbacks.rich_progress_bar" \
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}" \
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
    "callbacks.model_checkpoint.auto_insert_metric_name=False" \
    "callbacks.model_checkpoint.save_top_k=1" \
    "callbacks.model_checkpoint.save_last=True" \
    "trainer.accelerator=gpu" \
    "+trainer.devices=[0]" \
    "+trainer.enable_progress_bar=false" \
    "logger=csv" \
    "logger.csv.name=ffsp50_loss_only_pref"
}

worker_ffsp50_weighting() {
  local run_dir="${ROOT_DIR}/logs/train/runs/ffsp50_weighting_pref_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"

  require_file "${FFSP50_WEIGHT_PAIR_PATH}" "FFSP50 weighting-only best_pair.json"

  run_fg \
    "${PYTHON_BIN}" -u run.py \
    "experiment=scheduling/ffsp-matnet-po-paper50" \
    "hydra.run.dir=${run_dir}" \
    "model.loss_type=free_loss" \
    "+model.pref_pair_json_path=${FFSP50_WEIGHT_PAIR_PATH}" \
    "~callbacks.learning_rate_monitor" \
    "~callbacks.rich_progress_bar" \
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}" \
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
    "callbacks.model_checkpoint.auto_insert_metric_name=False" \
    "callbacks.model_checkpoint.save_top_k=1" \
    "callbacks.model_checkpoint.save_last=True" \
    "trainer.accelerator=gpu" \
    "+trainer.devices=[0]" \
    "+trainer.enable_progress_bar=false" \
    "logger=csv" \
    "logger.csv.name=ffsp50_weighting_pref"
}

worker_jssp10_loss_only() {
  local run_dir="${ROOT_DIR}/logs/train/runs/mgl-jssp-bopo-lossonly_10x10_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"

  require_file "${JSSP10_LOSS_PAIR_PATH}" "JSSP10x10 loss-only best_pair.json"
  prepare_bopo_jssp_data_if_needed

  run_fg \
    "${PYTHON_BIN}" -u run.py \
    "experiment=scheduling/mgl-jssp-bopo-paper" \
    "hydra.run.dir=${run_dir}" \
    "~callbacks.learning_rate_monitor" \
    "~callbacks.rich_progress_bar" \
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}" \
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
    "callbacks.model_checkpoint.auto_insert_metric_name=False" \
    "callbacks.model_checkpoint.save_top_k=1" \
    "callbacks.model_checkpoint.save_last=True" \
    "callbacks.model_checkpoint.every_n_epochs=1" \
    "trainer.accelerator=gpu" \
    "+trainer.devices=[0]" \
    "trainer.max_epochs=20" \
    "+trainer.enable_progress_bar=false" \
    "logger=csv" \
    "logger.csv.name=jssp10x10_loss_only_pref" \
    "+model.pref_pair_json_path=${JSSP10_LOSS_PAIR_PATH}" \
    "+model.allowed_shapes=[[10,10]]" \
    "+model.required_allowed_shapes=[[10,10]]" \
    "+model.expected_train_dataset_size=5000" \
    "+model.expected_val_dataset_size=100"
}

worker_jssp15_loss_only() {
  local run_dir="${ROOT_DIR}/logs/train/runs/mgl-jssp-bopo-lossonly_15x15_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"

  require_file "${JSSP15_LOSS_PAIR_PATH}" "JSSP15x15 loss-only best_pair.json"
  prepare_bopo_jssp_data_if_needed

  run_fg \
    "${PYTHON_BIN}" -u run.py \
    "experiment=scheduling/mgl-jssp-bopo-bucketed-multishape" \
    "hydra.run.dir=${run_dir}" \
    "~callbacks.learning_rate_monitor" \
    "~callbacks.rich_progress_bar" \
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}" \
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
    "callbacks.model_checkpoint.auto_insert_metric_name=False" \
    "callbacks.model_checkpoint.save_top_k=1" \
    "callbacks.model_checkpoint.save_last=True" \
    "callbacks.model_checkpoint.every_n_epochs=1" \
    "trainer.accelerator=gpu" \
    "+trainer.devices=[0]" \
    "trainer.max_epochs=20" \
    "+trainer.enable_progress_bar=false" \
    "logger=csv" \
    "logger.csv.name=jssp15x15_loss_only_pref" \
    "+model.pref_pair_json_path=${JSSP15_LOSS_PAIR_PATH}" \
    "+model.allowed_shapes=[[15,15]]" \
    "+model.required_allowed_shapes=[[15,15]]" \
    "+model.expected_train_dataset_size=5000" \
    "+model.expected_val_dataset_size=100"
}

run_worker() {
  local worker="$1"
  case "${worker}" in
    ffsp50-loss-only)
      worker_ffsp50_loss_only
      ;;
    ffsp50-weighting)
      worker_ffsp50_weighting
      ;;
    jssp10-loss-only)
      worker_jssp10_loss_only
      ;;
    jssp15-loss-only)
      worker_jssp15_loss_only
      ;;
    *)
      echo "ERROR: unknown worker: ${worker}" >&2
      exit 2
      ;;
  esac
}

spawn_worker() {
  local worker="$1"
  local gpu_id="$2"
  local log_path="${LOG_DIR}/${worker}_${TIMESTAMP}.out"
  local -a cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu_id}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "DRY_RUN=${DRY_RUN}"
    "FFSP50_LOSS_PAIR_PATH=${FFSP50_LOSS_PAIR_PATH}"
    "FFSP50_WEIGHT_PAIR_PATH=${FFSP50_WEIGHT_PAIR_PATH}"
    "JSSP10_LOSS_PAIR_PATH=${JSSP10_LOSS_PAIR_PATH}"
    "JSSP15_LOSS_PAIR_PATH=${JSSP15_LOSS_PAIR_PATH}"
    bash
    "$0"
    --worker
    "${worker}"
  )

  echo "[spawn] worker=${worker} gpu=${gpu_id}"
  echo "[spawn] log=${log_path}"
  echo "[spawn] cmd=$(quote_cmd "${cmd[@]}")"

  if [[ "${DRY_RUN}" != "0" ]]; then
    return 0
  fi

  nohup "${cmd[@]}" >"${log_path}" 2>&1 < /dev/null &
  echo "[spawn] pid=$!"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "ERROR: --worker requires a worker name" >&2
    exit 2
  fi
  worker_name="$2"
  shift 2
  run_worker "${worker_name}"
  exit 0
fi

spawn_worker "ffsp50-loss-only" "${FFSP50_LOSS_GPU}"
spawn_worker "ffsp50-weighting" "${FFSP50_WEIGHT_GPU}"
spawn_worker "jssp10-loss-only" "${JSSP10_LOSS_GPU}"
spawn_worker "jssp15-loss-only" "${JSSP15_LOSS_GPU}"
