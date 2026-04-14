#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-both}"
GPU_RL="0"
GPU_PO="0"

case "${MODE}" in
  rl|po)
    GPU_RL="${2:-0}"
    shift $(( $# >= 2 ? 2 : $# )) || true
    ;;
  both)
    GPU_RL="${2:-0}"
    GPU_PO="${3:-${GPU_RL}}"
    if [ $# -ge 3 ]; then
      shift 3
    else
      shift $#
    fi
    ;;
  *)
    ;;
esac

PO_ALPHA="${PO_ALPHA:-0.25}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p "${ROOT_DIR}/logs"
TS="$(date +%Y%m%d-%H%M%S)"

launch_run() {
  local label="$1"
  local gpu_id="$2"
  local experiment="$3"
  shift 3

  local run_dir="${ROOT_DIR}/logs/train/runs/${label}_${TS}"
  local ckpt_dir="${run_dir}/checkpoints"
  local log_path="${ROOT_DIR}/logs/${label}_${TS}.out"

  local cmd=(
    "$PYTHON_BIN" -u run.py
    "experiment=${experiment}"
    "hydra.run.dir=${run_dir}"
    "~callbacks.learning_rate_monitor"
    "~callbacks.rich_progress_bar"
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}"
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
    "logger.csv.name=${label}"
  )
  cmd+=("$@")

  echo "CUDA_VISIBLE_DEVICES=${gpu_id}"
  echo "Run dir: ${run_dir}"
  echo "Running (${label}): ${cmd[*]}"
  echo "Log: ${log_path}"

  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >"${log_path}" 2>&1 &
  local pid=$!
  echo "Started ${label} PID: ${pid}"
}

case "${MODE}" in
  rl)
    launch_run \
      "mgl-jssp-rl_10x10" \
      "${GPU_RL}" \
      "scheduling/mgl-jssp-rl-paper" \
      "$@"
    ;;
  po)
    launch_run \
      "mgl-jssp-po_10x10" \
      "${GPU_RL}" \
      "scheduling/mgl-jssp-po-paper" \
      "model.po_alpha=${PO_ALPHA}" \
      "$@"
    ;;
  both)
    launch_run \
      "mgl-jssp-rl_10x10" \
      "${GPU_RL}" \
      "scheduling/mgl-jssp-rl-paper" \
      "$@"
    launch_run \
      "mgl-jssp-po_10x10" \
      "${GPU_PO}" \
      "scheduling/mgl-jssp-po-paper" \
      "model.po_alpha=${PO_ALPHA}" \
      "$@"
    ;;
  *)
    echo "Usage:"
    echo "  $0 rl <gpu_id> [extra hydra overrides...]"
    echo "  $0 po <gpu_id> [extra hydra overrides...]"
    echo "  $0 both <gpu_rl> <gpu_po> [extra hydra overrides...]"
    echo
    echo "Environment overrides:"
    echo "  PO_ALPHA=0.25 MAX_EPOCHS=20 PYTHON_BIN=python"
    exit 1
    ;;
esac
