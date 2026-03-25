#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

epochs="${1:-100}"
seed="${2:-1234}"
gpu="${3:-0}"
log_dir="${4:-logs/op100_ls_only_single}"

mkdir -p "${log_dir}"

timestamp="$(date +%Y%m%d_%H%M%S)"
stdout_log="${log_dir}/op100_ls_only_seed${seed}_${timestamp}.log"
exit_log="${stdout_log%.log}.exitcode"
hydra_run_dir="logs/train/runs/op100_ls_only_seed${seed}_${timestamp}"

echo "Starting OP100 ls_only on cuda:${gpu}"
echo "  stdout: ${stdout_log}"
echo "  hydra : ${hydra_run_dir}"

(
  run_status=0

  log_exit() {
    status="${1:-0}"
    {
      echo "[run] variant=ls_only"
      echo "[run] gpu=${gpu}"
      echo "[run] finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "[run] exit_code=${status}"
    } | tee -a "${stdout_log}"
    printf "%s\n" "${status}" > "${exit_log}"
  }

  on_signal() {
    signal_name="$1"
    signal_status="$2"
    echo "[run] variant=ls_only received_signal=${signal_name} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${stdout_log}"
    log_exit "${signal_status}"
    exit "${signal_status}"
  }

  trap 'on_signal INT 130' INT
  trap 'on_signal TERM 143' TERM

  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONUNBUFFERED=1 \
  python run.py \
    experiment=routing/op100_am_mechanism \
    "hydra.run.dir=${hydra_run_dir}" \
    trainer.accelerator=gpu \
    ++trainer.devices=1 \
    trainer.max_epochs="${epochs}" \
    seed="${seed}" \
    logger.csv.name="csv/op100_ls_only/" \
    model.mechanism.variant=ls_only \
    model.ea_kwargs.improve_mode=ls_only \
    model.ea_kwargs.val_improve_mode=ls_only \
    2>&1 | tee "${stdout_log}" || run_status=$?

  log_exit "${run_status}"
  exit "${run_status}"
)
