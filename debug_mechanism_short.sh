#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

problem="${1:-cvrp}"
variant="${2:-eam}"
gpu="${3:-0}"
max_steps="${4:-300}"
seed="${5:-1234}"
log_dir="${6:-logs/mechanism_debug}"
shift $(( $# >= 6 ? 6 : $# ))
extra_overrides=("$@")

case "${problem}" in
  cvrp)
    experiment="routing/cvrp100_pomo_mechanism"
    run_name="cvrp100_${variant}_debug"
    ;;
  op)
    experiment="routing/op100_am_mechanism"
    run_name="op100_${variant}_debug"
    ;;
  *)
    echo "Unsupported problem: ${problem}. Use 'cvrp' or 'op'."
    exit 1
    ;;
esac

case "${variant}" in
  eam|random_only|ls_only|resample)
    ;;
  *)
    echo "Unsupported variant: ${variant}. Use 'eam', 'random_only', 'ls_only', or 'resample'."
    exit 1
    ;;
esac

mkdir -p "${log_dir}"
timestamp="$(date +%Y%m%d_%H%M%S)"
stdout_log="${log_dir}/${run_name}_steps${max_steps}_seed${seed}_${timestamp}.log"
exit_log="${stdout_log%.log}.exitcode"

echo "Problem: ${problem}"
echo "Variant: ${variant}"
echo "GPU: cuda:${gpu}"
echo "Max steps: ${max_steps}"
echo "Seed: ${seed}"
echo "Stdout log: ${stdout_log}"

cleanup() {
  status=$?
  {
    echo "[debug] finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[debug] exit_code=${status}"
  } | tee -a "${stdout_log}"
  printf "%s\n" "${status}" > "${exit_log}"
}

trap cleanup EXIT
trap 'echo "[debug] received_signal=INT at $(date '\''+%Y-%m-%d %H:%M:%S'\'')" | tee -a "${stdout_log}"' INT
trap 'echo "[debug] received_signal=TERM at $(date '\''+%Y-%m-%d %H:%M:%S'\'')" | tee -a "${stdout_log}"' TERM

CUDA_VISIBLE_DEVICES="${gpu}" \
PYTHONUNBUFFERED=1 \
python run.py \
  experiment="${experiment}" \
  trainer.accelerator=gpu \
  +trainer.devices=1 \
  trainer.max_steps="${max_steps}" \
  trainer.max_epochs=9999 \
  trainer.limit_val_batches=0 \
  trainer.num_sanity_val_steps=0 \
  trainer.log_every_n_steps=1 \
  test=False \
  seed="${seed}" \
  logger.csv.name="csv/${run_name}/" \
  callbacks.model_checkpoint.save_last=false \
  callbacks.model_checkpoint.save_top_k=0 \
  model.mechanism.variant="${variant}" \
  model.mechanism.log_every=1 \
  model.ea_kwargs.improve_mode="${variant}" \
  model.ea_kwargs.val_improve_mode="${variant}" \
  "${extra_overrides[@]}" 2>&1 | tee "${stdout_log}"
