#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

epochs="${1:-100}"
seed="${2:-1234}"
log_dir="${3:-logs/op100_mechanism_ablation}"
shift $(( $# >= 3 ? 3 : $# ))
extra_overrides=("$@")

variants=(resample random_only ls_only eam)
gpus=(0 1 2 3)

mkdir -p "${log_dir}"

pids=()
pid_to_variant=()

for idx in "${!variants[@]}"; do
  variant="${variants[$idx]}"
  gpu="${gpus[$idx]}"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  stdout_log="${log_dir}/op100_${variant}_seed${seed}_${timestamp}.log"
  exit_log="${stdout_log%.log}.exitcode"

  echo "Starting OP100 ${variant} on cuda:${gpu}"
  echo "  stdout: ${stdout_log}"
  echo "  exit  : ${exit_log}"

  (
    cleanup() {
      status=$?
      {
        echo "[run] variant=${variant}"
        echo "[run] gpu=${gpu}"
        echo "[run] finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "[run] exit_code=${status}"
      } | tee -a "${stdout_log}"
      printf "%s\n" "${status}" > "${exit_log}"
    }

    trap cleanup EXIT
    trap 'echo "[run] variant='"${variant}"' received_signal=INT at $(date '\''+%Y-%m-%d %H:%M:%S'\'')" | tee -a "'"${stdout_log}"'"' INT
    trap 'echo "[run] variant='"${variant}"' received_signal=TERM at $(date '\''+%Y-%m-%d %H:%M:%S'\'')" | tee -a "'"${stdout_log}"'"' TERM

    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONUNBUFFERED=1 \
    python run.py \
      experiment=routing/op100_am_mechanism \
      trainer.accelerator=gpu \
      ++trainer.devices=1 \
      trainer.max_epochs="${epochs}" \
      seed="${seed}" \
      logger.csv.name="csv/op100_${variant}/" \
      model.mechanism.variant="${variant}" \
      model.ea_kwargs.improve_mode="${variant}" \
      model.ea_kwargs.val_improve_mode="${variant}" \
      "${extra_overrides[@]}" 2>&1 | tee "${stdout_log}"
  ) &

  pids+=("$!")
  pid_to_variant+=("${variant}")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  variant="${pid_to_variant[$idx]}"
  if ! wait "${pid}"; then
    echo "Variant ${variant} failed"
    status=1
  fi
done

exit "${status}"
