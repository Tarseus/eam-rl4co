#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

epochs="${1:-100}"
seed="${2:-1234}"
log_dir="${3:-logs/mechanism_op100}"
extra_overrides=("${@:4}")

variants=(resample random_only ls_only eam)
gpus=(0 1 2 3)

mkdir -p "${log_dir}"

pids=()
for idx in "${!variants[@]}"; do
  variant="${variants[$idx]}"
  gpu="${gpus[$idx]}"
  log_file="${log_dir}/op100_${variant}.log"

  echo "Starting OP100 ${variant} on cuda:${gpu} -> ${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONUNBUFFERED=1 \
  nohup python run.py \
    experiment=routing/op100_am_mechanism \
    trainer.accelerator=gpu \
    +trainer.devices=1 \
    trainer.max_epochs="${epochs}" \
    seed="${seed}" \
    model.mechanism.variant="${variant}" \
    model.ea_kwargs.improve_mode="${variant}" \
    model.ea_kwargs.val_improve_mode="${variant}" \
    "${extra_overrides[@]}" > "${log_file}" 2>&1 &

  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done

exit "${status}"
