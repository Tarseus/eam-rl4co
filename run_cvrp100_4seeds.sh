#!/usr/bin/env bash
set -euo pipefail

model="${1:-pomo}"
epochs="${2:-200}"
log_dir="${3:-logs}"

seeds=(0 1 2 3)
gpus=(0 1 2 3)

pids=()
for idx in "${!seeds[@]}"; do
  seed="${seeds[$idx]}"
  gpu="${gpus[$idx]}"
  python run_cvrp100_pomo_eam.py \
    --model "${model}" \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --device "${gpu}" \
    --log-dir "${log_dir}" \
    --run-name "${model}_cvrp100_seed${seed}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
