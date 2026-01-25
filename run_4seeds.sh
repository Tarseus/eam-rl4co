#!/usr/bin/env bash
set -euo pipefail

model="${1:-pomo}"
epochs="${2:-100}"
log_dir="${3:-logs}"

seeds=(0 1 2 3)
gpus=(1 2 3 4)

pids=()
for idx in "${!seeds[@]}"; do
  seed="${seeds[$idx]}"
  gpu="${gpus[$idx]}"
  
  log_file="${log_dir}/${model}_seed${seed}.log"
  
  echo "Starting seed ${seed} on GPU ${gpu}, logging to ${log_file}"
  
  nohup python run_cvrp100_pomo_eam.py \
    --problem tsp \
    --problem-size 50 \
    --model "${model}" \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --device "${gpu}" \
    --log-dir "${log_dir}" \
    --run-name "${model}_cvrp100_seed${seed}" > "${log_file}" 2>&1 &
    
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
