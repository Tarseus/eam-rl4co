#!/usr/bin/env bash
set -euo pipefail

model="${1:-eam-pomo}"
epochs="${2:-200}"
log_dir="${3:-logs}"
problem="${4:-cvrp}"
problem_size="${5:-100}"
extra_args=()
if [[ $# -gt 5 ]]; then
  extra_args=("${@:6}")
fi

seeds=(0 1 2 3 4)
gpus=(1 2 3 4 5)

pids=()
for idx in "${!seeds[@]}"; do
  seed="${seeds[$idx]}"
  gpu="${gpus[$idx]}"
  
  log_file="${log_dir}/${model}_${problem}${problem_size}_seed${seed}.log"
  
  echo "Starting ${model} ${problem}${problem_size} seed ${seed} on GPU ${gpu}, logging to ${log_file}"
  
  nohup python run_cvrp100_pomo_eam.py \
    --model "${model}" \
    --problem "${problem}" \
    --problem-size "${problem_size}" \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --device "${gpu}" \
    --log-dir "${log_dir}" \
    --run-name "${model}_${problem}${problem_size}_seed${seed}" \
    "${extra_args[@]}" > "${log_file}" 2>&1 &
    
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
