#!/usr/bin/env bash
set -euo pipefail

repo=/data1/gushengda/eam-rl4co
python_bin=/data1/gushengda/anaconda3/envs/rlco1/bin/python
checkpoint=downloads/jssp15x15/weighting/checkpoint.ckpt
tag=20260717_2043
requested=("$@")

cd "$repo"
mkdir -p logs/codex_remote

launch_one() {
  local variant="$1"
  local gpu="$2"
  local method="$3"
  local lr="$4"
  local alpha="$5"
  local output="logs/scheduling_large_scale/h3_calibrated/jssp50x20/screen100/${variant}_${tag}"
  local log_path="logs/codex_remote/h3_screen100_${variant}_${tag}.log"

  if [[ -e "$output" || -e "$log_path" ]]; then
    echo "refusing_existing variant=$variant output=$output log=$log_path" >&2
    return 1
  fi
  mkdir -p "$output"

  setsid -f bash -lc "
    echo \$\$ > '$repo/$output/pid'
    exec env PYTHONPATH='$repo' CUDA_VISIBLE_DEVICES='$gpu' \
      '$python_bin' '$repo/scripts/train_jssp_large_objectives.py' \
      --method '$method' \
      --checkpoint '$checkpoint' \
      --steps 100 \
      --learning-rate '$lr' \
      --weight-decay 1e-6 \
      --alpha '$alpha' \
      --validation-count 32 \
      --validation-batch-size 8 \
      --validation-every 50 \
      --output-dir '$output' \
      --device cuda:0
  " > "$log_path" 2>&1 < /dev/null

  for _ in $(seq 1 30); do
    [[ -s "$output/pid" ]] && break
    sleep 0.1
  done
  if [[ ! -s "$output/pid" ]]; then
    echo "missing_pid variant=$variant log=$repo/$log_path" >&2
    return 1
  fi
  local pid
  pid=$(tr -d '[:space:]' < "$output/pid")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "exited_early variant=$variant pid=$pid log=$repo/$log_path" >&2
    tail -n 40 "$log_path" >&2 || true
    return 1
  fi
  echo "variant=$variant gpu=$gpu pid=$pid log=$repo/$log_path output=$repo/$output"
}

is_requested() {
  local variant="$1"
  if [[ ${#requested[@]} -eq 0 ]]; then
    return 0
  fi
  local item
  for item in "${requested[@]}"; do
    [[ "$item" == "$variant" ]] && return 0
  done
  return 1
}

is_requested usw_lr5e6 && launch_one usw_lr5e6 0 usw 5e-6 1.0
is_requested usw_lr2e6 && launch_one usw_lr2e6 1 usw 2e-6 1.0
is_requested asw_lr5e6_a1 && launch_one asw_lr5e6_a1 2 asw 5e-6 1.0
is_requested asw_lr5e6_a0 && launch_one asw_lr5e6_a0 3 asw 5e-6 0.0
is_requested asw_lr5e6_a025 && launch_one asw_lr5e6_a025 4 asw 5e-6 0.25
is_requested asw_lr5e6_a05 && launch_one asw_lr5e6_a05 6 asw 5e-6 0.5
