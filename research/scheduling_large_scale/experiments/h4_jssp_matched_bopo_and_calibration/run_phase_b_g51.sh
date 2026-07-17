#!/usr/bin/env bash
set -euo pipefail

repo=/data1/gushengda/eam-rl4co
python_bin=/data1/gushengda/anaconda3/envs/rlco1/bin/python
checkpoint=downloads/jssp15x15/weighting/checkpoint.ckpt
tag=20260718_0148
root=logs/scheduling_large_scale/h4_calibrated/jssp50x20/phase_b_screen300_${tag}

labels=(
  usw_lr1e6_wd1e6
  usw_lr5e7_wd1e6
  usw_lr1e6_wd0
  asw_lr2e6_a0
  asw_lr1e6_a0
  asw_lr1e6_a025
)
methods=(usw usw usw asw asw asw)
learning_rates=(1e-6 5e-7 1e-6 2e-6 1e-6 1e-6)
weight_decays=(1e-6 1e-6 0 1e-6 1e-6 1e-6)
alphas=(1.0 1.0 1.0 0.0 0.0 0.25)
requested=("$@")

is_requested() {
  local candidate="$1"
  if [[ ${#requested[@]} -eq 0 ]]; then
    return 0
  fi
  local item
  for item in "${requested[@]}"; do
    [[ "$item" == "$candidate" ]] && return 0
  done
  return 1
}

cd "$repo"
mkdir -p "$(dirname "$root")" logs/codex_remote

# Preflight every target before the first run so a partially old sequence can
# never be mistaken for one clean six-configuration screen.
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  is_requested "$label" || continue
  output_dir="$root/$label"
  log_path="logs/codex_remote/h4_phase_b_${label}_${tag}.log"
  if [[ -e "$output_dir" || -e "$log_path" ]]; then
    echo "refusing_existing label=$label output=$output_dir log=$log_path" >&2
    exit 1
  fi
done

echo "phase_b_start time=$(date -Is) root=$repo/$root gpu=${CUDA_VISIBLE_DEVICES:-unset}"
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  is_requested "$label" || continue
  method="${methods[$index]}"
  learning_rate="${learning_rates[$index]}"
  weight_decay="${weight_decays[$index]}"
  alpha="${alphas[$index]}"
  output_dir="$root/$label"
  log_path="logs/codex_remote/h4_phase_b_${label}_${tag}.log"

  echo "variant_start label=$label time=$(date -Is) output=$repo/$output_dir log=$repo/$log_path"
  env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
    "$python_bin" scripts/train_jssp_large_objectives.py \
      --method "$method" \
      --checkpoint "$checkpoint" \
      --steps 300 \
      --learning-rate "$learning_rate" \
      --weight-decay "$weight_decay" \
      --alpha "$alpha" \
      --validation-count 32 \
      --validation-batch-size 8 \
      --validation-every 50 \
      --output-dir "$output_dir" \
      --device cuda:0 2>&1 | tee "$log_path"

  test -s "$output_dir/history.jsonl"
  test -s "$output_dir/best.ckpt"
  echo "variant_complete label=$label time=$(date -Is)"
done
echo "phase_b_complete time=$(date -Is) root=$repo/$root"
