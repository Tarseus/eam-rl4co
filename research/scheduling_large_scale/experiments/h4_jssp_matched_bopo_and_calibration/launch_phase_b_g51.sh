#!/usr/bin/env bash
set -euo pipefail

repo=/data1/gushengda/eam-rl4co
python_bin=/data1/gushengda/anaconda3/envs/rlco1/bin/python
checkpoint=downloads/jssp15x15/weighting/checkpoint.ckpt

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one audited idle GPU}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "refusing_multiple_visible_gpus=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TAG" >&2
  exit 2
fi

tag="$1"
root="logs/scheduling_large_scale/h4_phase_b/jssp50x20/screen300_${tag}"

labels=(
  usw_lr1e6_wd1e6
  usw_lr5e7_wd1e6
  usw_lr1e6_wd0
  asw_lr2e6_a0
  asw_lr1e6_a0
  asw_lr1e6_a025
)
methods=(usw usw usw asw asw asw)
lrs=(1e-6 5e-7 1e-6 2e-6 1e-6 1e-6)
weight_decays=(1e-6 1e-6 0 1e-6 1e-6 1e-6)
alphas=(1.0 1.0 1.0 0.0 0.0 0.25)

cd "$repo"

if [[ -e "$root" ]]; then
  echo "refusing_existing_root=$repo/$root" >&2
  exit 3
fi

for label in "${labels[@]}"; do
  if [[ -e "$root/$label" ]]; then
    echo "refusing_existing_output=$repo/$root/$label" >&2
    exit 3
  fi
done

mkdir -p "$root"
printf 'launcher_pid=%s\n' "$$" > "$root/launcher.pid"
printf 'tag=%s\ncuda_visible_devices=%s\nstarted=%s\n' \
  "$tag" "$CUDA_VISIBLE_DEVICES" "$(date -Is)" > "$root/launcher.meta"

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  output="$root/$label"
  printf 'variant_start=%s time=%s output=%s\n' \
    "$label" "$(date -Is)" "$repo/$output"
  "$python_bin" scripts/train_jssp_large_objectives.py \
    --method "${methods[$index]}" \
    --checkpoint "$checkpoint" \
    --steps 300 \
    --learning-rate "${lrs[$index]}" \
    --weight-decay "${weight_decays[$index]}" \
    --alpha "${alphas[$index]}" \
    --validation-count 32 \
    --validation-batch-size 8 \
    --validation-every 50 \
    --output-dir "$output" \
    --device cuda:0
  printf 'variant_complete=%s time=%s\n' "$label" "$(date -Is)"
done

printf 'completed=%s\n' "$(date -Is)" >> "$root/launcher.meta"
