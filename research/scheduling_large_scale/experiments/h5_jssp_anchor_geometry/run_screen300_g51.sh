#!/usr/bin/env bash
set -euo pipefail

cd /data1/gushengda/eam-rl4co

PY=/data1/gushengda/anaconda3/envs/rlco1/bin/python
TAG=20260718_0426
ROOT=logs/scheduling_large_scale/h5_anchor_geometry/jssp50x20/screen300_${TAG}
LOG_ROOT=logs/codex_remote
ART_ROOT=research/scheduling_large_scale/experiments/h5_jssp_anchor_geometry/artifacts

if [[ -e "$ROOT" ]]; then
  echo "collision: output root exists: $ROOT" >&2
  exit 51
fi

mkdir -p "$ROOT" "$LOG_ROOT"

run_variant() {
  local label="$1"
  local method="$2"
  local lr="$3"
  local alpha="$4"
  local pair_flag="$5"
  local pair_path="$6"
  local out="$ROOT/$label"
  local log="$LOG_ROOT/h5_screen300_${label}_${TAG}.log"

  if [[ -e "$out" || -e "$log" ]]; then
    echo "collision: variant target exists: $out or $log" >&2
    exit 52
  fi

  echo "variant_start label=$label time=$(date -Is)"
  CUDA_VISIBLE_DEVICES=7 "$PY" scripts/train_jssp_large_objectives.py \
    --method "$method" \
    --checkpoint downloads/jssp15x15/weighting/checkpoint.ckpt \
    --steps 300 \
    --learning-rate "$lr" \
    --weight-decay 1e-6 \
    --alpha "$alpha" \
    --validation-count 32 \
    --validation-batch-size 8 \
    --validation-every 50 \
    --output-dir "$out" \
    "$pair_flag" "$pair_path" \
    --device cuda:0 2>&1 | tee "$log"
  echo "variant_complete label=$label time=$(date -Is)"
}

run_variant \
  usw_anchor_lr5e7 \
  usw \
  5e-7 \
  1.0 \
  --usw-pair \
  "$ART_ROOT/usw_stratified_anchor/best_pair.json"

run_variant \
  asw_anchor_lr1e6_a0 \
  asw \
  1e-6 \
  0.0 \
  --asw-pair \
  "$ART_ROOT/asw_stratified_anchor/best_pair.json"

echo "h5_screen300_complete time=$(date -Is) root=$ROOT"
