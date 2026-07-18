#!/usr/bin/env bash
set -euo pipefail

cd /data1/gushengda/eam-rl4co

PY=/data1/gushengda/anaconda3/envs/rlco1/bin/python
TAG=20260718_1230
ROOT=logs/scheduling_large_scale/h12_instance_gradient_clipping/jssp50x20/screen300_${TAG}
OUT=$ROOT/usw_h8_gradclip1
LOG=logs/codex_remote/h12_screen300_usw_h8_gradclip1_${TAG}.log
PAIR=research/scheduling_large_scale/experiments/h8_jssp_relative_gap_temperature/artifacts/usw_relative_gap_temperature/best_pair.json

if [[ -e "$ROOT" || -e "$LOG" ]]; then
  echo "collision: H12 screen target exists: $ROOT or $LOG" >&2
  exit 123
fi

if pgrep -af 'train_jssp_large_objectives.py' | grep -F 'h12_instance_gradient_clipping' >/dev/null; then
  echo "collision: an H12 trainer already exists" >&2
  exit 124
fi

mkdir -p "$ROOT" logs/codex_remote

CUDA_VISIBLE_DEVICES=7 "$PY" scripts/train_jssp_large_objectives.py \
  --method usw \
  --checkpoint downloads/jssp15x15/weighting/checkpoint.ckpt \
  --steps 300 \
  --learning-rate 1e-5 \
  --weight-decay 1e-6 \
  --max-grad-norm 1.0 \
  --validation-count 32 \
  --validation-batch-size 8 \
  --validation-every 50 \
  --output-dir "$OUT" \
  --usw-pair "$PAIR" \
  --device cuda:0 2>&1 | tee "$LOG"

echo "h12_screen300_complete time=$(date -Is) root=$ROOT"
