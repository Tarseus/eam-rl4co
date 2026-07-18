#!/usr/bin/env bash
set -euo pipefail

cd /data1/gushengda/eam-rl4co

PY=/data1/gushengda/anaconda3/envs/rlco1/bin/python
TAG=20260718_1254
ROOT=logs/scheduling_large_scale/h13_h8_to_asw_micro/jssp50x20/screen50_${TAG}
OUT=$ROOT/asw_micro_lr1e6
LOG=logs/codex_remote/h13_screen50_h8_to_asw_micro_${TAG}.log
CKPT=logs/scheduling_large_scale/h8_relative_gap_temperature/jssp50x20/continue500_from_step250_20260718_0655/usw_relative_gap_temperature/best.ckpt
PAIR=research/scheduling_large_scale/experiments/h9_jssp_source_normalized_asw/artifacts/asw_source_normalized/best_pair.json

if [[ -e "$ROOT" || -e "$LOG" ]]; then
  echo "collision: H13 screen target exists: $ROOT or $LOG" >&2
  exit 133
fi

if pgrep -af 'train_jssp_large_objectives.py' | grep -F 'h13_h8_to_asw_micro' >/dev/null; then
  echo "collision: an H13 trainer already exists" >&2
  exit 134
fi

mkdir -p "$ROOT" logs/codex_remote

CUDA_VISIBLE_DEVICES=5 "$PY" scripts/train_jssp_large_objectives.py \
  --method asw \
  --checkpoint "$CKPT" \
  --steps 50 \
  --learning-rate 1e-6 \
  --weight-decay 1e-6 \
  --alpha 0 \
  --data-start-index 500 \
  --validation-count 32 \
  --validation-batch-size 8 \
  --validation-every 10 \
  --output-dir "$OUT" \
  --asw-pair "$PAIR" \
  --device cuda:0 2>&1 | tee "$LOG"

echo "h13_screen50_complete time=$(date -Is) root=$ROOT"
