#!/usr/bin/env bash
set -euo pipefail

cd /data1/gushengda/eam-rl4co

PY=/data1/gushengda/anaconda3/envs/rlco1/bin/python
TAG=20260718_0720
ROOT=logs/scheduling_large_scale/h9_source_normalized_asw/jssp50x20/screen300_${TAG}
OUT=$ROOT/asw_source_normalized
LOG=logs/codex_remote/h9_screen300_asw_source_normalized_${TAG}.log
PAIR=research/scheduling_large_scale/experiments/h9_jssp_source_normalized_asw/artifacts/asw_source_normalized/best_pair.json

if [[ -e "$ROOT" || -e "$LOG" ]]; then
  echo "collision: H9 screen target exists: $ROOT or $LOG" >&2
  exit 93
fi

if pgrep -af 'train_jssp_large_objectives.py' | grep -F 'h9_source_normalized_asw' >/dev/null; then
  echo "collision: an H9 trainer already exists" >&2
  exit 94
fi

mkdir -p "$ROOT" logs/codex_remote

CUDA_VISIBLE_DEVICES=7 "$PY" scripts/train_jssp_large_objectives.py \
  --method asw \
  --checkpoint downloads/jssp15x15/weighting/checkpoint.ckpt \
  --steps 300 \
  --learning-rate 1e-5 \
  --weight-decay 1e-6 \
  --alpha 0 \
  --validation-count 32 \
  --validation-batch-size 8 \
  --validation-every 50 \
  --output-dir "$OUT" \
  --asw-pair "$PAIR" \
  --device cuda:0 2>&1 | tee "$LOG"

echo "h9_screen300_complete time=$(date -Is) root=$ROOT"
