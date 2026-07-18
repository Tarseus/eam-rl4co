#!/usr/bin/env bash
set -euo pipefail

cd /data1/gushengda/eam-rl4co

PY=/data1/gushengda/anaconda3/envs/rlco1/bin/python
TAG=20260718_1155
ROOT=logs/scheduling_large_scale/h11_concave_gap_temperature/jssp50x20/screen300_${TAG}
OUT=$ROOT/usw_concave_gap_temperature
LOG=logs/codex_remote/h11_screen300_usw_concave_gap_temperature_${TAG}.log
PAIR=research/scheduling_large_scale/experiments/h11_jssp_concave_gap_temperature/artifacts/usw_concave_gap_temperature/best_pair.json

if [[ -e "$ROOT" || -e "$LOG" ]]; then
  echo "collision: H11 screen target exists: $ROOT or $LOG" >&2
  exit 113
fi

if pgrep -af 'train_jssp_large_objectives.py' | grep -F 'h11_concave_gap_temperature' >/dev/null; then
  echo "collision: an H11 trainer already exists" >&2
  exit 114
fi

mkdir -p "$ROOT" logs/codex_remote

CUDA_VISIBLE_DEVICES=7 "$PY" scripts/train_jssp_large_objectives.py \
  --method usw \
  --checkpoint downloads/jssp15x15/weighting/checkpoint.ckpt \
  --steps 300 \
  --learning-rate 1e-5 \
  --weight-decay 1e-6 \
  --validation-count 32 \
  --validation-batch-size 8 \
  --validation-every 50 \
  --output-dir "$OUT" \
  --usw-pair "$PAIR" \
  --device cuda:0 2>&1 | tee "$LOG"

echo "h11_screen300_complete time=$(date -Is) root=$ROOT"
