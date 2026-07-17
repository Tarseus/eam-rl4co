#!/usr/bin/env bash
set -euo pipefail

cd /data1/gushengda/eam-rl4co

PY=/data1/gushengda/anaconda3/envs/rlco1/bin/python
TAG=20260718_0550
ROOT=logs/scheduling_large_scale/h7_trajectory_scale_norm/jssp50x20/smoke_${TAG}
OUT=$ROOT/usw_source_length_norm
LOG=logs/codex_remote/h7_smoke_usw_source_length_norm_${TAG}.log
PAIR=research/scheduling_large_scale/experiments/h7_jssp_trajectory_scale_norm/artifacts/usw_source_length_norm/best_pair.json

if [[ -e "$ROOT" || -e "$LOG" ]]; then
  echo "collision: H7 smoke target exists: $ROOT or $LOG" >&2
  exit 71
fi

if pgrep -af 'train_jssp_large_objectives.py' | grep -F 'h7_trajectory_scale_norm' >/dev/null; then
  echo "collision: an H7 trainer already exists" >&2
  exit 72
fi

mkdir -p "$ROOT" logs/codex_remote

CUDA_VISIBLE_DEVICES=7 "$PY" scripts/train_jssp_large_objectives.py \
  --method usw \
  --checkpoint downloads/jssp15x15/weighting/checkpoint.ckpt \
  --steps 1 \
  --learning-rate 1e-5 \
  --weight-decay 1e-6 \
  --validation-count 32 \
  --validation-batch-size 8 \
  --validation-every 1 \
  --output-dir "$OUT" \
  --usw-pair "$PAIR" \
  --device cuda:0 2>&1 | tee "$LOG"

echo "h7_smoke_complete time=$(date -Is) root=$ROOT"
