#!/usr/bin/env bash
set -euo pipefail

cd /data1/gushengda/eam-rl4co
export PYTHONPATH=/data1/gushengda/eam-rl4co:${PYTHONPATH:-}
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one collision-free GPU}"

LABEL=${1:?usage: run_eval_g51.sh bopo|h8_usw|h13_asw}
TAG=generated256_b128g1_20260718_1540
ROOT=logs/scheduling_large_scale/h15_locked_generated_test/jssp50x20/${TAG}
USW_PAIR=research/scheduling_large_scale/experiments/h8_jssp_relative_gap_temperature/artifacts/usw_relative_gap_temperature/best_pair.json
ASW_PAIR=research/scheduling_large_scale/experiments/h9_jssp_source_normalized_asw/artifacts/asw_source_normalized/best_pair.json

case "$LABEL" in
  bopo)
    METHOD=bopo; STEP=500
    SHA=ca0a0e4d0b90a74c91f09daf087d230e14ac81274cfc9ceb6dc9e6244c8fb44d
    CKPT=logs/scheduling_large_scale/h4_matched_bopo/jssp50x20/bopo_lr1e5_screen500_20260718_0110/best.ckpt
    ;;
  h8_usw)
    METHOD=usw; STEP=500
    SHA=9b5054b5a9b6038840c389a9802dc8d2e15a059c11a417a443ec930058b58ddc
    CKPT=logs/scheduling_large_scale/h8_relative_gap_temperature/jssp50x20/continue500_from_step250_20260718_0655/usw_relative_gap_temperature/best.ckpt
    ;;
  h13_asw)
    METHOD=asw; STEP=30
    SHA=0b25c8b50fda168b01bf6c0820e910c0cc72bf1066d5cb0a1eb70ada1ee4011c
    CKPT=logs/scheduling_large_scale/h13_h8_to_asw_micro/jssp50x20/screen50_20260718_1254/asw_micro_lr1e6/best.ckpt
    ;;
  *) echo "unknown label: $LABEL" >&2; exit 2 ;;
esac

OUT=$ROOT/$LABEL
LOG=logs/codex_remote/h15_jssp50x20_${LABEL}_${TAG}.log
if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "collision: H15 target exists: $OUT or $LOG" >&2
  exit 3
fi

/data1/gushengda/anaconda3/envs/rlco1/bin/python scripts/eval_jssp50x20_locked_generated.py \
  --label "$LABEL" --method "$METHOD" --checkpoint "$CKPT" \
  --checkpoint-sha256 "$SHA" --expected-step "$STEP" \
  --output-dir "$OUT" --device cuda:0 --count 256 \
  --generator-seed 314159265 --sampling-seed 271828182 \
  --B 128 --greedy 1 --usw-pair "$USW_PAIR" --asw-pair "$ASW_PAIR" \
  2>&1 | tee "$LOG"

echo "h15_eval_complete label=$LABEL time=$(date -Is) output=$OUT"

