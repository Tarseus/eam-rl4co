#!/usr/bin/env bash
set -euo pipefail

method="$1"
train_pid="$2"
gpu="$3"
checkpoint="$4"
run_dir="$5"

tail --pid="${train_pid}" -f /dev/null
test -f "${run_dir}/summary.json"

CUDA_VISIBLE_DEVICES="${gpu}" /data1/gushengda/anaconda3/envs/rlco1/bin/python \
  scripts/train_cvrp1000_objectives.py \
  --method "${method}" \
  --checkpoint "${checkpoint}" \
  --resume "${run_dir}/last.ckpt" \
  --capacity 50 \
  --num-starts 50 \
  --evaluate-only \
  --evaluation-file data/vrp/agfn_vrp1000_capacity50_test128.npz \
  --evaluation-instances 128 \
  --evaluation-starts 100 \
  --evaluation-augment 8 \
  --evaluation-precision 32-true \
  --evaluation-output "${run_dir}/official128.json" \
  --device cuda:0
