#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${ROOT_DIR}/logs/jssp15x15_sll_resume_${TIMESTAMP}.out"

mkdir -p "${ROOT_DIR}/logs"

nohup python -u run.py \
  experiment=scheduling/mgl-jssp-sll-paper \
  hydra.run.dir=/data1/gushengda/eam-rl4co/logs/train/runs/mgl-jssp-sll_15x15_20260422-153318 \
  ckpt_path=/data1/gushengda/eam-rl4co/logs/train/runs/mgl-jssp-sll_15x15_20260422-153318/checkpoints/last.ckpt \
  ~callbacks.learning_rate_monitor \
  ~callbacks.rich_progress_bar \
  callbacks.model_checkpoint.dirpath=/data1/gushengda/eam-rl4co/logs/train/runs/mgl-jssp-sll_15x15_20260422-153318/checkpoints \
  "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
  callbacks.model_checkpoint.auto_insert_metric_name=False \
  callbacks.model_checkpoint.save_top_k=1 \
  callbacks.model_checkpoint.save_last=True \
  callbacks.model_checkpoint.every_n_epochs=1 \
  trainer.accelerator=gpu \
  +trainer.devices=[5] \
  trainer.max_epochs=20 \
  +trainer.enable_progress_bar=false \
  logger=csv \
  logger.csv.name=mgl-jssp-sll_15x15 \
  +model.allowed_shapes=[[15,15]] \
  +model.required_allowed_shapes=[[15,15]] \
  >"${LOG_PATH}" 2>&1 < /dev/null &

echo "started jssp15x15 sll resume on gpu 5"
echo "log: ${LOG_PATH}"
