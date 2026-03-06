#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-routing/pomo-po4cops-tsp100-po}"
SEED="${SEED:-1234}"

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-100000}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-10000}"
TEST_DATA_SIZE="${TEST_DATA_SIZE:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
mkdir -p "$LOG_DIR"

start_job() {
  local env_name="$1"
  local num_loc="$2"
  local gpu_id="$3"
  local max_epochs="$4"
  local ckpt_every_n_epochs="$5"
  shift 5
  local alias_epochs=("$@")
  local data_subdir
  local data_prefix
  local ts
  local run_dir
  local ckpt_dir

  if [[ "$env_name" == "tsp" ]]; then
    data_subdir="tsp"
    data_prefix="tsp"
  else
    data_subdir="vrp"
    data_prefix="vrp"
  fi

  local run_name="po4cops_${env_name}${num_loc}_e${max_epochs}_seed${SEED}"
  ts="$(date +%Y%m%d-%H%M%S)"
  run_dir="${ROOT_DIR}/logs/train/runs/${run_name}_${ts}"
  ckpt_dir="${run_dir}/checkpoints"
  local log_path="${LOG_DIR}/${run_name}_${ts}.out"

  echo "[launch] ${run_name} gpu=${gpu_id} log=${log_path} run_dir=${run_dir}"

  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" -u run.py \
    "experiment=${BASE_EXPERIMENT}" \
    "hydra.run.dir=${run_dir}" \
    "~callbacks.learning_rate_monitor" \
    "~callbacks.rich_progress_bar" \
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}" \
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
    "callbacks.model_checkpoint.auto_insert_metric_name=False" \
    "callbacks.model_checkpoint.save_top_k=-1" \
    "callbacks.model_checkpoint.save_last=True" \
    "callbacks.model_checkpoint.every_n_epochs=${ckpt_every_n_epochs}" \
    "seed=${SEED}" \
    "env=${env_name}" \
    "env.data_dir=\${paths.root_dir}/data/${data_subdir}" \
    "env.val_file=${data_prefix}${num_loc}_val_seed4321.npz" \
    "env.test_file=${data_prefix}${num_loc}_test_seed1234.npz" \
    "env.generator_params.num_loc=${num_loc}" \
    "model.loss_type=po_loss" \
    "model.alpha=0.05" \
    "model.num_starts=${num_loc}" \
    "model.batch_size=${BATCH_SIZE}" \
    "model.train_data_size=${TRAIN_DATA_SIZE}" \
    "model.val_data_size=${VAL_DATA_SIZE}" \
    "model.test_data_size=${TEST_DATA_SIZE}" \
    "model.optimizer=Adam" \
    "model.optimizer_kwargs.lr=${LR}" \
    "model.optimizer_kwargs.weight_decay=${WEIGHT_DECAY}" \
    "model.lr_scheduler=MultiStepLR" \
    "model.lr_scheduler_kwargs.milestones=[3001]" \
    "model.lr_scheduler_kwargs.gamma=0.2" \
    "trainer.max_epochs=${max_epochs}" \
    "trainer.accelerator=gpu" \
    "trainer.devices=[0]" \
    "trainer.strategy=auto" \
    "trainer.precision=32-true" \
    "trainer.gradient_clip_val=null" \
    "trainer.accumulate_grad_batches=1" \
    "trainer.enable_progress_bar=false" \
    "trainer.log_every_n_steps=50" \
    "logger=csv" \
    "logger.csv.name=${run_name}" \
    >"${log_path}" 2>&1 &

  local train_pid=$!

  (
    while kill -0 "${train_pid}" 2>/dev/null; do
      for alias_epoch in "${alias_epochs[@]}"; do
        src_epoch="$(printf "%03d" $((alias_epoch - 1)))"
        dst_epoch="${alias_epoch}"
        if [[ -f "${ckpt_dir}/epoch_${src_epoch}.ckpt" && ! -f "${ckpt_dir}/epoch_${dst_epoch}.ckpt" ]]; then
          cp -f "${ckpt_dir}/epoch_${src_epoch}.ckpt" "${ckpt_dir}/epoch_${dst_epoch}.ckpt"
        fi
      done
      sleep 15
    done

    for alias_epoch in "${alias_epochs[@]}"; do
      src_epoch="$(printf "%03d" $((alias_epoch - 1)))"
      dst_epoch="${alias_epoch}"
      if [[ -f "${ckpt_dir}/epoch_${src_epoch}.ckpt" ]]; then
        cp -f "${ckpt_dir}/epoch_${src_epoch}.ckpt" "${ckpt_dir}/epoch_${dst_epoch}.ckpt"
      elif [[ "${alias_epoch}" == "${max_epochs}" && -f "${ckpt_dir}/last.ckpt" ]]; then
        cp -f "${ckpt_dir}/last.ckpt" "${ckpt_dir}/epoch_${dst_epoch}.ckpt"
      fi
    done
  ) >/dev/null 2>&1 &

  echo "[started] ${run_name} pid=${train_pid} ckpt_dir=${ckpt_dir}"
}

start_job "tsp" "50" "0" "100" "50" "50" "100"
start_job "cvrp" "50" "1" "100" "50" "50" "100"
start_job "cvrp" "100" "2" "200" "100" "100" "200"

echo "[done] launched 3 jobs"
