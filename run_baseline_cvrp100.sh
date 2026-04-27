#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${1:-0}"
shift || true

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/train/runs/cvrp100_baseline_${TS}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_PATH="${LOG_DIR}/cvrp100_baseline_${TS}.out"

has_seed_override=false
has_deterministic_override=false
has_max_epochs_override=false
has_devices_override=false
has_matmul_precision_override=false
disable_rich_progress_bar=true

for arg in "$@"; do
  case "$arg" in
    seed=*)
      has_seed_override=true
      ;;
    trainer.deterministic=*)
      has_deterministic_override=true
      ;;
    trainer.max_epochs=*)
      has_max_epochs_override=true
      ;;
    trainer.devices=*)
      has_devices_override=true
      ;;
    matmul_precision=*)
      has_matmul_precision_override=true
      ;;
    trainer.enable_progress_bar=true|trainer.enable_progress_bar=True)
      disable_rich_progress_bar=false
      ;;
    callbacks.rich_progress_bar=*|~callbacks.rich_progress_bar)
      disable_rich_progress_bar=false
      ;;
  esac
done

CMD=(
  "$PYTHON_BIN" -u run.py
  "experiment=routing/pomo-po4cops-cvrp100-po"
  "hydra.run.dir=${RUN_DIR}"
  "~callbacks.learning_rate_monitor"
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"
  "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
  "callbacks.model_checkpoint.auto_insert_metric_name=False"
  "callbacks.model_checkpoint.save_top_k=-1"
  "callbacks.model_checkpoint.save_last=True"
  "callbacks.model_checkpoint.every_n_epochs=100"
  "env=cvrp"
  "env.data_dir=\${paths.root_dir}/data/vrp"
  "env.val_file=vrp100_val_seed4321.npz"
  "env.test_file=vrp100_test_seed1234.npz"
  "env.generator_params.num_loc=100"
  "model.loss_type=po_loss"
  "model.alpha=0.05"
  "model.num_starts=100"
  "model.batch_size=64"
  "model.train_data_size=100000"
  "model.val_data_size=10000"
  "model.test_data_size=10000"
  "model.optimizer=Adam"
  "model.optimizer_kwargs.lr=3e-4"
  "model.optimizer_kwargs.weight_decay=1e-6"
  "model.lr_scheduler=MultiStepLR"
  "model.lr_scheduler_kwargs.milestones=[3001]"
  "model.lr_scheduler_kwargs.gamma=0.2"
  "trainer.accelerator=gpu"
  "trainer.strategy=auto"
  "trainer.precision=32-true"
  "trainer.gradient_clip_val=null"
  "trainer.accumulate_grad_batches=1"
  "trainer.enable_progress_bar=false"
  "trainer.log_every_n_steps=50"
  "logger=csv"
  "logger.csv.name=cvrp100_baseline"
)

if [[ "$disable_rich_progress_bar" == "true" ]]; then
  CMD+=("~callbacks.rich_progress_bar")
fi
if [[ "$has_seed_override" == "false" ]]; then
  CMD+=("seed=1234")
fi
if [[ "$has_deterministic_override" == "false" ]]; then
  CMD+=("trainer.deterministic=true")
fi
if [[ "$has_max_epochs_override" == "false" ]]; then
  CMD+=("trainer.max_epochs=200")
fi
if [[ "$has_devices_override" == "false" ]]; then
  CMD+=("trainer.devices=[0]")
fi
if [[ "$has_matmul_precision_override" == "false" ]]; then
  CMD+=("matmul_precision=highest")
fi

CMD+=("$@")

echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Running: ${CMD[*]}"
echo "Log: ${LOG_PATH}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
TRAIN_PID=$!

(
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    if [[ -f "${CKPT_DIR}/epoch_099.ckpt" && ! -f "${CKPT_DIR}/epoch_100.ckpt" ]]; then
      cp -f "${CKPT_DIR}/epoch_099.ckpt" "${CKPT_DIR}/epoch_100.ckpt"
    fi
    if [[ -f "${CKPT_DIR}/epoch_199.ckpt" && ! -f "${CKPT_DIR}/epoch_200.ckpt" ]]; then
      cp -f "${CKPT_DIR}/epoch_199.ckpt" "${CKPT_DIR}/epoch_200.ckpt"
    fi
    if [[ -f "${CKPT_DIR}/epoch_100.ckpt" ]]; then
      cp -f "${CKPT_DIR}/epoch_100.ckpt" "${ROOT_DIR}/baseline/cvrp100_epoch_100.ckpt"
    fi
    if [[ -f "${CKPT_DIR}/epoch_200.ckpt" ]]; then
      cp -f "${CKPT_DIR}/epoch_200.ckpt" "${ROOT_DIR}/baseline/cvrp100_epoch_200.ckpt"
    fi
    sleep 15
  done

  if [[ -f "${CKPT_DIR}/epoch_099.ckpt" ]]; then
    cp -f "${CKPT_DIR}/epoch_099.ckpt" "${CKPT_DIR}/epoch_100.ckpt"
  fi
  if [[ -f "${CKPT_DIR}/epoch_199.ckpt" ]]; then
    cp -f "${CKPT_DIR}/epoch_199.ckpt" "${CKPT_DIR}/epoch_200.ckpt"
  elif [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
    cp -f "${CKPT_DIR}/last.ckpt" "${CKPT_DIR}/epoch_200.ckpt"
  fi
  if [[ -f "${CKPT_DIR}/epoch_100.ckpt" ]]; then
    cp -f "${CKPT_DIR}/epoch_100.ckpt" "${ROOT_DIR}/baseline/cvrp100_epoch_100.ckpt"
  fi
  if [[ -f "${CKPT_DIR}/epoch_200.ckpt" ]]; then
    cp -f "${CKPT_DIR}/epoch_200.ckpt" "${ROOT_DIR}/baseline/cvrp100_epoch_200.ckpt"
  elif [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
    cp -f "${CKPT_DIR}/last.ckpt" "${ROOT_DIR}/baseline/cvrp100_epoch_200.ckpt"
  fi
) >/dev/null 2>&1 &

echo "Started PID: ${TRAIN_PID}"
tail -f "${LOG_PATH}"
