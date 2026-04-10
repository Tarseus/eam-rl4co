#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_RAW="${GPU_IDS:-3 5}"

IFS=' ' read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
if [[ "${#GPU_IDS[@]}" -ne 2 ]]; then
  echo "ERROR: GPU_IDS must contain exactly 2 GPU ids, got: ${GPU_IDS_RAW}" >&2
  echo "Example: GPU_IDS='4 6' bash $0 trainer.max_epochs=200" >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL
: "${HYDRA_FULL_ERROR:=1}"
export HYDRA_FULL_ERROR

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${ROOT_DIR}/logs/train/runs/tsp100_cvrp100_bopo_${TS}"
mkdir -p "$RUN_ROOT"

EXPERIMENT_KEYS=(
  "tsp100_bopo"
  "cvrp100_bopo"
)

EXPERIMENT_CONFIGS=(
  "routing/pomo-bopo-tsp100"
  "routing/pomo-bopo-cvrp100"
)

PIDS=()
LOG_PATHS=()
RUN_DIRS=()

has_seed_override=false
has_deterministic_override=false
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

for idx in "${!EXPERIMENT_KEYS[@]}"; do
  key="${EXPERIMENT_KEYS[$idx]}"
  experiment="${EXPERIMENT_CONFIGS[$idx]}"
  gpu_id="${GPU_IDS[$idx]}"
  run_dir="${RUN_ROOT}/${key}"
  log_path="${LOG_DIR}/${key}_${TS}.out"

  mkdir -p "$run_dir"

  cmd=(
    "$PYTHON_BIN" -u run.py
    "experiment=${experiment}"
    "hydra.run.dir=${run_dir}"
  )

  if [[ "$disable_rich_progress_bar" == "true" ]]; then
    cmd+=("~callbacks.rich_progress_bar")
  fi
  if [[ "$has_seed_override" == "false" ]]; then
    cmd+=("seed=1234")
  fi
  if [[ "$has_deterministic_override" == "false" ]]; then
    cmd+=("trainer.deterministic=false")
  fi
  if [[ "$has_devices_override" == "false" ]]; then
    cmd+=("trainer.devices=[${gpu_id}]")
  fi
  if [[ "$has_matmul_precision_override" == "false" ]]; then
    cmd+=("matmul_precision=highest")
  fi

  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi

  echo "[launch] ${key}"
  echo "  GPU: ${gpu_id}"
  echo "  Run dir: ${run_dir}"
  if [[ "${key}" == "cvrp100_bopo" ]]; then
    echo "  Note: CVRP100 BOPO is an RL4CO adaptation, not a paper-matched official BOPO baseline."
  fi
  echo "  Log: ${log_path}"
  echo "  Command: ${cmd[*]}"

  nohup "${cmd[@]}" >"${log_path}" 2>&1 &
  pid=$!

  PIDS+=("$pid")
  LOG_PATHS+=("$log_path")
  RUN_DIRS+=("$run_dir")

  echo "  PID: ${pid}"
done

echo
echo "Launched 2 BOPO experiments:"
for idx in "${!EXPERIMENT_KEYS[@]}"; do
  echo "  ${EXPERIMENT_KEYS[$idx]} | GPU ${GPU_IDS[$idx]} | PID ${PIDS[$idx]}"
  echo "    run: ${RUN_DIRS[$idx]}"
  echo "    log: ${LOG_PATHS[$idx]}"
done

echo
echo "Example monitoring commands:"
echo "  tail -f ${LOG_PATHS[0]}"
echo "  tail -f ${LOG_PATHS[1]}"
