#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_RAW="${GPU_IDS:-0 1 2 3 4 5}"
TASKS_RAW="${TASKS:-tsp50 cvrp50}"
METHODS_RAW="${METHODS:-bopo}"

IFS=' ' read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
IFS=' ' read -r -a REQUESTED_TASKS <<< "$TASKS_RAW"
IFS=' ' read -r -a REQUESTED_METHODS <<< "$METHODS_RAW"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_IDS must contain at least one GPU id." >&2
  echo "Example: GPU_IDS='0 1 2' bash $0" >&2
  exit 2
fi

if [[ "${#REQUESTED_TASKS[@]}" -lt 1 ]]; then
  echo "ERROR: TASKS must contain at least one task." >&2
  echo "Example: TASKS='tsp50 cvrp50' bash $0" >&2
  exit 2
fi

if [[ "${#REQUESTED_METHODS[@]}" -lt 1 ]]; then
  echo "ERROR: METHODS must contain at least one method." >&2
  echo "Example: METHODS='rl sll bopo' bash $0" >&2
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
RUN_ROOT="${ROOT_DIR}/logs/train/runs/tsp50_cvrp50_supplement_${TS}"
mkdir -p "$RUN_ROOT"

all_keys=()

declare -A EXPERIMENT_MAP=(
  ["tsp50_rl"]="routing/pomo-po4cops-tsp50-po"
  ["cvrp50_rl"]="routing/pomo-po4cops-cvrp50-po"
  ["tsp50_sll"]="routing/pomo-sll-tsp50"
  ["cvrp50_sll"]="routing/pomo-sll-cvrp50"
  ["tsp50_bopo"]="routing/pomo-bopo-tsp50"
  ["cvrp50_bopo"]="routing/pomo-bopo-cvrp50"
)

declare -A ENV_MAP=(
  ["tsp50_rl"]="tsp"
  ["cvrp50_rl"]="cvrp"
  ["tsp50_sll"]="tsp"
  ["cvrp50_sll"]="cvrp"
  ["tsp50_bopo"]="tsp"
  ["cvrp50_bopo"]="cvrp"
)

declare -A DATA_SUBDIR_MAP=(
  ["tsp50_rl"]="tsp"
  ["cvrp50_rl"]="vrp"
  ["tsp50_sll"]="tsp"
  ["cvrp50_sll"]="vrp"
  ["tsp50_bopo"]="tsp"
  ["cvrp50_bopo"]="vrp"
)

declare -A VAL_FILE_MAP=(
  ["tsp50_rl"]="tsp50_val_seed4321.npz"
  ["cvrp50_rl"]="vrp50_val_seed4321.npz"
  ["tsp50_sll"]="tsp50_val_seed4321.npz"
  ["cvrp50_sll"]="vrp50_val_seed4321.npz"
  ["tsp50_bopo"]="tsp50_val_seed4321.npz"
  ["cvrp50_bopo"]="vrp50_val_seed4321.npz"
)

declare -A TEST_FILE_MAP=(
  ["tsp50_rl"]="tsp50_test_seed1234.npz"
  ["cvrp50_rl"]="vrp50_test_seed1234.npz"
  ["tsp50_sll"]="tsp50_test_seed1234.npz"
  ["cvrp50_sll"]="vrp50_test_seed1234.npz"
  ["tsp50_bopo"]="tsp50_test_seed1234.npz"
  ["cvrp50_bopo"]="vrp50_test_seed1234.npz"
)

declare -A LOSS_OVERRIDE_MAP=(
  ["tsp50_rl"]="rl_loss"
  ["cvrp50_rl"]="rl_loss"
  ["tsp50_sll"]=""
  ["cvrp50_sll"]=""
  ["tsp50_bopo"]=""
  ["cvrp50_bopo"]=""
)

declare -A BOPO_SELECT_K_OVERRIDE_MAP=(
  ["tsp50_rl"]=""
  ["cvrp50_rl"]=""
  ["tsp50_sll"]=""
  ["cvrp50_sll"]=""
  ["tsp50_bopo"]="10"
  ["cvrp50_bopo"]="10"
)

PIDS=()
LOG_PATHS=()
RUN_DIRS=()
STARTED_KEYS=()

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

contains_value() {
  local needle="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ "$candidate" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

is_requested_key() {
  local key="$1"
  local task="${key%_*}"
  local method="${key##*_}"

  if ! contains_value "$task" "${REQUESTED_TASKS[@]}"; then
    return 1
  fi
  if ! contains_value "$method" "${REQUESTED_METHODS[@]}"; then
    return 1
  fi

  if [[ -z "${EXPERIMENT_MAP[$key]+x}" ]]; then
    echo "WARN: unsupported experiment key '${key}', skipping." >&2
    return 1
  fi

  return 0
}

build_requested_keys() {
  local task
  local method
  local key

  for task in "${REQUESTED_TASKS[@]}"; do
    case "$task" in
      tsp50|cvrp50)
        ;;
      *)
        echo "WARN: unsupported task '${task}', skipping." >&2
        continue
        ;;
    esac
    for method in "${REQUESTED_METHODS[@]}"; do
      case "$method" in
        rl|sll|bopo)
          ;;
        *)
          echo "WARN: unsupported method '${method}', skipping." >&2
          continue
          ;;
      esac
      key="${task}_${method}"
      if [[ -n "${EXPERIMENT_MAP[$key]+x}" ]]; then
        all_keys+=("$key")
      fi
    done
  done
}

build_requested_keys

if [[ "${#all_keys[@]}" -lt 1 ]]; then
  echo "No valid task/method combinations were requested." >&2
  exit 1
fi

launch_one() {
  local key="$1"
  local idx="$2"
  local experiment="${EXPERIMENT_MAP[$key]}"
  local env_name="${ENV_MAP[$key]}"
  local data_subdir="${DATA_SUBDIR_MAP[$key]}"
  local val_file="${VAL_FILE_MAP[$key]}"
  local test_file="${TEST_FILE_MAP[$key]}"
  local loss_override="${LOSS_OVERRIDE_MAP[$key]}"
  local bopo_select_k_override="${BOPO_SELECT_K_OVERRIDE_MAP[$key]}"
  local gpu_id="${GPU_IDS[$(( idx % ${#GPU_IDS[@]} ))]}"
  local run_dir="${RUN_ROOT}/${key}"
  local log_path="${LOG_DIR}/${key}_${TS}.out"

  mkdir -p "$run_dir"

  cmd=(
    "$PYTHON_BIN" -u run.py
    "experiment=${experiment}"
    "hydra.run.dir=${run_dir}"
    "env=${env_name}"
    "env.data_dir=\${paths.root_dir}/data/${data_subdir}"
    "env.val_file=${val_file}"
    "env.test_file=${test_file}"
    "env.generator_params.num_loc=50"
    "logger=csv"
    "logger.csv.name=${key}"
  )

  if [[ -n "$loss_override" ]]; then
    cmd+=("model.loss_type=${loss_override}")
  fi
  if [[ -n "$bopo_select_k_override" ]]; then
    cmd+=("model.bopo_select_k=${bopo_select_k_override}")
  fi

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
    cmd+=("trainer.devices=[0]")
  fi
  if [[ "$has_matmul_precision_override" == "false" ]]; then
    cmd+=("matmul_precision=highest")
  fi
  if [[ "$#" -gt 2 ]]; then
    cmd+=("${@:3}")
  fi

  echo "[launch] ${key}"
  echo "  GPU: ${gpu_id}"
  echo "  Run dir: ${run_dir}"
  echo "  Log: ${log_path}"
  echo "  Command: ${cmd[*]}"

  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >"${log_path}" 2>&1 &
  local pid=$!

  PIDS+=("$pid")
  LOG_PATHS+=("$log_path")
  RUN_DIRS+=("$run_dir")
  STARTED_KEYS+=("$key")

  echo "  PID: ${pid}"
}

idx=0
for key in "${all_keys[@]}"; do
  if ! is_requested_key "$key"; then
    continue
  fi
  launch_one "$key" "$idx" "$@"
  idx=$((idx + 1))
done

if [[ "${#STARTED_KEYS[@]}" -eq 0 ]]; then
  echo "No experiments launched. Check TASKS and METHODS." >&2
  exit 1
fi

echo
echo "Launched supplement baseline experiments:"
for i in "${!STARTED_KEYS[@]}"; do
  echo "  ${STARTED_KEYS[$i]} | GPU ${GPU_IDS[$(( i % ${#GPU_IDS[@]} ))]} | PID ${PIDS[$i]}"
  echo "    run: ${RUN_DIRS[$i]}"
  echo "    log: ${LOG_PATHS[$i]}"
done

echo
echo "Examples:"
echo "  bash $0"
echo "  GPU_IDS='0 1 2' bash $0"
echo "  TASKS='tsp50' METHODS='bopo' bash $0"
echo "  TASKS='cvrp50' METHODS='bopo' bash $0 trainer.max_epochs=500"
echo "  METHODS='rl sll bopo' bash $0"
