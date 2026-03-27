#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINES="bopo,rl,po"
GPU_MAP=""
WAIT_MODE=1

print_usage() {
  cat <<'EOF'
Usage:
  ./run_jssp_baselines.sh [options] [-- extra args passed to every baseline]

Options:
  --baselines LIST    Comma-separated subset of: bopo,rl,po. Default: bopo,rl,po
  --gpus MAP          Comma-separated gpu map, e.g. bopo=0,rl=1,po=2
  --no-wait           Start processes and exit without waiting
  -h, --help          Show this help

Examples:
  ./run_jssp_baselines.sh --baselines bopo,po --gpus bopo=0,po=1
  ./run_jssp_baselines.sh --baselines rl --gpus rl=3 -- trainer.max_epochs=30
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baselines)
      BASELINES="${2:-}"
      shift 2
      ;;
    --gpus)
      GPU_MAP="${2:-}"
      shift 2
      ;;
    --no-wait)
      WAIT_MODE=0
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

EXTRA_ARGS=("$@")

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/logs/train/runs"

IFS=',' read -r -a REQUESTED_BASELINES <<< "${BASELINES}"

declare -A VALID_BASELINES=(
  [bopo]=1
  [rl]=1
  [po]=1
)

declare -A GPU_BY_BASELINE=(
  [bopo]=0
  [rl]=0
  [po]=0
)

if [[ -n "${GPU_MAP}" ]]; then
  IFS=',' read -r -a GPU_ENTRIES <<< "${GPU_MAP}"
  for entry in "${GPU_ENTRIES[@]}"; do
    [[ -z "${entry}" ]] && continue
    key="${entry%%=*}"
    value="${entry#*=}"
    if [[ -z "${VALID_BASELINES[$key]:-}" ]]; then
      echo "Unsupported baseline in --gpus: ${key}" >&2
      exit 1
    fi
    GPU_BY_BASELINE["$key"]="$value"
  done
fi

TS="$(date +%Y%m%d-%H%M%S)"

declare -a PIDS=()
declare -a NAMES=()
declare -a LOGS=()

start_baseline() {
  local baseline="$1"
  local gpu_id="$2"
  local log_path="${ROOT_DIR}/logs/jssp_${baseline}_${TS}.out"
  local experiment="scheduling/mgl-jssp-${baseline}-bucketed-multishape"
  local run_dir="${ROOT_DIR}/logs/train/runs/mgl-jssp-${baseline}_${TS}"
  local ckpt_dir="${run_dir}/checkpoints"
  local cmd=(
    "${PYTHON_BIN}" -u run.py
    "experiment=${experiment}"
    "hydra.run.dir=${run_dir}"
    "~callbacks.learning_rate_monitor"
    "~callbacks.rich_progress_bar"
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}"
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
    "callbacks.model_checkpoint.auto_insert_metric_name=False"
    "callbacks.model_checkpoint.save_top_k=1"
    "callbacks.model_checkpoint.save_last=True"
    "callbacks.model_checkpoint.every_n_epochs=1"
    "trainer.accelerator=gpu"
    "+trainer.devices=[0]"
    "logger=csv"
    "logger.csv.name=${baseline}"
  )
  cmd+=("${EXTRA_ARGS[@]}")

  echo "[${baseline}] CUDA_VISIBLE_DEVICES=${gpu_id}"
  echo "[${baseline}] log=${log_path}"
  echo "[${baseline}] experiment=${experiment}"
  echo "[${baseline}] cmd=${cmd[*]}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >"${log_path}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${baseline}")
  LOGS+=("${log_path}")
}

"${PYTHON_BIN}" scripts/prepare_bopo_jsp_data.py > /dev/null

for baseline in "${REQUESTED_BASELINES[@]}"; do
  [[ -z "${baseline}" ]] && continue
  if [[ -z "${VALID_BASELINES[$baseline]:-}" ]]; then
    echo "Unsupported baseline: ${baseline}" >&2
    exit 1
  fi
  start_baseline "${baseline}" "${GPU_BY_BASELINE[$baseline]}"
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "No baselines selected." >&2
  exit 1
fi

echo
echo "Started baselines:"
for i in "${!PIDS[@]}"; do
  echo "  ${NAMES[$i]}: pid=${PIDS[$i]} log=${LOGS[$i]}"
done

if [[ "${WAIT_MODE}" -eq 0 ]]; then
  exit 0
fi

exit_code=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if wait "${pid}"; then
    echo "[${name}] finished successfully"
  else
    echo "[${name}] failed; see ${LOGS[$i]}" >&2
    exit_code=1
  fi
done

exit "${exit_code}"
