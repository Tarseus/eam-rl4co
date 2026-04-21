#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-PTP/configs/experiment/pref_loss_coevo/loss_only_tsp100_discovery_family_off.yaml}"
MODE="${2:-start}" # start | resume-latest | resume-dir
RESUME_DIR="${3:-}"
VISIBLE_GPUS="${4:-${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-}}}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL
if [[ -n "$VISIBLE_GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
fi

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_DIR}/pref_loss_tsp100_family_ablation_${TS}.out"

CMD=("$PYTHON_BIN" -u PTP/ptp_discovery/run_pref_loss_coevo.py --config "$CONFIG_PATH")
case "$MODE" in
  start)
    ;;
  resume-latest)
    CMD+=(--resume-latest)
    ;;
  resume-dir)
    if [[ -z "$RESUME_DIR" ]]; then
      echo "ERROR: MODE=resume-dir requires a 3rd arg RESUME_DIR" >&2
      exit 2
    fi
    CMD+=(--resume-dir "$RESUME_DIR")
    ;;
  *)
    echo "Usage:" >&2
    echo "  $0 [config_path] start [resume_dir_unused] [visible_gpus]" >&2
    echo "  $0 [config_path] resume-latest [resume_dir_unused] [visible_gpus]" >&2
    echo "  $0 [config_path] resume-dir <run_dir> [visible_gpus]" >&2
    exit 2
    ;;
esac

echo "Running: ${CMD[*]}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<inherited>}"
echo "Log: $LOG_PATH"
nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"

tail -f "$LOG_PATH"
