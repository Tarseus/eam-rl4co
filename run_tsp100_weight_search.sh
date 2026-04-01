#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-PTP/configs/experiment/pref_loss_coevo/tsp100_builder_weight_search_from_archive.yaml}"
MODE="${2:-start}" # start | resume-latest | resume-dir
RESUME_DIR="${3:-}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="Asia/Shanghai"
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_DIR}/tsp100_weight_search_${TS}.out"

CMD=(python -u PTP/ptp_discovery/run_pref_loss_coevo.py --config "$CONFIG_PATH")
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
    echo "  $0 [config_path] start" >&2
    echo "  $0 [config_path] resume-latest" >&2
    echo "  $0 [config_path] resume-dir <run_dir>" >&2
    exit 2
    ;;
esac

echo "Running: ${CMD[*]}"
echo "Log: $LOG_PATH"
nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"

tail -f "$LOG_PATH"
