#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-PTP/configs/experiment/pref_loss_coevo/loss_only_simple.yaml}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="Asia/Shanghai"
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_DIR}/pref_loss_loss_only_simple_${TS}.out"

CMD=(python -u PTP/ptp_discovery/run_pref_loss_coevo.py --config "$CONFIG_PATH")

echo "Running: ${CMD[*]}"
echo "Log: $LOG_PATH"
nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"

tail -f "$LOG_PATH"
