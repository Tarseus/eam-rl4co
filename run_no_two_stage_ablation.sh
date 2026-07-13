#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROBLEM="${1:-}"
MODE="${2:-start}" # start | resume-latest | resume-dir
RESUME_DIR="${3:-}"
VISIBLE_GPUS="${4:-${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run_no_two_stage_ablation.sh tsp100  [start|resume-latest|resume-dir] [resume_dir] [visible_gpus]
  ./run_no_two_stage_ablation.sh ffsp100 [start|resume-latest|resume-dir] [resume_dir] [visible_gpus]

Examples:
  OPENAI_API_KEY=... ./run_no_two_stage_ablation.sh tsp100 start "" 0,1,2,3
  OPENAI_API_KEY=... ./run_no_two_stage_ablation.sh ffsp100 resume-latest "" 0,1,2,3
EOF
}

config_for_problem() {
  case "$1" in
    tsp100)
      echo "PTP/configs/experiment/pref_loss_coevo/no_two_stage_tsp100_coevo.yaml"
      ;;
    ffsp100)
      echo "PTP/configs/experiment/pref_loss_coevo/no_two_stage_ffsp100_coevo.yaml"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

if [[ -z "$PROBLEM" ]]; then
  usage
  exit 2
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the environment; this script does not store API keys.}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

# OpenAI-compatible endpoint defaults for the new proxy/model. The Python
# client normalizes a full chat.completions endpoint to the SDK base URL.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api2.aigcbest.top/v1/chat/completions}"
export OPENAI_MODEL="${OPENAI_MODEL:-glm-z1-flash}"
export OPENAI_MODEL_MINI="${OPENAI_MODEL_MINI:-$OPENAI_MODEL}"
export OPENAI_MODEL_NANO="${OPENAI_MODEL_NANO:-$OPENAI_MODEL}"

# The project wraps SDK calls with its own retry loop. Keep SDK retries low so
# logging and backoff are controlled in one place.
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-0}"
export OPENAI_TIMEOUT_S="${OPENAI_TIMEOUT_S:-90}"
export OPENAI_CALL_MAX_ATTEMPTS="${OPENAI_CALL_MAX_ATTEMPTS:-10}"
export OPENAI_CALL_BACKOFF_S="${OPENAI_CALL_BACKOFF_S:-2}"
export OPENAI_CALL_BACKOFF_MAX_S="${OPENAI_CALL_BACKOFF_MAX_S:-90}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_DIR}/no_two_stage_${PROBLEM}_${TS}.out"

if [[ -n "$VISIBLE_GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
fi

CONFIG_PATH="$(config_for_problem "$PROBLEM")"
CONFIG_TO_RUN="$CONFIG_PATH"

if [[ -n "$VISIBLE_GPUS" || "$MODE" == "start" ]]; then
  TMP_CONFIG="${LOG_DIR}/no_two_stage_${PROBLEM}_${TS}.yaml"
  "$PYTHON_BIN" - "$CONFIG_PATH" "$TMP_CONFIG" "$VISIBLE_GPUS" "$MODE" <<'PY'
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
visible = [part.strip() for part in sys.argv[3].split(",") if part.strip()]
mode = sys.argv[4]
cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
logical = [f"cuda:{i}" for i in range(len(visible))]
if logical:
    cfg["devices"] = logical
    mp = cfg.get("mp", {}) or {}
    if isinstance(mp, dict):
        mp["processes"] = len(logical)
        cfg["mp"] = mp
if mode == "start":
    resume = cfg.get("resume", {}) or {}
    if not isinstance(resume, dict):
        resume = {}
    resume["enabled"] = False
    cfg["resume"] = resume
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(dst.as_posix())
PY
  CONFIG_TO_RUN="$TMP_CONFIG"
fi

CMD=("$PYTHON_BIN" -u PTP/ptp_discovery/run_pref_loss_coevo.py --config "$CONFIG_TO_RUN")
case "$MODE" in
  start)
    ;;
  resume-latest)
    CMD+=(--resume-latest)
    ;;
  resume-dir)
    if [[ -z "$RESUME_DIR" ]]; then
      echo "ERROR: MODE=resume-dir requires RESUME_DIR as the third argument." >&2
      exit 2
    fi
    CMD+=(--resume-dir "$RESUME_DIR")
    ;;
  *)
    usage
    exit 2
    ;;
esac

echo "Problem: $PROBLEM"
echo "Config: $CONFIG_TO_RUN"
echo "OPENAI_BASE_URL: $OPENAI_BASE_URL"
echo "OPENAI_MODEL: $OPENAI_MODEL"
echo "OPENAI_CALL_MAX_ATTEMPTS: $OPENAI_CALL_MAX_ATTEMPTS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<inherited>}"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "Log: $LOG_PATH"
echo "Running: ${CMD[*]}"

nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"
