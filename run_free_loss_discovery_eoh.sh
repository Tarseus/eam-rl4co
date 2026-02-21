#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ

CONFIG_PATH="${1:-configs/experiment/free_loss_discovery/rl4co.yaml}"
DEVICE="${2:-cuda}"                  # cpu | cuda | cuda:0 ...
MODE="${3:-start}"                   # start | resume-latest | resume-dir
RESUME_DIR="${4:-}"

# Optional: OpenAI settings (you can also put OPENAI_API_KEY=... in a .env at repo root).
: "${OPENAI_MODEL:=gpt-4.1}"
: "${OPENAI_BASE_URL:=https://api.openai.com/v1}"
: "${OPENAI_TIMEOUT_S:=60}"
: "${OPENAI_MAX_RETRIES:=2}"

export OPENAI_MODEL OPENAI_BASE_URL OPENAI_TIMEOUT_S OPENAI_MAX_RETRIES

# Ensure local packages are importable when running from anywhere.
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"

# Quick dependency check (skip hard fail; user may run in offline_mode=true).
python -c "import yaml, dotenv; import openai" >/dev/null 2>&1 || {
  echo "WARNING: EoH extras not installed. Run: pip install -e \".[eoh]\"" >&2
}

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_DIR}/free_loss_eoh_${TS}.out"

CMD=(python -u scripts/run_free_loss_discovery_rl4co.py --config "$CONFIG_PATH" --device "$DEVICE")
case "$MODE" in
  start)
    ;;
  resume-latest)
    CMD+=(--resume-latest)
    ;;
  resume-dir)
    if [[ -z "$RESUME_DIR" ]]; then
      echo "ERROR: MODE=resume-dir requires a 4th arg RESUME_DIR" >&2
      exit 2
    fi
    CMD+=(--resume-dir "$RESUME_DIR")
    ;;
  *)
    echo "Usage:" >&2
    echo "  $0 [config_path] [device] start" >&2
    echo "  $0 [config_path] [device] resume-latest" >&2
    echo "  $0 [config_path] [device] resume-dir <run_dir>" >&2
    exit 2
    ;;
esac

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "NOTE: OPENAI_API_KEY is not set in env. If your config does NOT set offline_mode=true," >&2
  echo "      EoH will fail. Set it in env, or create a .env at repo root with OPENAI_API_KEY=..." >&2
fi

echo "Running: ${CMD[*]}"
echo "Log: $LOG_PATH"
nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"
echo "LLM cache will be written under the run dir as: llm_cache.jsonl"

