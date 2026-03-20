#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_ID="${1:-}"
MODE="${2:-start}" # start | resume-latest | resume-dir
RESUME_DIR="${3:-}"
BASE_CONFIG="${4:-PTP/configs/experiment/pref_loss_coevo/loss_transfer_cvrp100_from_tsp100_elite.yaml}"

if [[ -z "$CUDA_ID" ]]; then
  echo "Usage: $0 <cuda_id> [start|resume-latest|resume-dir] [resume_dir] [config_path]" >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
: "${LOG_TZ:=Asia/Shanghai}"
export LOG_TZ
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_DIR}/pref_loss_cvrp100_from_tsp100_cuda${CUDA_ID}_${TS}.out"
TMP_CONFIG="${LOG_DIR}/pref_loss_cvrp100_from_tsp100_cuda${CUDA_ID}_${TS}.yaml"

python - "$BASE_CONFIG" "$TMP_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["devices"] = ["cuda:0"]
cfg["device"] = "cuda:0"
cfg["mp"] = {"enabled": False, "processes": 1, "start_method": "spawn"}
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst.as_posix())
PY

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"

CMD=(python -u PTP/ptp_discovery/run_pref_loss_coevo.py --config "$TMP_CONFIG")
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
    echo "Usage: $0 <cuda_id> [start|resume-latest|resume-dir] [resume_dir] [config_path]" >&2
    exit 2
    ;;
esac

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Config: $TMP_CONFIG"
echo "Running: ${CMD[*]}"
echo "Log: $LOG_PATH"
nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"
