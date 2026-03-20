#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cuda_ids...> [start|resume-latest|resume-dir] [resume_dir] [config_path]" >&2
  echo "Examples:" >&2
  echo "  $0 0 2 3 4" >&2
  echo "  $0 0,2,3,4" >&2
  echo "  $0 0 2 3 4 resume-latest" >&2
  exit 2
fi

GPU_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|resume-latest|resume-dir)
      break
      ;;
    *)
      GPU_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#GPU_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: missing CUDA ids" >&2
  exit 2
fi

MODE="${1:-start}" # start | resume-latest | resume-dir
if [[ $# -gt 0 ]]; then
  shift
fi
RESUME_DIR="${1:-}"
if [[ "$MODE" == "resume-dir" && $# -gt 0 ]]; then
  shift
fi
BASE_CONFIG="${1:-PTP/configs/experiment/pref_loss_coevo/loss_transfer_cvrp100_from_tsp100_elite.yaml}"

GPU_JOINED="${GPU_ARGS[*]}"
GPU_CSV="${GPU_JOINED// /,}"
IFS=',' read -r -a PHYSICAL_GPUS <<< "$GPU_CSV"
GPU_COUNT="${#PHYSICAL_GPUS[@]}"

if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "ERROR: failed to parse CUDA ids from: $GPU_CSV" >&2
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
GPU_TAG="${GPU_CSV//,/}"
LOG_PATH="${LOG_DIR}/pref_loss_cvrp100_from_tsp100_cuda${GPU_TAG}_${TS}.out"
TMP_CONFIG="${LOG_DIR}/pref_loss_cvrp100_from_tsp100_cuda${GPU_TAG}_${TS}.yaml"

python - "$BASE_CONFIG" "$TMP_CONFIG" "$GPU_COUNT" <<'PY'
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
gpu_count = int(sys.argv[3])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["devices"] = [f"cuda:{idx}" for idx in range(gpu_count)]
cfg["device"] = "cuda:0"
cfg["mp"] = {
    "enabled": bool(gpu_count > 1),
    "processes": gpu_count,
    "start_method": "spawn",
}
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst.as_posix())
PY

export CUDA_VISIBLE_DEVICES="${GPU_CSV}"

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
    echo "Usage: $0 <cuda_ids...> [start|resume-latest|resume-dir] [resume_dir] [config_path]" >&2
    exit 2
    ;;
esac

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Config: $TMP_CONFIG"
echo "Running: ${CMD[*]}"
echo "Log: $LOG_PATH"
nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"
tail -f "$LOG_PATH"
