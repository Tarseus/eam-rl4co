#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Usage:
#   ./run_full_train_latest_pair.sh [RUNS_ROOT] [EXPERIMENT] [hydra overrides...]
#
# Examples:
#   ./run_full_train_latest_pair.sh
#   ./run_full_train_latest_pair.sh runs/pref_loss_alternating_simple routing/pomo-po4cops-tsp100-po trainer.devices=[0]
#   ./run_full_train_latest_pair.sh ckpt_path=baseline/epoch_409.ckpt trainer.max_epochs=300
#
# Notes:
# - If the first arg contains '=' (Hydra override), RUNS_ROOT/EXPERIMENT fall back to defaults.
# - Uses `best_pair.json` from the latest run directory under RUNS_ROOT (lexicographic sort).
# - By default the script adds low-variance training overrides unless you explicitly pass your own:
#   `seed=1234`, `trainer.deterministic=true`, `trainer.devices=[0]`, `matmul_precision=highest`.

RUNS_ROOT_DEFAULT="runs/pref_loss_alternating_simple"
EXPERIMENT_DEFAULT="routing/pomo-po4cops-tsp100-po"

runs_root="$RUNS_ROOT_DEFAULT"
experiment="$EXPERIMENT_DEFAULT"

if [[ $# -gt 0 && "${1:-}" != *=* && "${1:-}" != -* ]]; then
  runs_root="$1"
  shift
fi
if [[ $# -gt 0 && "${1:-}" != *=* && "${1:-}" != -* ]]; then
  experiment="$1"
  shift
fi

if [[ ! -d "$runs_root" ]]; then
  echo "ERROR: runs_root does not exist: $runs_root" >&2
  exit 1
fi

latest="$(
  python - "$runs_root" <<'PY'
import os
import sys

runs_root = sys.argv[1]
cands = []
for name in os.listdir(runs_root):
    path = os.path.join(runs_root, name)
    if not os.path.isdir(path):
        continue
    best_pair = os.path.join(path, "best_pair.json")
    if not os.path.isfile(best_pair):
        continue
    # Prefer most recently updated best_pair.json (covers resumed runs).
    try:
        mtime = os.path.getmtime(best_pair)
    except OSError:
        mtime = 0.0
    cands.append((mtime, name))

if not cands:
    sys.exit(2)

cands.sort()
print(cands[-1][1])
PY
)" || true

if [[ -z "${latest:-}" ]]; then
  # Fallback: lexicographic on directory name (expects YYYYMMDD-HHMMSS style).
  candidates=()
  for d in "$runs_root"/*; do
    [[ -d "$d" ]] || continue
    [[ -f "$d/best_pair.json" ]] || continue
    candidates+=("$(basename "$d")")
  done
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "ERROR: no run dirs with best_pair.json found under: $runs_root" >&2
    exit 1
  fi
  IFS=$'\n' sorted=($(printf "%s\n" "${candidates[@]}" | sort))
  unset IFS
  latest="${sorted[$((${#sorted[@]} - 1))]}"
fi

best_pair_path="${runs_root}/${latest}/best_pair.json"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="${LOG_TZ:-Asia/Shanghai}"
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL
: "${HYDRA_FULL_ERROR:=1}"
export HYDRA_FULL_ERROR

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_DIR}/full_train_best_pair_${TS}.out"

CMD=(
  python -u run.py
  "experiment=${experiment}"
  "model.loss_type=free_loss"
  "model.pref_pair_json_path=${best_pair_path}"
)

has_seed_override=false
has_deterministic_override=true
has_devices_override=false
has_matmul_precision_override=false

# Lightning raises if enable_progress_bar=false but RichProgressBar is still in callbacks.
# Some experiment configs disable the progress bar but keep the callback via callbacks/default.yaml.
# Default to disabling the RichProgressBar callback unless the user explicitly overrides it or
# explicitly enables the progress bar.
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
if [[ "$disable_rich_progress_bar" == "true" ]]; then
  CMD+=("~callbacks.rich_progress_bar")
fi

if [[ "$has_seed_override" == "false" ]]; then
  CMD+=("seed=1234")
fi
if [[ "$has_deterministic_override" == "false" ]]; then
  CMD+=("trainer.deterministic=true")
fi
if [[ "$has_devices_override" == "false" ]]; then
  CMD+=("trainer.devices=[0]")
fi
if [[ "$has_matmul_precision_override" == "false" ]]; then
  CMD+=("matmul_precision=highest")
fi

CMD+=("$@")

echo "Latest search run: ${runs_root}/${latest}"
echo "Using best_pair: ${best_pair_path}"
echo "Running: ${CMD[*]}"
echo "Log: $LOG_PATH"

nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"

tail -f "$LOG_PATH"
