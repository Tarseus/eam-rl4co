#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${TZ:=Asia/Shanghai}"
export TZ

CONFIG_PATH="${1:-configs/experiment/free_loss_discovery/rl4co.yaml}"
DEVICE="${2:-cuda}"                  # cpu | cuda | cuda:0 ...
MODE="${3:-start}"                   # start | resume-latest | resume-dir
RESUME_DIR="${4:-}"

# Optional: external baseline folder (metrics.csv + checkpoint) for epoch-window comparisons.
# If `baseline/metrics.csv` exists, the script will auto-enable baseline comparison by
# creating a temporary config with a `baseline:` section, unless you explicitly set
# BASELINE_METRICS_CSV="" to disable.
: "${BASELINE_DIR:=baseline}"
: "${BASELINE_METRICS_CSV:=}"
: "${BASELINE_CKPT:=}"
: "${BASELINE_CKPT_EPOCH:=}"
: "${BASELINE_VAL_COLUMN:=val/reward}"

# Optional: OpenAI settings (you can also put OPENAI_API_KEY=... in a .env at repo root).
: "${OPENAI_MODEL:=gpt-5.2}"
: "${OPENAI_BASE_URL:=https://api.bltcy.ai/v1}"
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

if [[ -z "$BASELINE_METRICS_CSV" && -f "${ROOT_DIR}/${BASELINE_DIR}/metrics.csv" ]]; then
  BASELINE_METRICS_CSV="${BASELINE_DIR}/metrics.csv"
fi

if [[ -n "$BASELINE_METRICS_CSV" && -z "$BASELINE_CKPT" ]]; then
  # Pick the max epoch_*.ckpt under BASELINE_DIR (if any).
  BASELINE_CKPT="$(
    python -c "import glob,re,os; ckpts=glob.glob(os.path.join('${ROOT_DIR}','${BASELINE_DIR}','epoch_*.ckpt')); key=(lambda p: int(m.group(1)) if (m:=re.search(r'(?:^|[._-])epoch_(\\d+)(?:\\D|$)', os.path.basename(p))) else -1); print(max(ckpts, key=key) if ckpts else '')"
  )"
  if [[ -n "$BASELINE_CKPT" ]]; then
    # Normalize to a repo-relative path for the YAML.
    BASELINE_CKPT="${BASELINE_CKPT#${ROOT_DIR}/}"
  fi
fi

if [[ -n "$BASELINE_CKPT" && -z "$BASELINE_CKPT_EPOCH" ]]; then
  BASELINE_CKPT_EPOCH="$(
    python -c "import os,re; p=os.path.basename('${BASELINE_CKPT}'); m=re.search(r'(?:^|[._-])epoch_(\\d+)(?:\\D|$)', p); print(m.group(1) if m else '')"
  )"
fi

if [[ -n "$BASELINE_METRICS_CSV" && -n "$BASELINE_CKPT_EPOCH" ]]; then
  TMP_CFG="$(mktemp -t free_loss_eoh_cfg_XXXXXX.yaml)"
  python - "$CONFIG_PATH" "$TMP_CFG" "$BASELINE_METRICS_CSV" "$BASELINE_CKPT" "$BASELINE_CKPT_EPOCH" "$BASELINE_VAL_COLUMN" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, metrics_csv, ckpt, ckpt_epoch, val_col = sys.argv[1:]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8")) or {}
baseline = dict(cfg.get("baseline") or {})
baseline["metrics_csv"] = metrics_csv
baseline["val_column"] = val_col
baseline["checkpoint_epoch"] = int(ckpt_epoch)
if ckpt:
    baseline["checkpoint"] = ckpt
cfg["baseline"] = baseline
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  echo "Baseline enabled:"
  echo "  metrics_csv=$BASELINE_METRICS_CSV"
  echo "  checkpoint=$BASELINE_CKPT"
  echo "  checkpoint_epoch=$BASELINE_CKPT_EPOCH"
  echo "  val_column=$BASELINE_VAL_COLUMN"
  CONFIG_PATH="$TMP_CFG"
fi

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
