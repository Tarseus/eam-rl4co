#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_BASELINE_CKPT="baseline/jssp100_epoch_016.ckpt"

if [[ $# -lt 1 && -z "${JSSP_BASELINE_CKPT:-}" ]]; then
  echo "Usage: $0 <cuda_ids...> [start|resume-latest|resume-dir] [resume_dir] [config_path]" >&2
  echo "   or: JSSP_BASELINE_CKPT=/abs/path/to/ckpt $0 <cuda_ids...> [start|resume-latest|resume-dir] [resume_dir] [config_path]" >&2
  echo "Examples:" >&2
  echo "  $0 0 1 2 3" >&2
  echo "  $0 0,1,2,3" >&2
  echo "  JSSP_BASELINE_CKPT=/abs/path/to/jssp10x10.ckpt $0 0 1 2 3 resume-latest" >&2
  exit 2
fi

if [[ -n "${JSSP_BASELINE_CKPT:-}" ]]; then
  BASELINE_CKPT="$JSSP_BASELINE_CKPT"
else
  BASELINE_CKPT="$DEFAULT_BASELINE_CKPT"
fi

if [[ ! -f "$BASELINE_CKPT" ]]; then
  echo "ERROR: JSSP baseline checkpoint not found: $BASELINE_CKPT" >&2
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
BASE_CONFIG="${1:-PTP/configs/experiment/pref_loss_coevo/loss_transfer_jssp10x10_from_ffsp100_elite.yaml}"

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
LOG_PATH="${LOG_DIR}/pref_loss_jssp10x10_from_ffsp100_cuda${GPU_TAG}_${TS}.out"
TMP_CONFIG="${LOG_DIR}/pref_loss_jssp10x10_from_ffsp100_cuda${GPU_TAG}_${TS}.yaml"

"$PYTHON_BIN" - "$BASE_CONFIG" "$TMP_CONFIG" "$GPU_COUNT" "$BASELINE_CKPT" <<'PY'
import re
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
gpu_count = int(sys.argv[3])
baseline_ckpt = str(Path(sys.argv[4]).resolve())
match = re.search(r"epoch[_-]?(\d+)", Path(baseline_ckpt).name, flags=re.IGNORECASE)
checkpoint_epoch = int(match.group(1)) if match else None

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

cfg["devices"] = [f"cuda:{idx}" for idx in range(gpu_count)]
cfg["device"] = "cuda:0"
cfg["mp"] = {
    "enabled": bool(gpu_count > 1),
    "processes": gpu_count,
    "start_method": "spawn",
}

baseline_cfg = cfg.setdefault("baseline", {})
baseline_cfg["checkpoints"] = [baseline_ckpt]
if checkpoint_epoch is not None:
    baseline_cfg["checkpoint_epoch"] = int(checkpoint_epoch)

scenarios = baseline_cfg.get("scenarios", [])
if isinstance(scenarios, list):
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        nested = item.setdefault("baseline", {})
        nested["checkpoints"] = [baseline_ckpt]
        if checkpoint_epoch is not None:
            nested["checkpoint_epoch"] = int(checkpoint_epoch)

dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst.as_posix())
PY

"$PYTHON_BIN" - "$TMP_CONFIG" "$BASELINE_CKPT" <<'PY'
import sys
from pathlib import Path

import yaml

repo_root = Path(".").resolve()
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str((repo_root / "PTP").resolve()))

from PTP.fitness.free_loss_fidelity import run_rl4co_rollout_smoke_test  # noqa: E402
from PTP.fitness.ptp_high_fidelity import HighFidelityConfig  # noqa: E402

cfg_path = Path(sys.argv[1])
ckpt_path = str(Path(sys.argv[2]).resolve())
cfg_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

hf_cfg = HighFidelityConfig(
    backend=str(cfg_yaml.get("backend", "rl4co") or "rl4co"),
    env_name=str(cfg_yaml.get("env_name", "") or ""),
    policy_name=str(cfg_yaml.get("policy_name", "") or ""),
    generator_params=dict(cfg_yaml.get("generator_params", {}) or {}),
    policy_kwargs=dict(cfg_yaml.get("policy_kwargs", {}) or {}),
    train_problem_size=int(cfg_yaml.get("train_problem_size", 10) or 10),
    valid_problem_sizes=list(cfg_yaml.get("valid_problem_sizes", [10]) or [10]),
    train_batch_size=int(cfg_yaml.get("train_batch_size", 1) or 1),
    num_validation_episodes=int(cfg_yaml.get("num_validation_episodes", 16) or 16),
    validation_batch_size=int(cfg_yaml.get("validation_batch_size", 1) or 1),
    device="cpu",
)

result = run_rl4co_rollout_smoke_test(
    hf_cfg,
    init_checkpoint_path=ckpt_path,
    device="cpu",
    batch_size=1,
    num_rollouts=4,
)
if not bool(result.get("ok", False)):
    raise SystemExit(
        "ERROR: baseline checkpoint failed pref-loss MGL preflight.\n"
        f"checkpoint={ckpt_path}\n"
        f"policy_name={hf_cfg.policy_name}\n"
        f"error={result.get('error')}\n"
        f"traceback={result.get('error_traceback')}"
    )
print(f"Checkpoint preflight OK: {ckpt_path} ({result.get('rollout_strategy')})")
PY

export CUDA_VISIBLE_DEVICES="${GPU_CSV}"

CMD=("$PYTHON_BIN" -u PTP/ptp_discovery/run_pref_loss_coevo.py --config "$TMP_CONFIG")
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

echo "JSSP_BASELINE_CKPT=$(cd "$(dirname "$BASELINE_CKPT")" && pwd)/$(basename "$BASELINE_CKPT")"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Config: $TMP_CONFIG"
echo "Running: ${CMD[*]}"
echo "Log: $LOG_PATH"
nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
echo "Started PID: $!"
tail -f "$LOG_PATH"
