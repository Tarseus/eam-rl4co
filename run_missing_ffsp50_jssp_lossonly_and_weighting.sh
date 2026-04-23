#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

FFSP50_LOSS_GPU="${FFSP50_LOSS_GPU:-1}"
FFSP50_WEIGHT_GPU="${FFSP50_WEIGHT_GPU:-2}"
JSSP10_LOSS_GPU="${JSSP10_LOSS_GPU:-3}"
JSSP15_LOSS_GPU="${JSSP15_LOSS_GPU:-4}"

FFSP50_PO_BASELINE_CKPT="${FFSP50_PO_BASELINE_CKPT:-${ROOT_DIR}/baseline/ffsp_matnet_po_paper50_epoch_75.ckpt}"
JSSP10_BASELINE_CKPT="${JSSP10_BASELINE_CKPT:-${ROOT_DIR}/baseline/jssp100_epoch_016.ckpt}"
JSSP15_BASELINE_CKPT="${JSSP15_BASELINE_CKPT:-/data1/gushengda/eam-rl4co/logs/train/runs/mgl-jssp-bopo_bucketed-multishape_15x15_20260421-050017/checkpoints/last.ckpt}"
FFSP100_SOURCE_CHECKPOINT_JSON="${FFSP100_SOURCE_CHECKPOINT_JSON:-runs/pref_loss_ffsp100_discovery/20260403-142801/checkpoint.json}"

FFSP50_LOSS_ROOT="${FFSP50_LOSS_ROOT:-runs/pref_loss_ffsp50_discovery}"
FFSP50_WEIGHT_ROOT="${FFSP50_WEIGHT_ROOT:-runs/pref_builder_weight_search_ffsp50}"
JSSP10_LOSS_ROOT="${JSSP10_LOSS_ROOT:-runs/pref_loss_jssp10x10_from_ffsp100_elite}"
JSSP15_LOSS_ROOT="${JSSP15_LOSS_ROOT:-runs/pref_loss_jssp15x15_from_ffsp100_elite}"

WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-259200}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs"
TMP_CFG_DIR="${LOG_DIR}/generated_configs"

mkdir -p "${LOG_DIR}" "${TMP_CFG_DIR}" "${ROOT_DIR}/logs/train/runs"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="${LOG_TZ:-Asia/Shanghai}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export HYDRA_FULL_ERROR=1

usage() {
  cat <<'EOF'
Usage:
  bash run_missing_ffsp50_jssp_lossonly_and_weighting.sh

This script starts four background workers:
  1. FFSP50 loss-only discovery -> FFSP50 loss-only full train
  2. FFSP50 weighting search    -> FFSP50 weighting full train
  3. JSSP10x10 loss-only full train (uses latest discovered best_pair)
  4. JSSP15x15 loss-only discovery -> JSSP15x15 loss-only full train

Environment overrides:
  PYTHON_BIN=python
  FFSP50_LOSS_GPU=1
  FFSP50_WEIGHT_GPU=2
  JSSP10_LOSS_GPU=3
  JSSP15_LOSS_GPU=4
  FFSP50_PO_BASELINE_CKPT=/abs/path/to/ffsp50_po.ckpt
  JSSP10_BASELINE_CKPT=/abs/path/to/jssp10x10.ckpt
  JSSP15_BASELINE_CKPT=/abs/path/to/jssp15x15.ckpt
  FFSP100_SOURCE_CHECKPOINT_JSON=runs/pref_loss_ffsp100_discovery/.../checkpoint.json
  WAIT_TIMEOUT_SECONDS=259200
  POLL_SECONDS=30
  DRY_RUN=1
EOF
}

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+=" $(printf '%q' "${arg}")"
  done
  printf '%s\n' "${out# }"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

parse_epoch_from_ckpt() {
  local ckpt_path="$1"
  local base
  base="$(basename "${ckpt_path}")"
  if [[ "${base}" =~ ([0-9]{1,4}) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]#0}"
  else
    printf '%s\n' ""
  fi
}

find_latest_run_with_file() {
  local root="$1"
  local filename="$2"
  "$PYTHON_BIN" - "$root" "$filename" <<'PY'
import os
import sys

root = sys.argv[1]
filename = sys.argv[2]
if not os.path.isdir(root):
    sys.exit(1)

cands = []
for name in os.listdir(root):
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        continue
    target = os.path.join(path, filename)
    if not os.path.isfile(target):
        continue
    try:
        mtime = os.path.getmtime(target)
    except OSError:
        mtime = 0.0
    cands.append((mtime, path))

if not cands:
    sys.exit(2)

cands.sort()
print(cands[-1][1])
PY
}

wait_for_latest_run_with_file() {
  local root="$1"
  local filename="$2"
  local timeout_seconds="$3"
  local poll_seconds="$4"

  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if latest="$(find_latest_run_with_file "${root}" "${filename}" 2>/dev/null)"; then
      printf '%s\n' "${latest}"
      return 0
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_seconds )); then
      echo "ERROR: timeout waiting for ${filename} under ${root}" >&2
      return 1
    fi
    sleep "${poll_seconds}"
  done
}

run_fg() {
  echo "cmd=$(quote_cmd "$@")"
  if [[ "${DRY_RUN}" != "0" ]]; then
    return 0
  fi
  "$@"
}

prepare_bopo_jssp_data_if_needed() {
  if [[ "${PREPARED_BOPO_JSP_DATA:-0}" == "1" ]]; then
    return 0
  fi
  echo "[prep] scripts/prepare_bopo_jsp_data.py"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "${PYTHON_BIN}" scripts/prepare_bopo_jsp_data.py >/dev/null
  fi
  PREPARED_BOPO_JSP_DATA=1
}

make_ffsp50_loss_config() {
  local out_path="$1"
  local baseline_ckpt="$2"
  local checkpoint_epoch="$3"
  "${PYTHON_BIN}" - "$out_path" "$baseline_ckpt" "$checkpoint_epoch" <<'PY'
import sys
from pathlib import Path

import yaml

out_path = Path(sys.argv[1])
baseline_ckpt = str(Path(sys.argv[2]).resolve())
checkpoint_epoch = int(sys.argv[3]) if sys.argv[3] else None

src = Path("PTP/configs/experiment/pref_loss_coevo/loss_only_ffsp100_discovery.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

cfg["output_root"] = "runs/pref_loss_ffsp50_discovery"
cfg["devices"] = ["cuda:0"]
cfg["mp"] = {"enabled": False, "processes": 1, "start_method": "spawn"}
cfg["stage3_early_prune"]["scenario_name"] = "ffsp50"
cfg["train_problem_size"] = 50
cfg["valid_problem_sizes"] = [50]
cfg["generator_params"]["num_job"] = 50

baseline = cfg["baseline"]
baseline["checkpoint"] = baseline_ckpt
baseline["checkpoints"] = [baseline_ckpt]
if checkpoint_epoch is not None:
    baseline["checkpoint_epoch"] = checkpoint_epoch
for scenario in baseline.get("scenarios", []):
    scenario["name"] = "ffsp50"
    scenario["train_problem_size"] = 50
    scenario["valid_problem_sizes"] = [50]
    scenario["generator_params"]["num_job"] = 50
    nested = scenario.setdefault("baseline", {})
    nested["checkpoints"] = [baseline_ckpt]
    if checkpoint_epoch is not None:
        nested["checkpoint_epoch"] = checkpoint_epoch

out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out_path.as_posix())
PY
}

make_ffsp50_weight_config() {
  local out_path="$1"
  local baseline_ckpt="$2"
  local checkpoint_epoch="$3"
  local source_loss_path="$4"
  "${PYTHON_BIN}" - "$out_path" "$baseline_ckpt" "$checkpoint_epoch" "$source_loss_path" <<'PY'
import sys
from pathlib import Path

import yaml

out_path = Path(sys.argv[1])
baseline_ckpt = str(Path(sys.argv[2]).resolve())
checkpoint_epoch = int(sys.argv[3]) if sys.argv[3] else None
source_loss_path = sys.argv[4]

src = Path("PTP/configs/experiment/pref_loss_coevo/ffsp100_builder_weight_search_from_archive.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

cfg["output_root"] = "runs/pref_builder_weight_search_ffsp50"
cfg["devices"] = ["cuda:0"]
cfg["mp"] = {"enabled": False, "processes": 1, "start_method": "spawn"}
cfg["stage3_early_prune"]["scenario_name"] = "ffsp50"
cfg["loss_transfer_seed"]["source_loss_path"] = source_loss_path
cfg["train_problem_size"] = 50
cfg["valid_problem_sizes"] = [50]
cfg["proxy_problem_size"] = 50
cfg["generator_params"]["num_job"] = 50

baseline = cfg["baseline"]
baseline["checkpoint"] = baseline_ckpt
baseline["checkpoints"] = [baseline_ckpt]
if checkpoint_epoch is not None:
    baseline["checkpoint_epoch"] = checkpoint_epoch
for scenario in baseline.get("scenarios", []):
    scenario["name"] = "ffsp50"
    scenario["train_problem_size"] = 50
    scenario["valid_problem_sizes"] = [50]
    scenario["generator_params"]["num_job"] = 50
    nested = scenario.setdefault("baseline", {})
    nested["checkpoints"] = [baseline_ckpt]
    if checkpoint_epoch is not None:
        nested["checkpoint_epoch"] = checkpoint_epoch

out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out_path.as_posix())
PY
}

make_jssp15_loss_config() {
  local out_path="$1"
  local baseline_ckpt="$2"
  local checkpoint_epoch="$3"
  local source_checkpoint_json="$4"
  "${PYTHON_BIN}" - "$out_path" "$baseline_ckpt" "$checkpoint_epoch" "$source_checkpoint_json" <<'PY'
import sys
from pathlib import Path

import yaml

out_path = Path(sys.argv[1])
baseline_ckpt = str(Path(sys.argv[2]).resolve())
checkpoint_epoch = int(sys.argv[3]) if sys.argv[3] else None
source_checkpoint_json = sys.argv[4]

src = Path("PTP/configs/experiment/pref_loss_coevo/loss_transfer_jssp10x10_from_ffsp100_elite.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

cfg["output_root"] = "runs/pref_loss_jssp15x15_from_ffsp100_elite"
cfg["devices"] = ["cuda:0"]
cfg["mp"] = {"enabled": False, "processes": 1, "start_method": "spawn"}
cfg["loss_transfer_seed"]["source_checkpoint_path"] = source_checkpoint_json
cfg["stage3_early_prune"]["scenario_name"] = "jssp15x15"
cfg["train_problem_size"] = 15
cfg["valid_problem_sizes"] = [15]
cfg["generator_params"]["num_jobs"] = 15
cfg["generator_params"]["num_machines"] = 15
cfg["policy_kwargs"]["allowed_shapes"] = [[15, 15]]

baseline = cfg["baseline"]
baseline["checkpoints"] = [baseline_ckpt]
if checkpoint_epoch is not None:
    baseline["checkpoint_epoch"] = checkpoint_epoch
for scenario in baseline.get("scenarios", []):
    scenario["name"] = "jssp15x15"
    scenario["train_problem_size"] = 15
    scenario["valid_problem_sizes"] = [15]
    scenario["generator_params"]["num_jobs"] = 15
    scenario["generator_params"]["num_machines"] = 15
    nested = scenario.setdefault("baseline", {})
    nested["checkpoints"] = [baseline_ckpt]
    if checkpoint_epoch is not None:
        nested["checkpoint_epoch"] = checkpoint_epoch

out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out_path.as_posix())
PY
}

make_jssp10_loss_config() {
  local out_path="$1"
  local baseline_ckpt="$2"
  local checkpoint_epoch="$3"
  local source_checkpoint_json="$4"
  "${PYTHON_BIN}" - "$out_path" "$baseline_ckpt" "$checkpoint_epoch" "$source_checkpoint_json" <<'PY'
import sys
from pathlib import Path

import yaml

out_path = Path(sys.argv[1])
baseline_ckpt = str(Path(sys.argv[2]).resolve())
checkpoint_epoch = int(sys.argv[3]) if sys.argv[3] else None
source_checkpoint_json = sys.argv[4]

src = Path("PTP/configs/experiment/pref_loss_coevo/loss_transfer_jssp10x10_from_ffsp100_elite.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

cfg["output_root"] = "runs/pref_loss_jssp10x10_from_ffsp100_elite"
cfg["devices"] = ["cuda:0"]
cfg["mp"] = {"enabled": False, "processes": 1, "start_method": "spawn"}
cfg["loss_transfer_seed"]["source_checkpoint_path"] = source_checkpoint_json
cfg["stage3_early_prune"]["scenario_name"] = "jssp10x10"
cfg["train_problem_size"] = 10
cfg["valid_problem_sizes"] = [10]
cfg["generator_params"]["num_jobs"] = 10
cfg["generator_params"]["num_machines"] = 10
cfg["policy_kwargs"]["allowed_shapes"] = [[10, 10]]

baseline = cfg["baseline"]
baseline["checkpoints"] = [baseline_ckpt]
if checkpoint_epoch is not None:
    baseline["checkpoint_epoch"] = checkpoint_epoch
for scenario in baseline.get("scenarios", []):
    scenario["name"] = "jssp10x10"
    scenario["train_problem_size"] = 10
    scenario["valid_problem_sizes"] = [10]
    scenario["generator_params"]["num_jobs"] = 10
    scenario["generator_params"]["num_machines"] = 10
    nested = scenario.setdefault("baseline", {})
    nested["checkpoints"] = [baseline_ckpt]
    if checkpoint_epoch is not None:
        nested["checkpoint_epoch"] = checkpoint_epoch

out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out_path.as_posix())
PY
}

run_pref_loss_search() {
  local config_path="$1"
  run_fg "${PYTHON_BIN}" -u PTP/ptp_discovery/run_pref_loss_coevo.py --config "${config_path}"
}

run_ffsp_full_train() {
  local label="$1"
  local best_pair_path="$2"
  local size="$3"
  local run_dir="${ROOT_DIR}/logs/train/runs/${label}_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"

  run_fg \
    "${PYTHON_BIN}" -u run.py \
    "experiment=scheduling/ffsp-matnet-po-paper${size}" \
    "hydra.run.dir=${run_dir}" \
    "model.loss_type=free_loss" \
    "+model.pref_pair_json_path=${best_pair_path}" \
    "~callbacks.learning_rate_monitor" \
    "~callbacks.rich_progress_bar" \
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}" \
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
    "callbacks.model_checkpoint.auto_insert_metric_name=False" \
    "callbacks.model_checkpoint.save_top_k=1" \
    "callbacks.model_checkpoint.save_last=True" \
    "trainer.accelerator=gpu" \
    "+trainer.devices=[0]" \
    "+trainer.enable_progress_bar=false" \
    "logger=csv" \
    "logger.csv.name=${label}"
}

run_jssp10_full_train() {
  local best_pair_path="$1"
  local run_dir="${ROOT_DIR}/logs/train/runs/mgl-jssp-bopo-lossonly_10x10_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"

  prepare_bopo_jssp_data_if_needed

  run_fg \
    "${PYTHON_BIN}" -u run.py \
    "experiment=scheduling/mgl-jssp-bopo-paper" \
    "hydra.run.dir=${run_dir}" \
    "~callbacks.learning_rate_monitor" \
    "~callbacks.rich_progress_bar" \
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}" \
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
    "callbacks.model_checkpoint.auto_insert_metric_name=False" \
    "callbacks.model_checkpoint.save_top_k=1" \
    "callbacks.model_checkpoint.save_last=True" \
    "callbacks.model_checkpoint.every_n_epochs=1" \
    "trainer.accelerator=gpu" \
    "+trainer.devices=[0]" \
    "trainer.max_epochs=20" \
    "+trainer.enable_progress_bar=false" \
    "logger=csv" \
    "logger.csv.name=jssp10x10_loss_only_pref" \
    "+model.pref_pair_json_path=${best_pair_path}" \
    "+model.allowed_shapes=[[10,10]]" \
    "+model.required_allowed_shapes=[[10,10]]" \
    "+model.expected_train_dataset_size=5000" \
    "+model.expected_val_dataset_size=100"
}

run_jssp15_full_train() {
  local best_pair_path="$1"
  local run_dir="${ROOT_DIR}/logs/train/runs/mgl-jssp-bopo-lossonly_15x15_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"

  prepare_bopo_jssp_data_if_needed

  run_fg \
    "${PYTHON_BIN}" -u run.py \
    "experiment=scheduling/mgl-jssp-bopo-bucketed-multishape" \
    "hydra.run.dir=${run_dir}" \
    "~callbacks.learning_rate_monitor" \
    "~callbacks.rich_progress_bar" \
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}" \
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'" \
    "callbacks.model_checkpoint.auto_insert_metric_name=False" \
    "callbacks.model_checkpoint.save_top_k=1" \
    "callbacks.model_checkpoint.save_last=True" \
    "callbacks.model_checkpoint.every_n_epochs=1" \
    "trainer.accelerator=gpu" \
    "+trainer.devices=[0]" \
    "trainer.max_epochs=20" \
    "+trainer.enable_progress_bar=false" \
    "logger=csv" \
    "logger.csv.name=jssp15x15_loss_only_pref" \
    "+model.pref_pair_json_path=${best_pair_path}" \
    "+model.allowed_shapes=[[15,15]]" \
    "+model.required_allowed_shapes=[[15,15]]" \
    "+model.expected_train_dataset_size=5000" \
    "+model.expected_val_dataset_size=100"
}

worker_ffsp50_loss_only() {
  local epoch
  epoch="$(parse_epoch_from_ckpt "${FFSP50_PO_BASELINE_CKPT}")"
  require_file "${FFSP50_PO_BASELINE_CKPT}" "FFSP50 PO baseline checkpoint"

  local cfg_path="${TMP_CFG_DIR}/ffsp50_loss_only_${TIMESTAMP}.yaml"
  make_ffsp50_loss_config "${cfg_path}" "${FFSP50_PO_BASELINE_CKPT}" "${epoch}"

  echo "[ffsp50-loss-only] config=${cfg_path}"
  run_pref_loss_search "${cfg_path}"

  local latest_run
  latest_run="$(wait_for_latest_run_with_file "${FFSP50_LOSS_ROOT}" "best_pair.json" 60 5)"
  echo "[ffsp50-loss-only] latest_run=${latest_run}"
  run_ffsp_full_train "ffsp50_loss_only_pref" "${latest_run}/best_pair.json" 50
}

worker_ffsp50_weighting() {
  local epoch
  epoch="$(parse_epoch_from_ckpt "${FFSP50_PO_BASELINE_CKPT}")"
  require_file "${FFSP50_PO_BASELINE_CKPT}" "FFSP50 PO baseline checkpoint"

  local source_run
  source_run="$(wait_for_latest_run_with_file "${FFSP50_LOSS_ROOT}" "best_loss.json" "${WAIT_TIMEOUT_SECONDS}" "${POLL_SECONDS}")"
  local cfg_path="${TMP_CFG_DIR}/ffsp50_weight_search_${TIMESTAMP}.yaml"

  make_ffsp50_weight_config "${cfg_path}" "${FFSP50_PO_BASELINE_CKPT}" "${epoch}" "${source_run}/best_loss.json"

  echo "[ffsp50-weighting] source_loss=${source_run}/best_loss.json"
  echo "[ffsp50-weighting] config=${cfg_path}"
  run_pref_loss_search "${cfg_path}"

  local latest_run
  latest_run="$(wait_for_latest_run_with_file "${FFSP50_WEIGHT_ROOT}" "best_pair.json" 60 5)"
  echo "[ffsp50-weighting] latest_run=${latest_run}"
  run_ffsp_full_train "ffsp50_weighting_pref" "${latest_run}/best_pair.json" 50
}

worker_jssp10_loss_only() {
  local latest_run

  prepare_bopo_jssp_data_if_needed

  if latest_run="$(find_latest_run_with_file "${JSSP10_LOSS_ROOT}" "best_pair.json" 2>/dev/null)"; then
    echo "[jssp10x10-loss-only] using existing discovery=${latest_run}"
  else
    local epoch
    local cfg_path="${TMP_CFG_DIR}/jssp10x10_loss_only_${TIMESTAMP}.yaml"
    epoch="$(parse_epoch_from_ckpt "${JSSP10_BASELINE_CKPT}")"
    require_file "${JSSP10_BASELINE_CKPT}" "JSSP10x10 baseline checkpoint"
    require_file "${FFSP100_SOURCE_CHECKPOINT_JSON}" "FFSP100 source checkpoint json"
    make_jssp10_loss_config "${cfg_path}" "${JSSP10_BASELINE_CKPT}" "${epoch}" "${FFSP100_SOURCE_CHECKPOINT_JSON}"
    echo "[jssp10x10-loss-only] config=${cfg_path}"
    run_pref_loss_search "${cfg_path}"
    latest_run="$(wait_for_latest_run_with_file "${JSSP10_LOSS_ROOT}" "best_pair.json" "${WAIT_TIMEOUT_SECONDS}" "${POLL_SECONDS}")"
  fi

  run_jssp10_full_train "${latest_run}/best_pair.json"
}

worker_jssp15_loss_only() {
  local epoch
  epoch="$(parse_epoch_from_ckpt "${JSSP15_BASELINE_CKPT}")"
  require_file "${FFSP100_SOURCE_CHECKPOINT_JSON}" "FFSP100 source checkpoint json"
  require_file "${JSSP15_BASELINE_CKPT}" "JSSP15x15 baseline checkpoint"

  local cfg_path="${TMP_CFG_DIR}/jssp15x15_loss_only_${TIMESTAMP}.yaml"
  make_jssp15_loss_config "${cfg_path}" "${JSSP15_BASELINE_CKPT}" "${epoch}" "${FFSP100_SOURCE_CHECKPOINT_JSON}"

  echo "[jssp15x15-loss-only] config=${cfg_path}"
  run_pref_loss_search "${cfg_path}"

  local latest_run
  latest_run="$(wait_for_latest_run_with_file "${JSSP15_LOSS_ROOT}" "best_pair.json" 60 5)"
  echo "[jssp15x15-loss-only] latest_run=${latest_run}"
  run_jssp15_full_train "${latest_run}/best_pair.json"
}

run_worker() {
  local worker="$1"
  case "${worker}" in
    ffsp50-loss-only)
      worker_ffsp50_loss_only
      ;;
    ffsp50-weighting)
      worker_ffsp50_weighting
      ;;
    jssp10-loss-only)
      worker_jssp10_loss_only
      ;;
    jssp15-loss-only)
      worker_jssp15_loss_only
      ;;
    *)
      echo "ERROR: unknown worker: ${worker}" >&2
      exit 2
      ;;
  esac
}

spawn_worker() {
  local worker="$1"
  local gpu_id="$2"
  local log_path="${LOG_DIR}/${worker}_${TIMESTAMP}.out"
  local -a cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu_id}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "DRY_RUN=${DRY_RUN}"
    "FFSP50_PO_BASELINE_CKPT=${FFSP50_PO_BASELINE_CKPT}"
    "JSSP10_BASELINE_CKPT=${JSSP10_BASELINE_CKPT}"
    "JSSP15_BASELINE_CKPT=${JSSP15_BASELINE_CKPT}"
    "FFSP100_SOURCE_CHECKPOINT_JSON=${FFSP100_SOURCE_CHECKPOINT_JSON}"
    "FFSP50_LOSS_ROOT=${FFSP50_LOSS_ROOT}"
    "FFSP50_WEIGHT_ROOT=${FFSP50_WEIGHT_ROOT}"
    "JSSP10_LOSS_ROOT=${JSSP10_LOSS_ROOT}"
    "JSSP15_LOSS_ROOT=${JSSP15_LOSS_ROOT}"
    "WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS}"
    "POLL_SECONDS=${POLL_SECONDS}"
    bash
    "$0"
    --worker
    "${worker}"
  )

  echo "[spawn] worker=${worker} gpu=${gpu_id}"
  echo "[spawn] log=${log_path}"
  echo "[spawn] cmd=$(quote_cmd "${cmd[@]}")"

  if [[ "${DRY_RUN}" != "0" ]]; then
    return 0
  fi

  nohup "${cmd[@]}" >"${log_path}" 2>&1 < /dev/null &
  echo "[spawn] pid=$!"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "ERROR: --worker requires a worker name" >&2
    exit 2
  fi
  worker_name="$2"
  shift 2
  run_worker "${worker_name}"
  exit 0
fi

spawn_worker "ffsp50-loss-only" "${FFSP50_LOSS_GPU}"
spawn_worker "ffsp50-weighting" "${FFSP50_WEIGHT_GPU}"
spawn_worker "jssp10-loss-only" "${JSSP10_LOSS_GPU}"
spawn_worker "jssp15-loss-only" "${JSSP15_LOSS_GPU}"
