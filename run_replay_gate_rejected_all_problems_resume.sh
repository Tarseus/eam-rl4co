#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
GPU4="${GPU4:-4}"

PAIR_REASONS="${PAIR_REASONS:-cheap_gate_failed,pref_semantic_failed,co_gate_failed}"
MIN_PER_GENERATION="${MIN_PER_GENERATION:-2}"
PURE_NO_GATE="${PURE_NO_GATE:-1}"
DRY_RUN="${DRY_RUN:-0}"

LOSS_SAMPLE_SIZE="${LOSS_SAMPLE_SIZE:-20}"
WEIGHT_SAMPLE_SIZE="${WEIGHT_SAMPLE_SIZE:-10}"
SAMPLE_SEED="${SAMPLE_SEED:-1234}"
SAMPLE_ALL="${SAMPLE_ALL:-0}"

TSP_LOSS_RUN="${TSP_LOSS_RUN:-runs/pref_loss_tsp100_discovery/20260317-131507}"
TSP_WEIGHT_RUN="${TSP_WEIGHT_RUN:-runs/pref_builder_weight_search_tsp100/20260414-113757}"
CVRP_LOSS_RUN="${CVRP_LOSS_RUN:-runs/pref_loss_cvrp100_from_tsp100_elite/20260320-224008}"
CVRP_WEIGHT_RUN="${CVRP_WEIGHT_RUN:-runs/pref_builder_weight_search_cvrp100/20260416-182905}"
FFSP_LOSS_RUN="${FFSP_LOSS_RUN:-runs/pref_loss_ffsp100_discovery/20260403-142801}"
FFSP_WEIGHT_RUN="${FFSP_WEIGHT_RUN:-runs/pref_builder_weight_search_ffsp100/20260416-111514}"
JSSP_LOSS_RUN="${JSSP_LOSS_RUN:-runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409}"
JSSP_WEIGHT_RUN="${JSSP_WEIGHT_RUN:-runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs"
REPORT_DIR="${ROOT_DIR}/runs/replay_gate_rejected_reports"
mkdir -p "${LOG_DIR}" "${REPORT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="${LOG_TZ:-Asia/Shanghai}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export HYDRA_FULL_ERROR=1

usage() {
  cat <<'EOF'
Usage:
  bash run_replay_gate_rejected_all_problems_resume.sh

Behavior:
  - Replays TSP, CVRP, FFSP, and JSSP in order.
  - Runs both loss and weighting stages.
  - Skips any shard whose existing report is already complete for pure-no-gate.
  - Re-runs missing or incomplete shards only.

Defaults:
  GPU1=1 GPU2=2 GPU3=3 GPU4=4
  PAIR_REASONS=cheap_gate_failed,pref_semantic_failed,co_gate_failed
  MIN_PER_GENERATION=2
  LOSS_SAMPLE_SIZE=20
  WEIGHT_SAMPLE_SIZE=10
  SAMPLE_SEED=1234
  SAMPLE_ALL=0
  PURE_NO_GATE=1
  DRY_RUN=0
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

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

build_replay_cmd() {
  local run_dir="$1"
  local sample_size="$2"
  local num_shards="$3"
  local shard_index="$4"
  local output_path="$5"

  local -a cmd=(
    "${PYTHON_BIN}"
    "scripts/replay_gate_rejected_pairs.py"
    "--run-dir" "${run_dir}"
    "--device" "cuda:0"
    "--pair-reasons" "${PAIR_REASONS}"
    "--num-shards" "${num_shards}"
    "--shard-index" "${shard_index}"
    "--output" "${output_path}"
  )

  if [[ "${PURE_NO_GATE}" != "0" ]]; then
    cmd+=( "--pure-no-gate" )
  fi

  if [[ "${SAMPLE_ALL}" == "0" ]]; then
    cmd+=(
      "--sample-size" "${sample_size}"
      "--sample-seed" "${SAMPLE_SEED}"
      "--min-per-generation" "${MIN_PER_GENERATION}"
    )
  fi

  if [[ "${DRY_RUN}" != "0" ]]; then
    cmd+=( "--dry-run" )
  fi

  printf '%s\0' "${cmd[@]}"
}

expected_selected_count() {
  local run_dir="$1"
  local sample_size="$2"
  local shard_index="$3"

  local -a cmd=(
    "${PYTHON_BIN}"
    "scripts/replay_gate_rejected_pairs.py"
    "--run-dir" "${run_dir}"
    "--device" "cuda:0"
    "--pair-reasons" "${PAIR_REASONS}"
    "--num-shards" "4"
    "--shard-index" "${shard_index}"
    "--dry-run"
  )

  if [[ "${PURE_NO_GATE}" != "0" ]]; then
    cmd+=( "--pure-no-gate" )
  fi

  if [[ "${SAMPLE_ALL}" == "0" ]]; then
    cmd+=(
      "--sample-size" "${sample_size}"
      "--sample-seed" "${SAMPLE_SEED}"
      "--min-per-generation" "${MIN_PER_GENERATION}"
    )
  fi

  local json_out
  json_out="$("${cmd[@]}")"
  "${PYTHON_BIN}" -c 'import json, sys; print(json.loads(sys.stdin.read())["selected_count"])' <<<"${json_out}"
}

report_path_for_shard() {
  local output_prefix="$1"
  local shard_index="$2"

  local latest=""
  latest="$(ls -1t "${REPORT_DIR}/${output_prefix}_shard${shard_index}"_*.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
  else
    printf '%s\n' "${REPORT_DIR}/${output_prefix}_shard${shard_index}_${TIMESTAMP}.json"
  fi
}

report_is_complete() {
  local report_path="$1"
  local expected_count="$2"

  [[ -f "${report_path}" ]] || return 1

  "${PYTHON_BIN}" - "${report_path}" "${expected_count}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

if payload.get("pure_no_gate") is not True:
    raise SystemExit(1)

results = payload.get("results")
if not isinstance(results, list):
    raise SystemExit(1)

raise SystemExit(0 if len(results) == expected else 1)
PY
}

STARTED_PID=""

start_task() {
  local label="$1"
  local gpu_id="$2"
  local run_dir="$3"
  local sample_size="$4"
  local num_shards="$5"
  local shard_index="$6"
  local output_path="$7"
  local log_path="$8"

  require_dir "${run_dir}" "${label} run dir"

  local -a cmd=()
  while IFS= read -r -d '' arg; do
    cmd+=( "${arg}" )
  done < <(build_replay_cmd "${run_dir}" "${sample_size}" "${num_shards}" "${shard_index}" "${output_path}")

  echo "[${label}] gpu=${gpu_id}"
  echo "[${label}] log=${log_path}"
  echo "[${label}] out=${output_path}"
  echo "[${label}] cmd=$(quote_cmd "${cmd[@]}")"

  if [[ "${DRY_RUN}" != "0" ]]; then
    STARTED_PID=""
    env CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
    return 0
  fi

  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >"${log_path}" 2>&1 < /dev/null &
  STARTED_PID="$!"
  echo "[${label}] pid=${STARTED_PID}"
}

wait_for_group() {
  local problem="$1"
  shift
  local -a pids=( "$@" )
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if [[ -z "${pid}" ]]; then
      continue
    fi
    if ! wait "${pid}"; then
      echo "[${problem}] worker failed: pid=${pid}" >&2
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[${problem}] failed" >&2
    exit 1
  fi
}

run_four_shards_resume() {
  local label_prefix="$1"
  local run_dir="$2"
  local sample_size="$3"
  local output_prefix="$4"
  local log_prefix="$5"

  local -a pids=()
  local shard
  for shard in 0 1 2 3; do
    local expected_count
    expected_count="$(expected_selected_count "${run_dir}" "${sample_size}" "${shard}")"

    local report_path
    report_path="$(report_path_for_shard "${output_prefix}" "${shard}")"
    if report_is_complete "${report_path}" "${expected_count}"; then
      echo "[${label_prefix}_s${shard}] skip complete report=${report_path} expected=${expected_count}"
      continue
    fi

    local log_path="${LOG_DIR}/${log_prefix}_s${shard}_${TIMESTAMP}.out"
    start_task \
      "${label_prefix}_s${shard}" \
      "$(case "${shard}" in 0) echo "${GPU1}" ;; 1) echo "${GPU2}" ;; 2) echo "${GPU3}" ;; 3) echo "${GPU4}" ;; esac)" \
      "${run_dir}" \
      "${sample_size}" \
      4 \
      "${shard}" \
      "${report_path}" \
      "${log_path}"
    if [[ "${DRY_RUN}" == "0" ]]; then
      pids+=( "${STARTED_PID}" )
    fi
  done

  if [[ "${DRY_RUN}" == "0" && "${#pids[@]}" -gt 0 ]]; then
    wait_for_group "${label_prefix}" "${pids[@]}"
  fi
}

run_problem() {
  local problem="$1"
  local loss_run="$2"
  local weight_run="$3"

  echo "[problem] start ${problem}"

  echo "[problem] stage loss ${problem}"
  run_four_shards_resume \
    "${problem}_loss" "${loss_run}" "${LOSS_SAMPLE_SIZE}" \
    "${problem}_loss" "replay_${problem}_loss"

  echo "[problem] stage weight ${problem}"
  run_four_shards_resume \
    "${problem}_weight" "${weight_run}" "${WEIGHT_SAMPLE_SIZE}" \
    "${problem}_weight" "replay_${problem}_weight"

  echo "[problem] done ${problem}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

run_problem "tsp" "${TSP_LOSS_RUN}" "${TSP_WEIGHT_RUN}"
run_problem "cvrp" "${CVRP_LOSS_RUN}" "${CVRP_WEIGHT_RUN}"
run_problem "ffsp" "${FFSP_LOSS_RUN}" "${FFSP_WEIGHT_RUN}"
run_problem "jssp" "${JSSP_LOSS_RUN}" "${JSSP_WEIGHT_RUN}"
