#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-4}"
GPU4="${GPU4:-5}"

PAIR_REASONS="${PAIR_REASONS:-cheap_gate_failed,pref_semantic_failed,co_gate_failed}"
MIN_PER_GENERATION="${MIN_PER_GENERATION:-2}"
PURE_NO_GATE="${PURE_NO_GATE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Symbolic sampling budget. Sampling happens first, then each run is split into 2 shards.
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
  bash run_replay_gate_rejected_all_problems.sh

Execution order:
  1. TSP
  2. CVRP
  3. FFSP
  4. JSSP

Within each problem, 4 GPUs are reused in two stages:
  Stage A: loss-only replay on shard 0/4, 1/4, 2/4, 3/4
  Stage B: weighting-only replay on shard 0/4, 1/4, 2/4, 3/4

Defaults:
  GPU1=1 GPU2=2 GPU3=3 GPU4=4
  PAIR_REASONS=cheap_gate_failed,pref_semantic_failed,co_gate_failed
  MIN_PER_GENERATION=2
  LOSS_SAMPLE_SIZE=20
  WEIGHT_SAMPLE_SIZE=10
  SAMPLE_SEED=1234
  SAMPLE_ALL=0
  PURE_NO_GATE=0
  DRY_RUN=0

Notes:
  - Default mode is symbolic sampling, not full replay.
  - Set SAMPLE_ALL=1 to replay all rejected pairs in each selected run.
  - Set PURE_NO_GATE=1 to also disable sandbox/repair-related helpers.
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

run_four_shards() {
  local label_prefix="$1"
  local run_dir="$2"
  local sample_size="$3"
  local output_prefix="$4"
  local log_prefix="$5"

  if [[ "${DRY_RUN}" != "0" ]]; then
    start_task \
      "${label_prefix}_s0" "${GPU1}" "${run_dir}" "${sample_size}" 4 0 \
      "${REPORT_DIR}/${output_prefix}_shard0_${TIMESTAMP}.json" \
      "${LOG_DIR}/${log_prefix}_s0_${TIMESTAMP}.out"
    start_task \
      "${label_prefix}_s1" "${GPU2}" "${run_dir}" "${sample_size}" 4 1 \
      "${REPORT_DIR}/${output_prefix}_shard1_${TIMESTAMP}.json" \
      "${LOG_DIR}/${log_prefix}_s1_${TIMESTAMP}.out"
    start_task \
      "${label_prefix}_s2" "${GPU3}" "${run_dir}" "${sample_size}" 4 2 \
      "${REPORT_DIR}/${output_prefix}_shard2_${TIMESTAMP}.json" \
      "${LOG_DIR}/${log_prefix}_s2_${TIMESTAMP}.out"
    start_task \
      "${label_prefix}_s3" "${GPU4}" "${run_dir}" "${sample_size}" 4 3 \
      "${REPORT_DIR}/${output_prefix}_shard3_${TIMESTAMP}.json" \
      "${LOG_DIR}/${log_prefix}_s3_${TIMESTAMP}.out"
    return 0
  fi

  local -a pids=()

  start_task \
    "${label_prefix}_s0" "${GPU1}" "${run_dir}" "${sample_size}" 4 0 \
    "${REPORT_DIR}/${output_prefix}_shard0_${TIMESTAMP}.json" \
    "${LOG_DIR}/${log_prefix}_s0_${TIMESTAMP}.out"
  pids+=( "${STARTED_PID}" )

  start_task \
    "${label_prefix}_s1" "${GPU2}" "${run_dir}" "${sample_size}" 4 1 \
    "${REPORT_DIR}/${output_prefix}_shard1_${TIMESTAMP}.json" \
    "${LOG_DIR}/${log_prefix}_s1_${TIMESTAMP}.out"
  pids+=( "${STARTED_PID}" )

  start_task \
    "${label_prefix}_s2" "${GPU3}" "${run_dir}" "${sample_size}" 4 2 \
    "${REPORT_DIR}/${output_prefix}_shard2_${TIMESTAMP}.json" \
    "${LOG_DIR}/${log_prefix}_s2_${TIMESTAMP}.out"
  pids+=( "${STARTED_PID}" )

  start_task \
    "${label_prefix}_s3" "${GPU4}" "${run_dir}" "${sample_size}" 4 3 \
    "${REPORT_DIR}/${output_prefix}_shard3_${TIMESTAMP}.json" \
    "${LOG_DIR}/${log_prefix}_s3_${TIMESTAMP}.out"
  pids+=( "${STARTED_PID}" )

  wait_for_group "${label_prefix}" "${pids[@]}"
}

run_problem() {
  local problem="$1"
  local loss_run="$2"
  local weight_run="$3"

  echo "[problem] start ${problem}"

  echo "[problem] stage loss ${problem}"
  run_four_shards \
    "${problem}_loss" "${loss_run}" "${LOSS_SAMPLE_SIZE}" \
    "${problem}_loss" "replay_${problem}_loss"

  echo "[problem] stage weight ${problem}"
  run_four_shards \
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
