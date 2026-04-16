#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting TSP Full Training - Parallel"
echo "=========================================="
echo ""
echo "TSP100 will run on: cuda:0"
echo "TSP50 will run on: cuda:1 (using TSP100's best_pair)"
echo ""

# Export PYTHONPATH
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="${LOG_TZ:-Asia/Shanghai}"
: "${LOG_LEVEL:=INFO}"
export LOG_LEVEL
: "${HYDRA_FULL_ERROR:=1}"
export HYDRA_FULL_ERROR

# Function to find latest best_pair.json in a run dir
find_latest_best_pair() {
    local run_dir="$1"
    local latest_run=""
    local best_pair_path=""

    if [[ -d "$run_dir" ]]; then
        latest_run="$(
            python3 - "$run_dir" <<'PY'
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
    try:
        mtime = os.path.getmtime(best_pair)
    except OSError:
        mtime = 0.0
    cands.append((mtime, name))

if cands:
    cands.sort()
    print(cands[-1][1])
PY
        )" || true
    fi

    if [[ -z "${latest_run:-}" ]]; then
        echo ""
        return 1
    fi

    echo "${run_dir}/${latest_run}/best_pair.json"
}

# Function to run full training on specific GPU with given best_pair
run_full_train() {
    local best_pair_path="$1"
    local experiment="$2"
    local gpu_id="$3"
    local log_suffix="$4"

    echo "Starting ${log_suffix} on GPU ${gpu_id}..."
    echo "  Using best_pair: ${best_pair_path}"

    if [[ ! -f "$best_pair_path" ]]; then
        echo "ERROR: best_pair not found: $best_pair_path"
        return 1
    fi

    local log_path="${LOG_DIR}/full_train_${log_suffix}_$(date +%Y%m%d-%H%M%S).out"

    # Build command
    local cmd=(
        python3 -u run.py
        "experiment=${experiment}"
        "model.loss_type=free_loss"
        "+model.pref_pair_json_path=${best_pair_path}"
        "seed=1234"
        "trainer.deterministic=false"
        "trainer.devices=[0]"
        "matmul_precision=highest"
        "~callbacks.rich_progress_bar"
    )

    echo "  Running on GPU ${gpu_id}: ${cmd[*]}"
    echo "  Log: ${log_path}"

    # Run with specific GPU
    CUDA_VISIBLE_DEVICES="${gpu_id}" nohup "${cmd[@]}" >"${log_path}" 2>&1 &
    local pid=$!
    echo "  Started PID: ${pid}"
    echo ""

    echo "$pid"
}

# Find TSP100's best_pair
echo "Finding TSP100's best_pair..."
tsp100_best_pair=$(find_latest_best_pair "runs/pref_builder_weight_search_tsp100")

if [[ -z "$tsp100_best_pair" ]]; then
    echo "ERROR: Could not find best_pair.json for TSP100"
    exit 1
fi

echo "Found TSP100 best_pair: $tsp100_best_pair"
echo ""

# Run TSP100 on cuda:0
echo "Starting TSP100 full training..."
pid_tsp100=$(run_full_train \
    "$tsp100_best_pair" \
    "routing/pomo-po4cops-tsp100-po" \
    "0" \
    "tsp100")

sleep 3

# Run TSP50 on cuda:1, using TSP100's best_pair
echo "Starting TSP50 full training (using TSP100's best_pair)..."
pid_tsp50=$(run_full_train \
    "$tsp100_best_pair" \
    "routing/pomo-po4cops-tsp50-po" \
    "1" \
    "tsp50")

echo "=========================================="
echo "Both training jobs started!"
echo "=========================================="
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/full_train_tsp100_*.out"
echo "  tail -f logs/full_train_tsp50_*.out"
echo ""
echo "PIDs:"
echo "  TSP100: ${pid_tsp100:-N/A}"
echo "  TSP50: ${pid_tsp50:-N/A}"
