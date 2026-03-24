#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

launcher_log_dir="logs/op100_mechanism_ablation"
mkdir -p "${launcher_log_dir}"

timestamp="$(date +%Y%m%d_%H%M%S)"
launcher_log="${launcher_log_dir}/launcher_${timestamp}.log"

nohup bash run_op100_mechanism_ablations_4gpu.sh "$@" > "${launcher_log}" 2>&1 &
launcher_pid=$!

echo "Started OP100 mechanism ablations with nohup"
echo "  pid     : ${launcher_pid}"
echo "  launcher: ${launcher_log}"
