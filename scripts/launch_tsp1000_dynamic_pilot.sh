#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_INDEX OUTPUT_ROOT METHOD [METHOD ...]" >&2
  exit 2
fi

gpu_index="$1"
output_root="$2"
shift 2
methods=("$@")
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data1/gushengda/anaconda3/envs/rlco1/bin/python}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
mkdir -p "${repo_root}/${output_root}"

status=0
for method in "${methods[@]}"; do
  echo "[launcher] gpu=${gpu_index} method=${method}"
  CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" -u \
    "${repo_root}/scripts/train_tsp1000_objectives.py" \
    --method "${method}" \
    --target-size 1000 \
    --num-starts 20 \
    --bopo-select-k 10 \
    --steps 100 \
    --accumulate 1 \
    --seed 1234 \
    --learning-rate 1e-5 \
    --weight-decay 1e-6 \
    --device cuda:0 \
    --precision bf16-mixed \
    --validation-size 8 \
    --validation-every 25 \
    --validation-starts 1000 \
    --validation-augment 1 \
    --log-every 5 \
    --output-dir "${output_root}/${method}" || status=$?
done

exit "${status}"
