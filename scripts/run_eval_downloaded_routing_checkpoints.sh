#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
PRECISION="${PRECISION:-32-true}"
PROBLEMS="${PROBLEMS:-tsp50,tsp100,cvrp50,cvrp100,ffsp50,ffsp100}"
METHODS="${METHODS:-rl,po,bopo,sll,loss_only,weighting}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/downloads/manifest.json}"
OUTPUT_CSV="${OUTPUT_CSV:-${REPO_ROOT}/downloads/tsp_cvrp_ffsp_checkpoint_test_results.csv}"

# Leave blank to keep checkpoint defaults.
NUM_INSTANCES="${NUM_INSTANCES:-}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-}"
LIMIT="${LIMIT:-}"

EXTRA_ARGS=("$@")

echo "[routing-eval] repo_root=${REPO_ROOT}"
echo "[routing-eval] python=${PYTHON_BIN}"
echo "[routing-eval] device=${DEVICE}"
echo "[routing-eval] precision=${PRECISION}"
echo "[routing-eval] manifest=${MANIFEST_PATH}"
echo "[routing-eval] output_csv=${OUTPUT_CSV}"
echo "[routing-eval] problems=${PROBLEMS}"
echo "[routing-eval] methods=${METHODS}"
echo "[routing-eval] This wrapper targets tsp/cvrp/ffsp checkpoints from downloads/manifest.json."
echo "[routing-eval] Problem/method coverage from current manifest:"
echo "  - tsp50: bopo, sll, weighting"
echo "  - tsp100: po, bopo, sll, loss_only, weighting"
echo "  - cvrp50: bopo, sll"
echo "  - cvrp100: po, bopo, sll, loss_only, weighting"
echo "  - ffsp50: rl, po, bopo"
echo "  - ffsp100: po, bopo"
echo "[routing-eval] The Python evaluator will skip any entry that already has test/max_aug_reward."
echo "[routing-eval] FFSP note: current downloaded MatNet checkpoints use num_augment=0, so the CSV may record test/max_aug_reward as empty with an explanatory note."

CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/scripts/eval_downloaded_routing_checkpoints.py"
  "--manifest" "${MANIFEST_PATH}"
  "--output-csv" "${OUTPUT_CSV}"
  "--problems" "${PROBLEMS}"
  "--methods" "${METHODS}"
  "--device" "${DEVICE}"
  "--precision" "${PRECISION}"
  "--resume"
)

if [[ -n "${NUM_INSTANCES}" ]]; then
  CMD+=("--num-instances" "${NUM_INSTANCES}")
fi

if [[ -n "${TEST_BATCH_SIZE}" ]]; then
  CMD+=("--test-batch-size" "${TEST_BATCH_SIZE}")
fi

if [[ -n "${LIMIT}" ]]; then
  CMD+=("--limit" "${LIMIT}")
fi

if [[ "${FORCE_EVAL:-0}" == "1" ]]; then
  CMD+=("--force")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[routing-eval] command:"
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${REPO_ROOT}"
"${CMD[@]}"
