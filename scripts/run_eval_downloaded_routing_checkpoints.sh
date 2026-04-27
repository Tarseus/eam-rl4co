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
FFSP_AUG_FACTOR="${FFSP_AUG_FACTOR:-128}"
FFSP_AUG_BATCH_SIZE="${FFSP_AUG_BATCH_SIZE:-128}"
RESUME="${RESUME:-1}"

EXTRA_ARGS=("$@")

echo "[routing-eval] repo_root=${REPO_ROOT}"
echo "[routing-eval] python=${PYTHON_BIN}"
echo "[routing-eval] device=${DEVICE}"
echo "[routing-eval] precision=${PRECISION}"
echo "[routing-eval] manifest=${MANIFEST_PATH}"
echo "[routing-eval] output_csv=${OUTPUT_CSV}"
echo "[routing-eval] problems=${PROBLEMS}"
echo "[routing-eval] methods=${METHODS}"
echo "[routing-eval] ffsp_aug_factor=${FFSP_AUG_FACTOR}"
echo "[routing-eval] ffsp_aug_batch_size=${FFSP_AUG_BATCH_SIZE}"
echo "[routing-eval] resume=${RESUME}"
echo "[routing-eval] This wrapper targets tsp/cvrp/ffsp checkpoints from downloads/manifest.json."
echo "[routing-eval] Problem/method coverage from current manifest:"
echo "  - tsp50: bopo, sll, weighting"
echo "  - tsp100: po, bopo, sll, loss_only, weighting"
echo "  - cvrp50: bopo, sll"
echo "  - cvrp100: po, bopo, sll, loss_only, weighting"
echo "  - ffsp50: rl, po, bopo"
echo "  - ffsp100: po, bopo"
echo "[routing-eval] The Python evaluator will skip any entry that already has test/max_aug_reward."
echo "[routing-eval] FFSP note: max_aug_reward is computed by repeated RandomOneHot inference passes (default x${FFSP_AUG_FACTOR}, override with FFSP_AUG_FACTOR=...)."

CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/scripts/eval_downloaded_routing_checkpoints.py"
  "--manifest" "${MANIFEST_PATH}"
  "--output-csv" "${OUTPUT_CSV}"
  "--problems" "${PROBLEMS}"
  "--methods" "${METHODS}"
  "--device" "${DEVICE}"
  "--precision" "${PRECISION}"
  "--ffsp-aug-factor" "${FFSP_AUG_FACTOR}"
  "--ffsp-aug-batch-size" "${FFSP_AUG_BATCH_SIZE}"
)

if [[ "${RESUME}" == "1" ]]; then
  CMD+=("--resume")
fi

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
