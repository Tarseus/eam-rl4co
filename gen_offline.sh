#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-PTP/configs/experiment/pref_loss_coevo/alternating_simple.yaml}"

ENV_NAME="${ENV_NAME:-tsp}"
PROBLEM_SIZE="${PROBLEM_SIZE:-100}"
VALID_PROBLEM_SIZES="${VALID_PROBLEM_SIZES:-100}"

OFFLINE_DIR="${OFFLINE_DIR:-offline_data}"
BASELINE_DIR="${BASELINE_DIR:-baseline/mini_eval}"

GLOBAL_SEED="${GLOBAL_SEED:-1234}"
SCRATCH_INIT_SEED="${SCRATCH_INIT_SEED:-1234}"

OFFLINE_TRAIN_SIZE="${OFFLINE_TRAIN_SIZE:-64}"
OFFLINE_VAL_SIZE="${OFFLINE_VAL_SIZE:-32}"
NUM_VALIDATION_EPISODES="${NUM_VALIDATION_EPISODES:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"

CKPT_135="${CKPT_135:-baseline/epoch_135.ckpt}"
CKPT_409="${CKPT_409:-baseline/epoch_409.ckpt}"

K_LIST="${K_LIST:-200 1000}"

mkdir -p "$OFFLINE_DIR" "$BASELINE_DIR"

IFS=' ' read -r -a VALID_SIZE_ARRAY <<< "$VALID_PROBLEM_SIZES"
ALL_SIZES=("$PROBLEM_SIZE")
for SZ in "${VALID_SIZE_ARRAY[@]}"; do
  SKIP=0
  for EXISTING in "${ALL_SIZES[@]}"; do
    if [[ "$EXISTING" == "$SZ" ]]; then
      SKIP=1
      break
    fi
  done
  if [[ "$SKIP" -eq 0 ]]; then
    ALL_SIZES+=("$SZ")
  fi
done

OFFLINE_VAL_ARGS=()
for SZ in "${VALID_SIZE_ARRAY[@]}"; do
  OFFLINE_VAL_ARGS+=(--offline_val "$SZ" "${OFFLINE_DIR}/${ENV_NAME}${SZ}_val.pt")
done

echo "[gen_offline] root=$ROOT_DIR"
echo "[gen_offline] python=$PYTHON_BIN"
echo "[gen_offline] seed=$GLOBAL_SEED scratch_init_seed=$SCRATCH_INIT_SEED"
echo "[gen_offline] generating offline data: env=$ENV_NAME sizes=${ALL_SIZES[*]} train=$OFFLINE_TRAIN_SIZE val=$OFFLINE_VAL_SIZE"

"$PYTHON_BIN" scripts/precompute_offline_instances.py \
  --env "$ENV_NAME" \
  --sizes "${ALL_SIZES[@]}" \
  --train_size "$OFFLINE_TRAIN_SIZE" \
  --val_size "$OFFLINE_VAL_SIZE" \
  --seed "$GLOBAL_SEED" \
  --out_dir "$OFFLINE_DIR"

for K in $K_LIST; do
  OUT_JSON="${BASELINE_DIR}/baseline_minitrain_${ENV_NAME}${PROBLEM_SIZE}_K${K}.json"
  echo "[gen_offline] generating baseline mini-eval: K=$K out=$OUT_JSON"

  "$PYTHON_BIN" scripts/eval_baseline_minitrain.py \
    --config "$CONFIG_PATH" \
    --K "$K" \
    --train_problem_size "$PROBLEM_SIZE" \
    --valid_problem_sizes $VALID_PROBLEM_SIZES \
    --num_validation_episodes "$NUM_VALIDATION_EPISODES" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --offline_train "${OFFLINE_DIR}/${ENV_NAME}${PROBLEM_SIZE}_train.pt" \
    "${OFFLINE_VAL_ARGS[@]}" \
    --scratch_init_seed "$SCRATCH_INIT_SEED" \
    --ckpt_135 "$CKPT_135" \
    --ckpt_409 "$CKPT_409" \
    --out "$OUT_JSON"
done

echo "[gen_offline] done"
