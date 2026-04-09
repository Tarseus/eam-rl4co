#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU_ID="${1:-0}"
shift || true

# Match the FFSP100 loss-search baseline protocol as closely as possible for
# full training: same native PO loss family, same alpha/impl, same seed,
# batch size, and numeric precision. The underlying baseline script keeps the
# usual checkpoint/logging behavior.
exec bash "$ROOT_DIR/run_baseline_ffsp_matnet_po_paper.sh" \
  "$GPU_ID" \
  100 \
  "seed=1234" \
  "model.batch_size=32" \
  "trainer.precision=32-true" \
  "trainer.deterministic=false" \
  "matmul_precision=highest" \
  "model.po_impl=exponential" \
  "model.alpha=1.0" \
  "$@"
