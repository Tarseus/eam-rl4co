#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec bash "$ROOT_DIR/run_full_train_latest_pair.sh" \
  "runs/pref_loss_cvrp100_from_tsp100_elite" \
  "routing/pomo-po4cops-cvrp100-po" \
  "$@"
