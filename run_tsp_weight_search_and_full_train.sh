#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting TSP Weight Search and Full Train"
echo "=========================================="
echo ""
echo "TSP100 will run on: cuda:0"
echo "TSP50 will run on: cuda:1"
echo ""

# Step 1: Run Weight Search for both TSP100 and TSP50 in parallel
echo "Step 1: Starting Weight Search..."
echo "  Starting TSP100 weight search on cuda:0..."
./run_tsp100_weight_search.sh "" start "" "0"
sleep 2

echo "  Starting TSP50 weight search on cuda:1..."
./run_tsp50_weight_search.sh "" start "" "1"
sleep 2

echo ""
echo "Weight search jobs started in background!"
echo "Check logs at:"
echo "  - TSP100: logs/tsp100_weight_search_*.out"
echo "  - TSP50: logs/tsp50_weight_search_*.out"
echo ""
echo "Waiting for weight search to complete..."
echo "You can monitor progress with:"
echo "  tail -f logs/tsp100_weight_search_*.out"
echo "  tail -f logs/tsp50_weight_search_*.out"
echo ""
echo "After weight search completes, run full training with:"
echo "  ./run_tsp_full_train_only.sh"
