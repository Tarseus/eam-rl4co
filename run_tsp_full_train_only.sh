#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting TSP Full Training Only"
echo "=========================================="
echo ""
echo "TSP100 will run on: cuda:0"
echo "TSP50 will run on: cuda:1"
echo ""

# Create temporary wrapper scripts for full training to set specific GPUs
cat > /tmp/full_train_tsp100.sh << 'EOF'
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT_DIR"
exec bash ./run_full_train_latest_pair.sh \
  "runs/pref_builder_weight_search_tsp100" \
  "routing/pomo-po4cops-tsp100-po" \
  "trainer.devices=[0]"
EOF

cat > /tmp/full_train_tsp50.sh << 'EOF'
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
cd "$ROOT_DIR"
exec bash ./run_full_train_latest_pair.sh \
  "runs/pref_builder_weight_search_tsp50" \
  "routing/pomo-po4cops-tsp50-po" \
  "trainer.devices=[0]"
EOF

chmod +x /tmp/full_train_tsp100.sh /tmp/full_train_tsp50.sh

# Start TSP100 full training on cuda:0
echo "Starting TSP100 full training on cuda:0..."
LOG_TSP100="${LOG_DIR}/full_train_tsp100_$(date +%Y%m%d-%H%M%S).out"
nohup bash /tmp/full_train_tsp100.sh > "$LOG_TSP100" 2>&1 &
PID_TSP100=$!
echo "  TSP100 PID: $PID_TSP100"
echo "  Log: $LOG_TSP100"

sleep 2

# Start TSP50 full training on cuda:1
echo "Starting TSP50 full training on cuda:1..."
LOG_TSP50="${LOG_DIR}/full_train_tsp50_$(date +%Y%m%d-%H%M%S).out"
nohup bash /tmp/full_train_tsp50.sh > "$LOG_TSP50" 2>&1 &
PID_TSP50=$!
echo "  TSP50 PID: $PID_TSP50"
echo "  Log: $LOG_TSP50"

echo ""
echo "Both full training jobs started!"
echo "Monitor with:"
echo "  tail -f $LOG_TSP100"
echo "  tail -f $LOG_TSP50"
