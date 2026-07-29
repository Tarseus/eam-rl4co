#!/usr/bin/env bash
set -uo pipefail

repo="/data1/gushengda/codex_worktrees/joint-search-matched-init-hf1-run"
run_id="tsp100-joint-matched-init-hf1-r5-20260728-1802"
run_root="/data1/gushengda/eam-rl4co_runs/$run_id"
env_file="/data1/gushengda/eam-rl4co/.env"
python_bin="/data1/gushengda/anaconda3/envs/rlco1/bin/python"

cd "$run_root"
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a
export HTTPS_PROXY="http://127.0.0.1:18081"
export https_proxy="http://127.0.0.1:18081"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES="1,3,6,7"
export PYTHONPATH="$repo:$repo/PTP"

started_at="$(date -Is)"
"$python_bin" -u "$repo/PTP/ptp_discovery/run_pref_loss_coevo.py" \
  --config "$run_root/config.runtime.yaml" 2>&1 \
  | "$python_bin" -u "$run_root/rotate_stream.py" "$run_root/launcher" 90000000
train_exit="${PIPESTATUS[0]}"
printf '{"run_id":"%s","source_commit":"%s","started_at":"%s","ended_at":"%s","exit_code":%d}\n' \
  "$run_id" \
  "8d49846374e383ac574b7b0ff2a467cb734fa35a" \
  "$started_at" \
  "$(date -Is)" \
  "$train_exit" \
  > "$run_root/launcher_exit.json"
exit "$train_exit"
