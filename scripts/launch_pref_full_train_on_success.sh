#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/launch_pref_full_train_on_success.sh <tsp50|tsp100|cvrp50|ffsp50> <best_pair.json> <cuda_visible_devices> [resume_ckpt_path] [hydra_overrides...]" >&2
  exit 2
fi

problem="$1"
best_pair_path="$2"
cuda_visible_devices="$3"
resume_ckpt_path="${4:-}"
if [[ $# -gt 4 ]]; then
  extra_overrides=("${@:5}")
else
  extra_overrides=()
fi

PYTHON_BIN="${PYTHON_BIN:-/data1/gushengda/anaconda3/envs/rlco1/bin/python}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="${LOG_TZ:-Asia/Shanghai}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"

case "$problem" in
  tsp50)
    experiment="routing/pomo-po4cops-tsp50-po"
    label="tsp50_weighting_pref_target"
    ;;
  tsp100)
    experiment="routing/pomo-po4cops-tsp100-po"
    label="tsp100_fitness_correlation"
    ;;
  cvrp50)
    experiment="routing/pomo-po4cops-cvrp50-po"
    label="cvrp50_weighting_pref_target"
    ;;
  ffsp50)
    experiment="scheduling/ffsp-matnet-po-paper50"
    label="ffsp50_weighting_pref_target"
    trainer_devices_override="+trainer.devices=[0]"
    progress_bar_override="+trainer.enable_progress_bar=false"
    disable_cudnn_sdp=1
    ;;
  *)
    echo "ERROR: unsupported problem: ${problem}" >&2
    exit 2
    ;;
esac

if [[ -n "${RUN_LABEL_SUFFIX:-}" ]]; then
  safe_suffix="$(printf '%s' "$RUN_LABEL_SUFFIX" | tr -cd 'A-Za-z0-9_.-')"
  if [[ -z "$safe_suffix" ]]; then
    echo "ERROR: RUN_LABEL_SUFFIX contains no safe filename characters" >&2
    exit 2
  fi
  label="${label}_${safe_suffix}"
fi

: "${trainer_devices_override:=trainer.devices=[0]}"
: "${progress_bar_override:=trainer.enable_progress_bar=false}"
: "${disable_cudnn_sdp:=0}"

if [[ ! -f "$best_pair_path" ]]; then
  echo "ERROR: best_pair.json not found: ${best_pair_path}" >&2
  exit 1
fi

if [[ -n "$resume_ckpt_path" && ! -f "$resume_ckpt_path" ]]; then
  echo "ERROR: resume checkpoint not found: ${resume_ckpt_path}" >&2
  exit 1
fi

ts="$(date +%Y%m%d-%H%M%S)"
run_dir="${ROOT_DIR}/logs/train/runs/${label}_${ts}"
ckpt_dir="${run_dir}/checkpoints"
log_dir="${ROOT_DIR}/logs/codex_remote"
log_path="${log_dir}/${label}_full_train_${ts}.log"
mkdir -p "$ckpt_dir" "$log_dir"

python_entry=("$PYTHON_BIN" -u run.py)
if [[ "$disable_cudnn_sdp" == "1" ]]; then
  python_entry=(
    "$PYTHON_BIN" -u -c
    "import torch; torch.backends.cuda.enable_cudnn_sdp(False); from rl4co.tasks.train import train; train()"
  )
fi

cmd=(
  "${python_entry[@]}"
  "experiment=${experiment}"
  "hydra.run.dir=${run_dir}"
  "model.loss_type=free_loss"
  "+model.pref_pair_json_path=${best_pair_path}"
  "~callbacks.learning_rate_monitor"
  "~callbacks.rich_progress_bar"
  "callbacks.model_checkpoint.dirpath=${ckpt_dir}"
  "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
  "callbacks.model_checkpoint.auto_insert_metric_name=False"
  "callbacks.model_checkpoint.save_top_k=1"
  "callbacks.model_checkpoint.save_last=True"
  "trainer.accelerator=gpu"
  "${trainer_devices_override}"
  "${progress_bar_override}"
  "logger=csv"
  "logger.csv.name=${label}"
  "test=True"
)

if [[ -n "$resume_ckpt_path" ]]; then
  cmd+=("ckpt_path=${resume_ckpt_path}")
fi

cmd+=("${extra_overrides[@]}")

{
  echo "problem=${problem}"
  echo "best_pair=${best_pair_path}"
  echo "cuda_visible_devices=${cuda_visible_devices}"
  echo "resume_ckpt=${resume_ckpt_path}"
  printf "extra_overrides="
  printf "%q " "${extra_overrides[@]}"
  printf "\n"
  echo "disable_cudnn_sdp=${disable_cudnn_sdp}"
  echo "run_dir=${run_dir}"
  echo "log_path=${log_path}"
  printf "command="
  printf "%q " "${cmd[@]}"
  printf "\n"
} > "$log_path"

nohup "${cmd[@]}" >> "$log_path" 2>&1 < /dev/null &
pid="$!"
echo "pid=${pid}" >> "$log_path"
echo "started full train: problem=${problem} pid=${pid} log=${log_path}"
