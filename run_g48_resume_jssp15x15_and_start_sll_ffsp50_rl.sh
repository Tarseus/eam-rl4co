#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-g48}"
REMOTE_REPO="${REMOTE_REPO:-/data1/gushengda/eam-rl4co}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-/data1/gushengda/anaconda3/envs/rlco1/bin/python}"
UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
if [[ "${UNAME_S}" =~ ^(MINGW|MSYS|CYGWIN) ]] && command -v ssh.exe >/dev/null 2>&1; then
  DEFAULT_SSH_BIN="ssh.exe"
else
  DEFAULT_SSH_BIN="ssh"
fi
SSH_BIN="${SSH_BIN:-${DEFAULT_SSH_BIN}}"

JSSP_RL_GPU="${JSSP_RL_GPU:-1}"
JSSP_PO_GPU="${JSSP_PO_GPU:-2}"
JSSP_SLL_GPU="${JSSP_SLL_GPU:-3}"
FFSP50_RL_GPU="${FFSP50_RL_GPU:-4}"

JSSP_RESUME_MAX_EPOCHS="${JSSP_RESUME_MAX_EPOCHS:-20}"
JSSP_SLL_MAX_EPOCHS="${JSSP_SLL_MAX_EPOCHS:-20}"
FFSP50_RL_MAX_EPOCHS="${FFSP50_RL_MAX_EPOCHS:-150}"

DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  ./run_g48_resume_jssp15x15_and_start_sll_ffsp50_rl.sh

Environment overrides:
  HOST=g48
  SSH_BIN=ssh.exe
  REMOTE_REPO=/data1/gushengda/eam-rl4co
  REMOTE_PYTHON_BIN=/data1/gushengda/anaconda3/envs/rlco1/bin/python
  JSSP_RL_GPU=1
  JSSP_PO_GPU=2
  JSSP_SLL_GPU=3
  FFSP50_RL_GPU=4
  JSSP_RESUME_MAX_EPOCHS=20
  JSSP_SLL_MAX_EPOCHS=20
  FFSP50_RL_MAX_EPOCHS=150
  DRY_RUN=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "DRY_RUN=1"
fi

"${SSH_BIN}" "${HOST}" \
  "HOST=${HOST@Q} \
   REMOTE_REPO=${REMOTE_REPO@Q} \
   REMOTE_PYTHON_BIN=${REMOTE_PYTHON_BIN@Q} \
   JSSP_RL_GPU=${JSSP_RL_GPU@Q} \
   JSSP_PO_GPU=${JSSP_PO_GPU@Q} \
   JSSP_SLL_GPU=${JSSP_SLL_GPU@Q} \
   FFSP50_RL_GPU=${FFSP50_RL_GPU@Q} \
   JSSP_RESUME_MAX_EPOCHS=${JSSP_RESUME_MAX_EPOCHS@Q} \
   JSSP_SLL_MAX_EPOCHS=${JSSP_SLL_MAX_EPOCHS@Q} \
   FFSP50_RL_MAX_EPOCHS=${FFSP50_RL_MAX_EPOCHS@Q} \
   DRY_RUN=${DRY_RUN@Q} \
   bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

ROOT_DIR="${REMOTE_REPO}"
PYTHON_BIN="${REMOTE_PYTHON_BIN}"

RL_RUN_DIR="${ROOT_DIR}/logs/train/runs/mgl-jssp-rl_bucketed-multishape_15x15_20260421-050017"
PO_RUN_DIR="${ROOT_DIR}/logs/train/runs/mgl-jssp-po_bucketed-multishape_15x15_20260421-050017"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/logs/train/runs" "${ROOT_DIR}/baseline"

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+=" $(printf '%q' "${arg}")"
  done
  printf '%s\n' "${out# }"
}

start_bg() {
  local name="$1"
  local gpu_id="$2"
  local log_path="$3"
  shift 3

  local -a cmd=( "$@" )
  echo "[${name}] CUDA_VISIBLE_DEVICES=${gpu_id}"
  echo "[${name}] log=${log_path}"
  echo "[${name}] cmd=$(quote_cmd "${cmd[@]}")"

  if [[ "${DRY_RUN}" != "0" ]]; then
    echo "[${name}] dry-run only"
    return 0
  fi

  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >"${log_path}" 2>&1 < /dev/null &
  local pid=$!
  echo "[${name}] pid=${pid}"
}

ensure_checkpoint() {
  local run_dir="$1"
  local ckpt_path="${run_dir}/checkpoints/last.ckpt"
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "ERROR: missing checkpoint: ${ckpt_path}" >&2
    exit 1
  fi
}

ensure_not_running() {
  local name="$1"
  local needle="$2"
  local existing
  existing="$(pgrep -af "${needle}" || true)"
  if [[ -n "${existing}" ]]; then
    echo "[${name}] already running, skip"
    echo "${existing}"
    return 1
  fi
  return 0
}

resume_jssp_run() {
  local name="$1"
  local gpu_id="$2"
  local run_dir="$3"
  local experiment="$4"
  local logger_name="$5"
  shift 5

  ensure_checkpoint "${run_dir}"
  if ! ensure_not_running "${name}" "${run_dir}"; then
    return 0
  fi

  local ckpt_dir="${run_dir}/checkpoints"
  local ckpt_path="${ckpt_dir}/last.ckpt"
  local log_path="${ROOT_DIR}/logs/${name}_resume_${TIMESTAMP}.out"

  local -a cmd=(
    "${PYTHON_BIN}" -u run.py
    "experiment=${experiment}"
    "hydra.run.dir=${run_dir}"
    "ckpt_path=${ckpt_path}"
    "~callbacks.learning_rate_monitor"
    "~callbacks.rich_progress_bar"
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}"
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
    "callbacks.model_checkpoint.auto_insert_metric_name=False"
    "callbacks.model_checkpoint.save_top_k=1"
    "callbacks.model_checkpoint.save_last=True"
    "callbacks.model_checkpoint.every_n_epochs=1"
    "trainer.accelerator=gpu"
    "+trainer.devices=[0]"
    "trainer.max_epochs=${JSSP_RESUME_MAX_EPOCHS}"
    "+trainer.enable_progress_bar=false"
    "logger=csv"
    "logger.csv.name=${logger_name}"
    "+model.allowed_shapes=[[15,15]]"
    "+model.required_allowed_shapes=[[15,15]]"
  )
  cmd+=( "$@" )

  start_bg "${name}" "${gpu_id}" "${log_path}" "${cmd[@]}"
}

start_jssp_sll() {
  local run_dir="${ROOT_DIR}/logs/train/runs/mgl-jssp-sll_15x15_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"
  local log_path="${ROOT_DIR}/logs/mgl-jssp-sll_15x15_${TIMESTAMP}.out"

  if ! ensure_not_running "jssp15x15_sll" "logger.csv.name=mgl-jssp-sll_15x15"; then
    return 0
  fi

  local -a cmd=(
    "${PYTHON_BIN}" -u run.py
    "experiment=scheduling/mgl-jssp-sll-paper"
    "hydra.run.dir=${run_dir}"
    "~callbacks.learning_rate_monitor"
    "~callbacks.rich_progress_bar"
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}"
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
    "callbacks.model_checkpoint.auto_insert_metric_name=False"
    "callbacks.model_checkpoint.save_top_k=1"
    "callbacks.model_checkpoint.save_last=True"
    "callbacks.model_checkpoint.every_n_epochs=1"
    "trainer.accelerator=gpu"
    "+trainer.devices=[0]"
    "trainer.max_epochs=${JSSP_SLL_MAX_EPOCHS}"
    "+trainer.enable_progress_bar=false"
    "logger=csv"
    "logger.csv.name=mgl-jssp-sll_15x15"
    "+model.allowed_shapes=[[15,15]]"
    "+model.required_allowed_shapes=[[15,15]]"
  )

  start_bg "jssp15x15_sll" "${JSSP_SLL_GPU}" "${log_path}" "${cmd[@]}"
}

start_ffsp50_rl() {
  local run_dir="${ROOT_DIR}/logs/train/runs/ffsp_matnet_rl_50_${TIMESTAMP}"
  local ckpt_dir="${run_dir}/checkpoints"
  local log_path="${ROOT_DIR}/logs/ffsp_matnet_rl_50_${TIMESTAMP}.out"

  if ! ensure_not_running "ffsp50_rl" "logger.csv.name=ffsp_matnet_rl_50"; then
    return 0
  fi

  local -a cmd=(
    "${PYTHON_BIN}" -u run.py
    "experiment=scheduling/ffsp-matnet-rl-paper50"
    "hydra.run.dir=${run_dir}"
    "~callbacks.learning_rate_monitor"
    "~callbacks.rich_progress_bar"
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}"
    "callbacks.model_checkpoint.filename='epoch_{epoch:03d}'"
    "callbacks.model_checkpoint.auto_insert_metric_name=False"
    "callbacks.model_checkpoint.save_top_k=-1"
    "callbacks.model_checkpoint.save_last=True"
    "callbacks.model_checkpoint.every_n_epochs=1"
    "trainer.accelerator=gpu"
    "+trainer.devices=[0]"
    "trainer.max_epochs=${FFSP50_RL_MAX_EPOCHS}"
    "+trainer.enable_progress_bar=false"
    "logger=csv"
    "logger.csv.name=ffsp_matnet_rl_50"
  )

  start_bg "ffsp50_rl" "${FFSP50_RL_GPU}" "${log_path}" "${cmd[@]}"

  if [[ "${DRY_RUN}" != "0" ]]; then
    return 0
  fi

  (
    local train_pid
    train_pid="$(pgrep -n -f "${run_dir}")"
    local mid_src="epoch_074.ckpt"
    local mid_alias="epoch_075.ckpt"
    local final_src="epoch_149.ckpt"
    local final_alias="epoch_150.ckpt"

    while kill -0 "${train_pid}" 2>/dev/null; do
      if [[ -f "${ckpt_dir}/${mid_src}" && ! -f "${ckpt_dir}/${mid_alias}" ]]; then
        cp -f "${ckpt_dir}/${mid_src}" "${ckpt_dir}/${mid_alias}"
      fi
      if [[ -f "${ckpt_dir}/${final_src}" && ! -f "${ckpt_dir}/${final_alias}" ]]; then
        cp -f "${ckpt_dir}/${final_src}" "${ckpt_dir}/${final_alias}"
      fi
      if [[ -f "${ckpt_dir}/${mid_alias}" ]]; then
        cp -f "${ckpt_dir}/${mid_alias}" "${ROOT_DIR}/baseline/ffsp_matnet_rl_paper50_epoch_75.ckpt"
      fi
      if [[ -f "${ckpt_dir}/${final_alias}" ]]; then
        cp -f "${ckpt_dir}/${final_alias}" "${ROOT_DIR}/baseline/ffsp_matnet_rl_paper50_epoch_150.ckpt"
      fi
      if [[ -f "${ckpt_dir}/last.ckpt" ]]; then
        cp -f "${ckpt_dir}/last.ckpt" "${ROOT_DIR}/baseline/ffsp_matnet_rl_paper50_last.ckpt"
      fi
      sleep 15
    done
  ) >/dev/null 2>&1 &
}

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/PTP:${PYTHONPATH:-}"
export LOG_TZ="${LOG_TZ:-Asia/Shanghai}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export HYDRA_FULL_ERROR=1

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python bin does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/prepare_bopo_jsp_data.py >/dev/null

resume_jssp_run \
  "jssp15x15_rl" \
  "${JSSP_RL_GPU}" \
  "${RL_RUN_DIR}" \
  "scheduling/mgl-jssp-rl-bucketed-multishape" \
  "rl_15x15"

resume_jssp_run \
  "jssp15x15_po" \
  "${JSSP_PO_GPU}" \
  "${PO_RUN_DIR}" \
  "scheduling/mgl-jssp-po-bucketed-multishape" \
  "po_15x15" \
  "model.po_alpha=0.25"

start_jssp_sll
start_ffsp50_rl
REMOTE_SCRIPT
