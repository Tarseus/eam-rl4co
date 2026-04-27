#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOLVER_SITE="${ROOT_DIR}/.solver_site"
PYTHON_BIN="${PYTHON_BIN:-}"

fail() {
  printf '[run_classical_solver_benchmark] ERROR: %s\n' "$*" >&2
  exit 1
}

choose_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    [[ -x "${PYTHON_BIN}" ]] || fail "PYTHON_BIN is not executable: ${PYTHON_BIN}"
    return
  fi

  if [[ -x "/home/gsd/anaconda3/envs/rlco/bin/python" ]]; then
    PYTHON_BIN="/home/gsd/anaconda3/envs/rlco/bin/python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  fail "Could not find a Python interpreter."
}

if [[ ! -d "${SOLVER_SITE}" ]]; then
  if [[ "${AUTO_INSTALL:-0}" == "1" ]]; then
    "${ROOT_DIR}/scripts/install_classical_solvers.sh"
  else
    fail "Missing installed Python solver packages. Run scripts/install_classical_solvers.sh first, or set AUTO_INSTALL=1."
  fi
fi

choose_python
export PYTHONPATH="${SOLVER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/classical_solver_benchmark.py" "$@"
