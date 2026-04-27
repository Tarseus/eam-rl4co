#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
MINIZINC_BIN="${MINIZINC_BIN:-}"
MINIZINC_ENV_FILE="${MINIZINC_ENV_FILE:-${ROOT_DIR}/tools/minizinc_env.sh}"

fail() {
  printf '[run_classical_solver_minizinc_benchmark] ERROR: %s\n' "$*" >&2
  exit 1
}

choose_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    [[ -x "${PYTHON_BIN}" ]] || fail "PYTHON_BIN is not executable: ${PYTHON_BIN}"
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

if [[ -f "${MINIZINC_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${MINIZINC_ENV_FILE}"
fi

if [[ -z "${MINIZINC_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/tools/minizinc/bin/minizinc" ]]; then
    MINIZINC_BIN="${ROOT_DIR}/tools/minizinc/bin/minizinc"
  elif command -v minizinc >/dev/null 2>&1; then
    MINIZINC_BIN="$(command -v minizinc)"
  else
    MINIZINC_BIN=""
  fi
fi

if [[ -z "${MINIZINC_BIN}" || ! -x "${MINIZINC_BIN}" ]]; then
  if [[ "${AUTO_INSTALL:-0}" == "1" ]]; then
    "${ROOT_DIR}/scripts/install_minizinc_scheduling_solvers.sh"
    if [[ -f "${MINIZINC_ENV_FILE}" ]]; then
      # shellcheck disable=SC1090
      source "${MINIZINC_ENV_FILE}"
    fi
    if [[ -x "${ROOT_DIR}/tools/minizinc/bin/minizinc" ]]; then
      MINIZINC_BIN="${ROOT_DIR}/tools/minizinc/bin/minizinc"
    elif command -v minizinc >/dev/null 2>&1; then
      MINIZINC_BIN="$(command -v minizinc)"
    fi
  fi
fi

[[ -n "${MINIZINC_BIN}" && -x "${MINIZINC_BIN}" ]] || fail "MiniZinc executable not found. Run scripts/install_minizinc_scheduling_solvers.sh first, or set AUTO_INSTALL=1."

choose_python
exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/classical_solver_minizinc_benchmark.py" --minizinc-bin "${MINIZINC_BIN}" "$@"
