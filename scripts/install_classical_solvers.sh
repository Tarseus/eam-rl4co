#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOLVER_SITE="${ROOT_DIR}/.solver_site"
TOOLS_DIR="${ROOT_DIR}/tools"
LKH_VERSION="${LKH_VERSION:-3.0.13}"
LKH_DIR="${TOOLS_DIR}/LKH-${LKH_VERSION}"
LKH_TGZ="${TOOLS_DIR}/LKH-${LKH_VERSION}.tgz"
LKH_URL="${LKH_URL:-http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-${LKH_VERSION}.tgz}"
CONCORDE_ARCHIVE_NAME="${CONCORDE_ARCHIVE_NAME:-co031219.tgz}"
CONCORDE_SRC_DIR="${TOOLS_DIR}/concorde"
CONCORDE_TGZ="${TOOLS_DIR}/${CONCORDE_ARCHIVE_NAME}"
CONCORDE_URL="${CONCORDE_URL:-http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/${CONCORDE_ARCHIVE_NAME}}"
QSOPT_DIR="${TOOLS_DIR}/qsopt"
QSOPT_BIN_URL="${QSOPT_BIN_URL:-http://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt}"
QSOPT_LIB_URL="${QSOPT_LIB_URL:-http://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.a}"
QSOPT_HDR_URL="${QSOPT_HDR_URL:-http://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.h}"
ORTOOLS_VERSION="${ORTOOLS_VERSION:-9.12.4544}"
JOB_SHOP_LIB_VERSION="${JOB_SHOP_LIB_VERSION:-1.7.0}"
PYVRP_SPEC="${PYVRP_SPEC:-pyvrp>=0.9,<1.0}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy==1.26.4}"
PYTHON_BIN="${PYTHON_BIN:-}"

log() {
  printf '[install_classical_solvers] %s\n' "$*"
}

fail() {
  printf '[install_classical_solvers] ERROR: %s\n' "$*" >&2
  exit 1
}

download_with_python() {
  local target="$1"
  local primary_url="$2"
  "${PYTHON_BIN}" - "${target}" "${primary_url}" <<'PY'
import pathlib
import sys
import urllib.request

target = pathlib.Path(sys.argv[1])
url = sys.argv[2]
fallback_url = url.replace("https://", "http://", 1) if url.startswith("https://") else url
for candidate in (url, fallback_url):
    try:
        with urllib.request.urlopen(candidate) as response:
            target.write_bytes(response.read())
        print(target)
        break
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
else:
    raise last_exc
PY
}

choose_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    [[ -x "${PYTHON_BIN}" ]] || fail "PYTHON_BIN is not executable: ${PYTHON_BIN}"
    "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1 || fail "PYTHON_BIN has no usable pip: ${PYTHON_BIN}"
    return
  fi

  if [[ -x "/home/gsd/anaconda3/envs/rlco/bin/python" ]]; then
    PYTHON_BIN="/home/gsd/anaconda3/envs/rlco/bin/python"
    return
  fi

  if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  if command -v python >/dev/null 2>&1 && python -m pip --version >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  fail "Could not find a Python interpreter with pip. Set PYTHON_BIN=/path/to/python first."
}

ensure_build_tools() {
  if command -v make >/dev/null 2>&1 && { command -v gcc >/dev/null 2>&1 || command -v cc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1; }; then
    return
  fi

  log "Missing C build tools for LKH-3."

  if [[ "${AUTO_INSTALL_BUILD_TOOLS:-0}" != "1" ]]; then
    fail "Install make + C compiler first, or rerun with AUTO_INSTALL_BUILD_TOOLS=1 on a supported distro."
  fi

  if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    log "Installing build-essential via apt-get."
    sudo apt-get update
    sudo apt-get install -y build-essential
    return
  fi

  if command -v sudo >/dev/null 2>&1 && command -v dnf >/dev/null 2>&1; then
    log "Installing gcc/gcc-c++/make via dnf."
    sudo dnf install -y gcc gcc-c++ make
    return
  fi

  if command -v sudo >/dev/null 2>&1 && command -v yum >/dev/null 2>&1; then
    log "Installing gcc/gcc-c++/make via yum."
    sudo yum install -y gcc gcc-c++ make
    return
  fi

  fail "Could not auto-install build tools on this system."
}

install_python_packages() {
  if [[ "${CLEAN_SOLVER_SITE:-1}" == "1" ]]; then
    rm -rf "${SOLVER_SITE}"
  fi
  mkdir -p "${SOLVER_SITE}"
  log "Installing Python solver packages into ${SOLVER_SITE}"
  "${PYTHON_BIN}" -m pip install \
    --upgrade \
    --target "${SOLVER_SITE}" \
    "${NUMPY_SPEC}" \
    "ortools==${ORTOOLS_VERSION}" \
    "job-shop-lib==${JOB_SHOP_LIB_VERSION}" \
    "${PYVRP_SPEC}"
}

download_lkh() {
  mkdir -p "${TOOLS_DIR}"

  if [[ -x "${LKH_DIR}/LKH" ]]; then
    log "LKH-3 already built at ${LKH_DIR}/LKH"
    return
  fi

  if [[ ! -f "${LKH_TGZ}" ]]; then
    if [[ -n "${LKH_ARCHIVE_PATH:-}" ]]; then
      log "Using local LKH archive from ${LKH_ARCHIVE_PATH}"
      cp "${LKH_ARCHIVE_PATH}" "${LKH_TGZ}"
    else
      log "Downloading LKH-${LKH_VERSION} source archive."
      download_with_python "${LKH_TGZ}" "${LKH_URL}"
    fi
  fi

  if [[ ! -d "${LKH_DIR}" ]]; then
    log "Extracting LKH-${LKH_VERSION}."
    tar xvfz "${LKH_TGZ}" -C "${TOOLS_DIR}" >/dev/null
  fi
}

build_lkh() {
  if [[ "${SKIP_LKH:-0}" == "1" ]]; then
    log "Skipping LKH build because SKIP_LKH=1"
    return
  fi
  ensure_build_tools
  download_lkh
  log "Building LKH-3 in ${LKH_DIR}"
  make -C "${LKH_DIR}"
  [[ -x "${LKH_DIR}/LKH" ]] || fail "LKH build completed without producing ${LKH_DIR}/LKH"
}

download_qsopt() {
  if [[ -f "${QSOPT_DIR}/qsopt.a" && -f "${QSOPT_DIR}/qsopt.h" ]]; then
    return
  fi

  mkdir -p "${QSOPT_DIR}"
  log "Downloading QSopt artifacts into ${QSOPT_DIR}"
  download_with_python "${QSOPT_DIR}/qsopt" "${QSOPT_BIN_URL}"
  download_with_python "${QSOPT_DIR}/qsopt.a" "${QSOPT_LIB_URL}"
  download_with_python "${QSOPT_DIR}/qsopt.h" "${QSOPT_HDR_URL}"
  chmod +x "${QSOPT_DIR}/qsopt" || true
}

download_concorde_source() {
  mkdir -p "${TOOLS_DIR}"

  if [[ -x "${CONCORDE_SRC_DIR}/TSP/concorde" ]]; then
    log "Concorde already built at ${CONCORDE_SRC_DIR}/TSP/concorde"
    return
  fi

  if [[ ! -f "${CONCORDE_TGZ}" ]]; then
    if [[ -n "${CONCORDE_ARCHIVE_PATH:-}" ]]; then
      log "Using local Concorde archive from ${CONCORDE_ARCHIVE_PATH}"
      cp "${CONCORDE_ARCHIVE_PATH}" "${CONCORDE_TGZ}"
    else
      log "Downloading Concorde source archive."
      download_with_python "${CONCORDE_TGZ}" "${CONCORDE_URL}"
    fi
  fi

  if [[ ! -d "${CONCORDE_SRC_DIR}" ]]; then
    log "Extracting Concorde source."
    tar xvfz "${CONCORDE_TGZ}" -C "${TOOLS_DIR}" >/dev/null
  fi

  [[ -f "${CONCORDE_SRC_DIR}/configure" ]] || fail "Expected Concorde source at ${CONCORDE_SRC_DIR}"
}

build_concorde() {
  if [[ "${SKIP_CONCORDE:-0}" == "1" ]]; then
    log "Skipping Concorde build because SKIP_CONCORDE=1"
    return
  fi

  ensure_build_tools
  download_qsopt
  download_concorde_source
  log "Building Concorde against QSopt from ${QSOPT_DIR}"
  (
    cd "${CONCORDE_SRC_DIR}"
    if [[ ! -f Makefile ]]; then
      CFLAGS="${CONCORDE_CFLAGS:--g -O3 -fPIC}" ./configure --with-qsopt="${QSOPT_DIR}"
    fi
    make
  )
  [[ -x "${CONCORDE_SRC_DIR}/TSP/concorde" ]] || fail "Concorde build completed without producing ${CONCORDE_SRC_DIR}/TSP/concorde"
}

verify_install() {
  log "Verifying Python package imports."
  PYTHONPATH="${SOLVER_SITE}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["ortools", "job_shop_lib", "pyvrp"]
for mod in mods:
    imported = importlib.import_module(mod)
    print(f"{mod}: OK {getattr(imported, '__version__', 'unknown')}")
PY

  if [[ "${SKIP_LKH:-0}" == "1" ]]; then
    log "Skipped LKH verification because SKIP_LKH=1"
  else
    [[ -x "${LKH_DIR}/LKH" ]] || fail "Missing built LKH executable after install."
    log "Verified LKH executable at ${LKH_DIR}/LKH"
  fi

  if [[ "${SKIP_CONCORDE:-0}" == "1" ]]; then
    log "Skipped Concorde verification because SKIP_CONCORDE=1"
  else
    [[ -x "${CONCORDE_SRC_DIR}/TSP/concorde" ]] || fail "Missing built Concorde executable after install."
    [[ -x "${QSOPT_DIR}/qsopt" ]] || fail "Missing downloaded QSopt solver after install."
    log "Verified Concorde executable at ${CONCORDE_SRC_DIR}/TSP/concorde"
  fi
}

main() {
  choose_python
  log "Using Python interpreter: ${PYTHON_BIN}"
  install_python_packages
  build_lkh
  build_concorde
  verify_install
  log "Done."
}

main "$@"
