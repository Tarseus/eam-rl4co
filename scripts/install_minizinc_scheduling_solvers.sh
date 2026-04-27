#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
PYTHON_BIN="${PYTHON_BIN:-}"

MINIZINC_VERSION="${MINIZINC_VERSION:-2.9.5}"
MINIZINC_ARCHIVE_NAME="${MINIZINC_ARCHIVE_NAME:-MiniZincIDE-${MINIZINC_VERSION}-bundle-linux-x86_64.tgz}"
# Default URL is inferred from the standard MiniZinc GitHub release layout.
MINIZINC_URL="${MINIZINC_URL:-https://github.com/MiniZinc/MiniZincIDE/releases/download/${MINIZINC_VERSION}/${MINIZINC_ARCHIVE_NAME}}"
MINIZINC_ARCHIVE_PATH="${MINIZINC_ARCHIVE_PATH:-${TOOLS_DIR}/${MINIZINC_ARCHIVE_NAME}}"
MINIZINC_INSTALL_DIR="${MINIZINC_INSTALL_DIR:-${TOOLS_DIR}/MiniZincIDE-${MINIZINC_VERSION}-bundle-linux-x86_64}"
MINIZINC_LINK_DIR="${MINIZINC_LINK_DIR:-${TOOLS_DIR}/minizinc}"
MINIZINC_ENV_FILE="${MINIZINC_ENV_FILE:-${TOOLS_DIR}/minizinc_env.sh}"
DOWNLOAD_TIMEOUT_SEC="${DOWNLOAD_TIMEOUT_SEC:-120}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"

SCIP_INSTALL_METHOD="${SCIP_INSTALL_METHOD:-conda}"
SCIP_CONDA_CHANNEL="${SCIP_CONDA_CHANNEL:-conda-forge}"
SCIP_CONDA_PACKAGE="${SCIP_CONDA_PACKAGE:-scip}"
SCIP_CONDA_ENV="${SCIP_CONDA_ENV:-}"
REQUIRE_SCIP_IN_MINIZINC="${REQUIRE_SCIP_IN_MINIZINC:-0}"

log() {
  printf '[install_minizinc_scheduling_solvers] %s\n' "$*"
}

warn() {
  printf '[install_minizinc_scheduling_solvers] WARNING: %s\n' "$*" >&2
}

fail() {
  printf '[install_minizinc_scheduling_solvers] ERROR: %s\n' "$*" >&2
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

  fail "Could not find a Python interpreter. Set PYTHON_BIN=/path/to/python first."
}

download_with_python() {
  local target="$1"
  local primary_url="$2"
  local timeout_sec="$3"
  "${PYTHON_BIN}" - "${target}" "${primary_url}" "${timeout_sec}" <<'PY'
import pathlib
import sys
import urllib.request

target = pathlib.Path(sys.argv[1])
url = sys.argv[2]
timeout_sec = float(sys.argv[3])
fallback_url = url.replace("https://", "http://", 1) if url.startswith("https://") else url
last_exc = None
for candidate in (url, fallback_url):
    try:
        with urllib.request.urlopen(candidate, timeout=timeout_sec) as response:
            target.write_bytes(response.read())
        print(target)
        break
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
else:
    raise last_exc
PY
}

download_with_curl() {
  local target="$1"
  local url="$2"
  local timeout_sec="$3"
  local retries="$4"
  curl -L --fail --retry "${retries}" --connect-timeout "${timeout_sec}" --max-time 0 -o "${target}" "${url}"
}

download_with_wget() {
  local target="$1"
  local url="$2"
  local timeout_sec="$3"
  local retries="$4"
  wget --tries="${retries}" --timeout="${timeout_sec}" -O "${target}" "${url}"
}

download_archive() {
  local target="$1"
  local url="$2"
  local timeout_sec="$3"
  local retries="$4"
  local tmp_target="${target}.part"
  rm -f "${tmp_target}"

  if command -v curl >/dev/null 2>&1; then
    log "Downloading via curl from ${url}"
    if download_with_curl "${tmp_target}" "${url}" "${timeout_sec}" "${retries}"; then
      mv "${tmp_target}" "${target}"
      return
    fi
    warn "curl download failed for ${url}"
  fi

  if command -v wget >/dev/null 2>&1; then
    log "Downloading via wget from ${url}"
    if download_with_wget "${tmp_target}" "${url}" "${timeout_sec}" "${retries}"; then
      mv "${tmp_target}" "${target}"
      return
    fi
    warn "wget download failed for ${url}"
  fi

  log "Downloading via Python urllib from ${url}"
  if download_with_python "${tmp_target}" "${url}" "${timeout_sec}"; then
    mv "${tmp_target}" "${target}"
    return
  fi

  rm -f "${tmp_target}"
  fail \
"Failed to download MiniZinc archive from ${url}.
This machine may not be able to reach GitHub release assets.

You can recover in any of these ways:
1. Download ${MINIZINC_ARCHIVE_NAME} on another machine and upload it to:
   ${target}
2. Point MINIZINC_URL to a reachable mirror or internal artifact host.
3. If the archive is already uploaded elsewhere on this machine, set:
   MINIZINC_ARCHIVE_PATH=/abs/path/${MINIZINC_ARCHIVE_NAME}"
}

install_minizinc_bundle() {
  mkdir -p "${TOOLS_DIR}"

  if [[ ! -f "${MINIZINC_ARCHIVE_PATH}" ]]; then
    log "Downloading MiniZinc bundle ${MINIZINC_ARCHIVE_NAME}"
    download_archive "${MINIZINC_ARCHIVE_PATH}" "${MINIZINC_URL}" "${DOWNLOAD_TIMEOUT_SEC}" "${DOWNLOAD_RETRIES}"
  else
    log "Reusing existing MiniZinc archive at ${MINIZINC_ARCHIVE_PATH}"
  fi

  if [[ ! -d "${MINIZINC_INSTALL_DIR}" ]]; then
    log "Extracting MiniZinc bundle into ${MINIZINC_INSTALL_DIR}"
    mkdir -p "${MINIZINC_INSTALL_DIR}"
    tar xfz "${MINIZINC_ARCHIVE_PATH}" -C "${MINIZINC_INSTALL_DIR}" --strip-components=1
  else
    log "MiniZinc bundle already extracted at ${MINIZINC_INSTALL_DIR}"
  fi

  ln -sfn "${MINIZINC_INSTALL_DIR}" "${MINIZINC_LINK_DIR}"
  cat > "${MINIZINC_ENV_FILE}" <<EOF
export PATH="${MINIZINC_LINK_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${MINIZINC_LINK_DIR}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export QT_PLUGIN_PATH="${MINIZINC_LINK_DIR}/plugins\${QT_PLUGIN_PATH:+:\${QT_PLUGIN_PATH}}"
EOF
  chmod +x "${MINIZINC_ENV_FILE}" || true
  log "Wrote MiniZinc environment helper to ${MINIZINC_ENV_FILE}"
}

install_scip_with_conda() {
  if [[ "${SCIP_INSTALL_METHOD}" == "skip" ]]; then
    log "Skipping SCIP install because SCIP_INSTALL_METHOD=skip"
    return
  fi

  if ! command -v conda >/dev/null 2>&1 && ! command -v mamba >/dev/null 2>&1 && ! command -v micromamba >/dev/null 2>&1; then
    fail "SCIP_INSTALL_METHOD=${SCIP_INSTALL_METHOD} requires conda/mamba/micromamba on PATH."
  fi

  local installer=""
  if command -v mamba >/dev/null 2>&1; then
    installer="mamba"
  elif command -v micromamba >/dev/null 2>&1; then
    installer="micromamba"
  else
    installer="conda"
  fi

  local args=()
  if [[ -n "${SCIP_CONDA_ENV}" ]]; then
    if [[ "${installer}" == "micromamba" ]]; then
      args=(-n "${SCIP_CONDA_ENV}")
    else
      args=(-n "${SCIP_CONDA_ENV}")
    fi
    log "Installing SCIP into conda env ${SCIP_CONDA_ENV} via ${installer}"
  else
    log "Installing SCIP into the current conda/mamba environment via ${installer}"
  fi

  "${installer}" install -y "${args[@]}" -c "${SCIP_CONDA_CHANNEL}" "${SCIP_CONDA_PACKAGE}"
}

verify_minizinc_setup() {
  # shellcheck disable=SC1090
  source "${MINIZINC_ENV_FILE}"

  command -v minizinc >/dev/null 2>&1 || fail "MiniZinc executable is not on PATH after install."
  log "MiniZinc version:"
  local version_output=""
  if ! version_output="$(minizinc --version 2>&1)"; then
    printf '%s\n' "${version_output}" >&2
    if printf '%s' "${version_output}" | grep -q "GLIBC_"; then
      fail \
"MiniZinc bundle is not compatible with this machine's glibc.
Use scripts/install_minizinc_scheduling_solvers_from_source.sh instead, with locally uploaded source archives."
    fi
    fail "MiniZinc failed to start after install."
  fi
  printf '%s\n' "${version_output}"

  local solvers_output
  solvers_output="$(minizinc --solvers || true)"
  printf '%s\n' "${solvers_output}"

  if ! printf '%s' "${solvers_output}" | grep -qi "chuffed"; then
    fail "MiniZinc did not expose a Chuffed backend after install."
  fi

  if printf '%s' "${solvers_output}" | grep -qi "scip"; then
    log "MiniZinc can already see a SCIP backend."
    return
  fi

  if [[ "${REQUIRE_SCIP_IN_MINIZINC}" == "1" ]]; then
    fail "MiniZinc cannot see SCIP yet. Activate the target conda env or register SCIP with MiniZinc, then retry."
  fi

  warn "MiniZinc does not currently list SCIP. This is often because SCIP was installed into a conda env that is not active yet."
  warn "Activate that env before running the benchmark, or set REQUIRE_SCIP_IN_MINIZINC=1 to make this a hard failure."
}

main() {
  choose_python
  log "Using Python interpreter: ${PYTHON_BIN}"
  install_minizinc_bundle
  install_scip_with_conda
  verify_minizinc_setup
  log "Done."
}

main "$@"
