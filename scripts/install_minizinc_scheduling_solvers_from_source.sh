#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
SRC_DIR="${SRC_DIR:-${TOOLS_DIR}/src}"
BUILD_ROOT="${BUILD_ROOT:-${TOOLS_DIR}/build}"

MINIZINC_SOURCE_ARCHIVE_PATH="${MINIZINC_SOURCE_ARCHIVE_PATH:-${SRC_DIR}/libminizinc-src.tar.gz}"
CHUFFED_SOURCE_ARCHIVE_PATH="${CHUFFED_SOURCE_ARCHIVE_PATH:-${SRC_DIR}/chuffed-src.tar.gz}"

MINIZINC_INSTALL_DIR="${MINIZINC_INSTALL_DIR:-${TOOLS_DIR}/minizinc_source_install}"
CHUFFED_INSTALL_DIR="${CHUFFED_INSTALL_DIR:-${TOOLS_DIR}/chuffed_source_install}"
MINIZINC_LINK_DIR="${MINIZINC_LINK_DIR:-${TOOLS_DIR}/minizinc}"
MINIZINC_ENV_FILE="${MINIZINC_ENV_FILE:-${TOOLS_DIR}/minizinc_env.sh}"

SCIP_INSTALL_METHOD="${SCIP_INSTALL_METHOD:-conda}"
SCIP_CONDA_CHANNEL="${SCIP_CONDA_CHANNEL:-conda-forge}"
SCIP_CONDA_PACKAGE="${SCIP_CONDA_PACKAGE:-scip}"
SCIP_PREFIX_PATH="${SCIP_PREFIX_PATH:-${CONDA_PREFIX:-}}"
REQUIRE_SCIP_IN_MINIZINC="${REQUIRE_SCIP_IN_MINIZINC:-0}"

BUILD_PARALLEL="${BUILD_PARALLEL:-}"

log() {
  printf '[install_minizinc_scheduling_solvers_from_source] %s\n' "$*"
}

warn() {
  printf '[install_minizinc_scheduling_solvers_from_source] WARNING: %s\n' "$*" >&2
}

fail() {
  printf '[install_minizinc_scheduling_solvers_from_source] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || fail "Required command not found: ${cmd}"
}

choose_parallelism() {
  if [[ -n "${BUILD_PARALLEL}" ]]; then
    return
  fi

  if command -v nproc >/dev/null 2>&1; then
    BUILD_PARALLEL="$(nproc)"
    return
  fi

  BUILD_PARALLEL="4"
}

configure_cmake_project() {
  local src_root="$1"
  local build_dir="$2"
  shift 2

  mkdir -p "${build_dir}"
  (
    cd "${build_dir}"
    cmake "${src_root}" "$@"
  )
}

build_cmake_project() {
  local build_dir="$1"

  if cmake --build "${build_dir}" --parallel "${BUILD_PARALLEL}"; then
    return
  fi

  cmake --build "${build_dir}" -- -j"${BUILD_PARALLEL}"
}

install_cmake_project() {
  local build_dir="$1"

  if cmake --build "${build_dir}" --target install; then
    return
  fi

  (
    cd "${build_dir}"
    make install
  )
}

extract_archive_root() {
  local archive_path="$1"
  local dest_dir="$2"

  [[ -f "${archive_path}" ]] || fail "Archive not found: ${archive_path}"
  rm -rf "${dest_dir}"
  mkdir -p "${dest_dir}"
  tar xf "${archive_path}" -C "${dest_dir}"

  local entries=()
  mapfile -t entries < <(find "${dest_dir}" -mindepth 1 -maxdepth 1 | sort)
  if [[ "${#entries[@]}" -eq 1 && -d "${entries[0]}" ]]; then
    printf '%s\n' "${entries[0]}"
    return
  fi

  printf '%s\n' "${dest_dir}"
}

patch_chuffed_cmakelists_for_legacy_cmake() {
  local src_root="$1"
  local cmakelists_path="${src_root}/CMakeLists.txt"

  [[ -f "${cmakelists_path}" ]] || fail "Chuffed CMakeLists.txt not found at ${cmakelists_path}"

  python - "${cmakelists_path}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
original = text

old_block = """install(
  TARGETS chuffed chuffed_fzn
  EXPORT chuffed-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
"""

new_block = """install(
  TARGETS chuffed chuffed_fzn
  EXPORT chuffed-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
"""

old_config_install_block = """install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/chuffed-config.cmake"
  FILES "${CMAKE_CURRENT_BINARY_DIR}/chuffed-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chuffed
)
"""

new_config_install_block = """install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/chuffed-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/chuffed-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chuffed
)
"""

if old_block in text:
    text = text.replace(old_block, new_block, 1)
elif new_block in text:
    pass
else:
    raise SystemExit(
        "Could not find the expected Chuffed install block to patch. "
        "Please inspect CMakeLists.txt layout."
    )

if old_config_install_block in text:
    text = text.replace(old_config_install_block, new_config_install_block, 1)
elif new_config_install_block in text:
    pass
else:
    raise SystemExit(
        "Could not find the expected Chuffed config install block to patch. "
        "Please inspect CMakeLists.txt layout."
    )

if text != original:
    path.write_text(text, encoding="utf-8")
PY
}

install_scip_with_conda() {
  if [[ "${SCIP_INSTALL_METHOD}" == "skip" ]]; then
    log "Skipping SCIP install because SCIP_INSTALL_METHOD=skip"
    return
  fi

  require_command conda
  [[ -n "${CONDA_PREFIX:-}" ]] || fail \
"SCIP_INSTALL_METHOD=conda expects an activated conda environment.
Activate the target env first, or set SCIP_INSTALL_METHOD=skip."

  log "Installing SCIP into the active conda environment at ${CONDA_PREFIX}"
  conda install -y -c "${SCIP_CONDA_CHANNEL}" "${SCIP_CONDA_PACKAGE}"

  if command -v scip >/dev/null 2>&1; then
    log "SCIP executable detected at $(command -v scip)"
  else
    warn "SCIP executable was not found on PATH after conda install."
  fi

  if [[ -z "${SCIP_PREFIX_PATH}" ]]; then
    SCIP_PREFIX_PATH="${CONDA_PREFIX}"
  fi
}

build_chuffed() {
  local src_root
  src_root="$(extract_archive_root "${CHUFFED_SOURCE_ARCHIVE_PATH}" "${BUILD_ROOT}/src/chuffed")"
  local build_dir="${BUILD_ROOT}/chuffed-build"

  patch_chuffed_cmakelists_for_legacy_cmake "${src_root}"
  log "Building Chuffed from source at ${src_root}"
  configure_cmake_project "${src_root}" "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${CHUFFED_INSTALL_DIR}"
  build_cmake_project "${build_dir}"
  install_cmake_project "${build_dir}"
}

build_minizinc() {
  local src_root
  src_root="$(extract_archive_root "${MINIZINC_SOURCE_ARCHIVE_PATH}" "${BUILD_ROOT}/src/libminizinc")"
  local build_dir="${BUILD_ROOT}/libminizinc-build"
  local cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${MINIZINC_INSTALL_DIR}"
  )

  if [[ -n "${SCIP_PREFIX_PATH}" ]]; then
    cmake_args+=("-DCMAKE_PREFIX_PATH=${SCIP_PREFIX_PATH}")
  fi

  log "Building MiniZinc from source at ${src_root}"
  configure_cmake_project "${src_root}" "${build_dir}" "${cmake_args[@]}"
  build_cmake_project "${build_dir}"
  install_cmake_project "${build_dir}"
}

write_chuffed_solver_config() {
  local solver_dir="${MINIZINC_INSTALL_DIR}/share/minizinc/solvers"
  mkdir -p "${solver_dir}"

  cat > "${solver_dir}/chuffed.msc" <<EOF
{
  "id": "org.chuffed.chuffed",
  "name": "Chuffed",
  "description": "Chuffed FlatZinc executable",
  "version": "source-build",
  "mznlib": "${CHUFFED_INSTALL_DIR}/share/minizinc/chuffed",
  "executable": "${CHUFFED_INSTALL_DIR}/bin/fzn-chuffed",
  "tags": ["cp", "lcg", "int"],
  "stdFlags": ["-a", "-f", "-n", "-p", "-r", "-s", "-t", "-v"],
  "supportsMzn": false,
  "supportsFzn": true,
  "needsSolns2Out": true,
  "needsMznExecutable": false,
  "needsStdlibDir": false,
  "isGUIApplication": false
}
EOF
}

write_env_file() {
  local path_prefix="${MINIZINC_INSTALL_DIR}/bin:${CHUFFED_INSTALL_DIR}/bin"
  local ld_library_prefix="${MINIZINC_INSTALL_DIR}/lib:${CHUFFED_INSTALL_DIR}/lib"
  if [[ -n "${SCIP_PREFIX_PATH}" ]]; then
    path_prefix="${path_prefix}:${SCIP_PREFIX_PATH}/bin"
    ld_library_prefix="${SCIP_PREFIX_PATH}/lib:${ld_library_prefix}"
  fi

  cat > "${MINIZINC_ENV_FILE}" <<EOF
export PATH="${path_prefix}:\${PATH}"
export LD_LIBRARY_PATH="${ld_library_prefix}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export MZN_SOLVER_PATH="${MINIZINC_INSTALL_DIR}/share/minizinc/solvers\${MZN_SOLVER_PATH:+:\${MZN_SOLVER_PATH}}"
EOF
  chmod +x "${MINIZINC_ENV_FILE}" || true
  ln -sfn "${MINIZINC_INSTALL_DIR}" "${MINIZINC_LINK_DIR}"
  log "Wrote MiniZinc environment helper to ${MINIZINC_ENV_FILE}"
}

verify_setup() {
  # shellcheck disable=SC1090
  source "${MINIZINC_ENV_FILE}"

  command -v minizinc >/dev/null 2>&1 || fail "MiniZinc executable not found after source install."
  log "MiniZinc version:"
  minizinc --version

  local solvers_output
  solvers_output="$(minizinc --solvers || true)"
  printf '%s\n' "${solvers_output}"

  if ! printf '%s' "${solvers_output}" | grep -qi "chuffed"; then
    fail "MiniZinc did not expose a Chuffed backend after source install."
  fi

  if printf '%s' "${solvers_output}" | grep -qi "scip"; then
    log "MiniZinc can see a SCIP backend."
    return
  fi

  if [[ "${REQUIRE_SCIP_IN_MINIZINC}" == "1" ]]; then
    fail "MiniZinc cannot see SCIP yet. Check SCIP_PREFIX_PATH and rebuild MiniZinc."
  fi

  warn "MiniZinc does not currently list SCIP."
  warn "If you need SCIP through MiniZinc, install SCIP first and rebuild with SCIP_PREFIX_PATH set."
}

main() {
  require_command tar
  require_command cmake
  require_command c++
  choose_parallelism
  mkdir -p "${SRC_DIR}" "${BUILD_ROOT}" "${TOOLS_DIR}"

  log "Expecting uploaded source archives at:"
  log "  MINIZINC_SOURCE_ARCHIVE_PATH=${MINIZINC_SOURCE_ARCHIVE_PATH}"
  log "  CHUFFED_SOURCE_ARCHIVE_PATH=${CHUFFED_SOURCE_ARCHIVE_PATH}"

  install_scip_with_conda
  build_chuffed
  build_minizinc
  write_chuffed_solver_config
  write_env_file
  verify_setup
  log "Done."
}

main "$@"
