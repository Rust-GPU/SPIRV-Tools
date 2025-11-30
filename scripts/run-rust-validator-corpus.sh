#!/usr/bin/env bash
set -euo pipefail

# Runs the C++ validation corpus with the Rust validator forced on.
# Usage: scripts/run-rust-validator-corpus.sh /path/to/build-tests

build_dir="${1:-.}"
if [ ! -d "${build_dir}" ]; then
  echo "error: build directory '${build_dir}' does not exist" >&2
  exit 1
fi

# Heuristic: require a CTest config to avoid running in source tree by accident.
if [ ! -f "${build_dir}/CTestTestfile.cmake" ] && [ ! -d "${build_dir}/Testing" ]; then
  echo "error: '${build_dir}' does not look like a CTest build directory" >&2
  exit 1
fi

(
  cd "${build_dir}"
  SPIRV_TOOLS_FORCE_RUST_VALIDATOR=1 ctest --output-on-failure "$@"
)
