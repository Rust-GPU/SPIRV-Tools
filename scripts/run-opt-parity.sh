#!/usr/bin/env bash
# Copyright (c) 2025 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

# Runs the Rust-vs-C++ optimizer parity tests.
# Usage: scripts/run-opt-parity.sh [workspace-root]

workspace="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

find_cpp_opt() {
  if [[ -n "${SPIRV_CPP_OPT:-}" ]]; then
    if [[ -x "${SPIRV_CPP_OPT}" ]]; then
      printf '%s\n' "${SPIRV_CPP_OPT}"
      return 0
    fi
    echo "error: SPIRV_CPP_OPT='${SPIRV_CPP_OPT}' is not executable" >&2
    return 1
  fi
  if command -v spirv-opt >/dev/null 2>&1; then
    command -v spirv-opt
    return 0
  fi
  return 1
}

if ! cpp_opt_bin=$(find_cpp_opt); then
  echo "error: spirv-opt (C++ optimizer) not found; set SPIRV_CPP_OPT to the C++ binary" >&2
  exit 1
fi

echo "Using C++ optimizer: ${cpp_opt_bin}"
cd "${workspace}/rust"
SPIRV_CPP_OPT="${cpp_opt_bin}" cargo test -p spirv-tools-opt --test cpp_parity -- --nocapture
