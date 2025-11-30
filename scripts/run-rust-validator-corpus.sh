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
