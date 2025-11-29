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

# Smoke-test the arithmetic optimizer fuzz target with a bounded run.
# This is intentionally short so it can be run locally or in CI gates.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if ! command -v cargo-fuzz >/dev/null 2>&1; then
  echo "cargo-fuzz not installed; install with 'cargo install cargo-fuzz'." >&2
  exit 1
fi

RUSTFLAGS="-C debug-assertions=yes" \
  cargo fuzz run expr_opt --manifest-path rust/spirv-tools-opt/fuzz/Cargo.toml \
    -- -max_len=64 -runs=1000 -only_ascii=1
