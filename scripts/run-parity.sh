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

# Aggregate parity runner: validator corpus + optimizer parity + asm/dis parity.
# Usage: scripts/run-parity.sh /path/to/build-tests [workspace-root]

build_dir="${1:-build-tests}"
workspace="${2:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

echo "Running validator corpus with Rust validator..."
scripts/run-rust-validator-corpus.sh "${build_dir}"

echo "Running optimizer Rust vs. C++ parity..."
scripts/run-opt-parity.sh "${workspace}"

echo "Running assembler/disassembler Rust vs. C++ parity..."
scripts/run-asm-dis-parity.sh "${workspace}"
