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

# Compare the Rust optimizer against passthrough and, optionally, the C++ optimizer.
# Usage: ./rust/tools/hyperfine_opt_compare.sh <module.spv>
# Optionally set SPIRV_CPP_OPT to point to the C++ spirv-opt binary to include it.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <module.spv>" >&2
  exit 1
fi

if ! command -v hyperfine >/dev/null 2>&1; then
  echo "hyperfine is required: https://github.com/sharkdp/hyperfine" >&2
  exit 1
fi

INPUT="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building spirv-opt (release)..."
cargo build -p spirv-tools-cli --bin spirv-opt --release >/dev/null

RUST_OPT="$ROOT/target/release/spirv-opt"

CMD_RUST="$RUST_OPT $INPUT"
CMD_PASS="SPIRV_TOOLS_DISABLE_RUST_OPT=1 $RUST_OPT $INPUT"

CMDS=("$CMD_RUST" "$CMD_PASS")
NAMES=("rust-opt" "rust-opt-disable")

if [[ -n "${SPIRV_CPP_OPT:-}" ]]; then
  CMDS+=("$SPIRV_CPP_OPT -O $INPUT -o /dev/null")
  NAMES+=("cpp-opt")
fi

echo "Running hyperfine..."
hyperfine --warmup 3 --export-markdown "$ROOT/target/hyperfine-opt.md" \
  "${CMDS[@]}" --command-name "${NAMES[@]}"

echo "Results written to target/hyperfine-opt.md"
