#!/usr/bin/env bash
# Copyright (c) 2024 The Khronos Group Inc.
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

if [[ $# -lt 1 ]]; then
  echo "usage: SPIRV_CPP_OPT=/path/to/spirv-opt $0 <module.spv>" >&2
  exit 1
fi

if [[ -z "${SPIRV_CPP_OPT:-}" ]]; then
  echo "SPIRV_CPP_OPT is not set; skipping C++ comparison." >&2
  exit 0
fi

INPUT="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v hyperfine >/dev/null 2>&1; then
  echo "hyperfine is required: https://github.com/sharkdp/hyperfine" >&2
  exit 1
fi

echo "Building Rust spirv-opt binary..."
cargo build -p spirv-tools-cli --bin spirv-opt --release >/dev/null

RUST_OPT="$ROOT/target/release/spirv-opt"
CPP_OPT="$SPIRV_CPP_OPT"

hyperfine --warmup 3 \
  "$RUST_OPT $INPUT" \
  "$CPP_OPT -O $INPUT -o /dev/null"
