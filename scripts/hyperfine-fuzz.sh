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

# Hyperfine smoke benchmark for spirv-fuzz. Uses the Rust binary by default and
# includes the C++ binary when available. Skips gracefully if prerequisites are
# missing.

if ! command -v hyperfine >/dev/null 2>&1; then
  echo "hyperfine not found; skipping fuzz benchmark" >&2
  exit 0
fi

rust_fuzz=${CARGO_BIN_EXE_spirv-fuzz:-target/debug/spirv-fuzz}
rust_as=${CARGO_BIN_EXE_spirv-as:-target/debug/spirv-as}
if [[ ! -x "$rust_fuzz" || ! -x "$rust_as" ]]; then
  echo "spirv-fuzz or spirv-as binary not found; build them first" >&2
  exit 0
fi

cpp_fuzz=${SPIRV_CPP_FUZZ:-$(command -v spirv-fuzz || true)}
include_cpp=0
if [[ -n "$cpp_fuzz" && -x "$cpp_fuzz" ]]; then
  include_cpp=1
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
asm="$tmpdir/module.spvasm"
spv="$tmpdir/module.spv"
cat >"$asm" <<'ASM'
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
ASM

"$rust_as" "$asm" -o "$spv"

benchmarks=("$rust_fuzz $spv")
labels=("rust-fuzz")
if [[ $include_cpp -eq 1 ]]; then
  benchmarks+=("$cpp_fuzz $spv")
  labels+=("cpp-fuzz")
fi

cmd=(hyperfine --runs 3 --warmup 1)
for i in "${!benchmarks[@]}"; do
  cmd+=("--command-name" "${labels[$i]}" "${benchmarks[$i]}")
fi

"${cmd[@]}"
