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

# Run hyperfine benchmarks comparing Rust opt_block vs. C++ spirv-opt on a small compute shader.
# Requires: hyperfine, spirv-as, cargo, and spirv-opt (or set SPIRV_CPP_OPT).
#
# Usage: scripts/hyperfine-opt.sh [workspace-root]

workspace="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: $1 not found in PATH" >&2
    exit 1
  fi
}

require hyperfine
require spirv-as

find_cpp_opt() {
  if [[ -n "${SPIRV_CPP_OPT:-}" ]]; then
    if [[ -x "${SPIRV_CPP_OPT}" ]]; then
      printf '%s\n' "${SPIRV_CPP_OPT}"
      return 0
    fi
    echo "warning: SPIRV_CPP_OPT set but not executable: ${SPIRV_CPP_OPT}" >&2
  fi
  if command -v spirv-opt >/dev/null 2>&1; then
    command -v spirv-opt
    return 0
  fi
  return 1
}

cpp_opt_bin=$(find_cpp_opt || true)
if [[ -z "${cpp_opt_bin}" ]]; then
  echo "warning: spirv-opt not found; C++ baseline will be skipped" >&2
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "${tmpdir}"; }
trap cleanup EXIT

spirv_text="${tmpdir}/arith.comp"
cat >"${spirv_text}" <<'EOF'
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%c0 = OpConstant %u32 0
%c1 = OpConstant %u32 1
%c2 = OpConstant %u32 2
%c4 = OpConstant %u32 4
%main = OpFunction %void None %fn
%entry = OpLabel
%add = OpIAdd %u32 %c1 %c2
%mul = OpIMul %u32 %add %c4
%sub = OpISub %u32 %mul %c1
%div = OpUDiv %u32 %sub %c2
OpReturn
OpFunctionEnd
EOF

spirv_bin="${tmpdir}/arith.spv"
spirv-as "${spirv_text}" -o "${spirv_bin}"

rust_cmd="cargo run -p spirv-tools-opt --release --bin opt_block -- --force-rust ${spirv_bin} -o /dev/null"
passthrough_cmd="cargo run -p spirv-tools-opt --release --bin opt_block -- --passthrough ${spirv_bin} -o /dev/null"

bench_cmds=(
  "rust-opt=${rust_cmd}"
  "rust-passthrough=${passthrough_cmd}"
)

if [[ -n "${cpp_opt_bin}" ]]; then
  bench_cmds+=("cpp-opt=${cpp_opt_bin} -O ${spirv_bin} -o /dev/null")
fi

(
  cd "${workspace}"
  hyperfine "${bench_cmds[@]}"
)
