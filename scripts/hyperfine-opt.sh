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

# Benchmark Rust spirv-opt CLI against the C++ spirv-opt when available.
# This uses a small arithmetic-heavy module assembled on the fly.

if ! command -v hyperfine >/dev/null 2>&1; then
  echo "hyperfine not found; install it to run benchmarks." >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

ASM_PATH="${TMP_DIR}/input.spvasm"
SPV_PATH="${TMP_DIR}/input.spv"
OUT_RUST="${TMP_DIR}/out-rust.spv"
OUT_CPP="${TMP_DIR}/out-cpp.spv"

cat > "${ASM_PATH}" <<'ASM'
; SPIR-V
; Version: 1.0
; Generator: custom
; Bound: 20
; Schema: 0
OpCapability Shader
OpMemoryModel Logical Simple
OpEntryPoint Fragment %func "main"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 0
%c2 = OpConstant %int 2
%c3 = OpConstant %int 3
%c4 = OpConstant %int 4
%c5 = OpConstant %int 5
%c6 = OpConstant %int 6
%func = OpFunction %void None %fn
%entry = OpLabel
%add1 = OpIAdd %int %c4 %c5
%mul1 = OpIMul %int %add1 %c6
%sub1 = OpISub %int %mul1 %c2
%add2 = OpIAdd %int %sub1 %c3
%mul2 = OpIMul %int %add2 %c4
OpReturn
OpFunctionEnd
ASM

pushd "${REPO_ROOT}" >/dev/null
cargo run -p spirv-tools-cli --bin spirv-as --release -- "${ASM_PATH}" -o "${SPV_PATH}"

RUST_CMD="cargo run -p spirv-tools-cli --bin spirv-opt --release -- ${SPV_PATH} -o ${OUT_RUST}"

COMMANDS=("${RUST_CMD}")
if command -v spirv-opt >/dev/null 2>&1; then
  COMMANDS+=("spirv-opt ${SPV_PATH} -o ${OUT_CPP}")
else
  echo "C++ spirv-opt not found in PATH; benchmarking Rust path only." >&2
fi

hyperfine "${COMMANDS[@]}"
popd >/dev/null
