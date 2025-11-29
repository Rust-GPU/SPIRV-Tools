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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUST_OPT="${REPO_ROOT}/build-rust/tools/spirv-opt"
CPP_OPT="${SPIRV_CPP_OPT:-${REPO_ROOT}/build/tools/spirv-opt}"
ASSEMBLER="${REPO_ROOT}/build-rust/tools/spirv-as"

if [[ ! -x "${RUST_OPT}" ]]; then
  echo "[optimizer-parity] Rust spirv-opt not found at ${RUST_OPT}" >&2
  exit 1
fi

if [[ ! -x "${CPP_OPT}" ]]; then
  echo "[optimizer-parity] C++ spirv-opt not found at ${CPP_OPT} (set SPIRV_CPP_OPT)" >&2
  exit 1
fi

if [[ ! -x "${ASSEMBLER}" ]]; then
  echo "[optimizer-parity] spirv-as not found at ${ASSEMBLER}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

ASM_PATH="${TMP_DIR}/input.spvasm"
SPV_PATH="${TMP_DIR}/input.spv"
OUT_RUST="${TMP_DIR}/out-rust.spv"
OUT_CPP="${TMP_DIR}/out-cpp.spv"

cat >"${ASM_PATH}" <<'EOF'
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%int = OpTypeInt 32 1
%fnty = OpTypeFunction %void
%main = OpFunction %void None %fnty
%entry = OpLabel
%one = OpConstant %int 1
%two = OpConstant %int 2
%sum = OpIAdd %int %one %two
%mul = OpIMul %int %sum %two
OpReturn
OpFunctionEnd
EOF

"${ASSEMBLER}" "${ASM_PATH}" -o "${SPV_PATH}"

# Force the Rust optimizer path regardless of env disables.
SPIRV_TOOLS_DISABLE_RUST_OPT=0 "${RUST_OPT}" --force-rust "${SPV_PATH}" -o "${OUT_RUST}"
"${CPP_OPT}" "${SPV_PATH}" -o "${OUT_CPP}"

if ! cmp -s "${OUT_RUST}" "${OUT_CPP}"; then
  echo "[optimizer-parity] optimizer outputs differ"
  exit 2
fi

echo "[optimizer-parity] Rust and C++ optimizer outputs match"
