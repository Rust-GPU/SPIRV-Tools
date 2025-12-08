#!/usr/bin/env bash
set -euo pipefail

RUST_FUZZ=${RUST_FUZZ:-${CARGO_BIN_EXE_spirv-fuzz:-}}
RUST_AS=${RUST_AS:-${CARGO_BIN_EXE_spirv-as:-}}
CPP_FUZZ=${CPP_FUZZ:-${SPIRV_CPP_FUZZ:-$(command -v spirv-fuzz || true)}}

if [[ -z "${RUST_FUZZ}" || -z "${RUST_AS}" ]]; then
  echo "rust spirv-fuzz or spirv-as not found (set RUST_FUZZ/RUST_AS or build tests); exiting"
  exit 0
fi

if [[ -z "${CPP_FUZZ}" ]]; then
  echo "CPP fuzz binary not found (set SPIRV_CPP_FUZZ or ensure spirv-fuzz on PATH); skipping parity"
  exit 0
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

corpus=(
"OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"
"OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"
"OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"
"OpCapability RayTracingKHR
OpExtension \"SPV_KHR_ray_tracing\"
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %main \"main\" %payload
%void = OpTypeVoid
%u32 = OpTypeInt 32 0
%payload_ty = OpTypeStruct %u32
%ptr_payload = OpTypePointer IncomingRayPayloadKHR %payload_ty
%fn = OpTypeFunction %void
%payload = OpVariable %ptr_payload IncomingRayPayloadKHR
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"
)

for idx in "${!corpus[@]}"; do
  asm="${TMPDIR}/case_${idx}.spvasm"
  spv="${TMPDIR}/case_${idx}.spv"
  echo "${corpus[$idx]}" > "${asm}"
  "${RUST_AS}" "${asm}" -o "${spv}"

  rust_out="${TMPDIR}/rust_${idx}.spv"
  cpp_out="${TMPDIR}/cpp_${idx}.spv"

  "${RUST_FUZZ}" "${spv}" -o "${rust_out}"
  "${CPP_FUZZ}" "${spv}" -o "${cpp_out}"

  if ! cmp -s "${rust_out}" "${cpp_out}"; then
    echo "Mismatch on corpus ${idx} (${asm})"
    exit 1
  fi
done

echo "Rust and C++ spirv-fuzz outputs match on ${#corpus[@]} cases"
