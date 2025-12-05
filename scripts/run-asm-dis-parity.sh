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

# Runs Rust-vs-C++ assembler/disassembler parity over a corpus of .spvasm files.
# Usage: scripts/run-asm-dis-parity.sh [--include-errors] [workspace-root] [corpus-root]
#
# Set ASM_DIS_INCLUDE_ERRORS=1 (or pass --include-errors) to also compare exit
# codes/stderr for error-only .spvasm files under test/asm_dis_error_corpus.
#
# Requires the C++ tools `spirv-as` and `spirv-dis` (or env SPIRV_CPP_AS/SPIRV_CPP_DIS).
# Rust binaries are built on demand from `spirv-tools-cli`.

include_errors=${ASM_DIS_INCLUDE_ERRORS:-0}
if [[ "${1:-}" == "--include-errors" ]]; then
  include_errors=1
  shift
fi

workspace="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
corpus="${2:-${workspace}/test}"
error_corpus="${ASM_DIS_ERROR_CORPUS:-${workspace}/test/asm_dis_error_corpus}"

find_cpp_tool() {
  local env_var="$1"
  local fallback="$2"
  if [[ -n "${!env_var:-}" ]]; then
    if [[ -x "${!env_var}" ]]; then
      printf '%s\n' "${!env_var}"
      return 0
    fi
    echo "error: ${env_var}='${!env_var}' is not executable" >&2
    return 1
  fi
  if command -v "${fallback}" >/dev/null 2>&1; then
    command -v "${fallback}"
    return 0
  fi
  echo "error: ${fallback} not found; set ${env_var}" >&2
  return 1
}

build_rust_bins() {
  if [[ ! -x "${workspace}/rust/target/debug/spirv-as" ]] || [[ ! -x "${workspace}/rust/target/debug/spirv-dis" ]]; then
    echo "Building Rust CLI binaries..."
    (cd "${workspace}/rust" && cargo build -p spirv-tools-cli --bins >/dev/null)
  fi
}

run_error_corpus() {
  local dir="$1"

  if [[ ! -d "${dir}" ]]; then
    echo "warning: error corpus '${dir}' not found; skipping"
    return
  fi

  local files=()
  while IFS= read -r -d '' file; do
    files+=("$file")
  done < <(find "${dir}" -name '*.spvasm' -print0)

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "warning: no error .spvasm files found under '${dir}'"
    return
  fi

  local start_failures=${failures}
  echo "Processing ${#files[@]} error .spvasm files under ${dir}"

  for asm in "${files[@]}"; do
    rust_err="${tmpdir}/$(basename "${asm}").rust.err"
    cpp_err="${tmpdir}/$(basename "${asm}").cpp.err"

    set +e
    "${rust_as}" "${asm}" -o /dev/null >"${tmpdir}/rust.out" 2>"${rust_err}"
    rust_status=$?
    "${cpp_as}" "${asm}" -o /dev/null >"${tmpdir}/cpp.out" 2>"${cpp_err}"
    cpp_status=$?
    set -e

    if [[ ${rust_status} -eq 0 ]]; then
      echo "Rust spirv-as unexpectedly succeeded on ${asm}" >&2
      failures=$((failures + 1))
      continue
    fi
    if [[ ${cpp_status} -eq 0 ]]; then
      echo "C++ spirv-as unexpectedly succeeded on ${asm}" >&2
      failures=$((failures + 1))
      continue
    fi

    if [[ ${rust_status} -ne ${cpp_status} ]]; then
      echo "Exit status mismatch for ${asm}: rust=${rust_status}, cpp=${cpp_status}" >&2
      failures=$((failures + 1))
      continue
    fi

    if ! diff -u "${cpp_err}" "${rust_err}" >/dev/null 2>&1; then
      echo "stderr mismatch for ${asm}" >&2
      diff -u "${cpp_err}" "${rust_err}" >&2 || true
      failures=$((failures + 1))
    fi
  done

  if [[ ${failures} -eq ${start_failures} ]]; then
    echo "Assembler/disassembler error parity passed for ${#files[@]} file(s)."
  fi
}

if [[ ! -d "${corpus}" ]]; then
  echo "error: corpus root '${corpus}' not found" >&2
  exit 1
fi

cpp_as=$(find_cpp_tool "SPIRV_CPP_AS" "spirv-as")
cpp_dis=$(find_cpp_tool "SPIRV_CPP_DIS" "spirv-dis")
build_rust_bins
rust_as="${workspace}/rust/target/debug/spirv-as"
rust_dis="${workspace}/rust/target/debug/spirv-dis"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

files=()
while IFS= read -r -d '' file; do
  files+=("$file")
done < <(find "${corpus}" -path "${error_corpus}" -prune -o -name '*.spvasm' -print0)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "warning: no .spvasm files found under '${corpus}'"
fi

echo "Using C++ assembler: ${cpp_as}"
echo "Using C++ disassembler: ${cpp_dis}"
echo "Processing ${#files[@]} .spvasm files under ${corpus}"

failures=0
for asm in "${files[@]}"; do
  base="$(basename "${asm}")"
  rust_bin="${tmpdir}/${base}.rust.spv"
  cpp_bin="${tmpdir}/${base}.cpp.spv"

  if ! "${rust_as}" "${asm}" -o "${rust_bin}" >/dev/null 2>&1; then
    echo "Rust spirv-as failed on ${asm}" >&2
    failures=$((failures + 1))
    continue
  fi
  if ! "${cpp_as}" "${asm}" -o "${cpp_bin}" >/dev/null 2>&1; then
    echo "C++ spirv-as failed on ${asm}" >&2
    failures=$((failures + 1))
    continue
  fi
  if ! cmp -s "${rust_bin}" "${cpp_bin}"; then
    echo "Assembler output mismatch for ${asm}" >&2
    failures=$((failures + 1))
    continue
  fi

  rust_text="${tmpdir}/${base}.rust.txt"
  cpp_text="${tmpdir}/${base}.cpp.txt"
  rust_text_bin="${tmpdir}/${base}.rust.txt.spv"
  cpp_text_bin="${tmpdir}/${base}.cpp.txt.spv"

  if ! "${rust_dis}" "${cpp_bin}" >"${rust_text}" 2>/dev/null; then
    echo "Rust spirv-dis failed on ${asm}" >&2
    failures=$((failures + 1))
    continue
  fi
  if ! "${cpp_dis}" "${cpp_bin}" >"${cpp_text}" 2>/dev/null; then
    echo "C++ spirv-dis failed on ${asm}" >&2
    failures=$((failures + 1))
    continue
  fi

  if ! "${rust_as}" "-" -o "${rust_text_bin}" <"${rust_text}" >/dev/null 2>&1; then
    echo "Rust spirv-as failed reassembling rust disassembly for ${asm}" >&2
    failures=$((failures + 1))
    continue
  fi
  if ! "${rust_as}" "-" -o "${cpp_text_bin}" <"${cpp_text}" >/dev/null 2>&1; then
    echo "Rust spirv-as failed reassembling cpp disassembly for ${asm}" >&2
    failures=$((failures + 1))
    continue
  fi

  if ! cmp -s "${rust_text_bin}" "${cpp_text_bin}"; then
    echo "Disassembly round-trip mismatch for ${asm}" >&2
    failures=$((failures + 1))
    continue
  fi
done

if [[ ${include_errors} -eq 1 ]]; then
  run_error_corpus "${error_corpus}"
fi

if [[ ${failures} -ne 0 ]]; then
  echo "Parity failures: ${failures} file(s) differed." >&2
  exit 1
fi

echo "Assembler/disassembler parity passed for ${#files[@]} file(s)."
