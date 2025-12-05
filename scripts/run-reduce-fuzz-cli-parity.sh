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

# Compares CLI error handling between Rust and C++ for spirv-reduce and spirv-fuzz.
# Currently covers missing-input paths to keep exit codes/stdout/stderr aligned.
# Usage: scripts/run-reduce-fuzz-cli-parity.sh [workspace-root]
#
# Env overrides:
#   SPIRV_CPP_REDUCE / SPIRV_CPP_FUZZ

workspace="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

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
  echo "skip: ${fallback} not found; set ${env_var} to enable parity for this tool" >&2
  return 2
}

build_rust_bins() {
  local missing=0
  for bin in spirv-reduce spirv-fuzz; do
    if [[ ! -x "${workspace}/rust/target/debug/${bin}" ]]; then
      missing=1
      break
    fi
  done
  if [[ ${missing} -eq 1 ]]; then
    echo "Building Rust CLI binaries..."
    (cd "${workspace}/rust" && cargo build -p spirv-tools-cli --bins >/dev/null)
  fi
}

run_case() {
  local tool="$1"
  local label="$2"
  shift 2
  local args=("$@")

  local cpp_bin="${cpp_paths[${tool}]}"
  local rust_bin="${rust_paths[${tool}]}"

  local cpp_stdout cpp_stderr rust_stdout rust_stderr
  cpp_stdout="$(mktemp)"
  cpp_stderr="$(mktemp)"
  rust_stdout="$(mktemp)"
  rust_stderr="$(mktemp)"

  "${cpp_bin}" "${args[@]}" >"${cpp_stdout}" 2>"${cpp_stderr}"
  local cpp_rc=$?
  "${rust_bin}" "${args[@]}" >"${rust_stdout}" 2>"${rust_stderr}"
  local rust_rc=$?

  local failures=0
  if [[ ${cpp_rc} -ne ${rust_rc} ]]; then
    echo "[${tool}] ${label}: exit code mismatch (cpp=${cpp_rc}, rust=${rust_rc})" >&2
    failures=1
  fi
  if ! diff -u "${cpp_stdout}" "${rust_stdout}" >/dev/null; then
    echo "[${tool}] ${label}: stdout mismatch" >&2
    failures=1
  fi
  if ! diff -u "${cpp_stderr}" "${rust_stderr}" >/dev/null; then
    echo "[${tool}] ${label}: stderr mismatch" >&2
    failures=1
  fi

  rm -f "${cpp_stdout}" "${cpp_stderr}" "${rust_stdout}" "${rust_stderr}"
  return ${failures}
}

build_rust_bins

declare -A cpp_paths
declare -A rust_paths

tools=("spirv-reduce" "spirv-fuzz")
envs=("SPIRV_CPP_REDUCE" "SPIRV_CPP_FUZZ")

for i in "${!tools[@]}"; do
  tool="${tools[$i]}"
  env="${envs[$i]}"
  if ! cpp_bin=$(find_cpp_tool "${env}" "${tool}"); then
    status=$?
    if [[ ${status} -eq 2 ]]; then
      continue
    fi
    exit ${status}
  fi
  cpp_paths["${tool}"]="${cpp_bin}"
  rust_paths["${tool}"]="${workspace}/rust/target/debug/${tool}"
done

if [[ ${#cpp_paths[@]} -eq 0 ]]; then
  echo "No C++ reduce/fuzz toolchains found; skipping CLI error parity." >&2
  exit 0
fi

failures=0
missing="definitely-not-present.spv"

for tool in "${!cpp_paths[@]}"; do
  echo "Checking CLI error parity for ${tool}"
  if ! run_case "${tool}" "missing-input" "${missing}"; then
    failures=$((failures + 1))
  fi
done

if [[ ${failures} -ne 0 ]]; then
  echo "CLI error parity failures: ${failures}" >&2
  exit 1
fi

echo "CLI error parity passed for ${#cpp_paths[@]} tool(s)."
