#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the arithmetic optimizer fuzz target with a bounded run.
# This is intentionally short so it can be run locally or in CI gates.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if ! command -v cargo-fuzz >/dev/null 2>&1; then
  echo "cargo-fuzz not installed; install with 'cargo install cargo-fuzz'." >&2
  exit 1
fi

RUSTFLAGS="-C debug-assertions=yes" \
  cargo fuzz run expr_opt --manifest-path rust/spirv-tools-opt/fuzz/Cargo.toml \
    -- -max_len=64 -runs=1000 -only_ascii=1
