#!/usr/bin/env bash
set -euo pipefail

# Aggregate parity runner: validator corpus + optimizer parity.
# Usage: scripts/run-parity.sh /path/to/build-tests [workspace-root]

build_dir="${1:-build-tests}"
workspace="${2:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

echo "Running validator corpus with Rust validator..."
scripts/run-rust-validator-corpus.sh "${build_dir}"

echo "Running optimizer Rust vs. C++ parity..."
scripts/run-opt-parity.sh "${workspace}"
