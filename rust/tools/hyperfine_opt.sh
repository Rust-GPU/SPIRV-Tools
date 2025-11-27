#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v hyperfine >/dev/null 2>&1; then
  echo "hyperfine is required: https://github.com/sharkdp/hyperfine" >&2
  exit 1
fi

hyperfine --warmup 3 \
  'cargo bench -p spirv-tools-opt --bench opt --quiet'
