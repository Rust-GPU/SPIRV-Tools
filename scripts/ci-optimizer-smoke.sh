#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[optimizer-smoke] running fuzz smoke..."
if [[ -x "${ROOT}/scripts/fuzz-smoke.sh" ]]; then
  (cd "${ROOT}" && bash "${ROOT}/scripts/fuzz-smoke.sh")
else
  echo "[optimizer-smoke] fuzz smoke script not found; skipping"
fi

echo "[optimizer-smoke] running hyperfine benchmarks (if available)..."
if command -v hyperfine >/dev/null 2>&1; then
  if [[ -x "${ROOT}/scripts/hyperfine-opt.sh" ]]; then
    (cd "${ROOT}" && bash "${ROOT}/scripts/hyperfine-opt.sh" --runs 3 --warmup 1 || true)
  else
    echo "[optimizer-smoke] hyperfine-opt.sh not found; skipping"
  fi
else
  echo "[optimizer-smoke] hyperfine not installed; skipping benchmarks"
fi

echo "[optimizer-smoke] done"
