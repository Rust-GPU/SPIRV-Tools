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
