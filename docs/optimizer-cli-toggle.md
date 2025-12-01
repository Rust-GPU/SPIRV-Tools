# Rust Optimizer CLI Toggles

This document describes the knobs available when running the Rust arithmetic optimizer via the `spirv-opt` CLI wrapper. The goal is to make swapping between the Rust path and the legacy C++ `spirv-opt` explicit and reversible for downstreams.

## Flags

- `--passthrough` — bypasses the Rust optimizer entirely and echoes the input module unchanged. Useful when you want CLI behavior without optimization.
- `--cpp` — invokes the C++ `spirv-opt` binary found on `PATH` (or provided via the `SPIRV_CPP_OPT` env var in tests) instead of the Rust optimizer.
- `--force-rust` — runs the Rust optimizer even when the disable env var (below) is set.

## Environment

- `SPIRV_TOOLS_DISABLE_RUST_OPT=1` — disables the Rust optimizer by default. Can be overridden with `--force-rust`.
- `SPIRV_CPP_OPT=/path/to/spirv-opt` — optional in tests to point directly at a C++ `spirv-opt` binary when `PATH` lookup is not sufficient.

## Error Reporting

Errors are surfaced with typed messages:

- Input/IO failures (`failed to read SPIR-V module: …`)
- Misaligned input (`input size must be a multiple of 4 bytes`)
- Rust optimizer failures (`optimization failed: …`)
- C++ fallback failures (`cpp spirv-opt failed with status …: …`) that include the process exit status and stderr for easier debugging.

## Recommended Rollout Pattern

1. Default to the Rust optimizer.
2. Allow operators to force C++ via `--cpp` for side-by-side benchmarking.
3. Keep `--passthrough` as a last-resort escape hatch.
4. Prefer `--force-rust` when you want the Rust path even if the global disable env is set in a larger environment.
