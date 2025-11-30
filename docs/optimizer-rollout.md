# Rust Optimizer Rollout and Overrides

The Rust arithmetic optimizer is enabled by default for supported passes in the Rust tooling. Use the
flags/env below to roll it forward or back without rebuilding.

## CLI toggles (`opt_block`)
- `--force-rust`: force the Rust optimizer even if disables are set.
- `--passthrough`: bypass optimization and emit the input unchanged.

## Environment toggles (FFI/CLI)
- `SPIRV_TOOLS_FORCE_RUST_OPT=1`: prefer the Rust optimizer.
- `SPIRV_TOOLS_DISABLE_RUST_OPT=1`: disable the Rust optimizer and fall back to passthrough.

If both env vars are set, `SPIRV_TOOLS_DISABLE_RUST_OPT` wins. FFI overrides
(`set_rust_optimizer_override/clear_rust_optimizer_override`) sync these env hints for child
processes/tests.

## Parity and benchmarks
- Parity: `scripts/run-opt-parity.sh` compares Rust vs. C++ `spirv-opt` on the arithmetic corpus
  (requires `spirv-opt` in `PATH` or `SPIRV_CPP_OPT`).
- Benchmarks: `scripts/hyperfine-opt.sh` runs hyperfine against Rust `opt_block` (force/passthrough)
  and C++ `spirv-opt` on a sample module.

## Rollout guidance
1. Default: leave env unset; the Rust optimizer runs for supported blocks.
2. Roll forward: set `SPIRV_TOOLS_FORCE_RUST_OPT=1` (or `--force-rust`) to ensure the Rust path is
   used where available.
3. Roll back: set `SPIRV_TOOLS_DISABLE_RUST_OPT=1` (or use `--passthrough`) to bypass Rust
   optimization; use the C++ `spirv-opt` binary if needed for full C++ coverage.

