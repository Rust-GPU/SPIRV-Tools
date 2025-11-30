# Rust Validator Rollout and Overrides

The Rust validator is enabled by default when building with Rust support. Use the flags/env
below to roll it forward or back without rebuilding.

## CLI toggles (`spirv-val`)
- `--prefer-rust-validator`: default to the Rust validator when both are present (overrides C++ default).
- `--prefer-cpp-validator`: default to the legacy C++ validator when both are present.
- `--force-rust-validator`: force the Rust validator even when the C++ path is present.
- `--force-cpp-validator`: force the legacy C++ validator and skip the Rust path.

## Environment toggles
- `SPIRV_TOOLS_FORCE_RUST_VALIDATOR=1`: prefer Rust validation (honored by CLI/FFI).
- `SPIRV_TOOLS_DISABLE_RUST_VALIDATOR=1`: disable the Rust validator and use C++.
- `SPIRV_TOOLS_PREFER_RUST_VALIDATOR=1`: default to the Rust validator when both are built.
- `SPIRV_TOOLS_PREFER_CPP_VALIDATOR=1`: default to the C++ validator when both are built.

If both env vars are set, the disable flag wins. CLI flags override env preference.

## Rollout guidance
1. Default: leave env unset and the Rust validator will be chosen when available; if Rust support
   is not built, the CLI/FFI falls back to C++ automatically.
2. Roll forward: set `SPIRV_TOOLS_FORCE_RUST_VALIDATOR=1` (or pass
   `--force-rust-validator`) to ensure the Rust path is used.
3. Roll back: set `SPIRV_TOOLS_DISABLE_RUST_VALIDATOR=1` (or pass
   `--force-cpp-validator`) to pin to the C++ path.
4. Nudge default without forcing: use `--prefer-rust-validator` or
   `--prefer-cpp-validator` (or equivalent env overrides) to steer the default when both are built.

## Parity testing
- Run the existing corpus with the Rust path forced:
  `SPIRV_TOOLS_FORCE_RUST_VALIDATOR=1 ctest --output-on-failure` in a test-enabled build dir.
- For single binaries, `spirv-val --force-rust-validator ...` mirrors the above.
- For an end-to-end parity sweep (validator + optimizer), run `scripts/run-parity.sh`.
