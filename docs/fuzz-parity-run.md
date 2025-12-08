# Fuzz Parity: Rust vs. C++

These parity runs compare the Rust fuzz pipeline against the legacy C++ `spirv-fuzz` output. They are opt-in and skip cleanly when the C++ binary is unavailable.

## Prerequisites
- A `spirv-fuzz` binary on your `PATH` or pointed to via `SPIRV_CPP_FUZZ=/path/to/spirv-fuzz`.
- (Optional) A C++ fuzz bridge build (`SPIRV_BUILD_FUZZER=ON`) if you prefer the static lib; the Rust FFI will also drive the CLI binary directly when present.

## Running parity tests
```
cd rust
SPIRV_CPP_FUZZ=/path/to/spirv-fuzz cargo test -p spirv-tools-ffi --tests fuzz_cpp_parity
SPIRV_CPP_FUZZ=/path/to/spirv-fuzz cargo test -p spirv-tools-cli --tests fuzz_cli_cpp_parity
SPIRV_CPP_FUZZ=/path/to/spirv-fuzz cargo test -p spirv-tools-cli --tests spirv_reduce_fuzz_cfg_lint_parity

# or use the shell helper to diff outputs directly
SPIRV_CPP_FUZZ=/path/to/spirv-fuzz ./scripts/run-fuzz-parity.sh
```

The corpora include vertex/fragment/compute and a ray-generation module, plus ray interface edge cases (dangling ids, non-pointer/pointer-to-pointer payloads). When the binary is missing, the tests emit a skip message and pass.

## Notes
- The Rust `fuzz_module_with_cpp` helper now consults `SPIRV_CPP_FUZZ` or `PATH` and invokes the C++ CLI before falling back to the cxx bridge stub.
- Keep `SPIRV_CPP_FUZZ` unset to rely solely on the Rust pipeline.
