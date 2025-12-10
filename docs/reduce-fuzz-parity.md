# Reducer/Fuzzer Parity Guide

This note captures how to exercise Rust vs. C++ parity for `spirv-reduce` and `spirv-fuzz` across both CLI and FFI surfaces. The goal is to keep behavior aligned while we finish porting the reducer/fuzzer stacks into Rust.

## CLI Parity

- Tests in `rust/spirv-tools-cli/tests/spirv_reduce_fuzz_cfg_lint_parity.rs` run a corpus (vertex/fragment/compute plus raygen/miss/closest-hit/callable) through the Rust CLIs and compare outputs against the C++ tools when available (`SPIRV_CPP_REDUCE`/`SPIRV_CPP_FUZZ` or PATH discovery). Extend the corpus there with deterministic modules as the Rust reducer/fuzzer ports grow.
- Use `CARGO_TARGET_DIR=/tmp/spirv-tools-target cargo test -p spirv-tools-cli --manifest-path rust/Cargo.toml spirv_reduce_fuzz_cfg_lint_parity` to run the parity suite locally.

## FFI Parity

- FFI validation/assemble corpus parity lives in `rust/spirv-tools-ffi/tests/reduce_fuzz_ffi_parity.rs` and `reduce_fuzz_message_parity.rs`. These guard that FFI assemble/validate behavior matches the core Rust paths. FFI reducer parity against the C++ bridge is covered in `rust/spirv-tools-ffi/tests/reduce_cpp_parity.rs` (vertex/fragment/compute plus raygen/miss/closest-hit/callable), skipping cleanly when the bridge is unavailable. Mirror the CLI corpus for fuzzer FFI once the bridge is wired.
- Prefer deterministic modules that do not rely on external interestingness scripts so Rust-vs-C++ output diffs are stable.

## Message Consumer Routing

- When wiring reducer/fuzzer FFI entry points, ensure message-consumer callbacks fire in the same order and with the same payloads as the C API. Add tests that install a counting consumer on the C++ bridge and on the Rust path and assert identical sequencing.

## Toggle/Docs

- Keep the reducer/fuzzer parity tests opt-in for C++ comparisons (skip when binaries are absent) so CI remains green by default.
- Document any experimental toggles (e.g., enabling the Rust reducer/fuzzer by default) near `docs/optimizer-cli-toggle.md` and `docs/objdump-cli-toggle.md` once implemented.
