# Fuzz Bridge Enablement

To exercise the C++ fuzz bridge (and wire the Rust `fuzz_module` FFI to it), the SPIRV-Tools fuzz static library must be built alongside the rest of the C++ artifacts. When a `spirv-fuzz` binary is already available, the Rust FFI will invoke it directly as a fallback before calling into the cxx bridge. This keeps parity paths usable even if the static bridge library is not present.

## Prerequisites
- `SPIRV_BUILD_FUZZER=ON` passed to CMake.
- Protobuf sources available under `external/protobuf` (matching the upstream layout expected by SPIRV-Tools).
- Fuzz headers provided by the toolchain (e.g., `FuzzedDataProvider.h`); effcee will skip fuzz targets otherwise.

## Build
```
cmake -S . -B build-rust -DSPIRV_BUILD_FUZZER=ON
cmake --build build-rust --target SPIRV-Tools-fuzz
```

On success, `build-rust/source/fuzz/libSPIRV-Tools-fuzz.a` will exist. The Rust path no longer depends on this library; the C++ fuzz bridge is disabled in favor of the Rust pipeline. You can still build the library for C++ tooling, but the Rust FFI does not require it. If you have a C++ `spirv-fuzz` binary on your `PATH` (or point `SPIRV_CPP_FUZZ` at it), the FFI will use that binary for parity checks before falling back to the Rust pipeline.
