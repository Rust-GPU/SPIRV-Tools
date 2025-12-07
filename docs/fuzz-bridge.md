# Fuzz Bridge Enablement

To exercise the C++ fuzz bridge (and wire the Rust `fuzz_module` FFI to it), the SPIRV-Tools fuzz static library must be built alongside the rest of the C++ artifacts.

## Prerequisites
- `SPIRV_BUILD_FUZZER=ON` passed to CMake.
- Protobuf sources available under `external/protobuf` (matching the upstream layout expected by SPIRV-Tools).
- Fuzz headers provided by the toolchain (e.g., `FuzzedDataProvider.h`); effcee will skip fuzz targets otherwise.

## Build
```
cmake -S . -B build-rust -DSPIRV_BUILD_FUZZER=ON
cmake --build build-rust --target SPIRV-Tools-fuzz
```

On success, `build-rust/source/fuzz/libSPIRV-Tools-fuzz.a` will exist. The Rust path no longer depends on this library; the C++ fuzz bridge is disabled in favor of the Rust pipeline. You can still build the library for C++ tooling, but the Rust FFI does not require it.
