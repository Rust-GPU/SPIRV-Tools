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

On success, `build-rust/source/fuzz/libSPIRV-Tools-fuzz.a` will exist. The Rust build auto-detects this library, links it, and defines `SPIRV_TOOLS_HAS_FUZZ_LIB` for the cxx bridge. When absent, the fuzz FFI reports `ToolError::Disabled` with a diagnostic.
