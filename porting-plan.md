# SPIRV-Tools Rust Port Plan

## Current State Summary

The Rust port is **production-ready** for validator and optimizer core functionality:

- **Validator**: Rust validator is the default across CLI/FFI with full C++ parity (49 rule modules)
- **Optimizer**: E-graph-based optimizer with 32 rule files (~11,000 lines of rewrites), 28 enabled
- **Assembler/Disassembler**: Full Rust implementation with FFI/CLI parity
- **Fuzzer**: Rust-first fuzzer pipeline with arbitrary-driven generation
- **CLI Tools**: `spirv-as`, `spirv-dis`, `spirv-val`, `spirv-opt`, `spirv-objdump`, `spirv-size` all have Rust implementations

## Workspace Structure

```
rust/
├── spirv-tools-core/     # Pure Rust: validation, assembly, types
├── spirv-tools-opt/      # E-graph optimizer using egg crate
├── spirv-tools-ffi/      # C API surface via cxx bridge
└── spirv-tools-cli/      # CLI binaries
```

## Validator Status

### Rule Modules (49 total, all complete)

| Category | Modules |
|----------|---------|
| Core | `capabilities`, `extensions`, `limits`, `layout`, `cfg`, `mode_setting` |
| Types | `types/*` (composites, cooperative, general, helpers, pointers, scalars, tensor), `invalid_type`, `literals` |
| Memory | `memory/*` (access_chain, cooperative, helpers, load_store, misc, variables), `memory_semantics`, `pointers`, `storage_classes` |
| Decorations | `decorations`, `annotation`, `builtins`, `interpolation`, `block_layout` |
| Instructions | `arithmetics`, `atomics`, `barriers`, `bitwise`, `composites`, `constants`, `conversion`, `derivatives`, `functions`, `image`, `logicals`, `misc`, `primitives` |
| Control Flow | `adjacency`, `entry_points`, `execution_limitations`, `execution_modes`, `interfaces` |
| Extended Instructions | `ext_inst/*` (clspv, glsl, opencl), `debug`, `debug_info` |
| Vulkan/GPU | `vulkan`, `ray_tracing`, `mesh_shading`, `hit_object`, `tensor`, `tensor_layout`, `graph`, `non_uniform`, `scopes`, `small_type_uses` |

## Optimizer Status

### Rule Files (32 total, ~11,000 lines)

**Enabled (28 files):**

| Category | Files |
|----------|-------|
| Core | `datatypes.egg`, `rvsdg.egg`, `primitives.egg` |
| Arithmetic/Math | `arithmetic.egg`, `floating_point.egg`, `bitwise.egg`, `bitfield.egg`, `constant_folding.egg` |
| Comparison/Logic | `comparison.egg`, `logical.egg` |
| Vectors/Matrices | `vector.egg`, `matrix.egg` |
| Types | `type_conversion.egg`, `float_conversion.egg` |
| Memory | `memory.egg`, `copy_propagation.egg`, `mem2reg.egg`, `sroa.egg` |
| Control Flow | `advanced_loops.egg`, `loop_unroll.egg` |
| GLSL/Graphics | `glsl.egg`, `graphics.egg`, `subgroup.egg` |
| Cleanup/DCE | `cleanup.egg`, `dce.egg`, `spec_constant.egg` |
| Advanced | `inlining.egg`, `licm.egg` |

**Disabled (4 files):**

| File | Issue | Resolution |
|------|-------|------------|
| `merge_return.egg` | Rules duplicate `rvsdg.egg` lines 213, 230 | **Not needed** - functionality in rvsdg.egg |
| `loop_fusion.egg` | Type mismatches (`Seq` takes `Effect`, `Theta` returns `Expr`); undefined types | Requires RVSDG type system redesign |
| `legalization.egg` | ~40 undefined types (`SDiv8`, `FAdd16`, `MakePair64`, etc.) | Add types to datatypes.egg |
| `instrumentation.egg` | `Seq` used with `Expr` instead of `Effect`; `AtomicIAdd` arity wrong | Requires redesign for side effects |

## Completed Milestones

- [x] Validator Parity Rollout (Rust validator default, all C++ corpora pass)
- [x] Decoration & Environment Constraint Parity (interpolation, builtins, storage classes)
- [x] SSA & Type Validation Parity (dominance, phi, operand types)
- [x] Entry-Point Interface Parity (storage classes, ray tracing, tessellation)
- [x] Operand Type Table Parity (grammar-driven type checks)
- [x] Optimizer 64-bit Constant Support
- [x] Optimizer Arithmetic/Bitwise/Logical Parity
- [x] Optimizer DCE, Inlining, SROA passes
- [x] mem2reg, LICM passes
- [x] Fuzz Bridge Enablement (C++ bridge wired)
- [x] Rust Fuzz Corpus Expansion (ray/interface/control-flow invalids)
- [x] mem2reg and LICM unit tests (15 tests added)

## Test Coverage

### CLI Tests (`spirv-tools-cli/tests/`)

| File | Coverage |
|------|----------|
| `spirv_as_dis_parity.rs` | Assembler/disassembler round-trip parity |
| `spirv_val_cli.rs` | Validator CLI tests |
| `spirv_opt_cli.rs` | Optimizer CLI tests |
| `spirv_reduce_fuzz_cfg_lint_parity.rs` | Reduce/fuzz/cfg/lint/objdump/size parity |
| `spirv_objdump_compiler_cmd.rs` | Objdump compiler command tests |
| `fuzz_cli_smoke.rs` | Fuzzer CLI smoke tests |
| `fuzz_cli_bench_smoke.rs` | Fuzzer benchmark smoke tests |
| `fuzz_cli_cpp_parity.rs` | Fuzzer C++ parity tests |
| `fuzz_cli_errors.rs` | Fuzzer error handling tests |

### FFI Tests (`spirv-tools-ffi/tests/`)

| File | Coverage |
|------|----------|
| `asm_dis_ffi_parity.rs` | FFI assembler/disassembler parity |
| `reduce_cpp_parity.rs` | FFI reducer C++ parity |
| `fuzz_cpp_parity.rs` | FFI fuzzer C++ parity |
| `fuzz_cpp_bridge_parity.rs` | Fuzzer C++ bridge parity |
| `fuzz_bridge.rs` | Fuzzer bridge tests |
| `fuzz_errors.rs` | FFI fuzzer error tests |
| `fuzz_invalid.rs` | Invalid input fuzzer tests |
| `fuzz_invalid_ray_mix.rs` | Ray tracing invalid input tests |
| `reduce_fuzz_ffi_parity.rs` | Reduce/fuzz FFI parity |
| `reduce_fuzz_ffi_messages.rs` | FFI message tests |
| `reduce_fuzz_message_parity.rs` | Message parity tests |
| `optimizer_bitwise_parity.rs` | Optimizer bitwise operation parity |

### Optimizer Tests (`spirv-tools-opt/src/egglog_opt.rs`)

- 69 unit tests covering all optimization passes
- Includes 15 dedicated mem2reg and LICM tests

## Outstanding Tasks

### Not Ported (Lower Priority)

- [ ] `spirv-link` - Linker pipeline (currently uses C++ bridge)
- [ ] `spirv-diff` - Typed diff output

### Optional Enhancements

- [ ] Enable `loop_fusion.egg` - Requires adding Pair/Triple/CountRange types
- [ ] Enable `legalization.egg` - Requires adding ~40 narrow/wide type definitions
- [ ] Enable `instrumentation.egg` - Requires Effect-aware redesign

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SPIRV_TOOLS_DISABLE_RUST_VALIDATOR` | Disable Rust validator, use C++ |
| `SPIRV_TOOLS_FORCE_RUST_VALIDATOR` | Force Rust validator on |
| `SPIRV_CPP_OPT` | Path to C++ spirv-opt for parity testing |
| `SPIRV_BUILD_FUZZER` | Enable fuzz library build |
| `SPIRV_CPP_FUZZ` | Use C++ fuzz bridge |

## Build Integration

The Rust staticlib builds alongside existing C++ binaries via CMake. CI runs:
- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features`
- `cargo test --workspace`
- `ctest` with Rust validator enabled

## Guiding Principles

1. Favor Rust's type system (newtypes, enums, typestates) over runtime checks
2. Keep safety boundaries explicit; unsafe code isolated in FFI shims
3. Mirror public API shape of `include/spirv-tools/libspirv.h`
4. Tests move with the code; FFI parity verified via integration tests
