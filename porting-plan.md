# SPIRV-Tools Rust Port Plan

## Goals
- Reimplement SPIRV-Tools functionality in idiomatic Rust with exhaustive type safety (newtypes, enums, traits) so invalid states are unrepresentable.
- Maintain drop-in compatibility with the existing C API by exposing an `extern "C"` surface that mirrors `libspirv` so the Rust code can be swapped in incrementally.
- Keep the port incremental: move the smallest self-contained pieces at a time, land them behind the FFI adapter, and validate behavior with the existing test suite plus new Rust-specific tests.
- Ensure every Rust addition passes `cargo fmt` and `cargo clippy --all-targets --all-features` and integrates with existing CI expectations.

## Guiding Principles
1. Favor Rust's type system (newtypes, phantom data, `NonZeroU32`, `BitFlags`, typestates) over runtime checks.
2. Keep safety boundaries explicit. Unsafe code is isolated inside `spirv-tools-ffi` shims with documented invariants.
3. Mirror the public API shape of `include/spirv-tools/libspirv.h`. When the Rust implementation is incomplete, delegate to existing C++ implementations via FFI to avoid regressions.
4. Keep changes minimal and focused. Update this plan whenever scope, sequencing, or assumptions change.
5. Tests move with the code. Rust modules own their unit tests; FFI parity is verified with integration tests mirroring the original C tests.

## High-Level Phases
1. **Foundation / Workspace**
    - Introduce a Cargo workspace under `rust/` with crates:
      - `spirv-tools-ffi`: low-level FFI surface powered by the `cxx` crate so we can bind seamlessly with the existing C++ build.
     - `spirv-tools-core`: pure-Rust logic, zero `unsafe`, high-level API.
     - (Future) feature crates for optimizer, reducer, fuzzing, etc.
   - Provide build glue (CMake/Bazel) to compile the Rust staticlib alongside existing binaries.
2. **Core Data Model Port**
   - Port enums (`spv_result_t`, operand types, message severity, target environments, etc.) as Rust enums/newtypes with `repr(transparent)` and conversion traits.
   - Implement fundamental structs (diagnostics, context, target env, message consumer) with safe builders and invariants enforced via typestates.
3. **Binary/Text Infrastructure**
   - Port binary parser/assembler data structures (instruction representation, operand tables) while keeping algorithmic parity.
   - Provide safe APIs for assembling/disassembling modules.
4. **Validator/Optimizer/Tools**
   - Incrementally port validator passes, optimizer transformations, reducer/fuzzer infrastructure, prioritizing components with minimal external dependencies first.
5. **CLIs & Integration**
   - Wire Rust implementations into existing CLI tools through the shared C API.


## Completed Milestone: Validator Parity Rollout
Force the Rust validator through CLI/FFI, close remaining ordering/decorations gaps, and prepare to flip it on by default.

Tasks for this milestone:
- [x] Generate capability/extension ordering tables (including conditional variants) from the grammar and enforce them relative to debug/names/annotations/functions so layout parity matches C++. (mode-stage ordering is generated and enforced regardless of layout skipping)
 - [x] Extend decoration target/category constraints still present in the C++ tables (interpolation/barycentric/BuiltIn env quirks) with text/binary regressions. (Interpolation storage-class and fragment execution-model requirements enforced; interpolation decorations now enforce exclusivity within base and centroid/sample/patch classes; Sample decoration requires SampleRateShading capability; fragment-only BuiltIns now require a fragment entry point; barycentric BuiltIns now restricted to fragment entry points and Input storage; ray/mesh/shading-rate built-ins now require appropriate execution models; shading rate built-ins now enforce Input/Output storage by kind; mesh built-ins enforced to Output storage; BuiltIn pointee type checks added for fragment/tessellation/shading-rate/ray targets; compute invocation/workgroup BuiltIns now require compute entry points; ViewIndex requires MultiView and compute is disallowed; DeviceIndex requires DeviceGroup+extension; SM/ARM core built-ins now require ShaderSMBuiltinsNV/CoreBuiltinsARM and CoreBuiltinsARM requires its extension; subgroup id/size/local-invocation/ballot/num-enqueued built-ins now gate on GroupNonUniform/DeviceEnqueue and Kernel execution as appropriate.)
 - [x] Thread extension allowlists through conditional extensions/capabilities during full validation (not just layout) and add env-gated regressions (e.g., WebGPU/Vulkan splits). **(function-body extensions now rejected even when layout is skipped; allowlists cover conditional extensions)**
 - [x] Keep validated-module caching in place while the new ordering/decorations checks are added, ensuring CLI/FFI reuse validated words without re-parse.
 - [x] Run the full C++ validation corpora with the Rust path forced on (`SPIRV_TOOLS_DISABLE_RUST_VALIDATOR=0`) and fix mismatches until the Rust path can be the default without a flag. (Test-enabled CMake build `build-tests` runs all 24 ctests clean with the Rust validator enabled.)

## Completed Milestone: Rust Validator Default Rollout
Flip the Rust validator on by default across FFI/CLI and lock in env-specific BuiltIn allowlists.

Planned tasks:
- [x] Finish remaining env-specific BuiltIn allowlists/quirks (any lingering Vulkan/OpenCL/vendor shading-rate/mesh cases) with Rust regressions.
- [x] Add a rollout flag/toggle to make the Rust validator the default in CLI/FFI, with a clear fallback to C++.
- [x] Honor `SPIRV_TOOLS_FORCE_RUST_VALIDATOR` in the CLI path so downstreams can flip the Rust validator on without extra flags while keeping a C++ fallback.
- [x] Keep Vulkan rejecting legacy OpenGL-style built-ins (`VertexId`, `InstanceId`) with Rust-side regressions.
- [x] Gate fragment shading rate built-ins to Vulkan-only via Rust validation with regression coverage.
- [x] Gate mesh built-ins to Vulkan-only (extension-gated) with Rust regression coverage.
- [x] Provide a corpus runner (`scripts/run-rust-validator-corpus.sh`) that forces the Rust validator for CI/locals; wire into CI to guard parity.
- [x] Document rollout/rollback procedures and env overrides for downstream consumers.
- [x] Keep the corpus test job (`ctest` with `SPIRV_TOOLS_DISABLE_RUST_VALIDATOR=0`) in CI to guard parity (script available; wire into CI).

## Completed Milestone: Decoration & Env Constraint Parity
Close the remaining decoration/environment-specific gaps against the C++ validator tables.

Tasks for this milestone:
- [x] Enforce interpolation/barycentric decoration storage-class and execution-model constraints with grammar-driven tables and regressions.
- [x] Add env-specific BuiltIn allowlists (e.g., Mesh/Fragment shading built-ins, SM built-ins) with typed errors and binary tests. (Stage-specific allowlists and BuiltIn type checks added; compute-only built-ins gated to GLCompute/Kernel; kernel-only built-ins gated to Kernel and require Kernel capability; subgroup built-ins require GroupNonUniform (mask built-ins also require GroupNonUniformBallot/SubgroupBallotKHR); subgroup id/size/local-invocation regressions added; kernel enqueue counters now require DeviceEnqueue and Kernel execution model; SubgroupMaxSize now requires Kernel capability and Kernel execution model; SM/ARM GPU core built-ins now gate on ShaderSMBuiltinsNV/CoreBuiltinsARM and require their vendor extensions; fragment shading rate and mesh built-ins now Vulkan-only; legacy GL-style built-ins rejected in Vulkan.)
- [x] Validate decoration exclusivity/compatibility pairs (e.g., interpolation vs Sample/Flat mixes, sample-rate shading requirements) against the C++ suite and port any missing tests. (Interpolation exclusivity and Flat+Sample/Centroid conflicts covered; Sample requires SampleRateShading.)
- [x] Wire these constraints through FFI/CLI and keep validated-module caching active; run C++ corpora under the Rust path to confirm parity.

## Active Milestone: Optimizer Block Folding + FFI
Port the arithmetic optimizer to Rust with e-graph-driven rewrites, expose it through the FFI, and validate with Rust-side unit tests plus fuzzing/benchmarks. (Paused per rollout order; resume after validator parity lands.)

Tasks for this milestone:
- [x] Translate arithmetic ops (`OpConstant`, `OpIAdd`, `OpIMul`, `OpISub`, `OpSNegate`, `OpSDiv`, `OpUDiv`, `OpSRem`, `OpUMod`) into the Rust optimizer and fold trivially solvable expressions.
- [x] Expose a basic-block optimizer over FFI that returns reassembled SPIR-V words, preserving non-arithmetic instructions.
- [x] Add Rust tests mirroring the C++ optimizer expectations for pass-through and simple constant folding.
- [x] Add `cargo fuzz` coverage for SPIR-V block optimization to catch translation/rewriter panics (fuzz targets run without artificial e-graph limits).
- [x] Expand e-graph rewrites with algebraic cancellations and emit simplified SPIR-V blocks (while preserving result ids and stability when no cost improvement occurs).
- [x] Add criterion coverage for block-level arithmetic optimization (including small and medium block cases).
- [ ] Expand optimizer coverage with e-graph driven rewrites (egg) for algebraic simplifications beyond simple folds. (constant-offset hoisting for add/add and add/sub landed; constant-factor hoisting for mul chains; divisor merging for nested div chains; common-factor extraction for const-scaled sums and symbolic add/sub products; cancellation of common constant factors in div/rem and sub factoring; constant-flip for negated const multiplies; negation normalization across mul/sub/double-neg; pulling divisible constants out of mul/div chains; distributing constants over add/sub to expose folds; refactoring affine sums when constants share divisors; folding repeated adds into multiplies/shifts (x+x, x+x+x, x*4 via shift); merging nested const shifts; eliminating shift-by-zero forms; signed/unsigned power-of-two div strength reduction (unsigned shift; signed biased shift with bitmask); decomposing signed/unsigned remainder into div/mul/sub for const divisors; bitwise and folding/identity rules and mask-to-mod for pow2 masks; strength-reducing pow2 mul/div/umod into shifts; extend to distributivity/strength-reduction cases)
- [x] Add a fuzz target for straight-line arithmetic blocks to exercise `optimize_arith_block` end-to-end.
- [x] Add hyperfine benchmarks alongside criterion for optimizer passes to mirror C++ tooling.
- [ ] Wire the Rust optimizer into the CLI/FFI path behind a flag and start porting more C++ optimizer passes using e-graphs where beneficial.

## Active Milestone: Optimizer Integration in CLI/FFI
Connect the Rust arithmetic optimizer to user-facing entry points with safety toggles and parity tests.

Planned tasks:
- [x] Add a CLI flag and FFI toggle to route `spirv-opt` through the Rust optimizer for supported passes while falling back to C++ for the rest.
- [x] Port optimizer regressions for arithmetic canonicalization to Rust integration tests (CLI + FFI) to lock behavior.
- [x] Thread error handling through `thiserror` types at the optimizer boundary and surface structured errors to FFI/CLI callers.
- [x] Add end-to-end benchmarks (criterion + hyperfine) comparing Rust vs. C++ optimizer for supported passes. (hyperfine scripts now compare Rust, passthrough, and optional C++ spirv-opt)
- Add more affine/distributivity rewrites to expose constant folding (mixed add/sub chains, constant factorization) and port matching C++ arithmetic tests to Rust/FFI/CLI harnesses. (added add/sub operand cancellation, constant-chain merges, shared-addend cancellation including constant addends `(x+c)-(y+c)=>x-y`, commuted shared addends `(b+a)-(d+b)=>a-d`, chained sub merges `(x-y)+(y-z)=>x-z`, mirrored add/sub cancel `(x)+(y-x)=>y`, subtrahend cancellation `(x-y)+y=>x`, and factoring with commuted multiplicands `y*x + z*x` / `y*x - z*x` into `x*(y±z)` with unit coverage; broader factorization still pending and parity tests to mirror C++ remain to be ported)
- Expand e-graph coverage to additional algebraic identities and ensure the reconstructed SPIR-V preserves ids and layout invariants.
- [x] Add CI parity runner to diff Rust vs. C++ optimizer outputs on the arithmetic corpus; gate regressions.
- [x] Document optimizer parity runner usage for CI/locals.
- [x] Add hyperfine benchmark script comparing Rust opt_block, passthrough, and C++ spirv-opt for a sample module.

## Upcoming Milestone: Optimizer Parity vs C++ Arithmetic Pass
Align the Rust arithmetic optimizer with the legacy C++ arithmetic canonicalization passes.

Planned tasks:
- Collect a corpus of small arithmetic shaders and compare Rust vs. C++ optimizer outputs to detect mismatches.
- Add golden integration tests (CLI/FFI) that diff Rust-optimized modules against C++ outputs for supported ops.
  - Initial parity harnesses cover const add, add+negate, add zero, mul by zero/one, mul by -1, sub self/zero-left, double negation, div/rem by one (signed+unsigned), commutative add, and preserve div/rem-by-zero cases.
  - Added parity for distributing constant multiplication over addition and subtraction.
  - Added parity for affine GCD folding on constant add/sub cases.
- Extend e-graph rewrites to cover distributivity/simplification cases present in the C++ optimizer while keeping cost-based stability.
- Add a benchmark harness that runs both Rust and C++ optimizers via hyperfine for the arithmetic corpus to track performance deltas.
- Add a parity runner in CI that exercises `SPIRV_CPP_OPT` when available to keep regressions visible.


## Upcoming Milestone: Optimizer Default Rollout
Flip the Rust arithmetic optimizer on by default (for supported passes) while preserving a C++ fallback.

Planned tasks:
- Wire the optimizer parity runner (`scripts/run-opt-parity.sh`) into CI and track regressions over time.
- [x] Add an env/flag to force the Rust optimizer on by default in CLI/FFI (with disable override).
- Keep a benchmark guardrail (hyperfine/criterion) in CI or nightly to watch for regressions.
- [x] Document rollback/roll-forward instructions and env toggles for downstream users.

## Upcoming Milestone: Validator Default Enablement
Flip the Rust validator on by default across CLI/FFI while keeping an escape hatch.

Planned tasks:
- Add a rollout flag/env to enable the Rust validator by default in CLI/FFI with a clear C++ fallback.
- [x] Wire the validator toggle through CMake/Bazel build glue so downstreams inherit the default.
- Update CLI help/docs to describe the Rust-default behavior and override knobs.
- Keep the corpus/ctest parity runner in CI to guard the rollout and document rollback steps.
- [x] Expose CLI toggles (`--prefer-rust-validator` / `--prefer-cpp-validator`) to steer the default when both validators are available.
- [x] Add env-level defaults (`SPIRV_TOOLS_PREFER_RUST_VALIDATOR` / `SPIRV_TOOLS_PREFER_CPP_VALIDATOR`) to steer the default without forcing.

## Upcoming Milestone: Residual Layout & Structural Parity
- Finish capability/extension ordering gaps relative to the C++ validator (remaining conditional capability/extension placements and after-function checks) with Rust regressions.
- Close outstanding decoration placement/compatibility constraints in layout to mirror the C++ tables, adding paired text/binary tests.
- Add execution-model-aware entry-point interface storage-class validation (including kernel/ray/mesh-specific allowances) without regressing existing small-type capability checks; gate with Rust/C++ parity tests.
- Keep CLI/FFI parity runners updated to exercise these structural rules with the Rust validator forced on so rollout remains guarded.

## Completed Milestone: Layout Ordering Parity
Match the C++ validator’s capability/extension/import ordering and annotation placement rules in the Rust validator.

- [x] Enforce capability/extension/import ordering relative to debug/names/annotations/types/globals/functions, including conditional capabilities/extensions, with Rust regression tests.
- [x] Mirror annotation-placement rules (decorations, decoration groups, member decorations/names) against function/after-function sections with Rust tests.
- [x] Keep block-layout skipping from bypassing ordering/placement validation.
- [x] Verify ordering rules on binaries the text assembler would reorder (hand-built word tests) to mirror the C++ coverage.
- [x] Add ordering regressions for extensions/conditional extensions placed after entry points, names, and annotations to match C++ layout behavior.
- [x] Full ctest run with `SPIRV_TOOLS_DISABLE_RUST_VALIDATOR=0` passes (24 tests), matching the C++ suite with the Rust validator enabled.

## Active Milestone: Structural Validator Rules
Enforce target-environment specific structural rules and reuse validated modules across interfaces.

Tasks for this milestone:
- [x] Thread `ModuleWords`/`ValidModule` through CLI/FFI validation entry points so validated words can be reused without re-parsing.
- [x] Validate entry-point targets are `OpFunction`/`OpVariable` and report typed errors, with binary regression tests where the text assembler would reject inputs.
- [x] Add an environment-aware extension validation hook (`DisallowedExtension`) so target-specific allowlists can be enforced.
- [x] Reject extensions for WebGPU environments; extend per-target-environment allowlists in `TargetEnv::is_extension_allowed` with regression tests.
- [x] Build Vulkan/OpenCL capability allowlists (guaranteed + optional tables up through Vulkan 1.4/OpenCL 2.2), including extension- and capability-enabled cases (e.g., ray tracing extensions, OpenCL image capabilities gated on `ImageBasic`).
- [x] Add typed extension names plus env-aware extension gating (Vulkan/OpenCL prefix rules) and capability→extension dependencies (ray tracing, mesh shading, fragment shading rate/interlock, atomic float add/min/max, tile shading).
- [x] Apply grammar-driven instruction/operand capability/extension requirements and SPIR-V version gating for newer opcodes/capabilities.
- [x] Enforce structural links between entry points, execution modes, and decoration groups (execution modes must target entry points; group member decorations must target struct types and declared ids).
- [x] Reject invalid entry-point interfaces (function-scope variables, duplicate interface ids) and duplicate entry-point declarations for the same function/execution model with typed diagnostics.
- [x] Broaden decoration target constraints with paired text/binary tests (member targets, decoration categories).
- [ ] Extend capability/extension ordering and remaining decoration constraints in layout to mirror the C++ tables.
  - Added extra layout regressions rejecting capabilities after annotations and extensions after functions to mirror C++ ordering.
  - Added layout regressions for late `OpExtInstImport` and misordered `OpSamplerImageAddressingModeNV` to keep parity with the C++ layout tests.
  - Mirrored NV bindless sampler address mode rules (presence, uniqueness, bit-width validation) with binary regressions and layout-time checks.
  - Added Vulkan-only extension gating (e.g., `SPV_KHR_vulkan_memory_model`, ray tracing, descriptor indexing) so non-Vulkan environments reject them with explicit diagnostics.
  - Added Vulkan-only regressions for `SPV_KHR_workgroup_memory_explicit_layout`, `SPV_KHR_physical_storage_buffer`, the KHR/NV fragment shader barycentric extensions, `SPV_KHR_untyped_pointers`, `SPV_KHR_subgroup_uniform_control_flow`, `SPV_KHR_cooperative_matrix`, `SPV_KHR_maximal_reconvergence`, `SPV_KHR_ray_cull_mask`, `SPV_QCOM_cooperative_matrix_conversion`, and ARM vendor extensions to mirror the C++ allowlist.
  - Ported SPIR-V version gating for key extensions (ray tracing, descriptor indexing, fragment shader interlock/shading rate/density) to match C++ tables.
  - Added capability/extension version gates for physical storage buffers, storage-buffer/variable-pointer extensions, shader clock (capability + extension at SPIR-V 1.3), and DeviceGroup (SPIR-V 1.3). Continue importing remaining extension/capability version and ordering tables (e.g., maximal reconvergence, untyped pointers).
  - Pulled capability metadata from the SPIR-V grammar to drive version/extension lookups and layered in manual overrides where the grammar leaves extension-only features (e.g., shader clock). Added a regression for VariablePointers requiring VariablePointersStorageBuffer to align with the dependency tables.
  - Enforced grammar-driven capability dependencies (with a soft exception for Shader→Matrix) and added regressions for ray tracing requiring Shader, group non-uniform arithmetic requiring GroupNonUniform, and OpenCL DeviceEnqueue requiring Kernel.
  - Added layout regressions to lock ordering of capabilities/extensions relative to debug/names/annotations (including `OpSource`/`OpModuleProcessed` and name/annotation section ordering) to match the C++ validator expectations.
  - Added layout regressions covering misordered `OpSourceExtension`/`OpSourceContinued` after annotations to mirror the C++ validator behavior.
  - Added layout coverage for `OpString` ordering: rejecting strings after annotations and after the Names section to keep Debug1/Debug2 ordering aligned with the C++ validator.
  - Added a layout regression to ensure `OpModuleProcessed` follows the Names section (Debug3 after Debug2).
  - Added layout regressions for conditional capabilities/extensions (INTEL) to keep them in the capabilities/extensions sections ahead of debug/names/annotations.
  - Added layout regressions to reject conditional capabilities/extensions that trail the types-and-globals section, matching the C++ ordering expectations.
  - Added layout regressions for late extensions and imported instruction sets (extensions/ExtInstImport) to keep them ahead of annotations and debug names.
  - Added layout regressions for late capabilities (including conditional capabilities) appearing after annotations to keep section ordering tight.
  - Added a layout regression to ensure imported instruction sets (`OpExtInstImport`) do not appear after annotations.
  - Added a layout regression for `OpConditionalExtensionINTEL` appearing after debug/names to keep extension ordering consistent.
  - Added layout regressions for late extensions and capabilities after `OpExtInstImport` to lock the capabilities/extensions/import ordering.
  - Added layout regressions for capabilities or extensions appearing after `OpMemoryModel` to keep early sections strict.
  - Added a layout regression to reject debug names (`OpName`) that appear inside function bodies so debug/annotation instructions stay out of the function section.
  - Added layout regressions to keep annotation opcodes (`OpDecorationGroup`, `OpGroupDecorate`) out of function bodies and after-function sections.
  - Added layout regressions to keep member decoration opcodes (`OpMemberDecorate`, `OpGroupMemberDecorate`) confined to the annotations section.
  - Added layout regressions to ensure member naming (`OpMemberName`) stays in the names section and outside function bodies.
  - Added layout regressions to keep `OpMemberDecorateString` in the annotations section (not inside or after functions).
  - Added layout regressions to keep `OpDecorateString` in the annotations section and out of function bodies.
  - Added layout regressions to keep `OpDecorateId` confined to the annotations section (not inside or after functions).
  - Added layout regressions to keep `OpGroupDecorate`/`OpGroupMemberDecorate` out of function bodies in addition to after-function checks.
  - Added layout regressions rejecting capabilities, extensions, or imported instruction sets inside function bodies.
  - Added layout regressions rejecting extensions or imported instruction sets that appear after function definitions.
  - Added layout regressions to keep debug/source instructions (`OpSource`, `OpSourceExtension`, `OpSourceContinued`) out of function bodies and after-function sections.
  - Extended extension layout handling to cover conditional extensions and reject duplicates.
  - Added layout regressions to keep conditional extensions (`OpConditionalExtensionINTEL`) out of function bodies and after functions.
  - Added a binary regression with a well-formed module that ends in `OpConditionalExtensionINTEL` after a function to ensure ordering reports the conditional extension (not earlier layout violations).
  - Adjusted extension ordering/duplication regressions to use env-allowed extensions (and Vulkan env where necessary) so ordering/uniqueness failures trigger before allowlist gating, including NV bindless sampler address mode coverage.
  - Generator skips zero-valued bitflag enumerants when emitting operand requirements so clippy passes without bad-bitmask false positives while retaining capability/extension gating.
  - Updated misordered extension regression in the types/globals section to use an allowlisted extension string so the ordering error is exercised without hitting the extension allowlist.
  - Swapped remaining misordered extension layout regressions to use allowlisted extension strings and Vulkan env to isolate ordering diagnostics from env allowlists (debug/annotations/import/memory-model placements).
  - Brought the execution-mode/entry-point ordering regressions for extensions in line with the allowlisted/Vulkan approach so layout violations surface independent of env allowlists.
  - Converted the function/inside-function extension ordering regressions to use an allowlisted extension string and Vulkan env, keeping the ordering diagnostics focused.
  - Added layout regressions to keep conditional capabilities (`OpConditionalCapabilityINTEL`) out of function bodies and after functions.
  - Generated operand requirement tables now avoid unused value bindings so validator builds stay warning-free under cargo test/clippy.
  - Reject duplicate conditional capabilities during layout to mirror capability deduplication.
  - Added layout regression to reject `OpDecorationGroup` that appears after function definitions.
  - Added function entry block validation (functions must start with an `OpLabel`).
  - Added block terminator validation (each block must end with a terminator).
  - Added a regression for stray instructions after a terminator to mirror function/CFG rules.
  - Added branch/switch block target validation to ensure terminators reference existing blocks.
  - Added phi predecessor count validation based on collected predecessors.
  - Added a regression to ensure entry blocks have no predecessors.
  - Enforced function declarations to appear before function definitions and added tests for forward declarations versus missing entry labels.
  - Tightened phi validation: incoming blocks must exist, must be real predecessors, and must not be duplicated; added binary regressions for each case.
  - Validated function signatures against their `OpTypeFunction`: function types must be `OpTypeFunction`, return types must match, and parameter counts/types must align with the function type, with focused binary regressions.
  - Validated `OpTypeFunction` definitions themselves: return/parameter ids must name type instructions, parameters cannot be void, and malformed layouts now report typed errors with binary regressions.
  - Added return-type checks: non-void functions must use `OpReturnValue` with matching types, void functions must use `OpReturn`, and mismatches are reported with typed errors.
  - Captured the module header's declared SPIR-V version inside `ValidatedHeader`/`ValidModule` so version-aware rules can reuse it without re-parsing.
  - Version gating for capabilities, extensions, and instructions now uses the module-declared SPIR-V version (bounded by the target environment) so modules declaring older versions report precise diagnostics.
  - Exposed the effective (clamped) SPIR-V version on `ValidModule` so FFI/CLI callers can reuse the validated version without recomputing it.
  - Added operand-level regression for `ImageOperands::NON_PRIVATE_TEXEL` to require `VulkanMemoryModel` capability under Vulkan 1.2 when the capability is omitted.
  - Added operand-level regression for `MemoryAccess::NON_PRIVATE_POINTER` to require `VulkanMemoryModel` capability under Vulkan 1.2 when the capability is omitted.
  - Added operand-level regression for `MemoryAccess::MAKE_POINTER_AVAILABLE` to require `VulkanMemoryModel` capability under Vulkan 1.2 when the capability is omitted.
  - Added operand-level regressions for `OpCopyMemory` with `MemoryAccess::NON_PRIVATE_POINTER` (both SPIR-V 1.5 gating and VulkanMemoryModel capability) to mirror load/store coverage.
  - Added operand-level regressions for `OpCopyMemory` using `MAKE_POINTER_VISIBLE`/`MAKE_POINTER_AVAILABLE` to enforce SPIR-V 1.5 gating and VulkanMemoryModel capability under Vulkan 1.2.
  - Imported opcode and operand SPIR-V version requirements from the grammar and added regressions (e.g., `OpTerminateInvocation` gated at 1.6, `StorageBuffer` storage class at 1.3, `LoopControl DependencyLength` at 1.1) that exercise env clamping.
- [x] Cache validated modules across CLI/FFI invocations when the same input is reused, avoiding redundant parsing/validation.
- [ ] Expose wider structural rules (capability/extension ordering in layout, per-target decoration constraints) mirroring the C++ validator tables.
 - [x] Enforce basic-block structure: every block now requires an `OpLabel` and phi instructions must be contiguous at the top of the block, matching C++ CFG rules with Rust regressions.
 - [x] Structured merges cannot target their own header block (merge/continue/self), matching C++ structured control-flow checks.
 - [x] Enforce dominance: SSA value uses must be dominated by their definitions, and phi incoming values must dominate the incoming edge, with Rust regressions.
 - [x] Reject uses of undefined ids (module- and function-scope) to match C++ validator coverage.
 - [x] Validate that instruction result types reference type opcodes and report typed errors when they do not.
- [ ] Enable the Rust validator over the FFI/CLI by default once structural parity is sufficiently close to C++.
- [x] Run the C++ validation corpus with the Rust validator forced on (`SPIRV_TOOLS_DISABLE_RUST_VALIDATOR=0`) and close any remaining gaps; current build directory lacked test targets, so a test-enabled CMake build (`build-tests/`) was used and all 24 ctests passed with the Rust validator enabled.

## Completed Milestone: Capability/Extension Ordering Parity
Align capability/extension ordering and dependency enforcement with the C++ tables, including operand-level requirements and per-environment allowlists.

Tasks for this milestone:
- Import remaining capability/extension ordering tables so capabilities, extensions, and imports must appear before debug/annotations/functions, with text/binary regressions.
- Enforce per-environment capability/extension allowlists (beyond the Vulkan/OpenCL/WebGPU splits) using grammar metadata and C++ tables.
- Add operand-level capability/extension requirements from the grammar and test them with paired text/binary fixtures.
- Thread effective SPIR-V version and env clamping through these ordering checks for precise diagnostics.
- Keep CLI/FFI caching wired so validated modules reused across calls do not reparse/revalidate after ordering enforcement.
- Keep conditional capabilities/extensions/entry points in the parsed module so validation applies allowlists/version gates; conditional extensions now respect env allowlists with regressions for non-Vulkan/WebGPU targets.
- Treat conditional entry points equivalently to standard entry points when validating targets and execution modes; added binary regressions for invalid targets and execution-mode linkage.
- Added layout regressions to keep conditional entry points before debug/names and outside/after function bodies to mirror section ordering rules.
- Added layout coverage to ensure conditional extensions cannot follow imported instruction sets, keeping extension/import ordering aligned with the C++ validator.
- Added layout coverage to ensure conditional capabilities cannot follow imported instruction sets or the memory model, matching capability ordering rules.
- Added layout coverage to ensure conditional capabilities cannot trail the extensions section, keeping capability ordering strict.
- Added layout coverage to ensure conditional extensions cannot trail the memory model and conditional entry points cannot follow annotations, keeping section boundaries tight.
- Enforced `SPV_INTEL_function_variants` as the required extension for `SpecConditionalINTEL`/`FunctionVariantsINTEL` capabilities and added dependency/acceptance regressions.
- Blocked `SPV_INTEL_function_variants` for Vulkan environments to mirror target allowlists; added regression to ensure Vulkan rejects and Universal/OpenCL accept.
- Aligned extension version gates with the C++ tables (e.g., `SPV_KHR_vulkan_memory_model` now requires SPIR-V 1.3) and added regressions for NV shader invocation reorder and QCOM cooperative matrix conversion; Vulkan-only vendor extensions now gate NV/AMD/GOOGLE/EXT/QCOM prefixes to Vulkan with QCOM environment rejection covered by tests.
- Extended Vulkan-only extension gating to cover NV shader invocation reorder/cluster acceleration spheres, QCOM image processing/cooperative matrix conversion, and added regressions for NV shader invocation reorder in Universal targets.
- Added environment regressions for NV cluster acceleration structure and QCOM image_processing2 to keep Vulkan-only vendor extensions rejected for Universal targets.
- Tightened capability validation precedence so required extensions are reported before generic disallowance; vendor capability tests (NV ray tracing) now exercise the stricter ordering to match the C++ validator behavior.
- Generated instruction-class mapping from the grammar to drive layout section selection for capabilities/extensions/debug/annotations, keeping ordering checks aligned as new opcodes land.
- Generated a mode-setting layout map from the grammar so capabilities/extensions/memory model/entry points/execution modes route through layout checks without manual opcode lists.
- Added mode-setting ordering enforcement using the generated map and a regression that entry points cannot precede the memory model.
- Generated grammar-driven mode-setting stage/ordering tables (including conditional variants and extensions/imports) to remove hand-maintained opcode lists from the layout checker.
- Added a regression to ensure execution modes cannot precede the memory model, exercising the generated stage ordering.
- Generated grammar-driven helpers for capability/extension opcodes to back upcoming ordering enforcement without manual opcode lists.
- Routed layout capability/extension handling through the generated opcode helpers and added coverage to keep the grammar-driven classification in sync.
- Trimmed unused mode-setting kind generation so the emitted layout tables now focus on the stage map and opcode classifiers in use.
- Ran the full C++/CLI/FFI test corpus with the Rust path enabled (ctest in build-rust) and reached green across all validator/CLI/FFI suites, confirming ordering parity in practice.

## Completed Milestone: Extension Allowlists Parity
Environment-specific extension allowlists now mirror the C++ tables via the generated data set in `TargetEnv::is_extension_allowed`, with vendor gating (NV/AMDX/ARM Vulkan-only; INTEL/ALTERA OpenCL/Universal), version gates, and capability precedence covered by regressions (ray tracing adjuncts, cooperative matrices, tile shading, shader clock, fragment shading rate/density, motion blur, displacement micromaps).

## Completed Milestone: Enable Rust Validator by Default
With allowlists and capability/extension precedence aligned, the Rust validator now runs by default in the CLI/FFI path (with env/flag opt-outs) and the full C++ CLI/FFI/validator corpus passes with the Rust path enabled.

Planned tasks:
- Added a runtime toggle (default-on with env/override opt-out) so the FFI validator path prefers the Rust validator and falls back to C++ only when explicitly disabled; added unit coverage to lock the toggle behavior.
- Added CLI flags to force the Rust or C++ validator when the Rust target is built, defaulting to the Rust path unless an override or validator options (currently unsupported in Rust) are requested.
- Forwarded validator CLI options (layout relaxations, limits, friendly names) into the Rust validator via the FFI so custom flags no longer force a fallback to the C++ path.
- Apply the forwarded validator options inside the Rust validator (limits and layout relaxations) so CLI behavior matches the legacy validator without fallbacks.
- Audit remaining layout/decoration ordering rules against the C++ tables and add any missing regressions.
- Re-run the allowlist/capability matrix against the latest SPIR-V headers snapshot to catch drift before the default flip.
- Wire a feature flag to select the Rust validator by default in the CLI/FFI, keeping an opt-out for known gaps.
- Add and run a parity harness (Rust vs. C++ validator outputs) over a curated corpus in CI to detect drift.
- Run full test/CI cycles (Rust + C++) to confirm parity and update docs describing the default path.

## Upcoming Milestone: Validator Options Parity
Align Rust validator option semantics with the legacy C++ validator so CLI flags behave identically.

Planned tasks:
- Apply validator options inside the Rust validator (layout relaxations, friendly names, skip/enable layouts, local size id, offset texture operand, 32-bit bitwise) with option-aware tests.
- [x] Validate that `before_hlsl_legalization` permits the offset texture operand in Vulkan, mirroring the C++ validator behavior with a dedicated Rust regression.
- [x] Capture friendly names from `OpName`/`OpMemberName` when `use_friendly_names` is set and surface them on `ValidModule` via a typed table, with Rust regressions to keep both id and member names populated.
- [x] Gate `OpExecutionModeId LocalSizeId` by environment and the `allow_localsizeid` option, with regressions for Vulkan 1.0–1.2 defaults and the opt-in path.
- [x] Enforce Vulkan-only restrictions for image operand `Offset` (gather-only without `allow_offset_texture_operand`) and 32-bit-only bitwise operations unless `allow_vulkan_32_bit_bitwise` is set, with Rust regressions covering both the restricted and opt-in paths.
- [x] Provide friendly-name aware formatting helpers for `ValidationError` so diagnostics can render `%id (name)` when names are available.
- [x] Surface friendly names in FFI validation errors by parsing `OpName`/`OpMemberName` from the input when `use_friendly_names` is enabled.
- [x] Surface friendly names in the Rust CLI validator output by formatting validation errors with the collected name table.
- Accept layout relaxation flags (`relax_*`, `skip_block_layout`) in the Rust validator with option-aware tests to keep the CLI/FFI paths from falling back when these options are set. (`skip_block_layout` now bypasses layout ordering errors with a dedicated regression.)
- [x] Enforce `Block`/`BufferBlock` layouts when relaxations are disabled by requiring `OpMemberDecorate Offset` on every member and rejecting overlapping offsets, with Rust coverage guarding the strict path.
- [x] Honor layout relaxation flags for block layout by permitting scalar alignment for vectors (`relax_block_layout`, `uniform_buffer_standard_layout`, `scalar_block_layout`, `workgroup_scalar_block_layout`) while still validating offsets, alignment, and runtime-array placement.
- [x] Align relaxed block layout with array stride checks and vector straddle rules so misaligned strides and 16-byte straddles are rejected even under relaxed layouts, with Rust tests mirroring C++ expectations.
- [x] Enforce matrix stride alignment/size and row-major straddle rules under relaxed layouts, rejecting missing MatrixStride and strides smaller than column size.
- [x] Keep `relax_struct_store` parity for arrays by rejecting mismatched strides even under relaxed struct-store handling, with binary/regression coverage for the validation and compatibility helpers.
- [x] Enforce logical pointer rules for logical addressing (pointer-to-pointer allocations gated on VariablePointers* caps and Function/Private storage), with a `relax_logical_pointer` opt-out and Rust regressions.
- [x] Enforce `OpStore` pointer/object type compatibility with a `relax_struct_store` escape hatch for layout-compatible structs (struct/array recursion) and typed errors; added regressions covering both relaxed acceptance, array-length mismatches, and layout-relaxed acceptance (block layout relax flags).

With validator options parity achieved, proceed to capability/extension ordering and decoration/layout parity to close the remaining structural gaps.

## Active Milestone: Optimizer/Reducer Parity
Broaden optimizer/reducer parity by running the existing C++ regression corpus with the Rust path enabled and shoring up any gaps exposed by CLI/FFI flows.

Planned tasks:
- Run existing C++ optimizer/reducer suites with `SPIRV_TOOLS_DISABLE_RUST_OPT=0` and audit any output mismatches or diagnostics that differ from the legacy path.
- Add Rust-side fixes to close mismatches surfaced by the corpus without copying obvious C++ bugs; prefer e-graph rewrites and typed errors.
- Keep fuzz/bench scripts (cargo fuzz, criterion/hyperfine) up to date with the Rust path to catch performance or correctness regressions as parity expands.
- Validate the existing optimizer CLI regression suite with the Rust path enabled (done via `spirv_opt_cli_tools_tests` in CTest and `cargo test -p spirv-tools-cli`).
- Make C++ parity tests auto-discover `spirv-opt` on PATH when `SPIRV_CPP_OPT` is unset to reduce friction when running parity locally or in CI.

## Upcoming Milestone: Block/Layout Relaxations Parity
Implement the semantics of layout-related validator options so they match the C++ validator while keeping the Rust path active.

Tasks for this milestone:
- Apply `relax_block_layout`, `uniform_buffer_standard_layout`, `scalar_block_layout`, and `workgroup_scalar_block_layout` in layout/decorations validation to match Vulkan/OpenCL rules, with option-aware Rust tests.
- Apply `relax_struct_store` in memory/object validation to mirror C++ behavior and add Rust regressions for the relaxed store cases.
- Thread these options through any layout/memory checks not yet ported so CLI/FFI callers never fall back when they are set.
 - Ensure limit overrides (struct members, depth, locals, globals, switches, function args, control-flow depth, access chain indexes, id bound) are enforced with typed diagnostics. (Done for id bound, struct members/depth, locals/globals, function args, control-flow depth, switch branches, access chain indexes.)
 - Plumb option-aware diagnostics through the FFI/CLI and add integration coverage to keep the Rust path active when flags are set.

## Upcoming Milestone: Capability/Extension Parity
Bring the Rust validator to full capability/extension parity with the C++ tables using the grammar-driven dependency metadata and explicit per-environment rules.

Tasks for this milestone:
- Import the remaining capability/extension dependency tables (including operand-level requirements) from the grammar/C++ metadata and drive them from the effective SPIR-V version helper.
- Add paired text/binary regressions for version/extension/capability gating, covering env-clamped cases and Vulkan/OpenCL/WebGPU allowlists.
  - Added operand-version regressions for `ExecutionModeId`, memory-semantics `MakeVisible`/`MakeAvailable`/`OutputMemory`/`Volatile`, `NonUniform` decoration, image operands (`MakeTexelVisible`, `MakeTexelAvailable`, `NonPrivateTexel`, `VolatileTexel`, `SignExtend`, `ZeroExtend`, `Nontemporal`), and `MakePointerVisible`/`MakePointerAvailable`/`NonPrivatePointer`/`NonTemporal` memory accesses, and taught the Rust loader path to retain `OpExecutionModeId` when parsing binaries.
  - Manual capability override added for the `NonUniform` decoration to require `ShaderNonUniform` when present.
- Thread the effective version into any residual version checks (opcode tables, decoration/version ties) and ensure FFI/CLI callers can surface the clamped version for diagnostics without recomputation.
- Audit per-target allowlists for extensions/capabilities and align them with the C++ validator, extending typed errors/tests where gaps remain.

## Upcoming Milestone: Layout/Decoration Parity
Align remaining layout ordering rules and decoration target/category constraints with the C++ validator.

Planned tasks:
- Tighten capability/extension ordering to catch any remaining misorders relative to debug/names/annotations with paired text/binary regressions.
- Enforce outstanding decoration target category rules (member vs. non-member, category-specific targets) with focused binary tests.
- [x] Reject member-only decorations applied via `OpDecorate` (Offset, MatrixStride, RowMajor, ColMajor) so annotation opcodes stay in the correct form.
- [x] Add decoration target regressions for `ArrayStride` (non-array/pointer targets) and `BuiltIn WorkgroupSize` (must target constants) to mirror the C++ target-kind checks.
- [x] Add layout regression to ensure decorations recorded before `OpMemoryModel` produce the expected memory-model ordering error.
- [x] Add layout regression for `OpExtInstImport` before `OpMemoryModel` to lock ordering diagnostics.
- [x] Add layout regressions for decoration opcodes to ensure they remain in the annotations section and precede functions.
- [x] Add layout regressions for `OpDecorateId`/`OpDecorateString`/`OpMemberDecorateString` and group decorations to reject placement after the types-and-globals section.
- [x] Add layout regression to reject extensions (`OpExtension`) that appear after entry points.
- [x] Add layout regression to reject `OpMemoryModel` that appear after entry points to lock early-section ordering.
- [x] Add layout regressions to reject debug/source instructions before entry points or execution modes so debug sections remain ordered.
- [x] Add layout regressions to reject capabilities/extensions/imports that trail execution modes to keep early sections ordered.
- Reaudit per-environment decoration/capability allowlists and wire validated-module reuse through FFI/CLI caching where ordering rules apply.

## Completed Milestone: Disassembler Parity & CLI Integration
With the assembler covering the full operand space, this milestone brought the disassembler and CLI parity up to the same standard and wired the FFI so all build systems can opt into the Rust implementation.

Tasks for this milestone:
- [x] Finish binary-to-text option parity so the Rust disassembler never falls back unexpectedly.
  - [x] Implement `BinaryToTextOptions::PRINT` so stdout output matches the legacy toolchain.
  - [x] Route disassembly diagnostics through the Rust-owned context handle, mirroring the assembler path so message consumers keep working.
- [x] Surface CLI toggles for indent suppression, raw ids, nested indent, block reordering, comments, and color so the Rust `spirv-dis` mirrors the legacy flags.
- [x] Add literal-formatting (`--hex`) style toggles and update `SUPPORTED_OPTION_BITS` as coverage expands.
- [x] Align literal formatting with the legacy toolchain (signed integers, f16/f32/f64, NaN/Inf cases) and mirror the nested-indent spacing semantics, with new Rust unit tests covering these cases.
- [x] Mirror the disassembler improvements at the FFI/build-system layer (CMake/GN/Bazel) so the Rust disassembler can be enabled consistently, with documentation describing how to opt in. GN builds now honor `spirv_tools_enable_rust_target_env`/`spirv_tools_rust_profile`, while Bazel exposes the same knobs via `--define spirv_tools_enable_rust=true` and `spirv_tools_rust_profile`. Both paths build the Rust staticlib via `build_rust_ffi.py` and link the cxx bridge automatically.

### Vendor extension coverage

- [x] Regenerated the vendored `spirv`/`rspirv` crates from the repo's SPIRV-Headers snapshot so opcodes such as `OpConditionalExtensionINTEL` are part of the grammar.
- [x] Updated the rspirv loader to treat the INTEL conditional capability/extension/entry-point instructions as module-scope records and surfaced new FPEncoding operands through the assembler/disassembler.
- [x] Added a regression test (`disassembly_handles_conditional_extension_intel`) to ensure the Rust disassembler accepts binaries containing `OpConditionalExtensionINTEL` without falling back.
- [x] Honoured `PRESERVE_NUMERIC_IDS` in the Rust assembler/disassembler with regression tests mirroring `TextHandler.PreserveNumericIds` coverage.
- [ ] After the rspirv upgrade, re-tune the formatter to satisfy the legacy gtest fixtures (`BinaryToText.*`, `IndentTest.*`, string literal round-trips, float_controls2 round-trips). Key `BinaryToText/IndentTest` expectations (indent sample, nested-if, reordered-if) now have Rust parity tests and pass; remaining fixtures still need reconciliation. String literal escaping now mirrors the legacy toolchain (including multi-line literals), so the `StringLiterals/RoundTrip*` cases no longer require falling back.
  - The current gate now routes through the Rust disassembler for all option combinations (including default/zero options); diagnostics flow through the Rust context via the FFI, and `REORDER_BLOCKS` is handled natively.

## Active Milestone: Validator Core
Lay the groundwork for a Rust-native validator by implementing target-agnostic checks and wiring them behind a typed API so we can begin porting individual validation rules.

Tasks for this milestone:
- [x] Introduce a `validation` module with typed errors (`ValidationError`) and a `validate_module` entry point that parses binaries and performs invariant checks.
- [x] Validate id bounds (result ids, result types, operands) and detect duplicate result ids, with regression tests covering bound violations and duplicate definitions.
- [x] Require `OpMemoryModel` to be present before enabling function processing, with a focused regression test.
- [x] Enforce memory model ordering (reject functions before `OpMemoryModel`, duplicate memory model instructions, and out-of-order section placement) via a layout pre-pass with typed section and memory-model state.
- [x] Surface precise diagnostics for instructions that appear before `OpMemoryModel` so the Rust validator matches legacy error specificity while using newtyped ids/bounds.
- [x] Track declared capabilities and reject duplicates; introduce typed decoration/operand ids (including member targets) so structural checks operate on strongly-typed handles.
- [x] Carry a validated header (schema + bound) into `ValidModule`, reject zero-valued ids with typed wrappers, and add a text/binary `ValidatableModule` helper so validation can be driven from either representation ergonomically.
- [x] Let `ValidModule` own the validated words behind an `Arc<[u32]>` so validated modules can be shared without extra copies across FFI and CLI boundaries.
- [x] Wrap shared words in a `ModuleWords` newtype and surface `words_handle` for reuse to keep raw slices out of downstream APIs.
- [x] Add structural validation for decoration targets: member decorations must target structs and decoration groups must be declared before use. Classified `OpDecorationGroup`/`OpGroupDecorate` in the layout pass to keep logical ordering enforced.
- [x] Broaden structural checks: enforce unique extensions, ensure all decoration targets exist, and validate group/member decorations reference declared ids. CLI validation now uses `ModuleWords` to avoid extra copies while feeding the validator.
- [x] Track missing decoration targets explicitly (including group decorations) with typed errors so invalid ids are caught even when the assembler accepts the binary input.
- [x] Enforce layout ordering for capabilities/extensions and reject decoration targets that refer to undefined ids (including member targets) with focused binary regression tests.
- [x] Validate entry points reference declared ids (function and interface) with typed errors; added binary regression tests where text assembly would reject the malformed module.
- [x] Entry points now assert the referenced function is actually an `OpFunction` and interfaces are `OpVariable`, with binary regression tests covering invalid targets.
- [x] Add broader structural checks for logical layout ordering (capabilities/extensions/debug/annotations) before enabling the validator over the FFI/CLI.
- [x] Expose the Rust validator through the FFI and `spirv-val` CLI; feature-gate or deepen coverage as parity improves.

## Active Milestone: Matrix Layout Decorations
Row/column-major annotations plus matrix strides are still processed purely in C++. We now want the Rust assembler to record and validate those decorations so later passes (composite extract/insert, validator plumbing, CLI formatting) can rely on that metadata without falling back.

Tasks for this milestone:
- [x] Record row-major/column-major information for struct members (and the required `MatrixStride` operand) inside `ModuleBuilder::StructTypeInfo`, exposing helpers to look up/update member layout metadata after `OpMemberDecorate` / `OpDecorate`.
- [x] Emit diagnostics when a member receives conflicting major-ness decorations, when `MatrixStride` is missing for row/column-major members, and when decorations target non-struct members. Add focused assembler tests covering valid/invalid combinations.
- [x] Surface the stored metadata through the translator so upcoming composite instructions and future validator ports can consume it without re-parsing decorations.

## Active Milestone: Parser Diagnostics
Accurate diagnostics are critical when wiring the Rust assembler through the existing C API. We need to carry full line/column/index information through lexing/parsing so the emitted diagnostics can be compared directly against the legacy implementation.

Tasks for this milestone:
- [x] Teach the lexer/parser to honor an arbitrary source origin so each instruction line in `assemble_text` reports the correct global line/column/index even after trimming whitespace.
- [x] Add assembler tests that assert diagnostics originating from later lines (with indentation) report their true positions, preventing regressions.
- [x] Re-enable the Rust assembler for contexts that benefit from the improved diagnostics, document the behavior, and add FFI tests so `try_assemble_text` exercises the Rust path end-to-end.

## Active Milestone: Binary/Text Infrastructure
With the foundational workspace pieces in place, we now focus on the assembler and disassembler pipeline. This milestone delivers a Rust-native text/binary conversion path that can be swapped into the existing C entry points behind the FFI bridge.

Tasks for this milestone:
- [x] Build a strongly-typed lexer/token stream for SPIR-V assembly text that tracks source positions and quotes, so parsing can stay zero-copy where possible.
- [x] Model the intermediate instruction representation (IDs, operands, literal values) with newtypes that enforce the operand kinds the grammar expects.
- [x] Implement the assembler driver that consumes the lexer, consults the grammar tables, and emits binaries plus diagnostics through the Rust context.
  - [x] Cover core module metadata (capabilities, entry points, execution modes, pointer types, and global variables) plus basic block instructions such as loads, stores, and arithmetic.
  - [x] Support optional operands (memory access masks, alignment), composite instructions, access chains, and control-flow constructs so most shaders assemble entirely in Rust.
    - [x] Parse and encode memory access operands (alignment + pointer scopes) for `OpLoad`/`OpStore`.
    - [x] Extend optional operand coverage to copy-memory instructions (dual masks) and other ops that require literal/ID payloads.
    - [x] Teach the parser/translator to accept trailing annotation operands so `OpDecorate`/`OpMemberDecorate` can emit row-major matrix layouts (`MatrixStride`, `RowMajor`) without falling back to C++.
    - [x] Encode decoration operands (built-ins, linkage attributes, numeric IDs, bitflags such as `FPFastMathMode`) based on the grammar metadata so the Rust assembler covers the entire annotation space, including `OpDecorateId`.
    - [x] Add composite instruction coverage (e.g., `OpCompositeConstruct`, `OpVectorShuffle`) so complex data assembly no longer falls back.
      - [x] Implement `OpCompositeConstruct`, `OpTypeVector`, and `OpVectorShuffle` translators.
      - [x] Extend coverage to `OpCompositeExtract`, `OpCompositeInsert`, and `OpPhi` boolean helpers. Vector shuffle inputs now reuse tracked type metadata so we reject mismatched component counts/types.
      - [x] Track array/struct layouts to validate composite accesses (including array bounds) and add regression tests covering both valid and invalid cases.
      - [x] Extend the same metadata to matrices (column vectors + column counts) and nested aggregates so multi-level composite extracts/inserts no longer fall back.
      - [x] Track lexer spans on result/type identifiers and surface them through the translator so diagnostics report real line/column information instead of anonymous locations.
      - [x] Support GLSL/OpenCL extended instruction sets (named opcodes, literal integers, rounding modes, variadic operands) so `OpExtInst` can stay in Rust for those imports.
- [x] Expose the Rust assembler via `try_assemble_text`, returning success/failure and hooking diagnostics into the existing consumers. Fall back to the legacy C++ assembler only while gaps remain.
  - [x] Guard the FFI entry points so they fall back to the C++ assembler/disassembler unless explicitly re-enabled. This keeps CI green while we continue fleshing out opcode and option coverage on the Rust side.
    - [x] Allow the Rust disassembler to service `NO_HEADER` requests (and the incidental `PRINT` flag) while leaving other options to the legacy implementation, keeping the fallback behavior explicit in both Rust and C++.
    - [x] Enable byte-offset emission in the Rust disassembler so `NO_HEADER | SHOW_BYTE_OFFSET` requests stay entirely in Rust while unsupported combinations are rejected via typed formatting options.
    - [x] Teach the Rust disassembler to honor the `INDENT` option so CLI clients using `--no-header --no-color --raw-id --offsets` can stay entirely in Rust without losing the aligned opcode formatting.
    - [x] Support friendly-name emission via `BinaryToTextOptions::FRIENDLY_NAMES`, reusing `OpName` payloads when available and synthesizing stable `_N` identifiers so the default `spirv-dis` path (which requests friendly names) remains in Rust unless other unsupported options are toggled.
  - [x] Implement `NESTED_INDENT` so structured control flow is indented using merge information mirroring the C++ formatter, allowing `spirv-dis --nested-indent` to operate entirely within the Rust implementation.
  - [x] Honor the `COMMENT` option by tracking decoration metadata and per-instruction annotations, reproducing the byte-offset/decoration comment stream so `spirv-dis --comment` no longer falls back to C++.
  - [x] Reorder basic blocks on demand when `REORDER_BLOCKS` is set, using a simple CFG walk to mirror structured control-flow ordering so `spirv-dis --reorder-blocks` stays in Rust.
- [x] Support the `COLOR` flag by injecting ANSI escapes around IDs and comments when requested so `spirv-dis --color` produces colored output entirely in Rust.
- [x] Mirror the improvements in the disassembler path (options filtering, message routing) so both directions benefit from the Rust context. (Rust disassembler now defers diagnostics to the caller so C++ fallback can consume them without double-reporting; unsupported-option errors are surfaced alongside the fallback diagnostics.)
- [x] Add FFI/CLI-facing tests to ensure Rust disassembler diagnostics are surfaced when the Rust path fails and the C++ fallback runs, keeping message routing consistent across both implementations.
- [x] Canonicalize assembler layout to spec order while preserving in-section ordering, and add Arm Motion Engine opcode/name lookup so extended instructions round-trip via the Rust assembler/disassembler.

## Upcoming Milestone: Operand Requirements Parity
Tighten operand-level capability/extension/version enforcement to mirror the C++ validator tables.

Tasks for this milestone:
- [x] Import operand-level capability/extension/version requirement data from the SPIR-V grammar and enforce it during validation (including conditional operands).
- [x] Add binary/text regression tests covering operand requirements for representative instructions (e.g., memory semantics masks, subgroup scopes, and newer operand enums gated by extensions).
- Thread operand requirement failures through typed `ValidationError` variants so FFI/CLI callers receive structured diagnostics.
- Keep validated-module caching active to avoid re-validation when operand checks are enabled.

## Upcoming Milestone: Capability/Extension Ordering Parity
Align capability/extension ordering (including conditional variants) with the C++ tables and broaden layout ordering coverage.

Tasks for this milestone:
- Generate ordering tables for capabilities, extensions, and conditional variants from the grammar and enforce them relative to debug/names/annotations/functions.
- Add text/binary regressions mirroring C++ ordering checks (late capabilities/extensions/imports, conditional variants after functions, etc.).
- Thread effective SPIR-V version/env gating through ordering enforcement for precise diagnostics.
- Keep module caching enabled so repeated validations avoid reparsing after ordering checks are applied.

## Upcoming Milestone: Extension Allowlist Parity
Ensure extension allowlists are enforced consistently during layout and full-module validation, matching C++ behavior for conditional extensions and environment-specific bans.

Tasks for this milestone:
- Enforce extension allowlists (including conditional extensions) during full validation, not just layout, and return typed `DisallowedExtension` errors.
- Add integration tests covering conditional extensions in disallowed environments (e.g., WebGPU rejecting ray tracing extensions) and allowed environments (e.g., Vulkan).
- Keep extension gating data driven by the grammar allowlist and environment metadata.

## CLI Development
- [x] Introduce a `spirv-tools-cli` crate that wraps the core disassembly logic with `clap`, providing an initial `spirv-dis` binary capable of mirroring the `--no-header`/`--offsets` flags while reading from stdin or files.
- [x] Add `spirv-as` and `spirv-val` binaries so the Rust workspace mirrors the C++ tool surface, sharing reusable helpers for option parsing and file/stdin plumbing. The validator CLI currently delegates to the existing C++ validator over the FFI bridge and is covered by unit + integration tests to guard success/failure paths.

When the flag is enabled CMake continues to drive `cargo build -p spirv-tools-ffi` (profile configurable via `SPIRV_RUST_PROFILE`) and links the resulting staticlib into the core library while compiling the generated `rust/cxxbridge/spirv-tools-ffi.cc` shim.

Regenerate the bridge artifacts (`rust/cxxbridge/spirv-tools-ffi.{h,cc}`) after editing the Rust FFI surface with:

```
cxxbridge rust/spirv-tools-ffi/src/lib.rs --header > rust/cxxbridge/spirv-tools-ffi.h
cxxbridge rust/spirv-tools-ffi/src/lib.rs --output rust/cxxbridge/spirv-tools-ffi.cc
```

## Testing Strategy
- Unit tests live alongside Rust modules using `#[cfg(test)]` and cover exhaustive enum conversions and validation helpers.
- Cross-language FFI tests ensure the exported symbols maintain the same behavior as the C implementation.
- Long term: reuse/port existing `test/` suites into Rust integration tests.

## Progress Dashboard

| Area | Scope | Status |
|------|-------|--------|
| Target-environment helpers | Rust target-env types + diagnostics | ✅ Complete |
| Text assembler/disassembler stack | Lexer/parser, binary/text conversion, diagnostics, FFI plumbing | ✅ Complete |
| Validator/optimizer/reducer ports | Move validator passes, optimizer, reducer into Rust, expose via FFI | ⏳ In progress (layout pre-pass and id-bound checks wired) |
| Rust FFI bridge | Context ownership, assembler/disassembler exports, option sanitization | ~50 % (validator/optimizer exports TBD) |
| CLI workspace | `spirv-dis`, `spirv-as`, `spirv-val` implemented; remaining tools mirror C++ | ~35 % (3/8 CLIs shipped) |
| Build-system integration | CMake + GN + Bazel wiring for Rust staticlib | ✅ Complete for existing features |

Percentages are approximate and will be updated as new checklists are added for each subsystem.

## Open Questions / Risks
- Determining the best boundary for delegating back to existing C++ code during the transition.
- Build-system integration for consumers that currently rely on CMake/Bazel; for now, manual invocation via `cargo` is sufficient.
- Mapping of large switch-based operand logic into data-driven Rust structures without performance regressions.

## Next Up
1. Finish function/CFG parity: deepen SSA/type checking inside functions (operand types beyond calls/phi), validate interface linkage and descriptor set/binding rules, grow merge/continue/phi regressions (including duplicate label detection), and align dominance rules with the C++ validator where they still diverge.
2. Close any remaining decoration/env constraints that are still table-driven in C++ (e.g., leftover built-in placement quirks) and back them with Rust regressions.
3. Push validated-module caching through CLI/FFI plumbing where re-parsing still happens, keeping the Rust path zero-cost when reused.
4. Round out SPIR-V version/extension gating pulled from the grammar (instruction/operand-level) and document the Rust-default validator rollout knobs for CLI/FFI callers.
5. Keep validator rollout guardrails in CI: run the ctest/corpus jobs with `SPIRV_TOOLS_DISABLE_RUST_VALIDATOR=0`, and capture any deltas before flipping the default.

## Completed Milestone: Capability/Extension Ordering Parity
Align the Rust validator’s capability/extension ordering and layout checks with the C++ tables.

Tasks for this milestone:
- [x] Import capability/extension ordering tables (including conditional capabilities/extensions) and enforce them with text/binary regression tests (mode-stage/order checks are generated from the grammar; regressions cover conditional variants and late-section placements).
- [x] Mirror environment-specific decoration/layout constraints driven by the grammar data.
- [x] Thread validated-module reuse through FFI/CLI for these new checks to avoid reparsing.
- [x] Add regressions that exercise conditional capabilities/extensions around Debug/Names/Annotations/Types boundaries so layout errors surface identically to the C++ validator (coverage now includes extensions/capabilities after annotations, extinst import, execution modes, memory model, and inside functions).

## Completed Milestone: Operand Requirements Parity
Tighten operand-level capability/extension/version enforcement to mirror the C++ validator tables.

Tasks for this milestone:
- Import operand-level capability/extension/version requirement data from the SPIR-V grammar and enforce it during validation (including conditional operands).
- Add binary/text regression tests covering operand requirements for representative instructions (e.g., memory semantics masks, subgroup scopes, and newer operand enums gated by extensions).
- Thread operand requirement failures through typed `ValidationError` variants so FFI/CLI callers receive structured diagnostics.
- Keep validated-module caching active to avoid re-validation when operand checks are enabled.
- Added operand-level regressions for `Scope::QueueFamilyKHR` to require `VulkanMemoryModel` (failure without capability, acceptance with capability + extension under the Vulkan memory model).
- Added operand-level regressions for `Scope::ShaderCallKHR` to require `RayTracingKHR` (failure without capability, acceptance with capability + extension in Vulkan env).
- Added operand-level regressions for `ImageOperands::MAKE_TEXEL_VISIBLE/AVAILABLE` to require `VulkanMemoryModel` (fail without capability, pass with capability/extension under `VulkanKHR` memory model).
- Added operand-level regressions for `ImageOperands::NON_PRIVATE_TEXEL` to require `VulkanMemoryModel` (fail without capability, pass with capability/extension under `VulkanKHR` memory model).

## Active Milestone: Function and CFG Validation
Bring function-body validation in line with the C++ validator so the Rust validator can be enabled by default.

Tasks for this milestone:
- Validate function definitions: block ordering, structured control flow (merge/continue rules), and minimal well-formedness (single entry, required terminators).
- Enforce SSA/phi correctness (dominance of defs, matching predecessor counts/types) and type checking for instructions beyond the current structural pass.
- Detect unreachable basic blocks in functions and report them with typed diagnostics; added regression in Rust validator tests.
- Enforce structured merge rules: merges must immediately precede their terminators, selection merges pair with conditional/switch terminators, loop merge/continue targets must exist and be distinct, and structured terminators require a selection merge.
- [x] Enforce function call signatures (target must be a function definition; return/argument counts and types must match the callee) with Rust regressions.
- [x] Reject values defined in one function that are referenced in another, with Rust regression coverage.
- [x] Enforce function-scoped variables to use Function storage class with Rust regression coverage.
- [x] Enforce function-scoped variables to be declared in the entry block with Rust regression coverage.
- [x] Enforce entry-point interface variables to avoid Function storage class with Rust regression coverage.
- [x] Enforce uniqueness of entry-point interface ids with Rust regression coverage.
- Validate interface linkage for variables and descriptor sets/bindings where applicable per environment.
- Wire the Rust validator through FFI/CLI as the default path (behind a feature flag) once the above checks and layout parity are in place, backed by mirrored gtest/integration coverage.

## Active Milestone: Validator Parity Rollout
Run the full C++ validation suites with the Rust path forced on, close remaining gaps, and prepare to flip the default.

Tasks for this milestone:
- [x] Force-enable the Rust validator across CLI/FFI parity runs and record any mismatches against the C++ validator. (ctest suite passes with the Rust validator forced on)
- Close any remaining decoration/env-specific placement quirks (BuiltIn allowlists, interpolation/barycentric rules) with Rust regressions.
- Keep the plan updated with rollout status and fallbacks, and land a toggle to make the Rust validator the default once parity is demonstrated.

## Upcoming Milestone: Optimizer E-Graph Port
Drive the optimizer rewrite in Rust using `egg`/e-graphs with fuzzing and performance guardrails.

Planned tasks:
- [x] Scaffold an `egg`-powered optimizer crate (`spirv-tools-opt`) with algebraic rewrites and constant folding as the initial backbone.
- [x] Add a Criterion benchmark harness to track optimizer performance as rewrites expand.
- [x] Seed a fuzzing harness (`cargo fuzz` + `arbitrary`) for optimizer expressions, reusing the shared generator in `spirv-tools-opt` (see `rust/spirv-tools-opt/fuzz/fuzz_targets/expr_opt.rs`).
- [x] Extend optimizer rewrites to simplify algebraic identities (add zero, multiply by one/zero) with regression tests.
- [x] Translate a subset of SPIR-V arithmetic (OpConstant/OpIAdd/OpIMul/OpISub/OpSNegate/OpSDiv/OpUDiv/OpSRem/OpUMod) into e-graph expressions and collapse reducible blocks back to constants with round-trip tests.
- [x] Add divide/remainder/negation language support with folding guards (division-by-zero preserved) and block-level regressions.
- Model optimizer IR and rewrites in Rust with `egg`, keeping transformations zero-cost and type-safe.
- Expose optimizer controls through FFI/CLI compatible with the existing C API and binaries.
- Add fuzzing harnesses using `cargo fuzz` + `arbitrary` to stress rewrites and round-trip assembly/disassembly.
- Establish benchmarks with `criterion` (and `hyperfine` for CLI) to track regressions against the C++ optimizer.
- Port representative optimizer passes and their C++ tests into Rust unit/integration tests to validate e-graph results.
   - Added C++ parity coverage for shared-addend cancellation with symbolic terms, ensuring `(x+5)-(x+2)` folds identically in Rust and C++ paths.
   - Added C++ parity coverage for shared-addend cancellation that simplifies to zero: `(x+7)-(x+7)` -> `0` in both Rust and C++ optimizers.
   - Added C++ parity coverage for factoring a symbolic multiplier across subtracted constants: `(x*5)-(x*2)` => `3*x`, confirming the e-graph factoring rewrites match spirv-opt.
   - Added C++ parity coverage for factoring commuted symbolic multiplicands in both addition and subtraction: `(y*x)+(x*z)` => `x*(y+z)` and `(y*x)-(z*x)` => `x*(y-z)`, exercising commuted-operand factoring in Rust and C++.
   - Added C++ parity coverage for factoring constant multipliers out of commuted add/sub expressions: `(4*x)+(4*y)` => `4*(x+y)` and `(6*x)-(6*y)` => `6*(x-y)`, ensuring const factoring aligns between Rust and C++.
   - Added C++ parity coverage for factoring mixed symbolic/constant multiplicands: `(x*y)+(x*3)` => `x*(y+3)` and `(x*y)-(x*3)` => `x*(y-3)`, covering affine-like patterns in both optimizers.
  - Added C++ parity coverage for mixed-constant factoring with commuted multiplicands: `(2*x)+(x*3)` => `5*x` and `(2*x)-(x*3)` => wrapped `-1 * x`, exercising both positive and wrapping-negative constant combinations.
  - Added C++ parity coverage for mixed-constant factoring with positive difference: `(3*x)-(2*x)` => `1*x`, ensuring non-wrapping constant differences are simplified in both optimizers.
  - Added C++ parity coverage for zero-factor cancellation: `(0*x)+(0*y)` and `(0*x)-(0*y)` both fold to zero in Rust and C++ optimizers.
  - Added an FFI regression for affine GCD subtraction folding (`(14*2)-21 -> 7`) to keep the bridge aligned with the Rust optimizer behavior.
   - Added a short `cargo fuzz` smoke script (`scripts/fuzz-smoke.sh`) to keep the arithmetic optimizer fuzz target exercised with a bounded run.
   - Added an FFI regression for factoring linear combinations into a single constant result, ensuring the C bridge exercises the optimizer path.
   - Added a CLI-facing `opt_block` binary and integration test to optimize basic blocks on-disk, paving the way for drop-in CLI parity and hyperfine comparisons.
   - Added a hyperfine benchmark script (`scripts/hyperfine-opt.sh`) to compare the Rust optimizer CLI against the C++ spirv-opt when available.
   - Added a `--cpp` fallback flag to `spirv-opt` CLI to run the C++ binary for benchmarking/compatibility, while keeping the Rust path enabled by default.

## Upcoming Milestone: Optimizer FFI/CLI Integration
Expose the Rust optimizer through the existing C/C++ surfaces and provide CLI benchmarking.

Planned tasks:
- Add FFI hooks in `spirv-tools-ffi` to invoke the Rust optimizer on arithmetic/basic-block inputs, preserving the C API shape. **(in place: `optimize_basic_block` exposed, exercised by FFI tests)**
- Wire the Rust optimizer into the CLI with a feature flag and add a hyperfine benchmark script comparing Rust vs C++ optimizer paths. **(done: `spirv-opt` uses Rust by default, `--cpp` fallback for C++; `scripts/hyperfine-opt.sh` added)**
- Extend translation to cover common arithmetic/basic-block patterns (including div/rem/neg) and ensure non-arithmetic ops pass through untouched.
- Add fuzz targets for translated basic blocks to stress end-to-end translation + optimization.
- Port representative optimizer tests from the C++ suite to the Rust path to validate parity over FFI/CLI.
   - Added a CLI integration test covering div/rem/neg/shifts to keep translation coverage guarded in the Rust path.

## Upcoming Milestone: Optimizer Default & CI Guardrails
Flip the Rust optimizer on by default across FFI/CLI once parity is proven and guardrails are in place.

Tasks for this milestone:
- Add CI steps to run the hyperfine and fuzz smoke scripts (`scripts/hyperfine-opt.sh`, `scripts/fuzz-smoke.sh`) to catch perf/correctness regressions.
- [x] Add runtime toggles (env + FFI override) for the Rust optimizer so CLI/FFI callers can force-enable/disable independently of env defaults, with regression tests guarding the wrappers.
- [x] Expose a CLI `--force-rust` flag that ignores `SPIRV_TOOLS_DISABLE_RUST_OPT` for benchmarking/rollout control, with an integration test covering env override.
- [x] Add a CI-friendly optimizer smoke wrapper (`scripts/ci-optimizer-smoke.sh`) that runs fuzz smoke and optional hyperfine benchmarks when available.
- Wire the Rust optimizer through the main CXX bridge behind a feature flag and port representative C++ optimizer tests to the Rust path via that bridge.
- Add a nightly/longer-running fuzz job (separate from smoke) to stress end-to-end translation + optimization.
- Track a “Rust path by default” toggle and document rollout/rollback procedures for CLI/FFI consumers.
- Validator clippy is warning-free; ready for full corpus parity runs with the Rust validator forced on (`SPIRV_TOOLS_DISABLE_RUST_VALIDATOR=0`).
