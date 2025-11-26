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
- [x] Broaden decoration target constraints with paired text/binary tests (member targets, decoration categories).
- [ ] Extend capability/extension ordering and remaining decoration constraints in layout to mirror the C++ tables.
  - Added layout regressions for late `OpExtInstImport` and misordered `OpSamplerImageAddressingModeNV` to keep parity with the C++ layout tests.
  - Mirrored NV bindless sampler address mode rules (presence, uniqueness, bit-width validation) with binary regressions and layout-time checks.
  - Added Vulkan-only extension gating (e.g., `SPV_KHR_vulkan_memory_model`, ray tracing, descriptor indexing) so non-Vulkan environments reject them with explicit diagnostics.
  - Ported SPIR-V version gating for key extensions (ray tracing, descriptor indexing, fragment shader interlock/shading rate/density) to match C++ tables.
  - Added capability/extension version gates for physical storage buffers, storage-buffer/variable-pointer extensions, shader clock (capability + extension at SPIR-V 1.3), and DeviceGroup (SPIR-V 1.3). Continue importing remaining extension/capability version and ordering tables (e.g., maximal reconvergence, untyped pointers).
  - Pulled capability metadata from the SPIR-V grammar to drive version/extension lookups and layered in manual overrides where the grammar leaves extension-only features (e.g., shader clock). Added a regression for VariablePointers requiring VariablePointersStorageBuffer to align with the dependency tables.
  - Enforced grammar-driven capability dependencies (with a soft exception for Shader→Matrix) and added regressions for ray tracing requiring Shader, group non-uniform arithmetic requiring GroupNonUniform, and OpenCL DeviceEnqueue requiring Kernel.
  - Added layout regressions to lock ordering of capabilities/extensions relative to debug/names/annotations (including `OpSource`/`OpModuleProcessed` and name/annotation section ordering) to match the C++ validator expectations.
  - Added layout regressions covering misordered `OpSourceExtension`/`OpSourceContinued` after annotations to mirror the C++ validator behavior.
  - Added layout coverage for `OpString` ordering: rejecting strings after annotations and after the Names section to keep Debug1/Debug2 ordering aligned with the C++ validator.
  - Added a layout regression to ensure `OpModuleProcessed` follows the Names section (Debug3 after Debug2).
  - Added layout regressions for conditional capabilities/extensions (INTEL) to keep them in the capabilities/extensions sections ahead of debug/names/annotations.
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
  - Added a layout regression to keep `OpMemberDecorateString` in the annotations section ahead of functions.
  - Added layout regressions to keep `OpDecorateString` in the annotations section and out of function bodies.
  - Added layout regressions to keep `OpDecorateId` confined to the annotations section (not inside or after functions).
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
  - Imported opcode and operand SPIR-V version requirements from the grammar and added regressions (e.g., `OpTerminateInvocation` gated at 1.6, `StorageBuffer` storage class at 1.3, `LoopControl DependencyLength` at 1.1) that exercise env clamping.
- [x] Cache validated modules across CLI/FFI invocations when the same input is reused, avoiding redundant parsing/validation.
- [ ] Expose wider structural rules (capability/extension ordering in layout, per-target decoration constraints) mirroring the C++ validator tables.
- [ ] Enable the Rust validator over the FFI/CLI by default once structural parity is sufficiently close to C++.

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
- Add layout regressions for decoration opcodes to ensure they remain in the annotations section and precede functions.
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
- [ ] Add broader structural checks for logical layout ordering (capabilities/extensions/debug/annotations) before enabling the validator over the FFI/CLI.
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
- [ ] Mirror the improvements in the disassembler path (options filtering, message routing) so both directions benefit from the Rust context.

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
1. Broaden structural validation for decoration target categories and capability/extension ordering (text + binary coverage), keeping parity with the C++ validator tables.
   - Added layout-order regression tests for section ordering (ExtInstImport, debug/names/annotations) and BuiltIn target categories to lock in current behavior.
2. Push validated-module caching deeper through FFI/CLI entry points to avoid reparsing/validating identical inputs.
3. Fill in remaining SPIR-V version gating and per-instruction requirements from the grammar, then revisit disassembly fixtures once validator parity is solid.

## Upcoming Milestone: Capability/Extension Ordering Parity
Align the Rust validator’s capability/extension ordering and layout checks with the C++ tables.

Tasks for this milestone:
- Import capability/extension ordering tables (including conditional capabilities/extensions) and enforce them with text/binary regression tests.
- Mirror environment-specific decoration/layout constraints driven by the grammar data.
- Thread validated-module reuse through FFI/CLI for these new checks to avoid reparsing.

## Upcoming Milestone: Layout Ordering Parity
Tighten layout ordering to match the C++ validator’s section/decoration ordering rules.

Tasks for this milestone:
- Enforce capability/extension ordering relative to debug/names/annotations and module layout (no late section regressions), including conditional extensions/capabilities.
- Add decoration ordering/category checks that remain in the C++ tables (e.g., per-target-env decoration placement quirks) with paired text/binary regressions.
- Keep ValidModule caching wired through FFI/CLI for these ordering checks to avoid reparsing/renumbering.

## Upcoming Milestone: Function and CFG Validation
Bring function-body validation in line with the C++ validator so the Rust validator can be enabled by default.

Tasks for this milestone:
- Validate function definitions: block ordering, structured control flow (merge/continue rules), and minimal well-formedness (single entry, terminals).
- Enforce SSA/phi correctness (dominance of defs, matching predecessor counts/types) and type checking for instructions beyond the current structural pass.
- Validate interface linkage for variables and descriptor sets/bindings where applicable per environment.
- Wire the Rust validator through FFI/CLI as the default path (behind a feature flag) once the above checks and layout parity are in place, backed by mirrored gtest/integration coverage.
