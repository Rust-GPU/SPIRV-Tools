//! Validation error types for SPIR-V module validation.

use thiserror::Error;

use super::types::{
    CheckedBound, DeclaredBound, DecorationTargetId, DecorationTargetKind, ExtensionName, Id,
    IdKind, MemberDecorationTargetId, MemberIndex, MergeTargetKind, ResultId, TypeId,
};
use crate::{target_env::TargetEnv, version::SpirvVersion};

/// Errors that can arise when validating a SPIR-V module.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum ValidationError {
    // ========== CORE ==========
    /// The module failed to parse before validation could run.
    #[error("failed to parse module: {0}")]
    Parse(String),
    /// The module header is missing.
    #[error("module header is missing")]
    MissingHeader,
    /// The module is missing the required `OpMemoryModel` instruction.
    #[error("OpMemoryModel is required before any function definitions")]
    MissingMemoryModel,
    /// The module declared more than one memory model instruction.
    #[error("multiple OpMemoryModel instructions are not allowed")]
    DuplicateMemoryModel,
    /// A function definition appeared before the memory model.
    #[error("OpMemoryModel must appear before any function definitions")]
    FunctionBeforeMemoryModel,
    /// An instruction was found before the required memory model declaration.
    #[error("instruction {opcode:?} cannot appear before OpMemoryModel")]
    InstructionBeforeMemoryModel {
        /// The opcode that violated the ordering.
        opcode: rspirv::spirv::Op,
    },
    /// A validation limit was exceeded.
    #[error("validator limit {limit_kind} exceeded: found {found}, limit {limit}")]
    LimitExceeded {
        /// The limit kind (matches `spv_validator_limit`).
        limit_kind: u32,
        /// The configured limit value.
        limit: u32,
        /// The observed value.
        found: u32,
    },
    /// Global instructions are out of the required logical layout order.
    #[error("instruction {opcode:?} appears out of order in the logical layout")]
    LayoutOutOfOrder {
        /// The opcode that violated the ordering.
        opcode: rspirv::spirv::Op,
    },
    /// The module declared an invalid id bound (must be greater than zero).
    #[error("declared id bound {bound} is invalid")]
    InvalidIdBound {
        /// The declared id bound from the module header.
        bound: DeclaredBound,
    },
    /// The declared id bound exceeds a configured validator limit.
    #[error("declared id bound {declared} exceeds validator limit {limit}")]
    IdBoundExceedsLimit {
        /// The declared id bound from the module header.
        declared: DeclaredBound,
        /// The configured limit for the bound.
        limit: u32,
    },
    /// The module header declared a non-zero reserved word (must be zero).
    #[error("module reserved word must be zero (found {reserved})")]
    InvalidReservedWord {
        /// The declared reserved value from the module header.
        reserved: u32,
    },
    /// The module declared an id bound that is exceeded by at least one id.
    #[error("id {id} exceeds declared id bound {bound}")]
    IdExceedsBound {
        /// The offending id value.
        id: Id,
        /// The declared id bound from the module header.
        bound: CheckedBound,
    },
    /// Duplicate result ids were found in the module.
    #[error("id {id} is defined more than once")]
    DuplicateResultId {
        /// The result id that was defined multiple times.
        id: Id,
    },

    // ========== CAPABILITIES ==========
    /// Duplicate capability declarations were found.
    #[error("capability {capability:?} is declared more than once")]
    DuplicateCapability {
        /// The capability that was duplicated.
        capability: rspirv::spirv::Capability,
    },
    /// A capability is not permitted for the target environment.
    #[error("capability {capability:?} is not allowed for target environment {env:?}")]
    DisallowedCapability {
        /// The capability that was not allowed.
        capability: rspirv::spirv::Capability,
        /// The target environment in use.
        env: TargetEnv,
    },
    /// A capability requires an extension that was not declared.
    #[error("capability {capability:?} requires extension {required_extension}")]
    DisallowedCapabilityMissingExtension {
        /// The capability that was not allowed.
        capability: rspirv::spirv::Capability,
        /// The required extension name.
        required_extension: String,
    },
    /// A capability requires a newer SPIR-V version than the target environment provides.
    #[error(
        "capability {capability:?} requires SPIR-V version {required_version}, but target provides {target_version}"
    )]
    CapabilityRequiresSpirvVersion {
        /// The capability that was not allowed.
        capability: rspirv::spirv::Capability,
        /// The minimum SPIR-V version required.
        required_version: SpirvVersion,
        /// The target environment's SPIR-V version.
        target_version: SpirvVersion,
    },
    /// An instruction requires a newer SPIR-V version than the target environment provides.
    #[error(
        "instruction {opcode:?} requires SPIR-V version {required_version}, but target provides {target_version}"
    )]
    InstructionRequiresSpirvVersion {
        /// The opcode that is too new.
        opcode: rspirv::spirv::Op,
        /// The minimum SPIR-V version required.
        required_version: SpirvVersion,
        /// The target environment's SPIR-V version.
        target_version: SpirvVersion,
    },
    /// An operand requires a newer SPIR-V version than the target environment provides.
    #[error(
        "operand {operand_index} of {opcode:?} requires SPIR-V version {required_version}, but target provides {target_version}"
    )]
    OperandRequiresSpirvVersion {
        /// The opcode containing the operand.
        opcode: rspirv::spirv::Op,
        /// Index of the operand within the instruction.
        operand_index: usize,
        /// The minimum SPIR-V version required.
        required_version: SpirvVersion,
        /// The target environment's SPIR-V version.
        target_version: SpirvVersion,
    },
    /// A capability requires another capability that was not declared.
    #[error("capability {capability:?} requires capability {required_capability:?}")]
    MissingRequiredCapability {
        /// The capability that is missing.
        required_capability: rspirv::spirv::Capability,
        /// The capability that referenced it.
        capability: rspirv::spirv::Capability,
    },
    /// An instruction requires a capability that was not declared.
    #[error("instruction {opcode:?} requires capability {required_capability:?}")]
    MissingInstructionCapability {
        /// The opcode requiring the capability.
        opcode: rspirv::spirv::Op,
        /// The missing capability.
        required_capability: rspirv::spirv::Capability,
    },
    /// An instruction requires an extension that was not declared.
    #[error("instruction {opcode:?} requires extension {required_extension}")]
    MissingInstructionExtension {
        /// The opcode requiring the extension.
        opcode: rspirv::spirv::Op,
        /// The missing extension.
        required_extension: ExtensionName,
    },
    /// An operand requires a capability that was not declared.
    #[error("operand {operand_index} of {opcode:?} requires capability {required_capability:?}")]
    MissingOperandCapability {
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Operand index within the instruction.
        operand_index: usize,
        /// Missing capability.
        required_capability: rspirv::spirv::Capability,
    },
    /// An operand requires an extension that was not declared.
    #[error("operand {operand_index} of {opcode:?} requires extension {required_extension}")]
    MissingOperandExtension {
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Operand index within the instruction.
        operand_index: usize,
        /// Missing extension.
        required_extension: ExtensionName,
    },
    /// A variable containing a small (8- or 16-bit) element is not allowed in
    /// the given storage class without an additional capability.
    #[error(
        "Allocating a variable containing a {bit_width}-bit element in {storage_class:?} storage class requires an additional capability"
    )]
    SmallTypeMissingCapability {
        /// The element bit width (8 or 16).
        bit_width: u32,
        /// The storage class of the variable.
        storage_class: rspirv::spirv::StorageClass,
        /// The capability required to allow the allocation.
        required_capability: rspirv::spirv::Capability,
    },
    /// A variable containing a small (8- or 16-bit) element is disallowed in
    /// the given storage class.
    #[error(
        "Cannot allocate a variable containing a {bit_width}-bit type in {storage_class:?} storage class"
    )]
    SmallTypeDisallowedInStorageClass {
        /// The element bit width (8 or 16).
        bit_width: u32,
        /// The storage class of the variable.
        storage_class: rspirv::spirv::StorageClass,
    },

    // ========== EXTENSIONS ==========
    /// An extension requires a newer SPIR-V version than the target environment provides.
    #[error(
        "extension {extension} requires SPIR-V version {required_version}, but target provides {target_version}"
    )]
    ExtensionRequiresSpirvVersion {
        /// The extension name that is too new.
        extension: ExtensionName,
        /// The minimum SPIR-V version required by the extension.
        required_version: SpirvVersion,
        /// The target environment's SPIR-V version.
        target_version: SpirvVersion,
    },
    /// A decoration requires a newer SPIR-V version than the target environment provides.
    #[error(
        "decoration {decoration:?} requires SPIR-V version {required_version}, but target provides {target_version}"
    )]
    DecorationRequiresSpirvVersion {
        /// The decoration that is too new.
        decoration: rspirv::spirv::Decoration,
        /// The minimum SPIR-V version required by the decoration.
        required_version: SpirvVersion,
        /// The target environment's SPIR-V version.
        target_version: SpirvVersion,
    },
    /// A block decoration is applied to a struct that is used in a disallowed storage class.
    #[error(
        "decoration {decoration:?} cannot be used with storage class {storage_class:?} on a block-decorated type"
    )]
    InvalidBlockDecorationStorageClass {
        /// The decoration (`Block` or `BufferBlock`).
        decoration: rspirv::spirv::Decoration,
        /// The disallowed storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// A Binding/DescriptorSet decoration is applied to a disallowed storage class.
    #[error(
        "descriptor decorations are not permitted on storage class {storage_class:?} (expected resource classes)"
    )]
    InvalidDescriptorStorageClass {
        /// The storage class of the decorated variable.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// A resource interface variable in Vulkan is missing a DescriptorSet decoration.
    #[error("resource variable {variable:?} is missing a DescriptorSet decoration")]
    MissingDescriptorSetDecoration {
        /// The variable missing the decoration.
        variable: Id,
    },
    /// A resource interface variable in Vulkan is missing a Binding decoration.
    #[error("resource variable {variable:?} is missing a Binding decoration")]
    MissingBindingDecoration {
        /// The variable missing the decoration.
        variable: Id,
    },
    /// A struct used in a resource storage class is missing the required block decoration.
    #[error("storage class {storage_class:?} requires a Block-decorated struct type")]
    MissingBlockDecoration {
        /// The disallowed storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// A Vulkan buffer variable is missing the required Block decoration.
    #[error("Vulkan {storage_class:?} variable requires a Block-decorated struct (struct id {struct_id})")]
    VulkanBufferMissingBlockDecoration {
        /// The storage class of the variable.
        storage_class: rspirv::spirv::StorageClass,
        /// The struct type ID.
        struct_id: u32,
    },
    /// A Vulkan StorageBuffer variable has a BufferBlock decoration (not allowed).
    #[error(
        "Vulkan StorageBuffer variables cannot use BufferBlock decoration (struct id {struct_id})"
    )]
    VulkanStorageBufferHasBufferBlock {
        /// The struct type ID.
        struct_id: u32,
    },
    /// A Vulkan Uniform variable is missing Block or BufferBlock decoration.
    #[error("Vulkan Uniform variable requires a Block or BufferBlock-decorated struct (struct id {struct_id})")]
    VulkanUniformMissingBlockDecoration {
        /// The struct type ID.
        struct_id: u32,
    },
    /// An OpenGL uniform/storage block variable is missing a Binding decoration (ARB_gl_spirv).
    #[error("{storage_class:?} id {variable_id} is missing Binding decoration. From ARB_gl_spirv extension: Uniform and shader storage block variables must also be decorated with a Binding.")]
    OpenGlBufferMissingBindingDecoration {
        /// The storage class of the variable.
        storage_class: rspirv::spirv::StorageClass,
        /// The variable ID.
        variable_id: u32,
    },
    /// A Block/BufferBlock struct is missing ArrayStride decoration on an array member.
    #[error("Structure id {struct_id} decorated as {decoration_type} must be explicitly laid out with ArrayStride decorations")]
    BlockMissingArrayStride {
        /// The struct type ID.
        struct_id: u32,
        /// The decoration type (Block or BufferBlock).
        decoration_type: &'static str,
    },
    /// A Block/BufferBlock struct is missing MatrixStride decoration on a matrix member.
    #[error("Structure id {struct_id} decorated as {decoration_type} must be explicitly laid out with MatrixStride decorations")]
    BlockMissingMatrixStride {
        /// The struct type ID.
        struct_id: u32,
        /// The decoration type (Block or BufferBlock).
        decoration_type: &'static str,
    },
    /// A Block/BufferBlock struct is missing RowMajor/ColMajor decoration on a matrix member.
    #[error("Structure id {struct_id} decorated as {decoration_type} must be explicitly laid out with RowMajor or ColMajor decorations")]
    BlockMissingMatrixOrder {
        /// The struct type ID.
        struct_id: u32,
        /// The decoration type (Block or BufferBlock).
        decoration_type: &'static str,
    },
    /// A Location/Component decoration is applied to a disallowed storage class.
    #[error("location/component decorations are only permitted on Input/Output storage classes (found {storage_class:?})")]
    InvalidLocationStorageClass {
        /// The storage class of the decorated variable.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// A Component decoration used an out-of-range component value.
    #[error("Component decoration must be in the range [0, 3] (found {component})")]
    ComponentOutOfRange {
        /// The declared component value.
        component: u32,
    },
    /// A Component decoration was applied without a corresponding Location decoration.
    #[error("Component decoration requires a Location decoration on the same id")]
    ComponentMissingLocation,
    /// An interpolation decoration is applied to a disallowed storage class.
    #[error(
        "interpolation decoration {decoration:?} is only permitted on Input/Output storage classes (found {storage_class:?})"
    )]
    InterpolationDecorationInvalidStorageClass {
        /// The interpolation decoration applied.
        decoration: rspirv::spirv::Decoration,
        /// The storage class of the decorated variable.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// An interpolation decoration is used without a fragment entry point.
    #[error("interpolation decoration {decoration:?} requires a Fragment entry point")]
    InterpolationDecorationRequiresFragment {
        /// The interpolation decoration applied.
        decoration: rspirv::spirv::Decoration,
    },
    /// Two interpolation decorations from the same exclusivity class were applied.
    #[error(
        "interpolation decoration {decoration:?} conflicts with existing decoration {existing:?}"
    )]
    InterpolationDecorationConflict {
        /// The decoration being applied.
        decoration: rspirv::spirv::Decoration,
        /// The previously applied decoration in the same exclusivity class.
        existing: rspirv::spirv::Decoration,
    },
    /// A decoration is applied multiple times to the same ID when only one is allowed.
    #[error("decoration {decoration:?} applied multiple times to ID {target}")]
    DuplicateDecorationOnId {
        /// The duplicate decoration.
        decoration: rspirv::spirv::Decoration,
        /// The target ID.
        target: u32,
    },
    /// A member decoration is applied multiple times to the same member when only one is allowed.
    #[error("decoration {decoration:?} applied multiple times to ID {target} member {member}")]
    DuplicateMemberDecoration {
        /// The duplicate decoration.
        decoration: rspirv::spirv::Decoration,
        /// The struct type ID.
        target: u32,
        /// The member index.
        member: u32,
    },
    /// Two mutually exclusive decorations are applied to the same ID.
    #[error("ID {target} decorated with both {decoration1:?} and {decoration2:?} is not allowed")]
    MutuallyExclusiveDecorations {
        /// The first decoration.
        decoration1: rspirv::spirv::Decoration,
        /// The second decoration.
        decoration2: rspirv::spirv::Decoration,
        /// The target ID.
        target: u32,
    },
    /// Two mutually exclusive decorations are applied to the same struct member.
    #[error("ID {target} member {member} decorated with both {decoration1:?} and {decoration2:?} is not allowed")]
    MutuallyExclusiveMemberDecorations {
        /// The first decoration.
        decoration1: rspirv::spirv::Decoration,
        /// The second decoration.
        decoration2: rspirv::spirv::Decoration,
        /// The struct type ID.
        target: u32,
        /// The member index.
        member: u32,
    },
    /// A BuiltIn decoration is used without a fragment entry point when one is required.
    #[error("BuiltIn {builtin:?} requires a Fragment entry point")]
    BuiltInRequiresFragment {
        /// The BuiltIn applied.
        builtin: rspirv::spirv::BuiltIn,
    },
    /// A BuiltIn decoration is used without an allowed execution model being present.
    #[error("BuiltIn {builtin:?} requires one of the following execution models: {allowed:?}")]
    BuiltInRequiresExecutionModel {
        /// The BuiltIn applied.
        builtin: rspirv::spirv::BuiltIn,
        /// The allowed execution models for this BuiltIn.
        allowed: Vec<rspirv::spirv::ExecutionModel>,
    },
    /// A BuiltIn decoration requires a specific execution mode to be declared.
    #[error("BuiltIn {builtin:?} requires execution mode {required_mode:?}")]
    BuiltInRequiresExecutionModeDeclaration {
        /// The BuiltIn applied.
        builtin: rspirv::spirv::BuiltIn,
        /// The required execution mode.
        required_mode: rspirv::spirv::ExecutionMode,
    },
    /// A BuiltIn decoration targets a variable with an incompatible type.
    #[error("BuiltIn {builtin:?} requires type {expected}")]
    InvalidBuiltInType {
        /// The BuiltIn applied.
        builtin: rspirv::spirv::BuiltIn,
        /// A description of the expected type.
        expected: &'static str,
    },
    /// An interpolation decoration is used in an entry point where it is disallowed.
    #[error("decoration {decoration:?} on storage class {storage_class:?} is not allowed for execution model {execution_model:?}")]
    InterpolationDecorationInvalidForEntryPoint {
        /// The decoration applied.
        decoration: rspirv::spirv::Decoration,
        /// The storage class of the decorated variable.
        storage_class: rspirv::spirv::StorageClass,
        /// The disallowed execution model.
        execution_model: rspirv::spirv::ExecutionModel,
    },
    /// Fragment inputs carrying integers or 64-bit floats must be flat shaded.
    #[error("fragment input with integer or 64-bit float type must use the Flat decoration")]
    FragmentInputRequiresFlat,
    /// A decoration requires a capability that was not declared.
    #[error("decoration {decoration:?} requires capability {capability:?}")]
    DecorationRequiresCapability {
        /// The decoration applied.
        decoration: rspirv::spirv::Decoration,
        /// The missing capability.
        capability: rspirv::spirv::Capability,
    },
    /// A BuiltIn decoration requires a capability that was not declared.
    #[error("BuiltIn {builtin:?} requires capability {capability:?}")]
    BuiltInRequiresCapability {
        /// The BuiltIn applied.
        builtin: rspirv::spirv::BuiltIn,
        /// The missing capability.
        capability: rspirv::spirv::Capability,
    },
    /// A BuiltIn decoration is not allowed in the current environment.
    #[error("BuiltIn {builtin:?} is not allowed in target environment {env:?}")]
    BuiltInDisallowedForEnv {
        /// The BuiltIn applied.
        builtin: rspirv::spirv::BuiltIn,
        /// The target environment in which it was used.
        env: TargetEnv,
    },
    /// Tessellation level built-ins must also carry a Patch decoration.
    #[error("BuiltIn {builtin:?} requires a Patch decoration")]
    BuiltInRequiresPatchDecoration {
        /// The BuiltIn applied.
        builtin: rspirv::spirv::BuiltIn,
    },
    /// Location/Component decorations conflict with a BuiltIn decoration on the same id.
    #[error("Location/Component decorations cannot be applied to BuiltIn variables")]
    LocationConflictsWithBuiltIn,
    /// An entry point function has a LinkageAttributes decoration.
    #[error("Entry point {entry_point:?} cannot have LinkageAttributes decoration")]
    EntryPointHasLinkageAttributes {
        /// The entry point that was decorated.
        entry_point: Id,
    },
    /// An interpolation decoration is used on an Input variable in a vertex shader.
    #[error("Interpolation decoration on Input variable {variable:?} is not allowed in vertex shader (entry point {entry_point:?})")]
    InterpolationDecorationInvalidForVertexInput {
        /// The variable with the decoration.
        variable: Id,
        /// The entry point.
        entry_point: Id,
    },
    /// An interpolation decoration is used on an Output variable in a fragment shader.
    #[error("Interpolation decoration on Output variable {variable:?} is not allowed in fragment shader (entry point {entry_point:?})")]
    InterpolationDecorationInvalidForFragmentOutput {
        /// The variable with the decoration.
        variable: Id,
        /// The entry point.
        entry_point: Id,
    },
    /// A BuiltIn decoration is applied to a variable in a disallowed storage class.
    #[error("BuiltIn {builtin:?} cannot be applied to storage class {storage_class:?}")]
    InvalidBuiltInStorageClass {
        /// The built-in kind.
        builtin: rspirv::spirv::BuiltIn,
        /// The storage class of the decorated variable.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// A BuiltIn decoration has the wrong storage class direction for the execution model.
    #[error(
        "BuiltIn {builtin:?} cannot use {storage_class:?} in execution model {execution_model:?}"
    )]
    BuiltInWrongStorageClassForExecutionModel {
        /// The built-in kind.
        builtin: rspirv::spirv::BuiltIn,
        /// The storage class of the decorated variable.
        storage_class: rspirv::spirv::StorageClass,
        /// The execution model that disallows this combination.
        execution_model: rspirv::spirv::ExecutionModel,
    },
    /// `OpSamplerImageAddressingModeNV` was declared more than once.
    #[error("OpSamplerImageAddressingModeNV should only be provided once")]
    DuplicateSamplerImageAddressingMode,
    /// `OpSamplerImageAddressingModeNV` is required when using `BindlessTextureNV`.
    #[error("Missing required OpSamplerImageAddressingModeNV instruction")]
    MissingSamplerImageAddressingMode,
    /// `OpSamplerImageAddressingModeNV` used an invalid bit width.
    #[error("OpSamplerImageAddressingModeNV bitwidth should be 64 or 32 (found {bit_width})")]
    InvalidSamplerImageAddressingModeBitWidth {
        /// The declared bit width.
        bit_width: u32,
    },
    /// `OpSamplerImageAddressingModeNV` requires `BindlessTextureNV` capability.
    #[error(
        "OpSamplerImageAddressingModeNV supported only with extension SPV_NV_bindless_texture"
    )]
    SamplerImageAddressingModeNVRequiresBindlessTextureNV,
    /// Duplicate extension declarations were found.
    #[error("extension {extension} is declared more than once")]
    DuplicateExtension {
        /// The extension name that was duplicated.
        extension: ExtensionName,
    },
    /// An extension was declared that is not permitted in the target environment.
    #[error("extension {extension} is not allowed for target environment {env:?}")]
    DisallowedExtension {
        /// The extension name that is not allowed.
        extension: ExtensionName,
        /// The target environment in use.
        env: TargetEnv,
    },

    // ========== DECORATIONS ==========
    /// `OpExecutionModeId LocalSizeId` is not permitted for the current environment/options.
    #[error("LocalSizeId execution mode is not allowed for target environment {env:?}")]
    LocalSizeIdNotAllowed {
        /// The target environment in use.
        env: TargetEnv,
    },
    /// A function declaration (no basic blocks) must have Import linkage.
    #[error("function declaration {function:?} must have LinkageAttributes decoration with Import linkage type")]
    FunctionDeclarationMissingImportLinkage {
        /// The function ID.
        function: Id,
    },
    /// A function definition (with basic blocks) must not have Import linkage.
    #[error("function definition {function:?} may not be decorated with Import linkage type")]
    FunctionDefinitionHasImportLinkage {
        /// The function ID.
        function: Id,
    },
    /// An imported variable (with Import linkage) cannot be initialized.
    #[error("imported variable {variable:?} cannot have an initialization value")]
    ImportedVariableHasInitializer {
        /// The variable ID.
        variable: Id,
    },
    /// Component decoration value must be within legal range.
    #[error("Component decoration value {component} is out of valid range [0, {max_component})")]
    ComponentDecorationOutOfRange {
        /// The component value.
        component: u32,
        /// The maximum valid component.
        max_component: u32,
    },
    /// Component decoration combined with type width exceeds maximum.
    #[error("Component decoration value {component} combined with type width exceeds maximum (component + width/32 = {total})")]
    ComponentDecorationExceedsWidth {
        /// The component value.
        component: u32,
        /// The calculated total.
        total: u32,
    },
    /// FPRoundingMode decoration is only valid for conversion instructions.
    #[error("FPRoundingMode decoration on {opcode:?} is only valid for conversion instructions targeting 16-bit floats in the StorageBuffer, Uniform, or PhysicalStorageBuffer storage classes")]
    FPRoundingModeInvalidContext {
        /// The opcode being decorated.
        opcode: rspirv::spirv::Op,
    },
    /// FPRoundingMode is only valid for 16-bit float conversions.
    #[error("FPRoundingMode decoration requires 16-bit floating-point result type")]
    FPRoundingModeRequires16BitFloat,
    /// FPRoundingMode decoration requires writing to certain storage classes.
    #[error("FPRoundingMode decoration requires writes to StorageBuffer, Uniform, or PhysicalStorageBuffer storage classes")]
    FPRoundingModeInvalidStorageClass,
    /// FPRoundingMode decoration in Vulkan only allows RTE or RTZ.
    #[error(
        "In Vulkan, FPRoundingMode decoration only allows RTE or RTZ rounding modes, got {mode:?}"
    )]
    FPRoundingModeVulkanInvalidMode {
        /// The invalid rounding mode.
        mode: rspirv::spirv::FPRoundingMode,
    },
    /// Uniform decoration requires uniform control flow.
    #[error("Uniform decoration on {opcode:?} requires uniform control flow")]
    UniformDecorationRequiresUniformControlFlow {
        /// The opcode being decorated.
        opcode: rspirv::spirv::Op,
    },
    /// NoSignedWrap/NoUnsignedWrap decoration only applies to integer operations.
    #[error(
        "NoSignedWrap/NoUnsignedWrap decoration on {opcode:?} is only valid for integer arithmetic"
    )]
    IntegerWrapDecorationInvalidOp {
        /// The opcode being decorated.
        opcode: rspirv::spirv::Op,
    },
    /// RelaxedPrecision decoration is only valid in shader execution models.
    #[error("RelaxedPrecision decoration requires Shader capability")]
    RelaxedPrecisionRequiresShader,
    /// Block decoration applied to non-struct type.
    #[error("Block decoration can only be applied to struct types (found {opcode:?})")]
    BlockDecorationRequiresStruct {
        /// The opcode of the decorated type.
        opcode: rspirv::spirv::Op,
    },
    /// Buffer variables in Vulkan must use Block decoration (not BufferBlock).
    #[error("BufferBlock decoration is deprecated in Vulkan; use Block with StorageBuffer storage class")]
    BufferBlockDeprecatedInVulkan,
    /// Location decoration is applied to invalid target.
    #[error("Location decoration on ID {target:?} is not valid (requires variable with Input/Output storage class)")]
    LocationDecorationInvalidTarget {
        /// The target ID.
        target: Id,
    },
    /// Location values must form a contiguous range within shader interface.
    #[error("Location value {location} creates a gap or overlap in interface locations")]
    LocationValueCreateGap {
        /// The problematic location value.
        location: u32,
    },
    /// NonReadable decoration requires NonWritable or vice versa in some contexts.
    #[error("NonReadable decoration without NonWritable on ID {target:?} in image format that is not read-only")]
    NonReadableWithoutNonWritable {
        /// The target ID.
        target: Id,
    },
    /// Vulkan memory model deprecates certain decorations.
    #[error("Decoration {decoration:?} is deprecated when using the Vulkan memory model; use the memory model's semantics instead")]
    VulkanMemoryModelDeprecatesDecoration {
        /// The deprecated decoration.
        decoration: rspirv::spirv::Decoration,
    },

    // ========== MEMORY ==========
    /// Logical addressing forbids pointers to pointers for the given storage class.
    #[error(
        "In Logical addressing, variables cannot allocate a pointer to storage class {pointee_storage_class:?}"
    )]
    LogicalPointerPointeeStorageClassInvalid {
        /// The offending variable.
        variable: Id,
        /// The disallowed pointee storage class.
        pointee_storage_class: rspirv::spirv::StorageClass,
    },
    /// Logical addressing requires a capability for pointer-to-pointer allocations.
    #[error(
        "In Logical addressing, variables allocating pointers to {pointee_storage_class:?} require capability {required_capability:?}"
    )]
    LogicalPointerMissingCapability {
        /// The offending variable.
        variable: Id,
        /// The pointee storage class in question.
        pointee_storage_class: rspirv::spirv::StorageClass,
        /// The required capability.
        required_capability: rspirv::spirv::Capability,
    },
    /// Logical addressing requires function or private storage for pointer allocations.
    #[error(
        "In Logical addressing with variable pointers, variables allocating pointers must be in Function or Private storage classes (found {storage_class:?})"
    )]
    LogicalPointerInvalidStorageClass {
        /// The offending variable.
        variable: Id,
        /// The variable's storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// In logical addressing, a load/store pointer must come from a logical pointer-producing opcode.
    #[error("Op{instruction:?} Pointer <id> '{pointer}' is not a logical pointer.")]
    NotALogicalPointer {
        /// The load/store instruction.
        instruction: rspirv::spirv::Op,
        /// The pointer operand.
        pointer: Id,
        /// The opcode that produced the pointer (which is not a logical pointer producer).
        source_opcode: rspirv::spirv::Op,
    },
    /// The pointer and object types for an OpStore do not match.
    #[error(
        "OpStore pointer type {pointer_type:?} does not match object type {object_type:?} for pointer {pointer:?}"
    )]
    StoreTypeMismatch {
        /// The pointer being stored through.
        pointer: ResultId,
        /// The pointer's pointee type.
        pointer_type: TypeId,
        /// The stored object's type.
        object_type: TypeId,
    },
    /// Variable pointer points to an array of Block/BufferBlock-decorated structs.
    #[error(
        "Variable pointer {pointer:?} must not point to an array of Block- or BufferBlock-decorated structs"
    )]
    VariablePointerToBlockArray {
        /// The variable pointer instruction.
        pointer: Id,
    },
    /// Variable pointer points to a type containing a matrix.
    #[error(
        "Variable pointer {pointer:?} must not point to an object that is or contains a matrix"
    )]
    VariablePointerToMatrixType {
        /// The variable pointer instruction.
        pointer: Id,
    },
    /// Variable pointer points to a matrix column or component.
    #[error(
        "Variable pointer {pointer:?} must not point to a column or a component of a column of a matrix"
    )]
    VariablePointerToMatrixElement {
        /// The variable pointer instruction.
        pointer: Id,
    },
    /// Variable pointers selected from different buffers without VariablePointers capability.
    #[error(
        "Variable pointers in {pointer:?} must point into the same structure (or OpConstantNull) without VariablePointers capability"
    )]
    VariablePointerDifferentBuffers {
        /// The select/phi instruction producing the variable pointer.
        pointer: Id,
    },
    /// Instruction may not have logical pointer operands.
    #[error("Instruction {opcode:?} may not have a logical pointer operand")]
    LogicalPointerOperandNotAllowed {
        /// The opcode with invalid logical pointer operand.
        opcode: rspirv::spirv::Op,
    },
    /// Instruction requires variable pointer capability for logical pointer operand.
    #[error(
        "Instruction {opcode:?} may only have a logical pointer operand in the StorageBuffer or Workgroup storage classes with appropriate variable pointers capability"
    )]
    LogicalPointerOperandRequiresCapability {
        /// The opcode that requires capability.
        opcode: rspirv::spirv::Op,
    },
    /// Instruction may not return a logical pointer.
    #[error("Instruction {opcode:?} may not return a logical pointer")]
    LogicalPointerReturnNotAllowed {
        /// The opcode that cannot return logical pointer.
        opcode: rspirv::spirv::Op,
    },
    /// Instruction requires variable pointer capability to return logical pointer.
    #[error(
        "Instruction {opcode:?} may only return a logical pointer in the StorageBuffer or Workgroup storage classes with appropriate variable pointers capability"
    )]
    LogicalPointerReturnRequiresCapability {
        /// The opcode that requires capability.
        opcode: rspirv::spirv::Op,
    },
    /// A struct decorated for block layout violates layout rules.
    #[error("Struct {struct_type:?} has an invalid block layout: {reason}")]
    InvalidBlockLayout {
        /// The struct type id.
        struct_type: ResultId,
        /// Human-friendly reason for the failure.
        reason: String,
    },
    /// Image operand Offset is restricted to gather instructions unless explicitly allowed.
    #[error("Image operand Offset for {opcode:?} is only allowed with gather instructions in Vulkan unless the offset texture operand option is enabled")]
    OffsetTextureOperandDisallowed {
        /// The opcode using the restricted operand.
        opcode: rspirv::spirv::Op,
    },
    /// Bitwise operations in Vulkan require 32-bit types unless explicitly allowed.
    #[error(
        "Bitwise opcode {opcode:?} requires 32-bit integer types in Vulkan unless allow_vulkan_32_bit_bitwise is enabled (found {bit_width}-bit type)"
    )]
    VulkanBitwiseRequires32Bit {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
        /// The observed bit width.
        bit_width: u32,
    },
    /// A decoration group reference did not point to a declared group.
    #[error("decoration group {group} is not declared")]
    UnknownDecorationGroup {
        /// The group id that was missing.
        group: Id,
    },
    /// A decoration targeted an id that is not valid for that opcode.
    #[error("decoration target {target} is not defined")]
    MissingDecorationTarget {
        /// The target id that is missing.
        target: Id,
    },
    /// OpVariable result type is not a pointer type.
    #[error("OpVariable {variable:?} result type must be a pointer")]
    VariableResultTypeNotPointer {
        /// The offending variable.
        variable: Id,
    },
    /// OpVariable storage class in operand doesn't match result type.
    #[error("OpVariable {variable:?} storage class {operand_class:?} doesn't match type storage class {type_class:?}")]
    VariableStorageClassMismatch {
        /// The offending variable.
        variable: Id,
        /// Storage class from operand.
        operand_class: rspirv::spirv::StorageClass,
        /// Storage class from type.
        type_class: rspirv::spirv::StorageClass,
    },
    /// OpVariable cannot use Generic storage class.
    #[error("OpVariable {variable:?} cannot use Generic storage class")]
    VariableGenericStorageClass {
        /// The offending variable.
        variable: Id,
    },
    /// OpVariable cannot use PhysicalStorageBuffer storage class.
    #[error("OpVariable {variable:?} cannot use PhysicalStorageBuffer storage class")]
    VariablePhysicalStorageBuffer {
        /// The offending variable.
        variable: Id,
    },
    /// OpVariable pointee type contains bool in storage class that doesn't allow it.
    #[error("OpVariable {variable:?} contains bool type which is not allowed in {storage_class:?} storage class")]
    VariableContainsBool {
        /// The offending variable.
        variable: Id,
        /// The variable's storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// OpVariable initializer not found.
    #[error("OpVariable {variable:?} initializer {initializer:?} not found")]
    VariableInitializerNotFound {
        /// The offending variable.
        variable: Id,
        /// The missing initializer.
        initializer: Id,
    },
    /// OpVariable initializer is not a constant.
    #[error("OpVariable {variable:?} initializer {initializer:?} must be a constant")]
    VariableInitializerNotConstant {
        /// The offending variable.
        variable: Id,
        /// The non-constant initializer.
        initializer: Id,
    },
    /// OpVariable with Input storage class cannot have an initializer.
    #[error("OpVariable {variable:?} with Input storage class cannot have an initializer")]
    VariableInputHasInitializer {
        /// The offending variable.
        variable: Id,
    },
    /// OpLoad pointer operand is not a pointer type.
    #[error("OpLoad pointer operand {pointer:?} is not a pointer type")]
    LoadPointerNotPointerType {
        /// The pointer operand.
        pointer: Id,
    },
    /// OpLoad result type doesn't match pointee type.
    #[error("OpLoad result type {result_type:?} doesn't match pointee type {pointee_type:?}")]
    LoadResultTypeMismatch {
        /// The result type.
        result_type: TypeId,
        /// The expected pointee type.
        pointee_type: TypeId,
    },
    /// Cannot load a runtime array.
    #[error("OpLoad cannot load a runtime array")]
    LoadRuntimeArray,
    /// OpLoad cannot use MakePointerAvailable memory access.
    #[error("OpLoad cannot use MakePointerAvailable memory access")]
    LoadMakePointerAvailable,
    /// MakePointerVisible requires NonPrivatePointer.
    #[error("MakePointerVisible requires NonPrivatePointer memory access")]
    MakeVisibleRequiresNonPrivate,
    /// NonPrivatePointer used with invalid storage class.
    #[error("NonPrivatePointer memory access is not valid for {storage_class:?} storage class")]
    NonPrivatePointerInvalidStorageClass {
        /// The storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// OpStore pointer operand is not a pointer type.
    #[error("OpStore pointer operand {pointer:?} is not a pointer type")]
    StorePointerNotPointerType {
        /// The pointer operand.
        pointer: Id,
    },
    /// OpStore to read-only storage class.
    #[error("OpStore cannot write to {storage_class:?} storage class (pointer {pointer:?})")]
    StoreToReadOnlyStorageClass {
        /// The pointer being stored through.
        pointer: Id,
        /// The read-only storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// OpStore cannot use MakePointerVisible memory access.
    #[error("OpStore cannot use MakePointerVisible memory access")]
    StoreMakePointerVisible,
    /// MakePointerAvailable requires NonPrivatePointer.
    #[error("MakePointerAvailable requires NonPrivatePointer memory access")]
    MakeAvailableRequiresNonPrivate,
    /// PhysicalStorageBuffer requires Aligned memory access (VUID 4708).
    #[error("Memory accesses with PhysicalStorageBuffer must use Aligned (VUID-4708)")]
    PhysicalStorageBufferRequiresAligned,
    /// Memory access Aligned value must be a power of two.
    #[error("Memory access Aligned operand value {value} is not a power of two")]
    AlignedValueNotPowerOfTwo {
        /// The invalid alignment value.
        value: u32,
    },
    /// Memory access Aligned value is too small for the scalar type.
    #[error("Memory access Aligned operand value {alignment} is too small, the largest scalar type is {largest_scalar} bytes (VUID-6314)")]
    AlignedValueTooSmall {
        /// The alignment value.
        alignment: u32,
        /// The size of the largest scalar type in bytes.
        largest_scalar: u32,
    },
    /// OpArrayLength result type is not an integer.
    #[error("OpArrayLength {instruction:?} result type must be an integer")]
    ArrayLengthResultTypeNotInt {
        /// The instruction.
        instruction: Id,
    },
    /// OpArrayLength result type has invalid width.
    #[error("OpArrayLength {instruction:?} result type must be 32 or 64 bits (found {width})")]
    ArrayLengthResultTypeInvalidWidth {
        /// The instruction.
        instruction: Id,
        /// The invalid width.
        width: u32,
    },
    /// OpArrayLength result type is signed.
    #[error("OpArrayLength {instruction:?} result type must be unsigned")]
    ArrayLengthResultTypeSigned {
        /// The instruction.
        instruction: Id,
    },
    /// OpArrayLength structure operand is not a pointer.
    #[error("OpArrayLength {instruction:?} structure operand must be a pointer")]
    ArrayLengthStructureNotPointer {
        /// The instruction.
        instruction: Id,
    },
    /// OpArrayLength structure pointee is not a struct.
    #[error("OpArrayLength {instruction:?} must point to a struct")]
    ArrayLengthPointeeNotStruct {
        /// The instruction.
        instruction: Id,
    },
    /// OpArrayLength member index must be the last member.
    #[error("OpArrayLength {instruction:?} member index {member_index} must be last member {last_member}")]
    ArrayLengthMemberNotLast {
        /// The instruction.
        instruction: Id,
        /// The specified member index.
        member_index: usize,
        /// The last member index.
        last_member: usize,
    },
    /// OpArrayLength last member is not a runtime array.
    #[error("OpArrayLength {instruction:?} last member must be a runtime array")]
    ArrayLengthMemberNotRuntimeArray {
        /// The instruction.
        instruction: Id,
    },
    /// OpCopyMemory operand is not a pointer.
    #[error("OpCopyMemory {operand_name} operand {operand:?} must be a pointer")]
    CopyMemoryOperandNotPointer {
        /// The operand id.
        operand: Id,
        /// Which operand (target or source).
        operand_name: &'static str,
    },
    /// OpCopyMemory types don't match.
    #[error("OpCopyMemory target type {target_type:?} doesn't match source type {source_type:?}")]
    CopyMemoryTypeMismatch {
        /// Target pointee type.
        target_type: TypeId,
        /// Source pointee type.
        source_type: TypeId,
    },
    /// OpCopyMemorySized size is not an integer.
    #[error("OpCopyMemorySized size {size:?} must be an integer")]
    CopyMemorySizeNotInteger {
        /// The size operand.
        size: Id,
    },
    /// OpCopyMemorySized size is zero.
    #[error("OpCopyMemorySized size {size:?} must not be zero")]
    CopyMemorySizeZero {
        /// The size operand.
        size: Id,
    },

    // ========== ENTRY_POINTS ==========
    /// An entry point referenced an undefined id.
    #[error("entry point references unknown id {target}")]
    MissingEntryPointTarget {
        /// The missing target id.
        target: Id,
    },
    /// An entry point operand had the wrong kind.
    #[error("entry point operands are malformed")]
    InvalidEntryPointOperand,
    /// An entry point referenced an id of the wrong kind (non-function or non-interface variable).
    #[error(
        "entry point target {target} has opcode {opcode:?} which is invalid for this position"
    )]
    InvalidEntryPointTarget {
        /// The target id.
        target: Id,
        /// The opcode actually defining that id.
        opcode: rspirv::spirv::Op,
    },
    /// An entry point interface variable uses an invalid storage class.
    #[error(
        "entry point {entry_point:?} interface {interface:?} uses storage class {storage_class:?}"
    )]
    EntryPointInterfaceStorageClassInvalid {
        /// The entry-point function id.
        entry_point: Id,
        /// The interface variable id.
        interface: Id,
        /// The invalid storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// An entry point listed more than one interface variable with a storage class
    /// that must be unique for that entry point (Vulkan environments).
    #[error("entry point {entry_point:?} has more than one interface variable with storage class {storage_class:?}")]
    EntryPointInterfaceStorageClassDuplicate {
        /// The entry-point function id.
        entry_point: Id,
        /// The duplicated storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// Patch-decorated interface variables must be used only with tessellation execution models.
    #[error("Patch interface variables require tessellation execution models (found {execution_model:?})")]
    PatchDecorationRequiresTessellation {
        /// The execution model used by the entry point.
        execution_model: rspirv::spirv::ExecutionModel,
    },
    /// Entry-point interface variables consumed overlapping locations/components.
    #[error(
        "entry point {entry_point:?} has overlapping {storage_class:?} variables at location {location} component {component}: variable {first_var:?} and variable {second_var:?} both use this slot"
    )]
    EntryPointInterfaceLocationConflict {
        /// The entry-point function id.
        entry_point: Id,
        /// The storage class of the conflicting variables.
        storage_class: rspirv::spirv::StorageClass,
        /// The conflicting location.
        location: u32,
        /// The conflicting component.
        component: u32,
        /// The first variable that claimed this location.
        first_var: Id,
        /// The second variable that conflicts with the first.
        second_var: Id,
    },
    /// Execution mode requires a specific execution model and is invalid for this entry point.
    #[error(
        "execution mode {mode:?} requires execution model {allowed_models:?} but entry point {entry_point:?} uses {execution_model:?}"
    )]
    ExecutionModeRequiresExecutionModel {
        /// The entry point id.
        entry_point: Id,
        /// The execution mode being applied.
        mode: rspirv::spirv::ExecutionMode,
        /// The execution model actually used.
        execution_model: rspirv::spirv::ExecutionModel,
        /// Allowed execution models.
        allowed_models: Vec<rspirv::spirv::ExecutionModel>,
    },
    /// Execution mode uses an invalid literal value for the current environment/capabilities.
    #[error("execution mode {mode:?} on entry point {entry_point:?} uses invalid value {value}")]
    InvalidExecutionModeValue {
        /// The entry point id.
        entry_point: Id,
        /// The offending execution mode.
        mode: rspirv::spirv::ExecutionMode,
        /// The invalid literal value.
        value: u32,
    },
    /// An entry point interface variable uses a disallowed floating-point encoding for its storage
    /// class in Vulkan environments.
    #[error(
        "entry point interface {interface:?} in storage class {storage_class:?} uses disallowed floating-point encoding {encoding:?}"
    )]
    EntryPointInterfaceFloatEncodingInvalid {
        /// The interface variable id.
        interface: Id,
        /// The storage class of the interface variable.
        storage_class: rspirv::spirv::StorageClass,
        /// The offending FP encoding.
        encoding: rspirv::spirv::FPEncoding,
    },
    /// An entry point listed the same interface id more than once.
    #[error("entry point {entry_point:?} lists interface id {interface:?} more than once")]
    DuplicateEntryPointInterface {
        /// The entry-point function id.
        entry_point: Id,
        /// The duplicated interface id.
        interface: Id,
    },
    /// An entry point was declared more than once for the same function and execution model.
    #[error("entry point for function {function:?} with execution model {execution_model:?} is declared more than once")]
    DuplicateEntryPoint {
        /// The function id targeted by the entry point.
        function: Id,
        /// The execution model used by the duplicate entry point.
        execution_model: rspirv::spirv::ExecutionModel,
    },
    /// An execution mode targets a function that is not declared as an entry point.
    #[error("execution mode target {function} is not declared as an entry point")]
    ExecutionModeWithoutEntryPoint {
        /// The function targeted by the execution mode.
        function: Id,
    },
    /// A member decoration targeted an id that is not a struct type.
    #[error("member decorations must target struct types (target {target:?})")]
    MemberDecorationTargetNotStruct {
        /// The target id that was not a struct.
        target: MemberDecorationTargetId,
    },
    /// A member decoration referenced an out-of-range struct member.
    #[error(
        "member decoration index {member:?} is out of range for target {target:?} (member count {member_count})"
    )]
    MemberDecorationIndexOutOfRange {
        /// The struct target.
        target: DecorationTargetId,
        /// The member index that was too large.
        member: MemberIndex,
        /// The number of members in the struct.
        member_count: usize,
    },
    /// A decoration that must be applied with `OpMemberDecorate` was used with `OpDecorate`.
    #[error("decoration {decoration:?} must be applied with OpMemberDecorate")]
    MemberOnlyDecorationUsedWithDecorate {
        /// The member-only decoration.
        decoration: rspirv::spirv::Decoration,
    },
    /// A decoration targeted an id with an incompatible opcode.
    #[error(
        "decoration {decoration:?} requires target kind {expected}, but id {target} has opcode {found:?}"
    )]
    InvalidDecorationTargetKind {
        /// The decoration being applied.
        decoration: rspirv::spirv::Decoration,
        /// The target id.
        target: Id,
        /// The opcode defining the target.
        found: rspirv::spirv::Op,
        /// The expected target kind.
        expected: DecorationTargetKind,
    },
    /// An instruction used a zero id where a non-zero id is required.
    #[error("{kind} for {opcode:?} must be non-zero")]
    ZeroId {
        /// The category of id that was zero.
        kind: IdKind,
        /// The opcode containing the invalid id.
        opcode: rspirv::spirv::Op,
    },

    // ========== CFG ==========
    /// A function definition is missing its required entry block label.
    #[error("function {function:?} is missing its entry label")]
    MissingFunctionEntryBlock {
        /// The function missing its entry label.
        function: Id,
    },
    /// A function block is missing a terminating instruction.
    #[error("block {block:?} in function {function:?} is missing a terminator")]
    MissingBlockTerminator {
        /// The function containing the block.
        function: Id,
        /// The block missing a terminator.
        block: Id,
    },
    /// A block contains instructions after its terminator.
    #[error("block {block:?} in function {function:?} has instructions after its terminator")]
    InstructionsAfterTerminator {
        /// The function containing the block.
        function: Id,
        /// The block with stray instructions.
        block: Id,
    },
    /// A terminator references a block that does not exist in the function.
    #[error("block target {target:?} does not exist in function {function:?}")]
    MissingBlockTarget {
        /// The function containing the reference.
        function: Id,
        /// The missing block target.
        target: Id,
    },
    /// A merge instruction was not placed immediately before the block terminator.
    #[error("merge instruction in block {block:?} of function {function:?} must immediately precede the terminator")]
    MergeInstructionNotBeforeTerminator {
        /// The function containing the block.
        function: Id,
        /// The block containing the misplaced merge.
        block: Id,
    },
    /// A block contains multiple merge instructions.
    #[error("block {block:?} in function {function:?} contains multiple merge instructions")]
    DuplicateMergeInstruction {
        /// The function containing the block.
        function: Id,
        /// The block with duplicate merge instructions.
        block: Id,
    },
    /// A merge instruction targets a block that does not exist.
    #[error("{kind} target {target:?} in block {block:?} of function {function:?} does not exist")]
    MergeTargetMissing {
        /// The function containing the merge.
        function: Id,
        /// The block containing the merge.
        block: Id,
        /// Whether the missing target is the merge or continue target.
        kind: MergeTargetKind,
        /// The missing block id.
        target: Id,
    },
    /// A merge instruction targets a block that is not dominated by the header.
    #[error(
        "{kind} target {target:?} in block {block:?} of function {function:?} is not dominated by the header"
    )]
    MergeTargetNotDominated {
        /// The function containing the merge.
        function: Id,
        /// The block containing the merge.
        block: Id,
        /// Whether the target is the merge or continue target.
        kind: MergeTargetKind,
        /// The target block id.
        target: Id,
    },
    /// A merge or continue target aliases its own block.
    #[error("{kind} target {target:?} in block {block:?} of function {function:?} must not be the block itself")]
    MergeTargetIsBlock {
        /// The function containing the merge.
        function: Id,
        /// The block containing the merge.
        block: Id,
        /// Whether the offending target is the merge or continue target.
        kind: MergeTargetKind,
        /// The offending block id.
        target: Id,
    },
    /// A value is used in a block that is not dominated by its definition.
    #[error("value {value:?} in block {block:?} of function {function:?} is not dominated by its definition")]
    ValueNotDominated {
        /// The function containing the use.
        function: Id,
        /// The block where the value is used.
        block: Id,
        /// The value that is not dominated by its definition.
        value: Id,
    },
    /// A phi incoming value is not dominated by its definition along the incoming edge.
    #[error("phi incoming value {value:?} for predecessor {incoming:?} in block {block:?} of function {function:?} is not dominated by its definition")]
    PhiIncomingNotDominated {
        /// The function containing the phi.
        function: Id,
        /// The phi's block.
        block: Id,
        /// The incoming predecessor block.
        incoming: Id,
        /// The incoming value.
        value: Id,
    },
    /// An instruction references an id that is not defined in the module.
    #[error("use of undefined id {id:?}")]
    UndefinedId {
        /// Optional function containing the use.
        function: Option<Id>,
        /// The missing id.
        id: Id,
    },
    /// A function call targets an id that is not a function definition.
    #[error("call target {target:?} in function {function:?} is not a function (found {found:?})")]
    FunctionCallTargetNotFunction {
        /// The function containing the call.
        function: Id,
        /// The target id.
        target: Id,
        /// The opcode that defined the target.
        found: rspirv::spirv::Op,
    },
    /// A function call returns a value with a mismatched type.
    #[error("call in function {function:?} returns type {found:?}, expected {expected:?}")]
    FunctionCallResultTypeMismatch {
        /// The function containing the call.
        function: Id,
        /// The expected return type.
        expected: TypeId,
        /// The found return type.
        found: TypeId,
    },
    /// A function call provides the wrong number of arguments.
    #[error("call in function {function:?} has {found} arguments but callee expects {expected}")]
    FunctionCallArgumentCountMismatch {
        /// The function containing the call.
        function: Id,
        /// The expected argument count.
        expected: usize,
        /// The provided argument count.
        found: usize,
    },
    /// A function call provides an argument of the wrong type.
    #[error("call in function {function:?} expects parameter type {expected:?} but argument {argument:?} has type {found:?}")]
    FunctionCallArgumentTypeMismatch {
        /// The function containing the call.
        function: Id,
        /// The argument id.
        argument: Id,
        /// The expected parameter type.
        expected: TypeId,
        /// The argument's type.
        found: TypeId,
    },
    /// An instruction's result type does not match the expected type for the opcode.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} has result type {found:?} but expected {expected:?}"
    )]
    InstructionResultTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The expected result type.
        expected: TypeId,
        /// The found result type.
        found: TypeId,
    },
    /// An instruction operand has a type that does not match the instruction's result type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} expects operand {operand_index} to have type {expected:?} but found {found:?}"
    )]
    OperandTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The instruction opcode.
        instruction: rspirv::spirv::Op,
        /// The zero-based operand index.
        operand_index: usize,
        /// The expected operand type.
        expected: TypeId,
        /// The found operand type.
        found: TypeId,
    },
    /// Pointer comparison operands must be pointer-typed.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} expects operand {operand_index} to be a pointer type (found {found:?})"
    )]
    PointerComparisonOperandNotPointer {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The instruction opcode.
        instruction: rspirv::spirv::Op,
        /// The zero-based operand index.
        operand_index: usize,
        /// The non-pointer operand type that was provided.
        found: TypeId,
    },
    /// Pointer comparisons require specific capabilities for their storage class.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} on storage class {storage_class:?} requires capability {required_capability:?}"
    )]
    PointerComparisonMissingCapability {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The instruction opcode.
        instruction: rspirv::spirv::Op,
        /// The pointer storage class.
        storage_class: rspirv::spirv::StorageClass,
        /// The required capability.
        required_capability: rspirv::spirv::Capability,
    },
    /// Pointer comparisons are limited to specific storage classes.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} cannot operate on storage class {storage_class:?}"
    )]
    PointerComparisonInvalidStorageClass {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The instruction opcode.
        instruction: rspirv::spirv::Op,
        /// The storage class in use.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// An access chain base is not a pointer type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} requires a pointer base (found {base_type:?})"
    )]
    AccessChainBaseNotPointer {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The non-pointer base type.
        base_type: TypeId,
    },
    /// An access chain result type is not a pointer.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} requires a pointer result type (found {result_type:?})"
    )]
    AccessChainResultTypeNotPointer {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The non-pointer result type.
        result_type: TypeId,
    },
    /// An access chain index must be an integer scalar type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} expects operand {operand_index} to be an integer scalar index but found type {found:?}"
    )]
    AccessChainIndexTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The zero-based operand index for the index.
        operand_index: usize,
        /// The type of the offending operand.
        found: TypeId,
    },
    /// An access chain struct index must be a literal and within bounds.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} indexes struct type {composite_type:?} with invalid index {index} (bound {bound})"
    )]
    AccessChainStructIndexOutOfBounds {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The struct type being indexed.
        composite_type: TypeId,
        /// The provided index value.
        index: u32,
        /// The member bound.
        bound: u32,
    },
    /// A struct index was not a literal number.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} requires a literal struct index for composite type {composite_type:?}"
    )]
    AccessChainStructIndexNotLiteral {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The struct type being indexed.
        composite_type: TypeId,
    },
    /// An access chain targeted a non-composite type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} cannot index non-composite type {composite_type:?}"
    )]
    AccessChainNonCompositeTarget {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The non-composite type.
        composite_type: TypeId,
    },
    /// An access chain index has a negative value (not allowed in logical addressing).
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} has negative index value {value} at operand {operand_index}"
    )]
    AccessChainNegativeIndex {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The operand index.
        operand_index: usize,
        /// The negative value.
        value: i64,
    },
    /// An access chain result type does not match the computed target type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} expects result pointer to {expected:?} but found {found:?}"
    )]
    AccessChainResultTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The expected pointee type.
        expected: TypeId,
        /// The found pointee type.
        found: TypeId,
    },
    /// An access chain result storage class must match the base pointer storage class.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} has result storage class {result_storage_class:?} but base uses {base_storage_class:?}"
    )]
    AccessChainStorageClassMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// Base storage class.
        base_storage_class: rspirv::spirv::StorageClass,
        /// Result storage class.
        result_storage_class: rspirv::spirv::StorageClass,
    },

    // ========== RAW ACCESS CHAIN ==========
    /// OpRawAccessChainNV result type must be OpTypePointer.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?} requires result type to be OpTypePointer")]
    RawAccessChainResultNotPointer {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpRawAccessChainNV storage class must be StorageBuffer, PhysicalStorageBuffer, or Uniform.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?} requires storage class to be StorageBuffer, PhysicalStorageBuffer, or Uniform")]
    RawAccessChainInvalidStorageClass {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpRawAccessChainNV pointed type must not be Array, Matrix, or Struct.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?} must not point to OpTypeArray, OpTypeMatrix, or OpTypeStruct")]
    RawAccessChainInvalidPointedType {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpRawAccessChainNV Stride must be OpConstant.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?} requires Stride to be OpConstant")]
    RawAccessChainStrideNotConstant {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpRawAccessChainNV Stride type must be OpTypeInt.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?} requires Stride type to be OpTypeInt")]
    RawAccessChainStrideNotInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpRawAccessChainNV Index/Offset type must be 32-bit OpTypeInt.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?} requires {operand_name} to be 32-bit OpTypeInt")]
    RawAccessChainOperandNot32BitInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The operand name.
        operand_name: &'static str,
    },
    /// OpRawAccessChainNV Stride must not be zero with per-element robustness.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?} requires Stride to be non-zero when per-element robustness is used")]
    RawAccessChainStrideZeroWithRobustness {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpRawAccessChainNV robustness cannot use PhysicalStorageBuffer.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?} cannot use robustness with PhysicalStorageBuffer storage class")]
    RawAccessChainRobustnessWithPhysicalStorageBuffer {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpRawAccessChainNV per-component and per-element robustness are mutually exclusive.
    #[error("OpRawAccessChainNV in block {block:?} of function {function:?}: per-component and per-element robustness are mutually exclusive")]
    RawAccessChainMutuallyExclusiveRobustness {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ========== PTR ACCESS CHAIN ==========
    /// OpPtrAccessChain Element must be an integer.
    #[error("OpPtrAccessChain in block {block:?} of function {function:?} requires Element to be an integer")]
    PtrAccessChainElementNotInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpPtrAccessChain Element must be 0 for Block/BufferBlock decorated struct.
    #[error("OpPtrAccessChain in block {block:?} of function {function:?} requires Element to be 0 for Block- or BufferBlock-decorated structure")]
    PtrAccessChainElementMustBeZeroForBlock {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpPtrAccessChain Base type must have ArrayStride decoration.
    #[error("OpPtrAccessChain in block {block:?} of function {function:?} requires Base type to be decorated with ArrayStride")]
    PtrAccessChainMissingArrayStride {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpPtrAccessChain Workgroup storage class requires VariablePointers capability.
    #[error("OpPtrAccessChain in block {block:?} of function {function:?} with Workgroup storage class requires VariablePointers capability")]
    PtrAccessChainWorkgroupRequiresVariablePointers {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpPtrAccessChain StorageBuffer storage class requires VariablePointers capability.
    #[error("OpPtrAccessChain in block {block:?} of function {function:?} with StorageBuffer storage class requires VariablePointers or VariablePointersStorageBuffer capability")]
    PtrAccessChainStorageBufferRequiresVariablePointers {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpPtrAccessChain in Vulkan must use Workgroup, StorageBuffer, or PhysicalStorageBuffer.
    #[error("OpPtrAccessChain in block {block:?} of function {function:?} in Vulkan requires Workgroup, StorageBuffer, or PhysicalStorageBuffer storage class")]
    PtrAccessChainInvalidVulkanStorageClass {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ============================================================================
    // Cooperative Matrix Errors
    // ============================================================================
    /// OpCooperativeMatrixLengthKHR/NV result type must be OpTypeInt with width 32 and signedness 0.
    #[error("{op_name} result type must be OpTypeInt with width 32 and signedness 0")]
    CooperativeMatrixLengthResultTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixLengthKHR type operand must be OpTypeCooperativeMatrixKHR.
    #[error("OpCooperativeMatrixLengthKHR type must be OpTypeCooperativeMatrixKHR")]
    CooperativeMatrixLengthKhrTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixLengthNV type operand must be OpTypeCooperativeMatrixNV.
    #[error("OpCooperativeMatrixLengthNV type must be OpTypeCooperativeMatrixNV")]
    CooperativeMatrixLengthNvTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store result type or object type is not a cooperative matrix type.
    #[error("{op_name} {operand_name} is not a cooperative matrix type")]
    CooperativeMatrixLoadStoreTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The operand name (Result Type or Object type).
        operand_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store pointer is not a logical pointer.
    #[error("{op_name} Pointer is not a logical pointer")]
    CooperativeMatrixLoadStorePointerNotLogical {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store pointer type is not a pointer type.
    #[error("{op_name} type for pointer is not a pointer type")]
    CooperativeMatrixLoadStorePointerTypeInvalid {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store storage class is invalid.
    #[error("{op_name} storage class is not Workgroup, StorageBuffer, or PhysicalStorageBuffer")]
    CooperativeMatrixLoadStoreInvalidStorageClass {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store pointer pointee type must be scalar or vector.
    #[error("{op_name} Pointer's Type must be a scalar or vector type")]
    CooperativeMatrixLoadStorePointeeTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store stride operand must be a scalar integer type.
    #[error("{op_name} Stride operand must be a scalar integer type")]
    CooperativeMatrixLoadStoreStrideTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store (NV) column major operand must be a boolean constant.
    #[error("{op_name} ColumnMajor operand must be a boolean constant instruction")]
    CooperativeMatrixLoadStoreColumnMajorMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store (KHR) layout operand must be a 32-bit integer constant.
    #[error("{op_name} MemoryLayout operand must be a 32-bit integer constant instruction")]
    CooperativeMatrixLoadStoreLayoutMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix load/store (KHR) memory layout requires a stride.
    #[error("{op_name} MemoryLayout {layout} requires a Stride")]
    CooperativeMatrixLoadStoreLayoutRequiresStride {
        /// The opcode name.
        op_name: &'static str,
        /// The layout value.
        layout: u64,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ============================================================================
    // Cooperative Matrix MulAdd Errors
    // ============================================================================
    /// Cooperative matrix MulAdd operand is not a cooperative matrix type.
    #[error("{op_name} {operand_name} is not a cooperative matrix type")]
    CooperativeMatrixMulAddTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The operand name (A, B, C, or Result Type).
        operand_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix MulAdd scope mismatch.
    #[error("{op_name} cooperative matrix scopes must match")]
    CooperativeMatrixMulAddScopeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix MulAdd M dimension mismatch.
    #[error("{op_name} cooperative matrix 'M' dimension mismatch")]
    CooperativeMatrixMulAddMDimensionMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix MulAdd N dimension mismatch.
    #[error("{op_name} cooperative matrix 'N' dimension mismatch")]
    CooperativeMatrixMulAddNDimensionMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix MulAdd K dimension mismatch.
    #[error("{op_name} cooperative matrix 'K' dimension mismatch")]
    CooperativeMatrixMulAddKDimensionMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixPerElementOpNV Function operand is not a function.
    #[error("OpCooperativeMatrixPerElementOpNV Function <id> {function_id:?} is not a function")]
    CooperativeMatrixPerElementOpFunctionNotFunction {
        /// The function operand ID.
        function_id: Id,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixPerElementOpNV Matrix operand is not a cooperative matrix.
    #[error(
        "OpCooperativeMatrixPerElementOpNV Matrix <id> {matrix_id:?} is not a cooperative matrix"
    )]
    CooperativeMatrixPerElementOpMatrixNotCooperative {
        /// The matrix operand ID.
        matrix_id: Id,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixPerElementOpNV Result Type must match matrix type.
    #[error("OpCooperativeMatrixPerElementOpNV Result Type <id> {result_type_id:?} must match matrix type <id> {matrix_type_id:?}")]
    CooperativeMatrixPerElementOpResultTypeMismatch {
        /// The result type ID.
        result_type_id: Id,
        /// The matrix type ID.
        matrix_type_id: Id,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixPerElementOpNV function return type must match matrix component type.
    #[error("OpCooperativeMatrixPerElementOpNV function return type <id> {return_type_id:?} must match matrix component type <id> {component_type_id:?}")]
    CooperativeMatrixPerElementOpReturnTypeMismatch {
        /// The function return type ID.
        return_type_id: Id,
        /// The matrix component type ID.
        component_type_id: Id,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixPerElementOpNV function type must have at least three parameters.
    #[error("OpCooperativeMatrixPerElementOpNV function type <id> {function_type_id:?} must have at least three parameters")]
    CooperativeMatrixPerElementOpTooFewParameters {
        /// The function type ID.
        function_type_id: Id,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixPerElementOpNV function type first parameter must be a 32-bit integer.
    #[error("OpCooperativeMatrixPerElementOpNV function type first parameter type <id> {param_type_id:?} must be a 32-bit integer")]
    CooperativeMatrixPerElementOpParam0Not32BitInt {
        /// The parameter type ID.
        param_type_id: Id,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixPerElementOpNV function type second parameter must be a 32-bit integer.
    #[error("OpCooperativeMatrixPerElementOpNV function type second parameter type <id> {param_type_id:?} must be a 32-bit integer")]
    CooperativeMatrixPerElementOpParam1Not32BitInt {
        /// The parameter type ID.
        param_type_id: Id,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCooperativeMatrixPerElementOpNV function type third parameter must match matrix component type.
    #[error("OpCooperativeMatrixPerElementOpNV function type third parameter type <id> {param_type_id:?} must match matrix component type")]
    CooperativeMatrixPerElementOpParam2TypeMismatch {
        /// The parameter type ID.
        param_type_id: Id,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative matrix conversion type mismatch - both operands must be cooperative matrix types.
    #[error("{opcode:?} in function {function:?} block {block:?}: both operands must be cooperative matrix types (or both must be non-cooperative matrix types)")]
    CooperativeMatrixConversionTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The conversion opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Cooperative matrix conversion shape mismatch.
    #[error("{opcode:?} in function {function:?} block {block:?}: cooperative matrix {dimension} does not match between result and input types")]
    CooperativeMatrixShapeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The conversion opcode.
        opcode: rspirv::spirv::Op,
        /// The dimension that doesn't match (scope, rows, columns, or use).
        dimension: &'static str,
    },

    // ============================================================================
    // Cooperative Vector Errors
    // ============================================================================
    /// Cooperative vector load/store result type or object type is not a cooperative vector type.
    #[error("{op_name} {operand_name} is not a cooperative vector type")]
    CooperativeVectorLoadStoreTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The operand name (Result Type or Object type).
        operand_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector pointer is not a logical pointer.
    #[error("{op_name} Pointer is not a logical pointer")]
    CooperativeVectorPointerNotLogical {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector pointer type is not a pointer type.
    #[error("{op_name} type for pointer is not a pointer type")]
    CooperativeVectorPointerTypeInvalid {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector storage class is invalid.
    #[error("{op_name} storage class is not Workgroup, StorageBuffer, or PhysicalStorageBuffer")]
    CooperativeVectorInvalidStorageClass {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector pointer pointee type must be an array type.
    #[error("{op_name} Pointer's Type must be an array type")]
    CooperativeVectorPointeeTypeNotArray {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector pointer array element type must be scalar or vector.
    #[error("{op_name} Pointer's Type must be an array of scalar or vector type")]
    CooperativeVectorArrayElementTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector offset operand must be 32 or 64-bit integer.
    #[error("{op_name} Offset operand must be a 32 or 64-bit integer type")]
    CooperativeVectorOffsetTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector outer product A operand is not a cooperative vector type.
    #[error("{op_name} A type is not a cooperative vector type")]
    CooperativeVectorOuterProductATypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector outer product B operand is not a cooperative vector type.
    #[error("{op_name} B type is not a cooperative vector type")]
    CooperativeVectorOuterProductBTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector outer product A and B component types do not match.
    #[error("{op_name} A and B component types do not match")]
    CooperativeVectorOuterProductComponentTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector reduce sum V operand is not a cooperative vector type.
    #[error("{op_name} V type is not a cooperative vector type")]
    CooperativeVectorReduceSumTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector matrix mul result type is not a cooperative vector type.
    #[error("{op_name} result type is not a cooperative vector type")]
    CooperativeVectorMatrixMulResultTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector matrix mul result component type is invalid.
    #[error("{op_name} result component type is not a 32-bit int or 16/32-bit float")]
    CooperativeVectorMatrixMulResultComponentTypeMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector matrix mul M does not match result type component count.
    #[error(
        "{op_name} result type number of components {result_components} does not match M {m_value}"
    )]
    CooperativeVectorMatrixMulMDimensionMismatch {
        /// The opcode name.
        op_name: &'static str,
        /// The result type number of components.
        result_components: u32,
        /// The M value.
        m_value: u32,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Cooperative vector operand must be 32-bit integer.
    #[error("{op_name} {operand_name} operand must be a 32-bit integer")]
    CooperativeVectorOperandMustBe32BitInt {
        /// The opcode name.
        op_name: &'static str,
        /// The operand name.
        operand_name: &'static str,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ============================================================================
    // CompositeConstruct Errors
    // ============================================================================
    /// OpCompositeConstruct result type is not a composite type.
    #[error("OpCompositeConstruct result type must be a composite type (vector, matrix, array, or struct)")]
    CompositeConstructResultTypeInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct for vectors requires at least 2 constituents.
    #[error("OpCompositeConstruct for vector requires at least 2 constituents")]
    CompositeConstructVectorTooFewConstituents {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct constituent type mismatch for vector.
    #[error("OpCompositeConstruct constituent must be scalar or vector of the same component type as result")]
    CompositeConstructVectorConstituentTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct total components don't match vector size.
    #[error("OpCompositeConstruct total constituent components ({given}) must equal result vector size ({expected})")]
    CompositeConstructVectorComponentCountMismatch {
        /// Expected component count.
        expected: u32,
        /// Given component count.
        given: u32,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct column count mismatch for matrix.
    #[error("OpCompositeConstruct constituent count ({given}) must equal matrix column count ({expected})")]
    CompositeConstructMatrixColumnCountMismatch {
        /// Expected column count.
        expected: u32,
        /// Given column count.
        given: u32,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct constituent type mismatch for matrix.
    #[error("OpCompositeConstruct constituent type must match matrix column type")]
    CompositeConstructMatrixConstituentTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct element count mismatch for array.
    #[error("OpCompositeConstruct constituent count ({given}) must equal array element count ({expected})")]
    CompositeConstructArrayElementCountMismatch {
        /// Expected element count.
        expected: u64,
        /// Given element count.
        given: u64,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct constituent type mismatch for array.
    #[error("OpCompositeConstruct constituent type must match array element type")]
    CompositeConstructArrayConstituentTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct member count mismatch for struct.
    #[error("OpCompositeConstruct constituent count ({given}) must equal struct member count ({expected})")]
    CompositeConstructStructMemberCountMismatch {
        /// Expected member count.
        expected: u32,
        /// Given member count.
        given: u32,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct constituent type mismatch for struct member.
    #[error(
        "OpCompositeConstruct constituent type must match struct member type at index {index}"
    )]
    CompositeConstructStructConstituentTypeMismatch {
        /// The index of the mismatched member.
        index: u32,
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct for cooperative matrix must have exactly one constituent.
    #[error("OpCompositeConstruct for cooperative matrix must have exactly one constituent")]
    CompositeConstructCoopMatrixSingleConstituent {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCompositeConstruct cooperative matrix constituent type mismatch.
    #[error("OpCompositeConstruct constituent type must match cooperative matrix component type")]
    CompositeConstructCoopMatrixConstituentTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ============================================================================
    // CopyLogical Errors
    // ============================================================================
    /// OpCopyLogical result type must not equal operand type.
    #[error("OpCopyLogical result type must not equal the operand type")]
    CopyLogicalTypesEqual {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCopyLogical types do not logically match.
    #[error("OpCopyLogical result type does not logically match the operand type")]
    CopyLogicalTypesNotLogicallyMatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpCopyLogical cannot copy composites of 8/16-bit types with Shader capability.
    #[error("Cannot copy composites of 8- or 16-bit types")]
    CopyLogicalSmallTypeRestriction {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// A composite instruction requires at least one index operand.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} requires at least one index"
    )]
    CompositeInstructionMissingIndexes {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
    },
    /// A composite operand is not a composite type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} requires a composite operand (found {composite_type:?})"
    )]
    CompositeOperandNotComposite {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The non-composite operand type.
        composite_type: TypeId,
    },
    /// A composite index is out of bounds for the chosen member.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} indexes beyond composite type {composite_type:?} at position {index_position} (index {index}, bound {bound})"
    )]
    CompositeIndexOutOfBounds {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The composite type being indexed.
        composite_type: TypeId,
        /// The zero-based index position within the instruction.
        index_position: usize,
        /// The index value that was out of bounds.
        index: u32,
        /// The bound for that position.
        bound: u32,
    },
    /// OpCompositeExtract or OpCompositeInsert requires at least one index.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} requires at least one index"
    )]
    CompositeExtractInsertNoIndices {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
    },
    /// OpCompositeExtract or OpCompositeInsert has too many indices (max 255).
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} has {count} indices, exceeding the maximum of 255"
    )]
    CompositeExtractInsertTooManyIndices {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The number of indices found.
        count: usize,
    },
    /// A phi incoming value's type does not match the phi's result type.
    #[error("phi in block {block:?} of function {function:?} expects type {expected:?} but incoming value {incoming:?} has type {found:?}")]
    PhiIncomingTypeMismatch {
        /// The function containing the phi.
        function: Id,
        /// The phi's block.
        block: Id,
        /// The incoming value id.
        incoming: Id,
        /// The expected phi result type.
        expected: TypeId,
        /// The incoming value's type.
        found: TypeId,
    },

    // ========== TYPES ==========
    /// Duplicate non-aggregate type declaration.
    #[error("Duplicate non-aggregate type declarations are not allowed. Opcode: {opcode:?} id: {type_id:?}")]
    DuplicateTypeDeclaration {
        /// The opcode of the duplicate type.
        opcode: rspirv::spirv::Op,
        /// The ID of the duplicate type.
        type_id: TypeId,
    },
    /// OpTypeInt uses 8-bit width without Int8 capability.
    #[error("Using an 8-bit integer type requires the Int8 capability, or an extension that explicitly enables 8-bit integers")]
    TypeIntRequiresInt8Capability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeInt uses 16-bit width without Int16 capability.
    #[error("Using a 16-bit integer type requires the Int16 capability, or an extension that explicitly enables 16-bit integers")]
    TypeIntRequiresInt16Capability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeInt uses 64-bit width without Int64 capability.
    #[error("Using a 64-bit integer type requires the Int64 capability")]
    TypeIntRequiresInt64Capability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeInt uses an invalid bit width.
    #[error("Invalid number of bits ({width}) used for OpTypeInt")]
    TypeIntInvalidBitWidth {
        /// The type ID.
        type_id: TypeId,
        /// The invalid bit width.
        width: u32,
    },
    /// OpTypeInt has invalid signedness value.
    #[error("OpTypeInt has invalid signedness value {signedness} (must be 0 or 1)")]
    TypeIntInvalidSignedness {
        /// The type ID.
        type_id: TypeId,
        /// The invalid signedness value.
        signedness: u32,
    },
    /// OpTypeInt with Kernel capability must have signedness 0.
    #[error("The Signedness in OpTypeInt must always be 0 when Kernel capability is used")]
    TypeIntKernelRequiresUnsigned {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeFloat uses 16-bit width without Float16/Float16Buffer capability.
    #[error("Using a 16-bit floating point type requires the Float16 or Float16Buffer capability, or an extension that explicitly enables 16-bit floating point")]
    TypeFloatRequiresFloat16Capability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeFloat uses 64-bit width without Float64 capability.
    #[error("Using a 64-bit floating point type requires the Float64 capability")]
    TypeFloatRequiresFloat64Capability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeFloat uses an invalid bit width.
    #[error("Invalid number of bits ({width}) used for OpTypeFloat")]
    TypeFloatInvalidBitWidth {
        /// The type ID.
        type_id: TypeId,
        /// The invalid bit width.
        width: u32,
    },
    /// OpTypeFloat uses 8-bit width without Float8EXT capability.
    #[error("Using an 8-bit floating point type requires the Float8EXT capability")]
    TypeFloatRequiresFloat8Capability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeFloat uses 8-bit width without FPEncoding operand.
    #[error("8-bit floating point type requires an encoding")]
    TypeFloat8RequiresEncoding {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeFloat uses 8-bit width with unsupported FPEncoding.
    #[error("Unsupported 8-bit floating point encoding ({encoding:?})")]
    TypeFloat8UnsupportedEncoding {
        /// The type ID.
        type_id: TypeId,
        /// The unsupported encoding.
        encoding: rspirv::spirv::FPEncoding,
    },
    /// OpTypeVector component is not a scalar type.
    #[error("OpTypeVector Component Type is not a scalar type")]
    TypeVectorComponentNotScalar {
        /// The type ID.
        type_id: TypeId,
        /// The component type ID.
        component_type: TypeId,
    },
    /// OpTypeVector component is not a scalar or pointer type (with MaskedGatherScatterINTEL).
    #[error(
        "OpTypeVector Component Type is not a scalar or pointer type (MaskedGatherScatterINTEL)"
    )]
    TypeVectorComponentNotScalarOrPointer {
        /// The type ID.
        type_id: TypeId,
        /// The component type ID.
        component_type: TypeId,
    },
    /// OpTypeVector uses 8 or 16 components without Vector16 capability.
    #[error(
        "Having {component_count} components for OpTypeVector requires the Vector16 capability"
    )]
    TypeVectorRequiresVector16Capability {
        /// The type ID.
        type_id: TypeId,
        /// The number of components.
        component_count: u32,
    },
    /// OpTypeVector uses an invalid component count.
    #[error("Illegal number of components ({component_count}) for OpTypeVector")]
    TypeVectorInvalidComponentCount {
        /// The type ID.
        type_id: TypeId,
        /// The invalid component count.
        component_count: u32,
    },
    /// OpTypeMatrix column type is not a vector type.
    #[error("Columns in a matrix must be of type vector")]
    TypeMatrixColumnNotVector {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeMatrix column vector component type is not a float type.
    #[error("Matrix types can only be parameterized with floating-point types")]
    TypeMatrixComponentNotFloat {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeMatrix uses an invalid column count.
    #[error("Matrix types can only be parameterized as having only 2, 3, or 4 columns (found {column_count})")]
    TypeMatrixInvalidColumnCount {
        /// The type ID.
        type_id: TypeId,
        /// The invalid column count.
        column_count: u32,
    },
    /// OpTypeArray element type is void.
    #[error("OpTypeArray Element Type is a void type")]
    TypeArrayElementVoid {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeArray length is not a constant.
    #[error("OpTypeArray Length is not a scalar constant type")]
    TypeArrayLengthNotConstant {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeArray length is not an integer constant.
    #[error("OpTypeArray Length is not a constant integer type")]
    TypeArrayLengthNotInteger {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeArray length is zero or negative.
    #[error("OpTypeArray Length must be at least 1 (found {length})")]
    TypeArrayLengthInvalid {
        /// The type ID.
        type_id: TypeId,
        /// The invalid length value.
        length: i64,
    },
    /// OpTypeRuntimeArray element type is void.
    #[error("OpTypeRuntimeArray Element Type is a void type")]
    TypeRuntimeArrayElementVoid {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeStruct member is a self-reference.
    #[error("Structure members may not be self references")]
    TypeStructMemberSelfReference {
        /// The type ID of the struct.
        type_id: TypeId,
    },
    /// OpTypeStruct member is not a type.
    #[error("OpTypeStruct Member Type <id> {member_type:?} is not a type")]
    TypeStructMemberNotType {
        /// The type ID of the struct.
        type_id: TypeId,
        /// The invalid member type ID.
        member_type: TypeId,
    },
    /// OpTypeStruct member is void type.
    #[error("Structures cannot contain a void type")]
    TypeStructMemberVoid {
        /// The type ID of the struct.
        type_id: TypeId,
    },
    /// OpTypeStruct contains a struct member with BuiltIn decoration.
    #[error("Structure contains members with BuiltIn decoration and therefore may not be contained as a member of another structure type")]
    TypeStructContainsBuiltInStruct {
        /// The type ID of the struct.
        type_id: TypeId,
        /// The nested struct member type ID with BuiltIn decoration.
        member_type: TypeId,
    },
    /// OpTypeRuntimeArray is not the last member of the struct (Vulkan).
    #[error(
        "In Vulkan, OpTypeRuntimeArray must only be used for the last member of an OpTypeStruct"
    )]
    TypeStructRuntimeArrayNotLast {
        /// The type ID of the struct.
        type_id: TypeId,
    },
    /// OpTypeStruct with OpTypeRuntimeArray lacks Block/BufferBlock decoration (Vulkan).
    #[error("In Vulkan, OpTypeStruct containing an OpTypeRuntimeArray must be decorated with Block or BufferBlock")]
    TypeStructRuntimeArrayNoBlockDecoration {
        /// The type ID of the struct.
        type_id: TypeId,
    },
    /// Block/BufferBlock struct nested within another Block/BufferBlock.
    #[error("A Block or BufferBlock cannot be nested within another Block or BufferBlock")]
    TypeStructNestedBlockOrBufferBlock {
        /// The type ID of the struct.
        type_id: TypeId,
    },
    /// BuiltIn decoration not applied to all struct members.
    #[error("When BuiltIn decoration is applied to a structure-type member, all members of that structure type must also be decorated with BuiltIn")]
    TypeStructBuiltInNotAllMembers {
        /// The type ID of the struct.
        type_id: TypeId,
        /// Number of members with BuiltIn decoration.
        builtin_count: usize,
        /// Total number of members.
        total_count: usize,
    },
    /// OpTypeStruct contains an opaque type (Vulkan).
    #[error("In Vulkan, OpTypeStruct must not contain an opaque type")]
    TypeStructContainsOpaqueType {
        /// The type ID of the struct.
        type_id: TypeId,
    },
    /// OpTypePointer Type operand is not a type.
    #[error("OpTypePointer Type <id> {pointee_type:?} is not a type")]
    TypePointerTypeNotType {
        /// The type ID of the pointer.
        type_id: TypeId,
        /// The invalid pointee type ID.
        pointee_type: TypeId,
    },
    /// OpTypePointer uses invalid storage class for target environment.
    #[error("Invalid storage class {storage_class:?} for target environment")]
    TypePointerInvalidStorageClass {
        /// The type ID of the pointer.
        type_id: TypeId,
        /// The invalid storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// OpTypeForwardPointer target is not an OpTypePointer.
    #[error("Pointer type in OpTypeForwardPointer is not a pointer type")]
    ForwardPointerNotPointerType {
        /// The invalid target type ID.
        target_type: TypeId,
    },
    /// OpTypeForwardPointer storage class does not match pointer definition.
    #[error("Storage class in OpTypeForwardPointer does not match the pointer definition")]
    ForwardPointerStorageClassMismatch {
        /// The target type ID.
        target_type: TypeId,
        /// Storage class in forward pointer.
        forward_storage_class: rspirv::spirv::StorageClass,
        /// Storage class in pointer definition.
        pointer_storage_class: rspirv::spirv::StorageClass,
    },
    /// OpTypeForwardPointer must point to a structure.
    #[error("Forward pointers must point to a structure")]
    ForwardPointerNotPointingToStruct {
        /// The target type ID.
        target_type: TypeId,
    },
    /// OpTypeForwardPointer storage class must be PhysicalStorageBuffer in Vulkan.
    #[error("In Vulkan, OpTypeForwardPointer must have a storage class of PhysicalStorageBuffer")]
    ForwardPointerRequiresPhysicalStorageBuffer {
        /// The target type ID.
        target_type: TypeId,
        /// The actual storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// A result type refers to an instruction that is not a type declaration.
    #[error("instruction {instruction:?} has result type {result_type:?} defined by non-type opcode {found:?}")]
    ResultTypeNotType {
        /// The opcode of the instruction with the invalid type.
        instruction: rspirv::spirv::Op,
        /// The result type id.
        result_type: Id,
        /// The opcode that defined the result type id.
        found: rspirv::spirv::Op,
    },
    /// An operand refers to a value defined in another function.
    #[error("value {value:?} used in function {function:?} is defined in a different function")]
    ValueDefinedInAnotherFunction {
        /// The function containing the use.
        function: Id,
        /// The value that is defined elsewhere.
        value: Id,
    },
    /// A function-scoped variable uses a non-function storage class.
    #[error("variable {variable:?} in function {function:?} must use Function storage class (found {storage_class:?})")]
    FunctionVariableStorageClassMismatch {
        /// The function containing the variable.
        function: Id,
        /// The variable id.
        variable: Id,
        /// The provided storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// A function-scoped variable is not declared in the entry block.
    #[error("variable {variable:?} in function {function:?} must be declared in the entry block")]
    FunctionVariableNotInEntryBlock {
        /// The function containing the variable.
        function: Id,
        /// The variable id.
        variable: Id,
    },
    /// A merge instruction is paired with an invalid terminator.
    #[error("block {block:?} of function {function:?} has merge paired with invalid terminator {terminator:?}")]
    InvalidMergeTerminator {
        /// The function containing the merge.
        function: Id,
        /// The block containing the merge.
        block: Id,
        /// The unexpected terminator opcode.
        terminator: rspirv::spirv::Op,
    },
    /// A loop merge specifies the same block for merge and continue targets.
    #[error("loop merge in block {block:?} of function {function:?} uses the same target for merge and continue")]
    ContinueTargetMatchesMerge {
        /// The function containing the loop.
        function: Id,
        /// The block containing the loop merge.
        block: Id,
        /// The shared target id.
        target: Id,
    },
    /// A loop header has no back edge from the continue block to the header.
    #[error("loop header {header:?} in function {function:?} has no back edge from continue block {continue_target:?}")]
    LoopMissingBackEdge {
        /// The function containing the loop.
        function: Id,
        /// The loop header block.
        header: Id,
        /// The continue target block.
        continue_target: Id,
    },
    /// The continue block is not reachable from the loop header.
    #[error("continue block {continue_target:?} in function {function:?} is not reachable from loop header {header:?}")]
    ContinueNotReachable {
        /// The function containing the loop.
        function: Id,
        /// The loop header block.
        header: Id,
        /// The continue target block.
        continue_target: Id,
    },
    /// Unroll and DontUnroll loop controls are both specified.
    #[error("Unroll and DontUnroll loop controls must not both be specified")]
    LoopControlUnrollAndDontUnroll {
        /// The function containing the loop.
        function: Id,
        /// The block containing the loop merge.
        block: Id,
    },
    /// PeelCount and DontUnroll loop controls are both specified.
    #[error("PeelCount and DontUnroll loop controls must not both be specified")]
    LoopControlPeelCountAndDontUnroll {
        /// The function containing the loop.
        function: Id,
        /// The block containing the loop merge.
        block: Id,
    },
    /// PartialCount and DontUnroll loop controls are both specified.
    #[error("PartialCount and DontUnroll loop controls must not both be specified")]
    LoopControlPartialCountAndDontUnroll {
        /// The function containing the loop.
        function: Id,
        /// The block containing the loop merge.
        block: Id,
    },
    /// IterationMultiple loop control operand must be greater than zero.
    #[error("IterationMultiple loop control operand must be greater than zero")]
    LoopControlIterationMultipleZero {
        /// The function containing the loop.
        function: Id,
        /// The block containing the loop merge.
        block: Id,
    },
    /// Flatten and DontFlatten selection controls are both specified.
    #[error("Flatten and DontFlatten selection controls must not both be specified")]
    SelectionControlFlattenAndDontFlatten {
        /// The function containing the selection.
        function: Id,
        /// The block containing the selection merge.
        block: Id,
    },
    /// Inline and DontInline function controls are both specified.
    #[error("Inline and DontInline function controls must not both be specified")]
    FunctionControlInlineAndDontInline {
        /// The function with conflicting controls.
        function: Id,
    },
    /// A structured terminator is missing a required selection merge.
    #[error("block {block:?} of function {function:?} ends with {terminator:?} but lacks a selection merge")]
    MissingSelectionMerge {
        /// The function containing the block.
        function: Id,
        /// The block with the structured terminator.
        block: Id,
        /// The structured terminator opcode.
        terminator: rspirv::spirv::Op,
    },
    /// A basic block is missing its required `OpLabel`.
    #[error("function {function:?} contains a block without OpLabel at index {block_index}")]
    MissingBlockLabel {
        /// The function containing the malformed block.
        function: Id,
        /// The ordinal of the block within the function.
        block_index: usize,
    },
    /// A phi instruction appears after non-phi instructions in a block.
    #[error(
        "block {block:?} of function {function:?} has phi instructions after non-phi instructions"
    )]
    PhiAfterNonPhi {
        /// The function containing the block.
        function: Id,
        /// The block containing the misordered phi.
        block: Id,
    },
    /// OpPhi must not have void result type.
    #[error("OpPhi must not have void result type")]
    PhiVoidResultType {
        /// The function containing the phi.
        function: Id,
        /// The block containing the phi.
        block: Id,
    },
    /// OpPhi with pointer type requires VariablePointers capability in logical addressing.
    #[error("Using pointers with OpPhi requires capability VariablePointers or VariablePointersStorageBuffer")]
    PhiPointerRequiresVariablePointers {
        /// The function containing the phi.
        function: Id,
        /// The block containing the phi.
        block: Id,
    },
    /// OpPhi cannot have sampled image, image, or sampler result type.
    #[error("OpPhi result type cannot be {type_opcode:?}")]
    PhiInvalidResultType {
        /// The function containing the phi.
        function: Id,
        /// The block containing the phi.
        block: Id,
        /// The invalid result type opcode.
        type_opcode: rspirv::spirv::Op,
    },
    /// In SPIR-V 1.6 or later, BranchConditional True Label and False Label must be different.
    #[error("In SPIR-V 1.6 or later, True Label and False Label must be different labels")]
    BranchConditionalSameLabels {
        /// The function containing the branch.
        function: Id,
        /// The block containing the branch.
        block: Id,
    },
    /// In MaximallyReconvergesKHR execution mode, BranchConditional True Label and False Label must be different.
    #[error("In entry points using the MaximallyReconvergesKHR execution mode, True Label and False Label must be different labels")]
    BranchConditionalSameLabelsMaximalReconvergence {
        /// The function containing the branch.
        function: Id,
        /// The block containing the branch.
        block: Id,
    },
    /// In MaximallyReconvergesKHR execution mode, blocks cannot have multiple unique predecessors (except loop headers, merge targets, and switch targets).
    #[error("In entry points using the MaximallyReconvergesKHR execution mode, block {block:?} in function {function:?} must not have multiple unique predecessors")]
    MaximalReconvergenceMultiplePredecessors {
        /// The function containing the block.
        function: Option<Id>,
        /// The block with invalid multiple predecessors.
        block: Option<Id>,
    },
    /// A phi instruction has an unexpected number of incoming predecessors.
    #[error(
        "phi in block {block:?} of function {function:?} has {found} incoming values, expected {expected}"
    )]
    PhiPredecessorCountMismatch {
        /// The function containing the phi.
        function: Id,
        /// The block containing the phi.
        block: Id,
        /// The expected predecessor count.
        expected: usize,
        /// The number of incoming pairs provided by the phi.
        found: usize,
    },
    /// The entry block must not have predecessors.
    #[error("entry block {entry:?} in function {function:?} must not have predecessors")]
    EntryBlockHasPredecessor {
        /// The function containing the entry block.
        function: Id,
        /// The entry block id.
        entry: Id,
    },
    /// A function declaration appeared after a function definition.
    #[error("function declaration {function:?} appears after a function definition")]
    FunctionDeclarationAfterDefinition {
        /// The declared function id.
        function: Id,
    },
    /// A phi node references a block that does not exist in the function.
    #[error(
        "phi in block {block:?} of function {function:?} references missing block {incoming:?}"
    )]
    PhiIncomingBlockMissing {
        /// The function containing the phi.
        function: Id,
        /// The block containing the phi.
        block: Id,
        /// The missing incoming block id.
        incoming: Id,
    },
    /// A phi node lists an incoming block that is not a predecessor of the current block.
    #[error(
        "phi in block {block:?} of function {function:?} lists non-predecessor block {incoming:?}"
    )]
    PhiIncomingNotPredecessor {
        /// The function containing the phi.
        function: Id,
        /// The block containing the phi.
        block: Id,
        /// The incoming block that is not a predecessor.
        incoming: Id,
    },
    /// A phi node lists the same predecessor block more than once.
    #[error("phi in block {block:?} of function {function:?} lists predecessor {incoming:?} more than once")]
    PhiDuplicatePredecessor {
        /// The function containing the phi.
        function: Id,
        /// The block containing the phi.
        block: Id,
        /// The duplicate predecessor id.
        incoming: Id,
    },
    /// A vector shuffle operand is not a vector type.
    #[error(
        "vector shuffle in block {block:?} of function {function:?} uses operand {operand} with non-vector type {found:?}"
    )]
    VectorShuffleOperandNotVector {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The operand index (0 or 1).
        operand: u32,
        /// The non-vector type id.
        found: TypeId,
    },
    /// A vector shuffle operand element types do not match.
    #[error(
        "vector shuffle in block {block:?} of function {function:?} uses mismatched component types {first:?} and {second:?}"
    )]
    VectorShuffleComponentTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The first component type.
        first: TypeId,
        /// The second component type.
        second: TypeId,
    },
    /// The vector shuffle result type does not match the operands.
    #[error(
        "vector shuffle in block {block:?} of function {function:?} expects result type {result_type:?} to match operand component type {component_type:?}"
    )]
    VectorShuffleResultTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The result type.
        result_type: TypeId,
        /// The expected component type.
        component_type: TypeId,
    },
    /// The vector shuffle component count does not match the result type.
    #[error(
        "vector shuffle in block {block:?} of function {function:?} defines {operand_components} components but result type expects {result_components}"
    )]
    VectorShuffleComponentCountMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Number of components described by operands.
        operand_components: u32,
        /// Number of components in the result vector type.
        result_components: u32,
    },
    /// A vector shuffle index was out of range.
    #[error(
        "vector shuffle in block {block:?} of function {function:?} uses component index {value} beyond the available range [0, {max}]"
    )]
    VectorShuffleComponentOutOfRange {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The offending index value.
        value: u32,
        /// The maximum valid index.
        max: u32,
    },
    /// A vector operation operand is not a vector type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} uses operand {operand} with non-vector type {found:?}"
    )]
    VectorOperandNotVector {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The operand index (zero-based).
        operand: u32,
        /// The non-vector type.
        found: TypeId,
    },
    /// A vector operation uses an index with a non-integer type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} expects operand {operand_index} to be an integer scalar index but found type {found:?}"
    )]
    VectorIndexTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The index operand position (zero-based).
        operand_index: usize,
        /// The offending type.
        found: TypeId,
    },
    /// A vector dynamic operation cannot use 8- or 16-bit types in Shader capability.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?}: cannot {operation} a vector of 8- or 16-bit types"
    )]
    VectorDynamicLimitedType {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// "extract from" or "insert into".
        operation: &'static str,
    },
    /// A vector-times-scalar operand violates type rules.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} has mismatched vector/scalar types: vector {vector_type:?}, scalar {scalar_type:?}"
    )]
    VectorTimesScalarTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode (always `OpVectorTimesScalar`).
        instruction: rspirv::spirv::Op,
        /// The vector operand type.
        vector_type: TypeId,
        /// The scalar operand type.
        scalar_type: TypeId,
    },
    /// A matrix operation operand is not a matrix type.
    #[error(
        "instruction {instruction:?} in block {block:?} of function {function:?} uses operand {operand} with non-matrix type {found:?}"
    )]
    MatrixOperandNotMatrix {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode.
        instruction: rspirv::spirv::Op,
        /// The operand index (zero-based).
        operand: u32,
        /// The non-matrix type.
        found: TypeId,
    },
    /// A matrix/vector multiply has mismatched component types.
    #[error(
        "matrix/vector multiply in block {block:?} of function {function:?} uses component types {matrix_component:?} and {vector_component:?}"
    )]
    MatrixTimesVectorComponentTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The matrix component type.
        matrix_component: TypeId,
        /// The vector component type.
        vector_component: TypeId,
    },
    /// A matrix/vector multiply has incompatible dimensions.
    #[error(
        "matrix/vector multiply in block {block:?} of function {function:?} has {matrix_columns} matrix columns but vector has {vector_components} components"
    )]
    MatrixTimesVectorDimensionMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Matrix column count.
        matrix_columns: u32,
        /// Vector component count.
        vector_components: u32,
    },
    /// A vector/matrix multiply has mismatched component types.
    #[error(
        "vector/matrix multiply in block {block:?} of function {function:?} uses component types {vector_component:?} and {matrix_component:?}"
    )]
    VectorTimesMatrixComponentTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The vector component type.
        vector_component: TypeId,
        /// The matrix component type.
        matrix_component: TypeId,
    },
    /// A vector/matrix multiply has incompatible dimensions.
    #[error(
        "vector/matrix multiply in block {block:?} of function {function:?} has vector components {vector_components} but matrix rows {matrix_rows}"
    )]
    VectorTimesMatrixDimensionMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Vector component count.
        vector_components: u32,
        /// Matrix row count.
        matrix_rows: u32,
    },
    /// A vector/matrix multiply result type is invalid.
    #[error(
        "vector/matrix multiply in block {block:?} of function {function:?} expects a vector of {expected_components} components but found {found_components}"
    )]
    VectorTimesMatrixResultDimensionMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Expected result component count.
        expected_components: u32,
        /// Actual result component count.
        found_components: u32,
    },
    /// A matrix/matrix multiply has incompatible dimensions.
    #[error(
        "matrix/matrix multiply in block {block:?} of function {function:?} has left columns {left_columns} that do not match right rows {right_rows}"
    )]
    MatrixTimesMatrixDimensionMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Left matrix column count.
        left_columns: u32,
        /// Right matrix row count.
        right_rows: u32,
    },
    /// A matrix/matrix multiply uses mismatched component types.
    #[error(
        "matrix/matrix multiply in block {block:?} of function {function:?} uses component types {left_component:?} and {right_component:?}"
    )]
    MatrixTimesMatrixComponentTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Left matrix component type.
        left_component: TypeId,
        /// Right matrix component type.
        right_component: TypeId,
    },
    /// A matrix/matrix multiply result type is invalid.
    #[error(
        "matrix/matrix multiply in block {block:?} of function {function:?} expects a matrix with {expected_columns} columns of length {expected_rows} but found a different shape"
    )]
    MatrixTimesMatrixResultShapeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Expected column count.
        expected_columns: u32,
        /// Expected row count.
        expected_rows: u32,
    },
    /// A vector/matrix multiply result uses a mismatched component type.
    #[error(
        "vector/matrix multiply in block {block:?} of function {function:?} expects component type {expected:?} but found {found:?}"
    )]
    VectorTimesMatrixResultComponentTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Expected component type.
        expected: TypeId,
        /// Found component type.
        found: TypeId,
    },
    /// A matrix/matrix multiply result uses a mismatched component type.
    #[error(
        "matrix/matrix multiply in block {block:?} of function {function:?} expects component type {expected:?} but found {found:?}"
    )]
    MatrixTimesMatrixResultComponentTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// Expected component type.
        expected: TypeId,
        /// Found component type.
        found: TypeId,
    },

    // ========== FUNCTIONS ==========
    /// A function references a missing or invalid function type.
    #[error("function {function:?} has an invalid function type {type_id:?}")]
    InvalidFunctionType {
        /// The function id.
        function: Id,
        /// The referenced function type id.
        type_id: TypeId,
    },
    /// A function declared a different return type than its function type.
    #[error("function {function:?} return type {result_type:?} does not match function type {function_type:?}")]
    FunctionReturnTypeMismatch {
        /// The function id.
        function: Id,
        /// The return type on the function instruction.
        result_type: TypeId,
        /// The return type on the function type declaration.
        function_type: TypeId,
    },
    /// A function parameter list does not match the function type parameter list.
    #[error("function {function:?} expects {expected} parameters but found {found}")]
    FunctionParameterCountMismatch {
        /// The function id.
        function: Id,
        /// The expected number of parameters per the function type.
        expected: usize,
        /// The number of parameters present on the function.
        found: usize,
    },
    /// A function parameter type does not match the function type declaration.
    #[error("function {function:?} parameter {parameter:?} has type {found:?} but function type expects {expected:?}")]
    FunctionParameterTypeMismatch {
        /// The function id.
        function: Id,
        /// The parameter id.
        parameter: Id,
        /// The expected type id.
        expected: TypeId,
        /// The found type id.
        found: TypeId,
    },
    /// Invalid use of function result id.
    #[error("Invalid use of function result id {function:?} in {use_opcode:?}")]
    FunctionInvalidUse {
        /// The function id.
        function: Id,
        /// The opcode that uses the function in an invalid way.
        use_opcode: rspirv::spirv::Op,
    },
    /// A non-void function returned without a value.
    #[error("function {function:?} must return a value of type {expected:?}")]
    MissingReturnValue {
        /// The function id.
        function: Id,
        /// The expected return type.
        expected: TypeId,
    },
    /// A void function returned a value.
    #[error("function {function:?} returns a value but has void return type")]
    ReturnValueInVoidFunction {
        /// The function id.
        function: Id,
    },
    /// A function returned a value whose type mismatched its signature.
    #[error("function {function:?} returned value of type {found:?}, expected {expected:?}")]
    InvalidReturnValueType {
        /// The function id.
        function: Id,
        /// The expected return type.
        expected: TypeId,
        /// The found value type.
        found: TypeId,
    },
    /// An `OpTypeFunction` declaration is malformed.
    #[error("function type {type_id:?} is invalid")]
    InvalidTypeFunction {
        /// The invalid function type id.
        type_id: TypeId,
    },
    /// An `OpTypeFunction` parameter uses `OpTypeVoid`.
    #[error("function type {type_id:?} has parameter type {parameter:?} which must not be void")]
    FunctionTypeParameterVoid {
        /// The function type id.
        type_id: TypeId,
        /// The parameter type that was void.
        parameter: TypeId,
    },

    // ========== ARITHMETIC ==========
    /// An arithmetic operation has an invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected} (found {result_type:?})"
    )]
    ArithmeticResultTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the arithmetic operation.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// An arithmetic operation operand has a type mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operand {operand_index} with mismatched type for result {result_type:?}"
    )]
    ArithmeticOperandTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the arithmetic operation.
        opcode: rspirv::spirv::Op,
        /// The operand index with the mismatch.
        operand_index: usize,
        /// The result type that operands should match.
        result_type: TypeId,
    },
    /// OpDot with BFloat16 result type requires BFloat16DotProductKHR capability.
    #[error(
        "OpDot in block {block:?} of function {function:?} with BFloat16 result type {result_type:?} requires BFloat16DotProductKHR capability"
    )]
    DotBFloat16RequiresCapability {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The BFloat16 result type.
        result_type: TypeId,
    },
    /// OpOuterProduct has a vector size mismatch.
    #[error(
        "OpOuterProduct in block {block:?} of function {function:?} has operand {operand_index} with size {found}, expected {expected}"
    )]
    OuterProductVectorSizeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The operand index.
        operand_index: usize,
        /// Expected size.
        expected: u32,
        /// Found size.
        found: u32,
    },
    /// OpOuterProduct has mismatched component types.
    #[error(
        "OpOuterProduct in block {block:?} of function {function:?} has mismatched component types between operands"
    )]
    OuterProductComponentTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
    },
    /// Extended arithmetic result type is not a struct.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires struct result type"
    )]
    ExtendedArithmeticResultNotStruct {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the extended arithmetic operation.
        opcode: rspirv::spirv::Op,
    },
    /// Extended arithmetic struct has wrong member count.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires struct with 2 members, found {found}"
    )]
    ExtendedArithmeticStructMemberCount {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the extended arithmetic operation.
        opcode: rspirv::spirv::Op,
        /// Number of members found.
        found: usize,
    },
    /// Extended arithmetic struct members are not identical.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires struct members to be identical types"
    )]
    ExtendedArithmeticStructMembersNotIdentical {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the extended arithmetic operation.
        opcode: rspirv::spirv::Op,
    },
    /// Extended arithmetic operand type mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operand {operand_index} with mismatched type"
    )]
    ExtendedArithmeticOperandTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the extended arithmetic operation.
        opcode: rspirv::spirv::Op,
        /// The operand index.
        operand_index: usize,
    },
    /// Extended arithmetic struct member type is invalid.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires struct member types to be {expected}"
    )]
    ExtendedArithmeticMemberTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the extended arithmetic operation.
        opcode: rspirv::spirv::Op,
        /// Expected type description.
        expected: &'static str,
    },
    /// A bitwise operation has an invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected} (found {result_type:?})"
    )]
    BitwiseResultTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the bitwise operation.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// A bitwise operation operand has a type mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operand {operand_index} with invalid type for result {result_type:?}, expected {expected}"
    )]
    BitwiseOperandTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the bitwise operation.
        opcode: rspirv::spirv::Op,
        /// The operand index with the mismatch.
        operand_index: usize,
        /// The result type that operands should match.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// A bitwise operation operand has a dimension mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has {operand_name} with mismatched dimension for result {result_type:?}"
    )]
    BitwiseDimensionMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the bitwise operation.
        opcode: rspirv::spirv::Op,
        /// The operand name with the mismatch.
        operand_name: &'static str,
        /// The result type.
        result_type: TypeId,
    },
    /// A bitwise operation operand has a bit width mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has {operand_name} with mismatched bit width for result {result_type:?}"
    )]
    BitwiseBitWidthMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the bitwise operation.
        opcode: rspirv::spirv::Op,
        /// The operand name with the mismatch.
        operand_name: &'static str,
        /// The result type.
        result_type: TypeId,
    },

    // ========== CONVERSION ==========
    /// A conversion operation has an invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected} (found {result_type:?})"
    )]
    ConversionResultTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// A conversion operation has an invalid input type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires input to be {expected} for result {result_type:?}"
    )]
    ConversionInputTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
        /// Expected input type description.
        expected: &'static str,
    },
    /// A conversion operation has a dimension mismatch between input and result.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has input with different dimension than result {result_type:?}"
    )]
    ConversionDimensionMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
    },
    /// A width conversion operation has the same bit width for input and result.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires input to have different bit width from result {result_type:?}"
    )]
    ConversionSameBitWidth {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
    },
    /// A bitcast operation has mismatched total bit widths.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires input to have same total bit width as result {result_type:?}"
    )]
    ConversionBitWidthMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
    },
    /// A pointer conversion operation is not supported in Logical addressing mode.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} is not supported in Logical addressing mode"
    )]
    ConversionLogicalAddressingNotSupported {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
    },
    /// A pointer conversion operation has an invalid storage class.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires pointer storage class {expected}"
    )]
    ConversionInvalidStorageClass {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
        /// Expected storage class.
        expected: &'static str,
    },
    /// A pointer conversion requires 64-bit integer in Vulkan with PhysicalStorageBuffer64.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires 64-bit integer in Vulkan with PhysicalStorageBuffer64 addressing mode"
    )]
    ConversionRequires64BitInteger {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
    },
    /// A conversion with 8/16-bit types is not allowed in Shader capability.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?}: 8- or 16-bit types can only be used with width-only conversions"
    )]
    ConversionLimitedTypeNotAllowed {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the conversion operation.
        opcode: rspirv::spirv::Op,
    },

    // ========== LOGICALS ==========
    /// A logical operation has an invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected} (found {result_type:?})"
    )]
    LogicalResultTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the logical operation.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// A logical operation has an invalid operand type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires operand to be {expected} for result {result_type:?}"
    )]
    LogicalOperandTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the logical operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
        /// Expected operand type description.
        expected: &'static str,
    },
    /// A logical operation has a dimension mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operand with different dimension than result {result_type:?}"
    )]
    LogicalDimensionMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the logical operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
    },
    /// A logical operation has an operand type mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operands with mismatched types for result {result_type:?}"
    )]
    LogicalOperandTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the logical operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
    },
    /// A logical operation has a bit width mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operands with mismatched bit widths for result {result_type:?}"
    )]
    LogicalBitWidthMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the logical operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
    },

    // ========== COMPOSITES ==========
    /// A composite operation has an invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected} (found {result_type:?})"
    )]
    CompositeResultTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the composite operation.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// A composite operation has an invalid operand type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operand {operand_index} with invalid type for result {result_type:?}, expected {expected}"
    )]
    CompositeOperandTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the composite operation.
        opcode: rspirv::spirv::Op,
        /// The operand index with the invalid type.
        operand_index: usize,
        /// The result type.
        result_type: TypeId,
        /// Expected operand type description.
        expected: &'static str,
    },
    /// A composite operation has an operand type mismatch.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operand with type not matching result {result_type:?}"
    )]
    CompositeOperandTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the composite operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
    },
    /// OpTranspose dimension mismatch.
    #[error("OpTranspose in block {block:?} of function {function:?}: result matrix dimensions do not match the transpose of the input matrix")]
    TransposeDimensionMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// OpCopyLogical types are not logically matching.
    #[error("OpCopyLogical in block {block:?} of function {function:?}: result type and operand type are not logically matching")]
    CopyLogicalTypesNotLogicallyMatching {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// OpConstantComposite constituent type mismatch.
    #[error("OpConstantComposite constituent at index {index} has type that does not match expected member type")]
    ConstantCompositeConstituentTypeMismatch {
        /// The index of the mismatched constituent.
        index: usize,
    },

    /// A literal number has incorrectly encoded upper bits.
    #[error(
        "literal for id {id:?} with type {type_id:?} ({bit_width}-bit {}) has invalid upper bits - must be {}",
        if *is_signed { "signed" } else { "unsigned" },
        if *is_signed { "sign-extended" } else { "zero-extended" }
    )]
    LiteralUpperBitsInvalid {
        /// The id of the constant with the invalid literal.
        id: Id,
        /// The type id of the constant.
        type_id: TypeId,
        /// The bit width of the type.
        bit_width: u32,
        /// Whether the type is signed.
        is_signed: bool,
    },
    /// A derivative instruction has an invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected} (found {result_type:?})"
    )]
    DerivativeResultTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the derivative operation.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// A derivative instruction has mismatched operand and result types.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires P type to match result type {result_type:?}"
    )]
    DerivativeOperandTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the derivative operation.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
    },
    /// A derivative instruction requires a specific execution model.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires execution model from {allowed:?}"
    )]
    DerivativeRequiresExecutionModel {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the derivative operation.
        opcode: rspirv::spirv::Op,
        /// The allowed execution models.
        allowed: Vec<rspirv::spirv::ExecutionModel>,
    },
    /// Derivative instructions in GLCompute/MeshEXT/TaskEXT require derivative execution mode.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} with {execution_model:?} execution model requires DerivativeGroupQuadsKHR or DerivativeGroupLinearKHR execution mode"
    )]
    DerivativeRequiresExecutionMode {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the derivative operation.
        opcode: rspirv::spirv::Op,
        /// The execution model that requires derivative execution mode.
        execution_model: rspirv::spirv::ExecutionModel,
    },
    /// A barrier instruction has an invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected} (found {result_type:?})"
    )]
    BarrierResultTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the barrier operation.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// A barrier instruction has an invalid operand type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has operand {operand_index} with invalid type, expected {expected}"
    )]
    BarrierOperandTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the barrier operation.
        opcode: rspirv::spirv::Op,
        /// The operand index with the invalid type.
        operand_index: usize,
        /// Expected type description.
        expected: &'static str,
    },
    /// A barrier instruction requires a specific execution model.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires execution model from {allowed:?} in SPIR-V {spirv_version}"
    )]
    BarrierRequiresExecutionModel {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the barrier operation.
        opcode: rspirv::spirv::Op,
        /// The allowed execution models.
        allowed: Vec<rspirv::spirv::ExecutionModel>,
        /// The SPIR-V version in use.
        spirv_version: crate::version::SpirvVersion,
    },
    /// An atomic instruction has an invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected} (found {result_type:?})"
    )]
    AtomicResultTypeInvalid {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// An atomic instruction uses a forbidden storage class.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} uses storage class {storage_class:?}: {reason}"
    )]
    AtomicStorageClassForbidden {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
        /// The forbidden storage class.
        storage_class: rspirv::spirv::StorageClass,
        /// Why the storage class is forbidden.
        reason: &'static str,
    },
    /// An atomic instruction requires a capability that is not declared.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires capability {required_capability:?}"
    )]
    AtomicMissingCapability {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
        /// The required capability.
        required_capability: rspirv::spirv::Capability,
    },
    /// Atomic flag instruction requires pointer to 32-bit integer.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires Pointer to point to a 32-bit integer type"
    )]
    AtomicFlagPointerTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
    },
    /// Atomic flag instruction does not support untyped pointers.
    #[error(
        "Untyped pointers are not supported by atomic flag instructions ({opcode:?} in block {block:?} of function {function:?})"
    )]
    AtomicFlagUntypedPointerNotSupported {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
    },
    /// OpAtomicStore requires pointer to integer or float scalar type.
    #[error(
        "{opcode:?} in block {block:?} of function {function:?} expected Pointer to be a pointer to integer or float scalar type"
    )]
    AtomicStorePointerNotScalar {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
    },
    /// OpAtomicCompareExchange Comparator type mismatch.
    #[error(
        "{opcode:?} in block {block:?} of function {function:?} expected Comparator to be of type Result Type"
    )]
    AtomicCompareExchangeComparatorTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
    },
    /// OpAtomicStore Value type must match Pointer pointee type.
    #[error(
        "{opcode:?} in block {block:?} of function {function:?} expected Value type and the type pointed to by Pointer to be the same"
    )]
    AtomicStoreValueTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
    },
    /// Atomic Value type must match Result Type.
    #[error(
        "{opcode:?} in block {block:?} of function {function:?} expected Value to be of type Result Type"
    )]
    AtomicValueTypeMismatch {
        /// The function containing the instruction.
        function: Id,
        /// The block containing the instruction.
        block: Id,
        /// The opcode of the atomic operation.
        opcode: rspirv::spirv::Op,
    },
    /// OpFunction has an invalid function type.
    #[error("OpFunction {function:?} has Function Type {function_type:?} which is not {expected}")]
    FunctionTypeInvalid {
        /// The function with the invalid type.
        function: Id,
        /// The invalid function type reference.
        function_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// A constant instruction has an invalid result type.
    #[error("{opcode:?} has result type {result_type:?} but expected {expected}")]
    ConstantResultTypeInvalid {
        /// The constant opcode.
        opcode: rspirv::spirv::Op,
        /// The invalid result type.
        result_type: TypeId,
        /// Expected type description.
        expected: &'static str,
    },
    /// OpConstantNull has a type that cannot be null.
    #[error("OpConstantNull has result type {result_type:?} which cannot have a null value")]
    ConstantNullTypeNotNullable {
        /// The non-nullable result type.
        result_type: TypeId,
    },
    /// A constant composite has a constituent that is not a constant or undef.
    #[error("{opcode:?} constituent {constituent:?} is not a constant or undef")]
    ConstantCompositeConstituentNotConstant {
        /// The composite constant opcode.
        opcode: rspirv::spirv::Op,
        /// The invalid constituent.
        constituent: Id,
    },
    /// A constant composite has the wrong number of constituents.
    #[error(
        "{opcode:?} has {found} constituents but result type {result_type:?} expects {expected}"
    )]
    ConstantCompositeCountMismatch {
        /// The composite constant opcode.
        opcode: rspirv::spirv::Op,
        /// The result type.
        result_type: TypeId,
        /// Expected constituent count.
        expected: usize,
        /// Actual constituent count.
        found: usize,
    },
    /// OpSpecConstantOp uses an operation that requires a missing capability.
    #[error(
        "OpSpecConstantOp operation {inner_opcode:?} requires capability {required_capability:?}"
    )]
    SpecConstantOpMissingCapability {
        /// The inner operation.
        inner_opcode: rspirv::spirv::Op,
        /// The required capability.
        required_capability: rspirv::spirv::Capability,
    },
    /// OpSpecConstantOp uses UConvert before SPIR-V 1.4 without Kernel capability.
    #[error("Prior to SPIR-V 1.4, OpSpecConstantOp UConvert requires Kernel capability or extension SPV_AMD_gpu_shader_int16")]
    SpecConstantOpUConvertRequiresKernel,
    /// Cannot form constants of 8- or 16-bit types without full capabilities.
    #[error("Cannot form constants of 8- or 16-bit types")]
    ConstantSmallTypeNotAllowed,

    // ========== IMAGE ==========
    /// Reserved opcode used (never allowed).
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: this opcode is reserved and cannot be used")]
    ReservedOpcodeUsed {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// OpTypeImage has invalid operand count.
    #[error("OpTypeImage {type_id:?} has {actual} operands, expected at least {expected}")]
    ImageTypeInvalidOperandCount {
        /// The image type ID.
        type_id: Option<TypeId>,
        /// Expected minimum operand count.
        expected: usize,
        /// Actual operand count.
        actual: usize,
    },
    /// OpTypeImage with SubpassData dimension must not be arrayed.
    #[error("OpTypeImage {type_id:?} with SubpassData dimension must not be arrayed")]
    ImageTypeSubpassDataMustNotBeArrayed {
        /// The image type ID.
        type_id: Option<TypeId>,
    },
    /// OpTypeImage with SubpassData dimension must have Sampled = 2.
    #[error("OpTypeImage {type_id:?} with SubpassData dimension must have Sampled = 2")]
    ImageTypeSubpassDataSampledMustBeTwo {
        /// The image type ID.
        type_id: Option<TypeId>,
    },
    /// OpTypeImage with Buffer dimension requires a format in Vulkan.
    #[error("OpTypeImage {type_id:?} with Buffer dimension requires a format in Vulkan")]
    ImageTypeBufferFormatRequired {
        /// The image type ID.
        type_id: Option<TypeId>,
    },
    /// Multisampled image requires Sample operand.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} operates on multisampled image but missing Sample operand"
    )]
    ImageOperandSampleRequiredForMultisampled {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Multiple offset operands specified.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} has multiple offset operands (Offset, ConstOffset, ConstOffsets are mutually exclusive)"
    )]
    ImageOperandMultipleOffsets {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Bias operand requires implicit LOD operation.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} uses Bias operand but is not an implicit LOD operation"
    )]
    ImageOperandBiasRequiresImplicitLod {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Lod operand requires explicit LOD or fetch operation.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} uses Lod operand but is not an explicit LOD or fetch operation"
    )]
    ImageOperandLodRequiresExplicitLodOrFetch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Lod and Grad operands are mutually exclusive.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} uses both Lod and Grad operands which are mutually exclusive"
    )]
    ImageOperandLodAndGradMutuallyExclusive {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Grad operand requires explicit LOD operation.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} uses Grad operand but is not an explicit LOD operation"
    )]
    ImageOperandGradRequiresExplicitLod {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// ConstOffsets operand requires gather operation.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} uses ConstOffsets operand but is not a gather operation"
    )]
    ImageOperandConstOffsetsRequiresGather {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Offset operands cannot be used with Cube dimension.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} uses offset operand with Cube dimension which is not allowed"
    )]
    ImageOperandOffsetCannotBeUsedWithCube {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Implicit LOD requires Fragment execution model.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} uses implicit LOD which requires Fragment execution model"
    )]
    ImageImplicitLodRequiresFragment {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image read requires storage image (Sampled = 0 or 2).
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} reads from image with Sampled=1 (sampling-only), requires Sampled=0 or 2"
    )]
    ImageReadRequiresStorageImage {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image write requires storage image (Sampled = 0 or 2).
    #[error(
        "OpImageWrite in block {block:?} of function {function:?} writes to image with Sampled=1 (sampling-only), requires Sampled=0 or 2"
    )]
    ImageWriteRequiresStorageImage {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Image query has invalid result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be {expected}"
    )]
    ImageQueryResultTypeInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the query operation.
        opcode: rspirv::spirv::Op,
        /// Expected type description.
        expected: &'static str,
    },
    /// OpImageQuerySizeLod used with invalid dimension.
    #[error(
        "OpImageQuerySizeLod in block {block:?} of function {function:?} cannot be used with dimension {dim:?}"
    )]
    ImageQuerySizeLodInvalidDim {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The invalid dimension.
        dim: rspirv::spirv::Dim,
    },
    /// OpImageQuerySize used with invalid dimension.
    #[error(
        "OpImageQuerySize in block {block:?} of function {function:?} cannot be used with dimension {dim:?} for sampling-only images"
    )]
    ImageQuerySizeInvalidDim {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The invalid dimension.
        dim: rspirv::spirv::Dim,
    },
    /// OpImageQueryLod result must be float vector of size 2.
    #[error(
        "OpImageQueryLod in block {block:?} of function {function:?} requires result type to be float vector of size 2 (found size {actual})"
    )]
    ImageQueryLodResultSizeInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// Expected size.
        expected: u32,
        /// Actual size.
        actual: u32,
    },
    /// OpImageQueryLod requires Fragment execution model or derivative capability.
    #[error(
        "OpImageQueryLod in block {block:?} of function {function:?} requires Fragment execution model (derivatives are only available in fragment shaders)"
    )]
    ImageQueryLodRequiresFragment {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageQueryLod cannot be used with multisampled images.
    #[error(
        "OpImageQueryLod in block {block:?} of function {function:?} cannot be used with multisampled images"
    )]
    ImageQueryLodCannotUseMultisampled {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageQueryLevels cannot be used with Buffer, Rect, or SubpassData dimensions.
    #[error(
        "OpImageQueryLevels in block {block:?} of function {function:?} cannot be used with dimension {dim:?}"
    )]
    ImageQueryLevelsInvalidDim {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The invalid dimension.
        dim: rspirv::spirv::Dim,
    },
    /// OpImageQuerySamples requires a multisampled image.
    #[error(
        "OpImageQuerySamples in block {block:?} of function {function:?} requires a multisampled image (MS=1)"
    )]
    ImageQuerySamplesRequiresMultisampled {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ========== SAMPLED IMAGE ==========
    /// OpSampledImage result type must be OpTypeSampledImage.
    #[error("OpSampledImage in block {block:?} of function {function:?} requires result type to be OpTypeSampledImage")]
    SampledImageResultTypeMustBeSampledImage {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpSampledImage in Vulkan requires Image 'Sampled' parameter to be 1.
    #[error("OpSampledImage in block {block:?} of function {function:?} requires Image 'Sampled' parameter to be 1 in Vulkan")]
    SampledImageRequiresSampledOne {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpSampledImage cannot use SubpassData dimension.
    #[error("OpSampledImage in block {block:?} of function {function:?} cannot use Image with Dim SubpassData")]
    SampledImageCannotUseSubpassData {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpSampledImage Sampler operand must be of type OpTypeSampler.
    #[error("OpSampledImage in block {block:?} of function {function:?} requires Sampler to be of type OpTypeSampler")]
    SampledImageSamplerMustBeSamplerType {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ========== IMAGE TEXEL POINTER ==========
    /// OpImageTexelPointer result type must be a pointer.
    #[error("OpImageTexelPointer in block {block:?} of function {function:?} requires result type to be a pointer")]
    ImageTexelPointerResultMustBePointer {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageTexelPointer result type storage class must be Image.
    #[error("OpImageTexelPointer in block {block:?} of function {function:?} requires result type storage class to be Image")]
    ImageTexelPointerStorageClassMustBeImage {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageTexelPointer Coordinate must be integer scalar or vector.
    #[error("OpImageTexelPointer in block {block:?} of function {function:?} requires Coordinate to be integer scalar or vector")]
    ImageTexelPointerCoordMustBeIntScalarOrVector {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageTexelPointer Sample must be integer scalar.
    #[error("OpImageTexelPointer in block {block:?} of function {function:?} requires Sample to be integer scalar")]
    ImageTexelPointerSampleMustBeIntScalar {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageTexelPointer for non-multisampled images (MS=0) must have Sample as constant 0.
    #[error("OpImageTexelPointer in block {block:?} of function {function:?} for non-multisampled image (MS=0) requires Sample to be a constant with value 0")]
    ImageTexelPointerSampleMustBeZeroForNonMultisampled {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageTexelPointer cannot use SubpassData dimension.
    #[error("OpImageTexelPointer in block {block:?} of function {function:?} cannot use Image with Dim SubpassData")]
    ImageTexelPointerCannotUseSubpassData {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageTexelPointer cannot use TileImageDataEXT dimension.
    #[error("OpImageTexelPointer in block {block:?} of function {function:?} cannot use Image with Dim TileImageDataEXT")]
    ImageTexelPointerCannotUseTileImageData {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageTexelPointer Image format invalid for Vulkan.
    #[error("OpImageTexelPointer in block {block:?} of function {function:?} requires Image format to be R64i, R64ui, R32f, R32i, or R32ui for Vulkan (found {format:?})")]
    ImageTexelPointerFormatInvalidForVulkan {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The invalid format.
        format: rspirv::spirv::ImageFormat,
    },

    /// OpTypeSampledImage operand must be OpTypeImage.
    #[error("OpTypeSampledImage requires Image operand to be of type OpTypeImage")]
    TypeSampledImageOperandMustBeImage {
        /// The result type ID of the instruction.
        type_id: Option<TypeId>,
    },
    /// OpTypeSampledImage 'Sampled' parameter must be 0 or 1.
    #[error("OpTypeSampledImage requires Image 'Sampled' parameter to be 0 or 1")]
    TypeSampledImageSampledMustBeZeroOrOne {
        /// The result type ID of the instruction.
        type_id: Option<TypeId>,
    },
    /// OpTypeSampledImage in SPIR-V 1.6+ cannot use Buffer dimension.
    #[error("OpTypeSampledImage in SPIR-V 1.6 or later cannot use Buffer dimension")]
    TypeSampledImageBufferDimInvalid {
        /// The result type ID of the instruction.
        type_id: Option<TypeId>,
    },

    /// OpImage result type must be OpTypeImage.
    #[error("OpImage in block {block:?} of function {function:?} requires result type to be OpTypeImage")]
    ImageResultTypeMustBeImage {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImage operand must be OpTypeSampledImage.
    #[error("OpImage in block {block:?} of function {function:?} requires operand to be of type OpTypeSampledImage")]
    ImageOperandMustBeSampledImage {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImage sampled image type must match result type.
    #[error("OpImage in block {block:?} of function {function:?} requires sampled image inner type to match result type")]
    ImageSampledImageTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// OpImageSparseTexelsResident result type must be bool scalar.
    #[error("OpImageSparseTexelsResident in block {block:?} of function {function:?} requires result type to be bool scalar")]
    ImageSparseTexelsResidentResultMustBeBool {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpImageSparseTexelsResident Resident Code must be int scalar.
    #[error("OpImageSparseTexelsResident in block {block:?} of function {function:?} requires Resident Code to be int scalar")]
    ImageSparseTexelsResidentCodeMustBeInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// Sparse image sample result type must be a struct with 2 members.
    #[error("{opcode:?} in block {block:?} of function {function:?} requires result type to be a struct with 2 members")]
    ImageSparseSampleResultMustBeStruct {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Sparse image sample result type's first member (residency code) must be int scalar.
    #[error("{opcode:?} in block {block:?} of function {function:?} requires result struct's first member (residency code) to be int scalar")]
    ImageSparseSampleResidencyMustBeInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// OpSampledImage operand types must match except for depth.
    #[error("OpSampledImage in block {block:?} of function {function:?} requires image operand type to match result image type (except depth)")]
    SampledImageOperandTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ========== IMAGE COORDINATE AND DREF VALIDATION ==========
    /// Image sample instruction coordinate must be float scalar or vector.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} requires Coordinate to be float scalar or vector")]
    ImageCoordinateMustBeFloat {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image coordinate has insufficient components.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires Coordinate to have at least {required} components, found {actual}"
    )]
    ImageCoordinateInsufficientComponents {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
        /// Required coordinate components.
        required: u32,
        /// Actual coordinate components.
        actual: u32,
    },
    /// Image Dref operand must be 32-bit float scalar.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} requires Dref to be 32-bit float scalar")]
    ImageDrefMustBe32BitFloat {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image Dref operation cannot use 3D dimension in Vulkan.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} cannot use 3D dimension for Dref operations in Vulkan")]
    ImageDrefCannotUse3DInVulkan {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Sampling operation invalid for multisampled image.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} is a sampling operation which is invalid for multisampled images")]
    ImageSamplingInvalidForMultisample {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image sample result type must match image sampled type.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} requires result component type to match image 'Sampled Type'")]
    ImageSampleResultTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image sample result type must be 4-component vector.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} requires result type to be 4-component vector")]
    ImageSampleResultMustBe4ComponentVector {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image sampled image operand must be OpTypeSampledImage.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} requires Sampled Image operand to be of type OpTypeSampledImage")]
    ImageOperandMustBeSampledImageType {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// OpImageFetch requires Image 'Sampled' parameter to be 1.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: expected Image 'Sampled' parameter to be 1")]
    ImageFetchRequiresSampledImage {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// OpImageFetch image dimension cannot be Cube.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: image 'Dim' cannot be Cube")]
    ImageFetchDimCannotBeCube {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// OpSampledImage in SPIR-V 1.6+ cannot use Buffer dimension.
    #[error("OpSampledImage in block {block:?} of function {function:?} cannot use Buffer dimension in SPIR-V 1.6 or later")]
    SampledImageBufferDimInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpSampledImage result consumed in a different block.
    #[error("OpSampledImage result %{sampled_image_id} defined in block {def_block:?} of function {function:?} is consumed in different block {consumer_block:?}")]
    SampledImageConsumedInDifferentBlock {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block where OpSampledImage is defined.
        def_block: Option<Id>,
        /// The block where the result is consumed.
        consumer_block: Option<Id>,
        /// The OpSampledImage result ID.
        sampled_image_id: Id,
    },
    /// OpSampledImage result used in OpPhi or OpSelect.
    #[error("OpSampledImage result %{sampled_image_id} cannot be used as operand to {consumer_opcode:?} in block {block:?} of function {function:?}")]
    SampledImageUsedInPhiOrSelect {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the consumer instruction.
        block: Option<Id>,
        /// The OpSampledImage result ID.
        sampled_image_id: Id,
        /// The opcode that incorrectly uses the sampled image (OpPhi or OpSelect).
        consumer_opcode: rspirv::spirv::Op,
    },
    /// Image Proj operation cannot use multisampled images.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} is a Proj operation which cannot use multisampled images")]
    ImageProjCannotUseMultisample {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image Proj operation cannot use arrayed images.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} is a Proj operation which cannot use arrayed images")]
    ImageProjCannotUseArrayed {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image Proj operation requires Dim 1D, 2D, 3D, or Rect.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} is a Proj operation which requires Dim 1D, 2D, 3D, or Rect (found {dim:?})")]
    ImageProjInvalidDim {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
        /// The invalid dimension.
        dim: rspirv::spirv::Dim,
    },
    /// Image Dref operation cannot use multisampled images.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} is a Dref operation which cannot use multisampled images")]
    ImageDrefCannotUseMultisample {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image Read result must be 4-component vector in Vulkan.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} result type must be 4-component vector in Vulkan (found {actual_components} components)")]
    ImageReadResultMustBe4ComponentInVulkan {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
        /// Actual number of components found.
        actual_components: u32,
    },
    /// Image Gather requires Dim 2D, Cube, or Rect.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} requires Image Dim to be 2D, Cube, or Rect (found {dim:?})")]
    ImageGatherInvalidDim {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
        /// The invalid dimension.
        dim: rspirv::spirv::Dim,
    },
    /// Image Gather Component operand must be 32-bit int scalar.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} Component operand must be 32-bit int scalar")]
    ImageGatherComponentMustBe32BitInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },
    /// Image Gather Component operand must be constant in Vulkan.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} Component operand must be a constant in Vulkan")]
    ImageGatherComponentMustBeConstantInVulkan {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },

    // ========== QCOM IMAGE PROCESSING ==========
    /// QCOM image instruction missing required decoration.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} operand is missing required decoration {decoration:?}")]
    QCOMImageMissingDecoration {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
        /// The required decoration.
        decoration: rspirv::spirv::Decoration,
    },
    /// QCOM image instruction expects OpLoad.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?} expects operand to be OpLoad")]
    QCOMImageExpectsOpLoad {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode of the image operation.
        opcode: rspirv::spirv::Op,
    },

    // ========== EXTENDED INSTRUCTIONS ==========
    /// Extended instruction result type must be float scalar or vector.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires result type to be float scalar or vector")]
    ExtInstResultTypeMustBeFloat {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction result type must be float scalar.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires result type to be float scalar")]
    ExtInstResultTypeMustBeFloatScalar {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction result type must be int scalar or vector.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires result type to be int scalar or vector")]
    ExtInstResultTypeMustBeInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction operand type mismatch.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires all operands to match result type")]
    ExtInstOperandTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction operand must be int.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires operand to be int scalar or vector")]
    ExtInstOperandMustBeInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction operand must be float.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires operand to be float scalar or vector")]
    ExtInstOperandMustBeFloat {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction requires 32-bit int.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires 32-bit int type")]
    ExtInstRequires32BitInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction requires 16 or 32-bit float.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires 16 or 32-bit float type")]
    ExtInstRequires16Or32BitFloat {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction operand must be vec4 float.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires operand to be vec4 float")]
    ExtInstOperandMustBeVec4Float {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction operand must be vec2 float.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires operand to be vec2 float")]
    ExtInstOperandMustBeVec2Float {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction result type must be vec4 float.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires result type to be vec4 float")]
    ExtInstResultTypeMustBeVec4Float {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction result type must be vec2 float.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires result type to be vec2 float")]
    ExtInstResultTypeMustBeVec2Float {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction result type must be vec3 float.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires result type to be vec3 float")]
    ExtInstResultTypeMustBeVec3Float {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The extended instruction name.
        ext_inst_name: &'static str,
    },
    /// Extended instruction Refract eta must be float scalar.
    #[error("GLSL.std.450 Refract in block {block:?} of function {function:?} requires eta operand to be float scalar")]
    ExtInstEtaMustBeFloatScalar {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// Extended instruction Refract eta component type must match result component type.
    #[error("GLSL.std.450 Refract in block {block:?} of function {function:?}: eta float type must match the component type of the result")]
    ExtInstEtaTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// ModfStruct result type must be a struct with two identical float members.
    #[error("GLSL.std.450 ModfStruct in block {block:?} of function {function:?} requires result to be a struct with two identical float scalar/vector members")]
    GlslModfStructBadResult {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// ModfStruct operand type must match struct member type.
    #[error("GLSL.std.450 ModfStruct in block {block:?} of function {function:?} requires operand X to have the same type as struct members")]
    GlslModfStructOperandMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// FrexpStruct result type must be a struct with float and 32-bit int members.
    #[error("GLSL.std.450 FrexpStruct in block {block:?} of function {function:?} requires result to be a struct with float member and 32-bit int member")]
    GlslFrexpStructBadResult {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// FrexpStruct operand type must match first struct member type.
    #[error("GLSL.std.450 FrexpStruct in block {block:?} of function {function:?} requires operand X to have the same type as the first struct member")]
    GlslFrexpStructOperandMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// Ldexp Exp operand must be int scalar or vector.
    #[error("GLSL.std.450 Ldexp in block {block:?} of function {function:?} requires Exp operand to be int scalar or vector")]
    GlslLdexpExpMustBeInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// Ldexp Exp operand must be 32-bit.
    #[error("GLSL.std.450 Ldexp in block {block:?} of function {function:?} requires Exp operand to be 32-bit int")]
    GlslLdexpExpMustBe32Bit {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// Ldexp component count mismatch between result and Exp.
    #[error("GLSL.std.450 Ldexp in block {block:?} of function {function:?} requires Exp operand to have the same component count as result")]
    GlslLdexpComponentCountMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// Interpolate requires InterpolationFunction capability.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires InterpolationFunction capability")]
    GlslInterpolateRequiresCapability {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The name of the extended instruction.
        ext_inst_name: &'static str,
    },

    /// Interpolate result must be 32-bit float.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires result to be 32-bit float scalar or vector")]
    GlslInterpolateResultMustBe32BitFloat {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The name of the extended instruction.
        ext_inst_name: &'static str,
    },

    /// Interpolate interpolant must be Input storage class.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires Interpolant storage class to be Input")]
    GlslInterpolateInputStorageClass {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The name of the extended instruction.
        ext_inst_name: &'static str,
    },

    /// Interpolate interpolant pointee type must match result type.
    #[error("GLSL.std.450 {ext_inst_name} in block {block:?} of function {function:?} requires Interpolant pointee type to match result type")]
    GlslInterpolateTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The name of the extended instruction.
        ext_inst_name: &'static str,
    },

    /// InterpolateAtSample Sample must be 32-bit int.
    #[error("GLSL.std.450 InterpolateAtSample in block {block:?} of function {function:?} requires Sample to be int scalar")]
    GlslInterpolateSampleMustBeInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// InterpolateAtSample Sample must be 32-bit.
    #[error("GLSL.std.450 InterpolateAtSample in block {block:?} of function {function:?} requires Sample to be 32-bit int")]
    GlslInterpolateSampleMustBe32Bit {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// InterpolateAtOffset Offset must be vec2 of float.
    #[error("GLSL.std.450 InterpolateAtOffset in block {block:?} of function {function:?} requires Offset to be a vector of 2 floats")]
    GlslInterpolateOffsetMustBeVec2Float {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// InterpolateAtOffset Offset must be 32-bit float.
    #[error("GLSL.std.450 InterpolateAtOffset in block {block:?} of function {function:?} requires Offset to be a vector of 2 32-bit floats")]
    GlslInterpolateOffsetMustBe32Bit {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ========== OPENCL EXTENDED INSTRUCTIONS ==========
    /// OpenCL.std extended instruction result type must be float.
    #[error("OpenCL.std extended instruction in block {block:?} of function {function:?} requires result type to be a float scalar or vector")]
    OpenClExtInstResultTypeMustBeFloat {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpenCL.std extended instruction result type must be int.
    #[error("OpenCL.std extended instruction in block {block:?} of function {function:?} requires result type to be an integer scalar or vector")]
    OpenClExtInstResultTypeMustBeInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpenCL.std extended instruction has invalid vector dimension.
    #[error("OpenCL.std extended instruction in block {block:?} of function {function:?} requires result type to be a scalar or vector with 2, 3, 4, 8, or 16 components")]
    OpenClExtInstBadVectorDimension {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpenCL.std extended instruction operand type mismatch.
    #[error("OpenCL.std extended instruction in block {block:?} of function {function:?} requires all operands to match result type")]
    OpenClExtInstOperandTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpenCL.std cross result must be float vector.
    #[error("OpenCL.std cross in block {block:?} of function {function:?} requires result type to be a float vector")]
    OpenClCrossResultMustBeFloatVector {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpenCL.std cross result must be 3 or 4 component.
    #[error("OpenCL.std cross in block {block:?} of function {function:?} requires result type to be a 3 or 4 component vector")]
    OpenClCrossBadVectorDimension {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpenCL.std distance/length result must be float scalar.
    #[error("OpenCL.std geometry function in block {block:?} of function {function:?} requires result type to be a float scalar")]
    OpenClGeometryResultMustBeFloatScalar {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpenCL.std normalize result must be float vector.
    #[error("OpenCL.std normalize in block {block:?} of function {function:?} requires result type to be a float vector")]
    OpenClNormalizeResultMustBeFloatVector {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    // ========== NON_UNIFORM ==========
    /// Non-uniform group operation result must be a boolean scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result to be a boolean scalar"
    )]
    NonUniformResultMustBeBoolScalar {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Non-uniform predicate operand must be a boolean scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires predicate to be a boolean scalar"
    )]
    NonUniformPredicateMustBeBoolScalar {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Non-uniform value operand must be a scalar or vector.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires value to be a scalar or vector of integer, floating-point, or boolean type"
    )]
    NonUniformValueInvalidType {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Non-uniform value type must match result type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires value type to match result type"
    )]
    NonUniformValueTypeMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Non-uniform ID/Index/Mask/Delta operand must be unsigned integer scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires {operand_name} to be an unsigned integer scalar"
    )]
    NonUniformIdMustBeUnsignedInt {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The operand name (Id, Index, Mask, Delta, etc.).
        operand_name: &'static str,
    },
    /// Non-uniform ID operand must be a constant in SPIR-V 1.4 or earlier.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires {operand_name} to be a constant in SPIR-V 1.4 or earlier"
    )]
    NonUniformIdMustBeConstant {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The operand name.
        operand_name: &'static str,
    },
    /// Non-uniform result must be a scalar or vector of the expected type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result to be a {expected}"
    )]
    NonUniformResultTypeInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Expected type description.
        expected: &'static str,
    },
    /// Non-uniform ballot result must be a 4-component unsigned integer vector.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires result to be a 4-component unsigned integer vector"
    )]
    NonUniformBallotResultInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Non-uniform ballot value must be a 4-component unsigned integer vector.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires value to be a 4-component unsigned integer vector"
    )]
    NonUniformBallotValueInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// ClusterSize operand is required for ClusteredReduce group operation.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} with ClusteredReduce operation requires ClusterSize operand"
    )]
    NonUniformClusterSizeRequired {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Ballot operand is required for partitioned group operations.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} with partitioned operation requires Ballot operand"
    )]
    NonUniformBallotRequired {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// ClusterSize must be an unsigned integer scalar constant.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires ClusterSize to be an unsigned integer scalar constant"
    )]
    NonUniformClusterSizeInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Ballot operand must be a 4-component integer vector.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires Ballot to be a 4-component integer vector"
    )]
    NonUniformPartitionedBallotInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Vulkan restricts OpGroupNonUniformBallotBitCount group operations.
    #[error(
        "instruction OpGroupNonUniformBallotBitCount in block {block:?} of function {function:?} group operation must be Reduce, InclusiveScan, or ExclusiveScan in Vulkan"
    )]
    NonUniformBallotBitCountInvalidGroupOp {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// ClusterSize must be at least 1 and a power of 2.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} ClusterSize must be at least 1 and a power of 2"
    )]
    NonUniformClusterSizeMustBePowerOfTwo {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    // ========== RAY_TRACING ==========
    /// Ray tracing instruction requires specific execution model(s).
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} requires one of execution models: {allowed_models}"
    )]
    RayTracingInvalidExecutionModel {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The allowed execution models.
        allowed_models: &'static str,
    },
    /// Expected Acceleration Structure type.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} expected Acceleration Structure to be of type OpTypeAccelerationStructureKHR"
    )]
    RayTracingExpectedAccelerationStructure {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Ray Flags must be a 32-bit int scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Ray Flags must be a 32-bit int scalar"
    )]
    RayTracingInvalidRayFlags {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Cull Mask must be a 32-bit int scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Cull Mask must be a 32-bit int scalar"
    )]
    RayTracingInvalidCullMask {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Ray Origin must be a 32-bit float 3-component vector.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Ray Origin must be a 32-bit float 3-component vector"
    )]
    RayTracingInvalidRayOrigin {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Ray Direction must be a 32-bit float 3-component vector.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Ray Direction must be a 32-bit float 3-component vector"
    )]
    RayTracingInvalidRayDirection {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Ray TMin/TMax must be a 32-bit float scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Ray {param_name} must be a 32-bit float scalar"
    )]
    RayTracingInvalidRayT {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The parameter name (TMin or TMax).
        param_name: &'static str,
    },
    /// SBT Offset/Stride/Index must be a 32-bit int scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} {param_name} must be a 32-bit int scalar"
    )]
    RayTracingInvalidSbtParam {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The parameter name.
        param_name: &'static str,
    },
    /// Payload must be a variable with RayPayloadKHR or IncomingRayPayloadKHR storage class.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Payload must be a variable with storage class RayPayloadKHR or IncomingRayPayloadKHR"
    )]
    RayTracingInvalidPayload {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Callable data must be a variable with CallableDataKHR or IncomingCallableDataKHR storage class.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Callable Data must be a variable with storage class CallableDataKHR or IncomingCallableDataKHR"
    )]
    RayTracingInvalidCallableData {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Hit operand must be a 32-bit float scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Hit must be a 32-bit float scalar"
    )]
    RayTracingInvalidHit {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Hit Kind must be a 32-bit unsigned int scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Hit Kind must be a 32-bit unsigned int scalar"
    )]
    RayTracingInvalidHitKind {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Motion/Current time parameter must be a 32-bit float scalar.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Current time must be a 32-bit float scalar"
    )]
    RayTracingInvalidCurrentTime {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Ray Query must be a pointer to OpTypeRayQueryKHR.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Ray Query must be a pointer to OpTypeRayQueryKHR"
    )]
    RayQueryInvalidPointer {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Intersection ID must be a 32-bit int constant.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} Intersection ID must be a constant 32-bit int scalar"
    )]
    RayQueryInvalidIntersectionId {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Ray Query result type is invalid.
    #[error(
        "instruction {opcode:?} in block {block:?} of function {function:?} expected Result Type to be {expected}"
    )]
    RayQueryInvalidResultType {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Expected type description.
        expected: &'static str,
    },

    // ========== HIT_OBJECT ==========
    /// Hit object operand must be a memory object declaration.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Hit Object must be a memory object declaration")]
    HitObjectNotMemoryObject {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object operand must be a pointer.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Hit Object must be a pointer")]
    HitObjectNotPointer {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object type must be OpTypeHitObjectNV.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Type must be OpTypeHitObjectNV")]
    HitObjectInvalidType {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object attribute operand is invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Hit Object Attributes must be OpVariable of storage class HitObjectAttributeNV")]
    HitObjectAttributeInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object payload operand is invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Payload must be OpVariable of storage class RayPayloadKHR or IncomingRayPayloadKHR")]
    HitObjectPayloadInvalid {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object instruction result type is invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: expected Result Type to be {expected}")]
    HitObjectInvalidResultType {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Expected type description.
        expected: &'static str,
    },

    /// Hit object miss index must be 32-bit unsigned int.
    #[error("instruction OpHitObjectRecordMissNV in block {block:?} of function {function:?}: Miss Index must be a 32-bit unsigned int scalar")]
    HitObjectInvalidMissIndex {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// Hit object ray origin must be 32-bit float vec3.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Ray Origin must be a 32-bit float 3-component vector")]
    HitObjectInvalidRayOrigin {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object ray direction must be 32-bit float vec3.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Ray Direction must be a 32-bit float 3-component vector")]
    HitObjectInvalidRayDirection {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object ray T value must be 32-bit float.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Ray {param_name} must be a 32-bit float scalar")]
    HitObjectInvalidRayT {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Parameter name (TMin or TMax).
        param_name: &'static str,
    },

    /// Hit object hint must be 32-bit int.
    #[error(
        "instruction in block {block:?} of function {function:?}: Hint must be a 32-bit int scalar"
    )]
    HitObjectInvalidHint {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// Hit object bits must be 32-bit int.
    #[error(
        "instruction in block {block:?} of function {function:?}: Bits must be a 32-bit int scalar"
    )]
    HitObjectInvalidBits {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },

    /// Hit object optional operands mismatch (Hint and Bits must both be present or neither).
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Hint and Bits are optional together")]
    HitObjectOptionalOperandsMismatch {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object acceleration structure invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: expected Acceleration Structure to be of type OpTypeAccelerationStructureKHR")]
    HitObjectInvalidAccelStruct {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object ray flags invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Ray Flags must be a 32-bit int scalar")]
    HitObjectInvalidRayFlags {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object cull mask invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Cull mask must be a 32-bit unsigned int scalar")]
    HitObjectInvalidCullMask {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object SBT offset invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: SBT Offset must be a 32-bit unsigned int scalar")]
    HitObjectInvalidSBTOffset {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object SBT stride invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: SBT Stride must be a 32-bit unsigned int scalar")]
    HitObjectInvalidSBTStride {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object SBT index invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: SBT Index must be a 32-bit unsigned int scalar")]
    HitObjectInvalidSBTIndex {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object SBT record offset invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: SBT record offset must be a 32-bit unsigned int scalar")]
    HitObjectInvalidSBTRecordOffset {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object SBT record stride invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: SBT record stride must be a 32-bit unsigned int scalar")]
    HitObjectInvalidSBTRecordStride {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object instance ID invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Instance Id must be a 32-bit int scalar")]
    HitObjectInvalidInstanceId {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object primitive ID invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Primitive Id must be a 32-bit int scalar")]
    HitObjectInvalidPrimitiveId {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object geometry index invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Geometry Index must be a 32-bit int scalar")]
    HitObjectInvalidGeometryIndex {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object hit kind invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Hit Kind must be a 32-bit unsigned int scalar")]
    HitObjectInvalidHitKind {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Hit object current time invalid.
    #[error("instruction {opcode:?} in block {block:?} of function {function:?}: Current Time must be a 32-bit float scalar")]
    HitObjectInvalidCurrentTime {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    // ========== MESH_SHADING ==========
    /// OpEmitMeshTasksEXT requires TaskEXT execution model.
    #[error("OpEmitMeshTasksEXT requires TaskEXT execution model")]
    MeshShadingEmitMeshTasksWrongExecutionModel {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpEmitMeshTasksEXT Group Count must be a 32-bit unsigned int scalar.
    #[error(
        "instruction OpEmitMeshTasksEXT in block {block:?} of function {function:?} Group Count {component} must be a 32-bit unsigned int scalar"
    )]
    MeshShadingInvalidGroupCount {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The component (X, Y, or Z).
        component: &'static str,
    },
    /// OpEmitMeshTasksEXT Payload must be a variable.
    #[error(
        "instruction OpEmitMeshTasksEXT in block {block:?} of function {function:?} Payload must be the result of an OpVariable"
    )]
    MeshShadingPayloadMustBeVariable {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpEmitMeshTasksEXT Payload must have TaskPayloadWorkgroupEXT storage class.
    #[error(
        "instruction OpEmitMeshTasksEXT in block {block:?} of function {function:?} Payload OpVariable must have a storage class of TaskPayloadWorkgroupEXT"
    )]
    MeshShadingPayloadWrongStorageClass {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpSetMeshOutputsEXT requires MeshEXT execution model.
    #[error("OpSetMeshOutputsEXT requires MeshEXT execution model")]
    MeshShadingSetMeshOutputsWrongExecutionModel {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
    },
    /// OpSetMeshOutputsEXT count must be a 32-bit unsigned int scalar.
    #[error(
        "instruction OpSetMeshOutputsEXT in block {block:?} of function {function:?} {count_name} must be a 32-bit unsigned int scalar"
    )]
    MeshShadingInvalidOutputCount {
        /// The function containing the instruction.
        function: Option<Id>,
        /// The block containing the instruction.
        block: Option<Id>,
        /// The count name (Vertex Count or Primitive Count).
        count_name: &'static str,
    },
    /// PerPrimitiveEXT decoration applied to wrong storage class in Fragment.
    #[error(
        "PerPrimitiveEXT decoration must be applied only to variables in the Input Storage Class in the Fragment Execution Model"
    )]
    MeshShadingPerPrimitiveFragmentWrongStorageClass {
        /// The variable ID.
        variable_id: Id,
    },
    /// PerPrimitiveEXT decoration applied to wrong storage class in MeshEXT.
    #[error(
        "PerPrimitiveEXT decoration must be applied only to variables in the Output Storage Class in the MeshEXT Execution Model"
    )]
    MeshShadingPerPrimitiveMeshWrongStorageClass {
        /// The variable ID.
        variable_id: Id,
    },

    // ========== DEBUG ==========
    /// OpMemberName Type must be a struct type.
    #[error("OpMemberName Type <id> {type_id:?} is not a struct type")]
    DebugMemberNameNotStruct {
        /// The type ID.
        type_id: Id,
    },
    /// OpMemberName Member index is out of bounds.
    #[error(
        "OpMemberName Member <id> {member_index} index is larger than Type <id> {type_id:?}'s member count ({member_count})"
    )]
    DebugMemberNameIndexOutOfBounds {
        /// The type ID.
        type_id: Id,
        /// The member index provided.
        member_index: u32,
        /// The actual member count.
        member_count: u32,
    },
    /// OpLine Target must be an OpString.
    #[error("OpLine Target <id> {file_id:?} is not an OpString")]
    DebugLineTargetNotString {
        /// The file ID.
        file_id: Id,
    },

    // ========== MEMORY_SEMANTICS ==========
    /// Memory Semantics must be a 32-bit int.
    #[error("{opcode:?}: expected Memory Semantics to be a 32-bit int")]
    MemorySemanticsNotInt32 {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Memory Semantics ids must be OpConstant when Shader capability is present.
    #[error("Memory Semantics ids must be OpConstant when Shader capability is present")]
    MemorySemanticsNotConstantWithShader,
    /// Memory Semantics UniformMemory requires capability Shader.
    #[error("{opcode:?}: Memory Semantics UniformMemory requires capability Shader")]
    MemorySemanticsUniformMemoryRequiresShader {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Memory Semantics OutputMemoryKHR requires capability VulkanMemoryModelKHR.
    #[error(
        "{opcode:?}: Memory Semantics OutputMemoryKHR requires capability VulkanMemoryModelKHR"
    )]
    MemorySemanticsOutputMemoryRequiresVulkanMemoryModel {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Memory Semantics must have at most one non-relaxed memory order bit set.
    #[error("{opcode:?}: Memory Semantics must have at most one non-relaxed memory order bit set")]
    MemorySemanticsMultipleOrderBits {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// SequentiallyConsistent not allowed in Vulkan.
    #[error("{opcode:?}: Memory Semantics with SequentiallyConsistent memory order must not be used in the Vulkan API")]
    MemorySemanticsSequentiallyConsistentInVulkan {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Invalid memory order for atomic store/flag clear.
    #[error("{opcode:?}: MemorySemantics must not use Acquire or AcquireRelease memory order")]
    MemorySemanticsInvalidOrderForStore {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Invalid memory order for atomic load.
    #[error("{opcode:?}: MemorySemantics must not use Release or AcquireRelease memory order")]
    MemorySemanticsInvalidOrderForLoad {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Memory barrier must not use relaxed memory order in Vulkan.
    #[error("{opcode:?}: MemorySemantics must not use Relaxed memory order with OpMemoryBarrier")]
    MemorySemanticsRelaxedBarrierInVulkan {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Non-relaxed memory order requires storage class in Vulkan.
    #[error("{opcode:?}: Memory Semantics with a non-relaxed memory order must have at least one Vulkan-supported storage class semantics bit set")]
    MemorySemanticsOrderWithoutStorageClass {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Storage class semantics without memory order in Vulkan.
    #[error("{opcode:?}: Memory Semantics with at least one Vulkan-supported storage class semantics bit set must use a non-relaxed memory order")]
    MemorySemanticsStorageClassWithoutOrder {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// MakeAvailableKHR requires VulkanMemoryModelKHR capability.
    #[error(
        "{opcode:?}: Memory Semantics MakeAvailableKHR requires capability VulkanMemoryModelKHR"
    )]
    MemorySemanticsMakeAvailableRequiresVulkanMemoryModel {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// MakeAvailableKHR requires Release or AcquireRelease.
    #[error("{opcode:?}: Memory Semantics with MakeAvailable bit set must use Release or AcquireRelease memory order")]
    MemorySemanticsMakeAvailableRequiresRelease {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// MakeVisibleKHR requires VulkanMemoryModelKHR capability.
    #[error(
        "{opcode:?}: Memory Semantics MakeVisibleKHR requires capability VulkanMemoryModelKHR"
    )]
    MemorySemanticsMakeVisibleRequiresVulkanMemoryModel {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// MakeVisibleKHR requires Acquire or AcquireRelease.
    #[error("{opcode:?}: Memory Semantics with MakeVisible bit set must use Acquire or AcquireRelease memory order")]
    MemorySemanticsMakeVisibleRequiresAcquire {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Volatile requires VulkanMemoryModelKHR capability.
    #[error("{opcode:?}: Memory Semantics Volatile requires capability VulkanMemoryModelKHR")]
    MemorySemanticsVolatileRequiresVulkanMemoryModel {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Volatile must not be used with barrier instructions.
    #[error("{opcode:?}: Memory Semantics with Volatile bit set must not be used with barrier instructions")]
    MemorySemanticsVolatileWithBarrier {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Unequal memory semantics must not use Release or AcquireRelease.
    #[error(
        "{opcode:?}: Unequal Memory Semantics must not use Release or AcquireRelease memory order"
    )]
    MemorySemanticsUnequalInvalidOrder {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Unequal memory semantics must not be stronger than equal.
    #[error("{opcode:?}: Unequal Memory Semantics must not use a stronger memory order than the corresponding Equal Memory Semantics")]
    MemorySemanticsUnequalStrongerThanEqual {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Relaxed memory with Invocation scope in Vulkan.
    #[error("{opcode:?}: Vulkan specification requires Memory Semantics to be Relaxed if used with Invocation Memory Scope")]
    MemorySemanticsRequiresRelaxedWithInvocation {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    // ========== SCOPE ==========
    /// Scope must be a 32-bit int.
    #[error("{opcode:?}: expected scope to be a 32-bit int")]
    ScopeNotInt32 {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Scope ids must be OpConstant when Shader capability is present.
    #[error("Scope ids must be OpConstant when Shader capability is present")]
    ScopeNotConstantWithShader,
    /// Invalid scope value.
    #[error("Invalid scope value: {value}")]
    ScopeInvalidValue {
        /// The invalid scope value.
        value: u32,
    },
    /// Execution scope limited in Vulkan.
    #[error(
        "{opcode:?}: in Vulkan environment Execution Scope is limited to Workgroup and Subgroup"
    )]
    ScopeExecutionLimitedInVulkan {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Non-uniform group operations require Subgroup scope in Vulkan.
    #[error("{opcode:?}: in Vulkan environment Execution scope is limited to Subgroup")]
    ScopeNonUniformRequiresSubgroup {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Non-uniform group operations execution scope limited.
    #[error("{opcode:?}: Execution scope is limited to Subgroup or Workgroup")]
    ScopeNonUniformLimited {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// QueueFamilyKHR scope requires VulkanMemoryModelKHR capability.
    #[error("{opcode:?}: Memory Scope QueueFamilyKHR requires capability VulkanMemoryModelKHR")]
    ScopeQueueFamilyRequiresVulkanMemoryModel {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Device scope with VulkanMemoryModel requires VulkanMemoryModelDeviceScopeKHR.
    #[error("Use of device scope with VulkanKHR memory model requires the VulkanMemoryModelDeviceScopeKHR capability")]
    ScopeDeviceRequiresDeviceScopeCapability,
    /// Memory scope limited in Vulkan.
    #[error("{opcode:?}: in Vulkan environment Memory Scope is limited to Device, QueueFamily, Workgroup, ShaderCallKHR, Subgroup, or Invocation")]
    ScopeMemoryLimitedInVulkan {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Subgroup memory scope not allowed in Vulkan 1.0 without specific capabilities.
    #[error("{opcode:?}: in Vulkan 1.0 environment Memory Scope can not be Subgroup without SubgroupBallotKHR or SubgroupVoteKHR declared")]
    ScopeSubgroupNotAllowedVulkan10 {
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    // ========== INTERFACE ==========
    /// Interface variable contains a PhysicalStorageBuffer pointer.
    #[error(
        "Input/Output interface variable id <{variable_id:?}> contains a PhysicalStorageBuffer pointer"
    )]
    InterfaceContainsPhysicalStorageBuffer {
        /// The variable ID.
        variable_id: Id,
    },
    /// Entry point has more than one PushConstant variable.
    #[error("Entry-point {entry_point:?} has more than one variable with the PushConstant storage class")]
    InterfaceMultiplePushConstant {
        /// The entry point ID.
        entry_point: Option<Id>,
    },
    /// Entry point has more than one IncomingRayPayloadKHR variable.
    #[error("Entry-point {entry_point:?} has more than one variable with the IncomingRayPayloadKHR storage class")]
    InterfaceMultipleIncomingRayPayload {
        /// The entry point ID.
        entry_point: Option<Id>,
    },
    /// Entry point has more than one HitAttributeKHR variable.
    #[error("Entry-point {entry_point:?} has more than one variable with the HitAttributeKHR storage class")]
    InterfaceMultipleHitAttribute {
        /// The entry point ID.
        entry_point: Option<Id>,
    },
    /// Entry point has more than one IncomingCallableDataKHR variable.
    #[error("Entry-point {entry_point:?} has more than one variable with the IncomingCallableDataKHR storage class")]
    InterfaceMultipleIncomingCallableData {
        /// The entry point ID.
        entry_point: Option<Id>,
    },
    // Note: InterfaceLocationConflict was consolidated into EntryPointInterfaceLocationConflict
    /// Index decoration can only be applied to Output storage class variables.
    #[error("Index decoration on variable <{variable_id:?}> must be on Output storage class")]
    IndexDecorationNotOutput {
        /// The variable ID.
        variable_id: Id,
    },
    /// Index decoration can only be applied to Fragment execution model outputs.
    #[error("Index decoration on variable <{variable_id:?}> can only be applied to Fragment output variables")]
    IndexDecorationNotFragment {
        /// The variable ID.
        variable_id: Id,
    },

    // ========== MODE_SETTING ==========
    /// Entry point must be a function.
    #[error("OpEntryPoint Entry Point <id> {entry_point:?} is not a function")]
    EntryPointNotFunction {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Entry point return type must be void.
    #[error("OpEntryPoint Entry Point <id> {entry_point:?}'s function return type is not void")]
    EntryPointReturnTypeNotVoid {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Non-Kernel entry point must have zero parameters.
    #[error(
        "OpEntryPoint Entry Point <id> {entry_point:?}'s function parameter count is {param_count}, expected 0"
    )]
    EntryPointNonZeroParameters {
        /// The entry point ID.
        entry_point: Id,
        /// The actual parameter count.
        param_count: u32,
    },
    /// Fragment entry point has both OriginUpperLeft and OriginLowerLeft.
    #[error("Fragment execution model entry point {entry_point:?} can only specify one of OriginUpperLeft or OriginLowerLeft")]
    FragmentMultipleOriginModes {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Fragment entry point is missing origin mode.
    #[error("Fragment execution model entry point {entry_point:?} requires either OriginUpperLeft or OriginLowerLeft")]
    FragmentMissingOriginMode {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Fragment entry point has multiple depth modes.
    #[error(
        "Fragment execution model entry point {entry_point:?} can specify at most one of DepthGreater, DepthLess or DepthUnchanged"
    )]
    FragmentMultipleDepthModes {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Fragment entry point has multiple interlock modes.
    #[error("Fragment execution model entry point {entry_point:?} can specify at most one fragment shader interlock execution mode")]
    FragmentMultipleInterlockModes {
        /// The entry point ID.
        entry_point: Id,
    },
    /// OriginLowerLeft not allowed in Vulkan.
    #[error("In the Vulkan environment, the OriginLowerLeft execution mode must not be used")]
    VulkanOriginLowerLeftNotAllowed,
    /// PixelCenterInteger not allowed in Vulkan.
    #[error("In the Vulkan environment, the PixelCenterInteger execution mode must not be used")]
    VulkanPixelCenterIntegerNotAllowed,
    /// VulkanMemoryModelKHR capability requires VulkanKHR memory model.
    #[error("VulkanMemoryModelKHR capability must only be specified if the VulkanKHR memory model is used")]
    VulkanMemoryModelCapabilityRequiresVulkanKHR,
    /// Vulkan requires Logical or PhysicalStorageBuffer64 addressing model.
    #[error("Addressing model {addressing_model:?} is not valid in the Vulkan environment")]
    VulkanInvalidAddressingModel {
        /// The addressing model.
        addressing_model: rspirv::spirv::AddressingModel,
    },
    /// OpenCL requires Physical32 or Physical64 addressing model.
    #[error("Addressing model {addressing_model:?} is not valid in the OpenCL environment")]
    OpenCLInvalidAddressingModel {
        /// The addressing model.
        addressing_model: rspirv::spirv::AddressingModel,
    },
    /// OpenCL requires OpenCL memory model.
    #[error("Memory model {memory_model:?} is not valid in the OpenCL environment")]
    OpenCLInvalidMemoryModel {
        /// The memory model.
        memory_model: rspirv::spirv::MemoryModel,
    },
    /// CooperativeMatrixKHR with Shader requires VulkanMemoryModel.
    #[error("If the Shader and CooperativeMatrixKHR capabilities are declared, the VulkanMemoryModel capability must also be declared")]
    CooperativeMatrixRequiresVulkanMemoryModel,
    /// Duplicate execution mode.
    #[error("Execution mode {mode:?} for entry point {entry_point:?} is specified multiple times")]
    DuplicateExecutionMode {
        /// The entry point ID.
        entry_point: Id,
        /// The execution mode.
        mode: rspirv::spirv::ExecutionMode,
    },
    /// Duplicate execution mode per entry point.
    /// This is for modes that can only appear once per entry point.
    #[error("Execution mode {execution_mode:?} can only be specified once per entry point (entry point {entry_point})")]
    DuplicateExecutionModePerEntry {
        /// The entry point ID.
        entry_point: u32,
        /// The execution mode.
        execution_mode: rspirv::spirv::ExecutionMode,
    },
    /// Duplicate execution mode per operand.
    /// This is for modes like float control modes that can appear multiple times
    /// but only with different operands.
    #[error("Execution mode {execution_mode:?} with operand {operand} is specified multiple times for entry point {entry_point}")]
    DuplicateExecutionModePerOperand {
        /// The entry point ID.
        entry_point: u32,
        /// The execution mode.
        execution_mode: rspirv::spirv::ExecutionMode,
        /// The operand value that was duplicated.
        operand: u32,
    },
    /// Tessellation has multiple spacing modes.
    #[error("Tessellation execution model entry point {entry_point:?} can specify at most one of SpacingEqual, SpacingFractionalOdd or SpacingFractionalEven")]
    TessellationMultipleSpacingModes {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Tessellation has multiple primitive types.
    #[error("Tessellation execution model entry point {entry_point:?} can specify at most one of Triangles, Quads or Isolines")]
    TessellationMultiplePrimitiveTypes {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Tessellation has multiple vertex order modes.
    #[error("Tessellation execution model entry point {entry_point:?} can specify at most one of VertexOrderCw or VertexOrderCcw")]
    TessellationMultipleVertexOrderModes {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Geometry must specify exactly one input primitive type.
    #[error("Geometry execution model entry point {entry_point:?} must specify exactly one of InputPoints, InputLines, InputLinesAdjacency, Triangles or InputTrianglesAdjacency")]
    GeometryMissingInputPrimitiveType {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Geometry must specify exactly one output primitive type.
    #[error("Geometry execution model entry point {entry_point:?} must specify exactly one of OutputPoints, OutputLineStrip or OutputTriangleStrip")]
    GeometryMissingOutputPrimitiveType {
        /// The entry point ID.
        entry_point: Id,
    },
    /// MeshEXT must specify exactly one output primitive type.
    #[error("MeshEXT execution model entry point {entry_point:?} must specify exactly one of OutputPoints, OutputLinesEXT, or OutputTrianglesEXT")]
    MeshExtMissingOutputPrimitiveType {
        /// The entry point ID.
        entry_point: Id,
    },
    /// MeshEXT must specify both OutputPrimitivesEXT and OutputVertices.
    #[error("MeshEXT execution model entry point {entry_point:?} must specify both OutputPrimitivesEXT and OutputVertices")]
    MeshExtMissingOutputModes {
        /// The entry point ID.
        entry_point: Id,
    },
    /// GLCompute requires LocalSize, LocalSizeId, or WorkgroupSize decoration.
    #[error("In the Vulkan environment, GLCompute execution model entry point {entry_point:?} requires LocalSize, LocalSizeId execution mode or WorkgroupSize decoration")]
    VulkanGLComputeMissingLocalSize {
        /// The entry point ID.
        entry_point: Id,
    },
    /// LocalSize product is zero.
    #[error(
        "LocalSize execution mode must not have a product of zero (X = {x}, Y = {y}, Z = {z})"
    )]
    LocalSizeProductZero {
        /// X dimension.
        x: u32,
        /// Y dimension.
        y: u32,
        /// Z dimension.
        z: u32,
    },
    /// DerivativeGroupQuadsKHR requires LocalSize X and Y to be multiples of 2.
    #[error("LocalSize with DerivativeGroupQuadsKHR: X ({x}) and Y ({y}) must be multiples of 2")]
    DerivativeGroupQuadsRequiresMultipleOf2 {
        /// X dimension.
        x: u64,
        /// Y dimension.
        y: u64,
    },
    /// DerivativeGroupLinearKHR requires LocalSize product to be multiple of 4.
    #[error(
        "LocalSize with DerivativeGroupLinearKHR: product ({product}) must be a multiple of 4"
    )]
    DerivativeGroupLinearRequiresMultipleOf4 {
        /// The product of local size dimensions.
        product: u64,
    },
    /// OpExecutionModeId used with mode that doesn't take ID operands.
    #[error("OpExecutionModeId is only valid when the Mode operand is an execution mode that takes Extra Operands that are id operands (mode {mode:?})")]
    ExecutionModeIdInvalidMode {
        /// The offending execution mode.
        mode: rspirv::spirv::ExecutionMode,
    },
    /// OpExecutionMode used with mode that takes ID operands.
    #[error("OpExecutionMode is only valid when the Mode operand is an execution mode that takes no Extra Operands, or takes Extra Operands that are not id operands (mode {mode:?})")]
    ExecutionModeNonIdMode {
        /// The offending execution mode.
        mode: rspirv::spirv::ExecutionMode,
    },
    /// Execution mode operands must be constants.
    #[error("For OpExecutionModeId all Extra Operand ids must be constant instructions")]
    ExecutionModeIdOperandsNotConstant,
    /// Execution mode requires specific execution model.
    #[error("Execution mode {mode:?} can only be used with {required_model}")]
    ExecutionModeInvalidExecutionModel {
        /// The execution mode.
        mode: rspirv::spirv::ExecutionMode,
        /// The required execution model description.
        required_model: &'static str,
    },
    /// Execution mode target is not an entry point.
    #[error("OpExecutionMode Entry Point <id> {entry_point:?} is not the Entry Point operand of an OpEntryPoint")]
    ExecutionModeTargetNotEntryPoint {
        /// The entry point ID.
        entry_point: Id,
    },
    /// FPFastMathDefault conflicts with ContractionOff.
    #[error("FPFastMathDefault and ContractionOff execution modes cannot be applied to the same entry point")]
    FPFastMathDefaultConflictsWithContractionOff,
    /// FPFastMathDefault conflicts with SignedZeroInfNanPreserve.
    #[error("FPFastMathDefault and SignedZeroInfNanPreserve execution modes cannot be applied to the same entry point")]
    FPFastMathDefaultConflictsWithSignedZeroInfNanPreserve,
    /// FPFastMathDefault target type must be floating-point scalar.
    #[error("The Target Type operand of FPFastMathDefault must be a floating-point scalar type")]
    FPFastMathDefaultTargetNotFloatScalar,
    /// FPFastMathDefault bitmask is invalid.
    #[error("The Fast Math Default operand is an invalid bitmask value")]
    FPFastMathDefaultInvalidBitmask,
    /// FPFastMathDefault must not include Fast.
    #[error("The Fast Math Default operand must not include Fast")]
    FPFastMathDefaultIncludesFast,
    /// FPFastMathDefault with AllowTransform requires AllowContract and AllowReassoc.
    #[error("The Fast Math Default operand must include AllowContract and AllowReassoc when AllowTransform is specified")]
    FPFastMathDefaultAllowTransformRequiresContractReassoc,
    /// FPFastMathDefault operand must be non-specialization constant.
    #[error("The Fast Math Default operand must be a non-specialization constant")]
    FPFastMathDefaultNotConstant,
    /// Decoration conflicts with FPFastMathDefault execution mode.
    /// NoContraction and FPFastMathMode Fast cannot be used by entry points
    /// that have the FPFastMathDefault execution mode (SPV_KHR_float_controls2).
    #[error("{decoration} cannot be used by an entry point with the FPFastMathDefault execution mode (result id {result_id}, entry points: {entry_points:?})")]
    DecorationConflictsWithFPFastMathDefault {
        /// The result ID with the problematic decoration.
        result_id: u32,
        /// The decoration that conflicts (NoContraction or FPFastMathMode Fast).
        decoration: String,
        /// The entry points with FPFastMathDefault that reach this instruction.
        entry_points: Vec<u32>,
    },
    /// OutputVertices must be greater than 0 for MeshEXT.
    #[error("In mesh shaders using the MeshEXT Execution Model the OutputVertices Execution Mode must be greater than 0")]
    MeshExtOutputVerticesMustBeNonZero,
    /// OutputPrimitivesEXT must be greater than 0 for MeshEXT.
    #[error("In mesh shaders using the MeshEXT Execution Model the OutputPrimitivesEXT Execution Mode must be greater than 0")]
    MeshExtOutputPrimitivesMustBeNonZero,
    /// TileShadingRateQCOM x and y must be powers of 2.
    #[error("The TileShadingRateQCOM execution mode's x and y values must be powers of 2")]
    TileShadingRateQCOMNotPowerOf2,
    /// Fragment entry point has multiple AMD stencil ref front modes.
    #[error("Fragment execution model entry point {entry_point:?} can specify at most one of StencilRefUnchangedFrontAMD, StencilRefLessFrontAMD or StencilRefGreaterFrontAMD")]
    FragmentMultipleStencilRefFrontModes {
        /// The entry point ID.
        entry_point: Id,
    },
    /// Fragment entry point has multiple AMD stencil ref back modes.
    #[error("Fragment execution model entry point {entry_point:?} can specify at most one of StencilRefUnchangedBackAMD, StencilRefLessBackAMD or StencilRefGreaterBackAMD")]
    FragmentMultipleStencilRefBackModes {
        /// The entry point ID.
        entry_point: Id,
    },

    // ========== ANNOTATIONS ==========
    /// Vulkan does not allow GLSLShared or GLSLPacked decorations.
    #[error("Decoration {decoration:?} is not valid for the Vulkan execution environment")]
    VulkanDecorationNotAllowed {
        /// The decoration.
        decoration: rspirv::spirv::Decoration,
    },
    /// FPFastMathMode and NoContraction cannot be on the same target.
    #[error("FPFastMathMode and NoContraction cannot decorate the same target <{target_id:?}>")]
    FPFastMathModeConflictsWithNoContraction {
        /// The target ID.
        target_id: Id,
    },
    /// FPFastMathMode with AllowTransform requires AllowContract and AllowReassoc.
    #[error("AllowReassoc and AllowContract must be specified when AllowTransform is specified on target <{target_id:?}>")]
    FPFastMathAllowTransformRequiresContractReassoc {
        /// The target ID.
        target_id: Id,
    },
    /// Decoration requires OpDecorateId instead of OpDecorate.
    #[error("Decoration {decoration:?} taking ID parameters may not be used with OpDecorate")]
    DecorationRequiresDecorateId {
        /// The decoration.
        decoration: rspirv::spirv::Decoration,
    },
    /// Member decoration applied to non-member target.
    #[error("Decoration {decoration:?} can only be applied to structure members, not to <{target_id:?}>")]
    MemberDecorationOnNonMember {
        /// The decoration.
        decoration: rspirv::spirv::Decoration,
        /// The target ID.
        target_id: Id,
    },
    /// OpDecorateId target is a decoration group.
    #[error("OpDecorateId Target <id> {target_id:?} must not be an OpDecorationGroup instruction")]
    DecorateIdTargetIsDecorationGroup {
        /// The target ID.
        target_id: Id,
    },
    /// OpDecorateId used with decoration that doesn't take ID parameters.
    #[error("Decoration {decoration:?} that doesn't take ID parameters may not be used with OpDecorateId")]
    DecorationDoesNotTakeIdParameters {
        /// The decoration.
        decoration: rspirv::spirv::Decoration,
    },
    /// OpMemberDecorate target is not a struct type.
    #[error("OpMemberDecorate Structure type <id> {target_id:?} is not a struct type")]
    MemberDecorateTargetNotStruct {
        /// The target ID.
        target_id: Id,
    },
    /// OpMemberDecorate member index is out of bounds.
    #[error("Index {member_index} provided in OpMemberDecorate for struct <id> {struct_id:?} is out of bounds (struct has {member_count} members)")]
    MemberDecorateIndexOutOfBounds {
        /// The struct ID.
        struct_id: Id,
        /// The member index.
        member_index: u32,
        /// The actual member count.
        member_count: u32,
    },
    /// Decoration cannot be applied to structure members.
    #[error("Decoration {decoration:?} cannot be applied to structure member {member_index} of struct <{struct_id:?}>")]
    DecorationCannotBeOnMember {
        /// The decoration.
        decoration: rspirv::spirv::Decoration,
        /// The struct ID.
        struct_id: Id,
        /// The member index.
        member_index: u32,
    },
    /// Decoration group used in invalid context.
    #[error("Result id of OpDecorationGroup <{group_id:?}> can only be targeted by OpName, OpGroupDecorate, OpDecorate, OpDecorateId, and OpGroupMemberDecorate")]
    DecorationGroupInvalidUse {
        /// The decoration group ID.
        group_id: Id,
    },
    /// OpGroupDecorate first operand is not a decoration group.
    #[error("OpGroupDecorate Decoration group <id> {target_id:?} is not a decoration group")]
    GroupDecorateNotDecorationGroup {
        /// The target ID.
        target_id: Id,
    },
    /// OpGroupDecorate target is a decoration group.
    #[error("OpGroupDecorate may not target OpDecorationGroup <id> {target_id:?}")]
    GroupDecorateTargetIsDecorationGroup {
        /// The target ID.
        target_id: Id,
    },
    /// OpGroupMemberDecorate first operand is not a decoration group.
    #[error("OpGroupMemberDecorate Decoration group <id> {target_id:?} is not a decoration group")]
    GroupMemberDecorateNotDecorationGroup {
        /// The target ID.
        target_id: Id,
    },
    /// OpGroupMemberDecorate target is not a struct type.
    #[error("OpGroupMemberDecorate Structure type <id> {struct_id:?} is not a struct type")]
    GroupMemberDecorateTargetNotStruct {
        /// The struct ID.
        struct_id: Id,
    },
    /// OpGroupMemberDecorate member index is out of bounds.
    #[error("Index {member_index} provided in OpGroupMemberDecorate for struct <id> {struct_id:?} is out of bounds (struct has {member_count} members)")]
    GroupMemberDecorateIndexOutOfBounds {
        /// The struct ID.
        struct_id: Id,
        /// The member index.
        member_index: u32,
        /// The actual member count.
        member_count: u32,
    },
    /// Vulkan: Decoration requires specific storage class.
    #[error("{decoration:?} decoration must not be applied to this storage class in Vulkan (VUID-{vuid})")]
    VulkanDecorationStorageClassMismatch {
        /// The decoration.
        decoration: rspirv::spirv::Decoration,
        /// The target ID.
        target_id: Id,
        /// The actual storage class.
        storage_class: rspirv::spirv::StorageClass,
        /// The VUID number.
        vuid: u32,
    },
    /// Vulkan: Index decoration requires Output storage class.
    #[error("Index decoration must be in the Output storage class")]
    VulkanIndexDecorationNotOutput {
        /// The target ID.
        target_id: Id,
        /// The actual storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// Vulkan: Binding/DescriptorSet requires specific storage class.
    #[error("Binding/DescriptorSet decoration must be in the StorageBuffer, Uniform, or UniformConstant storage class (VUID-6491)")]
    VulkanBindingDecorationInvalidStorageClass {
        /// The target ID.
        target_id: Id,
        /// The actual storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// Vulkan: InputAttachmentIndex requires UniformConstant storage class.
    #[error(
        "InputAttachmentIndex decoration must be in the UniformConstant storage class (VUID-6678)"
    )]
    VulkanInputAttachmentIndexInvalidStorageClass {
        /// The target ID.
        target_id: Id,
        /// The actual storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// Vulkan: Flat/NoPerspective/Centroid/Sample require Input or Output storage class.
    #[error("{decoration:?} decoration storage class must be Input or Output (VUID-4670)")]
    VulkanInterpolationDecorationInvalidStorageClass {
        /// The decoration.
        decoration: rspirv::spirv::Decoration,
        /// The target ID.
        target_id: Id,
        /// The actual storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// Vulkan: PerVertexKHR requires Input storage class.
    #[error("PerVertexKHR decoration storage class must be Input (VUID-6777)")]
    VulkanPerVertexDecorationNotInput {
        /// The target ID.
        target_id: Id,
        /// The actual storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// Vulkan: PerVertexKHR can only be applied in Fragment execution model.
    #[error("PerVertexKHR can only be applied to Fragment Execution Models (VUID-6777)")]
    VulkanPerVertexDecorationNotFragment {
        /// The variable ID.
        variable_id: Id,
    },
    /// Vulkan: PerVertexKHR decorated variables must be declared as arrays.
    #[error("PerVertexKHR must be declared as arrays (VUID-6778)")]
    VulkanPerVertexDecorationNotArray {
        /// The variable ID.
        variable_id: Id,
    },

    // ========== TENSORS ==========
    /// OpTensorReadARM result type must be a scalar or array of scalar.
    #[error("OpTensorReadARM {instruction_id:?} Result Type must be a scalar type or array of scalar type")]
    TensorReadResultNotScalar {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },
    /// Tensor operand must be a ranked tensor.
    #[error("Tensor {instruction_id:?} must be an OpTypeTensorARM whose Rank is specified")]
    TensorNotRankedTensor {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },
    /// OpTensorQuerySizeARM result type must be an integer scalar.
    #[error("OpTensorQuerySizeARM {instruction_id:?} Result Type must be an integer type scalar")]
    TensorQuerySizeResultNotInt {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },
    /// OpTensorQuerySizeARM Dimension must be less than tensor rank.
    #[error("OpTensorQuerySizeARM {instruction_id:?} Dimension ({dimension}) must be less than the Rank of Tensor ({tensor_rank})")]
    TensorQuerySizeDimensionOutOfRange {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The dimension value.
        dimension: u64,
        /// The tensor rank.
        tensor_rank: u64,
    },
    /// OpTensorQuerySizeARM Dimension must be a constant.
    #[error("OpTensorQuerySizeARM {instruction_id:?} Dimension must come from a constant instruction of scalar integer type")]
    TensorQuerySizeDimensionNotConstant {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },
    /// OpCreateTensorLayoutNV result type must be OpTypeTensorLayoutNV.
    #[error("OpCreateTensorLayoutNV {instruction_id:?} Result Type must be a tensor layout type")]
    TensorLayoutResultNotTensorLayout {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },
    /// OpCreateTensorViewNV result type must be OpTypeTensorViewNV.
    #[error("OpCreateTensorViewNV {instruction_id:?} Result Type must be a tensor view type")]
    TensorViewResultNotTensorView {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },

    // ========== MISC ==========
    /// Invalid use of 8- or 16-bit result type.
    #[error("{opcode:?} {instruction_id:?}: Invalid use of 8- or 16-bit result")]
    InvalidSmallTypeUse {
        /// The instruction ID of the invalid use.
        instruction_id: Option<Id>,
        /// The opcode of the instruction using the small type.
        opcode: rspirv::spirv::Op,
    },
    /// OpLoad/OpStore with 8/16-bit type requires scalar, vector, or matrix - not array/struct.
    #[error("In function {function:?} block {block:?}: {opcode:?} of 8- or 16-bit type without Int8/Int16/Float16 capability must have scalar, vector, or matrix type, not composite")]
    LoadStoreSmallTypeComposite {
        /// The function ID.
        function: Id,
        /// The block ID.
        block: Id,
        /// The opcode (Load or Store).
        opcode: rspirv::spirv::Op,
    },
    /// OpSelect with pointer type requires VariablePointers capability.
    #[error("In function {function:?} block {block:?}: OpSelect with pointer result type {result_type:?} requires VariablePointers or VariablePointersStorageBuffer capability")]
    SelectPointerRequiresCapability {
        /// The function ID.
        function: Id,
        /// The block ID.
        block: Id,
        /// The result type ID.
        result_type: TypeId,
    },

    /// OpSelect with image/sampler type requires BindlessTextureNV capability.
    #[error("In function {function:?} block {block:?}: OpSelect with image/sampler result type {result_type:?} requires BindlessTextureNV capability")]
    SelectImageRequiresCapability {
        /// The function ID.
        function: Id,
        /// The block ID.
        block: Id,
        /// The result type ID.
        result_type: TypeId,
    },

    /// OpSelect result type is invalid.
    #[error("In function {function:?} block {block:?}: OpSelect result type {result_type:?} must be scalar, vector{}", if *supports_composites { ", or composite" } else { "" })]
    SelectResultTypeInvalid {
        /// The function ID.
        function: Id,
        /// The block ID.
        block: Id,
        /// The result type ID.
        result_type: TypeId,
        /// Whether composite types are supported (SPIR-V 1.4+).
        supports_composites: bool,
    },

    /// OpSelect condition must be bool scalar or vector.
    #[error("In function {function:?} block {block:?}: OpSelect condition must be bool scalar or vector, result type was {result_type:?}")]
    SelectConditionNotBool {
        /// The function ID.
        function: Id,
        /// The block ID.
        block: Id,
        /// The result type ID.
        result_type: TypeId,
    },

    /// OpSelect condition dimension must match result dimension.
    #[error("In function {function:?} block {block:?}: OpSelect condition and result {result_type:?} dimensions must match")]
    SelectDimensionMismatch {
        /// The function ID.
        function: Id,
        /// The block ID.
        block: Id,
        /// The result type ID.
        result_type: TypeId,
    },

    /// OpSelect objects must match result type.
    #[error("In function {function:?} block {block:?}: OpSelect objects must match result type {result_type:?}")]
    SelectObjectTypeMismatch {
        /// The function ID.
        function: Id,
        /// The block ID.
        block: Id,
        /// The result type ID.
        result_type: TypeId,
    },
    /// OpUndef cannot create void type.
    #[error("OpUndef {instruction_id:?}: Cannot create undefined values with void type")]
    UndefCannotBeVoid {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },

    /// OpUndef cannot create 8- or 16-bit types.
    #[error("OpUndef {instruction_id:?}: Cannot create undefined values with 8- or 16-bit types")]
    UndefCannotBeSmallType {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },

    /// OpReadClockKHR has invalid scope.
    #[error("In function {function:?} block {block:?}: OpReadClockKHR scope must be {expected}")]
    ShaderClockInvalidScope {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// Expected scope description.
        expected: &'static str,
    },

    /// OpReadClockKHR has invalid result type.
    #[error("In function {function:?} block {block:?}: OpReadClockKHR result must be 64-bit uint or vec2<u32>")]
    ShaderClockInvalidResultType {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
    },

    /// OpAssumeTrueKHR value must be bool.
    #[error(
        "In function {function:?} block {block:?}: OpAssumeTrueKHR value must be a boolean scalar"
    )]
    AssumeTrueNotBool {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
    },

    /// OpExpectKHR has invalid result type.
    #[error("In function {function:?} block {block:?}: OpExpectKHR result must be int or bool scalar/vector")]
    ExpectInvalidResultType {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
    },

    /// OpExpectKHR value type mismatch.
    #[error("In function {function:?} block {block:?}: OpExpectKHR Value type does not match result type")]
    ExpectValueTypeMismatch {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
    },

    /// OpExpectKHR expected value type mismatch.
    #[error("In function {function:?} block {block:?}: OpExpectKHR ExpectedValue type does not match result type")]
    ExpectExpectedValueTypeMismatch {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
    },

    /// OpIsHelperInvocationEXT must return bool.
    #[error("In function {function:?} block {block:?}: OpIsHelperInvocationEXT result must be bool scalar")]
    IsHelperInvocationNotBool {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
    },

    /// OpIsHelperInvocationEXT requires Fragment execution model.
    #[error("In function {function:?} block {block:?}: OpIsHelperInvocationEXT requires Fragment execution model")]
    IsHelperInvocationRequiresFragment {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
    },

    /// OpDemoteToHelperInvocationEXT requires Fragment execution model.
    #[error("In function {function:?} block {block:?}: OpDemoteToHelperInvocationEXT requires Fragment execution model")]
    DemoteToHelperRequiresFragment {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
    },

    /// Invocation interlock instructions require Fragment execution model.
    #[error(
        "In function {function:?} block {block:?}: {opcode:?} requires Fragment execution model"
    )]
    InvocationInterlockRequiresFragment {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Invocation interlock instructions require an interlock execution mode.
    #[error("In function {function:?} block {block:?}: {opcode:?} requires a fragment shader interlock execution mode (PixelInterlockOrderedEXT, PixelInterlockUnorderedEXT, SampleInterlockOrderedEXT, SampleInterlockUnorderedEXT, ShadingRateInterlockOrderedEXT, or ShadingRateInterlockUnorderedEXT)")]
    InvocationInterlockRequiresMode {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    // ========== GRAPH ==========
    /// OpTypeGraphARM has too few I/O types for the number of inputs.
    #[error("OpTypeGraphARM {instruction_id:?}: {num_io_types} I/O types provided but graph has {num_inputs} inputs")]
    GraphTypeTooFewIOTypes {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// Number of I/O types provided.
        num_io_types: usize,
        /// Number of inputs declared.
        num_inputs: u32,
    },

    /// OpTypeGraphARM must have at least one output.
    #[error("OpTypeGraphARM {instruction_id:?}: A graph type must have at least one output")]
    GraphTypeNoOutputs {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },

    /// OpTypeGraphARM I/O type is not a graph interface type.
    #[error("OpTypeGraphARM {instruction_id:?}: I/O type {io_type:?} is not a Graph Interface Type (tensor or tensor array)")]
    GraphTypeInvalidIOType {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The invalid I/O type.
        io_type: Id,
    },

    /// OpGraphConstantARM result type must be a tensor type.
    #[error("OpGraphConstantARM {instruction_id:?}: Result Type must be a tensor type")]
    GraphConstantNotTensorType {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },

    /// OpGraphConstantARM duplicate constant ID.
    #[error("OpGraphConstantARM {instruction_id:?}: No two OpGraphConstantARM may have the same GraphConstantID ({constant_id})")]
    GraphConstantDuplicateId {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The duplicate constant ID.
        constant_id: u32,
    },

    /// OpGraphARM result type must be OpTypeGraphARM.
    #[error("OpGraphARM {instruction_id:?}: Result Type must be an OpTypeGraphARM")]
    GraphInvalidResultType {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },

    /// OpGraphEntryPointARM Graph operand must be an OpGraphARM.
    #[error("OpGraphEntryPointARM {instruction_id:?}: Graph {graph_id:?} must be an OpGraphARM")]
    GraphEntryPointInvalidGraph {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The invalid graph ID.
        graph_id: Id,
    },

    /// OpGraphEntryPointARM interface count mismatch.
    #[error("OpGraphEntryPointARM {instruction_id:?}: Interface list contains {actual} IDs but graph type has {expected} I/Os")]
    GraphEntryPointInterfaceCountMismatch {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// Expected number of interfaces.
        expected: usize,
        /// Actual number of interfaces.
        actual: usize,
    },

    /// OpGraphEntryPointARM interface must be OpVariable.
    #[error("OpGraphEntryPointARM {instruction_id:?}: Interface {interface_id:?} must come from OpVariable")]
    GraphEntryPointInterfaceNotVariable {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The interface ID.
        interface_id: Id,
    },

    /// OpGraphEntryPointARM interface must have UniformConstant storage class.
    #[error("OpGraphEntryPointARM {instruction_id:?}: Interface {interface_id:?} must have UniformConstant Storage Class")]
    GraphEntryPointInterfaceNotUniformConstant {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The interface ID.
        interface_id: Id,
    },

    /// OpGraphEntryPointARM interface type mismatch.
    #[error("OpGraphEntryPointARM {instruction_id:?}: Interface {interface_id:?} type {actual_type:?} must match graph I/O type {expected_type:?}")]
    GraphEntryPointInterfaceTypeMismatch {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The interface ID.
        interface_id: Id,
        /// Expected type.
        expected_type: Id,
        /// Actual type.
        actual_type: Id,
    },

    /// OpGraphInputARM/OpGraphSetOutputARM index must be 32-bit integer.
    #[error("OpGraphInputARM {instruction_id:?}: {operand} must be a 32-bit integer")]
    GraphInputIndexNotInt32 {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The operand name.
        operand: &'static str,
    },

    /// OpGraphInputARM InputIndex out of range.
    #[error("OpGraphInputARM {instruction_id:?}: InputIndex {input_index} out of range (graph has {num_inputs} inputs)")]
    GraphInputIndexOutOfRange {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The input index value.
        input_index: u64,
        /// Number of inputs in graph.
        num_inputs: u64,
    },

    /// OpGraphInputARM ElementIndex not allowed.
    #[error(
        "OpGraphInputARM {instruction_id:?}: ElementIndex not allowed when input is not an array"
    )]
    GraphInputElementIndexNotAllowed {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },

    /// OpGraphInputARM result type mismatch.
    #[error("OpGraphInputARM {instruction_id:?}: Result type {actual_type:?} does not match expected type {expected_type:?}")]
    GraphInputTypeMismatch {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// Expected type.
        expected_type: Id,
        /// Actual type.
        actual_type: Id,
    },

    /// OpGraphSetOutputARM index must be 32-bit integer.
    #[error("OpGraphSetOutputARM {instruction_id:?}: {operand} must be a 32-bit integer")]
    GraphOutputIndexNotInt32 {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The operand name.
        operand: &'static str,
    },

    /// OpGraphSetOutputARM OutputIndex out of range.
    #[error("OpGraphSetOutputARM {instruction_id:?}: OutputIndex {output_index} out of range (graph has {num_outputs} outputs)")]
    GraphOutputIndexOutOfRange {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The output index value.
        output_index: u64,
        /// Number of outputs in graph.
        num_outputs: u64,
    },

    /// OpGraphSetOutputARM ElementIndex not allowed.
    #[error("OpGraphSetOutputARM {instruction_id:?}: ElementIndex not allowed when output is not an array")]
    GraphOutputElementIndexNotAllowed {
        /// The instruction ID.
        instruction_id: Option<Id>,
    },

    /// OpGraphSetOutputARM value type mismatch.
    #[error("OpGraphSetOutputARM {instruction_id:?}: Value type {actual_type:?} does not match expected type {expected_type:?}")]
    GraphOutputTypeMismatch {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// Expected type.
        expected_type: Id,
        /// Actual type.
        actual_type: Id,
    },

    /// Duplicate OpGraphInputARM with same InputIndex.
    #[error("OpGraphInputARM {instruction_id:?}: Duplicate InputIndex {input_index} in graph definition")]
    GraphDuplicateInputIndex {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The duplicate input index.
        input_index: u64,
    },

    /// Duplicate OpGraphSetOutputARM with same OutputIndex.
    #[error("OpGraphSetOutputARM {instruction_id:?}: Duplicate OutputIndex {output_index} in graph definition")]
    GraphDuplicateOutputIndex {
        /// The instruction ID.
        instruction_id: Option<Id>,
        /// The duplicate output index.
        output_index: u64,
    },

    // ========== INVALID_TYPES ==========
    /// Operation doesn't support BFloat16 type.
    #[error("In function {function:?} block {block:?}: {opcode:?} doesn't support BFloat16 type")]
    InvalidTypeBFloat16 {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Operation doesn't support FP8 E4M3/E5M2 types.
    #[error(
        "In function {function:?} block {block:?}: {opcode:?} doesn't support FP8 E4M3/E5M2 types"
    )]
    InvalidTypeFP8 {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    // ========== PRIMITIVES ==========
    /// Primitive instruction requires Geometry execution model.
    #[error(
        "In function {function:?} block {block:?}: {opcode:?} requires Geometry execution model"
    )]
    PrimitiveRequiresGeometry {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Stream primitive instruction requires Stream to be int scalar.
    #[error("In function {function:?} block {block:?}: {opcode:?} Stream must be int scalar")]
    StreamNotIntScalar {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Stream primitive instruction requires Stream to be constant.
    #[error(
        "In function {function:?} block {block:?}: {opcode:?} Stream must be constant instruction"
    )]
    StreamNotConstant {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    // ========== TENSOR_LAYOUTS ==========
    /// Tensor layout instruction has invalid result type.
    #[error("In function {function:?} block {block:?}: {opcode:?} result type is not {expected}")]
    TensorLayoutInvalidResultType {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Expected type name.
        expected: &'static str,
    },

    /// Tensor view instruction has invalid result type.
    #[error("In function {function:?} block {block:?}: {opcode:?} result type is not {expected}")]
    TensorViewInvalidResultType {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Expected type name.
        expected: &'static str,
    },

    /// Tensor operation has mismatched tensor type.
    #[error("In function {function:?} block {block:?}: {opcode:?} tensor operand type does not match result type")]
    TensorTypeMismatch {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// Tensor operation has unexpected number of operands.
    #[error("In function {function:?} block {block:?}: {opcode:?} unexpected number of operands: expected {expected}, got {actual}")]
    TensorUnexpectedOperandCount {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// Expected operand count.
        expected: usize,
        /// Actual operand count.
        actual: usize,
    },

    /// Tensor operation operand is not a 32-bit integer.
    #[error("In function {function:?} block {block:?}: {opcode:?} operand {operand_id} is not a 32-bit integer")]
    TensorOperandNotInt32 {
        /// The function ID.
        function: Option<Id>,
        /// The block ID.
        block: Option<Id>,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The operand ID.
        operand_id: Id,
    },

    // ========== COOPERATIVE_MATRIX_TYPE ==========
    /// OpTypeCooperativeMatrixKHR/NV component type must be a scalar numeric type.
    #[error(
        "{opcode:?} {type_id:?}: Component Type must be a scalar integer or floating-point type"
    )]
    TypeCooperativeMatrixComponentNotScalar {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// OpTypeCooperativeMatrixKHR/NV Scope must be a constant instruction.
    #[error("{opcode:?} {type_id:?}: Scope operand must be a constant instruction")]
    TypeCooperativeMatrixScopeNotConstant {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// OpTypeCooperativeMatrixKHR/NV Scope must be a scalar integer.
    #[error("{opcode:?} {type_id:?}: Scope operand must be a scalar integer type")]
    TypeCooperativeMatrixScopeNotInteger {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// OpTypeCooperativeMatrixKHR/NV Rows must be a constant instruction.
    #[error("{opcode:?} {type_id:?}: Rows operand must be a constant instruction")]
    TypeCooperativeMatrixRowsNotConstant {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// OpTypeCooperativeMatrixKHR/NV Rows must be a positive integer.
    #[error("{opcode:?} {type_id:?}: Rows operand must be a positive integer, found {value}")]
    TypeCooperativeMatrixRowsNotPositive {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The value found.
        value: i64,
    },

    /// OpTypeCooperativeMatrixKHR/NV Columns must be a constant instruction.
    #[error("{opcode:?} {type_id:?}: Columns operand must be a constant instruction")]
    TypeCooperativeMatrixColumnsNotConstant {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
    },

    /// OpTypeCooperativeMatrixKHR/NV Columns must be a positive integer.
    #[error("{opcode:?} {type_id:?}: Columns operand must be a positive integer, found {value}")]
    TypeCooperativeMatrixColumnsNotPositive {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The value found.
        value: i64,
    },

    /// OpTypeCooperativeMatrixKHR Use must be a constant instruction.
    #[error("OpTypeCooperativeMatrixKHR {type_id:?}: Use operand must be a constant instruction")]
    TypeCooperativeMatrixUseNotConstant {
        /// The type ID.
        type_id: TypeId,
    },

    // ========== DEBUG_INFO ==========
    /// Debug info operand must be an OpString.
    #[error("{instruction}: expected operand {operand_name} to be an OpString")]
    DebugInfoOperandNotString {
        /// The debug info instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
    },
    /// Debug info operand must be a constant.
    #[error("{instruction}: expected operand {operand_name} to be an OpConstant")]
    DebugInfoOperandNotConstant {
        /// The debug info instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
    },
    /// Debug info operand must be a specific debug instruction.
    #[error("{instruction}: expected operand {operand_name} to be {expected}")]
    DebugInfoOperandNotDebugInstruction {
        /// The debug info instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
        /// The expected debug instruction type.
        expected: &'static str,
    },
    /// Debug info operand must be a debug type.
    #[error("{instruction}: expected operand {operand_name} to be a debug type")]
    DebugInfoOperandNotDebugType {
        /// The debug info instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
    },
    /// Debug info operand must be a lexical scope.
    #[error("{instruction}: expected operand {operand_name} to be a lexical scope (DebugCompilationUnit, DebugFunction, DebugLexicalBlock, or DebugTypeComposite)")]
    DebugInfoOperandNotLexicalScope {
        /// The debug info instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
    },
    /// DebugTypeVector component count must be 1-4.
    #[error("DebugTypeVector: Component Count must be positive integer less than or equal to 4, found {count}")]
    DebugTypeVectorInvalidComponentCount {
        /// The invalid component count.
        count: u32,
    },

    // ========== ID VALIDATION ==========
    /// Reserved opcode used.
    #[error("Opcode {opcode:?} is reserved and cannot be used")]
    ReservedOpcode {
        /// The reserved opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Operand cannot be a type in this instruction.
    #[error("Operand {operand:?} cannot be a type")]
    OperandCannotBeType {
        /// The operand that was incorrectly a type.
        operand: Id,
    },
    /// Operand requires a type but received a non-typed value.
    #[error("Operand {operand:?} requires a type")]
    OperandRequiresType {
        /// The operand that required a type.
        operand: Id,
    },
    /// Non-semantic instruction result used in semantic instruction.
    #[error(
        "Operand {operand:?} in semantic instruction cannot be a non-semantic instruction result"
    )]
    NonSemanticUsedInSemantic {
        /// The non-semantic operand.
        operand: Id,
    },
    /// OpExtInstWithForwardRefsKHR only allowed with non-semantic instructions.
    #[error("OpExtInstWithForwardRefsKHR is only allowed with non-semantic instructions")]
    ExtInstWithForwardRefsNotNonSemantic,
    /// OpExtInstWithForwardRefsKHR must have at least one forward declared ID.
    #[error("OpExtInstWithForwardRefsKHR must have at least one forward declared ID")]
    ExtInstWithForwardRefsNoForwardRefs,
    /// Forward reference in type-generating instruction requires forward pointer.
    #[error("Operand {operand:?} requires a previous definition")]
    ForwardRefInTypeRequiresPreviousDef {
        /// The forward-referenced operand.
        operand: Id,
    },

    // ========== CLSPV REFLECTION ==========
    /// CLSpv reflection instruction result type must be void.
    #[error("NonSemantic.ClspvReflection.{instruction} result type must be void")]
    ClspvResultTypeMustBeVoid {
        /// The instruction name.
        instruction: &'static str,
    },
    /// CLSpv reflection instruction requires a minimum version.
    #[error("NonSemantic.ClspvReflection.{instruction} requires version {required} or later, but version {found} is being used")]
    ClspvVersionRequired {
        /// The instruction name.
        instruction: &'static str,
        /// Required version.
        required: u32,
        /// Found version.
        found: u32,
    },
    /// CLSpv reflection operand must be an OpString.
    #[error(
        "NonSemantic.ClspvReflection.{instruction}: operand {operand_name} must be an OpString"
    )]
    ClspvOperandMustBeString {
        /// The instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
    },
    /// CLSpv reflection operand must be a 32-bit unsigned integer constant.
    #[error("NonSemantic.ClspvReflection.{instruction}: operand {operand_name} must be an OpConstant with 32-bit unsigned integer type")]
    ClspvOperandMustBeUint32Constant {
        /// The instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
    },
    /// CLSpv reflection operand must be a Kernel instruction.
    #[error("NonSemantic.ClspvReflection.{instruction}: operand {operand_name} must be a Kernel reflection instruction")]
    ClspvOperandMustBeKernel {
        /// The instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
    },
    /// CLSpv reflection operand must be an ArgumentInfo instruction.
    #[error("NonSemantic.ClspvReflection.{instruction}: operand {operand_name} must be an ArgumentInfo reflection instruction")]
    ClspvOperandMustBeArgumentInfo {
        /// The instruction name.
        instruction: &'static str,
        /// The operand name.
        operand_name: &'static str,
    },
    /// CLSpv reflection instruction must be imported.
    #[error("NonSemantic.ClspvReflection must be imported via OpExtInstImport")]
    ClspvNotImported,

    // ========== TYPE VALIDATION EXTENSIONS ==========
    /// OpTypeCooperativeVectorNV Component Type must be scalar numerical.
    #[error("OpTypeCooperativeVectorNV Component Type {component_type:?} is not a scalar numerical type")]
    TypeCooperativeVectorComponentNotScalar {
        /// The type ID.
        type_id: TypeId,
        /// The component type ID.
        component_type: TypeId,
    },
    /// OpTypeCooperativeVectorNV component count must be a constant.
    #[error(
        "OpTypeCooperativeVectorNV component count {count_id:?} is not a constant instruction"
    )]
    TypeCooperativeVectorCountNotConstant {
        /// The type ID.
        type_id: TypeId,
        /// The count ID.
        count_id: Id,
    },
    /// OpTypeCooperativeVectorNV component count must be an integer type.
    #[error(
        "OpTypeCooperativeVectorNV component count {count_id:?} is not a constant integer type"
    )]
    TypeCooperativeVectorCountNotInteger {
        /// The type ID.
        type_id: TypeId,
        /// The count ID.
        count_id: Id,
    },
    /// OpTypeCooperativeVectorNV component count must be at least 1.
    #[error("OpTypeCooperativeVectorNV component count must be at least 1, found {value}")]
    TypeCooperativeVectorCountInvalid {
        /// The type ID.
        type_id: TypeId,
        /// The invalid count value.
        value: i64,
    },
    /// OpTypeUntypedPointerKHR requires WorkgroupMemoryExplicitLayoutKHR for Workgroup storage class.
    #[error("Workgroup storage class untyped pointers in Vulkan require WorkgroupMemoryExplicitLayoutKHR capability")]
    TypeUntypedPointerWorkgroupRequiresCapability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeUntypedPointerKHR has invalid storage class in Vulkan.
    #[error("In Vulkan, untyped pointers can only be used in explicitly laid out storage classes (StorageBuffer, PhysicalStorageBuffer, Uniform, PushConstant, Workgroup)")]
    TypeUntypedPointerInvalidStorageClass {
        /// The type ID.
        type_id: TypeId,
        /// The storage class.
        storage_class: rspirv::spirv::StorageClass,
    },
    /// OpTypeTensorLayoutNV or OpTypeTensorViewNV Dim must be 32-bit integer.
    #[error("{opcode:?} Dim {dim_id:?} is not a 32-bit integer")]
    TypeTensorDimNot32BitInteger {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The dim ID.
        dim_id: Id,
    },
    /// OpTypeTensorLayoutNV or OpTypeTensorViewNV Dim must be between 1 and 5.
    #[error("{opcode:?} Dim must be between 1 and 5, found {value}")]
    TypeTensorDimOutOfRange {
        /// The type ID.
        type_id: TypeId,
        /// The opcode.
        opcode: rspirv::spirv::Op,
        /// The invalid dim value.
        value: u64,
    },
    /// OpTypeTensorLayoutNV ClampMode must be 32-bit integer.
    #[error("OpTypeTensorLayoutNV ClampMode {clamp_id:?} is not a 32-bit integer")]
    TypeTensorLayoutClampNot32BitInteger {
        /// The type ID.
        type_id: TypeId,
        /// The clamp mode ID.
        clamp_id: Id,
    },
    /// OpTypeTensorLayoutNV ClampMode must be a valid TensorClampMode.
    #[error("OpTypeTensorLayoutNV ClampMode must be a valid TensorClampMode")]
    TypeTensorLayoutClampInvalid {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeTensorViewNV HasDimensions must be a boolean.
    #[error("OpTypeTensorViewNV HasDimensions {has_dim_id:?} is not a boolean value")]
    TypeTensorViewHasDimNotBool {
        /// The type ID.
        type_id: TypeId,
        /// The HasDimensions ID.
        has_dim_id: Id,
    },
    /// OpTypeTensorViewNV Permutation must be 32-bit integer.
    #[error("OpTypeTensorViewNV Permutation {permutation_id:?} is not a 32-bit integer")]
    TypeTensorViewPermutationNot32BitInteger {
        /// The type ID.
        type_id: TypeId,
        /// The permutation ID.
        permutation_id: Id,
    },
    /// OpTypeTensorViewNV Permutation value out of range.
    #[error("OpTypeTensorViewNV Permutation {permutation_id:?} must be a valid dimension")]
    TypeTensorViewPermutationOutOfRange {
        /// The type ID.
        type_id: TypeId,
        /// The permutation ID.
        permutation_id: Id,
    },
    /// OpTypeTensorViewNV Permutation values don't form a valid permutation.
    #[error("OpTypeTensorViewNV Permutation values don't form a valid permutation")]
    TypeTensorViewPermutationInvalid {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeTensorViewNV incorrect number of permutation values.
    #[error("OpTypeTensorViewNV incorrect number of permutation values")]
    TypeTensorViewPermutationCountMismatch {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeTensorARM Element Type must be a scalar type.
    #[error("OpTypeTensorARM Element Type {element_type:?} is not a scalar type")]
    TypeTensorARMElementNotScalar {
        /// The type ID.
        type_id: TypeId,
        /// The element type ID.
        element_type: TypeId,
    },
    /// OpTypeTensorARM Rank must be a constant instruction.
    #[error("OpTypeTensorARM Rank {rank_id:?} is not a constant instruction")]
    TypeTensorARMRankNotConstant {
        /// The type ID.
        type_id: TypeId,
        /// The rank ID.
        rank_id: Id,
    },
    /// OpTypeTensorARM Rank must have scalar integer type.
    #[error("OpTypeTensorARM Rank {rank_id:?} does not have a scalar integer type")]
    TypeTensorARMRankNotInteger {
        /// The type ID.
        type_id: TypeId,
        /// The rank ID.
        rank_id: Id,
    },
    /// OpTypeTensorARM Rank must be greater than 0.
    #[error("OpTypeTensorARM Rank must define a value greater than 0")]
    TypeTensorARMRankZero {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeTensorARM Shape must be a constant instruction.
    #[error("OpTypeTensorARM Shape {shape_id:?} is not a constant instruction")]
    TypeTensorARMShapeNotConstant {
        /// The type ID.
        type_id: TypeId,
        /// The shape ID.
        shape_id: Id,
    },
    /// OpTypeTensorARM Shape must be an array of integers.
    #[error("OpTypeTensorARM Shape {shape_id:?} is not an array of integer type whose Length equals Rank")]
    TypeTensorARMShapeNotIntegerArray {
        /// The type ID.
        type_id: TypeId,
        /// The shape ID.
        shape_id: Id,
    },
    /// OpTypeTensorARM Shape constituent must be greater than 0.
    #[error("OpTypeTensorARM Shape constituent {index} is not greater than 0")]
    TypeTensorARMShapeConstituentZero {
        /// The type ID.
        type_id: TypeId,
        /// The constituent index.
        index: usize,
    },
    /// OpTypeArray/OpTypeRuntimeArray containing Block/BufferBlock must not have ArrayStride.
    #[error("Array containing a Block or BufferBlock must not be decorated with ArrayStride")]
    TypeArrayBlockCannotHaveArrayStride {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeArray/OpTypeRuntimeArray element cannot be RuntimeArray in Vulkan.
    #[error("OpTypeArray/OpTypeRuntimeArray Element Type cannot be OpTypeRuntimeArray in Vulkan")]
    TypeArrayElementCannotBeRuntimeArray {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeFunction exceeds maximum function arguments.
    #[error("OpTypeFunction may not take more than {limit} arguments, has {count}")]
    TypeFunctionTooManyArguments {
        /// The type ID.
        type_id: TypeId,
        /// Maximum allowed arguments.
        limit: u32,
        /// Actual argument count.
        count: usize,
    },
    /// OpTypeFunction has invalid use.
    #[error("Invalid use of function type result id {type_id:?}")]
    TypeFunctionInvalidUse {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeCooperativeMatrixKHR requires BFloat16CooperativeMatrixKHR for BFloat16 component.
    #[error("OpTypeCooperativeMatrix with BFloat16 component type requires BFloat16CooperativeMatrixKHR capability")]
    TypeCooperativeMatrixBFloat16RequiresCapability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeCooperativeMatrixKHR requires Float8CooperativeMatrixEXT for FP8 component.
    #[error("OpTypeCooperativeMatrix with FP8 component type requires Float8CooperativeMatrixEXT capability")]
    TypeCooperativeMatrixFP8RequiresCapability {
        /// The type ID.
        type_id: TypeId,
    },
    /// OpTypeCooperativeMatrixKHR with ScopeWorkgroup requires LocalSize/LocalSizeId.
    #[error("OpTypeCooperativeMatrixKHR with ScopeWorkgroup used without specifying LocalSize or LocalSizeId for entry point {entry_point:?}")]
    TypeCooperativeMatrixWorkgroupNoLocalSize {
        /// The type ID.
        type_id: TypeId,
        /// The entry point ID.
        entry_point: Id,
    },
    /// OpTypeCooperativeMatrixKHR with ScopeWorkgroup used before LocalSizeId constant is defined.
    #[error("OpTypeCooperativeMatrixKHR with ScopeWorkgroup used before LocalSizeId constant value {constant_id:?} is defined")]
    TypeCooperativeMatrixLocalSizeNotDefined {
        /// The type ID.
        type_id: TypeId,
        /// The constant ID.
        constant_id: Id,
    },

    // ========== EXECUTION LIMITATIONS ==========
    /// A function in the entry point callgraph is incompatible with the execution model.
    #[error(
        "Entry point {entry_point:?}'s callgraph contains function {function:?} which is incompatible with execution model {execution_model:?}: {reason}"
    )]
    ExecutionModelIncompatible {
        /// The entry point ID.
        entry_point: Id,
        /// The incompatible function ID.
        function: Id,
        /// The execution model.
        execution_model: rspirv::spirv::ExecutionModel,
        /// The reason for incompatibility.
        reason: String,
    },

    // ========== LIFETIME ==========
    /// OpLifetimeStart/Stop pointer must be OpTypePointer.
    #[error("{opcode:?} pointer operand type must be OpTypePointer")]
    LifetimePointerNotTypePointer {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// OpLifetimeStart/Stop pointer must be in Function storage class.
    #[error("{opcode:?} pointer operand must be in the Function storage class")]
    LifetimePointerNotFunctionStorageClass {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// OpLifetimeStart/Stop non-zero size requires Addresses capability.
    #[error("{opcode:?} size is non-zero, but the Addresses capability is not declared")]
    LifetimeNonZeroSizeRequiresAddresses {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },

    // ========== GROUP OPERATIONS ==========
    /// Group operation result must be a boolean scalar type.
    #[error("{opcode:?}: result must be a boolean scalar type")]
    GroupResultMustBeBoolScalar {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Group operation predicate must be a boolean scalar type.
    #[error("{opcode:?}: predicate must be a boolean scalar type")]
    GroupPredicateMustBeBoolScalar {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Group broadcast result must be a scalar or vector of integer, float, or boolean.
    #[error(
        "{opcode:?}: result must be a scalar or vector of integer, floating-point, or boolean type"
    )]
    GroupBroadcastResultInvalidType {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Group operation value type must match result type.
    #[error("{opcode:?}: the type of Value must match the Result type")]
    GroupValueTypeMismatch {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Group float operation result must be float scalar or vector.
    #[error("{opcode:?}: result must be a scalar or vector of float type")]
    GroupResultMustBeFloatScalarOrVector {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Group float operation X type must match result type.
    #[error("{opcode:?}: the type of X must match the Result type")]
    GroupXTypeMismatch {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Group int operation result must be int scalar or vector.
    #[error("{opcode:?}: result must be a scalar or vector of integer type")]
    GroupResultMustBeIntScalarOrVector {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// GroupAsyncCopy result type must be OpTypeEvent.
    #[error("OpGroupAsyncCopy: the result type must be OpTypeEvent")]
    GroupAsyncCopyResultNotEvent,
    /// GroupAsyncCopy destination must be a pointer.
    #[error("OpGroupAsyncCopy: expected Destination to be a pointer")]
    GroupAsyncCopyDestNotPointer,
    /// GroupAsyncCopy destination has invalid storage class.
    #[error("OpGroupAsyncCopy: expected Destination to be a pointer with storage class Workgroup or CrossWorkgroup")]
    GroupAsyncCopyDestInvalidStorageClass,
    /// GroupAsyncCopy destination points to invalid type.
    #[error("OpGroupAsyncCopy: expected Destination to be a pointer to scalar or vector of floating-point type or integer type")]
    GroupAsyncCopyDestInvalidPointeeType,
    /// GroupAsyncCopy source and destination types don't match.
    #[error("OpGroupAsyncCopy: expected Destination and Source to be the same type")]
    GroupAsyncCopyTypeMismatch,
    /// GroupAsyncCopy storage class pairing invalid.
    #[error("OpGroupAsyncCopy: {message}")]
    GroupAsyncCopyStorageClassMismatch {
        /// Detailed message.
        message: String,
    },
    /// GroupAsyncCopy NumElements has wrong type.
    #[error("OpGroupAsyncCopy: NumElements must be a {bit_width}-bit int scalar when Addressing Model is {addressing_model}")]
    GroupAsyncCopyNumElementsInvalidType {
        /// Expected bit width.
        bit_width: u32,
        /// The addressing model name.
        addressing_model: String,
    },
    /// GroupAsyncCopy Stride has wrong type.
    #[error("OpGroupAsyncCopy: Stride must be a {bit_width}-bit int scalar when Addressing Model is {addressing_model}")]
    GroupAsyncCopyStrideInvalidType {
        /// Expected bit width.
        bit_width: u32,
        /// The addressing model name.
        addressing_model: String,
    },
    /// GroupAsyncCopy Event has wrong type.
    #[error("OpGroupAsyncCopy: expected Event to be type OpTypeEvent")]
    GroupAsyncCopyEventNotEvent,
    /// GroupWaitEvents NumEvents has wrong type.
    #[error("OpGroupWaitEvents: expected Num Events to be a 32-bit int scalar")]
    GroupWaitEventsNumEventsInvalidType,
    /// GroupWaitEvents Events List must be pointer.
    #[error("OpGroupWaitEvents: expected Events List to be a pointer")]
    GroupWaitEventsEventsListNotPointer,
    /// GroupWaitEvents Events List must point to OpTypeEvent.
    #[error("OpGroupWaitEvents: expected Events List to be a pointer to OpTypeEvent")]
    GroupWaitEventsEventsListNotEventPointer,

    // ========== DOT PRODUCT ==========
    /// Dot product result must be int scalar type.
    #[error("{opcode:?}: result must be an int scalar type")]
    DotProductResultNotIntScalar {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Dot product accumulator type must match result type.
    #[error("{opcode:?}: result must be the same as the Accumulator type")]
    DotProductAccumulatorTypeMismatch {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Dot product result must be unsigned int scalar type.
    #[error("{opcode:?}: result must be an unsigned int scalar type")]
    DotProductResultNotUnsignedIntScalar {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Dot product vectors must be the same type.
    #[error("{opcode:?}: 'Vector 1' and 'Vector 2' must be the same type")]
    DotProductVectorTypeMismatch {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Dot product packed result width too small.
    #[error("{opcode:?}: result width ({width}) must be greater than or equal to the packed vector width of 8")]
    DotProductPackedResultWidthTooSmall {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
        /// Actual result width.
        width: u32,
    },
    /// Dot product packed vectors missing PackedVectorFormat.
    #[error("{opcode:?}: 'Vector 1' and 'Vector 2' are a 32-bit int scalar, but no Packed Vector Format was provided")]
    DotProductPackedMissingFormat {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
    },
    /// Dot product vector operand invalid.
    #[error("{opcode:?}: {message}")]
    DotProductVectorInvalid {
        /// The offending opcode.
        opcode: rspirv::spirv::Op,
        /// Detailed error message.
        message: String,
    },
}
