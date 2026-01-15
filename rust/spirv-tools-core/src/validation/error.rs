//! Validation error types for SPIR-V module validation.

use thiserror::Error;

use super::types::{
    CheckedBound, DecorationTargetId, DecorationTargetKind, DeclaredBound, ExtensionName, Id,
    IdKind, MemberDecorationTargetId, MemberIndex, MergeTargetKind, ResultId, TypeId,
};
use crate::{target_env::TargetEnv, version::SpirvVersion};

/// Errors that can arise when validating a SPIR-V module.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum ValidationError {
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
    /// A BuiltIn decoration is applied to a variable in a disallowed storage class.
    #[error("BuiltIn {builtin:?} cannot be applied to storage class {storage_class:?}")]
    InvalidBuiltInStorageClass {
        /// The built-in kind.
        builtin: rspirv::spirv::BuiltIn,
        /// The storage class of the decorated variable.
        storage_class: rspirv::spirv::StorageClass,
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
    /// `OpExecutionModeId LocalSizeId` is not permitted for the current environment/options.
    #[error("LocalSizeId execution mode is not allowed for target environment {env:?}")]
    LocalSizeIdNotAllowed {
        /// The target environment in use.
        env: TargetEnv,
    },
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
    #[error(
        "{instruction:?} Pointer <id> {pointer:?} is not a logical pointer."
    )]
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
        "entry point {entry_point:?} has overlapping {storage_class:?} locations at location {location} component {component}"
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
    /// OpFunction has an invalid function type.
    #[error(
        "OpFunction {function:?} has Function Type {function_type:?} which is not {expected}"
    )]
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
    #[error("{opcode:?} has {found} constituents but result type {result_type:?} expects {expected}")]
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

    // ========================================================================
    // Image Instruction Errors
    // ========================================================================

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

    // =========================================================================
    // Non-Uniform Group Operation Errors
    // =========================================================================

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

    // =========================================================================
    // Ray Tracing Errors
    // =========================================================================

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

    // ==================== Mesh Shading Errors ====================
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

    // ==================== Debug Instruction Errors ====================
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

    // ==================== Memory Semantics Errors ====================
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
    #[error("{opcode:?}: Memory Semantics OutputMemoryKHR requires capability VulkanMemoryModelKHR")]
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
    #[error("{opcode:?}: Memory Semantics MakeAvailableKHR requires capability VulkanMemoryModelKHR")]
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
    #[error("{opcode:?}: Memory Semantics MakeVisibleKHR requires capability VulkanMemoryModelKHR")]
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
    #[error("{opcode:?}: Unequal Memory Semantics must not use Release or AcquireRelease memory order")]
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

    // ==================== Scope Errors ====================
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
    #[error("{opcode:?}: in Vulkan environment Execution Scope is limited to Workgroup and Subgroup")]
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

    // ==================== Interface Validation Errors ====================
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
    /// Entry point has conflicting location assignment.
    #[error(
        "Entry-point {entry_point:?} has conflicting {storage_class} location assignment at location {location}, component {component}"
    )]
    InterfaceLocationConflict {
        /// The entry point ID.
        entry_point: Option<Id>,
        /// The storage class (input or output).
        storage_class: &'static str,
        /// The conflicting location.
        location: u32,
        /// The conflicting component.
        component: u32,
    },
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

    // ==================== Mode Setting Errors ====================
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

    // ==================== Annotation Errors ====================
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

    // ==================== Tensor Errors ====================
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

    // ==================== Small Type Uses Errors ====================
    /// Invalid use of 8- or 16-bit result type.
    #[error("{opcode:?} {instruction_id:?}: Invalid use of 8- or 16-bit result")]
    InvalidSmallTypeUse {
        /// The instruction ID of the invalid use.
        instruction_id: Option<Id>,
        /// The opcode of the instruction using the small type.
        opcode: rspirv::spirv::Op,
    },
}
