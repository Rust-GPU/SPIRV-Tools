use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    fmt,
    hash::{Hash, Hasher},
    num::NonZeroU32,
    sync::Arc,
};

use rspirv::dr::Module;
use thiserror::Error;

use crate::{target_env::TargetEnv, version::SpirvVersion};

/// A non-zero SPIR-V id.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Id(NonZeroU32);

/// Result ids must be non-zero and unique within a module.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResultId(Id);

/// Type ids referenced by instructions (non-zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(Id);

/// Operand ids appearing in instruction operands (non-zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct OperandId(Id);

/// Decoration targets (non-zero ids referenced by decoration instructions).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DecorationTargetId(OperandId);

/// Member decoration targets capture the struct id plus the member index.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MemberDecorationTargetId {
    target: DecorationTargetId,
    member: MemberIndex,
}

/// A struct member index (can be zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MemberIndex(u32);

/// The schema (reserved word) from the module header.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Schema(u32);

/// A validated module header with a checked bound and schema.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ValidatedHeader {
    bound: CheckedBound,
    schema: Schema,
}

/// Shared, validated words backing a module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleWords(Arc<[u32]>);

/// A set of declared capabilities for a module.
#[derive(Debug, Default)]
struct CapabilitySet {
    values: HashSet<rspirv::spirv::Capability>,
}

impl CapabilitySet {
    fn insert_unchecked(
        &mut self,
        capability: rspirv::spirv::Capability,
    ) -> Result<(), ValidationError> {
        if !self.values.insert(capability) {
            return Err(ValidationError::DuplicateCapability { capability });
        }
        Ok(())
    }

    fn insert(&mut self, capability: rspirv::spirv::Capability) -> Result<(), ValidationError> {
        self.insert_unchecked(capability)
    }
}

/// A set of declared extensions for a module.
/// Strongly-typed extension name to avoid raw string misuse.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ExtensionName(String);

impl ExtensionName {
    /// Returns the underlying extension name as a string slice.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<&str> for ExtensionName {
    fn from(value: &str) -> Self {
        ExtensionName(value.to_string())
    }
}

impl From<String> for ExtensionName {
    fn from(value: String) -> Self {
        ExtensionName(value)
    }
}

impl std::fmt::Display for ExtensionName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A set of declared extensions for a module.
#[derive(Debug, Default)]
struct ExtensionSet {
    values: HashSet<ExtensionName>,
}

impl ExtensionSet {
    fn insert_unchecked(&mut self, extension: ExtensionName) -> Result<(), ValidationError> {
        if !self.values.insert(extension.clone()) {
            return Err(ValidationError::DuplicateExtension { extension });
        }
        Ok(())
    }

    fn insert(&mut self, extension: ExtensionName, env: TargetEnv) -> Result<(), ValidationError> {
        self.insert_unchecked(extension.clone())?;
        if !env.is_extension_allowed(&extension) {
            return Err(ValidationError::DisallowedExtension { extension, env });
        }
        Ok(())
    }
}

/// Errors produced when attempting to construct zero-valued ids.
#[derive(Debug, Error, Copy, Clone, PartialEq, Eq)]
#[error("ids must be non-zero")]
pub struct ZeroIdError;

impl Id {
    /// Wraps an existing non-zero id.
    pub fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    /// Returns the underlying non-zero id.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

macro_rules! id_wrapper {
    ($name:ident) => {
        impl $name {
            /// Wraps a non-zero `Id` in the typed wrapper.
            pub fn new(id: Id) -> Self {
                Self(id)
            }

            /// Unwraps the inner `Id`.
            pub fn into_inner(self) -> Id {
                self.0
            }
        }

        impl TryFrom<u32> for $name {
            type Error = ZeroIdError;

            fn try_from(value: u32) -> Result<Self, Self::Error> {
                Id::try_from(value).map(Self)
            }
        }

        impl From<NonZeroU32> for $name {
            fn from(value: NonZeroU32) -> Self {
                Self(Id::new(value))
            }
        }

        impl From<$name> for Id {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> Self {
                value.0.get()
            }
        }
    };
}

id_wrapper!(ResultId);
id_wrapper!(TypeId);
id_wrapper!(OperandId);

impl DecorationTargetId {
    /// Wraps a non-zero operand id in a decoration target.
    pub fn new(id: OperandId) -> Self {
        Self(id)
    }

    /// Returns the underlying operand id.
    pub fn into_inner(self) -> OperandId {
        self.0
    }
}

impl From<NonZeroU32> for DecorationTargetId {
    fn from(value: NonZeroU32) -> Self {
        DecorationTargetId::new(OperandId::from(value))
    }
}

impl From<DecorationTargetId> for Id {
    fn from(value: DecorationTargetId) -> Self {
        value.0.into_inner()
    }
}

impl TryFrom<u32> for DecorationTargetId {
    type Error = ZeroIdError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        OperandId::try_from(value).map(DecorationTargetId::new)
    }
}

impl MemberDecorationTargetId {
    /// Creates a member decoration target from a struct id and member index.
    pub fn new(target: DecorationTargetId, member: MemberIndex) -> Self {
        Self { target, member }
    }

    /// Returns the struct id being decorated.
    pub fn target(self) -> DecorationTargetId {
        self.target
    }

    /// Returns the member index being decorated.
    pub fn member(self) -> MemberIndex {
        self.member
    }
}

impl MemberIndex {
    /// Constructs a member index from a raw literal (zero is allowed).
    pub fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the underlying literal member index.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for DecorationTargetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let id: Id = (*self).into();
        id.fmt(f)
    }
}

impl fmt::Display for MemberIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Schema {
    /// The only valid schema value; SPIR-V reserves this header field.
    pub const ZERO: Schema = Schema(0);

    /// Validates the raw schema value from the module header.
    pub fn validate(raw: u32) -> Result<Self, ValidationError> {
        if raw == 0 {
            Ok(Schema::ZERO)
        } else {
            Err(ValidationError::InvalidReservedWord { reserved: raw })
        }
    }

    /// Returns the raw schema value (always zero for valid modules).
    pub fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl ValidatedHeader {
    /// Creates a validated header from its components.
    pub fn new(bound: CheckedBound, schema: Schema) -> Self {
        Self { bound, schema }
    }

    /// Parses and validates a module header, ensuring the bound and schema are valid.
    pub fn from_module(module: &Module) -> Result<Self, ValidationError> {
        let header = module
            .header
            .as_ref()
            .ok_or(ValidationError::MissingHeader)?;
        let schema = Schema::validate(header.reserved_word)?;
        let declared_bound = DeclaredBound(header.bound);
        let bound = CheckedBound::new(declared_bound).ok_or(ValidationError::InvalidIdBound {
            bound: declared_bound,
        })?;
        Ok(Self { bound, schema })
    }

    /// Returns the validated id bound associated with this header.
    pub fn bound(self) -> CheckedBound {
        self.bound
    }

    /// Returns the validated schema value (always zero for valid modules).
    pub fn schema(self) -> Schema {
        self.schema
    }
}

impl ModuleWords {
    /// Wraps already-owned SPIR-V words.
    pub fn new(words: Arc<[u32]>) -> Self {
        Self(words)
    }

    /// Clones the shared words as a slice reference.
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// Returns a shared reference-counted handle to the words.
    pub fn shared(&self) -> Arc<[u32]> {
        Arc::clone(&self.0)
    }

    /// Consumes the wrapper and returns the underlying `Arc`.
    pub fn into_arc(self) -> Arc<[u32]> {
        self.0
    }
}

impl From<Arc<[u32]>> for ModuleWords {
    fn from(words: Arc<[u32]>) -> Self {
        ModuleWords::new(words)
    }
}

impl From<Box<[u32]>> for ModuleWords {
    fn from(words: Box<[u32]>) -> Self {
        ModuleWords::new(words.into())
    }
}

impl From<ModuleWords> for Arc<[u32]> {
    fn from(words: ModuleWords) -> Self {
        words.into_arc()
    }
}

impl AsRef<[u32]> for ModuleWords {
    fn as_ref(&self) -> &[u32] {
        self.as_slice()
    }
}

impl From<Id> for u32 {
    fn from(id: Id) -> Self {
        id.0.get()
    }
}

impl From<NonZeroU32> for Id {
    fn from(value: NonZeroU32) -> Self {
        Id::new(value)
    }
}

impl TryFrom<u32> for Id {
    type Error = ZeroIdError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        NonZeroU32::new(value).map(Id).ok_or(ZeroIdError)
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A declared upper bound for SPIR-V ids (must be non-zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IdBound(NonZeroU32);

impl IdBound {
    /// Wraps an existing non-zero bound.
    pub fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    /// Attempts to create an id bound from a raw value, returning `None` if zero.
    pub fn from_raw(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }

    /// Returns the underlying non-zero bound value.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

impl From<IdBound> for u32 {
    fn from(bound: IdBound) -> Self {
        bound.0.get()
    }
}

impl From<NonZeroU32> for IdBound {
    fn from(value: NonZeroU32) -> Self {
        IdBound::new(value)
    }
}

impl fmt::Display for IdBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

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
}

/// Categories of ids that must be non-zero.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IdKind {
    /// Result id produced by an instruction.
    Result,
    /// Result type id associated with an instruction.
    ResultType,
    /// Ids that appear within operands.
    Operand,
}

impl fmt::Display for IdKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdKind::Result => write!(f, "result id"),
            IdKind::ResultType => write!(f, "result type id"),
            IdKind::Operand => write!(f, "operand id"),
        }
    }
}

/// Categories of targets required by specific decorations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DecorationTargetKind {
    /// Struct type targets.
    StructType,
    /// Array, runtime array, or pointer types.
    ArrayOrPointerType,
    /// Functions.
    Function,
    /// Functions or variables.
    FunctionOrVariable,
    /// Variable-like declarations (variables and untyped variables).
    Variable,
    /// Memory object declarations (variables, parameters, raw access chains).
    MemoryObjectDeclaration,
    /// Pointer types.
    Pointer,
    /// Scalar specialization constants.
    ScalarSpecConstant,
    /// Non-specialization constants.
    Constant,
}

impl fmt::Display for DecorationTargetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorationTargetKind::StructType => write!(f, "struct type"),
            DecorationTargetKind::ArrayOrPointerType => {
                write!(f, "array, runtime array, or pointer type")
            }
            DecorationTargetKind::Function => write!(f, "function"),
            DecorationTargetKind::FunctionOrVariable => write!(f, "function or variable"),
            DecorationTargetKind::Variable => write!(f, "variable"),
            DecorationTargetKind::MemoryObjectDeclaration => {
                write!(f, "memory object declaration")
            }
            DecorationTargetKind::Pointer => write!(f, "pointer type"),
            DecorationTargetKind::ScalarSpecConstant => write!(f, "scalar specialization constant"),
            DecorationTargetKind::Constant => write!(f, "constant"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Section {
    Capabilities,
    Extensions,
    ExtInstImport,
    MemoryModel,
    SamplerImageAddressMode,
    EntryPoint,
    ExecutionMode,
    Debug1,
    Debug2,
    Debug3,
    Annotations,
    TypesGlobals,
    FunctionDeclarations,
    Functions,
    GraphDefinitions,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FunctionState {
    Outside,
    Inside,
}

/// A declared (possibly zero) id bound from a module header.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DeclaredBound(pub u32);

impl std::fmt::Display for DeclaredBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<u32> for DeclaredBound {
    fn from(value: u32) -> Self {
        DeclaredBound(value)
    }
}

/// A validated (non-zero) id bound paired with the originally declared value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CheckedBound {
    declared: DeclaredBound,
    validated: IdBound,
}

impl CheckedBound {
    /// Creates a checked bound from a declared bound, returning `None` when the declared value is zero.
    pub fn new(declared: DeclaredBound) -> Option<Self> {
        IdBound::from_raw(declared.0).map(|validated| Self {
            declared,
            validated,
        })
    }

    /// Returns the originally declared bound (which may be zero).
    pub fn declared(self) -> DeclaredBound {
        self.declared
    }

    /// Returns the validated, non-zero bound.
    pub fn validated(self) -> IdBound {
        self.validated
    }
}

impl fmt::Display for CheckedBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.declared.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryModelState {
    Missing {
        first_violation: Option<rspirv::spirv::Op>,
    },
    Seen,
}

impl MemoryModelState {
    fn new() -> Self {
        Self::Missing {
            first_violation: None,
        }
    }

    fn record_violation(&mut self, opcode: rspirv::spirv::Op) {
        if let MemoryModelState::Missing { first_violation } = self {
            if first_violation.is_none() {
                *first_violation = Some(opcode);
            }
        }
    }

    fn mark_seen(&mut self) -> Result<(), ValidationError> {
        match self {
            MemoryModelState::Missing { .. } => {
                *self = MemoryModelState::Seen;
                Ok(())
            }
            MemoryModelState::Seen => Err(ValidationError::DuplicateMemoryModel),
        }
    }

    fn is_seen(&self) -> bool {
        matches!(self, MemoryModelState::Seen)
    }

    fn finalize(self) -> Result<(), ValidationError> {
        match self {
            MemoryModelState::Seen => Ok(()),
            MemoryModelState::Missing {
                first_violation: Some(opcode),
            } => Err(ValidationError::InstructionBeforeMemoryModel { opcode }),
            MemoryModelState::Missing {
                first_violation: None,
            } => Err(ValidationError::MissingMemoryModel),
        }
    }
}

/// A validated module containing the original binary plus the parsed representation.
#[derive(Debug)]
pub struct ValidModule {
    words: ModuleWords,
    module: Module,
    env: TargetEnv,
    header: ValidatedHeader,
}

impl ValidModule {
    /// Returns the validated words that were successfully checked.
    pub fn words(&self) -> &[u32] {
        self.words.as_slice()
    }

    /// Returns the parsed module corresponding to the validated words.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Returns the target environment this module was validated against.
    pub fn env(&self) -> TargetEnv {
        self.env
    }

    /// Returns the validated module header.
    pub fn header(&self) -> ValidatedHeader {
        self.header
    }

    /// Returns a shared handle to the validated words.
    pub fn words_handle(&self) -> ModuleWords {
        self.words.clone()
    }
}

/// Validates a SPIR-V module against invariants that can be checked without target-specific
/// knowledge.
pub fn validate_module(words: &[u32], env: TargetEnv) -> Result<(), ValidationError> {
    validate_words(ModuleWords::from(Arc::from(words)), env).map(|_| ())
}

/// A cache of validated modules keyed by target environment and module contents.
#[derive(Default)]
pub struct ValidModuleCache {
    entries: std::collections::HashMap<(TargetEnv, u64), Arc<ValidModule>>,
}

impl ValidModuleCache {
    /// Validate the provided binary words, returning a shared validated module and caching the result.
    pub fn validate_words(
        &mut self,
        words: &[u32],
        env: TargetEnv,
    ) -> Result<Arc<ValidModule>, ValidationError> {
        let hash = hash_words(words, env);
        if let Some(cached) = self.entries.get(&(env, hash)) {
            if cached.words_handle().as_slice() == words {
                return Ok(Arc::clone(cached));
            }
        }
        let validated = validate_words(ModuleWords::from(Arc::from(words)), env)?;
        let validated = Arc::new(validated);
        self.entries.insert((env, hash), Arc::clone(&validated));
        Ok(validated)
    }
}

fn validate_words(words: ModuleWords, env: TargetEnv) -> Result<ValidModule, ValidationError> {
    if let Some(&schema) = words.as_slice().get(4) {
        Schema::validate(schema)?;
    }
    run_layout_check(words.as_slice(), env)?;
    let mut loader = rspirv::dr::Loader::new();
    if let Err(error) = rspirv::binary::parse_words(words.as_slice(), &mut loader) {
        return Err(ValidationError::Parse(error.to_string()));
    }
    let module = loader.module();
    let header = ValidatedHeader::from_module(&module)?;
    let defined_ids = validate_id_bound(&module, header)?;
    let opcodes = collect_result_opcodes(&module);
    let definitions = collect_result_instructions(&module);
    let capabilities = collect_declared_capabilities(&module);
    let extensions = validate_extensions(&module, env)?;
    validate_capabilities(&module, env, &extensions)?;
    validate_sampler_image_addressing_mode(&module, &capabilities)?;
    validate_memory_model(&module)?;
    let struct_member_counts = validate_member_decorations(&module, &defined_ids)?;
    validate_decoration_groups(&module, &defined_ids, &opcodes, &struct_member_counts)?;
    validate_decorations(&module, &defined_ids)?;
    validate_decoration_target_categories(&module, &opcodes, &definitions, &capabilities)?;
    let entry_points = validate_entry_points(&module, &defined_ids, &opcodes)?;
    validate_execution_modes(&module, &entry_points)?;
    Ok(ValidModule {
        words,
        module,
        env,
        header,
    })
}

fn hash_words(words: &[u32], env: TargetEnv) -> u64 {
    let mut hasher = DefaultHasher::new();
    env.hash(&mut hasher);
    words.len().hash(&mut hasher);
    for word in words {
        word.hash(&mut hasher);
    }
    hasher.finish()
}

/// Input sources that can be validated before becoming a `ValidModule`.
pub enum MaybeValidModule<'a> {
    /// Pre-assembled SPIR-V words.
    Binary(&'a [u32]),
    /// SPIR-V assembly text to be assembled and validated.
    Text(&'a str),
}

impl<'a> MaybeValidModule<'a> {
    /// Validate the provided input, assembling text when necessary.
    pub fn validate(self, env: TargetEnv) -> Result<ValidModule, ValidationError> {
        match self {
            MaybeValidModule::Binary(words) => {
                validate_words(ModuleWords::from(Arc::from(words)), env)
            }
            MaybeValidModule::Text(text) => {
                let binary = ModuleWords::from(Arc::<[u32]>::from(
                    crate::assembly::assemble_text(text)
                        .map_err(|err| ValidationError::Parse(err.to_string()))?
                        .into_boxed_slice(),
                ));
                validate_words(binary, env)
            }
        }
    }
}

/// Convenience trait for validating either binary words or assembly text.
pub trait ValidatableModule<'a> {
    /// Validates the module input for the requested target environment.
    fn validate(self, env: TargetEnv) -> Result<ValidModule, ValidationError>;
}

impl<'a> ValidatableModule<'a> for &'a [u32] {
    fn validate(self, env: TargetEnv) -> Result<ValidModule, ValidationError> {
        MaybeValidModule::Binary(self).validate(env)
    }
}

impl<'a> ValidatableModule<'a> for &'a str {
    fn validate(self, env: TargetEnv) -> Result<ValidModule, ValidationError> {
        MaybeValidModule::Text(self).validate(env)
    }
}

fn run_layout_check(words: &[u32], _env: TargetEnv) -> Result<(), ValidationError> {
    struct LayoutChecker {
        memory_model_state: MemoryModelState,
        current_section: Section,
        function_state: FunctionState,
        capabilities: CapabilitySet,
        extensions: ExtensionSet,
        sampler_image_address_mode: Option<u32>,
    }

    impl LayoutChecker {
        fn new() -> Self {
            Self {
                memory_model_state: MemoryModelState::new(),
                current_section: Section::Capabilities,
                function_state: FunctionState::Outside,
                capabilities: CapabilitySet::default(),
                extensions: ExtensionSet::default(),
                sampler_image_address_mode: None,
            }
        }
    }

    impl rspirv::binary::Consumer for LayoutChecker {
        fn initialize(&mut self) -> rspirv::binary::ParseAction {
            rspirv::binary::ParseAction::Continue
        }

        fn finalize(&mut self) -> rspirv::binary::ParseAction {
            if let Err(err) = self.memory_model_state.finalize() {
                return rspirv::binary::ParseAction::Error(Box::new(err));
            }
            if self
                .capabilities
                .values
                .contains(&rspirv::spirv::Capability::BindlessTextureNV)
                && self.sampler_image_address_mode.is_none()
            {
                return rspirv::binary::ParseAction::Error(Box::new(
                    ValidationError::MissingSamplerImageAddressingMode,
                ));
            }
            rspirv::binary::ParseAction::Continue
        }

        fn consume_header(
            &mut self,
            header: rspirv::dr::ModuleHeader,
        ) -> rspirv::binary::ParseAction {
            if let Err(err) = Schema::validate(header.reserved_word) {
                return rspirv::binary::ParseAction::Error(Box::new(err));
            }
            rspirv::binary::ParseAction::Continue
        }

        fn consume_instruction(
            &mut self,
            inst: rspirv::dr::Instruction,
        ) -> rspirv::binary::ParseAction {
            if matches!(self.function_state, FunctionState::Inside) {
                match inst.class.opcode {
                    rspirv::spirv::Op::MemoryModel => {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::FunctionBeforeMemoryModel,
                        ));
                    }
                    rspirv::spirv::Op::FunctionEnd => {
                        self.function_state = FunctionState::Outside;
                    }
                    _ => {}
                }
                return rspirv::binary::ParseAction::Continue;
            }

            let opcode = inst.class.opcode;
            let section = instruction_section(self.current_section, &inst);

            match opcode {
                rspirv::spirv::Op::MemoryModel => {
                    if self.current_section > Section::MemoryModel {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::LayoutOutOfOrder {
                                opcode: rspirv::spirv::Op::MemoryModel,
                            },
                        ));
                    }
                    if let Err(err) = self.memory_model_state.mark_seen() {
                        return rspirv::binary::ParseAction::Error(Box::new(err));
                    }
                }
                rspirv::spirv::Op::Capability => {
                    if section < self.current_section {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::LayoutOutOfOrder {
                                opcode: rspirv::spirv::Op::Capability,
                            },
                        ));
                    }
                    if let Some(cap) = capability_operand(&inst) {
                        if let Err(err) = self.capabilities.insert(cap) {
                            return rspirv::binary::ParseAction::Error(Box::new(err));
                        }
                    }
                }
                rspirv::spirv::Op::Extension => {
                    if section < self.current_section {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::LayoutOutOfOrder {
                                opcode: rspirv::spirv::Op::Extension,
                            },
                        ));
                    }
                    if let Some(extension) = extension_operand(&inst) {
                        if let Err(err) = self.extensions.insert_unchecked(extension) {
                            return rspirv::binary::ParseAction::Error(Box::new(err));
                        }
                    }
                }
                rspirv::spirv::Op::Function => {
                    if !self.memory_model_state.is_seen() {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::FunctionBeforeMemoryModel,
                        ));
                    }
                    self.function_state = FunctionState::Inside;
                }
                rspirv::spirv::Op::SamplerImageAddressingModeNV => {
                    if self.sampler_image_address_mode.is_some() {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::DuplicateSamplerImageAddressingMode,
                        ));
                    }
                    let bit_width = match inst.operands.first() {
                        Some(rspirv::dr::Operand::LiteralBit32(value)) => *value,
                        Some(rspirv::dr::Operand::LiteralBit64(value)) => *value as u32,
                        _ => 0,
                    };
                    if bit_width != 32 && bit_width != 64 {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::InvalidSamplerImageAddressingModeBitWidth {
                                bit_width,
                            },
                        ));
                    }
                    self.sampler_image_address_mode = Some(bit_width);
                }
                _ => {}
            }

            if section < self.current_section {
                return rspirv::binary::ParseAction::Error(Box::new(
                    ValidationError::LayoutOutOfOrder { opcode },
                ));
            }
            self.current_section = self.current_section.max(section);
            if section > Section::MemoryModel && !self.memory_model_state.is_seen() {
                self.memory_model_state.record_violation(opcode);
            }
            rspirv::binary::ParseAction::Continue
        }
    }

    let mut checker = LayoutChecker::new();
    match rspirv::binary::parse_words(words, &mut checker) {
        Ok(()) => Ok(()),
        Err(rspirv::binary::ParseState::ConsumerError(err)) => {
            if let Some(validation) = err.downcast_ref::<ValidationError>() {
                Err(validation.clone())
            } else {
                Err(ValidationError::Parse(err.to_string()))
            }
        }
        Err(other) => Err(ValidationError::Parse(other.to_string())),
    }
}

fn instruction_section(current: Section, inst: &rspirv::dr::Instruction) -> Section {
    use rspirv::spirv::Op::*;
    let opcode = inst.class.opcode;
    let opname = inst.class.opname;
    if opname.starts_with("OpType")
        || opname.starts_with("OpConstant")
        || opname.starts_with("OpSpecConstant")
    {
        return Section::TypesGlobals;
    }
    match opcode {
        Capability | ConditionalCapabilityINTEL => Section::Capabilities,
        Extension | ConditionalExtensionINTEL => Section::Extensions,
        ExtInstImport => Section::ExtInstImport,
        MemoryModel => Section::MemoryModel,
        SamplerImageAddressingModeNV => Section::SamplerImageAddressMode,
        EntryPoint | ConditionalEntryPointINTEL => Section::EntryPoint,
        ExecutionMode | ExecutionModeId => Section::ExecutionMode,
        SourceContinued | Source | SourceExtension | String => Section::Debug1,
        Name | MemberName => Section::Debug2,
        ModuleProcessed => Section::Debug3,
        Decorate | DecorateId | MemberDecorate | DecorateString | MemberDecorateString
        | DecorationGroup | GroupDecorate | GroupMemberDecorate => Section::Annotations,
        Variable | UntypedVariableKHR => {
            if current == Section::TypesGlobals {
                Section::TypesGlobals
            } else {
                Section::Functions
            }
        }
        ExtInst | ExtInstWithForwardRefsKHR => {
            if current == Section::TypesGlobals {
                Section::TypesGlobals
            } else if current == Section::GraphDefinitions {
                Section::GraphDefinitions
            } else {
                Section::Functions
            }
        }
        Line | NoLine | Undef => {
            if current == Section::TypesGlobals {
                Section::TypesGlobals
            } else {
                Section::Functions
            }
        }
        Function | FunctionParameter | FunctionEnd => {
            if current == Section::FunctionDeclarations {
                Section::FunctionDeclarations
            } else {
                Section::Functions
            }
        }
        GraphEntryPointARM | GraphARM | GraphInputARM | GraphSetOutputARM | GraphEndARM => {
            Section::GraphDefinitions
        }
        CompositeExtract => {
            if current == Section::GraphDefinitions {
                Section::GraphDefinitions
            } else {
                Section::Functions
            }
        }
        _ => Section::Functions,
    }
}

fn capability_operand(inst: &rspirv::dr::Instruction) -> Option<rspirv::spirv::Capability> {
    inst.operands.iter().find_map(|operand| {
        if let rspirv::dr::Operand::Capability(cap) = operand {
            Some(*cap)
        } else {
            None
        }
    })
}

fn validate_capabilities(
    module: &Module,
    env: TargetEnv,
    extensions: &ExtensionSet,
) -> Result<(), ValidationError> {
    let declared: HashSet<_> = module
        .capabilities
        .iter()
        .filter_map(capability_operand)
        .collect();
    for inst in &module.capabilities {
        if let Some(capability) = capability_operand(inst) {
            if env.is_opencl()
                && matches!(
                    capability,
                    rspirv::spirv::Capability::LiteralSampler
                        | rspirv::spirv::Capability::Sampled1D
                        | rspirv::spirv::Capability::Image1D
                        | rspirv::spirv::Capability::SampledBuffer
                        | rspirv::spirv::Capability::ImageBuffer
                        | rspirv::spirv::Capability::ImageReadWrite
                )
                && !declared.contains(&rspirv::spirv::Capability::ImageBasic)
            {
                return Err(ValidationError::MissingRequiredCapability {
                    required_capability: rspirv::spirv::Capability::ImageBasic,
                    capability,
                });
            }
            if let Some(required_ext) = required_extension_for_capability(capability) {
                if !extensions
                    .values
                    .iter()
                    .any(|ext| ext.as_str() == required_ext)
                {
                    return Err(ValidationError::DisallowedCapabilityMissingExtension {
                        capability,
                        required_extension: required_ext.to_string(),
                    });
                }
            }
            let allowed_by_env = env.is_capability_allowed(capability);
            let allowed_by_extension = capability_allowed_by_extension(capability, extensions);
            let allowed_by_capability =
                capability_enabled_by_capability(env, capability, &declared);
            if !(allowed_by_env || allowed_by_extension || allowed_by_capability) {
                return Err(ValidationError::DisallowedCapability { capability, env });
            }
            if let Some(required_version) = required_spirv_version_for_capability(capability) {
                let target_version = env.spirv_version();
                if target_version < required_version {
                    return Err(ValidationError::CapabilityRequiresSpirvVersion {
                        capability,
                        required_version,
                        target_version,
                    });
                }
            }
            for required_cap in required_capabilities_for_capability(capability) {
                if !declared.contains(required_cap) {
                    return Err(ValidationError::MissingRequiredCapability {
                        required_capability: *required_cap,
                        capability,
                    });
                }
            }
        }
    }
    validate_instruction_requirements(module, &declared, extensions)?;
    Ok(())
}

fn capability_allowed_by_extension(
    capability: rspirv::spirv::Capability,
    extensions: &ExtensionSet,
) -> bool {
    required_extension_for_capability(capability)
        .map(|required_ext| {
            extensions
                .values
                .iter()
                .any(|ext| ext.as_str() == required_ext)
        })
        .unwrap_or(false)
}

fn capability_enabled_by_capability(
    env: TargetEnv,
    capability: rspirv::spirv::Capability,
    declared: &HashSet<rspirv::spirv::Capability>,
) -> bool {
    use rspirv::spirv::Capability::*;
    if !env.is_opencl() {
        return false;
    }
    if !declared.contains(&ImageBasic) {
        return false;
    }
    matches!(
        capability,
        LiteralSampler | Sampled1D | Image1D | SampledBuffer | ImageBuffer | ImageReadWrite
    )
}

fn validate_instruction_requirements(
    module: &Module,
    capabilities: &HashSet<rspirv::spirv::Capability>,
    extensions: &ExtensionSet,
) -> Result<(), ValidationError> {
    let target_version = module
        .header
        .as_ref()
        .map(|h| h.version)
        .unwrap_or_default();
    let target_version = SpirvVersion::from_word(target_version);
    for inst in module.all_inst_iter() {
        for &required_cap in inst.class.capabilities {
            if !capabilities.contains(&required_cap) {
                return Err(ValidationError::MissingInstructionCapability {
                    opcode: inst.class.opcode,
                    required_capability: required_cap,
                });
            }
        }
        for &required_ext in inst.class.extensions {
            if !extensions
                .values
                .iter()
                .any(|ext| ext.as_str() == required_ext)
            {
                return Err(ValidationError::MissingInstructionExtension {
                    opcode: inst.class.opcode,
                    required_extension: ExtensionName::from(required_ext),
                });
            }
        }
        if let Some(required_version) = required_spirv_version_for_opcode(inst.class.opcode) {
            if target_version < required_version {
                return Err(ValidationError::InstructionRequiresSpirvVersion {
                    opcode: inst.class.opcode,
                    required_version,
                    target_version,
                });
            }
        }
        for (index, operand) in inst.operands.iter().enumerate() {
            if matches!(operand, rspirv::dr::Operand::Capability(_)) {
                // Capability dependencies are validated separately to avoid over-constraining
                // the declaration order.
                continue;
            }
            for required_cap in operand.required_capabilities() {
                if !capabilities.contains(&required_cap) {
                    return Err(ValidationError::MissingOperandCapability {
                        opcode: inst.class.opcode,
                        operand_index: index,
                        required_capability: required_cap,
                    });
                }
            }
            for required_ext in operand.required_extensions() {
                if !extensions
                    .values
                    .iter()
                    .any(|ext| ext.as_str() == required_ext)
                {
                    return Err(ValidationError::MissingOperandExtension {
                        opcode: inst.class.opcode,
                        operand_index: index,
                        required_extension: ExtensionName::from(required_ext),
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_sampler_image_addressing_mode(
    module: &Module,
    capabilities: &HashSet<rspirv::spirv::Capability>,
) -> Result<(), ValidationError> {
    use rspirv::spirv::Op::SamplerImageAddressingModeNV;

    let mut declared_bit_width: Option<u32> = None;
    for inst in module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == SamplerImageAddressingModeNV)
    {
        if declared_bit_width.is_some() {
            return Err(ValidationError::DuplicateSamplerImageAddressingMode);
        }
        let bit_width = match inst.operands.first() {
            Some(rspirv::dr::Operand::LiteralBit32(value)) => *value,
            Some(rspirv::dr::Operand::LiteralBit64(value)) => *value as u32,
            _ => 0,
        };
        if bit_width != 32 && bit_width != 64 {
            return Err(ValidationError::InvalidSamplerImageAddressingModeBitWidth { bit_width });
        }
        declared_bit_width = Some(bit_width);
    }

    if capabilities.contains(&rspirv::spirv::Capability::BindlessTextureNV)
        && declared_bit_width.is_none()
    {
        return Err(ValidationError::MissingSamplerImageAddressingMode);
    }

    Ok(())
}

fn required_extension_for_capability(
    capability: rspirv::spirv::Capability,
) -> Option<&'static str> {
    use rspirv::spirv::Capability::*;
    match capability {
        BindlessTextureNV => Some("SPV_NV_bindless_texture"),
        RayTracingNV => Some("SPV_NV_ray_tracing"),
        RayTracingKHR => Some("SPV_KHR_ray_tracing"),
        RayQueryKHR => Some("SPV_KHR_ray_query"),
        RayTracingPositionFetchKHR => Some("SPV_KHR_ray_tracing_position_fetch"),
        CooperativeMatrixNV => Some("SPV_NV_cooperative_matrix"),
        MeshShadingNV => Some("SPV_NV_mesh_shader"),
        MeshShadingEXT => Some("SPV_EXT_mesh_shader"),
        FragmentShadingRateKHR => Some("SPV_KHR_fragment_shading_rate"),
        FragmentDensityEXT => Some("SPV_EXT_fragment_invocation_density"),
        FragmentShaderSampleInterlockEXT
        | FragmentShaderShadingRateInterlockEXT
        | FragmentShaderPixelInterlockEXT => Some("SPV_EXT_fragment_shader_interlock"),
        ImageFootprintNV => Some("SPV_NV_shader_image_footprint"),
        AtomicFloat32MinMaxEXT | AtomicFloat64MinMaxEXT | AtomicFloat16MinMaxEXT => {
            Some("SPV_EXT_shader_atomic_float_min_max")
        }
        AtomicFloat16AddEXT | AtomicFloat32AddEXT | AtomicFloat64AddEXT => {
            Some("SPV_EXT_shader_atomic_float_add")
        }
        AtomicFloat16VectorNV => Some("SPV_NV_shader_atomic_float"),
        ShaderSMBuiltinsNV => Some("SPV_NV_shader_sm_builtins"),
        TileShadingQCOM => Some("SPV_QCOM_tile_shading"),
        _ => None,
    }
}

fn required_spirv_version_for_capability(
    capability: rspirv::spirv::Capability,
) -> Option<SpirvVersion> {
    use rspirv::spirv::Capability::*;
    match capability {
        RayTracingKHR
        | RayTracingPositionFetchKHR
        | RayTracingNV
        | RayTracingMotionBlurNV
        | RayTracingOpacityMicromapEXT
        | RayTracingDisplacementMicromapNV
        | RayTracingSpheresGeometryNV
        | RayTracingLinearSweptSpheresGeometryNV
        | RayTracingClusterAccelerationStructureNV
        | RayQueryKHR
        | RayTracingProvisionalKHR => Some(SpirvVersion::new(1, 4)),
        MeshShadingEXT | MeshShadingNV => Some(SpirvVersion::new(1, 4)),
        FragmentShadingRateKHR | FragmentDensityEXT => Some(SpirvVersion::new(1, 5)),
        FragmentShaderSampleInterlockEXT
        | FragmentShaderShadingRateInterlockEXT
        | FragmentShaderPixelInterlockEXT => Some(SpirvVersion::new(1, 4)),
        AtomicFloat16AddEXT
        | AtomicFloat32AddEXT
        | AtomicFloat64AddEXT
        | AtomicFloat16MinMaxEXT
        | AtomicFloat32MinMaxEXT
        | AtomicFloat64MinMaxEXT
        | AtomicFloat16VectorNV => Some(SpirvVersion::new(1, 3)),
        TileShadingQCOM => Some(SpirvVersion::new(1, 6)),
        _ => None,
    }
}

fn required_spirv_version_for_extension(extension: &ExtensionName) -> Option<SpirvVersion> {
    let normalized = extension.as_str().to_ascii_lowercase();
    match normalized.as_str() {
        "spv_khr_vulkan_memory_model" | "spv_khr_workgroup_memory_explicit_layout" => {
            Some(SpirvVersion::new(1, 4))
        }
        "spv_khr_ray_tracing" | "spv_khr_ray_query" | "spv_khr_ray_tracing_position_fetch" => {
            Some(SpirvVersion::new(1, 4))
        }
        "spv_ext_fragment_shader_interlock" => Some(SpirvVersion::new(1, 4)),
        "spv_khr_fragment_shading_rate" | "spv_ext_fragment_invocation_density" => {
            Some(SpirvVersion::new(1, 5))
        }
        "spv_ext_descriptor_indexing" => Some(SpirvVersion::new(1, 5)),
        _ => None,
    }
}

fn required_spirv_version_for_opcode(opcode: rspirv::spirv::Op) -> Option<SpirvVersion> {
    match opcode {
        rspirv::spirv::Op::TypeAccelerationStructureKHR | rspirv::spirv::Op::TypeRayQueryKHR => {
            Some(SpirvVersion::new(1, 4))
        }
        _ => None,
    }
}

fn required_capabilities_for_capability(
    capability: rspirv::spirv::Capability,
) -> &'static [rspirv::spirv::Capability] {
    use rspirv::spirv::Capability::*;
    match capability {
        // Shader-based feature capabilities require the Shader capability.
        Geometry
        | Tessellation
        | MeshShadingNV
        | MeshShadingEXT
        | RayTracingNV
        | RayTracingKHR
        | RayQueryKHR
        | RayTracingMotionBlurNV
        | RayTracingOpacityMicromapEXT
        | RayTracingDisplacementMicromapNV
        | RayTracingSpheresGeometryNV
        | RayTracingLinearSweptSpheresGeometryNV
        | RayTracingClusterAccelerationStructureNV
        | RayTracingPositionFetchKHR
        | FragmentShadingRateKHR
        | FragmentDensityEXT
        | FragmentShaderSampleInterlockEXT
        | FragmentShaderShadingRateInterlockEXT
        | FragmentShaderPixelInterlockEXT
        | SampleRateShading
        | ImageFootprintNV
        | ShaderSMBuiltinsNV
        | AtomicFloat16AddEXT
        | AtomicFloat32AddEXT
        | AtomicFloat64AddEXT
        | AtomicFloat16MinMaxEXT
        | AtomicFloat32MinMaxEXT
        | AtomicFloat64MinMaxEXT
        | AtomicFloat16VectorNV
        | TileShadingQCOM => &[Shader],
        // OpenCL address-related capabilities require Kernel.
        Addresses | GenericPointer | DeviceEnqueue | Pipes => &[Kernel],
        _ => &[],
    }
}

fn extension_operand(inst: &rspirv::dr::Instruction) -> Option<ExtensionName> {
    inst.operands.iter().find_map(|operand| {
        if let rspirv::dr::Operand::LiteralString(extension) = operand {
            Some(ExtensionName::from(extension.as_str()))
        } else {
            None
        }
    })
}

fn validate_extensions(module: &Module, env: TargetEnv) -> Result<ExtensionSet, ValidationError> {
    let mut extensions = ExtensionSet::default();
    let target_version = env.spirv_version();
    for inst in &module.extensions {
        if let Some(extension) = extension_operand(inst) {
            let required_check = extension.clone();
            extensions.insert(extension, env)?;
            if let Some(required_version) = required_spirv_version_for_extension(&required_check) {
                if target_version < required_version {
                    return Err(ValidationError::ExtensionRequiresSpirvVersion {
                        extension: required_check,
                        required_version,
                        target_version,
                    });
                }
            }
        }
    }
    Ok(extensions)
}

fn member_decoration_target(inst: &rspirv::dr::Instruction) -> Option<MemberDecorationTargetId> {
    use rspirv::spirv::Op::*;
    match inst.class.opcode {
        MemberDecorate | MemberDecorateString => {
            let mut operands = inst.operands.iter();
            let target = operands.find_map(|op| {
                if let rspirv::dr::Operand::IdRef(id) = op {
                    DecorationTargetId::try_from(*id).ok()
                } else {
                    None
                }
            })?;
            let member_index = operands.find_map(|op| {
                if let rspirv::dr::Operand::LiteralBit32(member) = op {
                    Some(MemberIndex::new(*member))
                } else {
                    None
                }
            })?;
            Some(MemberDecorationTargetId::new(target, member_index))
        }
        _ => None,
    }
}

fn check_id(id: Id, bound: CheckedBound) -> Option<ValidationError> {
    if id.get() >= bound.validated().get() {
        Some(ValidationError::IdExceedsBound { id, bound })
    } else {
        None
    }
}

fn validate_memory_model(module: &Module) -> Result<(), ValidationError> {
    if module.memory_model.is_none() {
        return Err(ValidationError::MissingMemoryModel);
    }
    Ok(())
}

fn validate_id_bound(
    module: &Module,
    header: ValidatedHeader,
) -> Result<HashSet<ResultId>, ValidationError> {
    let bound = header.bound();
    let mut results: HashSet<ResultId> = HashSet::new();

    for instruction in module.all_inst_iter() {
        validate_instruction_ids(&mut results, instruction, bound)?;
    }

    Ok(results)
}

fn validate_member_decorations(
    module: &Module,
    defined_ids: &HashSet<ResultId>,
) -> Result<HashMap<ResultId, usize>, ValidationError> {
    let mut types: HashMap<ResultId, rspirv::spirv::Op> = HashMap::new();
    let mut struct_member_counts: HashMap<ResultId, usize> = HashMap::new();
    for inst in &module.types_global_values {
        if let Some(result_id) = inst.result_id {
            let id = ResultId::try_from(result_id).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::Result,
                opcode: inst.class.opcode,
            })?;
            if inst.class.opcode == rspirv::spirv::Op::TypeStruct {
                struct_member_counts.insert(id, inst.operands.len());
            }
            types.insert(id, inst.class.opcode);
        }
    }

    for inst in &module.annotations {
        if matches!(
            inst.class.opcode,
            rspirv::spirv::Op::MemberDecorate | rspirv::spirv::Op::MemberDecorateString
        ) {
            if let Some(target) = member_decoration_target(inst) {
                let target_id =
                    ResultId::try_from(u32::from(Id::from(target.target()))).map_err(|_| {
                        ValidationError::ZeroId {
                            kind: IdKind::Operand,
                            opcode: inst.class.opcode,
                        }
                    })?;
                if !defined_ids.contains(&target_id) {
                    return Err(ValidationError::MissingDecorationTarget {
                        target: target.target().into_inner().into_inner(),
                    });
                }
                let op = types.get(&target_id).copied();
                if op != Some(rspirv::spirv::Op::TypeStruct) {
                    return Err(ValidationError::MemberDecorationTargetNotStruct { target });
                }
                if let Some(member_count) = struct_member_counts.get(&target_id) {
                    if (target.member().0 as usize) >= *member_count {
                        return Err(ValidationError::MemberDecorationIndexOutOfRange {
                            target: target.target(),
                            member: target.member(),
                            member_count: *member_count,
                        });
                    }
                }
            }
        }
    }

    Ok(struct_member_counts)
}

fn validate_decoration_groups(
    module: &Module,
    defined_ids: &HashSet<ResultId>,
    opcodes: &HashMap<ResultId, rspirv::spirv::Op>,
    struct_member_counts: &HashMap<ResultId, usize>,
) -> Result<(), ValidationError> {
    let groups: HashSet<ResultId> = module
        .annotations
        .iter()
        .filter_map(|inst| {
            if inst.class.opcode == rspirv::spirv::Op::DecorationGroup {
                inst.result_id.and_then(|id| ResultId::try_from(id).ok())
            } else {
                None
            }
        })
        .collect();

    for inst in &module.annotations {
        match inst.class.opcode {
            rspirv::spirv::Op::GroupDecorate | rspirv::spirv::Op::GroupMemberDecorate => {
                let group = inst.operands.iter().find_map(|op| {
                    if let rspirv::dr::Operand::IdRef(id) = op {
                        ResultId::try_from(*id).ok()
                    } else {
                        None
                    }
                });
                if let Some(group) = group {
                    if !groups.contains(&group) {
                        return Err(ValidationError::UnknownDecorationGroup {
                            group: group.into_inner(),
                        });
                    }
                }
                let mut operands = inst.operands.iter().skip(1);
                while let Some(operand) = operands.next() {
                    if let rspirv::dr::Operand::IdRef(id) = operand {
                        let target =
                            ResultId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                                kind: IdKind::Operand,
                                opcode: inst.class.opcode,
                            })?;
                        if !defined_ids.contains(&target) {
                            return Err(ValidationError::MissingDecorationTarget {
                                target: target.into_inner(),
                            });
                        }
                        if inst.class.opcode == rspirv::spirv::Op::GroupMemberDecorate {
                            let member_index = operands
                                .next()
                                .and_then(|op| {
                                    if let rspirv::dr::Operand::LiteralBit32(member) = op {
                                        Some(*member)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(0);
                            if let Some(opcode) = opcodes.get(&target) {
                                if *opcode != rspirv::spirv::Op::TypeStruct {
                                    let target_operand = OperandId::try_from(u32::from(target))
                                        .expect("validated non-zero id");
                                    return Err(ValidationError::MemberDecorationTargetNotStruct {
                                        target: MemberDecorationTargetId::new(
                                            DecorationTargetId::new(target_operand),
                                            MemberIndex::new(member_index),
                                        ),
                                    });
                                }
                            }
                            if let Some(member_count) = struct_member_counts.get(&target) {
                                if (member_index as usize) >= *member_count {
                                    return Err(ValidationError::MemberDecorationIndexOutOfRange {
                                        target: DecorationTargetId::new(
                                            OperandId::try_from(u32::from(target)).unwrap(),
                                        ),
                                        member: MemberIndex::new(member_index),
                                        member_count: *member_count,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn validate_decorations(
    module: &Module,
    defined_ids: &HashSet<ResultId>,
) -> Result<(), ValidationError> {
    for inst in &module.annotations {
        match inst.class.opcode {
            rspirv::spirv::Op::Decorate | rspirv::spirv::Op::DecorateId => {
                let mut operands = inst.operands.iter();
                let target = operands.find_map(|op| {
                    if let rspirv::dr::Operand::IdRef(id) = op {
                        ResultId::try_from(*id).ok()
                    } else {
                        None
                    }
                });
                if let Some(target) = target {
                    if !defined_ids.contains(&target) {
                        return Err(ValidationError::MissingDecorationTarget {
                            target: target.into_inner(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn is_scalar_spec_constant(opcode: rspirv::spirv::Op) -> bool {
    matches!(
        opcode,
        rspirv::spirv::Op::SpecConstantTrue
            | rspirv::spirv::Op::SpecConstantFalse
            | rspirv::spirv::Op::SpecConstant
    )
}

fn is_constant_opcode(opcode: rspirv::spirv::Op) -> bool {
    matches!(
        opcode,
        rspirv::spirv::Op::Constant
            | rspirv::spirv::Op::ConstantTrue
            | rspirv::spirv::Op::ConstantFalse
            | rspirv::spirv::Op::SpecConstantTrue
            | rspirv::spirv::Op::SpecConstantFalse
            | rspirv::spirv::Op::SpecConstant
            | rspirv::spirv::Op::SpecConstantComposite
    )
}

fn is_memory_object_declaration(opcode: rspirv::spirv::Op) -> bool {
    matches!(
        opcode,
        rspirv::spirv::Op::Variable
            | rspirv::spirv::Op::UntypedVariableKHR
            | rspirv::spirv::Op::FunctionParameter
            | rspirv::spirv::Op::RawAccessChainNV
    )
}

fn is_pointer_type(
    type_id: Option<u32>,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    type_id
        .and_then(|id| ResultId::try_from(id).ok())
        .and_then(|id| definitions.get(&id))
        .map(|inst| {
            matches!(
                inst.class.opcode,
                rspirv::spirv::Op::TypePointer | rspirv::spirv::Op::TypeUntypedPointerKHR
            )
        })
        .unwrap_or(false)
}

fn validate_decoration_target_categories(
    module: &Module,
    opcodes: &HashMap<ResultId, rspirv::spirv::Op>,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    capabilities: &HashSet<rspirv::spirv::Capability>,
) -> Result<(), ValidationError> {
    for inst in &module.annotations {
        if !matches!(
            inst.class.opcode,
            rspirv::spirv::Op::Decorate | rspirv::spirv::Op::DecorateId
        ) {
            continue;
        }
        let mut operands = inst.operands.iter();
        let target = match operands.next() {
            Some(rspirv::dr::Operand::IdRef(id)) => {
                ResultId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                    kind: IdKind::Operand,
                    opcode: inst.class.opcode,
                })?
            }
            _ => continue,
        };
        let decoration = match operands.next() {
            Some(rspirv::dr::Operand::Decoration(dec)) => *dec,
            _ => continue,
        };
        let target_inst = match definitions.get(&target) {
            Some(inst) => inst,
            None => continue,
        };
        let opcode = match opcodes.get(&target) {
            Some(opcode) => *opcode,
            None => continue,
        };
        let target_id = Id::try_from(u32::from(target)).expect("non-zero id validated");
        let target_type_id = target_inst.result_type;

        let expected = match decoration {
            rspirv::spirv::Decoration::SpecId => {
                if !is_scalar_spec_constant(opcode) {
                    Some(DecorationTargetKind::ScalarSpecConstant)
                } else {
                    None
                }
            }
            rspirv::spirv::Decoration::Block
            | rspirv::spirv::Decoration::BufferBlock
            | rspirv::spirv::Decoration::GLSLShared
            | rspirv::spirv::Decoration::GLSLPacked
            | rspirv::spirv::Decoration::CPacked => {
                if opcode != rspirv::spirv::Op::TypeStruct {
                    Some(DecorationTargetKind::StructType)
                } else {
                    None
                }
            }
            rspirv::spirv::Decoration::ArrayStride => {
                if matches!(
                    opcode,
                    rspirv::spirv::Op::TypeArray
                        | rspirv::spirv::Op::TypeRuntimeArray
                        | rspirv::spirv::Op::TypePointer
                        | rspirv::spirv::Op::TypeUntypedPointerKHR
                ) {
                    None
                } else {
                    Some(DecorationTargetKind::ArrayOrPointerType)
                }
            }
            rspirv::spirv::Decoration::BuiltIn => {
                let builtin = operands.next().and_then(|op| {
                    if let rspirv::dr::Operand::BuiltIn(value) = op {
                        Some(*value)
                    } else if let rspirv::dr::Operand::LiteralBit32(raw) = op {
                        rspirv::spirv::BuiltIn::from_u32(*raw)
                    } else {
                        None
                    }
                });
                if capabilities.contains(&rspirv::spirv::Capability::Shader)
                    && builtin == Some(rspirv::spirv::BuiltIn::WorkgroupSize)
                    && !is_constant_opcode(opcode)
                {
                    Some(DecorationTargetKind::Constant)
                } else if matches!(
                    opcode,
                    rspirv::spirv::Op::Variable | rspirv::spirv::Op::UntypedVariableKHR
                ) || is_constant_opcode(opcode)
                {
                    None
                } else {
                    Some(DecorationTargetKind::Variable)
                }
            }
            rspirv::spirv::Decoration::NoPerspective
            | rspirv::spirv::Decoration::Flat
            | rspirv::spirv::Decoration::Patch
            | rspirv::spirv::Decoration::Centroid
            | rspirv::spirv::Decoration::Sample
            | rspirv::spirv::Decoration::Restrict
            | rspirv::spirv::Decoration::Aliased
            | rspirv::spirv::Decoration::Volatile
            | rspirv::spirv::Decoration::Coherent
            | rspirv::spirv::Decoration::NonWritable
            | rspirv::spirv::Decoration::NonReadable
            | rspirv::spirv::Decoration::XfbBuffer
            | rspirv::spirv::Decoration::XfbStride
            | rspirv::spirv::Decoration::Component
            | rspirv::spirv::Decoration::Stream
            | rspirv::spirv::Decoration::RestrictPointer
            | rspirv::spirv::Decoration::AliasedPointer
            | rspirv::spirv::Decoration::PerPrimitiveEXT => {
                if !is_memory_object_declaration(opcode) {
                    Some(DecorationTargetKind::MemoryObjectDeclaration)
                } else if !is_pointer_type(target_type_id, definitions) {
                    Some(DecorationTargetKind::Pointer)
                } else {
                    None
                }
            }
            rspirv::spirv::Decoration::Invariant
            | rspirv::spirv::Decoration::Constant
            | rspirv::spirv::Decoration::Location
            | rspirv::spirv::Decoration::Index
            | rspirv::spirv::Decoration::Binding
            | rspirv::spirv::Decoration::DescriptorSet
            | rspirv::spirv::Decoration::InputAttachmentIndex => {
                if matches!(
                    opcode,
                    rspirv::spirv::Op::Variable | rspirv::spirv::Op::UntypedVariableKHR
                ) {
                    None
                } else {
                    Some(DecorationTargetKind::Variable)
                }
            }
            rspirv::spirv::Decoration::LinkageAttributes => {
                if matches!(
                    opcode,
                    rspirv::spirv::Op::Function
                        | rspirv::spirv::Op::Variable
                        | rspirv::spirv::Op::UntypedVariableKHR
                ) {
                    None
                } else {
                    Some(DecorationTargetKind::FunctionOrVariable)
                }
            }
            _ => None,
        };

        if let Some(expected) = expected {
            return Err(ValidationError::InvalidDecorationTargetKind {
                decoration,
                target: target_id,
                found: opcode,
                expected,
            });
        }
    }
    Ok(())
}

fn validate_entry_points(
    module: &Module,
    defined_ids: &HashSet<ResultId>,
    opcodes: &HashMap<ResultId, rspirv::spirv::Op>,
) -> Result<HashSet<ResultId>, ValidationError> {
    let mut entry_points = HashSet::new();
    for ep in &module.entry_points {
        let mut operands = ep.operands.iter();
        // First operand is ExecutionModel; skip it.
        let _ = operands.next();
        let function_id = match operands.next() {
            Some(rspirv::dr::Operand::IdRef(id)) => {
                ResultId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                    kind: IdKind::Operand,
                    opcode: rspirv::spirv::Op::EntryPoint,
                })?
            }
            _ => continue,
        };
        if !defined_ids.contains(&function_id) {
            return Err(ValidationError::MissingEntryPointTarget {
                target: function_id.into_inner(),
            });
        }
        if let Some(opcode) = opcodes.get(&function_id) {
            if *opcode != rspirv::spirv::Op::Function {
                return Err(ValidationError::InvalidEntryPointTarget {
                    target: function_id.into_inner(),
                    opcode: *opcode,
                });
            }
        }
        entry_points.insert(function_id);
        // Skip the name operand.
        let _ = operands.next();
        for operand in operands {
            if let rspirv::dr::Operand::IdRef(id) = operand {
                let interface_id =
                    ResultId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                        kind: IdKind::Operand,
                        opcode: rspirv::spirv::Op::EntryPoint,
                    })?;
                if !defined_ids.contains(&interface_id) {
                    return Err(ValidationError::MissingEntryPointTarget {
                        target: interface_id.into_inner(),
                    });
                }
                if let Some(opcode) = opcodes.get(&interface_id) {
                    if *opcode != rspirv::spirv::Op::Variable {
                        return Err(ValidationError::InvalidEntryPointTarget {
                            target: interface_id.into_inner(),
                            opcode: *opcode,
                        });
                    }
                }
            }
        }
    }
    Ok(entry_points)
}

fn validate_execution_modes(
    module: &Module,
    entry_points: &HashSet<ResultId>,
) -> Result<(), ValidationError> {
    for mode in &module.execution_modes {
        let mut operands = mode.operands.iter();
        // First operand is the target entry point function.
        let Some(rspirv::dr::Operand::IdRef(target)) = operands.next() else {
            continue;
        };
        let function = ResultId::try_from(*target).map_err(|_| ValidationError::ZeroId {
            kind: IdKind::Operand,
            opcode: mode.class.opcode,
        })?;
        if !entry_points.contains(&function) {
            return Err(ValidationError::ExecutionModeWithoutEntryPoint {
                function: function.into_inner(),
            });
        }
    }
    Ok(())
}

fn collect_result_opcodes(module: &Module) -> HashMap<ResultId, rspirv::spirv::Op> {
    let mut map = HashMap::new();
    for inst in module.all_inst_iter() {
        if let Some(result_id) = inst.result_id {
            if let Ok(id) = ResultId::try_from(result_id) {
                map.insert(id, inst.class.opcode);
            }
        }
    }
    map
}

fn collect_result_instructions(module: &Module) -> HashMap<ResultId, rspirv::dr::Instruction> {
    let mut map = HashMap::new();
    for inst in module.all_inst_iter() {
        if let Some(result_id) = inst.result_id {
            if let Ok(id) = ResultId::try_from(result_id) {
                map.insert(id, inst.clone());
            }
        }
    }
    map
}

fn collect_declared_capabilities(module: &Module) -> HashSet<rspirv::spirv::Capability> {
    module
        .capabilities
        .iter()
        .filter_map(capability_operand)
        .collect()
}

fn validate_instruction_ids(
    results: &mut HashSet<ResultId>,
    instruction: &rspirv::dr::Instruction,
    bound: CheckedBound,
) -> Result<(), ValidationError> {
    let opcode = instruction.class.opcode;
    if let Some(id) = instruction.result_id {
        let valid_id = ResultId::try_from(id).map_err(|_| ValidationError::ZeroId {
            kind: IdKind::Result,
            opcode,
        })?;
        if !results.insert(valid_id) {
            return Err(ValidationError::DuplicateResultId {
                id: valid_id.into_inner(),
            });
        }
        if let Some(error) = check_id(valid_id.into_inner(), bound) {
            return Err(error);
        }
    }
    if let Some(result_type) = instruction.result_type {
        let result_type = TypeId::try_from(result_type).map_err(|_| ValidationError::ZeroId {
            kind: IdKind::ResultType,
            opcode,
        })?;
        if let Some(error) = check_id(result_type.into_inner(), bound) {
            return Err(error);
        }
    }
    for operand in &instruction.operands {
        if let rspirv::dr::Operand::IdRef(id) = operand {
            let operand_id = OperandId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::Operand,
                opcode,
            })?;
            let id = Id::from(operand_id);
            if let Some(error) = check_id(id, bound) {
                return Err(error);
            }
        }
    }

    if let Some(member_target) = member_decoration_target(instruction) {
        if let Some(error) = check_id(member_target.target().into_inner().into_inner(), bound) {
            return Err(error);
        }
        // Member index itself is a literal and does not participate in bound checking.
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        validate_module, CheckedBound, DeclaredBound, DecorationTargetId, DecorationTargetKind,
        ExtensionName, Id, IdKind, MaybeValidModule, MemberDecorationTargetId, MemberIndex,
        ModuleWords, OperandId, Schema, SpirvVersion, ValidModuleCache, ValidatableModule,
        ValidationError,
    };
    use crate::assembly::assemble_text;
    use crate::target_env::TargetEnv;
    use std::num::NonZeroU32;
    use std::sync::Arc;

    fn op(word_count: u16, opcode: u16) -> u32 {
        ((word_count as u32) << 16) | opcode as u32
    }

    #[test]
    fn validate_module_rejects_missing_header() {
        let binary = vec![0x07230203, 0, 0, 0, 0];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::MissingMemoryModel);
    }

    #[test]
    fn validate_module_rejects_ids_beyond_bound() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
        ]
        .join("\n");
        let mut binary = assemble_text(&text).expect("assemble");
        // Clamp the declared id bound to 1, which is lower than any type id emitted.
        binary[3] = 1;
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::IdExceedsBound {
                id: Id::new(NonZeroU32::new(1).unwrap()),
                bound: CheckedBound::new(DeclaredBound(1)).unwrap()
            }
        );
    }

    #[test]
    fn validate_module_requires_memory_model() {
        let text = [
            "OpCapability Shader",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::InstructionBeforeMemoryModel {
                opcode: rspirv::spirv::Op::TypeVoid,
            }
        );
    }

    #[test]
    fn names_section_must_follow_debug_section() {
        // OpName (Names section) precedes OpSource (Debug section), which should trigger an ordering error.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            0x00030005, // OpName %1 "x" (names)
            1,
            0x0000_0078,
            op(3, 3), // OpSource Unknown 0 (debug section after names -> error)
            0,
            0,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Source
            }
        );
    }

    #[test]
    fn annotations_must_follow_names() {
        // OpDecorate (Annotations) placed before OpName (Names) should trigger ordering diagnostics.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 71), // OpDecorate %1 RelaxedPrecision (annotations after names -> error)
            1,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            0x00030005, // OpName %1 "x" (names)
            1,
            0x0000_0078,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Name
            }
        );
    }

    #[test]
    fn execution_mode_must_follow_entry_point() {
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 16), // OpExecutionMode %1 OriginUpperLeft (before EntryPoint)
            1,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(5, 15), // OpEntryPoint Fragment %1 "main"
            rspirv::spirv::ExecutionModel::Fragment as u32,
            1,
            0x6e69616d,
            0,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::EntryPoint
            }
        );
    }

    #[test]
    fn sampler_image_address_mode_must_precede_entry_points() {
        // The text assembler rejects this ordering, so keep a hand-crafted binary with
        // OpSamplerImageAddressingModeNV placed after OpEntryPoint to exercise the validator.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version 1.0
            0,          // generator
            5,          // bound (ids 1..4)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // OpCapability BindlessTextureNV
            rspirv::spirv::Capability::BindlessTextureNV as u32,
            op(7, 10), // OpExtension "SPV_NV_bindless_texture"
            0x5f56_5053,
            0x625f_564e,
            0x6c64_6e69,
            0x5f73_7365,
            0x7478_6574,
            0x0065_7275,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(5, 15), // OpEntryPoint GLCompute %3 "main"
            rspirv::spirv::ExecutionModel::GLCompute as u32,
            3,
            0x6e69616d,
            0,
            op(2, rspirv::spirv::Op::SamplerImageAddressingModeNV as u16), // OpSamplerImageAddressingModeNV 64 (misordered after entry point)
            64,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::SamplerImageAddressingModeNV
            }
        );
    }

    #[test]
    fn sampler_image_address_mode_is_required_when_bindless_capability_declared() {
        // BindlessTextureNV requires a single SamplerImageAddressingModeNV declaration.
        let text = [
            "OpCapability Shader",
            "OpCapability BindlessTextureNV",
            "OpExtension \"SPV_NV_bindless_texture\"",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint GLCompute %func \"main\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%func = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let expected = ValidationError::MissingSamplerImageAddressingMode;

        let text_error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("sampler image address mode is required for bindless capability");
        assert_eq!(text_error, expected);

        let binary = assemble_text(&text).expect("assemble");
        let binary_error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .expect_err("binary should also require sampler image address mode");
        assert_eq!(binary_error, expected);
    }

    #[test]
    fn sampler_image_address_mode_rejects_invalid_bit_width() {
        // The assembler enforces valid bit widths, so use a hand-built binary with an invalid value.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version 1.0
            0,          // generator
            5,          // bound (ids 1..4)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // OpCapability BindlessTextureNV
            rspirv::spirv::Capability::BindlessTextureNV as u32,
            op(7, 10), // OpExtension "SPV_NV_bindless_texture"
            0x5f56_5053,
            0x625f_564e,
            0x6c64_6e69,
            0x5f73_7365,
            0x7478_6574,
            0x0065_7275,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, rspirv::spirv::Op::SamplerImageAddressingModeNV as u16), // invalid bit width
            16,
            op(5, 15), // OpEntryPoint GLCompute %3 "main"
            rspirv::spirv::ExecutionModel::GLCompute as u32,
            3,
            0x6e69616d,
            0,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let expected = ValidationError::InvalidSamplerImageAddressingModeBitWidth { bit_width: 16 };
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, expected);
    }

    #[test]
    fn sampler_image_address_mode_rejects_duplicates() {
        // Keep two declarations in the binary to bypass assembler canonicalization.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version 1.0
            0,          // generator
            6,          // bound (ids 1..5)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // OpCapability BindlessTextureNV
            rspirv::spirv::Capability::BindlessTextureNV as u32,
            op(7, 10), // OpExtension "SPV_NV_bindless_texture"
            0x5f56_5053,
            0x625f_564e,
            0x6c64_6e69,
            0x5f73_7365,
            0x7478_6574,
            0x0065_7275,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, rspirv::spirv::Op::SamplerImageAddressingModeNV as u16), // first declaration
            64,
            op(2, rspirv::spirv::Op::SamplerImageAddressingModeNV as u16), // duplicate
            64,
            op(5, 15), // OpEntryPoint GLCompute %3 "main"
            rspirv::spirv::ExecutionModel::GLCompute as u32,
            3,
            0x6e69616d,
            0,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::DuplicateSamplerImageAddressingMode);
    }

    #[test]
    fn validate_module_detects_duplicate_result_ids() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeVoid",
            "%1 = OpTypeInt 32 0",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::DuplicateResultId {
                id: Id::new(NonZeroU32::new(1).unwrap())
            }
        );
        let error = MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::DuplicateResultId {
                id: Id::new(NonZeroU32::new(1).unwrap())
            }
        );
    }

    #[test]
    fn validate_module_accepts_valid_binary() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .expect("valid module");
        MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .expect("valid module");
    }

    #[test]
    fn validate_module_checks_operand_ids_against_bound() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let mut binary = assemble_text(&text).expect("assemble");
        // Force a bound that is too small for the function type/result ids.
        binary[3] = 2;
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::IdExceedsBound {
                id: Id::new(NonZeroU32::new(2).unwrap()),
                bound: CheckedBound::new(DeclaredBound(2)).unwrap(),
            }
        );
    }

    #[test]
    fn validate_module_rejects_memory_model_after_function() {
        // The assembler canonicalizes layout, so build a binary with OpMemoryModel placed after the
        // function body to exercise the layout check directly.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version 1.0
            0,          // generator
            5,          // bound
            0,          // schema
            op(2, 17),  // OpCapability Shader
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %3
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(3, 14),  // OpMemoryModel Logical GLSL450 (misordered)
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::FunctionBeforeMemoryModel);
    }

    #[test]
    fn validate_module_rejects_duplicate_memory_model() {
        // The text path drops duplicate memory models, so keep a hand-built binary to assert the
        // validator rejects them.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            op(2, 17), // OpCapability Shader
            1,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 14), // Duplicate OpMemoryModel
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::DuplicateMemoryModel);
    }

    #[test]
    fn capability_must_appear_before_types() {
        // The assembler reorders sections, so preserve the out-of-order capability via binary.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(2, 17), // OpCapability Shader (out of order)
            rspirv::spirv::Capability::Shader as u32,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Capability
            }
        );
    }

    #[test]
    fn extension_must_precede_types_and_globals() {
        // Keep the extension misordered in binary form; the assembler canonicalizes this section.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            6,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(6, 10), // OpExtension "SPV_KHR_ray_tracing" (misordered)
            0x5f56_5053,
            0x5f52_484b,
            0x5f79_6172,
            0x6361_7274,
            0x0067_6e69,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn capability_cannot_follow_extension_section() {
        // Extensions must precede additional capabilities; a capability after an extension is out of order.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            op(6, 10), // OpExtension "SPV_KHR_ray_tracing"
            0x5f56_5053,
            0x5f52_484b,
            0x5f79_6172,
            0x6361_7274,
            0x0067_6e69,
            op(2, 17), // OpCapability Shader (out of order after extension)
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Capability
            }
        );
    }

    #[test]
    fn ext_inst_import_must_precede_types_and_globals() {
        // Place OpExtInstImport after a type to trigger layout ordering.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            3,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            0x0006000b, // OpExtInstImport %2 "GLSL.std.450" (misordered)
            2,
            0x4c53_4c47,
            0x2e73_7464,
            0x3035_342e,
            0,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ExtInstImport
            }
        );
    }

    #[test]
    fn ext_inst_import_cannot_follow_memory_model() {
        // The assembler canonicalizes layout, so construct the binary manually to keep
        // OpExtInstImport after OpMemoryModel and ensure the validator flags it.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            3,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            0x0006000b, // OpExtInstImport %2 "GLSL.std.450" (too late)
            2,
            0x4c53_4c47,
            0x2e73_7464,
            0x3035_342e,
            0,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ExtInstImport
            }
        );
    }

    #[test]
    fn validate_module_rejects_duplicate_capability() {
        // The assembler deduplicates capabilities; construct the binary manually to keep both.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // Duplicate OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DuplicateCapability {
                capability: rspirv::spirv::Capability::Shader
            }
        );
    }

    #[test]
    fn vulkan_rejects_kernel_capability() {
        let text = [
            "OpCapability Kernel",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("Vulkan should reject Kernel capability");
        assert_eq!(
            error,
            ValidationError::DisallowedCapability {
                capability: rspirv::spirv::Capability::Kernel,
                env: TargetEnv::Vulkan1_2
            }
        );
    }

    #[test]
    fn vulkan_rejects_opencl_only_capabilities() {
        let text = [
            "OpCapability DeviceEnqueue",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("Vulkan should reject DeviceEnqueue capability");
        assert_eq!(
            error,
            ValidationError::DisallowedCapability {
                capability: rspirv::spirv::Capability::DeviceEnqueue,
                env: TargetEnv::Vulkan1_2
            }
        );
    }

    #[test]
    fn vulkan_1_0_rejects_group_non_uniform() {
        let text = [
            "OpCapability Shader",
            "OpCapability GroupNonUniform",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("GroupNonUniform is optional from Vulkan 1.1+");
        assert_eq!(
            error,
            ValidationError::DisallowedCapability {
                capability: rspirv::spirv::Capability::GroupNonUniform,
                env: TargetEnv::Vulkan1_0
            }
        );
    }

    #[test]
    fn vulkan_1_1_allows_group_non_uniform() {
        let text = [
            "OpCapability Shader",
            "OpCapability GroupNonUniform",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let module = text
            .as_str()
            .validate(TargetEnv::Vulkan1_1)
            .expect("Vulkan 1.1 allows GroupNonUniform");
        assert_eq!(module.env(), TargetEnv::Vulkan1_1);
    }

    #[test]
    fn vulkan_1_0_rejects_vulkan_memory_model() {
        let text = [
            "OpCapability Shader",
            "OpCapability VulkanMemoryModel",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("Vulkan memory model is 1.2+ optional");
        assert_eq!(
            error,
            ValidationError::DisallowedCapability {
                capability: rspirv::spirv::Capability::VulkanMemoryModel,
                env: TargetEnv::Vulkan1_0
            }
        );
    }

    #[test]
    fn vulkan_1_2_allows_physical_storage_buffer_addresses() {
        let text = [
            "OpCapability Shader",
            "OpCapability PhysicalStorageBufferAddresses",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let module = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("PhysicalStorageBufferAddresses is optional in Vulkan 1.2");
        assert_eq!(module.env(), TargetEnv::Vulkan1_2);
    }

    #[test]
    fn opencl_allows_optional_float64() {
        let text = [
            "OpCapability Kernel",
            "OpCapability Addresses",
            "OpCapability Float64",
            "OpMemoryModel Physical32 OpenCL",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        text.as_str()
            .validate(TargetEnv::OpenCl1_2)
            .expect("Float64 is optional in OpenCL 1.2");
    }

    #[test]
    fn opencl_embedded_allows_optional_float64() {
        let text = [
            "OpCapability Kernel",
            "OpCapability Addresses",
            "OpCapability Float64",
            "OpMemoryModel Physical32 OpenCL",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        text.as_str()
            .validate(TargetEnv::OpenClEmbedded1_2)
            .expect("Float64 is optional in OpenCL 1.2 embedded");
    }

    #[test]
    fn webgpu_rejects_non_shader_capabilities() {
        let text = [
            "OpCapability Kernel",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::WebGpu0)
            .expect_err("WebGPU should reject Kernel capability");
        assert_eq!(
            error,
            ValidationError::DisallowedCapability {
                capability: rspirv::spirv::Capability::Kernel,
                env: TargetEnv::WebGpu0
            }
        );
    }

    #[test]
    fn opencl_rejects_shader_capability() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical OpenCL",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::OpenCl2_2)
            .expect_err("OpenCL should reject Shader capability");
        assert_eq!(
            error,
            ValidationError::DisallowedCapability {
                capability: rspirv::spirv::Capability::Shader,
                env: TargetEnv::OpenCl2_2
            }
        );
    }

    #[test]
    fn opencl_rejects_vulkan_specific_extension() {
        let text = [
            "OpCapability Kernel",
            "OpExtension \"SPV_KHR_vulkan_memory_model\"",
            "OpMemoryModel Logical OpenCL",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::OpenCl2_2)
            .expect_err("OpenCL should reject Vulkan-specific extensions");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
                env: TargetEnv::OpenCl2_2
            }
        );
    }

    #[test]
    fn universal_rejects_vulkan_specific_extension() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_vulkan_memory_model\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("Universal env should reject Vulkan-only extension");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn vulkan_memory_model_extension_requires_spirv_1_4() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_vulkan_memory_model\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("SPIR-V 1.4 is required for Vulkan memory model extension");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
                required_version: SpirvVersion::new(1, 4),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        // A newer environment should accept the extension.
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.4+");
    }

    #[test]
    fn ray_tracing_extensions_require_spirv_1_4() {
        for ext in &[
            "SPV_KHR_ray_tracing",
            "SPV_KHR_ray_query",
            "SPV_KHR_ray_tracing_position_fetch",
        ] {
            let text = [
                "OpCapability Shader",
                &format!("OpExtension \"{ext}\""),
                "OpMemoryModel Logical GLSL450",
                "%void = OpTypeVoid",
                "%fn = OpTypeFunction %void",
                "%main = OpFunction %void None %fn",
                "%entry = OpLabel",
                "OpReturn",
                "OpFunctionEnd",
            ]
            .join("\n");
            let error = text
                .as_str()
                .validate(TargetEnv::Vulkan1_0)
                .expect_err("extension should require SPIR-V 1.4");
            assert_eq!(
                error,
                ValidationError::ExtensionRequiresSpirvVersion {
                    extension: ExtensionName::from(*ext),
                    required_version: SpirvVersion::new(1, 4),
                    target_version: TargetEnv::Vulkan1_0.spirv_version(),
                }
            );

            text.as_str()
                .validate(TargetEnv::Vulkan1_2)
                .expect("extension should be accepted with SPIR-V 1.4+");
        }
    }

    #[test]
    fn descriptor_indexing_extension_requires_spirv_1_5() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_EXT_descriptor_indexing\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("descriptor indexing should require SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_EXT_descriptor_indexing"),
                required_version: SpirvVersion::new(1, 5),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.5+");
    }

    #[test]
    fn fragment_shader_interlock_extension_requires_spirv_1_4() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_EXT_fragment_shader_interlock\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("fragment shader interlock should require SPIR-V 1.4");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_EXT_fragment_shader_interlock"),
                required_version: SpirvVersion::new(1, 4),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.4+");
    }

    #[test]
    fn non_opencl_env_rejects_opencl_extension() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_opencl_enqueue\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("Vulkan should reject OpenCL-specific extension");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_opencl_enqueue"),
                env: TargetEnv::Vulkan1_2
            }
        );
    }

    #[test]
    fn vulkan_allows_optional_geometry_capability() {
        let text = [
            "OpCapability Shader",
            "OpCapability Geometry",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let module = text
            .as_str()
            .validate(TargetEnv::Vulkan1_0)
            .expect("optional Vulkan capability should be permitted");
        assert_eq!(module.env(), TargetEnv::Vulkan1_0);
    }

    #[test]
    fn opencl_requires_image_basic_for_image_capabilities() {
        let text = [
            "OpCapability Kernel",
            "OpCapability Addresses",
            "OpCapability Image1D",
            "OpMemoryModel Physical32 OpenCL",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::OpenCl2_0)
            .expect_err("Image1D requires ImageBasic");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::ImageBasic,
                capability: rspirv::spirv::Capability::Image1D
            }
        );

        let text_with_basic = [
            "OpCapability Kernel",
            "OpCapability Addresses",
            "OpCapability ImageBasic",
            "OpCapability Image1D",
            "OpMemoryModel Physical32 OpenCL",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        text_with_basic
            .as_str()
            .validate(TargetEnv::OpenCl2_0)
            .expect("ImageBasic enables other image capabilities");
    }

    #[test]
    fn opencl_embedded_rejects_int64_capability() {
        let text = [
            "OpCapability Kernel",
            "OpCapability Int64",
            "OpMemoryModel Physical32 OpenCL",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::OpenClEmbedded1_2)
            .expect_err("embedded OpenCL should reject Int64");
        assert_eq!(
            error,
            ValidationError::DisallowedCapability {
                capability: rspirv::spirv::Capability::Int64,
                env: TargetEnv::OpenClEmbedded1_2
            }
        );
    }

    #[test]
    fn valid_module_cache_reuses_entries() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let mut cache = ValidModuleCache::default();
        let first = cache
            .validate_words(&binary, TargetEnv::Universal1_6)
            .expect("first validation");
        let second = cache
            .validate_words(&binary, TargetEnv::Universal1_6)
            .expect("cached validation");
        assert_eq!(
            Arc::as_ptr(&first),
            Arc::as_ptr(&second),
            "cached entries should reuse the same allocation"
        );
    }

    #[test]
    fn execution_mode_requires_entry_point() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "OpExecutionMode %main OriginUpperLeft",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("execution modes must target entry points");
        assert_eq!(
            error,
            ValidationError::ExecutionModeWithoutEntryPoint {
                function: Id::try_from(3).unwrap()
            }
        );
    }

    #[test]
    fn execution_mode_must_target_entry_point_function() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Fragment %main \"main\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%helper = OpFunction %void None %fn",
            "%hentry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "OpExecutionMode %helper OriginUpperLeft",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("execution mode should target entry point");
        assert!(matches!(
            error,
            ValidationError::ExecutionModeWithoutEntryPoint { function }
                if function == Id::try_from(4).unwrap()
        ));
    }

    #[test]
    fn execution_mode_accepts_entry_point() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Fragment %main \"main\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "OpExecutionMode %main OriginUpperLeft",
        ]
        .join("\n");
        text.as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("execution mode targets entry point");
    }

    #[test]
    fn capability_requiring_extension_must_declare_it() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("capability should require extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::RayTracingKHR,
                required_extension: "SPV_KHR_ray_tracing".to_string()
            }
        );

        let text_with_extension = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let validated = text_with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension present should allow capability");
        assert_eq!(validated.header().schema(), Schema::ZERO);
    }

    #[test]
    fn validate_module_rejects_duplicate_extension() {
        // Hand-assemble a module with duplicate OpExtension instructions.
        let extension_word = 0x0006_000a; // word count 6, opcode OpExtension (10)
        let extension_words = [
            0x5f56_5053, // "SPV_"
            0x5f52_484b, // "KHR_"
            0x5f79_6172, // "ray_"
            0x6361_7274, // "trac"
            0x0067_6e69, // "ing\0"
        ];
        let binary = [
            0x0723_0203, // magic
            0x0001_0000, // version
            0,           // generator
            6,           // bound (ids up to 5)
            0,           // schema
            0x0002_0011, // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            extension_word,
            extension_words[0],
            extension_words[1],
            extension_words[2],
            extension_words[3],
            extension_words[4],
            extension_word, // duplicate extension
            extension_words[0],
            extension_words[1],
            extension_words[2],
            extension_words[3],
            extension_words[4],
            0x0003_000e, // OpMemoryModel Logical GLSL450
            0,
            1,
            0x0002_0013, // OpTypeVoid %1
            1,
        ];
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::DuplicateExtension {
                extension: ExtensionName::from("SPV_KHR_ray_tracing")
            }
        );
    }

    #[test]
    fn webgpu_disallows_extensions_for_text_and_binary() {
        let module_text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        let expected_error = ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_ray_tracing"),
            env: TargetEnv::WebGpu0,
        };

        let text_error = module_text
            .as_str()
            .validate(TargetEnv::WebGpu0)
            .unwrap_err();
        assert_eq!(text_error, expected_error);

        let binary = assemble_text(&module_text).expect("assemble");
        let binary_error = binary.as_slice().validate(TargetEnv::WebGpu0).unwrap_err();
        assert_eq!(binary_error, expected_error);

        let validated = binary
            .as_slice()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted for Vulkan environments");
        assert_eq!(validated.header().schema(), Schema::ZERO);
    }

    #[test]
    fn capability_requires_min_spirv_version() {
        let text = [
            "OpCapability Shader",
            "OpCapability FragmentShadingRateKHR",
            "OpExtension \"SPV_KHR_fragment_shading_rate\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_3)
            .expect_err("requires SPIR-V 1.5");
        match error {
            ValidationError::CapabilityRequiresSpirvVersion {
                capability,
                required_version,
                target_version,
            } => {
                assert_eq!(
                    capability,
                    rspirv::spirv::Capability::FragmentShadingRateKHR
                );
                assert_eq!(required_version, SpirvVersion::new(1, 5));
                assert_eq!(target_version, SpirvVersion::new(1, 3));
            }
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(
                    extension,
                    ExtensionName::from("SPV_KHR_fragment_shading_rate")
                );
                assert_eq!(required_version, SpirvVersion::new(1, 5));
                assert_eq!(target_version, SpirvVersion::new(1, 3));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        text.as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("succeeds on newer SPIR-V");
    }

    #[test]
    fn instruction_requires_capability() {
        use rspirv::{binary::Assemble, dr::Builder};
        let mut builder = Builder::new();
        builder.set_version(1, 5);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        builder.type_acceleration_structure_khr();
        let module = builder.module();
        let words = module.assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("missing RayTracingKHR capability");
        assert_eq!(
            error,
            ValidationError::MissingInstructionCapability {
                opcode: rspirv::spirv::Op::TypeAccelerationStructureKHR,
                required_capability: rspirv::spirv::Capability::RayTracingNV
            }
        );
    }

    #[test]
    fn operand_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%i32 = OpTypeInt 32 0",
            "%ptr_fn_i32 = OpTypePointer Function %i32",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "%var = OpVariable %ptr_fn_i32 Function",
            "%val = OpLoad %i32 %var NonPrivatePointer",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("missing vulkan memory model capability");
        assert_eq!(
            error,
            ValidationError::MissingOperandCapability {
                opcode: rspirv::spirv::Op::Load,
                operand_index: 1,
                required_capability: rspirv::spirv::Capability::VulkanMemoryModel
            }
        );
    }

    #[test]
    fn instruction_requires_spirv_version() {
        use rspirv::{binary::Assemble, dr::Builder};
        let mut builder = Builder::new();
        builder.set_version(1, 3);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.capability(rspirv::spirv::Capability::RayQueryKHR);
        builder.extension("SPV_KHR_ray_query");
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        builder.type_ray_query_khr();
        let module = builder.module();
        let words = module.assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("Ray query requires SPIR-V 1.4+");
        assert_eq!(
            error,
            ValidationError::InstructionRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::TypeRayQueryKHR,
                required_version: SpirvVersion::new(1, 4),
                target_version: SpirvVersion::new(1, 3),
            }
        );
    }

    #[test]
    fn validate_module_rejects_zero_result_id() {
        // The assembler never emits id 0; keep this binary hand-crafted to drive the zero-id path.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid with a zero result id
            0,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::ZeroId {
                kind: IdKind::Result,
                opcode: rspirv::spirv::Op::TypeVoid
            }
        );
    }

    #[test]
    fn validate_module_rejects_zero_operand_id() {
        // Text assembly forbids %0 operands, so build the binary directly to cover the check.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %0 (invalid operand id)
            2,
            0,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::ZeroId {
                kind: IdKind::Operand,
                opcode: rspirv::spirv::Op::TypeFunction
            }
        );
    }

    #[test]
    fn validate_module_rejects_non_zero_schema() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
        ]
        .join("\n");
        let mut binary = assemble_text(&text).expect("assemble");
        // Reserved word must be zero; flip it to a non-zero value to trigger the validation error.
        binary[4] = 1;
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::InvalidReservedWord { reserved: 1 });
    }

    #[test]
    fn member_decorate_requires_struct_target() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%u32 = OpTypeInt 32 0",
            "OpMemberDecorate %u32 0 Offset 0",
        ]
        .join("\n");
        let error = MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::MemberDecorationTargetNotStruct {
                target: MemberDecorationTargetId::new(
                    DecorationTargetId::try_from(1).unwrap(),
                    MemberIndex::new(0)
                )
            }
        );
    }

    #[test]
    fn member_decorate_requires_valid_member_index() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%u32 = OpTypeInt 32 0",
            "%vec2 = OpTypeStruct %u32 %u32",
            "OpMemberDecorate %vec2 2 Offset 0",
        ]
        .join("\n");
        let error = MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::MemberDecorationIndexOutOfRange {
                target: DecorationTargetId::try_from(2).unwrap(),
                member: MemberIndex::new(2),
                member_count: 2
            }
        );
    }

    #[test]
    fn group_decorate_requires_declared_group() {
        // The text assembler refuses to emit binaries with invalid decoration groups, so we
        // hand-build the binary to drive the validator directly.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            2,          // bound (ids up to 1)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            0x0006000b, // OpExtInstImport %1 "GLSL.std.450"
            1,
            0x4c53_4c47,
            0x2e73_7464,
            0x3035_342e,
            0,         // null terminator for the import string
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            0x0003004a, // OpGroupDecorate %1 %1 (invalid group id)
            1,
            1,
        ];
        let expected = ValidationError::UnknownDecorationGroup {
            group: Id::try_from(1).unwrap(),
        };
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(error, expected);
    }

    #[test]
    fn decorate_requires_declared_target() {
        // The text assembler enforces target existence up front, so use a binary to ensure the
        // validator catches the missing target.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            3,          // bound (ids up to 2)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            0x00030047, // OpDecorate %2 RelaxedPrecision (target %2 is undefined)
            2,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
        ];
        let expected = ValidationError::MissingDecorationTarget {
            target: Id::try_from(2).unwrap(),
        };
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(error, expected);
    }

    #[test]
    fn group_member_decorate_requires_declared_targets() {
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            5,          // bound (ids up to 4)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 73), // OpDecorationGroup %1
            1,
            op(4, 75), // OpGroupMemberDecorate %1 %4 0 (target %4 is undefined)
            1,
            4,
            0,
            op(2, 19), // OpTypeVoid %2
            2,
            op(3, 33), // OpTypeFunction %3 %2
            3,
            2,
        ];
        let expected = ValidationError::MissingDecorationTarget {
            target: Id::try_from(4).unwrap(),
        };
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(error, expected);
    }

    #[test]
    fn group_member_decorate_requires_struct_targets() {
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            6,          // bound (ids up to 5)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 73), // OpDecorationGroup %1
            1,
            op(4, 75), // OpGroupMemberDecorate %1 %4 0 (%4 is not a struct)
            1,
            4,
            0,
            op(4, 21), // OpTypeInt %2 32
            2,
            32,
            0,
            op(3, 33), // OpTypeFunction %3 %2
            3,
            2,
            op(3, 22), // OpTypeFloat %4 32 (non-struct target)
            4,
            32,
        ];
        let expected = ValidationError::MemberDecorationTargetNotStruct {
            target: MemberDecorationTargetId::new(
                DecorationTargetId::new(OperandId::try_from(4u32).unwrap()),
                MemberIndex::new(0),
            ),
        };
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(error, expected);
    }

    #[test]
    fn spec_id_requires_scalar_specialization_constant() {
        let text = [
            "OpCapability Addresses",
            "OpCapability Kernel",
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeInt 32 1",
            "%2 = OpConstant %1 1",
            "OpDecorate %2 SpecId 7",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble spec id decoration");
        let expected = ValidationError::InvalidDecorationTargetKind {
            decoration: rspirv::spirv::Decoration::SpecId,
            target: Id::try_from(2).unwrap(),
            found: rspirv::spirv::Op::Constant,
            expected: DecorationTargetKind::ScalarSpecConstant,
        };

        for module in [
            MaybeValidModule::Text(text.as_str()),
            MaybeValidModule::Binary(binary.as_slice()),
        ] {
            let error = module
                .validate(TargetEnv::Universal1_6)
                .expect_err("SpecId must target scalar specialization constants");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn block_requires_struct_type_target() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeInt 32 0",
            "OpDecorate %1 Block",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble block decoration");
        let expected = ValidationError::InvalidDecorationTargetKind {
            decoration: rspirv::spirv::Decoration::Block,
            target: Id::try_from(1).unwrap(),
            found: rspirv::spirv::Op::TypeInt,
            expected: DecorationTargetKind::StructType,
        };

        for module in [
            MaybeValidModule::Text(text.as_str()),
            MaybeValidModule::Binary(binary.as_slice()),
        ] {
            let error = module
                .validate(TargetEnv::Universal1_6)
                .expect_err("Block must target a struct type");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn array_stride_requires_array_or_pointer_target() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeInt 32 0",
            "OpDecorate %1 ArrayStride 16",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble array stride decoration");
        let expected = ValidationError::InvalidDecorationTargetKind {
            decoration: rspirv::spirv::Decoration::ArrayStride,
            target: Id::try_from(1).unwrap(),
            found: rspirv::spirv::Op::TypeInt,
            expected: DecorationTargetKind::ArrayOrPointerType,
        };

        for module in [
            MaybeValidModule::Text(text.as_str()),
            MaybeValidModule::Binary(binary.as_slice()),
        ] {
            let error = module
                .validate(TargetEnv::Universal1_6)
                .expect_err("ArrayStride must target array/runtime array/pointer types");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn workgroup_size_builtin_requires_constant_when_shader() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeInt 32 0",
            "%2 = OpTypeVector %1 3",
            "%3 = OpTypePointer Input %2",
            "%4 = OpVariable %3 Input",
            "OpDecorate %4 BuiltIn WorkgroupSize",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble workgroup size builtin");
        let expected = ValidationError::InvalidDecorationTargetKind {
            decoration: rspirv::spirv::Decoration::BuiltIn,
            target: Id::try_from(4).unwrap(),
            found: rspirv::spirv::Op::Variable,
            expected: DecorationTargetKind::Constant,
        };

        for module in [
            MaybeValidModule::Text(text.as_str()),
            MaybeValidModule::Binary(binary.as_slice()),
        ] {
            let error = module
                .validate(TargetEnv::Universal1_6)
                .expect_err("WorkgroupSize must target a constant when Shader is declared");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn memory_object_decorations_require_memory_objects() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeInt 32 0",
            "%2 = OpConstant %1 0",
            "OpDecorate %2 NoPerspective",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble NoPerspective decoration");
        let expected = ValidationError::InvalidDecorationTargetKind {
            decoration: rspirv::spirv::Decoration::NoPerspective,
            target: Id::try_from(2).unwrap(),
            found: rspirv::spirv::Op::Constant,
            expected: DecorationTargetKind::MemoryObjectDeclaration,
        };

        for module in [
            MaybeValidModule::Text(text.as_str()),
            MaybeValidModule::Binary(binary.as_slice()),
        ] {
            let error = module
                .validate(TargetEnv::Universal1_6)
                .expect_err("memory object decorations must target memory object declarations");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn linkage_attributes_require_function_or_variable() {
        let text = [
            "OpCapability Shader",
            "OpCapability Linkage",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeInt 32 0",
            "OpDecorate %1 LinkageAttributes \"foo\" Import",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble LinkageAttributes decoration");
        let expected = ValidationError::InvalidDecorationTargetKind {
            decoration: rspirv::spirv::Decoration::LinkageAttributes,
            target: Id::try_from(1).unwrap(),
            found: rspirv::spirv::Op::TypeInt,
            expected: DecorationTargetKind::FunctionOrVariable,
        };

        for module in [
            MaybeValidModule::Text(text.as_str()),
            MaybeValidModule::Binary(binary.as_slice()),
        ] {
            let error = module
                .validate(TargetEnv::Universal1_6)
                .expect_err("LinkageAttributes must target functions or variables");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn builtin_requires_variable_or_constant_targets() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "OpDecorate %main BuiltIn Position",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble BuiltIn on function");
        let expected = ValidationError::InvalidDecorationTargetKind {
            decoration: rspirv::spirv::Decoration::BuiltIn,
            target: Id::try_from(3).unwrap(),
            found: rspirv::spirv::Op::Function,
            expected: DecorationTargetKind::Variable,
        };

        for module in [
            MaybeValidModule::Text(text.as_str()),
            MaybeValidModule::Binary(binary.as_slice()),
        ] {
            let error = module
                .validate(TargetEnv::Universal1_6)
                .expect_err("BuiltIn must target variables/constants");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn entry_point_function_must_reference_function_op() {
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            5,          // bound
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(5, 15), // OpEntryPoint Vertex %1 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            1,
            0x6e69616d, // "main" (null terminator implicit via padding)
            0,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %3 None %2
            2,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::InvalidEntryPointTarget {
                target: Id::try_from(1).unwrap(),
                opcode: rspirv::spirv::Op::TypeVoid
            }
        );
    }

    #[test]
    fn entry_point_interfaces_must_reference_variables() {
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            6,          // bound
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(6, 15), // OpEntryPoint Vertex %3 "main" %1 (interface %1 is not a variable)
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69616d,
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %3 None %2
            2,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::InvalidEntryPointTarget {
                target: Id::try_from(1).unwrap(),
                opcode: rspirv::spirv::Op::TypeVoid
            }
        );
    }

    #[test]
    fn validate_module_reports_missing_memory_model_without_other_globals() {
        // A module that declares only capabilities should still fail for a missing memory model.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            1,          // bound
            0,          // schema
            op(2, 17),  // OpCapability Shader
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::MissingMemoryModel);
    }

    #[test]
    fn validatable_trait_covers_text_and_binary() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
        ]
        .join("\n");
        let valid_text = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("valid text");
        assert_eq!(valid_text.header().schema(), Schema::ZERO);

        let binary = assemble_text(&text).expect("assemble");
        let valid_binary = binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect("valid binary");
        assert_eq!(
            valid_binary.header().bound().declared(),
            DeclaredBound(binary[3])
        );
    }

    #[test]
    fn valid_module_shares_words_without_copying() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let valid = binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect("valid module");
        let handle = valid.words_handle();
        assert_eq!(handle.as_slice(), binary.as_slice());
        let arc_from_handle = handle.shared();
        let arc_from_valid = valid.words_handle().shared();
        assert_eq!(
            Arc::as_ptr(&arc_from_handle),
            Arc::as_ptr(&arc_from_valid),
            "validated modules should share backing storage"
        );

        let module_words: ModuleWords = ModuleWords::from(arc_from_handle);
        assert_eq!(module_words.as_slice(), binary.as_slice());
    }
}
