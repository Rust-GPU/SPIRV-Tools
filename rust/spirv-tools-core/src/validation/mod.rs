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

mod capability_info;
use capability_info::capability_info_from_grammar;
mod instruction_classes;
use instruction_classes::{instruction_class, InstructionClass};
mod instruction_layout;
use instruction_layout::{is_capability_opcode, is_extension_opcode, mode_stage, ModeStage};
mod instruction_versions;
use instruction_versions::grammar_required_spirv_version_for_opcode;
mod operand_versions;
use operand_versions::grammar_required_spirv_version_for_operand;
mod operand_requirements;
use operand_requirements::{
    grammar_required_capabilities_for_operand, grammar_required_extensions_for_operand,
};
use std::collections::BTreeMap;

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
    version: SpirvVersion,
    bound: CheckedBound,
    schema: Schema,
}

/// Shared, validated words backing a module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleWords(Arc<[u32]>);

/// Validator options mirrored from the C++ validator settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationOptions {
    /// Permit relaxed struct store handling.
    pub relax_struct_store: bool,
    /// Permit logical pointer relaxations.
    pub relax_logical_pointer: bool,
    /// Permit relaxed block layout.
    pub relax_block_layout: bool,
    /// Enable uniform buffer standard layout.
    pub uniform_buffer_standard_layout: bool,
    /// Enable scalar block layout.
    pub scalar_block_layout: bool,
    /// Enable workgroup scalar block layout.
    pub workgroup_scalar_block_layout: bool,
    /// Skip block layout validation entirely.
    pub skip_block_layout: bool,
    /// Allow LocalSizeId decoration.
    pub allow_localsizeid: bool,
    /// Allow offset texture operand usage.
    pub allow_offset_texture_operand: bool,
    /// Allow Vulkan 32-bit bitwise operations.
    pub allow_vulkan_32_bit_bitwise: bool,
    /// Enable pre-HLSL legalization relaxations.
    pub before_hlsl_legalization: bool,
    /// Use friendly names for diagnostics.
    pub use_friendly_names: bool,
    /// Validator limit overrides keyed by the limit enum value.
    pub limits: BTreeMap<u32, u32>,
}

/// User-facing names collected from debug instructions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriendlyNames {
    id_names: HashMap<u32, String>,
    member_names: HashMap<(u32, MemberIndex), String>,
}

impl FriendlyNames {
    /// Constructs a friendly-name table from raw id/member maps.
    pub fn from_parts(
        id_names: HashMap<u32, String>,
        member_names: HashMap<(u32, MemberIndex), String>,
    ) -> Self {
        FriendlyNames {
            id_names,
            member_names,
        }
    }

    /// Returns any `OpName`-provided name for the given result id.
    pub fn id(&self, id: u32) -> Option<&str> {
        self.id_names.get(&id).map(String::as_str)
    }

    /// Returns any `OpMemberName`-provided name for the given struct/member pair.
    pub fn member(&self, struct_id: u32, member: MemberIndex) -> Option<&str> {
        self.member_names
            .get(&(struct_id, member))
            .map(String::as_str)
    }

    /// Formats an id with a friendly suffix when available (e.g., `%5 (foo)`).
    pub fn format_id(&self, id: u32) -> String {
        if let Some(name) = self.id(id) {
            format!("%{id} ({name})")
        } else {
            format!("%{id}")
        }
    }

    /// Formats a struct member with a friendly suffix when available.
    pub fn format_member(&self, struct_id: u32, member: MemberIndex) -> String {
        if let Some(name) = self.member(struct_id, member) {
            format!("%{struct_id}.{member} ({name})")
        } else {
            format!("%{struct_id}.{member}")
        }
    }

    /// Accesses the raw id→name table.
    pub fn id_names(&self) -> &HashMap<u32, String> {
        &self.id_names
    }

    /// Accesses the raw (struct id, member)→name table.
    pub fn member_names(&self) -> &HashMap<(u32, MemberIndex), String> {
        &self.member_names
    }
}

/// Formats a validation error, appending friendly names when provided.
pub fn format_validation_error(error: &ValidationError, names: Option<&FriendlyNames>) -> String {
    match (error, names) {
        (ValidationError::ExecutionModeWithoutEntryPoint { function }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::InvalidEntryPointTarget { target, .. }, Some(names)) => {
            names.format_id((*target).into())
        }
        (ValidationError::FunctionDeclarationAfterDefinition { function }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::MissingFunctionEntryBlock { function }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::MissingBlockTerminator { function, block }, Some(names)) => format!(
            "{} in block {}",
            names.format_id((*function).into()),
            names.format_id((*block).into())
        ),
        (ValidationError::InstructionsAfterTerminator { function, block }, Some(names)) => format!(
            "{} in block {}",
            names.format_id((*function).into()),
            names.format_id((*block).into())
        ),
        (ValidationError::MissingBlockTarget { function, target }, Some(names)) => format!(
            "{} missing block {}",
            names.format_id((*function).into()),
            names.format_id((*target).into())
        ),
        (ValidationError::MissingReturnValue { function, .. }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::ReturnValueInVoidFunction { function }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::FunctionTypeParameterVoid { type_id, parameter }, Some(names)) => {
            format!(
                "{} parameter {}",
                names.format_id((*type_id).into()),
                names.format_id((*parameter).into())
            )
        }
        (ValidationError::FunctionReturnTypeMismatch { function, .. }, Some(names))
        | (ValidationError::FunctionParameterCountMismatch { function, .. }, Some(names))
        | (ValidationError::FunctionParameterTypeMismatch { function, .. }, Some(names)) => {
            names.format_id((*function).into())
        }
        _ => error.to_string(),
    }
}

/// Attempts to render a validation error with friendly names derived from the provided module words.
pub fn format_validation_error_from_words(
    words: &[u32],
    options: &ValidationOptions,
    error: &ValidationError,
) -> String {
    if !options.use_friendly_names {
        return error.to_string();
    }
    let names = collect_friendly_names(words);
    format_validation_error(error, names.as_ref())
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            relax_struct_store: false,
            relax_logical_pointer: false,
            relax_block_layout: false,
            uniform_buffer_standard_layout: false,
            scalar_block_layout: false,
            workgroup_scalar_block_layout: false,
            skip_block_layout: false,
            allow_localsizeid: false,
            allow_offset_texture_operand: false,
            allow_vulkan_32_bit_bitwise: false,
            before_hlsl_legalization: false,
            use_friendly_names: true,
            limits: BTreeMap::new(),
        }
    }
}

impl Hash for ValidationOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.relax_struct_store.hash(state);
        self.relax_logical_pointer.hash(state);
        self.relax_block_layout.hash(state);
        self.uniform_buffer_standard_layout.hash(state);
        self.scalar_block_layout.hash(state);
        self.workgroup_scalar_block_layout.hash(state);
        self.skip_block_layout.hash(state);
        self.allow_localsizeid.hash(state);
        self.allow_offset_texture_operand.hash(state);
        self.allow_vulkan_32_bit_bitwise.hash(state);
        self.before_hlsl_legalization.hash(state);
        self.use_friendly_names.hash(state);
        for (k, v) in &self.limits {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl ValidationOptions {
    /// Returns a copy of the options with the given limit override applied.
    pub fn with_limit(mut self, kind: u32, value: u32) -> Self {
        self.limits.insert(kind, value);
        self
    }
}

/// Limit kind for the maximum number of struct members.
pub const LIMIT_MAX_STRUCT_MEMBERS: u32 = 0;
/// Limit kind for maximum struct nesting depth.
pub const LIMIT_MAX_STRUCT_DEPTH: u32 = 1;
/// Limit kind for maximum local variables.
pub const LIMIT_MAX_LOCAL_VARIABLES: u32 = 2;
/// Limit kind for maximum global variables.
pub const LIMIT_MAX_GLOBAL_VARIABLES: u32 = 3;
/// Limit kind for maximum switch branches.
pub const LIMIT_MAX_SWITCH_BRANCHES: u32 = 4;
/// Limit kind for maximum function arguments.
pub const LIMIT_MAX_FUNCTION_ARGS: u32 = 5;
/// Limit kind for maximum control-flow nesting depth.
pub const LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH: u32 = 6;
/// Limit kind for maximum access-chain indexes.
pub const LIMIT_MAX_ACCESS_CHAIN_INDEXES: u32 = 7;
const LIMIT_MAX_ID_BOUND: u32 = 8;

/// A simple snapshot of validator limits keyed by the limit enum value.
pub type ValidationLimits = BTreeMap<u32, u32>;

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

fn merge_versions(
    grammar: Option<SpirvVersion>,
    manual: Option<SpirvVersion>,
) -> Option<SpirvVersion> {
    match (grammar, manual) {
        (Some(grammar), Some(manual)) => Some(if grammar > manual { grammar } else { manual }),
        (Some(grammar), None) => Some(grammar),
        (None, Some(manual)) => Some(manual),
        (None, None) => None,
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
    pub fn new(version: SpirvVersion, bound: CheckedBound, schema: Schema) -> Self {
        Self {
            version,
            bound,
            schema,
        }
    }

    /// Parses and validates a module header, ensuring the bound and schema are valid.
    pub fn from_module(module: &Module) -> Result<Self, ValidationError> {
        let header = module
            .header
            .as_ref()
            .ok_or(ValidationError::MissingHeader)?;
        let schema = Schema::validate(header.reserved_word)?;
        let version = SpirvVersion::from_word(header.version);
        let declared_bound = DeclaredBound(header.bound);
        let bound = CheckedBound::new(declared_bound).ok_or(ValidationError::InvalidIdBound {
            bound: declared_bound,
        })?;
        Ok(Self {
            version,
            bound,
            schema,
        })
    }

    /// Returns the validated id bound associated with this header.
    pub fn bound(self) -> CheckedBound {
        self.bound
    }

    /// Returns the module's declared SPIR-V version.
    pub fn version(self) -> SpirvVersion {
        self.version
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
    effective_version: SpirvVersion,
    options: ValidationOptions,
    friendly_names: Option<FriendlyNames>,
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

    /// Returns the SPIR-V version actually used during validation (module version clamped to env).
    pub fn effective_version(&self) -> SpirvVersion {
        self.effective_version
    }

    /// Returns the declared SPIR-V version from the module header.
    pub fn module_version(&self) -> SpirvVersion {
        self.header.version()
    }

    /// Returns the validated module header.
    pub fn header(&self) -> ValidatedHeader {
        self.header
    }

    /// Returns a shared handle to the validated words.
    pub fn words_handle(&self) -> ModuleWords {
        self.words.clone()
    }

    /// Returns the validator options applied during validation.
    pub fn options(&self) -> &ValidationOptions {
        &self.options
    }

    /// Returns friendly names applied during validation (if enabled).
    pub fn friendly_names(&self) -> Option<&FriendlyNames> {
        self.friendly_names.as_ref()
    }
}

/// Validates a SPIR-V module against invariants that can be checked without target-specific
/// knowledge.
pub fn validate_module(words: &[u32], env: TargetEnv) -> Result<(), ValidationError> {
    validate_module_with_options(words, env, ValidationOptions::default())
}

/// Validates a SPIR-V module with explicit validator options.
pub fn validate_module_with_options(
    words: &[u32],
    env: TargetEnv,
    options: ValidationOptions,
) -> Result<(), ValidationError> {
    validate_words(ModuleWords::from(Arc::from(words)), env, options).map(|_| ())
}

/// A cache of validated modules keyed by target environment and module contents.
#[derive(Default)]
pub struct ValidModuleCache {
    entries: std::collections::HashMap<(TargetEnv, u64, ValidationOptions), Arc<ValidModule>>,
}

impl ValidModuleCache {
    /// Validate the provided binary words, returning a shared validated module and caching the result.
    pub fn validate_words(
        &mut self,
        words: &[u32],
        env: TargetEnv,
    ) -> Result<Arc<ValidModule>, ValidationError> {
        self.validate_words_with_options(words, env, ValidationOptions::default())
    }

    /// Validate with explicit options.
    pub fn validate_words_with_options(
        &mut self,
        words: &[u32],
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<Arc<ValidModule>, ValidationError> {
        let hash = hash_words(words, env);
        if let Some(cached) = self.entries.get(&(env, hash, options.clone())) {
            if cached.words_handle().as_slice() == words {
                return Ok(Arc::clone(cached));
            }
        }
        let validated = validate_words(ModuleWords::from(Arc::from(words)), env, options.clone())?;
        let validated = Arc::new(validated);
        self.entries
            .insert((env, hash, options), Arc::clone(&validated));
        Ok(validated)
    }
}

fn validate_words(
    words: ModuleWords,
    env: TargetEnv,
    options: ValidationOptions,
) -> Result<ValidModule, ValidationError> {
    if let Some(&schema) = words.as_slice().get(4) {
        Schema::validate(schema)?;
    }
    if !options.skip_block_layout {
        run_layout_check(words.as_slice(), env)?;
    }
    let module = parse_module(words.as_slice())?;
    validate_extension_allowlist(&module, env)?;
    let header = ValidatedHeader::from_module(&module)?;
    if let Some(&limit) = options.limits.get(&LIMIT_MAX_ID_BOUND) {
        if header.bound.declared().0 > limit {
            return Err(ValidationError::IdBoundExceedsLimit {
                declared: header.bound.declared(),
                limit,
            });
        }
    }
    let module_version = header.version();
    let target_version = effective_spirv_version(env, module_version);
    let defined_ids = validate_id_bound(&module, header)?;
    let opcodes = collect_result_opcodes(&module);
    let definitions = collect_result_instructions(&module);
    let capabilities = collect_declared_capabilities(&module);
    let extensions = validate_extensions(&module, env, target_version)?;
    validate_capabilities(&module, env, target_version, &extensions)?;
    validate_sampler_image_addressing_mode(&module, &capabilities)?;
    validate_memory_model(&module)?;
    validate_type_functions(&module, &opcodes)?;
    let struct_member_counts = validate_member_decorations(&module, &defined_ids)?;
    enforce_logical_pointer_rules(&module, &definitions, &capabilities, &options)?;
    enforce_struct_member_limit(&struct_member_counts, &options)?;
    enforce_struct_depth_limit(&module, &definitions, &options)?;
    validate_decoration_groups(&module, &defined_ids, &opcodes, &struct_member_counts)?;
    validate_decorations(&module, &defined_ids)?;
    validate_decoration_target_categories(&module, &opcodes, &definitions, &capabilities)?;
    enforce_store_type_compatibility(&module, &definitions, &options)?;
    let entry_points = validate_entry_points(&module, &defined_ids, &opcodes)?;
    validate_execution_modes(&module, &entry_points, env, &options)?;
    validate_functions(&module)?;
    enforce_function_arg_limit(&module, &options)?;
    enforce_variable_limits(&module, &options)?;
    enforce_switch_branch_limit(&module, &options)?;
    enforce_access_chain_limit(&module, &options)?;
    enforce_offset_texture_operand_rule(&module, env, &options)?;
    enforce_vulkan_bitwise_widths(&module, env, &definitions, &options)?;
    enforce_block_layout_rules(&module, &definitions, &options)?;
    let friendly_names = options
        .use_friendly_names
        .then(|| build_friendly_name_table(&module));
    Ok(ValidModule {
        words,
        module,
        env,
        header,
        effective_version: target_version,
        options,
        friendly_names,
    })
}

fn build_friendly_name_table(module: &Module) -> FriendlyNames {
    let mut id_names = HashMap::new();
    let mut member_names = HashMap::new();
    for inst in &module.debug_names {
        match inst.class.opcode {
            rspirv::spirv::Op::Name => {
                if let (
                    Some(rspirv::dr::Operand::IdRef(id)),
                    Some(rspirv::dr::Operand::LiteralString(name)),
                ) = (inst.operands.first(), inst.operands.get(1))
                {
                    id_names.insert(*id, name.clone());
                }
            }
            rspirv::spirv::Op::MemberName => {
                if let (
                    Some(rspirv::dr::Operand::IdRef(struct_id)),
                    Some(rspirv::dr::Operand::LiteralBit32(member)),
                    Some(rspirv::dr::Operand::LiteralString(name)),
                ) = (
                    inst.operands.first(),
                    inst.operands.get(1),
                    inst.operands.get(2),
                ) {
                    member_names.insert((*struct_id, MemberIndex(*member)), name.clone());
                }
            }
            _ => {}
        }
    }
    FriendlyNames {
        id_names,
        member_names,
    }
}

fn collect_friendly_names(words: &[u32]) -> Option<FriendlyNames> {
    let mut loader = rspirv::dr::Loader::new();
    if rspirv::binary::parse_words(words, &mut loader).is_err() {
        return None;
    }
    let module = loader.module();
    Some(build_friendly_name_table(&module))
}

fn validate_functions(module: &Module) -> Result<(), ValidationError> {
    let definitions = collect_result_instructions(module);
    let result_types = collect_result_types(module)?;
    let mut seen_definition = false;
    for function in &module.functions {
        let function_id = function
            .def
            .as_ref()
            .and_then(|inst| inst.result_id)
            .and_then(|raw| Id::try_from(raw).ok())
            .unwrap_or(Id::try_from(1).expect("non-zero literal"));

        let is_declaration = function.blocks.is_empty() && function.parameters.is_empty();
        let signature = validate_function_signature(function_id, function, &definitions)?;
        if is_declaration {
            if seen_definition {
                return Err(ValidationError::FunctionDeclarationAfterDefinition {
                    function: function_id,
                });
            }
            continue;
        }
        seen_definition = true;
        let return_type = signature.return_type;
        let return_is_void = is_void_type(return_type, &definitions);

        let entry_block =
            function
                .blocks
                .first()
                .ok_or(ValidationError::MissingFunctionEntryBlock {
                    function: function_id,
                })?;
        let entry_label_id = entry_block
            .label
            .as_ref()
            .and_then(|inst| inst.result_id)
            .and_then(|raw| Id::try_from(raw).ok())
            .unwrap_or(function_id);

        let block_ids: std::collections::HashSet<Id> = function
            .blocks
            .iter()
            .filter_map(|block| {
                block
                    .label
                    .as_ref()
                    .and_then(|inst| inst.result_id)
                    .and_then(|raw| Id::try_from(raw).ok())
            })
            .collect();

        let missing_entry_label = entry_block
            .label
            .as_ref()
            .map(|label| label.class.opcode != rspirv::spirv::Op::Label)
            .unwrap_or(true);
        if missing_entry_label {
            return Err(ValidationError::MissingFunctionEntryBlock {
                function: function_id,
            });
        }

        let mut predecessors: std::collections::HashMap<Id, std::collections::HashSet<Id>> =
            block_ids
                .iter()
                .copied()
                .map(|id| (id, Default::default()))
                .collect();

        for block in &function.blocks {
            let block_label_id = block
                .label
                .as_ref()
                .and_then(|inst| inst.result_id)
                .and_then(|raw| Id::try_from(raw).ok())
                .unwrap_or(entry_label_id);
            let mut first_terminator_index = None;
            for (index, inst) in block.instructions.iter().enumerate() {
                if rspirv::grammar::reflect::is_block_terminator(inst.class.opcode) {
                    first_terminator_index = Some(index);
                    break;
                }
            }
            if first_terminator_index.is_none() {
                return Err(ValidationError::MissingBlockTerminator {
                    function: function_id,
                    block: block_label_id,
                });
            }
            let terminator_index = first_terminator_index.unwrap();
            if terminator_index + 1 < block.instructions.len() {
                return Err(ValidationError::InstructionsAfterTerminator {
                    function: function_id,
                    block: block_label_id,
                });
            }

            let terminator_inst = &block.instructions[terminator_index];
            let check_target = |operand: &rspirv::dr::Operand| -> Result<(), ValidationError> {
                if let rspirv::dr::Operand::IdRef(raw) = operand {
                    if let Ok(target) = Id::try_from(*raw) {
                        if !block_ids.contains(&target) {
                            return Err(ValidationError::MissingBlockTarget {
                                function: function_id,
                                target,
                            });
                        }
                    }
                }
                Ok(())
            };

            match terminator_inst.class.opcode {
                rspirv::spirv::Op::Return => {
                    if !return_is_void {
                        return Err(ValidationError::MissingReturnValue {
                            function: function_id,
                            expected: return_type,
                        });
                    }
                }
                rspirv::spirv::Op::ReturnValue => {
                    if return_is_void {
                        return Err(ValidationError::ReturnValueInVoidFunction {
                            function: function_id,
                        });
                    }
                    if let Some(rspirv::dr::Operand::IdRef(raw)) = terminator_inst.operands.first()
                    {
                        if let Ok(value_id) = ResultId::try_from(*raw) {
                            let value_type = result_types.get(&value_id).copied().ok_or(
                                ValidationError::InvalidReturnValueType {
                                    function: function_id,
                                    expected: return_type,
                                    found: return_type,
                                },
                            )?;
                            if value_type != return_type {
                                return Err(ValidationError::InvalidReturnValueType {
                                    function: function_id,
                                    expected: return_type,
                                    found: value_type,
                                });
                            }
                        }
                    }
                }
                rspirv::spirv::Op::Branch => {
                    if let Some(op) = terminator_inst.operands.first() {
                        check_target(op)?;
                        if let rspirv::dr::Operand::IdRef(raw) = op {
                            if let Ok(target) = Id::try_from(*raw) {
                                if let Some(preds) = predecessors.get_mut(&target) {
                                    preds.insert(block_label_id);
                                }
                            }
                        }
                    }
                }
                rspirv::spirv::Op::BranchConditional => {
                    for op in terminator_inst.operands.iter().skip(1).take(2) {
                        check_target(op)?;
                        if let rspirv::dr::Operand::IdRef(raw) = op {
                            if let Ok(target) = Id::try_from(*raw) {
                                if let Some(preds) = predecessors.get_mut(&target) {
                                    preds.insert(block_label_id);
                                }
                            }
                        }
                    }
                }
                rspirv::spirv::Op::Switch => {
                    for (index, op) in terminator_inst.operands.iter().enumerate() {
                        if index == 0 {
                            continue; // selector
                        }
                        // operands alternate: default target then pairs of (literal, target)
                        if index == 1 || index % 2 == 0 {
                            check_target(op)?;
                            if let rspirv::dr::Operand::IdRef(raw) = op {
                                if let Ok(target) = Id::try_from(*raw) {
                                    if let Some(preds) = predecessors.get_mut(&target) {
                                        preds.insert(block_label_id);
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(entry_preds) = predecessors.get(&entry_label_id) {
            if !entry_preds.is_empty() {
                return Err(ValidationError::EntryBlockHasPredecessor {
                    function: function_id,
                    entry: entry_label_id,
                });
            }
        }

        for block in &function.blocks {
            let block_label_id = block
                .label
                .as_ref()
                .and_then(|inst| inst.result_id)
                .and_then(|raw| Id::try_from(raw).ok())
                .unwrap_or(entry_label_id);
            let expected_preds = predecessors
                .get(&block_label_id)
                .map(|set| set.len())
                .unwrap_or(0);
            for inst in &block.instructions {
                if inst.class.opcode == rspirv::spirv::Op::Phi {
                    let mut seen_incoming: std::collections::HashSet<Id> = Default::default();
                    for pair in inst.operands.chunks(2) {
                        if let Some(rspirv::dr::Operand::IdRef(raw_incoming)) = pair.get(1) {
                            if let Ok(incoming_block) = Id::try_from(*raw_incoming) {
                                if !block_ids.contains(&incoming_block) {
                                    return Err(ValidationError::PhiIncomingBlockMissing {
                                        function: function_id,
                                        block: block_label_id,
                                        incoming: incoming_block,
                                    });
                                }
                                if let Some(preds) = predecessors.get(&block_label_id) {
                                    if !preds.contains(&incoming_block) {
                                        return Err(ValidationError::PhiIncomingNotPredecessor {
                                            function: function_id,
                                            block: block_label_id,
                                            incoming: incoming_block,
                                        });
                                    }
                                }
                                if !seen_incoming.insert(incoming_block) {
                                    return Err(ValidationError::PhiDuplicatePredecessor {
                                        function: function_id,
                                        block: block_label_id,
                                        incoming: incoming_block,
                                    });
                                }
                            }
                        }
                    }
                    let pair_count = inst.operands.len() / 2;
                    if pair_count != expected_preds {
                        return Err(ValidationError::PhiPredecessorCountMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected: expected_preds,
                            found: pair_count,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
struct FunctionSignature {
    return_type: TypeId,
}

fn validate_function_signature(
    function_id: Id,
    function: &rspirv::dr::Function,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Result<FunctionSignature, ValidationError> {
    let function_def = match &function.def {
        Some(def) => def,
        None => {
            return Err(ValidationError::MissingFunctionEntryBlock {
                function: function_id,
            })
        }
    };

    let function_type_id = match function_def.operands.get(1) {
        Some(rspirv::dr::Operand::IdRef(raw)) => {
            TypeId::try_from(*raw).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::Operand,
                opcode: function_def.class.opcode,
            })?
        }
        _ => {
            return Err(ValidationError::InvalidFunctionType {
                function: function_id,
                type_id: TypeId::new(function_id),
            })
        }
    };

    let function_type_result =
        ResultId::try_from(u32::from(function_type_id)).map_err(|_| ValidationError::ZeroId {
            kind: IdKind::Operand,
            opcode: function_def.class.opcode,
        })?;
    let Some(function_type_inst) = definitions.get(&function_type_result) else {
        return Err(ValidationError::InvalidFunctionType {
            function: function_id,
            type_id: function_type_id,
        });
    };
    if function_type_inst.class.opcode != rspirv::spirv::Op::TypeFunction {
        return Err(ValidationError::InvalidFunctionType {
            function: function_id,
            type_id: function_type_id,
        });
    }

    let result_type = function_def
        .result_type
        .and_then(|raw| TypeId::try_from(raw).ok())
        .ok_or(ValidationError::InvalidFunctionType {
            function: function_id,
            type_id: function_type_id,
        })?;

    let return_type = match function_type_inst.operands.first() {
        Some(rspirv::dr::Operand::IdRef(raw)) => {
            TypeId::try_from(*raw).map_err(|_| ValidationError::InvalidFunctionType {
                function: function_id,
                type_id: function_type_id,
            })?
        }
        _ => {
            return Err(ValidationError::InvalidFunctionType {
                function: function_id,
                type_id: function_type_id,
            })
        }
    };

    if result_type != return_type {
        return Err(ValidationError::FunctionReturnTypeMismatch {
            function: function_id,
            result_type,
            function_type: return_type,
        });
    }

    let mut expected_params: Vec<TypeId> = Vec::new();
    for op in function_type_inst.operands.iter().skip(1) {
        match op {
            rspirv::dr::Operand::IdRef(raw) => {
                let ty = TypeId::try_from(*raw).map_err(|_| ValidationError::ZeroId {
                    kind: IdKind::Operand,
                    opcode: function_type_inst.class.opcode,
                })?;
                expected_params.push(ty);
            }
            _ => {
                return Err(ValidationError::InvalidFunctionType {
                    function: function_id,
                    type_id: function_type_id,
                });
            }
        }
    }

    if expected_params.len() != function.parameters.len() {
        return Err(ValidationError::FunctionParameterCountMismatch {
            function: function_id,
            expected: expected_params.len(),
            found: function.parameters.len(),
        });
    }

    for (expected_type, param_inst) in expected_params.iter().zip(&function.parameters) {
        let parameter = param_inst
            .result_id
            .and_then(|raw| Id::try_from(raw).ok())
            .unwrap_or(function_id);
        let param_type = param_inst
            .result_type
            .and_then(|raw| TypeId::try_from(raw).ok())
            .ok_or(ValidationError::FunctionParameterTypeMismatch {
                function: function_id,
                parameter,
                expected: *expected_type,
                found: TypeId::new(parameter),
            })?;
        if param_type != *expected_type {
            return Err(ValidationError::FunctionParameterTypeMismatch {
                function: function_id,
                parameter,
                expected: *expected_type,
                found: param_type,
            });
        }
    }

    Ok(FunctionSignature { return_type })
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
        self.validate_with_options(env, ValidationOptions::default())
    }

    /// Validate the provided input with explicit options, assembling text when necessary.
    pub fn validate_with_options(
        self,
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<ValidModule, ValidationError> {
        match self {
            MaybeValidModule::Binary(words) => {
                validate_words(ModuleWords::from(Arc::from(words)), env, options)
            }
            MaybeValidModule::Text(text) => {
                let binary = ModuleWords::from(Arc::<[u32]>::from(
                    crate::assembly::assemble_text(text)
                        .map_err(|err| ValidationError::Parse(err.to_string()))?
                        .into_boxed_slice(),
                ));
                validate_words(binary, env, options)
            }
        }
    }
}

/// Convenience trait for validating either binary words or assembly text.
pub trait ValidatableModule<'a> {
    /// Validates the module input for the requested target environment.
    fn validate(self, env: TargetEnv) -> Result<ValidModule, ValidationError>
    where
        Self: Sized,
    {
        self.validate_with_options(env, ValidationOptions::default())
    }

    /// Validates the module input for the requested target environment with explicit options.
    fn validate_with_options(
        self,
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<ValidModule, ValidationError>
    where
        Self: Sized;
}

impl<'a> ValidatableModule<'a> for &'a [u32] {
    fn validate_with_options(
        self,
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<ValidModule, ValidationError> {
        MaybeValidModule::Binary(self).validate_with_options(env, options)
    }
}

impl<'a> ValidatableModule<'a> for &'a str {
    fn validate_with_options(
        self,
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<ValidModule, ValidationError> {
        MaybeValidModule::Text(self).validate_with_options(env, options)
    }
}

fn parse_module(words: &[u32]) -> Result<rspirv::dr::Module, ValidationError> {
    let collected = {
        struct ModuleSideInstructionCollector {
            execution_modes: Vec<rspirv::dr::Instruction>,
            conditional_extensions: Vec<rspirv::dr::Instruction>,
            conditional_capabilities: Vec<rspirv::dr::Instruction>,
            conditional_entry_points: Vec<rspirv::dr::Instruction>,
        }

        impl rspirv::binary::Consumer for ModuleSideInstructionCollector {
            fn initialize(&mut self) -> rspirv::binary::ParseAction {
                rspirv::binary::ParseAction::Continue
            }

            fn finalize(&mut self) -> rspirv::binary::ParseAction {
                rspirv::binary::ParseAction::Continue
            }

            fn consume_header(
                &mut self,
                _header: rspirv::dr::ModuleHeader,
            ) -> rspirv::binary::ParseAction {
                rspirv::binary::ParseAction::Continue
            }

            fn consume_instruction(
                &mut self,
                inst: rspirv::dr::Instruction,
            ) -> rspirv::binary::ParseAction {
                match inst.class.opcode {
                    rspirv::spirv::Op::ExecutionModeId => self.execution_modes.push(inst),
                    rspirv::spirv::Op::ConditionalExtensionINTEL => {
                        self.conditional_extensions.push(inst)
                    }
                    rspirv::spirv::Op::ConditionalCapabilityINTEL => {
                        self.conditional_capabilities.push(inst)
                    }
                    rspirv::spirv::Op::ConditionalEntryPointINTEL => {
                        self.conditional_entry_points.push(inst)
                    }
                    _ => {}
                }
                rspirv::binary::ParseAction::Continue
            }
        }

        let mut collector = ModuleSideInstructionCollector {
            execution_modes: Vec::new(),
            conditional_extensions: Vec::new(),
            conditional_capabilities: Vec::new(),
            conditional_entry_points: Vec::new(),
        };
        match rspirv::binary::parse_words(words, &mut collector) {
            Ok(()) => collector,
            Err(error) => return Err(ValidationError::Parse(error.to_string())),
        }
    };

    let filtered_words = {
        if words.len() < 5 {
            return Err(ValidationError::Parse(
                "module header is incomplete".to_string(),
            ));
        }
        let mut filtered = Vec::with_capacity(words.len());
        filtered.extend_from_slice(&words[..5]);
        let mut index = 5;
        while index < words.len() {
            let word = words[index];
            let word_count = (word >> 16) as usize;
            let opcode = word & 0xFFFF;
            if word_count == 0 {
                return Err(ValidationError::Parse(
                    "invalid instruction with zero word count".to_string(),
                ));
            }
            if index + word_count > words.len() {
                return Err(ValidationError::Parse(
                    "invalid instruction length exceeding module size".to_string(),
                ));
            }
            if !matches!(
                opcode,
                x if x == rspirv::spirv::Op::ExecutionModeId as u32
                    || x == rspirv::spirv::Op::ConditionalExtensionINTEL as u32
                    || x == rspirv::spirv::Op::ConditionalCapabilityINTEL as u32
                    || x == rspirv::spirv::Op::ConditionalEntryPointINTEL as u32
            ) {
                filtered.extend_from_slice(&words[index..index + word_count]);
            }
            index += word_count;
        }
        filtered
    };

    let mut loader = rspirv::dr::Loader::new();
    if let Err(error) = rspirv::binary::parse_words(&filtered_words, &mut loader) {
        return Err(ValidationError::Parse(error.to_string()));
    }
    let mut module = loader.module();
    module.execution_modes.extend(collected.execution_modes);
    module.extensions.extend(collected.conditional_extensions);
    module
        .capabilities
        .extend(collected.conditional_capabilities);
    module
        .entry_points
        .extend(collected.conditional_entry_points);
    Ok(module)
}

fn run_layout_check(words: &[u32], env: TargetEnv) -> Result<(), ValidationError> {
    struct LayoutChecker {
        memory_model_state: MemoryModelState,
        current_section: Section,
        function_state: FunctionState,
        capabilities: CapabilitySet,
        extensions: ExtensionSet,
        sampler_image_address_mode: Option<u32>,
        env: TargetEnv,
        mode_stage: ModeStage,
    }

    impl LayoutChecker {
        fn new(env: TargetEnv) -> Self {
            Self {
                memory_model_state: MemoryModelState::new(),
                current_section: Section::Capabilities,
                function_state: FunctionState::Outside,
                capabilities: CapabilitySet::default(),
                extensions: ExtensionSet::default(),
                sampler_image_address_mode: None,
                env,
                mode_stage: ModeStage::Capabilities,
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
                let opcode = inst.class.opcode;
                match opcode {
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
                let section = instruction_section(self.current_section, &inst);
                if section <= Section::Annotations {
                    return rspirv::binary::ParseAction::Error(Box::new(
                        ValidationError::LayoutOutOfOrder { opcode },
                    ));
                }
                return rspirv::binary::ParseAction::Continue;
            }

            let opcode = inst.class.opcode;
            let section = instruction_section(self.current_section, &inst);

            if let Some(stage) = mode_stage(opcode) {
                if stage < self.mode_stage {
                    return rspirv::binary::ParseAction::Error(Box::new(
                        ValidationError::LayoutOutOfOrder { opcode },
                    ));
                }
                self.mode_stage = self.mode_stage.max(stage);
            }

            if is_extension_opcode(opcode) {
                if section < self.current_section {
                    return rspirv::binary::ParseAction::Error(Box::new(
                        ValidationError::LayoutOutOfOrder { opcode },
                    ));
                }
                if let Some(extension) = extension_operand(&inst) {
                    if let Err(err) = self.extensions.insert(extension, self.env) {
                        return rspirv::binary::ParseAction::Error(Box::new(err));
                    }
                }
            } else if is_capability_opcode(opcode) {
                if section < self.current_section {
                    return rspirv::binary::ParseAction::Error(Box::new(
                        ValidationError::LayoutOutOfOrder { opcode },
                    ));
                }
                if let Some(cap) = capability_operand(&inst) {
                    if let Err(err) = self.capabilities.insert(cap) {
                        return rspirv::binary::ParseAction::Error(Box::new(err));
                    }
                }
            } else if opcode == rspirv::spirv::Op::MemoryModel {
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
            } else {
                match opcode {
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

    let mut checker = LayoutChecker::new(env);
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
    if let Some(class) = instruction_class(opcode) {
        match class {
            InstructionClass::Annotation => return Section::Annotations,
            InstructionClass::ConstantCreation | InstructionClass::TypeDeclaration => {
                return Section::TypesGlobals;
            }
            InstructionClass::Extension => match opcode {
                Extension | ConditionalExtensionINTEL => return Section::Extensions,
                ExtInstImport => return Section::ExtInstImport,
                _ => {}
            },
            InstructionClass::ModeSetting => {
                if let Some(stage) = mode_stage(opcode) {
                    return match stage {
                        ModeStage::Capabilities => Section::Capabilities,
                        ModeStage::Extensions => Section::Extensions,
                        ModeStage::ExtInstImport => Section::ExtInstImport,
                        ModeStage::MemoryModel => Section::MemoryModel,
                        ModeStage::EntryPoint => Section::EntryPoint,
                        ModeStage::ExecutionMode => Section::ExecutionMode,
                    };
                }
            }
            InstructionClass::Debug => match opcode {
                SourceContinued | Source | SourceExtension | String => return Section::Debug1,
                Name | MemberName => return Section::Debug2,
                ModuleProcessed => return Section::Debug3,
                Line | NoLine => {}
                _ => return Section::Debug1,
            },
        }
    }

    let opname = inst.class.opname;
    if opname.starts_with("OpType")
        || opname.starts_with("OpConstant")
        || opname.starts_with("OpSpecConstant")
    {
        return Section::TypesGlobals;
    }
    match opcode {
        SamplerImageAddressingModeNV => Section::SamplerImageAddressMode,
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
    target_version: SpirvVersion,
    extensions: &ExtensionSet,
) -> Result<(), ValidationError> {
    let declared: HashSet<_> = module
        .capabilities
        .iter()
        .filter_map(capability_operand)
        .collect();
    for inst in &module.capabilities {
        if let Some(capability) = capability_operand(inst) {
            let grammar_requirements = capability_info_from_grammar(capability);
            let allowed_by_env = env.is_capability_allowed(capability);
            if !allowed_by_env && capability == rspirv::spirv::Capability::VulkanMemoryModel {
                return Err(ValidationError::DisallowedCapability { capability, env });
            }
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
            let grammar_version = grammar_requirements.required_version;
            let required_version = merge_versions(
                grammar_version,
                manual_required_spirv_version_for_capability(capability),
            );
            let manual_required_extension = required_extension_for_capability(capability);
            let always_require_extension = manual_required_extension
                .map(extension_always_required)
                .unwrap_or(false);
            let version_allows_core = required_version
                .map(|required| target_version >= required)
                .unwrap_or(false);
            let grammar_requires_extension = !grammar_requirements.required_extensions.is_empty()
                && (grammar_version.is_none_or(|required| target_version < required)
                    || always_require_extension);
            let manual_requires_extension = manual_required_extension.is_some()
                && (always_require_extension || !version_allows_core);

            if grammar_requires_extension {
                for &required_ext in grammar_requirements.required_extensions {
                    if !extension_allowed_in_env(required_ext, env) {
                        return Err(ValidationError::DisallowedExtension {
                            extension: ExtensionName::from(required_ext),
                            env,
                        });
                    }
                }
            }
            if manual_requires_extension {
                if let Some(required_ext) = manual_required_extension {
                    if !extension_allowed_in_env(required_ext, env) {
                        return Err(ValidationError::DisallowedExtension {
                            extension: ExtensionName::from(required_ext),
                            env,
                        });
                    }
                }
            }

            if let Some(required_version) = required_version {
                if target_version < required_version {
                    let has_required_extension =
                        !grammar_requirements.required_extensions.is_empty()
                            || manual_required_extension.is_some();
                    if !allowed_by_env && !has_required_extension {
                        return Err(ValidationError::DisallowedCapability { capability, env });
                    }
                    return Err(ValidationError::CapabilityRequiresSpirvVersion {
                        capability,
                        required_version,
                        target_version,
                    });
                }
            }
            if !grammar_requirements.required_extensions.is_empty() && grammar_requires_extension {
                for &required_ext in grammar_requirements.required_extensions {
                    if !extension_allowed_in_env(required_ext, env) {
                        return Err(ValidationError::DisallowedExtension {
                            extension: ExtensionName::from(required_ext),
                            env,
                        });
                    }
                    if !has_extension(extensions, required_ext) {
                        return Err(ValidationError::DisallowedCapabilityMissingExtension {
                            capability,
                            required_extension: required_ext.to_string(),
                        });
                    }
                }
            }
            if let Some(required_ext) = manual_required_extension {
                if manual_requires_extension {
                    if !extension_allowed_in_env(required_ext, env) {
                        return Err(ValidationError::DisallowedExtension {
                            extension: ExtensionName::from(required_ext),
                            env,
                        });
                    }
                    if !has_extension(extensions, required_ext) {
                        return Err(ValidationError::DisallowedCapabilityMissingExtension {
                            capability,
                            required_extension: required_ext.to_string(),
                        });
                    }
                }
            }

            let allowed_by_extension = capability_allowed_by_extension(env, capability, extensions);
            let allowed_by_capability =
                capability_enabled_by_capability(env, capability, &declared);
            if !(allowed_by_env || allowed_by_extension || allowed_by_capability) {
                return Err(ValidationError::DisallowedCapability { capability, env });
            }
            for required_cap in grammar_requirements
                .required_capabilities
                .iter()
                .chain(required_capabilities_for_capability(capability).iter())
            {
                if is_soft_dependency(capability, *required_cap) {
                    continue;
                }
                if !declared.contains(required_cap) {
                    return Err(ValidationError::MissingRequiredCapability {
                        required_capability: *required_cap,
                        capability,
                    });
                }
            }
        }
    }
    validate_instruction_requirements(module, target_version, &declared, extensions)?;
    Ok(())
}

fn capability_allowed_by_extension(
    env: TargetEnv,
    capability: rspirv::spirv::Capability,
    extensions: &ExtensionSet,
) -> bool {
    let grammar_requirements = capability_info_from_grammar(capability);
    grammar_requirements
        .required_extensions
        .iter()
        .any(|required_ext| {
            extension_allowed_in_env(required_ext, env) && has_extension(extensions, required_ext)
        })
        || required_extension_for_capability(capability)
            .map(|required_ext| {
                extension_allowed_in_env(required_ext, env)
                    && has_extension(extensions, required_ext)
            })
            .unwrap_or(false)
}

fn has_extension(extensions: &ExtensionSet, required_extension: &str) -> bool {
    extensions
        .values
        .iter()
        .any(|ext| ext.as_str() == required_extension)
}

fn extension_allowed_in_env(extension: &str, env: TargetEnv) -> bool {
    env.is_extension_allowed(&ExtensionName::from(extension))
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

fn is_soft_dependency(
    capability: rspirv::spirv::Capability,
    required_capability: rspirv::spirv::Capability,
) -> bool {
    matches!(
        (capability, required_capability),
        (
            rspirv::spirv::Capability::Shader,
            rspirv::spirv::Capability::Matrix
        )
    )
}

fn validate_instruction_requirements(
    module: &Module,
    module_version: SpirvVersion,
    capabilities: &HashSet<rspirv::spirv::Capability>,
    extensions: &ExtensionSet,
) -> Result<(), ValidationError> {
    let target_version = module_version;
    for inst in module.all_inst_iter() {
        if matches!(
            inst.class.opcode,
            rspirv::spirv::Op::Decorate | rspirv::spirv::Op::DecorateId
        ) {
            if let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) {
                if matches!(
                    decoration,
                    rspirv::spirv::Decoration::Offset
                        | rspirv::spirv::Decoration::MatrixStride
                        | rspirv::spirv::Decoration::RowMajor
                        | rspirv::spirv::Decoration::ColMajor
                ) {
                    return Err(ValidationError::MemberOnlyDecorationUsedWithDecorate {
                        decoration: *decoration,
                    });
                }
            }
        }
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
            let resolved_operand = resolve_id_operand(module, operand);
            let operand = resolved_operand.as_ref().unwrap_or(operand);
            if matches!(operand, rspirv::dr::Operand::Capability(_)) {
                // Capability dependencies are validated separately to avoid over-constraining
                // the declaration order.
                continue;
            }
            if let Some(required_version) = required_spirv_version_for_operand(operand) {
                if target_version < required_version {
                    return Err(ValidationError::OperandRequiresSpirvVersion {
                        opcode: inst.class.opcode,
                        operand_index: index,
                        required_version,
                        target_version,
                    });
                }
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
            for &required_cap in manual_required_capabilities_for_operand(operand) {
                if !capabilities.contains(&required_cap) {
                    return Err(ValidationError::MissingOperandCapability {
                        opcode: inst.class.opcode,
                        operand_index: index,
                        required_capability: required_cap,
                    });
                }
            }
            for required_cap in grammar_required_capabilities_for_operand(operand) {
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
            for required_ext in grammar_required_extensions_for_operand(operand) {
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
        CooperativeMatrixKHR => Some("SPV_KHR_cooperative_matrix"),
        BindlessTextureNV => Some("SPV_NV_bindless_texture"),
        RayTracingNV => Some("SPV_NV_ray_tracing"),
        RayTracingKHR => Some("SPV_KHR_ray_tracing"),
        RayQueryKHR => Some("SPV_KHR_ray_query"),
        RayTracingPositionFetchKHR => Some("SPV_KHR_ray_tracing_position_fetch"),
        RayTracingMotionBlurNV => Some("SPV_NV_ray_tracing_motion_blur"),
        CooperativeMatrixNV => Some("SPV_NV_cooperative_matrix"),
        MeshShadingNV => Some("SPV_NV_mesh_shader"),
        MeshShadingEXT => Some("SPV_EXT_mesh_shader"),
        FragmentShadingRateKHR => Some("SPV_KHR_fragment_shading_rate"),
        FragmentDensityEXT => Some("SPV_EXT_fragment_invocation_density"),
        FragmentShaderSampleInterlockEXT
        | FragmentShaderShadingRateInterlockEXT
        | FragmentShaderPixelInterlockEXT => Some("SPV_EXT_fragment_shader_interlock"),
        ImageFootprintNV => Some("SPV_NV_shader_image_footprint"),
        RayTracingLinearSweptSpheresGeometryNV => Some("SPV_NV_linear_swept_spheres"),
        RayTracingDisplacementMicromapNV => Some("SPV_NV_displacement_micromap"),
        RayTracingOpacityMicromapEXT => Some("SPV_EXT_opacity_micromap"),
        AtomicFloat32MinMaxEXT | AtomicFloat64MinMaxEXT | AtomicFloat16MinMaxEXT => {
            Some("SPV_EXT_shader_atomic_float_min_max")
        }
        AtomicFloat16AddEXT | AtomicFloat32AddEXT | AtomicFloat64AddEXT => {
            Some("SPV_EXT_shader_atomic_float_add")
        }
        AtomicFloat16VectorNV => Some("SPV_NV_shader_atomic_float"),
        ShaderSMBuiltinsNV => Some("SPV_NV_shader_sm_builtins"),
        ShaderClockKHR => Some("SPV_KHR_shader_clock"),
        TileShadingQCOM => Some("SPV_QCOM_tile_shading"),
        SpecConditionalINTEL | FunctionVariantsINTEL => Some("SPV_INTEL_function_variants"),
        _ => None,
    }
}

fn extension_always_required(extension: &str) -> bool {
    extension.starts_with("SPV_NV_")
        || extension.starts_with("SPV_EXT_")
        || extension.starts_with("SPV_AMD_")
        || extension.starts_with("SPV_QCOM_")
        || matches!(
            extension,
            "SPV_KHR_ray_tracing"
                | "SPV_KHR_ray_query"
                | "SPV_KHR_ray_tracing_position_fetch"
                | "SPV_KHR_vulkan_memory_model"
                | "SPV_KHR_shader_clock"
        )
}

fn manual_required_spirv_version_for_capability(
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
        ShaderClockKHR => Some(SpirvVersion::new(1, 3)),
        DeviceGroup => Some(SpirvVersion::new(1, 3)),
        AtomicFloat16AddEXT
        | AtomicFloat32AddEXT
        | AtomicFloat64AddEXT
        | AtomicFloat16MinMaxEXT
        | AtomicFloat32MinMaxEXT
        | AtomicFloat64MinMaxEXT
        | AtomicFloat16VectorNV => Some(SpirvVersion::new(1, 3)),
        TileShadingQCOM => Some(SpirvVersion::new(1, 6)),
        PhysicalStorageBufferAddresses => Some(SpirvVersion::new(1, 4)),
        _ => None,
    }
}

fn required_spirv_version_for_extension(extension: &ExtensionName) -> Option<SpirvVersion> {
    let normalized = extension.as_str().to_ascii_lowercase();
    match normalized.as_str() {
        "spv_khr_vulkan_memory_model" | "spv_qcom_cooperative_matrix_conversion" => {
            Some(SpirvVersion::new(1, 3))
        }
        "spv_khr_workgroup_memory_explicit_layout" => Some(SpirvVersion::new(1, 4)),
        "spv_khr_physical_storage_buffer" => Some(SpirvVersion::new(1, 4)),
        "spv_khr_ray_tracing" | "spv_khr_ray_query" | "spv_khr_ray_tracing_position_fetch" => {
            Some(SpirvVersion::new(1, 4))
        }
        "spv_ext_mesh_shader"
        | "spv_nv_shader_invocation_reorder"
        | "spv_nv_cluster_acceleration_structure"
        | "spv_nv_linear_swept_spheres"
        | "spv_ext_shader_invocation_reorder"
        | "spv_qcom_image_processing"
        | "spv_qcom_image_processing2" => Some(SpirvVersion::new(1, 4)),
        "spv_qcom_tile_shading" => Some(SpirvVersion::new(1, 6)),
        "spv_ext_fragment_shader_interlock" => Some(SpirvVersion::new(1, 4)),
        "spv_khr_fragment_shading_rate" | "spv_ext_fragment_invocation_density" => {
            Some(SpirvVersion::new(1, 5))
        }
        "spv_khr_storage_buffer_storage_class" | "spv_khr_variable_pointers" => {
            Some(SpirvVersion::new(1, 3))
        }
        "spv_khr_shader_clock" | "spv_khr_device_group" => Some(SpirvVersion::new(1, 3)),
        "spv_khr_maximal_reconvergence" => Some(SpirvVersion::new(1, 6)),
        "spv_ext_descriptor_indexing" => Some(SpirvVersion::new(1, 5)),
        _ => None,
    }
}

fn manual_required_spirv_version_for_opcode(opcode: rspirv::spirv::Op) -> Option<SpirvVersion> {
    match opcode {
        rspirv::spirv::Op::TypeAccelerationStructureKHR | rspirv::spirv::Op::TypeRayQueryKHR => {
            Some(SpirvVersion::new(1, 4))
        }
        _ => None,
    }
}

fn manual_required_spirv_version_for_operand(
    operand: &rspirv::dr::Operand,
) -> Option<SpirvVersion> {
    match operand {
        rspirv::dr::Operand::MemoryAccess(mask)
            if mask.contains(rspirv::spirv::MemoryAccess::NONTEMPORAL) =>
        {
            Some(SpirvVersion::new(1, 6))
        }
        _ => None,
    }
}

fn manual_required_capabilities_for_operand(
    operand: &rspirv::dr::Operand,
) -> &'static [rspirv::spirv::Capability] {
    match operand {
        rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::NonUniform) => {
            &[rspirv::spirv::Capability::ShaderNonUniform]
        }
        _ => &[],
    }
}

fn required_spirv_version_for_opcode(opcode: rspirv::spirv::Op) -> Option<SpirvVersion> {
    merge_versions(
        grammar_required_spirv_version_for_opcode(opcode),
        manual_required_spirv_version_for_opcode(opcode),
    )
}

fn required_spirv_version_for_operand(operand: &rspirv::dr::Operand) -> Option<SpirvVersion> {
    merge_versions(
        grammar_required_spirv_version_for_operand(operand),
        manual_required_spirv_version_for_operand(operand),
    )
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
        VariablePointers => &[VariablePointersStorageBuffer],
        VariablePointersStorageBuffer => &[Shader],
        GroupNonUniformVote
        | GroupNonUniformArithmetic
        | GroupNonUniformBallot
        | GroupNonUniformShuffle
        | GroupNonUniformShuffleRelative
        | GroupNonUniformClustered
        | GroupNonUniformQuad => &[GroupNonUniform],
        SubgroupDispatch => &[DeviceEnqueue],
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

fn validate_extension_allowlist(module: &Module, env: TargetEnv) -> Result<(), ValidationError> {
    for inst in &module.extensions {
        if let Some(extension) = extension_operand(inst) {
            if !env.is_extension_allowed(&extension) {
                return Err(ValidationError::DisallowedExtension { extension, env });
            }
        }
    }
    Ok(())
}

fn validate_extensions(
    module: &Module,
    env: TargetEnv,
    target_version: SpirvVersion,
) -> Result<ExtensionSet, ValidationError> {
    let mut extensions = ExtensionSet::default();
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

/// Returns the SPIR-V version to use when validating a module for a given target.
/// This clamps the module-declared version to the environment's supported maximum
/// so version checks remain deterministic.
fn effective_spirv_version(env: TargetEnv, module_version: SpirvVersion) -> SpirvVersion {
    let env_version = env.spirv_version();
    if env_version.meets_or_exceeds(module_version) {
        module_version
    } else {
        env_version
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

fn enforce_struct_member_limit(
    struct_member_counts: &HashMap<ResultId, usize>,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    if let Some(&limit) = options.limits.get(&LIMIT_MAX_STRUCT_MEMBERS) {
        for &member_count in struct_member_counts.values() {
            if member_count as u32 > limit {
                return Err(ValidationError::LimitExceeded {
                    limit_kind: LIMIT_MAX_STRUCT_MEMBERS,
                    limit,
                    found: member_count as u32,
                });
            }
        }
    }
    Ok(())
}

fn enforce_function_arg_limit(
    module: &Module,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    if let Some(&limit) = options.limits.get(&LIMIT_MAX_FUNCTION_ARGS) {
        for function in &module.functions {
            let arg_count = function.parameters.len() as u32;
            if arg_count > limit {
                return Err(ValidationError::LimitExceeded {
                    limit_kind: LIMIT_MAX_FUNCTION_ARGS,
                    limit,
                    found: arg_count,
                });
            }
        }
    }
    Ok(())
}

fn enforce_struct_depth_limit(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    let Some(&limit) = options.limits.get(&LIMIT_MAX_STRUCT_DEPTH) else {
        return Ok(());
    };

    fn depth_for(
        ty: ResultId,
        defs: &HashMap<ResultId, rspirv::dr::Instruction>,
        memo: &mut HashMap<ResultId, u32>,
        visiting: &mut HashSet<ResultId>,
    ) -> u32 {
        if let Some(&cached) = memo.get(&ty) {
            return cached;
        }
        if visiting.contains(&ty) {
            return 1;
        }
        let Some(inst) = defs.get(&ty) else {
            return 0;
        };
        if inst.class.opcode != rspirv::spirv::Op::TypeStruct {
            memo.insert(ty, 0);
            return 0;
        }
        visiting.insert(ty);
        let mut max_child = 0u32;
        for operand in &inst.operands {
            if let rspirv::dr::Operand::IdRef(raw) = operand {
                if let Ok(child) = ResultId::try_from(*raw) {
                    let child_depth = depth_for(child, defs, memo, visiting);
                    max_child = max_child.max(child_depth);
                }
            }
        }
        visiting.remove(&ty);
        let depth = 1 + max_child;
        memo.insert(ty, depth);
        depth
    }

    let mut memo = HashMap::new();
    let mut visiting = HashSet::new();
    for inst in &module.types_global_values {
        if let Some(result_id) = inst.result_id {
            if inst.class.opcode == rspirv::spirv::Op::TypeStruct {
                if let Ok(id) = ResultId::try_from(result_id) {
                    let depth = depth_for(id, definitions, &mut memo, &mut visiting);
                    if depth > limit {
                        return Err(ValidationError::LimitExceeded {
                            limit_kind: LIMIT_MAX_STRUCT_DEPTH,
                            limit,
                            found: depth,
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

fn enforce_variable_limits(
    module: &Module,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    if let Some(&limit) = options.limits.get(&LIMIT_MAX_GLOBAL_VARIABLES) {
        let globals = module
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == rspirv::spirv::Op::Variable)
            .count() as u32;
        if globals > limit {
            return Err(ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_GLOBAL_VARIABLES,
                limit,
                found: globals,
            });
        }
    }

    if let Some(&limit) = options.limits.get(&LIMIT_MAX_LOCAL_VARIABLES) {
        let mut locals: u32 = 0;
        for function in &module.functions {
            for block in &function.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode == rspirv::spirv::Op::Variable {
                        if let Some(rspirv::dr::Operand::StorageClass(
                            rspirv::spirv::StorageClass::Function,
                        )) = inst.operands.first()
                        {
                            locals = locals.saturating_add(1);
                        }
                    }
                }
            }
        }
        if locals > limit {
            return Err(ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_LOCAL_VARIABLES,
                limit,
                found: locals,
            });
        }
    }

    if let Some(&limit) = options.limits.get(&LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH) {
        let mut max_depth = 0u32;
        for function in &module.functions {
            let mut depth = 0i32;
            for block in &function.blocks {
                for inst in &block.instructions {
                    match inst.class.opcode {
                        rspirv::spirv::Op::SelectionMerge | rspirv::spirv::Op::LoopMerge => {
                            depth = depth.saturating_add(1);
                            max_depth = max_depth.max(depth as u32);
                        }
                        rspirv::spirv::Op::Branch | rspirv::spirv::Op::BranchConditional => {
                            depth = (depth - 1).max(0);
                        }
                        _ => {}
                    }
                }
            }
        }
        if max_depth > limit {
            return Err(ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH,
                limit,
                found: max_depth,
            });
        }
    }

    Ok(())
}

fn enforce_switch_branch_limit(
    module: &Module,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    let Some(&limit) = options.limits.get(&LIMIT_MAX_SWITCH_BRANCHES) else {
        return Ok(());
    };

    for function in &module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == rspirv::spirv::Op::Switch {
                    // Operand order: selector id, default target id, then literal/target pairs.
                    let operands = &inst.operands;
                    if operands.len() < 2 {
                        continue;
                    }
                    let pair_count = (operands.len().saturating_sub(2)) / 2;
                    let branches = 1 + pair_count as u32; // include default target
                    if branches > limit {
                        return Err(ValidationError::LimitExceeded {
                            limit_kind: LIMIT_MAX_SWITCH_BRANCHES,
                            limit,
                            found: branches,
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

fn enforce_access_chain_limit(
    module: &Module,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    let Some(&limit) = options.limits.get(&LIMIT_MAX_ACCESS_CHAIN_INDEXES) else {
        return Ok(());
    };

    let access_chain_opcodes = [
        rspirv::spirv::Op::AccessChain,
        rspirv::spirv::Op::InBoundsAccessChain,
        rspirv::spirv::Op::PtrAccessChain,
        rspirv::spirv::Op::InBoundsPtrAccessChain,
        rspirv::spirv::Op::UntypedPtrAccessChainKHR,
        rspirv::spirv::Op::UntypedInBoundsPtrAccessChainKHR,
    ];

    let check_inst = |inst: &rspirv::dr::Instruction| -> Result<(), ValidationError> {
        if !access_chain_opcodes.contains(&inst.class.opcode) {
            return Ok(());
        }
        let num_operands = inst.operands.len();
        let indexes = match inst.class.opcode {
            rspirv::spirv::Op::AccessChain | rspirv::spirv::Op::InBoundsAccessChain => {
                num_operands.saturating_sub(1)
            }
            rspirv::spirv::Op::PtrAccessChain
            | rspirv::spirv::Op::InBoundsPtrAccessChain
            | rspirv::spirv::Op::UntypedPtrAccessChainKHR
            | rspirv::spirv::Op::UntypedInBoundsPtrAccessChainKHR => num_operands.saturating_sub(2),
            _ => 0,
        } as u32;

        if indexes > limit {
            return Err(ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_ACCESS_CHAIN_INDEXES,
                limit,
                found: indexes,
            });
        }
        Ok(())
    };

    for inst in &module.types_global_values {
        check_inst(inst)?;
    }

    for function in &module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                check_inst(inst)?;
            }
        }
    }

    Ok(())
}

fn enforce_logical_pointer_rules(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    capabilities: &HashSet<rspirv::spirv::Capability>,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    if options.relax_logical_pointer {
        return Ok(());
    }

    let addressing_model = module
        .memory_model
        .as_ref()
        .and_then(|inst| inst.operands.first())
        .and_then(|op| match op {
            rspirv::dr::Operand::AddressingModel(model) => Some(*model),
            _ => None,
        });
    let is_logical = matches!(
        addressing_model,
        Some(
            rspirv::spirv::AddressingModel::Logical
                | rspirv::spirv::AddressingModel::PhysicalStorageBuffer64
        )
    );
    if !is_logical {
        return Ok(());
    }

    for inst in module
        .types_global_values
        .iter()
        .chain(module.functions.iter().flat_map(|f| f.all_inst_iter()))
    {
        if inst.class.opcode != rspirv::spirv::Op::Variable {
            continue;
        }
        let Some(result_type) = inst.result_type else {
            continue;
        };
        let Ok(type_id) = TypeId::try_from(result_type) else {
            continue;
        };
        let Some(type_result_id) = ResultId::try_from(u32::from(type_id)).ok() else {
            continue;
        };
        let Some(type_inst) = definitions.get(&type_result_id) else {
            continue;
        };
        if type_inst.class.opcode != rspirv::spirv::Op::TypePointer
            && type_inst.class.opcode != rspirv::spirv::Op::TypeUntypedPointerKHR
        {
            continue;
        }
        let pointee_type_id = match type_inst.operands.get(1) {
            Some(rspirv::dr::Operand::IdRef(raw)) => ResultId::try_from(*raw).ok(),
            _ => None,
        };
        let Some(pointee_inst) = pointee_type_id.and_then(|id| definitions.get(&id)) else {
            continue;
        };
        if pointee_inst.class.opcode != rspirv::spirv::Op::TypePointer
            && pointee_inst.class.opcode != rspirv::spirv::Op::TypeUntypedPointerKHR
        {
            continue;
        }
        let pointee_storage_class = match pointee_inst.operands.first() {
            Some(rspirv::dr::Operand::StorageClass(sc)) => *sc,
            _ => continue,
        };
        if pointee_storage_class == rspirv::spirv::StorageClass::PhysicalStorageBuffer {
            continue;
        }

        let variable = inst
            .result_id
            .and_then(|id| Id::try_from(id).ok())
            .unwrap_or_else(|| Id::try_from(1).unwrap());

        match pointee_storage_class {
            rspirv::spirv::StorageClass::StorageBuffer => {
                if !capabilities.contains(&rspirv::spirv::Capability::VariablePointersStorageBuffer)
                {
                    return Err(ValidationError::LogicalPointerMissingCapability {
                        variable,
                        pointee_storage_class,
                        required_capability:
                            rspirv::spirv::Capability::VariablePointersStorageBuffer,
                    });
                }
            }
            rspirv::spirv::StorageClass::Workgroup => {
                if !capabilities.contains(&rspirv::spirv::Capability::VariablePointers) {
                    return Err(ValidationError::LogicalPointerMissingCapability {
                        variable,
                        pointee_storage_class,
                        required_capability: rspirv::spirv::Capability::VariablePointers,
                    });
                }
            }
            _ => {
                return Err(ValidationError::LogicalPointerPointeeStorageClassInvalid {
                    variable,
                    pointee_storage_class,
                });
            }
        }

        let var_storage_class = inst
            .operands
            .first()
            .and_then(|op| match op {
                rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                _ => None,
            })
            .unwrap_or(rspirv::spirv::StorageClass::Function);
        if var_storage_class != rspirv::spirv::StorageClass::Function
            && var_storage_class != rspirv::spirv::StorageClass::Private
        {
            return Err(ValidationError::LogicalPointerInvalidStorageClass {
                variable,
                storage_class: var_storage_class,
            });
        }
    }

    Ok(())
}

fn enforce_store_type_compatibility(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    for inst in module.all_inst_iter() {
        if inst.class.opcode != rspirv::spirv::Op::Store {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(ptr_id_raw)) = inst.operands.first() else {
            continue;
        };
        let Some(rspirv::dr::Operand::IdRef(obj_id_raw)) = inst.operands.get(1) else {
            continue;
        };
        let Ok(ptr_id) = ResultId::try_from(*ptr_id_raw) else {
            continue;
        };
        let Some(ptr_inst) = definitions.get(&ptr_id) else {
            continue;
        };
        let Some(ptr_type_raw) = ptr_inst.result_type else {
            continue;
        };
        let Ok(ptr_type_id) = TypeId::try_from(ptr_type_raw) else {
            continue;
        };
        let Some(ptr_type_result) = ResultId::try_from(u32::from(ptr_type_id)).ok() else {
            continue;
        };
        let Some(ptr_type_inst) = definitions.get(&ptr_type_result) else {
            continue;
        };
        if ptr_type_inst.class.opcode != rspirv::spirv::Op::TypePointer {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(pointee_raw)) = ptr_type_inst.operands.get(1) else {
            continue;
        };
        let Ok(pointee_id) = TypeId::try_from(*pointee_raw) else {
            continue;
        };

        let Ok(obj_id) = ResultId::try_from(*obj_id_raw) else {
            continue;
        };
        let Some(obj_inst) = definitions.get(&obj_id) else {
            continue;
        };
        let Some(obj_type_raw) = obj_inst.result_type else {
            continue;
        };
        let Ok(obj_type_id) = TypeId::try_from(obj_type_raw) else {
            continue;
        };

        if pointee_id == obj_type_id {
            continue;
        }

        if options.relax_struct_store {
            let layout_relaxed = options.relax_block_layout
                || options.uniform_buffer_standard_layout
                || options.scalar_block_layout
                || options.workgroup_scalar_block_layout;
            if layout_relaxed {
                continue;
            }
            if layout_compatible_types(
                pointee_id,
                obj_type_id,
                module,
                definitions,
                &mut HashSet::new(),
            ) {
                continue;
            }
        }

        return Err(ValidationError::StoreTypeMismatch {
            pointer: ptr_id,
            pointer_type: pointee_id,
            object_type: obj_type_id,
        });
    }

    Ok(())
}

fn layout_compatible_types(
    a: TypeId,
    b: TypeId,
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visiting: &mut HashSet<TypeId>,
) -> bool {
    if a == b {
        return true;
    }
    if !visiting.insert(a) {
        return false;
    }
    let Some(result_a) = ResultId::try_from(u32::from(a)).ok() else {
        visiting.remove(&a);
        return false;
    };
    let Some(result_b) = ResultId::try_from(u32::from(b)).ok() else {
        visiting.remove(&a);
        return false;
    };
    let Some(inst_a) = definitions.get(&result_a) else {
        visiting.remove(&a);
        return false;
    };
    let Some(inst_b) = definitions.get(&result_b) else {
        visiting.remove(&a);
        return false;
    };
    let compatible = match (inst_a.class.opcode, inst_b.class.opcode) {
        (rspirv::spirv::Op::TypeStruct, rspirv::spirv::Op::TypeStruct) => {
            if inst_a.operands.len() != inst_b.operands.len() {
                false
            } else {
                inst_a
                    .operands
                    .iter()
                    .zip(&inst_b.operands)
                    .all(|(op_a, op_b)| match (op_a, op_b) {
                        (rspirv::dr::Operand::IdRef(id_a), rspirv::dr::Operand::IdRef(id_b)) => {
                            let Ok(type_a) = TypeId::try_from(*id_a) else {
                                return false;
                            };
                            let Ok(type_b) = TypeId::try_from(*id_b) else {
                                return false;
                            };
                            layout_compatible_types(type_a, type_b, module, definitions, visiting)
                        }
                        _ => false,
                    })
            }
        }
        (rspirv::spirv::Op::TypeArray, rspirv::spirv::Op::TypeArray) => inst_a
            .operands
            .first()
            .and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id_a) => TypeId::try_from(*id_a).ok(),
                _ => None,
            })
            .and_then(|elem_a| {
                let elem_b = inst_b.operands.first().and_then(|op| match op {
                    rspirv::dr::Operand::IdRef(id_b) => TypeId::try_from(*id_b).ok(),
                    _ => None,
                })?;
                let len_a = array_length(inst_a, definitions);
                let len_b = array_length(inst_b, definitions);
                Some((elem_a, elem_b, len_a, len_b))
            })
            .is_some_and(|(elem_a, elem_b, len_a, len_b)| {
                let stride_a = array_stride(module, result_a);
                let stride_b = array_stride(module, result_b);
                len_a == len_b
                    && stride_a == stride_b
                    && layout_compatible_types(elem_a, elem_b, module, definitions, visiting)
            }),
        (rspirv::spirv::Op::TypeRuntimeArray, rspirv::spirv::Op::TypeRuntimeArray) => inst_a
            .operands
            .first()
            .and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id_a) => TypeId::try_from(*id_a).ok(),
                _ => None,
            })
            .and_then(|elem_a| {
                inst_b
                    .operands
                    .first()
                    .and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id_b) => TypeId::try_from(*id_b).ok(),
                        _ => None,
                    })
                    .map(|elem_b| (elem_a, elem_b))
            })
            .is_some_and(|(elem_a, elem_b)| {
                layout_compatible_types(elem_a, elem_b, module, definitions, visiting)
            }),
        (rspirv::spirv::Op::TypeVector, rspirv::spirv::Op::TypeVector) => {
            let (elem_a, count_a) = vector_info(inst_a);
            let (elem_b, count_b) = vector_info(inst_b);
            elem_a
                .zip(elem_b)
                .zip(count_a.zip(count_b))
                .is_some_and(|((a, b), (ca, cb))| {
                    ca == cb && layout_compatible_types(a, b, module, definitions, visiting)
                })
        }
        (rspirv::spirv::Op::TypeMatrix, rspirv::spirv::Op::TypeMatrix) => {
            let (col_a, count_a) = matrix_info(inst_a);
            let (col_b, count_b) = matrix_info(inst_b);
            col_a
                .zip(col_b)
                .zip(count_a.zip(count_b))
                .is_some_and(|((a, b), (ca, cb))| {
                    ca == cb && layout_compatible_types(a, b, module, definitions, visiting)
                })
        }
        _ => false,
    };
    visiting.remove(&a);
    compatible
}

fn array_length(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let len_id = match inst.operands.get(1) {
        Some(rspirv::dr::Operand::IdRef(id)) => ResultId::try_from(*id).ok()?,
        _ => return None,
    };
    let len_inst = definitions.get(&len_id)?;
    if len_inst.class.opcode != rspirv::spirv::Op::Constant {
        return None;
    }
    match len_inst.operands.first() {
        Some(rspirv::dr::Operand::LiteralBit32(v)) => Some(*v),
        Some(rspirv::dr::Operand::LiteralBit64(v)) => u32::try_from(*v).ok(),
        _ => None,
    }
}

fn vector_info(inst: &rspirv::dr::Instruction) -> (Option<TypeId>, Option<u32>) {
    let elem = inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    });
    let count = inst.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
        rspirv::dr::Operand::LiteralBit64(v) => u32::try_from(*v).ok(),
        _ => None,
    });
    (elem, count)
}

fn matrix_info(inst: &rspirv::dr::Instruction) -> (Option<TypeId>, Option<u32>) {
    let column = inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    });
    let count = inst.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
        rspirv::dr::Operand::LiteralBit64(v) => u32::try_from(*v).ok(),
        _ => None,
    });
    (column, count)
}

fn enforce_block_layout_rules(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    if options.skip_block_layout {
        return Ok(());
    }

    let scalar_layout = options.scalar_block_layout || options.workgroup_scalar_block_layout;
    let relax_layout =
        options.relax_block_layout || options.uniform_buffer_standard_layout || scalar_layout;

    let block_structs = collect_block_structs(module);
    for (struct_id, storage_classes) in block_structs {
        let Some(struct_inst) = definitions.get(&struct_id) else {
            continue;
        };
        if struct_inst.class.opcode != rspirv::spirv::Op::TypeStruct {
            continue;
        }
        if struct_inst.operands.is_empty() {
            continue;
        }
        let member_offsets = collect_member_offsets(module, struct_id);
        let member_count = struct_inst.operands.len();
        for (index, operand) in struct_inst.operands.iter().enumerate() {
            let Some(offset) = member_offsets.get(&MemberIndex(index as u32)) else {
                return Err(ValidationError::InvalidBlockLayout {
                    struct_type: struct_id,
                    reason: "missing OpMemberDecorate Offset".to_string(),
                });
            };
            let rspirv::dr::Operand::IdRef(member_type_id_raw) = operand else {
                continue;
            };
            let Ok(member_type_id) = TypeId::try_from(*member_type_id_raw) else {
                continue;
            };
            let Ok(member_result_id) = ResultId::try_from(u32::from(member_type_id)) else {
                continue;
            };
            let Some(member_inst) = definitions.get(&member_result_id) else {
                continue;
            };
            let Some(alignment) = type_alignment(
                member_type_id,
                definitions,
                &mut HashSet::new(),
                scalar_layout,
            ) else {
                continue;
            };
            if member_inst.class.opcode == rspirv::spirv::Op::TypeRuntimeArray {
                if index + 1 != member_count {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: "runtime array member must be the final struct member".to_string(),
                    });
                }
                if let Some(stride) = array_stride(module, member_result_id) {
                    if stride % alignment != 0 {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!(
                                "runtime array stride {stride} is not aligned to {alignment}"
                            ),
                        });
                    }
                }
                // Runtime array must be last; remaining checks do not apply.
                continue;
            }
            if member_inst.class.opcode == rspirv::spirv::Op::TypeArray {
                if let Some(stride) = array_stride(module, member_result_id) {
                    if stride % alignment != 0 {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!("array stride {stride} is not aligned to {alignment}"),
                        });
                    }
                    if let Some(rspirv::dr::Operand::IdRef(elem_raw)) = member_inst.operands.first()
                    {
                        if let Ok(elem_type) = TypeId::try_from(*elem_raw) {
                            if let Some(elem_size) =
                                type_layout_size(elem_type, definitions, &mut HashSet::new())
                            {
                                if elem_size > stride {
                                    return Err(ValidationError::InvalidBlockLayout {
                                        struct_type: struct_id,
                                        reason: format!(
                                            "array stride {stride} is smaller than element size {elem_size}"
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            if member_inst.class.opcode == rspirv::spirv::Op::TypeMatrix {
                let stride = member_matrix_stride(module, struct_id, MemberIndex(index as u32))
                    .ok_or_else(|| ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: "matrix member is missing MatrixStride".to_string(),
                    })?;
                if stride % alignment != 0 {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: format!("matrix stride {stride} is not aligned to {alignment}"),
                    });
                }
                let (column_type, _) = matrix_info(member_inst);
                if let Some(col_ty) = column_type {
                    if let Some(col_size) =
                        type_layout_size(col_ty, definitions, &mut HashSet::new())
                    {
                        if col_size > stride {
                            return Err(ValidationError::InvalidBlockLayout {
                                struct_type: struct_id,
                                reason: format!(
                                    "matrix stride {stride} is smaller than column size {col_size}"
                                ),
                            });
                        }
                        if relax_layout
                            && !scalar_layout
                            && member_is_row_major(module, struct_id, MemberIndex(index as u32))
                            && col_size > 16
                            && (offset % 16).saturating_add(col_size) > 16
                        {
                            return Err(ValidationError::InvalidBlockLayout {
                                struct_type: struct_id,
                                reason: "row-major matrix straddles 16-byte boundary under relaxed layout".to_string(),
                            });
                        }
                    }
                }
            }
            let Some(size) = type_layout_size(member_type_id, definitions, &mut HashSet::new())
            else {
                continue;
            };
            // Alignment rules: scalar block layout always uses scalar alignment; relaxed block
            // layout allows vectors to align to their scalar element size, otherwise require
            // alignment to the computed base alignment.
            if relax_layout
                && !scalar_layout
                && member_inst.class.opcode == rspirv::spirv::Op::TypeVector
            {
                let Some(scalar_align) = vector_scalar_alignment(member_inst, definitions) else {
                    continue;
                };
                if offset % scalar_align != 0 {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: format!(
                            "member offset {offset} is not aligned to vector scalar element size {}",
                            scalar_align
                        ),
                    });
                }
                let Some(vector_size) =
                    type_layout_size(member_type_id, definitions, &mut HashSet::new())
                else {
                    continue;
                };
                if vector_size > 16 && (offset % 16).saturating_add(vector_size) > 16 {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: "vector member straddles 16-byte boundary under relaxed layout"
                            .to_string(),
                    });
                }
            } else if member_inst.class.opcode == rspirv::spirv::Op::TypeMatrix {
                if let Some(stride) =
                    member_matrix_stride(module, struct_id, MemberIndex(index as u32))
                {
                    if stride % alignment != 0 {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!("matrix stride {stride} is not aligned to {alignment}"),
                        });
                    }
                    let (column_type, _) = matrix_info(member_inst);
                    if let Some(col_ty) = column_type {
                        if let Some(col_size) =
                            type_layout_size(col_ty, definitions, &mut HashSet::new())
                        {
                            if col_size > stride {
                                return Err(ValidationError::InvalidBlockLayout {
                                    struct_type: struct_id,
                                    reason: format!(
                                        "matrix stride {stride} is smaller than column size {col_size}"
                                    ),
                                });
                            }
                        }
                    }
                }
            } else if offset % alignment != 0 {
                return Err(ValidationError::InvalidBlockLayout {
                    struct_type: struct_id,
                    reason: format!(
                        "member offset {offset} is not aligned to required alignment {alignment}"
                    ),
                });
            }

            let next_offset = offset.saturating_add(size);
            // Ensure no overlap with the next member offset (if any).
            if let Some(next) = member_offsets
                .get(&MemberIndex((index as u32) + 1))
                .copied()
            {
                if next < next_offset {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: "member offsets overlap".to_string(),
                    });
                }
            }
        }

        // Basic storage-class specific hint: Workgroup layout relaxations may be requested.
        if storage_classes.contains(&rspirv::spirv::StorageClass::Workgroup)
            && !options.workgroup_scalar_block_layout
        {
            // No additional action; reserved for future stricter checks.
        }
    }

    Ok(())
}

fn collect_block_structs(
    module: &Module,
) -> HashMap<ResultId, HashSet<rspirv::spirv::StorageClass>> {
    let mut structs: HashMap<ResultId, HashSet<rspirv::spirv::StorageClass>> = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode == rspirv::spirv::Op::Decorate {
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
            ) = (inst.operands.first(), inst.operands.get(1))
            {
                if *decoration == rspirv::spirv::Decoration::Block
                    || *decoration == rspirv::spirv::Decoration::BufferBlock
                {
                    if let Ok(struct_id) = ResultId::try_from(*target) {
                        structs.entry(struct_id).or_default();
                    }
                }
            }
        }
    }

    // Map struct ids to storage classes where they are used.
    for var in &module.types_global_values {
        if var.class.opcode != rspirv::spirv::Op::Variable {
            continue;
        }
        let Some(rspirv::dr::Operand::StorageClass(sc)) = var.operands.first() else {
            continue;
        };
        let Some(result_type) = var.result_type else {
            continue;
        };
        let Ok(ptr_type_id) = TypeId::try_from(result_type) else {
            continue;
        };
        let Some(ptr_inst) = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(u32::from(ptr_type_id)))
        else {
            continue;
        };
        if ptr_inst.class.opcode != rspirv::spirv::Op::TypePointer {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(pointee)) = ptr_inst.operands.get(1) else {
            continue;
        };
        if let Ok(struct_id) = ResultId::try_from(*pointee) {
            structs.entry(struct_id).or_default().insert(*sc);
        }
    }

    structs
}

fn collect_member_offsets(module: &Module, struct_id: ResultId) -> HashMap<MemberIndex, u32> {
    let mut offsets = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode == rspirv::spirv::Op::MemberDecorate {
            let mut operands = inst.operands.iter();
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::LiteralBit32(member)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
            ) = (operands.next(), operands.next(), operands.next())
            {
                if *decoration == rspirv::spirv::Decoration::Offset {
                    if let Ok(target_id) = ResultId::try_from(*target) {
                        if target_id == struct_id {
                            if let Some(rspirv::dr::Operand::LiteralBit32(offset)) = operands.next()
                            {
                                offsets.insert(MemberIndex(*member), *offset);
                            }
                        }
                    }
                }
            }
        }
    }
    offsets
}

fn type_layout_size(
    ty: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visiting: &mut HashSet<TypeId>,
) -> Option<u32> {
    if !visiting.insert(ty) {
        return None;
    }
    let inst = definitions.get(&ResultId::try_from(u32::from(ty)).ok()?)?;
    let size = match inst.class.opcode {
        rspirv::spirv::Op::TypeInt | rspirv::spirv::Op::TypeFloat => {
            inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::LiteralBit32(bits) => Some(*bits / 8),
                _ => None,
            })
        }
        rspirv::spirv::Op::TypeVector => {
            let (elem, count) = vector_info(inst);
            let (elem, count) = (elem?, count?);
            let elem_size = type_layout_size(elem, definitions, visiting)?;
            Some(elem_size.saturating_mul(count))
        }
        rspirv::spirv::Op::TypeMatrix => {
            let (column, count) = matrix_info(inst);
            let (column, count) = (column?, count?);
            let col_size = type_layout_size(column, definitions, visiting)?;
            Some(col_size.saturating_mul(count))
        }
        rspirv::spirv::Op::TypeArray => {
            let elem = inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
                _ => None,
            })?;
            let elem_size = type_layout_size(elem, definitions, visiting)?;
            let len = array_length(inst, definitions)?;
            Some(elem_size.saturating_mul(len))
        }
        rspirv::spirv::Op::TypeRuntimeArray => None, // unsized
        rspirv::spirv::Op::TypeStruct => {
            let mut offset: u32 = 0;
            for op in &inst.operands {
                let ty = match op {
                    rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok()?,
                    _ => return None,
                };
                let size = type_layout_size(ty, definitions, visiting)?;
                offset = offset.saturating_add(size);
            }
            Some(offset)
        }
        _ => None,
    };
    visiting.remove(&ty);
    size
}

fn type_alignment(
    ty: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visiting: &mut HashSet<TypeId>,
    scalar_layout: bool,
) -> Option<u32> {
    if !visiting.insert(ty) {
        return None;
    }
    let inst = definitions.get(&ResultId::try_from(u32::from(ty)).ok()?)?;
    let alignment = match inst.class.opcode {
        rspirv::spirv::Op::TypeInt | rspirv::spirv::Op::TypeFloat => {
            inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::LiteralBit32(bits) => Some(*bits / 8),
                _ => None,
            })
        }
        rspirv::spirv::Op::TypeVector => {
            let (elem, count) = vector_info(inst);
            let (elem, count) = (elem?, count?);
            let elem_align = type_alignment(elem, definitions, visiting, scalar_layout)?;
            if scalar_layout {
                Some(elem_align)
            } else {
                elem_align.checked_mul(count)
            }
        }
        rspirv::spirv::Op::TypeMatrix => {
            // Matrix alignment follows its column vector alignment.
            let (column, _) = matrix_info(inst);
            let column = column?;
            type_alignment(column, definitions, visiting, scalar_layout)
        }
        rspirv::spirv::Op::TypeArray | rspirv::spirv::Op::TypeRuntimeArray => {
            let elem = inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
                _ => None,
            })?;
            type_alignment(elem, definitions, visiting, scalar_layout)
        }
        rspirv::spirv::Op::TypeStruct => {
            let mut max_align = 1;
            for op in &inst.operands {
                let ty = match op {
                    rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok()?,
                    _ => return None,
                };
                let align = type_alignment(ty, definitions, visiting, scalar_layout)?;
                max_align = max_align.max(align);
            }
            Some(max_align)
        }
        _ => None,
    };
    visiting.remove(&ty);
    alignment
}

fn vector_scalar_alignment(
    vector_inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let elem = vector_inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    })?;
    type_alignment(elem, definitions, &mut HashSet::new(), true)
}

fn array_stride(module: &Module, array_type: ResultId) -> Option<u32> {
    for inst in &module.annotations {
        if inst.class.opcode == rspirv::spirv::Op::Decorate {
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
                Some(rspirv::dr::Operand::LiteralBit32(stride)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            ) {
                if *decoration == rspirv::spirv::Decoration::ArrayStride {
                    if let Ok(target_id) = ResultId::try_from(*target) {
                        if target_id == array_type {
                            return Some(*stride);
                        }
                    }
                }
            }
        }
    }
    None
}

fn member_is_row_major(module: &Module, struct_id: ResultId, member: MemberIndex) -> bool {
    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::MemberDecorate {
            continue;
        }
        let mut ops = inst.operands.iter();
        let Some(rspirv::dr::Operand::IdRef(target)) = ops.next() else {
            continue;
        };
        let Ok(target_id) = ResultId::try_from(*target) else {
            continue;
        };
        if target_id != struct_id {
            continue;
        }
        let Some(rspirv::dr::Operand::LiteralBit32(member_idx)) = ops.next() else {
            continue;
        };
        if *member_idx != member.0 {
            continue;
        }
        let Some(rspirv::dr::Operand::Decoration(decoration)) = ops.next() else {
            continue;
        };
        if *decoration == rspirv::spirv::Decoration::RowMajor {
            return true;
        }
    }
    false
}

fn member_matrix_stride(module: &Module, struct_id: ResultId, member: MemberIndex) -> Option<u32> {
    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::MemberDecorate {
            continue;
        }
        let mut ops = inst.operands.iter();
        let Some(rspirv::dr::Operand::IdRef(target)) = ops.next() else {
            continue;
        };
        let Ok(target_id) = ResultId::try_from(*target) else {
            continue;
        };
        if target_id != struct_id {
            continue;
        }
        let Some(rspirv::dr::Operand::LiteralBit32(member_idx)) = ops.next() else {
            continue;
        };
        if *member_idx != member.0 {
            // Continue scanning; some producers may reorder decorations.
            continue;
        }
        let Some(rspirv::dr::Operand::Decoration(decoration)) = ops.next() else {
            continue;
        };
        if *decoration != rspirv::spirv::Decoration::MatrixStride {
            continue;
        }
        if let Some(rspirv::dr::Operand::LiteralBit32(stride)) = ops.next() {
            return Some(*stride);
        }
    }
    None
}

fn enforce_offset_texture_operand_rule(
    module: &Module,
    env: TargetEnv,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    if options.allow_offset_texture_operand || options.before_hlsl_legalization {
        return Ok(());
    }
    if !is_vulkan_env(env) {
        return Ok(());
    }

    let gather_opcodes = [
        rspirv::spirv::Op::ImageGather,
        rspirv::spirv::Op::ImageDrefGather,
        rspirv::spirv::Op::ImageSparseGather,
        rspirv::spirv::Op::ImageSparseDrefGather,
    ];

    for inst in module.all_inst_iter() {
        let has_offset = inst.operands.iter().any(|op| {
            matches!(
                op,
                rspirv::dr::Operand::ImageOperands(mask)
                    if mask.contains(rspirv::spirv::ImageOperands::OFFSET)
            )
        });
        if has_offset && !gather_opcodes.contains(&inst.class.opcode) {
            return Err(ValidationError::OffsetTextureOperandDisallowed {
                opcode: inst.class.opcode,
            });
        }
    }

    Ok(())
}

fn enforce_vulkan_bitwise_widths(
    module: &Module,
    env: TargetEnv,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    options: &ValidationOptions,
) -> Result<(), ValidationError> {
    if options.allow_vulkan_32_bit_bitwise || !is_vulkan_env(env) {
        return Ok(());
    }

    let bitwise_opcodes = [
        rspirv::spirv::Op::ShiftRightLogical,
        rspirv::spirv::Op::ShiftRightArithmetic,
        rspirv::spirv::Op::ShiftLeftLogical,
        rspirv::spirv::Op::BitwiseOr,
        rspirv::spirv::Op::BitwiseXor,
        rspirv::spirv::Op::BitwiseAnd,
        rspirv::spirv::Op::Not,
    ];

    for inst in module.all_inst_iter() {
        if !bitwise_opcodes.contains(&inst.class.opcode) {
            continue;
        }
        let Some(raw_type) = inst.result_type else {
            continue;
        };
        let Ok(type_id) = TypeId::try_from(raw_type) else {
            continue;
        };
        if let Some(bit_width) = int_bit_width(type_id, definitions) {
            if bit_width != 32 {
                return Err(ValidationError::VulkanBitwiseRequires32Bit {
                    opcode: inst.class.opcode,
                    bit_width,
                });
            }
        }
    }

    Ok(())
}

fn int_bit_width(
    type_id: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let Ok(result_id) = ResultId::try_from(u32::from(type_id)) else {
        return None;
    };
    let inst = definitions.get(&result_id)?;
    match inst.class.opcode {
        rspirv::spirv::Op::TypeInt => inst.operands.first().and_then(|op| match op {
            rspirv::dr::Operand::LiteralBit32(width) => Some(*width),
            rspirv::dr::Operand::LiteralBit64(width) => Some(*width as u32),
            _ => None,
        }),
        rspirv::spirv::Op::TypeVector => {
            let component = match inst.operands.first() {
                Some(rspirv::dr::Operand::IdRef(raw)) => TypeId::try_from(*raw).ok()?,
                _ => return None,
            };
            int_bit_width(component, definitions)
        }
        _ => None,
    }
}

fn is_vulkan_env(env: TargetEnv) -> bool {
    matches!(
        env,
        TargetEnv::Vulkan1_0
            | TargetEnv::Vulkan1_1
            | TargetEnv::Vulkan1_1Spirv1_4
            | TargetEnv::Vulkan1_2
            | TargetEnv::Vulkan1_3
            | TargetEnv::Vulkan1_4
    )
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
                if let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) {
                    if matches!(
                        decoration,
                        rspirv::spirv::Decoration::Offset
                            | rspirv::spirv::Decoration::MatrixStride
                            | rspirv::spirv::Decoration::RowMajor
                            | rspirv::spirv::Decoration::ColMajor
                    ) {
                        return Err(ValidationError::MemberOnlyDecorationUsedWithDecorate {
                            decoration: *decoration,
                        });
                    }
                }
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

fn constant_u32(module: &Module, id: u32) -> Option<u32> {
    module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(id) && is_constant_opcode(inst.class.opcode))
        .and_then(|inst| inst.operands.last())
        .and_then(|operand| match operand {
            rspirv::dr::Operand::LiteralBit32(value) => Some(*value),
            _ => None,
        })
}

fn resolve_id_operand(
    module: &Module,
    operand: &rspirv::dr::Operand,
) -> Option<rspirv::dr::Operand> {
    match operand {
        rspirv::dr::Operand::IdMemorySemantics(id) => constant_u32(module, *id).map(|value| {
            rspirv::dr::Operand::MemorySemantics(
                rspirv::spirv::MemorySemantics::from_bits_truncate(value),
            )
        }),
        rspirv::dr::Operand::IdScope(id) => constant_u32(module, *id).and_then(|value| {
            rspirv::spirv::Scope::from_u32(value).map(rspirv::dr::Operand::Scope)
        }),
        _ => None,
    }
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
        let entry_opcode = ep.class.opcode;
        let mut operands = ep.operands.iter();
        if entry_opcode == rspirv::spirv::Op::ConditionalEntryPointINTEL {
            // Skip the condition operand.
            let _ = operands.next();
        }
        // Next operand is ExecutionModel; skip it.
        let _ = operands.next();
        let function_id = match operands.next() {
            Some(rspirv::dr::Operand::IdRef(id)) => {
                ResultId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                    kind: IdKind::Operand,
                    opcode: entry_opcode,
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
                        opcode: entry_opcode,
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
    env: TargetEnv,
    options: &ValidationOptions,
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

        if let Some(execution_mode) = execution_mode_from_operand(mode.operands.get(1)) {
            if execution_mode == rspirv::spirv::ExecutionMode::LocalSizeId
                && !local_size_id_allowed(env, options)
            {
                return Err(ValidationError::LocalSizeIdNotAllowed { env });
            }
        }
    }
    Ok(())
}

fn execution_mode_from_operand(
    operand: Option<&rspirv::dr::Operand>,
) -> Option<rspirv::spirv::ExecutionMode> {
    match operand {
        Some(rspirv::dr::Operand::ExecutionMode(mode)) => Some(*mode),
        Some(rspirv::dr::Operand::LiteralBit32(raw)) => {
            rspirv::spirv::ExecutionMode::from_u32(*raw)
        }
        _ => None,
    }
}

fn local_size_id_allowed(env: TargetEnv, options: &ValidationOptions) -> bool {
    match env {
        TargetEnv::Vulkan1_0
        | TargetEnv::Vulkan1_1
        | TargetEnv::Vulkan1_1Spirv1_4
        | TargetEnv::Vulkan1_2 => options.allow_localsizeid,
        _ => true,
    }
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

fn collect_result_types(module: &Module) -> Result<HashMap<ResultId, TypeId>, ValidationError> {
    let mut map = HashMap::new();
    for inst in module.all_inst_iter() {
        if let (Some(result_id), Some(result_type)) = (inst.result_id, inst.result_type) {
            let id = ResultId::try_from(result_id).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::Result,
                opcode: inst.class.opcode,
            })?;
            let ty = TypeId::try_from(result_type).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::ResultType,
                opcode: inst.class.opcode,
            })?;
            map.insert(id, ty);
        }
    }
    Ok(map)
}

fn is_void_type(type_id: TypeId, definitions: &HashMap<ResultId, rspirv::dr::Instruction>) -> bool {
    let raw: u32 = Id::from(type_id).into();
    let Ok(result_id) = ResultId::try_from(raw) else {
        return false;
    };
    definitions
        .get(&result_id)
        .map(|inst| inst.class.opcode == rspirv::spirv::Op::TypeVoid)
        .unwrap_or(false)
}

fn collect_declared_capabilities(module: &Module) -> HashSet<rspirv::spirv::Capability> {
    module
        .capabilities
        .iter()
        .filter_map(capability_operand)
        .collect()
}

fn is_type_opcode(opcode: rspirv::spirv::Op) -> bool {
    matches!(
        opcode,
        rspirv::spirv::Op::TypeVoid
            | rspirv::spirv::Op::TypeBool
            | rspirv::spirv::Op::TypeInt
            | rspirv::spirv::Op::TypeFloat
            | rspirv::spirv::Op::TypeVector
            | rspirv::spirv::Op::TypeMatrix
            | rspirv::spirv::Op::TypeImage
            | rspirv::spirv::Op::TypeSampler
            | rspirv::spirv::Op::TypeSampledImage
            | rspirv::spirv::Op::TypeArray
            | rspirv::spirv::Op::TypeRuntimeArray
            | rspirv::spirv::Op::TypeStruct
            | rspirv::spirv::Op::TypeOpaque
            | rspirv::spirv::Op::TypePointer
            | rspirv::spirv::Op::TypeFunction
            | rspirv::spirv::Op::TypeEvent
            | rspirv::spirv::Op::TypeDeviceEvent
            | rspirv::spirv::Op::TypeReserveId
            | rspirv::spirv::Op::TypeQueue
            | rspirv::spirv::Op::TypePipe
            | rspirv::spirv::Op::TypeForwardPointer
            | rspirv::spirv::Op::TypePipeStorage
            | rspirv::spirv::Op::TypeNamedBarrier
            | rspirv::spirv::Op::TypeAccelerationStructureKHR
            | rspirv::spirv::Op::TypeCooperativeMatrixKHR
            | rspirv::spirv::Op::TypeCooperativeMatrixNV
            | rspirv::spirv::Op::TypeRayQueryKHR
            | rspirv::spirv::Op::TypeHitObjectNV
    )
}

fn validate_type_functions(
    module: &Module,
    opcodes: &HashMap<ResultId, rspirv::spirv::Op>,
) -> Result<(), ValidationError> {
    for inst in &module.types_global_values {
        if inst.class.opcode != rspirv::spirv::Op::TypeFunction {
            continue;
        }
        let type_id = inst
            .result_id
            .and_then(|raw| TypeId::try_from(raw).ok())
            .ok_or(ValidationError::ZeroId {
                kind: IdKind::Result,
                opcode: inst.class.opcode,
            })?;

        let mut operands = inst.operands.iter();
        let return_type = match operands.next() {
            Some(rspirv::dr::Operand::IdRef(raw)) => TypeId::try_from(*raw)
                .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?,
            _ => {
                return Err(ValidationError::InvalidTypeFunction { type_id });
            }
        };

        let return_id = ResultId::try_from(u32::from(return_type))
            .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?;
        let return_opcode = opcodes
            .get(&return_id)
            .copied()
            .ok_or(ValidationError::InvalidTypeFunction { type_id })?;
        if !is_type_opcode(return_opcode) {
            return Err(ValidationError::InvalidTypeFunction { type_id });
        }

        for op in operands {
            let param_type = match op {
                rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw)
                    .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?,
                _ => {
                    return Err(ValidationError::InvalidTypeFunction { type_id });
                }
            };
            let param_id = ResultId::try_from(u32::from(param_type))
                .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?;
            let param_opcode = opcodes
                .get(&param_id)
                .copied()
                .ok_or(ValidationError::InvalidTypeFunction { type_id })?;
            if param_opcode == rspirv::spirv::Op::TypeVoid {
                return Err(ValidationError::FunctionTypeParameterVoid {
                    type_id,
                    parameter: param_type,
                });
            }
            if !is_type_opcode(param_opcode) {
                return Err(ValidationError::InvalidTypeFunction { type_id });
            }
        }
    }
    Ok(())
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
        validate_module, validate_module_with_options, validate_words, CheckedBound, DeclaredBound,
        DecorationTargetId, DecorationTargetKind, ExtensionName, Id, IdKind, MaybeValidModule,
        MemberDecorationTargetId, MemberIndex, ModuleWords, OperandId, ResultId, Schema,
        SpirvVersion, TypeId, ValidModuleCache, ValidatableModule, ValidationError,
    };
    use crate::assembly::assemble_text;
    use crate::target_env::TargetEnv;
    use crate::validation::{
        array_stride, collect_result_instructions, enforce_store_type_compatibility,
        format_validation_error, format_validation_error_from_words, layout_compatible_types,
        parse_module, FriendlyNames, ValidationOptions,
    };
    use rspirv::spirv::{Capability, FunctionControl, MemoryModel, Op};
    use std::collections::{HashMap, HashSet};
    use std::num::NonZeroU32;
    use std::sync::Arc;

    fn op(word_count: u16, opcode: u16) -> u32 {
        ((word_count as u32) << 16) | opcode as u32
    }

    const EXT_SPV_GOOGLE_DECORATE_STRING_WORDS: [u32; 7] = [
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ];

    #[test]
    fn opcode_helpers_classify_capabilities_and_extensions() {
        use super::instruction_layout::{is_capability_opcode, is_extension_opcode};

        assert!(is_capability_opcode(Op::Capability));
        assert!(is_capability_opcode(Op::ConditionalCapabilityINTEL));
        assert!(is_extension_opcode(Op::Extension));
        assert!(is_extension_opcode(Op::ConditionalExtensionINTEL));
        assert!(!is_capability_opcode(Op::Extension));
        assert!(!is_extension_opcode(Op::Capability));
        assert!(!is_extension_opcode(Op::ExtInstImport));
    }

    #[test]
    fn mode_stage_orders_mode_settings() {
        use super::instruction_layout::{mode_stage, ModeStage};

        assert_eq!(mode_stage(Op::Capability), Some(ModeStage::Capabilities));
        assert_eq!(
            mode_stage(Op::ConditionalCapabilityINTEL),
            Some(ModeStage::Capabilities)
        );
        assert_eq!(mode_stage(Op::Extension), Some(ModeStage::Extensions));
        assert_eq!(
            mode_stage(Op::ConditionalExtensionINTEL),
            Some(ModeStage::Extensions)
        );
        assert_eq!(
            mode_stage(Op::ConditionalEntryPointINTEL),
            Some(ModeStage::EntryPoint)
        );
        assert_eq!(
            mode_stage(Op::ExtInstImport),
            Some(ModeStage::ExtInstImport)
        );
        assert_eq!(mode_stage(Op::MemoryModel), Some(ModeStage::MemoryModel));
        assert_eq!(mode_stage(Op::EntryPoint), Some(ModeStage::EntryPoint));
        assert_eq!(
            mode_stage(Op::ExecutionMode),
            Some(ModeStage::ExecutionMode)
        );
        assert_eq!(
            mode_stage(Op::ExecutionModeId),
            Some(ModeStage::ExecutionMode)
        );
        assert_eq!(mode_stage(Op::TypeVoid), None);
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
    fn operand_requires_capability_from_grammar() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Fragment %main \"main\"",
            "%void = OpTypeVoid",
            "%u32 = OpTypeInt 32 0",
            "%ptr = OpTypePointer Input %u32",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "%var = OpVariable %ptr Input",
            "OpDecorate %var BuiltIn SubgroupSize",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::MissingOperandCapability {
                opcode: rspirv::spirv::Op::Decorate,
                operand_index: 2,
                required_capability: rspirv::spirv::Capability::Kernel
            }
        );
    }

    #[test]
    fn operand_requires_extension_from_grammar() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Fragment %main \"main\"",
            "OpExecutionMode %main SubgroupUniformControlFlowKHR",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::MissingOperandExtension {
                opcode: rspirv::spirv::Op::ExecutionMode,
                operand_index: 1,
                required_extension: ExtensionName::from("SPV_KHR_subgroup_uniform_control_flow"),
            }
        );
    }

    #[test]
    fn conditional_extension_rejected_when_disallowed() {
        use rspirv::binary::Assemble;
        use rspirv::dr::Instruction;
        use rspirv::spirv::{AddressingModel, Capability, MemoryModel, Op};

        let mut module = rspirv::dr::Module::new();
        module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![rspirv::dr::Operand::Capability(Capability::Shader)],
        ));
        module.memory_model = Some(Instruction::new(
            Op::MemoryModel,
            None,
            None,
            vec![
                rspirv::dr::Operand::AddressingModel(AddressingModel::Logical),
                rspirv::dr::Operand::MemoryModel(MemoryModel::GLSL450),
            ],
        ));
        module.extensions.push(Instruction::new(
            Op::ConditionalExtensionINTEL,
            None,
            None,
            vec![rspirv::dr::Operand::LiteralString(
                "SPV_KHR_ray_tracing".into(),
            )],
        ));
        module.header = Some(rspirv::dr::ModuleHeader::new(5));
        let void = rspirv::dr::Operand::IdRef(1);
        module
            .types_global_values
            .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
        module.types_global_values.push(Instruction::new(
            Op::TypeFunction,
            Some(2),
            None,
            vec![void.clone()],
        ));
        let mut func = rspirv::dr::Function::new();
        func.def = Some(Instruction::new(
            Op::Function,
            Some(3),
            Some(1),
            vec![
                rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                rspirv::dr::Operand::IdRef(2),
            ],
        ));
        let mut block = rspirv::dr::Block::new();
        block.label = Some(Instruction::new(Op::Label, Some(4), None, vec![]));
        block
            .instructions
            .push(Instruction::new(Op::Return, None, None, vec![]));
        func.blocks.push(block);
        func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
        module.functions.push(func);

        let binary = module.assemble();
        let error = validate_module(&binary, TargetEnv::WebGpu0).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("KHR_ray_tracing"),
                env: TargetEnv::WebGpu0
            }
        );
    }

    #[test]
    fn conditional_extension_after_functions_rejected_for_ordering() {
        // Hand-rolled module: capability + memory model, then types/functions,
        // then a conditional extension placed after the function to trigger
        // layout ordering validation.
        let binary = vec![
            0x0723_0203,
            0x0001_0600,
            0,
            5, // bound
            0,
            op(2, Op::Capability as u16),
            Capability::Shader as u32,
            op(3, Op::MemoryModel as u16),
            rspirv::spirv::AddressingModel::Logical as u32,
            MemoryModel::GLSL450 as u32,
            op(2, Op::TypeVoid as u16),
            1,
            op(3, Op::TypeFunction as u16),
            2, // result id
            1, // return type
            op(5, Op::Function as u16),
            1, // result type (void)
            3, // result id
            FunctionControl::NONE.bits(),
            2, // fn type
            op(2, Op::Label as u16),
            4, // label
            op(1, Op::Return as u16),
            op(1, Op::FunctionEnd as u16),
            op(6, Op::ConditionalExtensionINTEL as u16),
            0x5f565053, // "SPV_"
            0x5f52484b, // "KHR_"
            0x5f796172, // "ray_"
            0x63617274, // "trac"
            0x00676e69, // "ing\0"
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn ext_inst_import_requires_memory_model() {
        // OpExtInstImport before OpMemoryModel should be reported as a memory-model ordering violation.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            3,          // bound (ids up to 2)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            0x0006000b, // OpExtInstImport %1 "GLSL.std.450"
            1,
            0x4c53_4c47, // "GLSL"
            0x6474_732e, // ".std"
            0x3035_342e, // ".450"
            0,           // padding/null terminator
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::MissingMemoryModel);
    }

    #[test]
    fn conditional_extension_must_precede_types_and_globals() {
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
            op(2, 19), // OpTypeVoid %1 (types/globals)
            1,
            op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (after types -> error)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn conditional_capability_must_precede_types_and_globals() {
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
            op(2, 19), // OpTypeVoid %1 (types/globals)
            1,
            op(3, 6250), // OpConditionalCapabilityINTEL %2 Linkage (after types -> error)
            2,
            rspirv::spirv::Capability::Linkage as u32,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
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
    fn decorations_cannot_follow_functions() {
        use rspirv::{binary::Assemble, dr::Builder, spirv::Decoration, spirv::Op};

        let mut builder = Builder::new();
        builder.set_version(1, 0);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        let void = builder.type_void();
        let fn_type = builder.type_function(void, std::iter::empty::<u32>());
        let fn_id = builder
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let mut words = builder.module().assemble();
        words.push(op(3, Op::Decorate as u16));
        words.push(fn_id);
        words.push(Decoration::RelaxedPrecision as u32);

        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Decorate
            }
        );
    }

    #[test]
    fn decorations_cannot_follow_types_and_globals() {
        // Annotations must appear before the types-and-globals section.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            2,          // bound (ids up to 1)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, rspirv::spirv::Op::TypeStruct as u16), // %1 = OpTypeStruct
            1,
            op(3, rspirv::spirv::Op::Decorate as u16), // OpDecorate %1 Block (after types -> error)
            1,
            rspirv::spirv::Decoration::Block as u32,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Decorate
            }
        );
    }

    #[test]
    fn decoration_group_cannot_follow_types_and_globals() {
        // Annotation section opcodes such as OpDecorationGroup must appear before the types/globals section.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            2,          // bound (ids up to 1)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, rspirv::spirv::Op::TypeStruct as u16), // %1 = OpTypeStruct
            1,
            op(2, rspirv::spirv::Op::DecorationGroup as u16), // OpDecorationGroup %1 (misordered)
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorationGroup
            }
        );
    }

    #[test]
    fn group_decorate_cannot_follow_types_and_globals() {
        // OpGroupDecorate must remain in the annotations section; it is invalid after types/globals.
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
            op(2, 73), // OpDecorationGroup %1 (annotations section)
            1,
            op(2, rspirv::spirv::Op::TypeStruct as u16), // %2 = OpTypeStruct
            2,
            op(3, rspirv::spirv::Op::GroupDecorate as u16), // OpGroupDecorate %1 %2 (after types -> error)
            1,
            2,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::GroupDecorate
            }
        );
    }

    #[test]
    fn group_member_decorate_cannot_follow_types_and_globals() {
        // OpGroupMemberDecorate must also stay in the annotations section before types/globals.
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
            op(2, 73), // OpDecorationGroup %1 (annotations)
            1,
            op(2, rspirv::spirv::Op::TypeStruct as u16), // %2 = OpTypeStruct
            2,
            op(4, rspirv::spirv::Op::GroupMemberDecorate as u16), // OpGroupMemberDecorate %1 %2 0 (after types -> error)
            1,
            2,
            0,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::GroupMemberDecorate
            }
        );
    }

    #[test]
    fn decorate_id_cannot_follow_types_and_globals() {
        // OpDecorateId belongs to the annotations section; placing it after globals is invalid.
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
            op(4, rspirv::spirv::Op::TypeInt as u16), // %1 = OpTypeInt 32 0
            1,
            32,
            0,
            op(4, rspirv::spirv::Op::TypePointer as u16), // %2 = OpTypePointer Function %1
            2,
            rspirv::spirv::StorageClass::Function as u32,
            1,
            op(4, rspirv::spirv::Op::Constant as u16), // %3 = OpConstant %1 4
            1,
            3,
            4,
            op(4, rspirv::spirv::Op::Variable as u16), // %4 = OpVariable %2 Function
            2,
            4,
            rspirv::spirv::StorageClass::Function as u32,
            op(4, rspirv::spirv::Op::DecorateId as u16), // OpDecorateId %4 AlignmentId %3 (after types/globals -> error)
            4,
            rspirv::spirv::Decoration::AlignmentId as u32,
            3,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorateId
            }
        );
    }

    #[test]
    fn decorate_string_cannot_follow_types_and_globals() {
        // OpDecorateString must appear in the annotations section before types/globals.
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
            op(4, rspirv::spirv::Op::TypeInt as u16), // %1 = OpTypeInt 32 0
            1,
            32,
            0,
            op(4, rspirv::spirv::Op::TypePointer as u16), // %2 = OpTypePointer Function %1
            2,
            rspirv::spirv::StorageClass::Function as u32,
            1,
            op(4, rspirv::spirv::Op::Variable as u16), // %3 = OpVariable %2 Function
            2,
            3,
            rspirv::spirv::StorageClass::Function as u32,
            op(4, rspirv::spirv::Op::DecorateString as u16), // OpDecorateString %3 UserSemantic "foo" (after globals -> error)
            3,
            rspirv::spirv::Decoration::UserSemantic as u32,
            0x006f_6f66, // "foo"
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorateString
            }
        );
    }

    #[test]
    fn member_decorate_string_cannot_follow_types_and_globals() {
        // OpMemberDecorateString also belongs to the annotations section and must not follow types/globals.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            4,          // bound (ids up to 3)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, rspirv::spirv::Op::TypeInt as u16), // %1 = OpTypeInt 32 0
            1,
            32,
            0,
            op(3, rspirv::spirv::Op::TypeStruct as u16), // %2 = OpTypeStruct %1
            2,
            1,
            op(5, rspirv::spirv::Op::MemberDecorateString as u16), // OpMemberDecorateString %2 0 UserSemantic "foo" (after type -> error)
            2,
            0,
            rspirv::spirv::Decoration::UserSemantic as u32,
            0x006f_6f66, // "foo"
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemberDecorateString
            }
        );
    }

    #[test]
    fn decorations_must_follow_entry_points() {
        // Annotations must not precede the entry-point section.
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
            op(3, rspirv::spirv::Op::Decorate as u16), // OpDecorate %1 RelaxedPrecision (before entry point -> error)
            1,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(2, rspirv::spirv::Op::TypeVoid as u16), // %1 = OpTypeVoid
            1,
            op(3, rspirv::spirv::Op::TypeFunction as u16), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, rspirv::spirv::Op::Function as u16), // %3 = OpFunction %1 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253),                                  // OpReturn
            op(1, 56),                                   // OpFunctionEnd
            op(5, rspirv::spirv::Op::EntryPoint as u16), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
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
    fn extensions_cannot_follow_entry_points() {
        // OpExtension must appear before the entry-point section.
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
            op(5, 15), // OpEntryPoint Vertex %1 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            1,
            0x6e69_616d, // "main"
            0,
            op(8, rspirv::spirv::Op::Extension as u16), // OpExtension "SPV_GOOGLE_decorate_string" (after entry point -> error)
            0x5f56_5053,                                // "SPV_"
            0x474f_4f47,                                // "GOOG"
            0x645f_454c,                                // "LE_d"
            0x726f_6365,                                // "ecor"
            0x5f65_7461,                                // "ate_"
            0x6972_7473,                                // "stri"
            0x0000_676e,                                // "ng\0"
            op(2, 19),                                  // OpTypeVoid %2
            2,
            op(3, 33), // OpTypeFunction %3 %2
            3,
            2,
            op(5, 54), // OpFunction %1 None %3
            2,
            1,
            0,
            3,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn decorations_cannot_appear_inside_functions() {
        // Hand-built binary with a decoration inside the function body to ensure layout checking
        // rejects annotations in the function section.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54),  // OpFunction %1 %4 None %2
            1,          // result type
            4,          // result id
            0,          // FunctionControl None
            2,          // function type
            op(2, 248), // OpLabel %3
            3,
            op(3, 71), // OpDecorate %3 RelaxedPrecision (illegal in function section)
            3,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Decorate
            }
        );
    }

    #[test]
    fn decorations_cannot_precede_memory_model() {
        // Missing OpMemoryModel with a decoration recorded before any other violation should
        // surface an InstructionBeforeMemoryModel error referencing the decoration opcode.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            2,          // bound (ids up to 1)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 71), // OpDecorate %1 RelaxedPrecision
            1,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(2, 19), // OpTypeVoid %1 (appears after the decoration but still before memory model)
            1,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::InstructionBeforeMemoryModel {
                opcode: rspirv::spirv::Op::Decorate
            }
        );
    }

    #[test]
    fn member_decorations_cannot_appear_inside_functions() {
        // MemberDecorate belongs to the annotations section; placing it inside a function should
        // be rejected by layout validation.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            7,          // bound (ids up to 6)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(3, 30), // OpTypeStruct %1 %2
            1,
            2,
            op(2, 19), // OpTypeVoid %3
            3,
            op(3, 33), // OpTypeFunction %4 %3
            4,
            3,
            op(5, 54), // OpFunction %3 %5 None %4
            3,
            5,
            0,
            4,
            op(2, 248), // OpLabel %6
            6,
            op(4, 72), // OpMemberDecorate %1 0 RowMajor (inside function -> error)
            1,
            0,
            rspirv::spirv::Decoration::RowMajor as u32,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemberDecorate
            }
        );
    }

    #[test]
    fn member_decorate_cannot_follow_functions() {
        // MemberDecorate must appear in the annotations section; placing it after functions should
        // trigger a layout error.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            7,          // bound (ids up to 6)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(3, 30), // OpTypeStruct %1 %2
            1,
            2,
            op(2, 19), // OpTypeVoid %3
            3,
            op(3, 33), // OpTypeFunction %4 %3
            4,
            3,
            op(5, 54), // OpFunction %3 %5 None %4
            3,
            5,
            0,
            4,
            op(2, 248), // OpLabel %6
            6,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(4, 72),  // OpMemberDecorate %1 0 RowMajor (after functions -> error)
            1,
            0,
            rspirv::spirv::Decoration::RowMajor as u32,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemberDecorate
            }
        );
    }

    #[test]
    fn decoration_group_cannot_appear_inside_functions() {
        // OpDecorationGroup belongs to the annotations section; ensure it is rejected when placed
        // inside a function body.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(2, 73), // OpDecorationGroup %5 (illegal inside function section)
            5,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorationGroup
            }
        );
    }

    #[test]
    fn decoration_group_cannot_follow_functions() {
        // OpDecorationGroup must appear in the annotations section before functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(2, 73),  // OpDecorationGroup %5 (after functions -> error)
            5,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorationGroup
            }
        );
    }

    #[test]
    fn group_decorate_cannot_follow_functions() {
        // OpGroupDecorate must stay in the annotations section; placing it after functions should
        // be rejected by the layout pass.
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
            op(2, 73), // OpDecorationGroup %1 (annotations section)
            1,
            op(2, 19), // OpTypeVoid %2
            2,
            op(3, 33), // OpTypeFunction %3 %2
            3,
            2,
            op(5, 54), // OpFunction %2 %4 None %3
            2,
            4,
            0,
            3,
            op(2, 248), // OpLabel %5
            5,
            op(1, 253),                                     // OpReturn
            op(1, 56),                                      // OpFunctionEnd
            op(3, rspirv::spirv::Op::GroupDecorate as u16), // OpGroupDecorate %1 %4 (after functions -> error)
            1,
            4,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::GroupDecorate
            }
        );
    }

    #[test]
    fn group_decorate_cannot_appear_inside_functions() {
        // OpGroupDecorate is an annotation and must not appear in the function section.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            7,          // bound (ids up to 6)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 73), // OpDecorationGroup %1 (annotations section)
            1,
            op(2, 19), // OpTypeVoid %2
            2,
            op(3, 33), // OpTypeFunction %3 %2
            3,
            2,
            op(5, 54), // OpFunction %2 %4 None %3
            2,
            4,
            0,
            3,
            op(2, 248), // OpLabel %5
            5,
            op(3, rspirv::spirv::Op::GroupDecorate as u16), // OpGroupDecorate %1 %4 (inside function -> error)
            1,
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::GroupDecorate
            }
        );
    }

    #[test]
    fn group_member_decorate_cannot_follow_functions() {
        // OpGroupMemberDecorate must remain in the annotations section; placing it after functions
        // should be rejected.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            8,          // bound (ids up to 7)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 73), // OpDecorationGroup %1
            1,
            op(4, 21), // OpTypeInt %3 32 0
            3,
            32,
            0,
            op(3, 30), // OpTypeStruct %2 %3
            2,
            3,
            op(2, 19), // OpTypeVoid %4
            4,
            op(3, 33), // OpTypeFunction %5 %4
            5,
            4,
            op(5, 54), // OpFunction %4 %6 None %5
            4,
            6,
            0,
            5,
            op(2, 248), // OpLabel %7
            7,
            op(1, 253),                                           // OpReturn
            op(1, 56),                                            // OpFunctionEnd
            op(4, rspirv::spirv::Op::GroupMemberDecorate as u16), // OpGroupMemberDecorate %1 %2 0 (after functions -> error)
            1,
            2,
            0,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::GroupMemberDecorate
            }
        );
    }

    #[test]
    fn member_names_cannot_appear_inside_functions() {
        // OpMemberName belongs to the names section; placing it inside a function should be
        // rejected by layout validation.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            7,          // bound (ids up to 6)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %1 32 0
            1,
            32,
            0,
            op(3, 30), // OpTypeStruct %2 %1
            2,
            1,
            op(2, 19), // OpTypeVoid %3
            3,
            op(3, 33), // OpTypeFunction %4 %3
            4,
            3,
            op(5, 54), // OpFunction %3 %5 None %4
            3,
            5,
            0,
            4,
            op(2, 248), // OpLabel %6
            6,
            op(4, 6), // OpMemberName %2 0 "f" (inside function -> error)
            2,
            0,
            0x0000_0066, // "f"
            op(1, 253),  // OpReturn
            op(1, 56),   // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemberName
            }
        );
    }

    #[test]
    fn decorate_string_cannot_appear_inside_functions() {
        // OpDecorateString is an annotation and must not appear in a function body.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            9,          // bound (ids up to 8)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %1 32 0
            1,
            32,
            0,
            op(4, 32), // OpTypePointer %2 Function %1
            2,
            rspirv::spirv::StorageClass::Function as u32,
            1,
            op(2, 19), // OpTypeVoid %3
            3,
            op(4, 33), // OpTypeFunction %4 %3 %2
            4,
            3,
            2,
            op(5, 54), // OpFunction %3 %5 None %4
            3,
            5,
            0,
            4,
            op(2, 248), // OpLabel %6
            6,
            op(4, 59), // OpVariable %2 %7 Function
            2,         // result type
            7,         // result id
            rspirv::spirv::StorageClass::Function as u32,
            op(4, rspirv::spirv::Op::DecorateString as u16), // OpDecorateString %7 UserSemantic "foo" (inside function -> error)
            7,
            rspirv::spirv::Decoration::UserSemantic as u32,
            0x006f_6f66, // "foo"
            op(1, 253),  // OpReturn
            op(1, 56),   // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorateString
            }
        );
    }

    #[test]
    fn decorate_string_cannot_follow_functions() {
        // OpDecorateString must appear in the annotations section before functions.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            9,          // bound (ids up to 8)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %1 32 0
            1,
            32,
            0,
            op(4, 32), // OpTypePointer %2 Function %1
            2,
            rspirv::spirv::StorageClass::Function as u32,
            1,
            op(2, 19), // OpTypeVoid %3
            3,
            op(4, 33), // OpTypeFunction %4 %3 %2
            4,
            3,
            2,
            op(5, 54), // OpFunction %3 %5 None %4
            3,
            5,
            0,
            4,
            op(2, 248), // OpLabel %6
            6,
            op(4, 59), // OpVariable %2 %7 Function
            2,         // result type
            7,         // result id
            rspirv::spirv::StorageClass::Function as u32,
            op(1, 253),                                      // OpReturn
            op(1, 56),                                       // OpFunctionEnd
            op(4, rspirv::spirv::Op::DecorateString as u16), // OpDecorateString %7 UserSemantic "foo" (after functions -> error)
            7,
            rspirv::spirv::Decoration::UserSemantic as u32,
            0x006f_6f66, // "foo"
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorateString
            }
        );
    }

    #[test]
    fn decorate_id_cannot_appear_inside_functions() {
        // OpDecorateId is an annotation and must not appear in the function section.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            10,         // bound (ids up to 9)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %1 32 0
            1,
            32,
            0,
            op(4, 32), // OpTypePointer %2 Function %1
            2,
            rspirv::spirv::StorageClass::Function as u32,
            1,
            op(4, 43), // OpConstant %1 4 -> %3
            1,
            3,
            4,
            op(2, 19), // OpTypeVoid %4
            4,
            op(4, 33), // OpTypeFunction %5 %4 %2
            5,
            4,
            2,
            op(5, 54), // OpFunction %4 %6 None %5
            4,
            6,
            0,
            5,
            op(2, 248), // OpLabel %7
            7,
            op(4, 59), // OpVariable %2 %8 Function
            2,
            8,
            rspirv::spirv::StorageClass::Function as u32,
            op(4, rspirv::spirv::Op::DecorateId as u16), // OpDecorateId %8 AlignmentId %3 (inside function -> error)
            8,
            rspirv::spirv::Decoration::AlignmentId as u32,
            3,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorateId
            }
        );
    }

    #[test]
    fn decorate_id_cannot_follow_functions() {
        // OpDecorateId must remain in the annotations section ahead of functions.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            10,         // bound (ids up to 9)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %1 32 0
            1,
            32,
            0,
            op(4, 32), // OpTypePointer %2 Function %1
            2,
            rspirv::spirv::StorageClass::Function as u32,
            1,
            op(4, 43), // OpConstant %1 4 -> %3
            1,
            3,
            4,
            op(2, 19), // OpTypeVoid %4
            4,
            op(4, 33), // OpTypeFunction %5 %4 %2
            5,
            4,
            2,
            op(5, 54), // OpFunction %4 %6 None %5
            4,
            6,
            0,
            5,
            op(2, 248), // OpLabel %7
            7,
            op(4, 59), // OpVariable %2 %8 Function
            2,
            8,
            rspirv::spirv::StorageClass::Function as u32,
            op(1, 253),                                  // OpReturn
            op(1, 56),                                   // OpFunctionEnd
            op(4, rspirv::spirv::Op::DecorateId as u16), // OpDecorateId %8 AlignmentId %3 (after functions -> error)
            8,
            rspirv::spirv::Decoration::AlignmentId as u32,
            3,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::DecorateId
            }
        );
    }

    #[test]
    fn member_decorate_string_cannot_follow_functions() {
        // OpMemberDecorateString is an annotation and must appear before functions.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            7,          // bound (ids up to 6)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %1 32 0
            1,
            32,
            0,
            op(3, 30), // OpTypeStruct %2 %1
            2,
            1,
            op(2, 19), // OpTypeVoid %3
            3,
            op(3, 33), // OpTypeFunction %4 %3
            4,
            3,
            op(5, 54), // OpFunction %3 %5 None %4
            3,
            5,
            0,
            4,
            op(2, 248), // OpLabel %6
            6,
            op(1, 253),                                            // OpReturn
            op(1, 56),                                             // OpFunctionEnd
            op(5, rspirv::spirv::Op::MemberDecorateString as u16), // OpMemberDecorateString %2 0 UserSemantic "foo"
            2,                                                     // target
            0,                                                     // member index
            rspirv::spirv::Decoration::UserSemantic as u32,
            0x006f_6f66, // "foo"
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemberDecorateString
            }
        );
    }

    #[test]
    fn group_member_decorate_cannot_appear_inside_functions() {
        // OpGroupMemberDecorate is an annotation and must not appear in the function section.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            8,          // bound (ids up to 7)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 73), // OpDecorationGroup %1
            1,
            op(4, 21), // OpTypeInt %3 32 0
            3,
            32,
            0,
            op(3, 30), // OpTypeStruct %2 %3
            2,
            3,
            op(2, 19), // OpTypeVoid %4
            4,
            op(3, 33), // OpTypeFunction %5 %4
            5,
            4,
            op(5, 54), // OpFunction %4 %6 None %5
            4,
            6,
            0,
            5,
            op(2, 248), // OpLabel %7
            7,
            op(4, rspirv::spirv::Op::GroupMemberDecorate as u16), // OpGroupMemberDecorate %1 %2 0 (inside function -> error)
            1,
            2,
            0,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::GroupMemberDecorate
            }
        );
    }

    #[test]
    fn capability_cannot_appear_inside_functions() {
        // Capabilities belong to the module header; placing one in the function section should
        // trigger a layout error.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54),  // OpFunction %1 %3 None %2
            1,          // result type
            3,          // result id
            0,          // FunctionControl None
            2,          // function type
            op(2, 248), // OpLabel %4
            4,
            op(2, 17), // OpCapability Kernel (illegal inside function)
            rspirv::spirv::Capability::Kernel as u32,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
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
    fn extension_cannot_appear_inside_functions() {
        // Extensions belong to the early module sections; reject an extension in the function body.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54),  // OpFunction %1 %3 None %2
            1,          // result type
            3,          // result id
            0,          // FunctionControl None
            2,          // function type
            op(2, 248), // OpLabel %4
            4,
            op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (illegal inside function)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn extension_cannot_follow_functions() {
        // Extensions must appear before functions; placing one after functions should be rejected.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(8, 10),  // OpExtension "SPV_GOOGLE_decorate_string" (after functions -> error)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
        ];

        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn source_cannot_appear_inside_functions() {
        // Debug/Source instructions must not appear inside function bodies.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54),  // OpFunction %1 %3 None %2
            1,          // result type
            3,          // result id
            0,          // FunctionControl None
            2,          // function type
            op(2, 248), // OpLabel %4
            4,
            op(3, 3), // OpSource GLSL 450 (illegal inside function)
            rspirv::spirv::SourceLanguage::GLSL as u32,
            450,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
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
    fn source_cannot_follow_functions() {
        // Debug/Source instructions must stay in the debug section before functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(3, 3),   // OpSource GLSL 450 (after functions -> error)
            rspirv::spirv::SourceLanguage::GLSL as u32,
            450,
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
    fn source_extension_cannot_appear_inside_functions() {
        // OpSourceExtension must remain in the Debug1 section, not inside functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(2, 4), // OpSourceExtension "ext" (illegal inside function)
            0x0074_7865,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::SourceExtension
            }
        );
    }

    #[test]
    fn source_extension_cannot_follow_functions() {
        // OpSourceExtension belongs to Debug1 and must appear before functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(2, 4),   // OpSourceExtension "ext" (after functions -> error)
            0x0074_7865,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::SourceExtension
            }
        );
    }

    #[test]
    fn source_continued_cannot_appear_inside_functions() {
        // OpSourceContinued must remain in the Debug1 section, not inside functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(2, 2), // OpSourceContinued "c" (illegal inside function)
            0x0000_0063,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::SourceContinued
            }
        );
    }

    #[test]
    fn source_continued_cannot_follow_functions() {
        // OpSourceContinued belongs to Debug1 and must appear before functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(2, 2),   // OpSourceContinued "c" (after functions -> error)
            0x0000_0063,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::SourceContinued
            }
        );
    }

    #[test]
    fn memory_model_cannot_appear_inside_functions() {
        // OpMemoryModel must appear before functions; reject it inside a function body.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            5,          // bound (ids up to 4)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54),  // OpFunction %1 %3 None %2
            1,          // result type
            3,          // result id
            0,          // FunctionControl None
            2,          // function type
            op(2, 248), // OpLabel %4
            4,
            op(3, 14), // OpMemoryModel Logical GLSL450 (illegal inside function)
            0,
            1,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::FunctionBeforeMemoryModel);
    }

    #[test]
    fn ext_inst_import_cannot_appear_inside_functions() {
        // Imported instruction sets must be declared before functions; reject occurrences in the
        // function section.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54),  // OpFunction %1 %3 None %2
            1,          // result type
            3,          // result id
            0,          // FunctionControl None
            2,          // function type
            op(2, 248), // OpLabel %4
            4,
            op(3, rspirv::spirv::Op::ExtInstImport as u16), // OpExtInstImport %5 "G" (illegal inside function)
            5,
            0x0000_0047, // "G"
            op(1, 253),  // OpReturn
            op(1, 56),   // OpFunctionEnd
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
    fn ext_inst_import_cannot_follow_functions() {
        // Imported instruction sets must precede functions; reject when placed after function
        // definitions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253),                                     // OpReturn
            op(1, 56),                                      // OpFunctionEnd
            op(3, rspirv::spirv::Op::ExtInstImport as u16), // OpExtInstImport %5 "G" (after functions -> error)
            5,
            0x0000_0047, // "G"
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
    fn member_decorate_string_cannot_appear_inside_functions() {
        // OpMemberDecorateString must stay in the annotations section; placing it inside a
        // function body should be rejected.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            8,          // bound (ids up to 7)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %1 32 0
            1,
            32,
            0,
            op(3, 30), // OpTypeStruct %2 %1
            2,
            1,
            op(2, 19), // OpTypeVoid %3
            3,
            op(3, 33), // OpTypeFunction %4 %3
            4,
            3,
            op(5, 54), // OpFunction %3 %5 None %4
            3,
            5,
            0,
            4,
            op(2, 248), // OpLabel %6
            6,
            op(5, rspirv::spirv::Op::MemberDecorateString as u16), // OpMemberDecorateString %2 0 UserSemantic "foo" (inside function -> error)
            2,
            0,
            rspirv::spirv::Decoration::UserSemantic as u32,
            0x006f_6f66, // "foo"
            op(1, 253),  // OpReturn
            op(1, 56),   // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemberDecorateString
            }
        );
    }

    #[test]
    fn member_name_cannot_follow_functions() {
        // OpMemberName must remain in the names section; placing it after functions should be
        // rejected by layout validation.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            7,          // bound (ids up to 6)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 21), // OpTypeInt %1 32 0
            1,
            32,
            0,
            op(3, 30), // OpTypeStruct %2 %1
            2,
            1,
            op(2, 19), // OpTypeVoid %3
            3,
            op(3, 33), // OpTypeFunction %4 %3
            4,
            3,
            op(5, 54), // OpFunction %3 %5 None %4
            3,
            5,
            0,
            4,
            op(2, 248), // OpLabel %6
            6,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(4, 6),   // OpMemberName %2 0 "f" (after functions -> error)
            2,
            0,
            0x0000_0066, // "f"
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemberName
            }
        );
    }

    #[test]
    fn debug_names_cannot_appear_inside_functions() {
        // Hand-built binary with OpName in the function section to ensure it is rejected.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(3, 5), // OpName %4 "fn" (invalid inside function section)
            4,
            0x006e_0066, // "fn"
            op(1, 253),  // OpReturn
            op(1, 56),   // OpFunctionEnd
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
    fn capability_cannot_follow_entry_points() {
        // Capabilities must be declared before entry points.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(2, 17), // OpCapability Geometry (misordered after entry point)
            rspirv::spirv::Capability::Geometry as u32,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::Capability
            }
        );
    }

    #[test]
    fn ext_inst_import_cannot_follow_entry_points() {
        // Imported instruction sets must appear before entry points.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            0x0006000b, // OpExtInstImport %5 "GLSL.std.450" (misordered after entry point)
            5,
            0x4c53_4c47, // "GLSL"
            0x6474_732e, // ".std"
            0x3035_342e, // ".450"
            0,           // null terminator
            op(2, 19),   // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ExtInstImport
            }
        );
    }

    #[test]
    fn memory_model_cannot_follow_entry_points() {
        // OpMemoryModel must be declared before the entry-point section.
        let binary = vec![
            0x0723_0203, // magic
            0x0001_0000, // version
            0,           // generator
            5,           // bound (ids up to 4)
            0,           // schema
            op(2, 17),   // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(5, 15), // OpEntryPoint Vertex %3 "main" (before memory model -> error)
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::MemoryModel
            }
        );
    }

    #[test]
    fn entry_points_must_precede_debug_names() {
        // OpEntryPoint must appear before the debug/names sections.
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
            op(4, 5), // OpName %3 "main" (debug names before entry point)
            3,
            0x6e69_616d, // "main"
            0,
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::EntryPoint
            }
        );
    }

    #[test]
    fn execution_modes_must_precede_debug_names() {
        // OpExecutionMode must appear before the debug/names sections.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(4, 5), // OpName %3 "main" (debug names before execution mode)
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft (after debug names -> error)
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ExecutionMode
            }
        );
    }

    #[test]
    fn entry_points_must_precede_debug_instructions() {
        // Debug/source instructions must not appear before entry points.
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
            op(3, 3), // OpSource Unknown 0 (debug instruction before entry point -> error)
            0,
            0,
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::EntryPoint
            }
        );
    }

    #[test]
    fn execution_modes_must_precede_debug_instructions() {
        // Debug/source instructions must not appear before execution modes.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 3), // OpSource Unknown 0 (debug instruction before execution mode -> error)
            0,
            0,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ExecutionMode
            }
        );
    }

    #[test]
    fn entry_points_cannot_follow_types_and_globals() {
        // Types/globals belong after entry points.
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
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(5, 15),  // OpEntryPoint Vertex %3 "main" (misordered after types/globals)
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
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
    fn execution_modes_cannot_follow_annotations() {
        // Execution modes must appear before the annotations section.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 71), // OpDecorate %3 RelaxedPrecision (annotations before execution mode)
            3,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft (misordered after annotations)
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ExecutionMode
            }
        );
    }

    #[test]
    fn execution_modes_cannot_follow_types_and_globals() {
        // Execution modes must appear before the types/globals section.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2 (types/globals section already begun)
            1,
            3,
            0,
            2,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft (misordered after types/globals)
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ExecutionMode
            }
        );
    }

    #[test]
    fn execution_modes_cannot_follow_functions() {
        // Execution modes must appear before function bodies.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(3, 16),  // OpExecutionMode %3 OriginUpperLeft (after functions -> error)
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ExecutionMode
            }
        );
    }

    #[test]
    fn capability_cannot_follow_execution_modes() {
        // Capabilities must precede the execution-mode section.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(2, 17), // OpCapability Geometry (misordered after execution mode)
            rspirv::spirv::Capability::Geometry as u32,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::Capability
            }
        );
    }

    #[test]
    fn extension_cannot_follow_execution_modes() {
        // Extensions must precede the execution-mode section.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered after execution mode)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn ext_inst_import_cannot_follow_execution_modes() {
        // Imported instruction sets must precede the execution-mode section.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            7,          // bound (ids up to 6)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            0x0006000b, // OpExtInstImport %5 "GLSL.std.450" (misordered after execution mode)
            5,
            0x4c53_4c47, // "GLSL"
            0x6474_732e, // ".std"
            0x3035_342e, // ".450"
            0,           // null terminator
            op(2, 19),   // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ExtInstImport
            }
        );
    }

    #[test]
    fn conditional_extension_cannot_follow_execution_modes() {
        // Conditional extensions must precede the execution-mode section.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(3, rspirv::spirv::Op::ConditionalExtensionINTEL as u16), // misordered after execution mode
            0x0000_0058,                                                // "X"
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn conditional_capability_cannot_follow_execution_modes() {
        // Conditional capabilities must also precede execution modes.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, 16), // OpExecutionMode %3 OriginUpperLeft
            3,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(3, rspirv::spirv::Op::ConditionalCapabilityINTEL as u16), // misordered after execution mode
            rspirv::spirv::Capability::InputAttachment as u32,
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn entry_points_cannot_follow_annotations() {
        // Entry points must appear before the annotations section.
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
            op(3, 71), // OpDecorate %3 RelaxedPrecision (annotations before entry point)
            3,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(5, 15), // OpEntryPoint Vertex %3 "main" (misordered after annotations)
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::EntryPoint
            }
        );
    }

    #[test]
    fn entry_points_cannot_follow_functions() {
        // Entry points cannot trail function definitions.
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
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(5, 15),  // OpEntryPoint Vertex %3 "main" (misordered after functions)
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
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
    fn conditional_capability_cannot_follow_entry_points() {
        // Conditional capabilities must also be declared before entry points.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, rspirv::spirv::Op::ConditionalCapabilityINTEL as u16), // misordered after entry point
            rspirv::spirv::Capability::InputAttachment as u32,
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn conditional_extension_cannot_follow_entry_points() {
        // Conditional extensions must be declared before entry points.
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
            op(5, 15), // OpEntryPoint Vertex %3 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(3, rspirv::spirv::Op::ConditionalExtensionINTEL as u16), // misordered after entry point
            0x0000_0058,                                                // "X"
            0,
            op(2, 19), // %1 = OpTypeVoid
            1,
            op(3, 33), // %2 = OpTypeFunction %1
            2,
            1,
            op(5, 54), // %3 = OpFunction %1 None %2
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
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
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
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
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
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("sampler image address mode is required for bindless capability");
        assert_eq!(text_error, expected);

        let binary = assemble_text(&text).expect("assemble");
        let binary_error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Vulkan1_2)
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
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
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
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
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
    fn validated_module_exposes_module_version() {
        use super::effective_spirv_version;
        let binary = vec![
            0x07230203, // magic number
            SpirvVersion::new(1, 5).to_word(),
            0,         // generator
            1,         // bound
            0,         // schema
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let module = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .expect("module should validate");
        assert_eq!(module.module_version(), SpirvVersion::new(1, 5));
        assert_eq!(module.header().version(), SpirvVersion::new(1, 5));
        assert_eq!(
            module.effective_version(),
            effective_spirv_version(TargetEnv::Universal1_6, SpirvVersion::new(1, 5))
        );
    }

    #[test]
    fn effective_version_reflects_env_clamp_on_valid_module() {
        use rspirv::{binary::Assemble, dr::Builder};
        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.extension("SPV_KHR_terminate_invocation");
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        let void = builder.type_void();
        let fn_type = builder.type_function(void, std::iter::empty::<u32>());
        builder
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        let words = builder.module().assemble();
        let module = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_0)
            .expect("module should validate with env clamp");
        assert_eq!(module.module_version(), SpirvVersion::new(1, 6));
        assert_eq!(
            module.effective_version(),
            TargetEnv::Vulkan1_0.spirv_version()
        );
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
    fn function_requires_entry_label() {
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            7,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(4, 33), // OpTypeFunction %3 %1 %2
            3,
            1,
            2,
            op(5, 54), // OpFunction %4 None %3 (missing OpLabel)
            1,
            4,
            0,
            3,
            op(3, 55), // OpFunctionParameter %5 %2
            2,
            5,
            op(1, 56), // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::MissingFunctionEntryBlock {
                function: Id::try_from(4).unwrap()
            }
        );
    }

    #[test]
    fn function_declarations_must_precede_definitions() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeVoid",
            "%2 = OpTypeFunction %1",
            "%3 = OpFunction %1 None %2",
            "OpFunctionEnd",
            "%4 = OpFunction %1 None %2",
            "%5 = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let result = MaybeValidModule::Text(&text).validate(TargetEnv::Universal1_6);
        assert!(result.is_ok(), "unexpected validation error: {result:?}");
    }

    #[test]
    fn function_declaration_after_definition_is_rejected() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeVoid",
            "%2 = OpTypeFunction %1",
            "%3 = OpFunction %1 None %2",
            "%4 = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "%5 = OpFunction %1 None %2",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::FunctionDeclarationAfterDefinition {
                function: Id::try_from(5).unwrap()
            }
        );
    }

    #[test]
    fn phi_requires_incoming_block_to_exist() {
        // Function has a single predecessor for %merge; the phi references a missing block id.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            11,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(3, 33), // OpTypeFunction %3 %1
            3,
            1,
            op(3, 1), // OpUndef %4 %2
            2,
            4,
            op(5, 54), // OpFunction %5 None %3
            1,
            5,
            0,
            3,
            op(2, 248), // OpLabel %6
            6,
            op(2, 249), // OpBranch %7
            7,
            op(2, 248), // OpLabel %7
            7,
            op(7, 245), // OpPhi %2 %9 %4 %6 %4 %10 (missing incoming block)
            2,
            9,
            4,
            6,
            4,
            10,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::PhiIncomingBlockMissing {
                function: Id::try_from(5).unwrap(),
                block: Id::try_from(7).unwrap(),
                incoming: Id::try_from(10).unwrap()
            }
        );
    }

    #[test]
    fn phi_incoming_block_must_be_predecessor() {
        // %merge only has predecessor %6, but the phi lists %8 which does not branch to it.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            12,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(3, 33), // OpTypeFunction %3 %1
            3,
            1,
            op(3, 1), // OpUndef %4 %2
            2,
            4,
            op(5, 54), // OpFunction %5 None %3
            1,
            5,
            0,
            3,
            op(2, 248), // OpLabel %6
            6,
            op(2, 249), // OpBranch %7
            7,
            op(2, 248), // OpLabel %8
            8,
            op(1, 253), // OpReturn
            op(2, 248), // OpLabel %7
            7,
            op(5, 245), // OpPhi %2 %9 %4 %8 (incoming block not a predecessor)
            2,
            9,
            4,
            8,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::PhiIncomingNotPredecessor {
                function: Id::try_from(5).unwrap(),
                block: Id::try_from(7).unwrap(),
                incoming: Id::try_from(8).unwrap()
            }
        );
    }

    #[test]
    fn phi_cannot_duplicate_predecessor() {
        // %merge has only predecessor %6, but the phi lists %6 twice.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            11,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(3, 33), // OpTypeFunction %3 %1
            3,
            1,
            op(3, 1), // OpUndef %4 %2
            2,
            4,
            op(5, 54), // OpFunction %5 None %3
            1,
            5,
            0,
            3,
            op(2, 248), // OpLabel %6
            6,
            op(2, 249), // OpBranch %7
            7,
            op(2, 248), // OpLabel %7
            7,
            op(7, 245), // OpPhi %2 %9 %4 %6 %4 %6 (duplicate incoming block)
            2,
            9,
            4,
            6,
            4,
            6,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::PhiDuplicatePredecessor {
                function: Id::try_from(5).unwrap(),
                block: Id::try_from(7).unwrap(),
                incoming: Id::try_from(6).unwrap()
            }
        );
    }

    #[test]
    fn function_type_must_be_type_function() {
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
            op(4, 21), // OpTypeInt %2 32 0 (used incorrectly as function type)
            2,
            32,
            0,
            op(5, 54), // OpFunction %3 None %2 (invalid function type)
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
            ValidationError::InvalidFunctionType {
                function: Id::try_from(3).unwrap(),
                type_id: TypeId::try_from(2).unwrap()
            }
        );
    }

    #[test]
    fn function_return_type_must_match_function_type() {
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
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(3, 33), // OpTypeFunction %3 %2 (return type int)
            3,
            2,
            op(5, 54), // OpFunction %4 None %3 (return type void)
            1,
            4,
            0,
            3,
            op(2, 248), // OpLabel %5
            5,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::FunctionReturnTypeMismatch {
                function: Id::try_from(4).unwrap(),
                result_type: TypeId::try_from(1).unwrap(),
                function_type: TypeId::try_from(2).unwrap(),
            }
        );
    }

    #[test]
    fn function_parameter_count_must_match_function_type() {
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            7,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(4, 33), // OpTypeFunction %3 %1 %2 (expects one parameter)
            3,
            1,
            2,
            op(5, 54), // OpFunction %4 None %3 (no parameters provided)
            1,
            4,
            0,
            3,
            op(2, 248), // OpLabel %5
            5,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::FunctionParameterCountMismatch {
                function: Id::try_from(4).unwrap(),
                expected: 1,
                found: 0,
            }
        );
    }

    #[test]
    fn function_parameter_types_must_match_function_type() {
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            9,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(3, 22), // OpTypeFloat %3 32
            3,
            32,
            op(4, 33), // OpTypeFunction %4 %1 %2 (expects int parameter)
            4,
            1,
            2,
            op(5, 54), // OpFunction %5 None %4
            1,
            5,
            0,
            4,
            op(3, 55), // OpFunctionParameter %3 %6 (float instead of int)
            3,
            6,
            op(2, 248), // OpLabel %7
            7,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::FunctionParameterTypeMismatch {
                function: Id::try_from(5).unwrap(),
                parameter: Id::try_from(6).unwrap(),
                expected: TypeId::try_from(2).unwrap(),
                found: TypeId::try_from(3).unwrap(),
            }
        );
    }

    #[test]
    fn type_function_requires_type_operands() {
        // The function type references a non-type operand as its parameter.
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
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(4, 43), // OpConstant %3 %2 0 (invalid as function parameter type)
            4,
            3,
            0,
            op(4, 33), // OpTypeFunction %4 %1 %3 (param not a type)
            4,
            1,
            3,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::InvalidTypeFunction {
                type_id: TypeId::try_from(4).unwrap()
            }
        );
    }

    #[test]
    fn type_function_parameters_cannot_be_void() {
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
            op(4, 33), // OpTypeFunction %2 %1 %1 (void parameter)
            3,
            1,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::FunctionTypeParameterVoid {
                type_id: TypeId::try_from(3).unwrap(),
                parameter: TypeId::try_from(1).unwrap(),
            }
        );
    }

    #[test]
    fn type_function_return_must_be_type() {
        // Return type id does not reference a type instruction.
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
            op(4, 21), // OpTypeInt %1 32 0
            4,
            2,
            0,
            op(4, 43), // OpConstant %2 %1 0 (used as return type)
            2,
            1,
            0,
            op(3, 33), // OpTypeFunction %3 %2 (invalid return type)
            3,
            2,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::InvalidTypeFunction {
                type_id: TypeId::try_from(3).unwrap()
            }
        );
    }

    #[test]
    fn block_requires_terminator() {
        // A block must end with a terminator instruction.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            4,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
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
            op(2, 248), // OpLabel %4 (no terminator follows)
            4,
            op(1, 56), // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        if let ValidationError::Parse(message) = error {
            assert!(
                message.contains("block without terminator"),
                "unexpected parse error: {message}"
            );
        } else {
            panic!("expected parse error, got {error:?}");
        }
    }

    #[test]
    fn block_cannot_have_instructions_after_terminator() {
        // A terminator must end the block; trailing instructions are invalid.
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
            op(1, 0),   // OpNop (illegal after terminator)
            op(1, 56),  // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        if let ValidationError::Parse(message) = error {
            assert!(
                message.contains("instruction") && message.contains("not inside block"),
                "unexpected parse error: {message}"
            );
        } else {
            panic!("expected parse error, got {error:?}");
        }
    }

    #[test]
    fn branch_requires_existing_target() {
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
            op(2, 249), // OpBranch %5 (undefined target)
            5,
            op(1, 56), // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::MissingBlockTarget {
                function: Id::try_from(3).unwrap(),
                target: Id::try_from(5).unwrap()
            }
        );
    }

    #[test]
    fn switch_requires_existing_targets() {
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            8,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(4, 21), // OpTypeInt %2 32 0
            2,
            32,
            0,
            op(3, 33), // OpTypeFunction %3 %1
            3,
            1,
            op(5, 54), // OpFunction %4 None %3
            1,
            4,
            0,
            3,
            op(2, 248), // OpLabel %5
            5,
            op(5, 128), // OpSwitch %6 %7 0 %7 (both %6 and %7 undefined)
            6,
            7,
            0,
            7,
            op(1, 56), // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        if let ValidationError::Parse(message) = error {
            assert!(
                message.contains("block") && message.contains("terminator"),
                "unexpected parse error: {message}"
            );
        } else {
            panic!("expected parse error, got {error:?}");
        }
    }

    #[test]
    fn entry_block_cannot_have_predecessors() {
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            7,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
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
            op(2, 248), // OpLabel %4 (entry)
            4,
            op(2, 249), // OpBranch %5
            5,
            op(2, 248), // OpLabel %5 (second block)
            5,
            op(2, 249), // OpBranch %4 (back to entry)
            4,
            op(1, 56), // OpFunctionEnd
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::EntryBlockHasPredecessor {
                function: Id::try_from(3).unwrap(),
                entry: Id::try_from(4).unwrap()
            }
        );
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
    fn capability_cannot_follow_memory_model() {
        // Capabilities must be declared before the memory model section.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 17), // OpCapability Shader (misordered after memory model)
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
            op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn extension_cannot_follow_memory_model() {
        // Extensions must appear before the memory model section.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered after memory model)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
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
            op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string"
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
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
    fn capability_cannot_follow_debug_names() {
        // Once debug names begin, capabilities are no longer allowed.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            0x0003_0005, // OpName %1 "x"
            1,
            0x0000_0078,
            op(2, 17), // OpCapability Float64 (misordered after debug section)
            rspirv::spirv::Capability::Float64 as u32,
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
    fn capability_cannot_follow_annotations() {
        // Capabilities must be declared before annotations.
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
            op(2, 73), // OpDecorationGroup %1 (annotations section)
            1,
            op(2, 17), // OpCapability Geometry (misordered after annotations)
            rspirv::spirv::Capability::Geometry as u32,
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
    fn conditional_capability_cannot_follow_debug_names() {
        // OpConditionalCapabilityINTEL (capabilities section) cannot be placed after debug names.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            3,
            0,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 5), // OpName %1 "x" (names/debug2 section)
            1,
            0x0000_0078,
            op(3, 6250), // OpConditionalCapabilityINTEL %1 Shader (misordered after names)
            1,
            rspirv::spirv::Capability::Shader as u32,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn conditional_capability_cannot_follow_annotations() {
        // Conditional capabilities must be declared before annotations.
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
            op(2, 73), // OpDecorationGroup %1 (annotations)
            1,
            op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry (misordered after annotations)
            1,
            rspirv::spirv::Capability::Geometry as u32,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn conditional_capability_cannot_follow_extensions() {
        // Conditional capabilities must appear before the extensions/imports sections.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(2, 10), // OpExtension "e"
            0x0000_0065,
            op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry (misordered after extensions)
            1,
            rspirv::spirv::Capability::Geometry as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn conditional_capability_cannot_follow_ext_inst_import() {
        // Conditional capabilities must precede extension imports.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(3, 11), // OpExtInstImport %1 "G" (imports section)
            1,
            0x0000_0047, // "G"
            op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry (misordered after imports)
            1,
            rspirv::spirv::Capability::Geometry as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn conditional_capability_cannot_follow_memory_model() {
        // Conditional capabilities must be declared before the memory model.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry (misordered after memory model)
            1,
            rspirv::spirv::Capability::Geometry as u32,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn conditional_capability_cannot_appear_inside_functions() {
        // Conditional capabilities belong to the capabilities section, not inside functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(3, 6250), // OpConditionalCapabilityINTEL %1 Shader (inside function -> error)
            1,
            rspirv::spirv::Capability::Shader as u32,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn conditional_capability_cannot_follow_functions() {
        // Conditional capabilities must appear before functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253),  // OpReturn
            op(1, 56),   // OpFunctionEnd
            op(3, 6250), // OpConditionalCapabilityINTEL %1 Shader (after functions -> error)
            1,
            rspirv::spirv::Capability::Shader as u32,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
            }
        );
    }

    #[test]
    fn duplicate_conditional_capability_is_rejected() {
        // Duplicate conditional capabilities should be rejected just like regular capabilities.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            2,          // bound (ids up to 1)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry
            1,
            rspirv::spirv::Capability::Geometry as u32,
            op(3, 6250), // duplicate conditional capability
            1,
            rspirv::spirv::Capability::Geometry as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DuplicateCapability {
                capability: rspirv::spirv::Capability::Geometry
            }
        );
    }

    #[test]
    fn extension_cannot_follow_debug_names() {
        // Extensions must appear before debug/names/annotations sections.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            0x0003_0005, // OpName %1 "x"
            1,
            0x0000_0078,
            op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered after debug)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn conditional_extension_cannot_follow_debug_names() {
        // OpConditionalExtensionINTEL must appear before debug/names/annotations sections.
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
            0x0003_0005, // OpName %1 "x"
            1,
            0x0000_0078,
            op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (misordered after debug)
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn conditional_extension_cannot_appear_inside_functions() {
        // Conditional extensions belong to the extensions section, not inside functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (inside function -> error)
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn conditional_extension_cannot_follow_functions() {
        // Conditional extensions must appear before functions.
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
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253),  // OpReturn
            op(1, 56),   // OpFunctionEnd
            op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (after functions -> error)
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        ];

        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn conditional_extension_cannot_follow_annotations() {
        // OpConditionalExtensionINTEL (extensions section) must not appear after annotations.
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
            op(2, 73), // OpDecorationGroup %1 (annotations)
            1,
            op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (misordered)
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn conditional_extension_cannot_follow_ext_inst_import() {
        // Conditional extensions must precede imported instruction sets.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            3,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(6, 11), // OpExtInstImport %1 "GLSL.std.450"
            1,
            0x4c53_4c47, // "GLSL"
            0x2e74_7364, // ".std"
            0x2e30_3534, // ".450"
            0,
            op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (misordered after import)
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn conditional_extension_cannot_follow_memory_model() {
        // Conditional extensions must appear before the memory model.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (misordered after memory model)
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
            EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
            }
        );
    }

    #[test]
    fn extension_cannot_follow_annotations() {
        // Extensions must appear before annotations.
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
            op(2, 73), // OpDecorationGroup %1 (annotations)
            1,
            op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered after annotations)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn extension_cannot_follow_ext_inst_import() {
        // Extensions must appear before imports; a later extension is out of order.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 11), // OpExtInstImport %1 "GLSL.std.450"
            1,
            0x004c_5347, // "GLS"
            op(8, 10),   // OpExtension "SPV_GOOGLE_decorate_string" (misordered after imports)
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Extension
            }
        );
    }

    #[test]
    fn names_cannot_follow_annotations() {
        // Names/debug instructions must precede annotations.
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
            op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
            1,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(3, 5), // OpName %1 "x" (misordered after annotations)
            1,
            0x0000_0078, // "x"
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Decorate
            }
        );
    }

    #[test]
    fn capability_cannot_follow_ext_inst_import() {
        // Capabilities must precede extension imports.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(3, 11), // OpExtInstImport %1 "GLSL.std.450" (imports section)
            1,
            0x004c_5347,
            op(2, 17), // OpCapability Geometry (misordered after imports)
            rspirv::spirv::Capability::Geometry as u32,
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
    fn ext_inst_import_cannot_follow_debug_names() {
        // OpExtInstImport belongs to the extensions/imports section and cannot follow debug names.
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
            op(3, 5), // OpName %1 "x" (names/debug2 section)
            1,
            0x0000_0078,
            op(3, 11), // OpExtInstImport %1 "GLSL.std.450" (misordered after names)
            1,
            0x004c_5347, // "GLS"
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
    fn ext_inst_import_cannot_follow_annotations() {
        // OpExtInstImport must precede the annotations section.
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
            op(2, 73), // OpDecorationGroup %1 (annotations)
            1,
            op(3, 11), // OpExtInstImport %1 "GLSL.std.450" (misordered after annotations)
            1,
            0x004c_5347, // "GLS"
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
    fn debug_instructions_cannot_follow_annotations() {
        // OpSource (debug) must not appear after the annotations section.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
            1,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(3, 3), // OpSource Unknown 0 (misordered after annotations)
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
    fn string_cannot_follow_annotations() {
        // OpString (debug) must not appear after the annotations section.
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
            op(2, 73), // OpDecorationGroup %1 (annotation section)
            1,
            op(3, 7), // OpString %2 "s" (misordered after annotations)
            2,
            0x0000_0073,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::String
            }
        );
    }

    #[test]
    fn string_cannot_follow_names() {
        // Debug1 instructions (OpString) must precede the Names section.
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
            op(3, 5), // OpName %1 "x" (names section)
            1,
            0x0000_0078,
            op(3, 7), // OpString %2 "s" (misordered after names section)
            2,
            0x0000_0073,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::String
            }
        );
    }

    #[test]
    fn source_extension_cannot_follow_annotations() {
        // OpSourceExtension (debug) must not appear after the annotations section.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
            1,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(2, 4), // OpSourceExtension "ext" (misordered after annotations)
            0x0074_7865,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::SourceExtension
            }
        );
    }

    #[test]
    fn source_continued_cannot_follow_annotations() {
        // OpSourceContinued (debug) must not appear after the annotations section.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            2,
            0,
            op(2, 17), // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 3), // OpSource Unknown 0 (establish debug section)
            0,
            0,
            op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
            1,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(2, 2), // OpSourceContinued "c" (misordered after annotations)
            0x0000_0063,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::SourceContinued
            }
        );
    }

    #[test]
    fn module_processed_must_precede_annotations() {
        // OpModuleProcessed belongs to the debug section and must precede annotations.
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
            op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
            1,
            rspirv::spirv::Decoration::RelaxedPrecision as u32,
            op(2, 330),  // OpModuleProcessed "tag" (misordered after annotations)
            0x0067_6174, // "tag"
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::Decorate
            }
        );
    }

    #[test]
    fn module_processed_must_follow_names() {
        // OpModuleProcessed (Debug3) must appear after the Names section.
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
            op(2, 330),  // OpModuleProcessed "tag" (appearing before names)
            0x0067_6174, // "tag"
            op(3, 5),    // OpName %1 "x" (out of order after ModuleProcessed)
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
    fn module_processed_must_precede_types_and_globals() {
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
            op(2, 330),  // OpModuleProcessed "tag" (too late after types/globals)
            0x0067_6174, // "tag"
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ModuleProcessed
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
    fn opencl_rejects_nv_vendor_extension() {
        // OpenCL should reject NV vendor extensions.
        let ext_words = [
            1599492179, 1834964558, 1600680805, 1684105331, 29285, // "SPV_NV_mesh_shader\0"
        ];
        let binary = [
            0x0723_0203, // magic
            0x0001_0000, // version
            0,           // generator
            2,           // bound
            0,           // schema
            0x0006_000a, // OpExtension, word count 6
            ext_words[0],
            ext_words[1],
            ext_words[2],
            ext_words[3],
            ext_words[4],
            0x0003_000e, // OpMemoryModel Logical OpenCL
            0,
            2,
        ];
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::OpenCl2_2)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_mesh_shader"),
                env: TargetEnv::OpenCl2_2
            }
        );
    }

    #[test]
    fn universal_rejects_nv_vendor_extension() {
        let ext_words = [
            1599492179, 1834964558, 1600680805, 1684105331, 29285, // "SPV_NV_mesh_shader\0"
        ];
        let binary = [
            0x0723_0203, // magic
            0x0001_0000, // version
            0,           // generator
            2,           // bound
            0,           // schema
            0x0006_000a, // OpExtension, word count 6
            ext_words[0],
            ext_words[1],
            ext_words[2],
            ext_words[3],
            ext_words[4],
            0x0003_000e, // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_mesh_shader"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn universal_rejects_nv_shader_invocation_reorder() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_NV_shader_invocation_reorder\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text.as_str().validate(TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_shader_invocation_reorder"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn universal_rejects_nv_cluster_acceleration_structure() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_NV_cluster_acceleration_structure\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text.as_str().validate(TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_cluster_acceleration_structure"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn universal_rejects_qcom_image_processing2() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_QCOM_image_processing2\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text.as_str().validate(TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_QCOM_image_processing2"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn vulkan_accepts_nv_vendor_extension() {
        let ext_words = [
            1599492179, 1834964558, 1600680805, 1684105331, 29285, // "SPV_NV_mesh_shader\0"
        ];
        let binary = [
            0x0723_0203, // magic
            0x0001_0000, // version
            0,           // generator
            2,           // bound
            0,           // schema
            0x0002_0011, // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            0x0006_000a, // OpExtension, word count 6
            ext_words[0],
            ext_words[1],
            ext_words[2],
            ext_words[3],
            ext_words[4],
            0x0003_000e, // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let validated = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Vulkan1_2)
            .expect("NV vendor extension should be allowed for Vulkan");
        assert_eq!(validated.env(), TargetEnv::Vulkan1_2);
    }

    #[test]
    fn vulkan_accepts_google_vendor_extension() {
        let ext_words = [
            1599492179, 1196379975, 1683965260, 1919902565, 1600484449, 1769108595, 26478,
        ]; // "SPV_GOOGLE_decorate_string\0"
        let binary = [
            0x0723_0203, // magic
            0x0001_0000, // version
            0,           // generator
            2,           // bound
            0,           // schema
            0x0002_0011, // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            0x0008_000a, // OpExtension, word count 8
            ext_words[0],
            ext_words[1],
            ext_words[2],
            ext_words[3],
            ext_words[4],
            ext_words[5],
            ext_words[6],
            0x0003_000e, // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let validated = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Vulkan1_2)
            .expect("Google vendor extension should be allowed for Vulkan");
        assert_eq!(validated.env(), TargetEnv::Vulkan1_2);
    }

    #[test]
    fn vulkan_rejects_intel_vendor_extension() {
        let intel_function_variants_ext = [
            1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
        ];
        let binary = [
            0x0723_0203, // magic
            0x0001_0000, // version
            0,           // generator
            2,           // bound
            0,           // schema
            0x0002_0011, // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            0x0008_000a, // OpExtension, word count 8
            intel_function_variants_ext[0],
            intel_function_variants_ext[1],
            intel_function_variants_ext[2],
            intel_function_variants_ext[3],
            intel_function_variants_ext[4],
            intel_function_variants_ext[5],
            intel_function_variants_ext[6],
            0x0003_000e, // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = MaybeValidModule::Binary(&binary)
            .validate(TargetEnv::Vulkan1_2)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_INTEL_function_variants"),
                env: TargetEnv::Vulkan1_2
            }
        );
    }

    #[test]
    fn qcom_extension_requires_vulkan_environment() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_QCOM_image_processing\"",
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
            .expect_err("QCOM extension should be disallowed outside Vulkan");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_QCOM_image_processing"),
                env: TargetEnv::Universal1_6
            }
        );

        let validated = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("QCOM extension should be accepted for Vulkan");
        assert_eq!(validated.env(), TargetEnv::Vulkan1_2);
    }

    #[test]
    fn qcom_image_processing_requires_spirv_1_4() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_QCOM_image_processing\"",
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
            .expect_err("SPIR-V 1.4 is required for QCOM image processing");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_QCOM_image_processing"),
                required_version: SpirvVersion::new(1, 4),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.4+");
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
    fn vulkan_memory_model_extension_requires_spirv_1_3() {
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
            .expect_err("SPIR-V 1.3 is required for Vulkan memory model extension");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
                required_version: SpirvVersion::new(1, 3),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        // A newer environment should accept the extension.
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.4+");
    }

    #[test]
    fn qcom_cooperative_matrix_conversion_requires_spirv_1_3() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_QCOM_cooperative_matrix_conversion\"",
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
            .expect_err("SPIR-V 1.3 is required for QCOM cooperative matrix conversion");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_QCOM_cooperative_matrix_conversion"),
                required_version: SpirvVersion::new(1, 3),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.3+");
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
    fn capability_requires_declared_vendor_extension() {
        let text = [
            "OpCapability RayTracingNV",
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
            .expect_err("Vendor capability without required extension should be rejected");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::RayTracingNV,
                required_extension: "SPV_NV_ray_tracing".to_string()
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability RayTracingNV",
            "OpExtension \"SPV_NV_ray_tracing\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("Vendor capability should be allowed with its extension declared");
    }

    #[test]
    fn vendor_capability_requiring_disallowed_extension_reports_env_error() {
        let text = [
            "OpCapability RayTracingNV",
            "OpExtension \"SPV_NV_ray_tracing\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "Capability should be rejected when its required extension is disallowed",
            );
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_NV_ray_tracing"),
                    env
                }
            );
        }
    }

    #[test]
    fn cooperative_matrix_nv_capability_rejected_outside_vulkan() {
        let text = [
            "OpCapability Shader",
            "OpCapability CooperativeMatrixNV",
            "OpExtension \"SPV_NV_cooperative_matrix\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "CooperativeMatrixNV should be rejected when its extension is disallowed",
            );
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_NV_cooperative_matrix"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("CooperativeMatrixNV should be accepted for Vulkan targets");
    }

    #[test]
    fn tile_shading_capability_rejected_outside_vulkan() {
        let text = [
            "OpCapability Shader",
            "OpCapability TileShadingQCOM",
            "OpExtension \"SPV_QCOM_tile_shading\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("TileShadingQCOM should be rejected when its extension is disallowed");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_QCOM_tile_shading"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_4)
            .expect("TileShadingQCOM should be accepted for Vulkan targets");
    }

    #[test]
    fn ray_tracing_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
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

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("RayTracingKHR should be rejected when its extension is disallowed");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_KHR_ray_tracing"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("RayTracingKHR should be accepted for Vulkan targets");
    }

    #[test]
    fn mesh_shading_nv_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability MeshShadingNV",
            "OpExtension \"SPV_NV_mesh_shader\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("MeshShadingNV should be rejected when its extension is disallowed");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_NV_mesh_shader"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("MeshShadingNV should be accepted for Vulkan targets");
    }

    #[test]
    fn mesh_shading_ext_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability MeshShadingEXT",
            "OpExtension \"SPV_EXT_mesh_shader\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("MeshShadingEXT should be rejected when its extension is disallowed");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_EXT_mesh_shader"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("MeshShadingEXT should be accepted for Vulkan targets");
    }

    #[test]
    fn cooperative_matrix_khr_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability CooperativeMatrixKHR",
            "OpExtension \"SPV_KHR_cooperative_matrix\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "CooperativeMatrixKHR should be rejected when its extension is disallowed",
            );
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_KHR_cooperative_matrix"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("CooperativeMatrixKHR should be accepted for Vulkan targets");
    }

    #[test]
    fn ray_tracing_motion_blur_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingNV",
            "OpCapability RayTracingMotionBlurNV",
            "OpExtension \"SPV_NV_ray_tracing\"",
            "OpExtension \"SPV_NV_ray_tracing_motion_blur\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "RayTracingMotionBlurNV should be rejected when its extension is disallowed",
            );
            match error {
                ValidationError::DisallowedExtension {
                    extension,
                    env: actual_env,
                } => {
                    assert_eq!(actual_env, env);
                    assert!(
                        extension == ExtensionName::from("SPV_NV_ray_tracing_motion_blur")
                            || extension == ExtensionName::from("SPV_NV_ray_tracing"),
                        "unexpected extension blocked: {extension:?}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("RayTracingMotionBlurNV should be accepted for Vulkan targets");
    }

    #[test]
    fn ray_tracing_displacement_micromap_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability RayTracingNV",
            "OpCapability RayTracingDisplacementMicromapNV",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_NV_ray_tracing\"",
            "OpExtension \"SPV_NV_displacement_micromap\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "RayTracingDisplacementMicromapNV should be rejected when its extension is disallowed",
            );
            match error {
                ValidationError::DisallowedExtension {
                    extension,
                    env: actual_env,
                } => {
                    assert_eq!(actual_env, env);
                    assert!(
                        extension == ExtensionName::from("SPV_NV_displacement_micromap")
                            || extension == ExtensionName::from("SPV_NV_ray_tracing")
                            || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                        "unexpected extension blocked: {extension:?}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("RayTracingDisplacementMicromapNV should be accepted for Vulkan targets");
    }

    #[test]
    fn ray_tracing_linear_swept_spheres_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability RayTracingLinearSweptSpheresGeometryNV",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_NV_linear_swept_spheres\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "RayTracingLinearSweptSpheresGeometryNV should be rejected when its extension is disallowed",
            );
            match error {
                ValidationError::DisallowedExtension {
                    extension,
                    env: actual_env,
                } => {
                    assert_eq!(actual_env, env);
                    assert!(
                        extension == ExtensionName::from("SPV_NV_linear_swept_spheres")
                            || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                        "unexpected extension blocked: {extension:?}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("RayTracingLinearSweptSpheresGeometryNV should be accepted for Vulkan targets");
    }

    #[test]
    fn ray_tracing_opacity_micromap_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability RayTracingOpacityMicromapEXT",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_EXT_opacity_micromap\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "RayTracingOpacityMicromapEXT should be rejected when its extension is disallowed",
            );
            match error {
                ValidationError::DisallowedExtension {
                    extension,
                    env: actual_env,
                } => {
                    assert_eq!(actual_env, env);
                    assert!(
                        extension == ExtensionName::from("SPV_EXT_opacity_micromap")
                            || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                        "unexpected extension blocked: {extension:?}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("RayTracingOpacityMicromapEXT should be accepted for Vulkan targets");
    }

    #[test]
    fn shader_invocation_reorder_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability ShaderInvocationReorderNV",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_NV_shader_invocation_reorder\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "ShaderInvocationReorderNV should be rejected when its extension is disallowed",
            );
            match error {
                ValidationError::DisallowedExtension {
                    extension,
                    env: actual_env,
                } => {
                    assert_eq!(actual_env, env);
                    assert!(
                        extension == ExtensionName::from("SPV_NV_shader_invocation_reorder")
                            || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                        "unexpected extension blocked: {extension:?}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("ShaderInvocationReorderNV should be accepted for Vulkan targets");
    }

    #[test]
    fn cluster_acceleration_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability RayTracingClusterAccelerationStructureNV",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_NV_cluster_acceleration_structure\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "RayTracingClusterAccelerationStructureNV should be rejected when its extension is disallowed",
            );
            match error {
                ValidationError::DisallowedExtension {
                    extension,
                    env: actual_env,
                } => {
                    assert_eq!(actual_env, env);
                    assert!(
                        extension == ExtensionName::from("SPV_NV_cluster_acceleration_structure")
                            || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                        "unexpected extension blocked: {extension:?}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        text.as_str().validate(TargetEnv::Vulkan1_2).expect(
            "RayTracingClusterAccelerationStructureNV should be accepted for Vulkan targets",
        );
    }

    #[test]
    fn shader_sm_builtins_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability ShaderSMBuiltinsNV",
            "OpExtension \"SPV_NV_shader_sm_builtins\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "ShaderSMBuiltinsNV should be rejected when its extension is disallowed",
            );
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_NV_shader_sm_builtins"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("ShaderSMBuiltinsNV should be accepted for Vulkan targets");
    }

    #[test]
    fn fragment_shader_interlock_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability FragmentShaderPixelInterlockEXT",
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

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "FragmentShaderPixelInterlockEXT should be rejected when its extension is disallowed",
            );
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_EXT_fragment_shader_interlock"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("FragmentShaderPixelInterlockEXT should be accepted for Vulkan targets");
    }

    #[test]
    fn image_footprint_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability ImageFootprintNV",
            "OpExtension \"SPV_NV_shader_image_footprint\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("ImageFootprintNV should be rejected when its extension is disallowed");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_NV_shader_image_footprint"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("ImageFootprintNV should be accepted for Vulkan targets");
    }

    #[test]
    fn shader_atomic_float_add_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability AtomicFloat32AddEXT",
            "OpExtension \"SPV_EXT_shader_atomic_float_add\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "AtomicFloat32AddEXT should be rejected when its extension is disallowed",
            );
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_EXT_shader_atomic_float_add"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("AtomicFloat32AddEXT should be accepted for Vulkan targets");
    }

    #[test]
    fn fragment_shading_rate_capability_rejected_outside_vulkan_even_with_extension() {
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

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "FragmentShadingRateKHR should be rejected when its extension is disallowed",
            );
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_KHR_fragment_shading_rate"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("FragmentShadingRateKHR should be accepted for Vulkan targets");
    }

    #[test]
    fn fragment_invocation_density_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability FragmentDensityEXT",
            "OpExtension \"SPV_EXT_fragment_invocation_density\"",
            "OpExtension \"SPV_NV_shading_rate\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text.as_str().validate(env).expect_err(
                "FragmentDensityEXT should be rejected when its extension is disallowed",
            );
            match error {
                ValidationError::DisallowedExtension {
                    extension,
                    env: actual_env,
                } => {
                    assert_eq!(actual_env, env);
                    assert!(
                        extension == ExtensionName::from("SPV_EXT_fragment_invocation_density")
                            || extension == ExtensionName::from("SPV_NV_shading_rate"),
                        "unexpected extension in disallowance: {extension:?}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("FragmentDensityEXT should be accepted for Vulkan targets");
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
    fn nv_shader_invocation_reorder_requires_spirv_1_4() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_NV_shader_invocation_reorder\"",
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
            .validate(TargetEnv::Vulkan1_1)
            .expect_err("SPIR-V 1.4 is required for NV shader invocation reorder");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_NV_shader_invocation_reorder"),
                required_version: SpirvVersion::new(1, 4),
                target_version: TargetEnv::Vulkan1_1.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.3+");
    }

    #[test]
    fn ext_shader_invocation_reorder_requires_spirv_1_4() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_EXT_shader_invocation_reorder\"",
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
            .validate(TargetEnv::Vulkan1_1)
            .expect_err("SPIR-V 1.4 is required for EXT shader invocation reorder");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_EXT_shader_invocation_reorder"),
                required_version: SpirvVersion::new(1, 4),
                target_version: TargetEnv::Vulkan1_1.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.4+");
    }

    #[test]
    fn extension_version_check_respects_module_version() {
        use rspirv::{binary::Assemble, dr::Builder};
        let mut builder = Builder::new();
        builder.set_version(1, 0);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.extension("SPV_KHR_vulkan_memory_model");
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        let void = builder.type_void();
        let fn_type = builder.type_function(void, std::iter::empty::<u32>());
        builder
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("module version 1.0 cannot use Vulkan memory model");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
                required_version: SpirvVersion::new(1, 3),
                target_version: SpirvVersion::new(1, 0),
            }
        );
    }

    #[test]
    fn extension_version_clamps_to_env_when_module_is_newer() {
        use rspirv::{binary::Assemble, dr::Builder};
        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.extension("SPV_EXT_fragment_shader_interlock");
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        let void = builder.type_void();
        let fn_type = builder.type_function(void, std::iter::empty::<u32>());
        builder
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("env version should clamp module version when gating extension");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_EXT_fragment_shader_interlock"),
                required_version: SpirvVersion::new(1, 4),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );
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
    fn fragment_invocation_density_extension_requires_spirv_1_5() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_EXT_fragment_invocation_density\"",
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
            .expect_err("fragment invocation density requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_EXT_fragment_invocation_density"),
                required_version: SpirvVersion::new(1, 5),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.5+");
    }

    #[test]
    fn physical_storage_buffer_extension_requires_spirv_1_4() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_physical_storage_buffer\"",
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
            .expect_err("physical storage buffer requires SPIR-V 1.4");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_KHR_physical_storage_buffer"),
                required_version: SpirvVersion::new(1, 4),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.4+");
    }

    #[test]
    fn storage_buffer_storage_class_extension_requires_spirv_1_3() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
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
            .validate(TargetEnv::Universal1_2)
            .expect_err("storage buffer storage class requires SPIR-V 1.3");
        // Either version gating or environment rejection is acceptable as long as the extension is disallowed.
        match error {
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(
                    extension,
                    ExtensionName::from("SPV_KHR_storage_buffer_storage_class")
                );
                assert_eq!(required_version, SpirvVersion::new(1, 3));
                assert_eq!(target_version, SpirvVersion::new(1, 2));
            }
            ValidationError::DisallowedExtension { extension, env } => {
                assert_eq!(
                    extension,
                    ExtensionName::from("SPV_KHR_storage_buffer_storage_class")
                );
                assert_eq!(env, TargetEnv::Universal1_2);
            }
            other => panic!("unexpected error: {other:?}"),
        }

        text.as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("extension should be accepted with SPIR-V 1.3+");
    }

    #[test]
    fn variable_pointers_extension_requires_spirv_1_3() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_variable_pointers\"",
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
            .validate(TargetEnv::Universal1_2)
            .expect_err("variable pointers requires SPIR-V 1.3");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_KHR_variable_pointers"),
                required_version: SpirvVersion::new(1, 3),
                target_version: SpirvVersion::new(1, 2),
            }
        );

        text.as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("extension should be accepted with SPIR-V 1.3+");
    }

    #[test]
    fn shader_clock_extension_requires_spirv_1_3() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_shader_clock\"",
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
            .expect_err("shader clock requires SPIR-V 1.3");
        match error {
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(extension, ExtensionName::from("SPV_KHR_shader_clock"));
                assert_eq!(required_version, SpirvVersion::new(1, 3));
                assert_eq!(target_version, TargetEnv::Vulkan1_0.spirv_version());
            }
            other => panic!("unexpected error: {other:?}"),
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.3+");
    }

    #[test]
    fn device_group_extension_requires_spirv_1_3() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_device_group\"",
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
            .expect_err("device group requires SPIR-V 1.3");
        assert_eq!(
            error,
            ValidationError::ExtensionRequiresSpirvVersion {
                extension: ExtensionName::from("SPV_KHR_device_group"),
                required_version: SpirvVersion::new(1, 3),
                target_version: TargetEnv::Vulkan1_0.spirv_version(),
            }
        );

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("extension should be accepted with SPIR-V 1.3+");
    }

    #[test]
    fn maximal_reconvergence_extension_rejected_outside_vulkan() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_maximal_reconvergence\"",
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
            .expect_err("maximal reconvergence is Vulkan-only");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_maximal_reconvergence"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn ray_cull_mask_extension_rejected_outside_vulkan() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_ray_cull_mask\"",
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
            .expect_err("ray cull mask is Vulkan-only");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_ray_cull_mask"),
                env: TargetEnv::Universal1_6
            }
        );
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
            "OpCapability Sampled1D",
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
        match error {
            ValidationError::MissingRequiredCapability {
                required_capability,
                capability,
            } => {
                assert_eq!(required_capability, rspirv::spirv::Capability::ImageBasic);
                assert!(
                    capability == rspirv::spirv::Capability::Image1D
                        || capability == rspirv::spirv::Capability::Sampled1D
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }

        let text_with_basic = [
            "OpCapability Kernel",
            "OpCapability Addresses",
            "OpCapability ImageBasic",
            "OpCapability Sampled1D",
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
    fn valid_module_cache_accounts_for_options() {
        use crate::validation::ValidationOptions;

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
            .validate_words_with_options(
                &binary,
                TargetEnv::Universal1_6,
                ValidationOptions::default(),
            )
            .expect("first validation");

        let mut relaxed = ValidationOptions {
            relax_struct_store: true,
            ..ValidationOptions::default()
        };
        relaxed.limits.insert(7, 42);

        let second = cache
            .validate_words_with_options(&binary, TargetEnv::Universal1_6, relaxed)
            .expect("validation with options");

        assert_ne!(
            Arc::as_ptr(&first),
            Arc::as_ptr(&second),
            "options should participate in the cache key"
        );
    }

    #[test]
    fn global_variable_limit_enforced() {
        use crate::validation::{ValidationOptions, LIMIT_MAX_GLOBAL_VARIABLES};

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%ptr = OpTypePointer Uniform %void",
            "%g0 = OpVariable %ptr Uniform",
            "%g1 = OpVariable %ptr Uniform",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let mut options = ValidationOptions::default();
        options.limits.insert(LIMIT_MAX_GLOBAL_VARIABLES, 1);

        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect_err("global variable limit should be enforced");
        assert_eq!(
            err,
            ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_GLOBAL_VARIABLES,
                limit: 1,
                found: 2
            }
        );
    }

    #[test]
    fn local_variable_limit_enforced() {
        use crate::validation::{ValidationOptions, LIMIT_MAX_LOCAL_VARIABLES};

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%ptr = OpTypePointer Function %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "%l0 = OpVariable %ptr Function",
            "%l1 = OpVariable %ptr Function",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let mut options = ValidationOptions::default();
        options.limits.insert(LIMIT_MAX_LOCAL_VARIABLES, 1);

        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect_err("local variable limit should be enforced");
        assert_eq!(
            err,
            ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_LOCAL_VARIABLES,
                limit: 1,
                found: 2
            }
        );
    }

    #[test]
    fn control_flow_depth_limit_enforced() {
        use crate::validation::{ValidationOptions, LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH};

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%bool = OpTypeBool",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpSelectionMerge %merge None",
            "OpBranchConditional %bool %then %merge",
            "%then = OpLabel",
            "OpReturn",
            "%merge = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let mut options = ValidationOptions::default();
        options
            .limits
            .insert(LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH, 0);

        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect_err("control flow nesting limit should be enforced");
        assert_eq!(
            err,
            ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH,
                limit: 0,
                found: 1
            }
        );
    }

    #[test]
    fn access_chain_index_limit_enforced() {
        use crate::validation::{ValidationOptions, LIMIT_MAX_ACCESS_CHAIN_INDEXES};

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%u32 = OpTypeInt 32 0",
            "%ptr = OpTypePointer Function %u32",
            "%zero = OpConstant %u32 0",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "%var = OpVariable %ptr Function",
            "%ac = OpAccessChain %ptr %var %zero %zero",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let mut options = ValidationOptions::default();
        options.limits.insert(LIMIT_MAX_ACCESS_CHAIN_INDEXES, 1);

        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect_err("access chain index limit should be enforced");
        assert_eq!(
            err,
            ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_ACCESS_CHAIN_INDEXES,
                limit: 1,
                found: 2
            }
        );
    }

    #[test]
    fn struct_depth_limit_enforced() {
        use crate::validation::{ValidationOptions, LIMIT_MAX_STRUCT_DEPTH};

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%i32 = OpTypeInt 32 0",
            "%inner = OpTypeStruct %i32",
            "%outer = OpTypeStruct %inner",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let mut options = ValidationOptions::default();
        options.limits.insert(LIMIT_MAX_STRUCT_DEPTH, 1);

        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect_err("struct depth limit should be enforced");
        assert_eq!(
            err,
            ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_STRUCT_DEPTH,
                limit: 1,
                found: 2
            }
        );
    }

    #[test]
    fn friendly_names_table_captures_op_name() {
        use crate::validation::ValidationOptions;

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpName %main \"friendly\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let options = ValidationOptions {
            use_friendly_names: true,
            ..ValidationOptions::default()
        };

        let module = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("validation should succeed");
        let names = module
            .friendly_names()
            .expect("friendly names should be present");
        let function_id = module
            .module()
            .functions
            .first()
            .and_then(|f| f.def.as_ref())
            .and_then(|inst| inst.result_id)
            .expect("function should have a result id");
        assert_eq!(names.id(function_id), Some("friendly"));
    }

    #[test]
    fn friendly_names_table_captures_member_name() {
        use crate::validation::ValidationOptions;

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpName %S \"Struct\"",
            "OpMemberName %S 1 \"member\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%S = OpTypeStruct %uint %uint",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let options = ValidationOptions {
            use_friendly_names: true,
            ..ValidationOptions::default()
        };

        let module = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("validation should succeed");
        let names = module
            .friendly_names()
            .expect("friendly names should be present");
        let struct_id = module
            .module()
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::TypeStruct)
            .and_then(|inst| inst.result_id)
            .expect("struct should have a result id");
        assert_eq!(names.id(struct_id), Some("Struct"));
        assert_eq!(names.member(struct_id, MemberIndex(1)), Some("member"));
    }

    #[test]
    fn localsizeid_disallowed_without_option_in_older_vulkan() {
        use crate::validation::ValidationOptions;
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{ExecutionMode, ExecutionModel, FunctionControl};

        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let fn_ty = builder.type_function(void, []);
        let uint = builder.type_int(32, 0);
        let local_size = builder.constant_bit32(uint, 1);

        let entry_point = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        builder.entry_point(ExecutionModel::GLCompute, entry_point, "main", []);
        builder.execution_mode_id(
            entry_point,
            ExecutionMode::LocalSizeId,
            [local_size, local_size, local_size],
        );

        let words = builder.module().assemble();
        let err = words
            .as_slice()
            .validate_with_options(TargetEnv::Vulkan1_2, ValidationOptions::default())
            .expect_err("LocalSizeId should be disallowed without the option");
        assert_eq!(
            err,
            ValidationError::LocalSizeIdNotAllowed {
                env: TargetEnv::Vulkan1_2
            }
        );
    }

    #[test]
    fn localsizeid_allowed_with_option() {
        use crate::validation::ValidationOptions;
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{ExecutionMode, ExecutionModel, FunctionControl};

        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let fn_ty = builder.type_function(void, []);
        let uint = builder.type_int(32, 0);
        let local_size = builder.constant_bit32(uint, 1);

        let entry_point = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        builder.entry_point(ExecutionModel::GLCompute, entry_point, "main", []);
        builder.execution_mode_id(
            entry_point,
            ExecutionMode::LocalSizeId,
            [local_size, local_size, local_size],
        );

        let words = builder.module().assemble();
        let options = ValidationOptions {
            allow_localsizeid: true,
            ..ValidationOptions::default()
        };
        words
            .as_slice()
            .validate_with_options(TargetEnv::Vulkan1_1, options)
            .expect("LocalSizeId should be allowed when option is enabled");
    }

    #[test]
    fn offset_texture_operand_disallowed_by_default_in_vulkan() {
        use crate::validation::ValidationOptions;
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{
            AddressingModel, Capability, Dim, ExecutionModel, FunctionControl, ImageFormat,
            ImageOperands, MemoryModel,
        };

        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(Capability::Shader);
        builder.capability(Capability::ImageGatherExtended);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void = builder.type_void();
        let float = builder.type_float(32, None);
        let v2float = builder.type_vector(float, 2);
        let i32 = builder.type_int(32, 1);
        let v2i = builder.type_vector(i32, 2);
        let float_zero = builder.constant_bit32(float, 0.0f32.to_bits());
        let int_zero = builder.constant_bit32(i32, 0);
        let zero_offset = builder.constant_composite(v2i, [int_zero, int_zero]);
        let image = builder.type_image(float, Dim::Dim2D, 0, 0, 0, 1, ImageFormat::Unknown, None);
        let sampled_image = builder.type_sampled_image(image);
        let fn_ty = builder.type_function(void, [sampled_image, v2float]);

        let entry = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .unwrap();
        let image_param = builder.function_parameter(sampled_image).unwrap();
        let coord_param = builder.function_parameter(v2float).unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_sample_explicit_lod(
                v2float,
                None,
                image_param,
                coord_param,
                ImageOperands::LOD | ImageOperands::OFFSET,
                [
                    rspirv::dr::Operand::IdRef(float_zero),
                    rspirv::dr::Operand::IdRef(zero_offset),
                ],
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        builder.entry_point(ExecutionModel::Fragment, entry, "main", []);

        let binary = builder.module().assemble();
        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Vulkan1_2, ValidationOptions::default())
            .expect_err("Offset operand should be restricted to gather ops in Vulkan by default");
        assert_eq!(
            err,
            ValidationError::OffsetTextureOperandDisallowed {
                opcode: rspirv::spirv::Op::ImageSampleExplicitLod
            }
        );
    }

    #[test]
    fn offset_texture_operand_allowed_with_option() {
        use crate::validation::ValidationOptions;
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{
            AddressingModel, Capability, Dim, ExecutionModel, FunctionControl, ImageFormat,
            ImageOperands, MemoryModel,
        };

        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(Capability::Shader);
        builder.capability(Capability::ImageGatherExtended);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void = builder.type_void();
        let float = builder.type_float(32, None);
        let v2float = builder.type_vector(float, 2);
        let i32 = builder.type_int(32, 1);
        let v2i = builder.type_vector(i32, 2);
        let float_zero = builder.constant_bit32(float, 0.0f32.to_bits());
        let int_zero = builder.constant_bit32(i32, 0);
        let zero_offset = builder.constant_composite(v2i, [int_zero, int_zero]);
        let image = builder.type_image(float, Dim::Dim2D, 0, 0, 0, 1, ImageFormat::Unknown, None);
        let sampled_image = builder.type_sampled_image(image);
        let fn_ty = builder.type_function(void, [sampled_image, v2float]);

        let entry = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .unwrap();
        let image_param = builder.function_parameter(sampled_image).unwrap();
        let coord_param = builder.function_parameter(v2float).unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_sample_explicit_lod(
                v2float,
                None,
                image_param,
                coord_param,
                ImageOperands::LOD | ImageOperands::OFFSET,
                [
                    rspirv::dr::Operand::IdRef(float_zero),
                    rspirv::dr::Operand::IdRef(zero_offset),
                ],
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        builder.entry_point(ExecutionModel::Fragment, entry, "main", []);

        let binary = builder.module().assemble();
        let options = ValidationOptions {
            allow_offset_texture_operand: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Vulkan1_2, options)
            .expect("Offset operand should be allowed when option is enabled");
    }

    #[test]
    fn offset_texture_operand_allowed_before_hlsl_legalization() {
        use crate::validation::ValidationOptions;
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{
            AddressingModel, Capability, Dim, ExecutionModel, FunctionControl, ImageFormat,
            ImageOperands, MemoryModel,
        };

        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(Capability::Shader);
        builder.capability(Capability::ImageGatherExtended);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void = builder.type_void();
        let float = builder.type_float(32, None);
        let v2float = builder.type_vector(float, 2);
        let i32 = builder.type_int(32, 1);
        let v2i = builder.type_vector(i32, 2);
        let float_zero = builder.constant_bit32(float, 0.0f32.to_bits());
        let int_zero = builder.constant_bit32(i32, 0);
        let zero_offset = builder.constant_composite(v2i, [int_zero, int_zero]);
        let image = builder.type_image(float, Dim::Dim2D, 0, 0, 0, 1, ImageFormat::Unknown, None);
        let sampled_image = builder.type_sampled_image(image);
        let fn_ty = builder.type_function(void, [sampled_image, v2float]);

        let entry = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .unwrap();
        let image_param = builder.function_parameter(sampled_image).unwrap();
        let coord_param = builder.function_parameter(v2float).unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_sample_explicit_lod(
                v2float,
                None,
                image_param,
                coord_param,
                ImageOperands::LOD | ImageOperands::OFFSET,
                [
                    rspirv::dr::Operand::IdRef(float_zero),
                    rspirv::dr::Operand::IdRef(zero_offset),
                ],
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        builder.entry_point(ExecutionModel::Fragment, entry, "main", []);

        let binary = builder.module().assemble();
        let options = ValidationOptions {
            before_hlsl_legalization: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Vulkan1_2, options)
            .expect("Offset operand should be allowed when using the pre-HLSL legalization option");
    }

    #[test]
    fn bitwise_ops_require_32bit_in_vulkan_by_default() {
        use crate::validation::ValidationOptions;
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel};

        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(Capability::Shader);
        builder.capability(Capability::Int64);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let u64 = builder.type_int(64, 0);
        let fn_ty = builder.type_function(u64, [u64, u64]);
        builder
            .begin_function(u64, None, FunctionControl::NONE, fn_ty)
            .unwrap();
        let a = builder.function_parameter(u64).unwrap();
        let b = builder.function_parameter(u64).unwrap();
        builder.begin_block(None).unwrap();
        let or = builder.bitwise_or(u64, None, a, b).unwrap();
        builder.ret_value(or).unwrap();
        builder.end_function().unwrap();

        let binary = builder.module().assemble();
        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Vulkan1_1, ValidationOptions::default())
            .expect_err("64-bit bitwise ops should be disallowed by default in Vulkan");
        assert_eq!(
            err,
            ValidationError::VulkanBitwiseRequires32Bit {
                opcode: rspirv::spirv::Op::BitwiseOr,
                bit_width: 64
            }
        );
    }

    #[test]
    fn bitwise_ops_allow_non_32bit_when_option_enabled() {
        use crate::validation::ValidationOptions;
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel};

        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(Capability::Shader);
        builder.capability(Capability::Int64);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let u64 = builder.type_int(64, 0);
        let fn_ty = builder.type_function(u64, [u64, u64]);
        builder
            .begin_function(u64, None, FunctionControl::NONE, fn_ty)
            .unwrap();
        let a = builder.function_parameter(u64).unwrap();
        let b = builder.function_parameter(u64).unwrap();
        builder.begin_block(None).unwrap();
        let or = builder.bitwise_or(u64, None, a, b).unwrap();
        builder.ret_value(or).unwrap();
        builder.end_function().unwrap();

        let binary = builder.module().assemble();
        let options = ValidationOptions {
            allow_vulkan_32_bit_bitwise: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Vulkan1_1, options)
            .expect("64-bit bitwise ops should be allowed when option is enabled");
    }

    #[test]
    fn friendly_name_helpers_format_ids_and_members() {
        use crate::validation::ValidationOptions;

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpName %func \"friendly\"",
            "OpName %S \"Struct\"",
            "OpMemberName %S 0 \"field\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%S = OpTypeStruct %uint",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let options = ValidationOptions {
            use_friendly_names: true,
            ..ValidationOptions::default()
        };
        let module = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("validation should succeed");
        let names = module.friendly_names().expect("friendly names present");

        let (&named_func_id, _) = names
            .id_names()
            .iter()
            .find(|(_, name)| name.as_str() == "friendly")
            .expect("function name should be present");
        let formatted_func = names.format_id(named_func_id);
        assert!(
            formatted_func.contains("(friendly)"),
            "expected friendly suffix, got {formatted_func}"
        );

        let (&struct_id, _) = names
            .id_names()
            .iter()
            .find(|(_, name)| name.as_str() == "Struct")
            .expect("struct name should be present");
        let formatted_member = names.format_member(struct_id, MemberIndex(0));
        assert!(
            formatted_member.contains("(field)"),
            "expected member friendly suffix, got {formatted_member}"
        );
    }

    #[test]
    fn format_validation_error_uses_friendly_names_when_available() {
        use std::iter::FromIterator;

        let id = 42;
        let names = FriendlyNames::from_parts(
            HashMap::from_iter([(id, "named_func".to_string())]),
            HashMap::new(),
        );
        let error = ValidationError::ExecutionModeWithoutEntryPoint {
            function: Id::try_from(id).unwrap(),
        };
        let rendered = format_validation_error(&error, Some(&names));
        assert!(
            rendered.contains("named_func"),
            "expected friendly name in rendered error, got {rendered}"
        );

        let fallback = format_validation_error(&error, None);
        assert!(
            !fallback.contains("named_func"),
            "fallback should omit friendly name"
        );
    }

    #[test]
    fn format_validation_error_from_words_parses_names() {
        use crate::validation::ValidationOptions;

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpExecutionMode %main LocalSize 1 1 1",
            "OpName %main \"friendly\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        let binary = assemble_text(&text).expect("assemble");
        let options = ValidationOptions::default();
        let error = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options.clone())
            .expect_err("missing entry point should fail");
        let rendered = format_validation_error_from_words(binary.as_slice(), &options, &error);
        assert!(
            rendered.contains("friendly"),
            "expected rendered error to include friendly name, got {rendered}"
        );
    }

    #[test]
    fn layout_relaxation_flags_are_accepted() {
        use crate::validation::ValidationOptions;

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
        let options = ValidationOptions {
            relax_struct_store: true,
            relax_logical_pointer: true,
            relax_block_layout: true,
            uniform_buffer_standard_layout: true,
            scalar_block_layout: true,
            workgroup_scalar_block_layout: true,
            skip_block_layout: true,
            ..ValidationOptions::default()
        };

        binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("layout relaxation flags should be accepted");
    }

    #[test]
    fn skip_block_layout_allows_misordered_globals() {
        // OpExtension after type declarations should trigger a layout error unless skipped.
        let binary = vec![
            0x0723_0203, // magic
            0x0001_0000, // version
            0,           // generator
            6,           // bound
            0,           // schema
            op(2, 17),   // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(3, 11), // OpExtInstImport %3 "GLSL.std.450" (misordered after types)
            3,
            0x4c5347,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];

        use crate::validation::ValidationOptions;

        let err = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ExtInstImport
            }
        ));

        let options = ValidationOptions {
            skip_block_layout: true,
            ..ValidationOptions::default()
        };
        validate_module_with_options(&binary, TargetEnv::Universal1_6, options)
            .expect("skip_block_layout should bypass layout ordering errors");
    }

    #[test]
    fn logical_pointer_disallows_pointee_storage_class_without_relaxation() {
        let text = [
            "OpCapability Shader",
            "OpCapability VectorComputeINTEL",
            "OpCapability VectorAnyINTEL",
            "OpExtension \"SPV_INTEL_vector_compute\"",
            "OpMemoryModel Logical GLSL450",
            "%float = OpTypeFloat 32",
            "%ptr_uniform_float = OpTypePointer Uniform %float",
            "%ptr_private_ptr_uniform = OpTypePointer Private %ptr_uniform_float",
            "%var = OpVariable %ptr_private_ptr_uniform Private",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let err = binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect_err("logical pointer rules should reject pointers to Input");
        if let ValidationError::LogicalPointerPointeeStorageClassInvalid {
            pointee_storage_class: rspirv::spirv::StorageClass::Uniform,
            ..
        } = err
        {
        } else {
            panic!("unexpected error: {err:?}");
        }

        let options = ValidationOptions {
            relax_logical_pointer: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("relax_logical_pointer should permit pointer-to-pointer");
    }

    #[test]
    fn logical_pointer_requires_variable_pointer_capabilities() {
        let text = [
            "OpCapability Shader",
            "OpCapability VectorComputeINTEL",
            "OpCapability VectorAnyINTEL",
            "OpExtension \"SPV_INTEL_vector_compute\"",
            "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
            "OpExtension \"SPV_KHR_variable_pointers\"",
            "OpMemoryModel Logical GLSL450",
            "%int = OpTypeInt 32 0",
            "%ptr_sb_int = OpTypePointer StorageBuffer %int",
            "%ptr_private_ptr_sb = OpTypePointer Private %ptr_sb_int",
            "%var = OpVariable %ptr_private_ptr_sb Private",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");

        let err = binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect_err("missing VariablePointersStorageBuffer capability should error");
        if let ValidationError::LogicalPointerMissingCapability {
            required_capability: rspirv::spirv::Capability::VariablePointersStorageBuffer,
            ..
        } = err
        {
        } else {
            panic!("unexpected error: {err:?}");
        }

        let with_capability = [
            "OpCapability Shader",
            "OpCapability VectorComputeINTEL",
            "OpCapability VectorAnyINTEL",
            "OpCapability VariablePointersStorageBuffer",
            "OpExtension \"SPV_INTEL_vector_compute\"",
            "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
            "OpExtension \"SPV_KHR_variable_pointers\"",
            "OpMemoryModel Logical GLSL450",
            "%int = OpTypeInt 32 0",
            "%ptr_sb_int = OpTypePointer StorageBuffer %int",
            "%ptr_private_ptr_sb = OpTypePointer Private %ptr_sb_int",
            "%var = OpVariable %ptr_private_ptr_sb Private",
        ]
        .join("\n");
        assemble_text(&with_capability)
            .expect("assemble")
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect("declaring capability should permit pointer-to-pointer");
    }

    #[test]
    fn logical_pointer_rejects_non_function_or_private_storage_class() {
        let text = [
            "OpCapability Shader",
            "OpCapability VariablePointersStorageBuffer",
            "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
            "OpExtension \"SPV_KHR_variable_pointers\"",
            "OpMemoryModel Logical GLSL450",
            "%float = OpTypeFloat 32",
            "%ptr_sb_float = OpTypePointer StorageBuffer %float",
            "%ptr_sb_ptr = OpTypePointer StorageBuffer %ptr_sb_float",
            "%var = OpVariable %ptr_sb_ptr StorageBuffer",
        ]
        .join("\n");

        let binary = assemble_text(&text).expect("assemble");
        let err = binary
            .as_slice()
            .validate_with_options(
                TargetEnv::Universal1_6,
                ValidationOptions {
                    // Skip block layout so we exercise logical-pointer rules directly.
                    skip_block_layout: true,
                    ..ValidationOptions::default()
                },
            )
            .expect_err("logical pointer should reject non-Function/Private storage class");
        assert!(
            matches!(
                err,
                ValidationError::LogicalPointerInvalidStorageClass {
                    storage_class: rspirv::spirv::StorageClass::StorageBuffer,
                    ..
                }
            ),
            "expected storage-class diagnostic, got {err:?}"
        );
    }

    #[test]
    fn friendly_names_populated_on_valid_module() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpName %struct \"MyStruct\"",
            "OpMemberName %struct 0 \"field0\"",
            "%void = OpTypeVoid",
            "%struct = OpTypeStruct %void",
        ]
        .join("\n");

        let binary = assemble_text(&text).expect("assemble");
        let valid = binary
            .as_slice()
            .validate_with_options(
                TargetEnv::Universal1_6,
                ValidationOptions {
                    use_friendly_names: true,
                    ..ValidationOptions::default()
                },
            )
            .expect("validation should succeed");

        let names = valid
            .friendly_names()
            .expect("friendly names should be populated when enabled");
        let struct_id = valid
            .module()
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::TypeStruct)
            .and_then(|inst| inst.result_id)
            .expect("struct should have a result id");
        assert_eq!(names.id(struct_id), Some("MyStruct"));
        assert_eq!(
            names.member(struct_id, MemberIndex(0)),
            Some("field0"),
            "member names should be recorded"
        );
    }

    #[test]
    fn friendly_names_disabled_when_option_off() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpName %struct \"MyStruct\"",
            "%void = OpTypeVoid",
            "%struct = OpTypeStruct %void",
        ]
        .join("\n");

        let binary = assemble_text(&text).expect("assemble");
        let valid = binary
            .as_slice()
            .validate_with_options(
                TargetEnv::Universal1_6,
                ValidationOptions {
                    use_friendly_names: false,
                    ..ValidationOptions::default()
                },
            )
            .expect("validation should succeed");
        assert!(
            valid.friendly_names().is_none(),
            "friendly names should be omitted when disabled"
        );
    }

    #[test]
    fn relax_struct_store_allows_layout_compatible_structs() {
        use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};

        fn inst(
            opcode: rspirv::spirv::Op,
            result_type: Option<u32>,
            result_id: Option<u32>,
            operands: Vec<rspirv::dr::Operand>,
        ) -> Instruction {
            Instruction::new(opcode, result_type, result_id, operands)
        }

        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(11));
        module.capabilities.push(inst(
            rspirv::spirv::Op::Capability,
            None,
            None,
            vec![rspirv::dr::Operand::Capability(
                rspirv::spirv::Capability::Shader,
            )],
        ));
        module.memory_model = Some(inst(
            rspirv::spirv::Op::MemoryModel,
            None,
            None,
            vec![
                rspirv::dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                rspirv::dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
            ],
        ));
        module.types_global_values.extend([
            inst(rspirv::spirv::Op::TypeVoid, None, Some(1), vec![]),
            inst(
                rspirv::spirv::Op::TypeInt,
                None,
                Some(2),
                vec![
                    rspirv::dr::Operand::LiteralBit32(32),
                    rspirv::dr::Operand::LiteralBit32(0),
                ],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(3),
                vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(4),
                vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
            ),
            inst(
                rspirv::spirv::Op::TypePointer,
                None,
                Some(5),
                vec![
                    rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Function),
                    rspirv::dr::Operand::IdRef(3),
                ],
            ),
            inst(
                rspirv::spirv::Op::TypeFunction,
                None,
                Some(6),
                vec![rspirv::dr::Operand::IdRef(1)],
            ),
        ]);

        module.functions.push(rspirv::dr::Function {
            def: Some(inst(
                rspirv::spirv::Op::Function,
                Some(1),
                Some(7),
                vec![
                    rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                    rspirv::dr::Operand::IdRef(6),
                ],
            )),
            end: Some(inst(rspirv::spirv::Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![rspirv::dr::Block {
                label: Some(inst(rspirv::spirv::Op::Label, None, Some(8), vec![])),
                instructions: vec![
                    inst(
                        rspirv::spirv::Op::Variable,
                        Some(5),
                        Some(9),
                        vec![rspirv::dr::Operand::StorageClass(
                            rspirv::spirv::StorageClass::Function,
                        )],
                    ),
                    inst(rspirv::spirv::Op::Undef, Some(4), Some(10), vec![]),
                    inst(
                        rspirv::spirv::Op::Store,
                        None,
                        None,
                        vec![
                            rspirv::dr::Operand::IdRef(9),
                            rspirv::dr::Operand::IdRef(10),
                        ],
                    ),
                    inst(rspirv::spirv::Op::Return, None, None, vec![]),
                ],
            }],
        });

        module.entry_points.push(inst(
            rspirv::spirv::Op::EntryPoint,
            None,
            None,
            vec![
                rspirv::dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Vertex),
                rspirv::dr::Operand::IdRef(7),
                rspirv::dr::Operand::LiteralString("main".to_string()),
            ],
        ));

        let binary = module.assemble();
        let err = binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect_err("struct store types should mismatch by default");
        assert!(matches!(err, ValidationError::StoreTypeMismatch { .. }));

        let options = ValidationOptions {
            relax_struct_store: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("relax_struct_store should permit layout-compatible structs");
    }

    #[test]
    fn relax_struct_store_rejects_mismatched_array_lengths() {
        use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};

        fn inst(
            opcode: rspirv::spirv::Op,
            result_type: Option<u32>,
            result_id: Option<u32>,
            operands: Vec<rspirv::dr::Operand>,
        ) -> Instruction {
            Instruction::new(opcode, result_type, result_id, operands)
        }

        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(21));
        module.capabilities.push(inst(
            rspirv::spirv::Op::Capability,
            None,
            None,
            vec![rspirv::dr::Operand::Capability(
                rspirv::spirv::Capability::Shader,
            )],
        ));
        module.memory_model = Some(inst(
            rspirv::spirv::Op::MemoryModel,
            None,
            None,
            vec![
                rspirv::dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                rspirv::dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
            ],
        ));
        module.types_global_values.extend([
            inst(rspirv::spirv::Op::TypeVoid, None, Some(1), vec![]),
            inst(
                rspirv::spirv::Op::TypeInt,
                None,
                Some(2),
                vec![
                    rspirv::dr::Operand::LiteralBit32(32),
                    rspirv::dr::Operand::LiteralBit32(0),
                ],
            ),
            inst(
                rspirv::spirv::Op::Constant,
                Some(2),
                Some(3),
                vec![rspirv::dr::Operand::LiteralBit32(2)],
            ),
            inst(
                rspirv::spirv::Op::Constant,
                Some(2),
                Some(4),
                vec![rspirv::dr::Operand::LiteralBit32(3)],
            ),
            inst(
                rspirv::spirv::Op::TypeArray,
                None,
                Some(5),
                vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
            ),
            inst(
                rspirv::spirv::Op::TypeArray,
                None,
                Some(6),
                vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(7),
                vec![rspirv::dr::Operand::IdRef(5)],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(8),
                vec![rspirv::dr::Operand::IdRef(6)],
            ),
            inst(
                rspirv::spirv::Op::TypePointer,
                None,
                Some(9),
                vec![
                    rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Function),
                    rspirv::dr::Operand::IdRef(7),
                ],
            ),
            inst(
                rspirv::spirv::Op::TypeFunction,
                None,
                Some(10),
                vec![rspirv::dr::Operand::IdRef(1)],
            ),
            inst(rspirv::spirv::Op::TypeVoid, None, Some(20), vec![]),
        ]);

        module.functions.push(rspirv::dr::Function {
            def: Some(inst(
                rspirv::spirv::Op::Function,
                Some(1),
                Some(11),
                vec![
                    rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                    rspirv::dr::Operand::IdRef(10),
                ],
            )),
            end: Some(inst(rspirv::spirv::Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![rspirv::dr::Block {
                label: Some(inst(rspirv::spirv::Op::Label, None, Some(12), vec![])),
                instructions: vec![
                    inst(
                        rspirv::spirv::Op::Variable,
                        Some(9),
                        Some(13),
                        vec![rspirv::dr::Operand::StorageClass(
                            rspirv::spirv::StorageClass::Function,
                        )],
                    ),
                    inst(rspirv::spirv::Op::Undef, Some(8), Some(14), vec![]),
                    inst(
                        rspirv::spirv::Op::Store,
                        None,
                        None,
                        vec![
                            rspirv::dr::Operand::IdRef(13),
                            rspirv::dr::Operand::IdRef(14),
                        ],
                    ),
                    inst(rspirv::spirv::Op::Return, None, None, vec![]),
                ],
            }],
        });

        module.entry_points.push(inst(
            rspirv::spirv::Op::EntryPoint,
            None,
            None,
            vec![
                rspirv::dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Vertex),
                rspirv::dr::Operand::IdRef(11),
                rspirv::dr::Operand::LiteralString("main".to_string()),
            ],
        ));

        let binary = module.assemble();

        // Relaxation should still reject mismatched array lengths.
        let options = ValidationOptions {
            relax_struct_store: true,
            ..ValidationOptions::default()
        };
        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect_err("array length mismatch should not be considered layout-compatible");
        if let ValidationError::StoreTypeMismatch { .. } = err {
        } else {
            panic!("unexpected error: {err:?}");
        }
    }

    #[test]
    fn relax_struct_store_rejects_mismatched_array_stride() {
        use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};

        fn inst(
            opcode: rspirv::spirv::Op,
            result_type: Option<u32>,
            result_id: Option<u32>,
            operands: Vec<rspirv::dr::Operand>,
        ) -> Instruction {
            Instruction::new(opcode, result_type, result_id, operands)
        }

        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(15));
        module.capabilities.push(inst(
            rspirv::spirv::Op::Capability,
            None,
            None,
            vec![rspirv::dr::Operand::Capability(
                rspirv::spirv::Capability::Shader,
            )],
        ));
        module.memory_model = Some(inst(
            rspirv::spirv::Op::MemoryModel,
            None,
            None,
            vec![
                rspirv::dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                rspirv::dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
            ],
        ));
        module.types_global_values.extend([
            inst(rspirv::spirv::Op::TypeVoid, None, Some(1), vec![]),
            inst(
                rspirv::spirv::Op::TypeInt,
                None,
                Some(2),
                vec![
                    rspirv::dr::Operand::LiteralBit32(32),
                    rspirv::dr::Operand::LiteralBit32(0),
                ],
            ),
            inst(
                rspirv::spirv::Op::Constant,
                Some(2),
                Some(3),
                vec![rspirv::dr::Operand::LiteralBit32(2)],
            ),
            inst(
                rspirv::spirv::Op::TypeArray,
                None,
                Some(4),
                vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
            ),
            inst(
                rspirv::spirv::Op::TypeArray,
                None,
                Some(5),
                vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(6),
                vec![rspirv::dr::Operand::IdRef(4)],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(7),
                vec![rspirv::dr::Operand::IdRef(5)],
            ),
            inst(
                rspirv::spirv::Op::TypePointer,
                None,
                Some(8),
                vec![
                    rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Function),
                    rspirv::dr::Operand::IdRef(6),
                ],
            ),
            inst(
                rspirv::spirv::Op::TypeFunction,
                None,
                Some(9),
                vec![rspirv::dr::Operand::IdRef(1)],
            ),
        ]);
        module.annotations.extend([
            inst(
                rspirv::spirv::Op::Decorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(4),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::ArrayStride),
                    rspirv::dr::Operand::LiteralBit32(4),
                ],
            ),
            inst(
                rspirv::spirv::Op::Decorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(5),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::ArrayStride),
                    rspirv::dr::Operand::LiteralBit32(8),
                ],
            ),
        ]);

        module.functions.push(rspirv::dr::Function {
            def: Some(inst(
                rspirv::spirv::Op::Function,
                Some(1),
                Some(10),
                vec![
                    rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                    rspirv::dr::Operand::IdRef(9),
                ],
            )),
            end: Some(inst(rspirv::spirv::Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![rspirv::dr::Block {
                label: Some(inst(rspirv::spirv::Op::Label, None, Some(11), vec![])),
                instructions: vec![
                    inst(
                        rspirv::spirv::Op::Variable,
                        Some(8),
                        Some(12),
                        vec![rspirv::dr::Operand::StorageClass(
                            rspirv::spirv::StorageClass::Function,
                        )],
                    ),
                    inst(rspirv::spirv::Op::Undef, Some(7), Some(13), vec![]),
                    inst(
                        rspirv::spirv::Op::Store,
                        None,
                        None,
                        vec![
                            rspirv::dr::Operand::IdRef(12),
                            rspirv::dr::Operand::IdRef(13),
                        ],
                    ),
                    inst(rspirv::spirv::Op::Return, None, None, vec![]),
                ],
            }],
        });

        module.entry_points.push(inst(
            rspirv::spirv::Op::EntryPoint,
            None,
            None,
            vec![
                rspirv::dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Vertex),
                rspirv::dr::Operand::IdRef(10),
                rspirv::dr::Operand::LiteralString("main".to_string()),
            ],
        ));

        // Stride metadata must be respected when comparing array layouts.
        let definitions = collect_result_instructions(&module);
        assert_eq!(
            array_stride(&module, ResultId::try_from(4).unwrap()),
            Some(4)
        );
        assert_eq!(
            array_stride(&module, ResultId::try_from(5).unwrap()),
            Some(8)
        );
        assert!(
            !layout_compatible_types(
                TypeId::try_from(6).unwrap(),
                TypeId::try_from(7).unwrap(),
                &module,
                &definitions,
                &mut HashSet::new()
            ),
            "array stride mismatch should render types layout-incompatible"
        );
        let options = ValidationOptions {
            relax_struct_store: true,
            ..ValidationOptions::default()
        };
        let err = enforce_store_type_compatibility(&module, &definitions, &options);
        assert!(
            matches!(err, Err(ValidationError::StoreTypeMismatch { .. })),
            "layout mismatch should be rejected even under relax_struct_store"
        );

        let binary = module.assemble();
        let parsed = parse_module(binary.as_slice())
            .expect("assembled module should round-trip through parser");
        assert_eq!(
            array_stride(&parsed, ResultId::try_from(4).unwrap()),
            Some(4)
        );
        assert_eq!(
            array_stride(&parsed, ResultId::try_from(5).unwrap()),
            Some(8)
        );
        let parsed_definitions = collect_result_instructions(&parsed);
        assert!(
            !layout_compatible_types(
                TypeId::try_from(6).unwrap(),
                TypeId::try_from(7).unwrap(),
                &parsed,
                &parsed_definitions,
                &mut HashSet::new()
            ),
            "parsed module should preserve stride mismatch"
        );
        let err = enforce_store_type_compatibility(&parsed, &parsed_definitions, &options);
        assert!(
            matches!(err, Err(ValidationError::StoreTypeMismatch { .. })),
            "parsed module should also reject incompatible strides"
        );
        let validation_result = validate_words(
            ModuleWords::from(Arc::from(binary.as_slice())),
            TargetEnv::Universal1_6,
            options.clone(),
        );
        match validation_result {
            Err(ValidationError::StoreTypeMismatch { .. }) => {}
            Err(other) => panic!("full validation path failed with unexpected error: {other:?}"),
            Ok(_) => panic!("full validation path should reject incompatible strides"),
        }
        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect_err("array stride mismatch should still fail without layout relaxation");
        assert!(matches!(err, ValidationError::StoreTypeMismatch { .. }));

        let relaxed = ValidationOptions {
            relax_struct_store: true,
            relax_block_layout: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, relaxed)
            .expect("layout relaxation should bypass array stride mismatch");
    }

    #[test]
    fn relax_struct_store_with_layout_relaxation_accepts_incompatible_structs() {
        use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};

        fn inst(
            opcode: rspirv::spirv::Op,
            result_type: Option<u32>,
            result_id: Option<u32>,
            operands: Vec<rspirv::dr::Operand>,
        ) -> Instruction {
            Instruction::new(opcode, result_type, result_id, operands)
        }

        // S0 has two members, S1 has one; store should pass when both relax_struct_store
        // and a block-layout relaxation flag are set.
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(11));
        module.capabilities.push(inst(
            rspirv::spirv::Op::Capability,
            None,
            None,
            vec![rspirv::dr::Operand::Capability(
                rspirv::spirv::Capability::Shader,
            )],
        ));
        module.memory_model = Some(inst(
            rspirv::spirv::Op::MemoryModel,
            None,
            None,
            vec![
                rspirv::dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                rspirv::dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
            ],
        ));
        module.types_global_values.extend([
            inst(rspirv::spirv::Op::TypeVoid, None, Some(1), vec![]),
            inst(
                rspirv::spirv::Op::TypeInt,
                None,
                Some(2),
                vec![
                    rspirv::dr::Operand::LiteralBit32(32),
                    rspirv::dr::Operand::LiteralBit32(0),
                ],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(3),
                vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(4),
                vec![rspirv::dr::Operand::IdRef(2)],
            ),
            inst(
                rspirv::spirv::Op::TypePointer,
                None,
                Some(5),
                vec![
                    rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Function),
                    rspirv::dr::Operand::IdRef(3),
                ],
            ),
            inst(
                rspirv::spirv::Op::TypeFunction,
                None,
                Some(6),
                vec![rspirv::dr::Operand::IdRef(1)],
            ),
        ]);

        module.functions.push(rspirv::dr::Function {
            def: Some(inst(
                rspirv::spirv::Op::Function,
                Some(1),
                Some(7),
                vec![
                    rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                    rspirv::dr::Operand::IdRef(6),
                ],
            )),
            end: Some(inst(rspirv::spirv::Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![rspirv::dr::Block {
                label: Some(inst(rspirv::spirv::Op::Label, None, Some(8), vec![])),
                instructions: vec![
                    inst(
                        rspirv::spirv::Op::Variable,
                        Some(5),
                        Some(9),
                        vec![rspirv::dr::Operand::StorageClass(
                            rspirv::spirv::StorageClass::Function,
                        )],
                    ),
                    inst(rspirv::spirv::Op::Undef, Some(4), Some(10), vec![]),
                    inst(
                        rspirv::spirv::Op::Store,
                        None,
                        None,
                        vec![
                            rspirv::dr::Operand::IdRef(9),
                            rspirv::dr::Operand::IdRef(10),
                        ],
                    ),
                    inst(rspirv::spirv::Op::Return, None, None, vec![]),
                ],
            }],
        });

        module.entry_points.push(inst(
            rspirv::spirv::Op::EntryPoint,
            None,
            None,
            vec![
                rspirv::dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Vertex),
                rspirv::dr::Operand::IdRef(7),
                rspirv::dr::Operand::LiteralString("main".to_string()),
            ],
        ));

        let binary = module.assemble();
        let options = ValidationOptions {
            relax_struct_store: true,
            relax_block_layout: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("relax_struct_store with layout relaxation should allow mismatched structs");
    }

    #[test]
    fn block_layout_requires_member_offsets() {
        use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};

        fn inst(
            opcode: rspirv::spirv::Op,
            result_type: Option<u32>,
            result_id: Option<u32>,
            operands: Vec<rspirv::dr::Operand>,
        ) -> Instruction {
            Instruction::new(opcode, result_type, result_id, operands)
        }

        fn make_block_struct(member_offsets: Option<Vec<u32>>) -> Vec<u32> {
            let mut module = Module::new();
            module.header = Some(ModuleHeader::new(8));
            module.capabilities.push(inst(
                rspirv::spirv::Op::Capability,
                None,
                None,
                vec![rspirv::dr::Operand::Capability(
                    rspirv::spirv::Capability::Shader,
                )],
            ));
            module.memory_model = Some(inst(
                rspirv::spirv::Op::MemoryModel,
                None,
                None,
                vec![
                    rspirv::dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                    rspirv::dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
                ],
            ));
            module.types_global_values.extend([
                inst(rspirv::spirv::Op::TypeVoid, None, Some(1), vec![]),
                inst(
                    rspirv::spirv::Op::TypeInt,
                    None,
                    Some(2),
                    vec![
                        rspirv::dr::Operand::LiteralBit32(32),
                        rspirv::dr::Operand::LiteralBit32(0),
                    ],
                ),
                inst(
                    rspirv::spirv::Op::TypeStruct,
                    None,
                    Some(3),
                    vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
                ),
            ]);
            module.annotations.push(inst(
                rspirv::spirv::Op::Decorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(3),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
                ],
            ));
            if let Some(offsets) = member_offsets {
                for (index, offset) in offsets.into_iter().enumerate() {
                    module.annotations.push(inst(
                        rspirv::spirv::Op::MemberDecorate,
                        None,
                        None,
                        vec![
                            rspirv::dr::Operand::IdRef(3),
                            rspirv::dr::Operand::LiteralBit32(index as u32),
                            rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                            rspirv::dr::Operand::LiteralBit32(offset),
                        ],
                    ));
                }
            }
            module.assemble()
        }

        let binary = make_block_struct(None);
        let err = binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect_err("missing member offsets should fail block layout");
        match err {
            ValidationError::InvalidBlockLayout {
                struct_type,
                reason,
                ..
            } => {
                assert_eq!(u32::from(struct_type), 3);
                assert!(reason.contains("Offset"), "unexpected reason: {reason:?}");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        let relax_options = ValidationOptions {
            relax_block_layout: true,
            ..ValidationOptions::default()
        };
        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, relax_options)
            .expect_err("relax_block_layout should still require offsets");
        assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));

        let options = ValidationOptions {
            skip_block_layout: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("skip_block_layout should skip member offset enforcement");
    }

    #[test]
    fn block_layout_rejects_overlapping_offsets() {
        use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};

        fn inst(
            opcode: rspirv::spirv::Op,
            result_type: Option<u32>,
            result_id: Option<u32>,
            operands: Vec<rspirv::dr::Operand>,
        ) -> Instruction {
            Instruction::new(opcode, result_type, result_id, operands)
        }

        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(8));
        module.capabilities.push(inst(
            rspirv::spirv::Op::Capability,
            None,
            None,
            vec![rspirv::dr::Operand::Capability(
                rspirv::spirv::Capability::Shader,
            )],
        ));
        module.memory_model = Some(inst(
            rspirv::spirv::Op::MemoryModel,
            None,
            None,
            vec![
                rspirv::dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                rspirv::dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
            ],
        ));
        module.types_global_values.extend([
            inst(rspirv::spirv::Op::TypeVoid, None, Some(1), vec![]),
            inst(
                rspirv::spirv::Op::TypeInt,
                None,
                Some(2),
                vec![
                    rspirv::dr::Operand::LiteralBit32(32),
                    rspirv::dr::Operand::LiteralBit32(0),
                ],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(3),
                vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
            ),
        ]);
        module.annotations.extend([
            inst(
                rspirv::spirv::Op::Decorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(3),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
                ],
            ),
            inst(
                rspirv::spirv::Op::MemberDecorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(3),
                    rspirv::dr::Operand::LiteralBit32(0),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                    rspirv::dr::Operand::LiteralBit32(0),
                ],
            ),
            inst(
                rspirv::spirv::Op::MemberDecorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(3),
                    rspirv::dr::Operand::LiteralBit32(1),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                    rspirv::dr::Operand::LiteralBit32(2),
                ],
            ),
        ]);

        let binary = module.assemble();
        let err = binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect_err("overlapping member offsets should fail block layout");
        match err {
            ValidationError::InvalidBlockLayout {
                struct_type,
                reason,
                ..
            } => {
                assert_eq!(u32::from(struct_type), 3);
                assert!(reason.contains("overlap"), "unexpected reason: {reason:?}");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        let relax_options = ValidationOptions {
            relax_block_layout: true,
            ..ValidationOptions::default()
        };
        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, relax_options)
            .expect_err("relax_block_layout should still enforce overlap constraints");
        assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));

        let options = ValidationOptions {
            skip_block_layout: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect("skip_block_layout should skip overlap checks");
    }

    #[test]
    fn relax_block_layout_allows_scalar_vector_alignment() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpDecorate %struct Block",
            "OpMemberDecorate %struct 0 Offset 0",
            "OpMemberDecorate %struct 1 Offset 4",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%struct = OpTypeStruct %int %vec2",
        ]
        .join("\n");
        let err = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("vector offset should require base alignment in strict layout");
        if let ValidationError::InvalidBlockLayout { reason, .. } = err {
            assert!(reason.contains("aligned"), "unexpected reason: {reason:?}");
        } else {
            panic!("unexpected error: {err:?}");
        }

        let relax = ValidationOptions {
            relax_block_layout: true,
            ..ValidationOptions::default()
        };
        text.as_str()
            .validate_with_options(TargetEnv::Universal1_6, relax)
            .expect("relax_block_layout should permit scalar-aligned vectors");

        let scalar = ValidationOptions {
            scalar_block_layout: true,
            ..ValidationOptions::default()
        };
        text.as_str()
            .validate_with_options(TargetEnv::Universal1_6, scalar)
            .expect("scalar_block_layout should permit scalar alignment for vectors");
    }

    #[test]
    fn uniform_buffer_standard_layout_allows_scalar_vector_alignment() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpDecorate %struct Block",
            "OpMemberDecorate %struct 0 Offset 0",
            "OpMemberDecorate %struct 1 Offset 4",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%struct = OpTypeStruct %int %vec2",
        ]
        .join("\n");
        let relax = ValidationOptions {
            uniform_buffer_standard_layout: true,
            ..ValidationOptions::default()
        };
        text.as_str()
            .validate_with_options(TargetEnv::Universal1_6, relax)
            .expect("uniform_buffer_standard_layout should permit scalar-aligned vectors");
    }

    #[test]
    fn workgroup_scalar_block_layout_uses_scalar_alignment() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpDecorate %struct Block",
            "OpMemberDecorate %struct 0 Offset 0",
            "OpMemberDecorate %struct 1 Offset 4",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%struct = OpTypeStruct %int %vec2",
        ]
        .join("\n");
        let relax = ValidationOptions {
            workgroup_scalar_block_layout: true,
            ..ValidationOptions::default()
        };
        text.as_str()
            .validate_with_options(TargetEnv::Universal1_6, relax)
            .expect("workgroup_scalar_block_layout should permit scalar alignment for vectors");
    }

    #[test]
    fn array_stride_must_align_to_element() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpDecorate %struct Block",
            "OpMemberDecorate %struct 0 Offset 0",
            "OpMemberDecorate %struct 1 Offset 16",
            "OpDecorate %arr ArrayStride 6",
            "%int = OpTypeInt 32 0",
            "%arr = OpTypeArray %int %len",
            "%len = OpConstant %int 2",
            "%struct = OpTypeStruct %arr %int",
        ]
        .join("\n");
        let err = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("array stride not aligned to element size should fail");
        assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
    }

    #[test]
    fn vector_straddle_rejected_under_relax() {
        let text = [
            "OpCapability Shader",
            "OpCapability Float64",
            "OpMemoryModel Logical GLSL450",
            "OpDecorate %struct Block",
            "OpMemberDecorate %struct 0 Offset 0",
            "OpMemberDecorate %struct 1 Offset 8",
            "%f64 = OpTypeFloat 64",
            "%v3 = OpTypeVector %f64 3",
            "%struct = OpTypeStruct %f64 %v3",
        ]
        .join("\n");
        let err = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("misaligned vector should fail");
        assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));

        let relax = ValidationOptions {
            relax_block_layout: true,
            ..ValidationOptions::default()
        };
        let err = text
            .as_str()
            .validate_with_options(TargetEnv::Universal1_6, relax)
            .expect_err("relaxed layout still rejects improper vector straddle");
        assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
    }

    #[test]
    fn matrix_stride_alignment_and_size() {
        use crate::validation::{
            collect_result_instructions, enforce_block_layout_rules, parse_module,
        };
        let text = [
            "OpCapability Shader",
            "OpCapability Float64",
            "OpMemoryModel Logical GLSL450",
            "%f64 = OpTypeFloat 64",
            "%v2 = OpTypeVector %f64 2",
            "%mat2 = OpTypeMatrix %v2 2",
            "%struct = OpTypeStruct %v2 %mat2",
            "OpDecorate %struct Block",
            "OpMemberDecorate %struct 0 Offset 0",
            "OpMemberDecorate %struct 1 Offset 32",
            "OpMemberDecorate %struct 1 RowMajor",
            "OpMemberDecorate %struct 1 MatrixStride 8",
        ]
        .join("\n");
        let words = assemble_text(&text).expect("assemble");
        let module = parse_module(&words).expect("parse");
        let definitions = collect_result_instructions(&module);
        let err = enforce_block_layout_rules(&module, &definitions, &ValidationOptions::default())
            .expect_err("matrix stride smaller than column size should fail");
        assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));

        let aligned = [
            "OpCapability Shader",
            "OpCapability Float64",
            "OpMemoryModel Logical GLSL450",
            "%f64 = OpTypeFloat 64",
            "%v2 = OpTypeVector %f64 2",
            "%mat2 = OpTypeMatrix %v2 2",
            "%struct = OpTypeStruct %v2 %mat2",
            "OpDecorate %struct Block",
            "OpMemberDecorate %struct 0 Offset 0",
            "OpMemberDecorate %struct 1 Offset 32",
            "OpMemberDecorate %struct 1 RowMajor",
            "OpMemberDecorate %struct 1 MatrixStride 16",
        ]
        .join("\n");
        let aligned_words = assemble_text(&aligned).expect("assemble");
        let aligned_module = parse_module(&aligned_words).expect("parse");
        let definitions = collect_result_instructions(&aligned_module);
        enforce_block_layout_rules(&aligned_module, &definitions, &ValidationOptions::default())
            .expect("aligned matrix stride should pass");
    }

    #[test]
    fn runtime_array_must_be_last_member() {
        use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};

        fn inst(
            opcode: rspirv::spirv::Op,
            result_type: Option<u32>,
            result_id: Option<u32>,
            operands: Vec<rspirv::dr::Operand>,
        ) -> Instruction {
            Instruction::new(opcode, result_type, result_id, operands)
        }

        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(8));
        module.capabilities.push(inst(
            rspirv::spirv::Op::Capability,
            None,
            None,
            vec![rspirv::dr::Operand::Capability(
                rspirv::spirv::Capability::Shader,
            )],
        ));
        module.memory_model = Some(inst(
            rspirv::spirv::Op::MemoryModel,
            None,
            None,
            vec![
                rspirv::dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                rspirv::dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
            ],
        ));
        module.types_global_values.extend([
            inst(
                rspirv::spirv::Op::TypeInt,
                None,
                Some(1),
                vec![
                    rspirv::dr::Operand::LiteralBit32(32),
                    rspirv::dr::Operand::LiteralBit32(0),
                ],
            ),
            inst(
                rspirv::spirv::Op::TypeRuntimeArray,
                None,
                Some(2),
                vec![rspirv::dr::Operand::IdRef(1)],
            ),
            inst(
                rspirv::spirv::Op::TypeStruct,
                None,
                Some(3),
                vec![
                    rspirv::dr::Operand::IdRef(2),
                    rspirv::dr::Operand::IdRef(1),
                    rspirv::dr::Operand::IdRef(1),
                ],
            ),
        ]);
        module.annotations.extend([
            inst(
                rspirv::spirv::Op::Decorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(2),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::ArrayStride),
                    rspirv::dr::Operand::LiteralBit32(4),
                ],
            ),
            inst(
                rspirv::spirv::Op::Decorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(3),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
                ],
            ),
            inst(
                rspirv::spirv::Op::MemberDecorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(3),
                    rspirv::dr::Operand::LiteralBit32(0),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                    rspirv::dr::Operand::LiteralBit32(0),
                ],
            ),
            inst(
                rspirv::spirv::Op::MemberDecorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(3),
                    rspirv::dr::Operand::LiteralBit32(1),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                    rspirv::dr::Operand::LiteralBit32(16),
                ],
            ),
            inst(
                rspirv::spirv::Op::MemberDecorate,
                None,
                None,
                vec![
                    rspirv::dr::Operand::IdRef(3),
                    rspirv::dr::Operand::LiteralBit32(2),
                    rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                    rspirv::dr::Operand::LiteralBit32(32),
                ],
            ),
        ]);

        let binary = module.assemble();
        let err = binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect_err("runtime array must be the final member");
        assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));

        let skip = ValidationOptions {
            skip_block_layout: true,
            ..ValidationOptions::default()
        };
        binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, skip)
            .expect("skip_block_layout should bypass runtime array placement rule");
    }

    #[test]
    fn switch_branch_limit_enforced() {
        use crate::validation::{
            enforce_switch_branch_limit, ValidationOptions, LIMIT_MAX_SWITCH_BRANCHES,
        };
        use rspirv::dr::{Instruction, Module, Operand};

        // Build a minimal module with an OpSwitch that exceeds the configured limit.
        let switch_inst = Instruction::new(
            rspirv::spirv::Op::Switch,
            None,
            None,
            vec![
                Operand::IdRef(1),
                Operand::IdRef(2),
                Operand::LiteralBit32(0),
                Operand::IdRef(3),
                Operand::LiteralBit32(1),
                Operand::IdRef(4),
            ],
        );
        let block = rspirv::dr::Block {
            label: None,
            instructions: vec![switch_inst],
        };
        let function = rspirv::dr::Function {
            def: None,
            parameters: Vec::new(),
            blocks: vec![block],
            end: None,
        };
        let module = Module {
            header: None,
            capabilities: Vec::new(),
            extensions: Vec::new(),
            ext_inst_imports: Vec::new(),
            memory_model: None,
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            debug_string_source: Vec::new(),
            debug_names: Vec::new(),
            debug_module_processed: Vec::new(),
            annotations: Vec::new(),
            types_global_values: Vec::new(),
            functions: vec![function],
        };

        let mut options = ValidationOptions::default();
        options.limits.insert(LIMIT_MAX_SWITCH_BRANCHES, 2);

        let err = enforce_switch_branch_limit(&module, &options)
            .expect_err("switch branch limit should be enforced");
        assert_eq!(
            err,
            ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_SWITCH_BRANCHES,
                limit: 2,
                found: 3
            }
        );
    }

    #[test]
    fn id_bound_limit_is_enforced() {
        use crate::validation::{ValidationOptions, LIMIT_MAX_ID_BOUND};

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
        let mut options = ValidationOptions::default();
        options.limits.insert(LIMIT_MAX_ID_BOUND, 3);

        let err = binary
            .as_slice()
            .validate_with_options(TargetEnv::Universal1_6, options)
            .expect_err("id bound should exceed configured limit");
        assert_eq!(
            err,
            ValidationError::IdBoundExceedsLimit {
                declared: DeclaredBound(5),
                limit: 3
            }
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
    fn conditional_entry_point_must_precede_debug_names() {
        let intel_function_variants_ext = [
            1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
        ];
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            9,          // bound (ids up to 8)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // OpCapability SpecConditionalINTEL
            rspirv::spirv::Capability::SpecConditionalINTEL as u32,
            0x0008_000a, // OpExtension "SPV_INTEL_function_variants"
            intel_function_variants_ext[0],
            intel_function_variants_ext[1],
            intel_function_variants_ext[2],
            intel_function_variants_ext[3],
            intel_function_variants_ext[4],
            intel_function_variants_ext[5],
            intel_function_variants_ext[6],
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(4, 5), // OpName %4 "main"
            4,
            0x6e69_616d,
            0,
            op(6, 6249), // OpConditionalEntryPointINTEL %5 Vertex %4 "main"
            5,
            rspirv::spirv::ExecutionModel::Vertex as u32,
            4,
            0x6e69_616d,
            0,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %4 None %2
            1,
            4,
            0,
            2,
            op(2, 248), // OpLabel %6
            6,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalEntryPointINTEL
            }
        );
    }

    #[test]
    fn conditional_entry_point_cannot_follow_functions() {
        let intel_function_variants_ext = [
            1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
        ];
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            7,          // bound (ids up to 6)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // OpCapability SpecConditionalINTEL
            rspirv::spirv::Capability::SpecConditionalINTEL as u32,
            0x0008_000a, // OpExtension "SPV_INTEL_function_variants"
            intel_function_variants_ext[0],
            intel_function_variants_ext[1],
            intel_function_variants_ext[2],
            intel_function_variants_ext[3],
            intel_function_variants_ext[4],
            intel_function_variants_ext[5],
            intel_function_variants_ext[6],
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %1 %3 None %2
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253),  // OpReturn
            op(1, 56),   // OpFunctionEnd
            op(6, 6249), // OpConditionalEntryPointINTEL %5 Vertex %3 "main"
            5,
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d,
            0,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalEntryPointINTEL
            }
        );
    }

    #[test]
    fn conditional_entry_point_must_reference_function() {
        let intel_function_variants_ext = [
            1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
        ];
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            8,          // bound (ids up to 7)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // OpCapability SpecConditionalINTEL
            rspirv::spirv::Capability::SpecConditionalINTEL as u32,
            0x0008_000a, // OpExtension "SPV_INTEL_function_variants"
            intel_function_variants_ext[0],
            intel_function_variants_ext[1],
            intel_function_variants_ext[2],
            intel_function_variants_ext[3],
            intel_function_variants_ext[4],
            intel_function_variants_ext[5],
            intel_function_variants_ext[6],
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(6, 6249), // OpConditionalEntryPointINTEL %5 Vertex %5 "main"
            5,
            rspirv::spirv::ExecutionModel::Vertex as u32,
            5,
            0x6e69_616d,
            0,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(2, 20), // OpTypeBool %3
            3,
            op(3, 41), // OpConstantTrue %3 %5
            3,
            5,
            op(5, 54), // OpFunction %1 %6 None %2
            1,
            6,
            0,
            2,
            op(2, 248), // OpLabel %7
            7,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::InvalidEntryPointTarget {
                target: Id::try_from(5).unwrap(),
                opcode: rspirv::spirv::Op::ConstantTrue
            }
        );
    }

    #[test]
    fn conditional_entry_point_cannot_follow_annotations() {
        let intel_function_variants_ext = [
            1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
        ];
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            5,          // bound (ids up to 4)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // OpCapability SpecConditionalINTEL
            rspirv::spirv::Capability::SpecConditionalINTEL as u32,
            0x0008_000a, // OpExtension "SPV_INTEL_function_variants"
            intel_function_variants_ext[0],
            intel_function_variants_ext[1],
            intel_function_variants_ext[2],
            intel_function_variants_ext[3],
            intel_function_variants_ext[4],
            intel_function_variants_ext[5],
            intel_function_variants_ext[6],
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 73), // OpDecorationGroup %1 (annotations)
            1,
            op(6, 6249), // OpConditionalEntryPointINTEL %2 Vertex %3 "main"
            2,
            rspirv::spirv::ExecutionModel::Vertex as u32,
            3,
            0x6e69_616d, // "main"
            0,
            op(2, 19), // OpTypeVoid %4
            4,
            op(3, 33), // OpTypeFunction %5 %4
            5,
            4,
            op(5, 54), // OpFunction %4 %3 None %5
            4,
            3,
            0,
            5,
            op(2, 248), // OpLabel %6
            6,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::ConditionalEntryPointINTEL
            }
        );
    }

    #[test]
    fn spec_conditional_capability_requires_extension() {
        let text = [
            "OpCapability Kernel",
            "OpCapability Linkage",
            "OpCapability SpecConditionalINTEL",
            "OpMemoryModel Logical OpenCL",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("SpecConditionalINTEL requires SPV_INTEL_function_variants");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::SpecConditionalINTEL,
                required_extension: "SPV_INTEL_function_variants".to_string()
            }
        );
    }

    #[test]
    fn function_variants_capability_requires_spec_conditional() {
        let text = [
            "OpCapability Kernel",
            "OpCapability Linkage",
            "OpCapability FunctionVariantsINTEL",
            "OpExtension \"SPV_INTEL_function_variants\"",
            "OpMemoryModel Logical OpenCL",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("FunctionVariantsINTEL requires SpecConditionalINTEL capability");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::SpecConditionalINTEL,
                capability: rspirv::spirv::Capability::FunctionVariantsINTEL
            }
        );
    }

    #[test]
    fn function_variants_capability_accepts_extension_and_dependency() {
        let text = [
            "OpCapability Kernel",
            "OpCapability Linkage",
            "OpCapability SpecConditionalINTEL",
            "OpCapability FunctionVariantsINTEL",
            "OpExtension \"SPV_INTEL_function_variants\"",
            "OpMemoryModel Logical OpenCL",
        ]
        .join("\n");
        text.as_str().validate(TargetEnv::Universal1_6).expect(
            "FunctionVariantsINTEL should be accepted with required extension and capability",
        );
    }

    #[test]
    fn function_variants_extension_rejected_for_vulkan() {
        let text = [
            "OpCapability Shader",
            "OpCapability SpecConditionalINTEL",
            "OpCapability FunctionVariantsINTEL",
            "OpExtension \"SPV_INTEL_function_variants\"",
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
            .expect_err("Intel function variants extension should be rejected for Vulkan");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_INTEL_function_variants"),
                env: TargetEnv::Vulkan1_2
            }
        );

        text.as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("Universal environment should accept vendor extensions");
    }

    #[test]
    fn conditional_entry_point_accepts_execution_modes() {
        let intel_function_variants_ext = [
            1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
        ];
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            9,          // bound (ids up to 8)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(2, 17), // OpCapability SpecConditionalINTEL
            rspirv::spirv::Capability::SpecConditionalINTEL as u32,
            0x0008_000a, // OpExtension "SPV_INTEL_function_variants"
            intel_function_variants_ext[0],
            intel_function_variants_ext[1],
            intel_function_variants_ext[2],
            intel_function_variants_ext[3],
            intel_function_variants_ext[4],
            intel_function_variants_ext[5],
            intel_function_variants_ext[6],
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(6, 6249), // OpConditionalEntryPointINTEL %5 Vertex %7 "main"
            5,
            rspirv::spirv::ExecutionModel::Vertex as u32,
            7,
            0x6e69_616d,
            0,
            op(3, 16), // OpExecutionMode %7 OriginUpperLeft
            7,
            rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(2, 20), // OpTypeBool %3
            3,
            op(3, 41), // OpConstantTrue %3 %5
            3,
            5,
            op(5, 54), // OpFunction %1 %7 None %2
            1,
            7,
            0,
            2,
            op(2, 248), // OpLabel %8
            8,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .expect("conditional entry points should participate in execution-mode validation");
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
    fn cooperative_matrix_capability_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability CooperativeMatrixNV",
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
            .expect_err("CooperativeMatrixNV requires its enabling extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::CooperativeMatrixNV,
                required_extension: "SPV_NV_cooperative_matrix".to_string()
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability CooperativeMatrixNV",
            "OpExtension \"SPV_NV_cooperative_matrix\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let validated = with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("capability should be accepted with required extension");
        assert_eq!(validated.header().schema(), Schema::ZERO);
    }

    #[test]
    fn cooperative_matrix_khr_capability_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability CooperativeMatrixKHR",
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
            .expect_err("CooperativeMatrixKHR requires its enabling extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::CooperativeMatrixKHR,
                required_extension: "SPV_KHR_cooperative_matrix".to_string()
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability CooperativeMatrixKHR",
            "OpExtension \"SPV_KHR_cooperative_matrix\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let validated = with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("capability should be accepted with required extension");
        assert_eq!(validated.header().schema(), Schema::ZERO);
    }

    #[test]
    fn ray_tracing_motion_blur_capability_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingNV",
            "OpCapability RayTracingMotionBlurNV",
            "OpExtension \"SPV_NV_ray_tracing\"",
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
            .expect_err("RayTracingMotionBlurNV requires its enabling extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::RayTracingMotionBlurNV,
                required_extension: "SPV_NV_ray_tracing_motion_blur".to_string()
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability RayTracingNV",
            "OpCapability RayTracingMotionBlurNV",
            "OpExtension \"SPV_NV_ray_tracing\"",
            "OpExtension \"SPV_NV_ray_tracing_motion_blur\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("capability should be accepted with required extensions");
    }

    #[test]
    fn ray_tracing_displacement_micromap_capability_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability RayTracingNV",
            "OpCapability RayTracingDisplacementMicromapNV",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_NV_ray_tracing\"",
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
            .expect_err("RayTracingDisplacementMicromapNV requires its enabling extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::RayTracingDisplacementMicromapNV,
                required_extension: "SPV_NV_displacement_micromap".to_string()
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability RayTracingNV",
            "OpCapability RayTracingDisplacementMicromapNV",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_NV_ray_tracing\"",
            "OpExtension \"SPV_NV_displacement_micromap\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("capability should be accepted with required extensions");
    }

    #[test]
    fn ray_tracing_linear_swept_spheres_capability_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingLinearSweptSpheresGeometryNV",
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
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("RayTracingLinearSweptSpheresGeometryNV requires its enabling extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::RayTracingLinearSweptSpheresGeometryNV,
                required_extension: "SPV_NV_linear_swept_spheres".to_string()
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability RayTracingLinearSweptSpheresGeometryNV",
            "OpCapability RayTracingKHR",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_NV_linear_swept_spheres\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("capability should be accepted with required extensions");
    }

    #[test]
    fn ray_tracing_opacity_micromap_capability_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability RayTracingOpacityMicromapEXT",
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
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("RayTracingOpacityMicromapEXT requires its enabling extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::RayTracingOpacityMicromapEXT,
                required_extension: "SPV_EXT_opacity_micromap".to_string()
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability RayTracingKHR",
            "OpCapability RayTracingOpacityMicromapEXT",
            "OpExtension \"SPV_KHR_ray_tracing\"",
            "OpExtension \"SPV_EXT_opacity_micromap\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("capability should be accepted with required extensions");
    }

    #[test]
    fn shader_clock_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_KHR_shader_clock");
    }

    #[test]
    fn fragment_shader_barycentric_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_KHR_fragment_shader_barycentric");
    }

    #[test]
    fn qcom_cooperative_matrix_conversion_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_QCOM_cooperative_matrix_conversion");
    }

    #[test]
    fn untyped_pointers_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_KHR_untyped_pointers");
    }

    #[test]
    fn subgroup_uniform_control_flow_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_KHR_subgroup_uniform_control_flow");
    }

    #[test]
    fn nv_fragment_shader_barycentric_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_NV_fragment_shader_barycentric");
    }

    #[test]
    fn workgroup_memory_explicit_layout_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_KHR_workgroup_memory_explicit_layout");
    }

    #[test]
    fn physical_storage_buffer_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_KHR_physical_storage_buffer");
    }

    #[test]
    fn shader_clock_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability ShaderClockKHR",
            "OpExtension \"SPV_KHR_shader_clock\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_6,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("ShaderClockKHR should be rejected when its extension is disallowed");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_KHR_shader_clock"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("ShaderClockKHR should be accepted for Vulkan targets");
    }

    #[test]
    fn tile_shading_capability_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability TileShadingQCOM",
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
            .validate(TargetEnv::Vulkan1_3)
            .expect_err("TileShadingQCOM requires its enabling extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::TileShadingQCOM,
                required_extension: "SPV_QCOM_tile_shading".to_string()
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability TileShadingQCOM",
            "OpExtension \"SPV_QCOM_tile_shading\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let validated = with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_3)
            .expect("capability should be accepted with required extension");
        assert_eq!(validated.header().schema(), Schema::ZERO);
    }

    #[test]
    fn universal_rejects_tile_shading_extension() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_QCOM_tile_shading\"",
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
            .expect_err("Tile shading extension should be Vulkan-only");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_QCOM_tile_shading"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn tile_shading_extension_requires_spirv_1_6() {
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_QCOM_tile_shading\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text.as_str().validate(TargetEnv::Universal1_6).expect_err(
            "tile shading extension should require SPIR-V 1.6 or be disallowed in Universal",
        );
        match error {
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(extension, ExtensionName::from("SPV_QCOM_tile_shading"));
                assert_eq!(required_version, SpirvVersion::new(1, 6));
                assert_eq!(target_version, TargetEnv::Universal1_6.spirv_version());
            }
            ValidationError::DisallowedExtension { extension, env } => {
                assert_eq!(extension, ExtensionName::from("SPV_QCOM_tile_shading"));
                assert_eq!(env, TargetEnv::Universal1_6);
            }
            other => panic!("unexpected error: {other:?}"),
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_4)
            .expect("extension should be accepted with SPIR-V 1.6+");
    }

    fn module_with_extension(extension: &str) -> String {
        module_with_extension_custom(
            extension,
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
        )
    }

    fn opencl_module_with_extension(extension: &str) -> String {
        module_with_extension_custom(
            extension,
            "OpCapability Kernel",
            "OpMemoryModel Logical OpenCL",
        )
    }

    fn module_with_extension_custom(
        extension: &str,
        capability: &str,
        memory_model: &str,
    ) -> String {
        [
            capability,
            &format!("OpExtension \"{extension}\""),
            memory_model,
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n")
    }

    #[test]
    fn nvx_extensions_are_vulkan_only() {
        let text = module_with_extension("SPV_NVX_multiview_per_view_attributes");
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("NVX extensions should be accepted for Vulkan targets");

        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("NVX extensions are Vulkan-only");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NVX_multiview_per_view_attributes"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn amdx_extensions_are_vulkan_only() {
        let text = module_with_extension("SPV_AMDX_shader_enqueue");
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("AMDX extensions should be accepted for Vulkan targets");

        let error = text
            .as_str()
            .validate(TargetEnv::OpenCl2_2)
            .expect_err("AMDX extensions are Vulkan-only");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_AMDX_shader_enqueue"),
                env: TargetEnv::OpenCl2_2
            }
        );
    }

    #[test]
    fn arm_extensions_are_vulkan_only() {
        let text = module_with_extension("SPV_ARM_graph");
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("ARM extensions should be accepted for Vulkan targets");

        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("ARM extensions are Vulkan-only");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_ARM_graph"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn altera_extensions_reject_vulkan() {
        let text = opencl_module_with_extension("SPV_ALTERA_fpga_memory_attributes");
        text.as_str()
            .validate(TargetEnv::OpenCl2_2)
            .expect("ALTERA extensions are permitted for OpenCL targets");
        text.as_str()
            .validate(TargetEnv::Universal1_5)
            .expect("ALTERA extensions are permitted for universal targets");

        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("ALTERA extensions should be rejected for Vulkan");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_ALTERA_fpga_memory_attributes"),
                env: TargetEnv::Vulkan1_2
            }
        );
    }

    #[test]
    fn universal_allows_google_and_amd_extensions() {
        let google = module_with_extension("SPV_GOOGLE_decorate_string");
        google
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("GOOGLE extensions should be allowed for universal environments");

        let amd = module_with_extension("SPV_AMD_shader_ballot");
        amd.as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("AMD extensions should be allowed for universal environments");
    }

    #[test]
    fn opencl_rejects_google_and_amd_vendor_extensions() {
        let google = opencl_module_with_extension("SPV_GOOGLE_decorate_string");
        let error = google
            .as_str()
            .validate(TargetEnv::OpenCl2_1)
            .expect_err("GOOGLE vendor extensions are not permitted for OpenCL targets");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_GOOGLE_decorate_string"),
                env: TargetEnv::OpenCl2_1
            }
        );

        let amd = opencl_module_with_extension("SPV_AMD_shader_ballot");
        let error = amd
            .as_str()
            .validate(TargetEnv::OpenCl2_1)
            .expect_err("AMD vendor extensions are not permitted for OpenCL targets");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_AMD_shader_ballot"),
                env: TargetEnv::OpenCl2_1
            }
        );
    }

    #[test]
    fn opengl_rejects_google_vendor_extension() {
        let text = module_with_extension("SPV_GOOGLE_decorate_string");
        let error = text
            .as_str()
            .validate(TargetEnv::OpenGl4_5)
            .expect_err("OpenGL should reject GOOGLE vendor extensions");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_GOOGLE_decorate_string"),
                env: TargetEnv::OpenGl4_5
            }
        );
    }

    #[test]
    fn opengl_rejects_amd_vendor_extension() {
        let text = module_with_extension("SPV_AMD_shader_trinary_minmax");
        let error = text
            .as_str()
            .validate(TargetEnv::OpenGl4_5)
            .expect_err("OpenGL should reject AMD vendor extensions");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_AMD_shader_trinary_minmax"),
                env: TargetEnv::OpenGl4_5
            }
        );
    }

    #[test]
    fn vulkan_memory_model_extension_is_vulkan_only() {
        let text = module_with_extension("SPV_KHR_vulkan_memory_model");
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("Vulkan memory model should be accepted for Vulkan targets");

        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect_err("Vulkan memory model should be rejected for non-Vulkan targets");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn intel_function_variants_allowed_for_opencl_and_universal_only() {
        let opencl_text = opencl_module_with_extension("SPV_INTEL_function_variants");
        opencl_text
            .as_str()
            .validate(TargetEnv::OpenCl2_2)
            .expect("INTEL function variants should be accepted for OpenCL targets");

        let universal_text = module_with_extension("SPV_INTEL_function_variants");
        universal_text
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("INTEL function variants should be accepted for universal targets");

        for env in [TargetEnv::Vulkan1_2, TargetEnv::OpenGl4_5] {
            let error = universal_text
                .as_str()
                .validate(env)
                .expect_err("INTEL function variants should be rejected outside OpenCL/Universal");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_INTEL_function_variants"),
                    env
                }
            );
        }
    }

    #[test]
    fn mesh_shader_extension_is_vulkan_only() {
        let text = module_with_extension("SPV_EXT_mesh_shader");
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("Mesh shader extension should be accepted for Vulkan targets");

        for env in [
            TargetEnv::OpenCl2_2,
            TargetEnv::Universal1_6,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("Mesh shader extension should be rejected outside Vulkan");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_EXT_mesh_shader"),
                    env
                }
            );
        }
    }

    fn assert_vulkan_only_extension(name: &str) {
        let text = module_with_extension(name);
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .unwrap_or_else(|_| panic!("{name} should be accepted for Vulkan targets"));

        for env in [
            TargetEnv::OpenCl2_2,
            TargetEnv::Universal1_6,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("{name} should be rejected outside Vulkan");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from(name),
                    env
                }
            );
        }
    }

    #[test]
    fn descriptor_indexing_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_EXT_descriptor_indexing");
    }

    #[test]
    fn fragment_shader_interlock_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_EXT_fragment_shader_interlock");
    }

    #[test]
    fn fragment_invocation_density_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_EXT_fragment_invocation_density");
    }

    #[test]
    fn shader_atomic_float_min_max_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_EXT_shader_atomic_float_min_max");
    }

    #[test]
    fn shader_invocation_reorder_ext_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_EXT_shader_invocation_reorder");
    }

    #[test]
    fn shader_atomic_float_add_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_EXT_shader_atomic_float_add");
    }

    #[test]
    fn qcom_image_processing_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_QCOM_image_processing");
    }

    #[test]
    fn opencl_environment_accepts_opencl_extension() {
        let text = opencl_module_with_extension("SPV_KHR_opencl_enqueue");
        text.validate(TargetEnv::OpenCl2_2)
            .expect("OpenCL targets should accept OpenCL-specific extensions");
    }

    #[test]
    fn validate_module_rejects_duplicate_extension() {
        // Hand-assemble a module with duplicate OpExtension instructions.
        let extension_word = 0x0008_000a; // word count 8, opcode OpExtension (10)
        let extension_words = [
            0x5f56_5053, // "SPV_"
            0x474f_4f47, // "GOOG"
            0x645f_454c, // "LE_d"
            0x726f_6365, // "ecor"
            0x5f65_7461, // "ate_"
            0x6972_7473, // "stri"
            0x0000_676e, // "ng\0"
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
            extension_words[5],
            extension_words[6],
            extension_word, // duplicate extension
            extension_words[0],
            extension_words[1],
            extension_words[2],
            extension_words[3],
            extension_words[4],
            extension_words[5],
            extension_words[6],
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
                extension: ExtensionName::from("SPV_GOOGLE_decorate_string")
            }
        );
    }

    #[test]
    fn validate_module_rejects_duplicate_conditional_extension() {
        // Duplicate OpConditionalExtensionINTEL instructions should be rejected.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            3,          // bound (ids up to 2)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string"
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
            op(8, 6248), // duplicate
            0x5f56_5053,
            0x474f_4f47,
            0x645f_454c,
            0x726f_6365,
            0x5f65_7461,
            0x6972_7473,
            0x0000_676e,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];

        let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DuplicateExtension {
                extension: ExtensionName::from("GOOGLE_decorate_string")
            }
        );
    }

    #[test]
    fn conditional_extension_rejected_in_non_vulkan_env() {
        // Vulkan-only conditional extensions must be rejected for non-Vulkan targets.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            6,          // bound (ids up to 5)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            0x0009_1868, // OpConditionalExtensionINTEL %1 "SPV_KHR_vulkan_memory_model"
            1,           // condition id (non-zero to satisfy parsing)
            0x5f56_5053, // "SPV_"
            0x5f52_484b, // "KHR_"
            0x6b6c_7576, // "vulk"
            0x6d5f_6e61, // "an_m"
            0x726f_6d65, // "emor"
            0x6f6d_5f79, // "y_mo"
            0x006c_6564, // "del\0"
            op(3, 14),   // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %2
            2,
            op(3, 33), // OpTypeFunction %3 %2
            3,
            2,
            op(5, 54), // OpFunction %2 %4 None %3
            2,
            4,
            0,
            3,
            op(2, 248), // OpLabel %5
            5,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
                env: TargetEnv::Universal1_6
            }
        );
    }

    #[test]
    fn entry_point_cannot_precede_memory_model() {
        // Entry points are mode-setting instructions that must follow the memory model.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            4,          // bound (ids up to 3)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(5, 15), // OpEntryPoint Vertex %1 "main"
            rspirv::spirv::ExecutionModel::Vertex as u32,
            1,
            0x6e69_616d, // "main"
            0,           // string padding
            op(3, 14),   // OpMemoryModel Logical GLSL450 (misordered after entry point)
            0,
            1,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_5).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemoryModel
            }
        );
    }

    #[test]
    fn execution_mode_cannot_precede_memory_model() {
        // Execution modes must follow the memory model stage.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            2,          // bound (ids up to 1)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            op(6, 16), // OpExecutionMode %1 LocalSize 1 1 1 (misordered before memory model)
            1,
            rspirv::spirv::ExecutionMode::LocalSize as u32,
            1,
            1,
            1,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];

        let error = validate_module(&binary, TargetEnv::Universal1_5).unwrap_err();
        assert_eq!(
            error,
            ValidationError::LayoutOutOfOrder {
                opcode: rspirv::spirv::Op::MemoryModel
            }
        );
    }

    #[test]
    fn conditional_extension_rejected_in_webgpu() {
        // WebGPU forbids all extensions, including conditional ones.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            6,          // bound (ids up to 5)
            0,          // schema
            op(2, 17),  // OpCapability Shader
            rspirv::spirv::Capability::Shader as u32,
            0x0008_1868, // OpConditionalExtensionINTEL %1 "SPV_KHR_shader_clock"
            1,           // condition id
            0x5f56_5053, // "SPV_"
            0x5f52_484b, // "KHR_"
            0x6461_6873, // "shad"
            0x635f_7265, // "er_c"
            0x6b63_6f6c, // "lock"
            0x0000_0000, // null terminator padding
            op(3, 14),   // OpMemoryModel Logical GLSL450
            0,
            1,
            op(2, 19), // OpTypeVoid %2
            2,
            op(3, 33), // OpTypeFunction %3 %2
            3,
            2,
            op(5, 54), // OpFunction %2 %4 None %3
            2,
            4,
            0,
            3,
            op(2, 248), // OpLabel %5
            5,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
        ];

        let error = validate_module(&binary, TargetEnv::WebGpu0).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_shader_clock"),
                env: TargetEnv::WebGpu0
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
            .validate(TargetEnv::Vulkan1_0)
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
                assert_eq!(target_version, TargetEnv::Vulkan1_0.spirv_version());
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
                assert_eq!(target_version, TargetEnv::Vulkan1_0.spirv_version());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("succeeds on newer SPIR-V");
    }

    #[test]
    fn effective_spirv_version_clamps_to_env() {
        use super::effective_spirv_version;
        assert_eq!(
            effective_spirv_version(TargetEnv::Vulkan1_0, SpirvVersion::new(1, 3)),
            TargetEnv::Vulkan1_0.spirv_version()
        );
        assert_eq!(
            effective_spirv_version(TargetEnv::Vulkan1_3, SpirvVersion::new(1, 1)),
            SpirvVersion::new(1, 1)
        );
    }

    #[test]
    fn capability_version_check_respects_module_version() {
        use rspirv::{binary::Assemble, dr::Builder};
        let mut builder = Builder::new();
        builder.set_version(1, 0);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.capability(rspirv::spirv::Capability::DeviceGroup);
        builder.extension("SPV_KHR_device_group");
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        let void = builder.type_void();
        let fn_type = builder.type_function(void, std::iter::empty::<u32>());
        builder
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("module version 1.0 should reject DeviceGroup (needs 1.3)");
        match error {
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(extension, ExtensionName::from("SPV_KHR_device_group"));
                assert_eq!(required_version, SpirvVersion::new(1, 3));
                assert_eq!(target_version, TargetEnv::Vulkan1_0.spirv_version());
            }
            ValidationError::CapabilityRequiresSpirvVersion {
                capability,
                required_version,
                target_version,
            } => {
                assert_eq!(capability, rspirv::spirv::Capability::DeviceGroup);
                assert_eq!(required_version, SpirvVersion::new(1, 3));
                assert_eq!(target_version, TargetEnv::Vulkan1_0.spirv_version());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn capability_version_clamps_to_env_when_module_is_newer() {
        use rspirv::{binary::Assemble, dr::Builder};
        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.capability(rspirv::spirv::Capability::RayTracingKHR);
        builder.extension("SPV_KHR_ray_tracing");
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        let void = builder.type_void();
        let fn_type = builder.type_function(void, std::iter::empty::<u32>());
        builder
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();
        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("capability should be gated by env-clamped version");
        match error {
            ValidationError::CapabilityRequiresSpirvVersion {
                capability,
                required_version,
                target_version,
            } => {
                assert_eq!(capability, rspirv::spirv::Capability::RayTracingKHR);
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 0));
            }
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(extension, ExtensionName::from("SPV_KHR_ray_tracing"));
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 0));
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn capability_version_clamps_for_binary_modules() {
        // Binary declares SPIR-V 1.6 and RayTracingKHR capability; Vulkan 1.0 should clamp.
        let binary = vec![
            0x07230203, // magic
            SpirvVersion::new(1, 6).to_word(),
            0,         // generator
            1,         // bound
            0,         // schema
            op(2, 17), // OpCapability
            rspirv::spirv::Capability::RayTracingKHR as u32,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Vulkan1_0)
            .expect_err("RayTracingKHR requires SPIR-V 1.4+ and should clamp to env");
        match error {
            ValidationError::CapabilityRequiresSpirvVersion {
                capability,
                required_version,
                target_version,
            } => {
                assert_eq!(capability, rspirv::spirv::Capability::RayTracingKHR);
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 0));
            }
            ValidationError::DisallowedCapability { capability, env } => {
                assert_eq!(capability, rspirv::spirv::Capability::RayTracingKHR);
                assert_eq!(env, TargetEnv::Vulkan1_0);
            }
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(extension, ExtensionName::from("SPV_KHR_ray_tracing"));
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 0));
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn instruction_requires_spirv_version_from_grammar() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.extension("SPV_KHR_terminate_invocation");
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        let void = builder.type_void();
        let fn_type = builder.type_function(void, std::iter::empty::<u32>());
        builder
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.terminate_invocation().unwrap();
        builder.end_function().unwrap();

        let module = builder.module();
        assert!(
            module
                .extensions
                .iter()
                .any(|inst| super::extension_operand(inst)
                    == Some(ExtensionName::from("SPV_KHR_terminate_invocation"))),
            "extension must be declared for opcode that requires it"
        );

        let words = module.assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_5)
            .expect_err("OpTerminateInvocation should require SPIR-V 1.6");
        assert_eq!(
            error,
            ValidationError::InstructionRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::TerminateInvocation,
                required_version: SpirvVersion::new(1, 6),
                target_version: SpirvVersion::new(1, 5),
            }
        );
    }

    #[test]
    fn memory_model_vulkan_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.extension("SPV_KHR_vulkan_memory_model");
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Vulkan,
        );

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_1Spirv1_4)
            .expect_err("Vulkan memory model requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::MemoryModel,
                operand_index: 1,
                required_version: SpirvVersion::new(1, 5),
                target_version: TargetEnv::Vulkan1_1Spirv1_4.spirv_version(),
            }
        );
    }

    #[test]
    fn memory_semantics_make_visible_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
        let semantics =
            builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::MAKE_VISIBLE.bits());

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .control_barrier(workgroup_scope, workgroup_scope, semantics)
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("MakeVisible semantics requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ControlBarrier,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn memory_semantics_make_available_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
        let semantics =
            builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::MAKE_AVAILABLE.bits());

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .control_barrier(workgroup_scope, workgroup_scope, semantics)
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("MakeAvailable semantics requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ControlBarrier,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn memory_semantics_volatile_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
        let semantics =
            builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::VOLATILE.bits());

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .control_barrier(workgroup_scope, workgroup_scope, semantics)
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("Volatile semantics requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ControlBarrier,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn decoration_non_uniform_requires_spirv_1_5() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%bool = OpTypeBool",
            "%ptr = OpTypePointer Uniform %bool",
            "%var = OpVariable %ptr Uniform",
            "OpDecorate %var NonUniform",
        ]
        .join("\n");

        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_4)
            .expect_err("NonUniform decoration requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::Decorate,
                operand_index: 1,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn decoration_non_uniform_requires_capability() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%bool = OpTypeBool",
            "%ptr = OpTypePointer Uniform %bool",
            "%var = OpVariable %ptr Uniform",
            "OpDecorate %var NonUniform",
        ]
        .join("\n");

        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_5)
            .expect_err("NonUniform decoration requires ShaderNonUniform capability");
        assert_eq!(
            error,
            ValidationError::MissingOperandCapability {
                opcode: rspirv::spirv::Op::Decorate,
                operand_index: 1,
                required_capability: rspirv::spirv::Capability::ShaderNonUniform,
            }
        );
    }

    #[test]
    fn image_operands_make_texel_visible_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let int = builder.type_int(32, 0);
        let float = builder.type_float(32, None);
        let v2int = builder.type_vector(int, 2);
        let v4float = builder.type_vector(float, 4);
        let image = builder.type_image(
            float,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            2,
            rspirv::spirv::ImageFormat::Rgba32f,
            None,
        );
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let int_0 = builder.constant_bit32(int, 0);
        let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
        let coord = builder.constant_composite(v2int, [int_0, int_0]);
        let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
        let img = builder.variable(
            ptr,
            None,
            rspirv::spirv::StorageClass::UniformConstant,
            None,
        );

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        let semantics =
            builder.constant_bit32(int, rspirv::spirv::MemorySemantics::MAKE_VISIBLE.bits());
        builder
            .image_write(
                img,
                coord,
                texel,
                Some(rspirv::spirv::ImageOperands::MAKE_TEXEL_VISIBLE),
                [rspirv::dr::Operand::IdRef(semantics)],
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("MakeTexelVisible image operand requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ImageWrite,
                operand_index: 3,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn image_operands_nontemporal_requires_spirv_1_6() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 5);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let int = builder.type_int(32, 0);
        let float = builder.type_float(32, None);
        let v2int = builder.type_vector(int, 2);
        let v4float = builder.type_vector(float, 4);
        let image = builder.type_image(
            float,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            2,
            rspirv::spirv::ImageFormat::Rgba32f,
            None,
        );
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let int_0 = builder.constant_bit32(int, 0);
        let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
        let coord = builder.constant_composite(v2int, [int_0, int_0]);
        let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
        let img = builder.variable(
            ptr,
            None,
            rspirv::spirv::StorageClass::UniformConstant,
            None,
        );

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_write(
                img,
                coord,
                texel,
                Some(rspirv::spirv::ImageOperands::NONTEMPORAL),
                std::iter::empty::<rspirv::dr::Operand>(),
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_5)
            .expect_err("Nontemporal image operand requires SPIR-V 1.6");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ImageWrite,
                operand_index: 3,
                required_version: SpirvVersion::new(1, 6),
                target_version: SpirvVersion::new(1, 5),
            }
        );
    }

    #[test]
    fn image_operands_make_texel_available_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let int = builder.type_int(32, 0);
        let float = builder.type_float(32, None);
        let v2int = builder.type_vector(int, 2);
        let v4float = builder.type_vector(float, 4);
        let image = builder.type_image(
            float,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            2,
            rspirv::spirv::ImageFormat::Rgba32f,
            None,
        );
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let int_0 = builder.constant_bit32(int, 0);
        let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
        let coord = builder.constant_composite(v2int, [int_0, int_0]);
        let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
        let img = builder.variable(
            ptr,
            None,
            rspirv::spirv::StorageClass::UniformConstant,
            None,
        );
        let semantics =
            builder.constant_bit32(int, rspirv::spirv::MemorySemantics::MAKE_AVAILABLE.bits());

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_write(
                img,
                coord,
                texel,
                Some(rspirv::spirv::ImageOperands::MAKE_TEXEL_AVAILABLE),
                [rspirv::dr::Operand::IdRef(semantics)],
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("MakeTexelAvailable image operand requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ImageWrite,
                operand_index: 3,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn image_operands_sign_extend_requires_spirv_1_4() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 3);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let int = builder.type_int(32, 0);
        let float = builder.type_float(32, None);
        let v2int = builder.type_vector(int, 2);
        let v4float = builder.type_vector(float, 4);
        let image = builder.type_image(
            float,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            2,
            rspirv::spirv::ImageFormat::Rgba32f,
            None,
        );
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let int_0 = builder.constant_bit32(int, 0);
        let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
        let coord = builder.constant_composite(v2int, [int_0, int_0]);
        let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
        let img = builder.variable(
            ptr,
            None,
            rspirv::spirv::StorageClass::UniformConstant,
            None,
        );

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_write(
                img,
                coord,
                texel,
                Some(rspirv::spirv::ImageOperands::SIGN_EXTEND),
                std::iter::empty::<rspirv::dr::Operand>(),
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_3)
            .expect_err("SignExtend image operand requires SPIR-V 1.4");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ImageWrite,
                operand_index: 3,
                required_version: SpirvVersion::new(1, 4),
                target_version: SpirvVersion::new(1, 3),
            }
        );
    }

    #[test]
    fn image_operands_zero_extend_requires_spirv_1_4() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 3);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let int = builder.type_int(32, 0);
        let float = builder.type_float(32, None);
        let v2int = builder.type_vector(int, 2);
        let v4float = builder.type_vector(float, 4);
        let image = builder.type_image(
            float,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            2,
            rspirv::spirv::ImageFormat::Rgba32f,
            None,
        );
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let int_0 = builder.constant_bit32(int, 0);
        let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
        let coord = builder.constant_composite(v2int, [int_0, int_0]);
        let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
        let img = builder.variable(
            ptr,
            None,
            rspirv::spirv::StorageClass::UniformConstant,
            None,
        );

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_write(
                img,
                coord,
                texel,
                Some(rspirv::spirv::ImageOperands::ZERO_EXTEND),
                std::iter::empty::<rspirv::dr::Operand>(),
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_3)
            .expect_err("ZeroExtend image operand requires SPIR-V 1.4");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ImageWrite,
                operand_index: 3,
                required_version: SpirvVersion::new(1, 4),
                target_version: SpirvVersion::new(1, 3),
            }
        );
    }

    #[test]
    fn image_operands_non_private_texel_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let int = builder.type_int(32, 0);
        let float = builder.type_float(32, None);
        let v2int = builder.type_vector(int, 2);
        let v4float = builder.type_vector(float, 4);
        let image = builder.type_image(
            float,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            2,
            rspirv::spirv::ImageFormat::Rgba32f,
            None,
        );
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let int_0 = builder.constant_bit32(int, 0);
        let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
        let coord = builder.constant_composite(v2int, [int_0, int_0]);
        let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
        let img = builder.variable(
            ptr,
            None,
            rspirv::spirv::StorageClass::UniformConstant,
            None,
        );

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_write(
                img,
                coord,
                texel,
                Some(rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL),
                std::iter::empty::<rspirv::dr::Operand>(),
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("NonPrivateTexel image operand requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ImageWrite,
                operand_index: 3,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn image_operands_volatile_texel_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let int = builder.type_int(32, 0);
        let float = builder.type_float(32, None);
        let v2int = builder.type_vector(int, 2);
        let v4float = builder.type_vector(float, 4);
        let image = builder.type_image(
            float,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            2,
            rspirv::spirv::ImageFormat::Rgba32f,
            None,
        );
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let int_0 = builder.constant_bit32(int, 0);
        let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
        let coord = builder.constant_composite(v2int, [int_0, int_0]);
        let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
        let img = builder.variable(
            ptr,
            None,
            rspirv::spirv::StorageClass::UniformConstant,
            None,
        );

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .image_write(
                img,
                coord,
                texel,
                Some(rspirv::spirv::ImageOperands::VOLATILE_TEXEL),
                std::iter::empty::<rspirv::dr::Operand>(),
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("VolatileTexel image operand requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ImageWrite,
                operand_index: 3,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn memory_semantics_output_memory_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
        let semantics =
            builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::OUTPUT_MEMORY.bits());

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .control_barrier(workgroup_scope, workgroup_scope, semantics)
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("OutputMemory semantics requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ControlBarrier,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn memory_access_make_pointer_visible_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let value = builder.constant_bit32(uint, 0);
        let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
        builder
            .store(
                var,
                value,
                Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_VISIBLE),
                [rspirv::dr::Operand::IdScope(scope)],
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("MakePointerVisible memory access requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::Store,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn memory_access_make_pointer_available_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let value = builder.constant_bit32(uint, 0);
        let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
        builder
            .store(
                var,
                value,
                Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_AVAILABLE),
                [rspirv::dr::Operand::IdScope(scope)],
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("MakePointerAvailable memory access requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::Store,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn memory_access_non_private_pointer_requires_spirv_1_5() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 4);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let value = builder.constant_bit32(uint, 0);
        let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .store(
                var,
                value,
                Some(rspirv::spirv::MemoryAccess::NON_PRIVATE_POINTER),
                std::iter::empty::<rspirv::dr::Operand>(),
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_4)
            .expect_err("NonPrivatePointer memory access requires SPIR-V 1.5");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::Store,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 5),
                target_version: SpirvVersion::new(1, 4),
            }
        );
    }

    #[test]
    fn memory_access_nontemporal_requires_spirv_1_6() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 5);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let value = builder.constant_bit32(uint, 0);
        let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);

        builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder
            .store(
                var,
                value,
                Some(rspirv::spirv::MemoryAccess::NONTEMPORAL),
                std::iter::empty::<rspirv::dr::Operand>(),
            )
            .unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_5)
            .expect_err("NonTemporal memory access requires SPIR-V 1.6");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::Store,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 6),
                target_version: SpirvVersion::new(1, 5),
            }
        );
    }

    #[test]
    fn storage_buffer_requires_spirv_1_3() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%int = OpTypeInt 32 0",
            "%ptr = OpTypePointer StorageBuffer %int",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "%var = OpVariable %ptr StorageBuffer",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_2)
            .expect_err("StorageBuffer storage class requires SPIR-V 1.3");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::TypePointer,
                operand_index: 0,
                required_version: SpirvVersion::new(1, 3),
                target_version: SpirvVersion::new(1, 2),
            }
        );
    }

    #[test]
    fn loop_control_dependency_length_requires_spirv_1_1() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%bool = OpTypeBool",
            "%fn = OpTypeFunction %void",
            "%true = OpConstantTrue %bool",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpBranch %loop",
            "%loop = OpLabel",
            "OpLoopMerge %merge %continue DependencyLength 1",
            "OpBranch %continue",
            "%continue = OpLabel",
            "OpBranchConditional %true %loop %merge",
            "%merge = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let error = text
            .as_str()
            .validate(TargetEnv::Universal1_0)
            .expect_err("DependencyLength loop control requires SPIR-V 1.1");
        assert_eq!(
            error,
            ValidationError::OperandRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::LoopMerge,
                operand_index: 2,
                required_version: SpirvVersion::new(1, 1),
                target_version: SpirvVersion::new(1, 0),
            }
        );
    }

    #[test]
    fn execution_mode_id_requires_spirv_1_2() {
        use rspirv::{binary::Assemble, dr::Builder};

        let mut builder = Builder::new();
        builder.set_version(1, 2);
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );

        let void = builder.type_void();
        let uint = builder.type_int(32, 0);
        let function_type = builder.type_function(void, std::iter::empty::<u32>());
        let local_size_x = builder.constant_bit32(uint, 1);
        let local_size_y = builder.constant_bit32(uint, 1);
        let local_size_z = builder.constant_bit32(uint, 1);

        let entry_point = builder
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::NONE,
                function_type,
            )
            .unwrap();
        builder.begin_block(None).unwrap();
        builder.ret().unwrap();
        builder.end_function().unwrap();

        builder.entry_point(
            rspirv::spirv::ExecutionModel::Vertex,
            entry_point,
            "main",
            [],
        );
        builder.execution_mode_id(
            entry_point,
            rspirv::spirv::ExecutionMode::LocalSizeId,
            [local_size_x, local_size_y, local_size_z],
        );

        let words = builder.module().assemble();
        let error = words
            .as_slice()
            .validate(TargetEnv::Universal1_1)
            .expect_err("ExecutionModeId::LocalSizeId requires SPIR-V 1.2");
        assert_eq!(
            error,
            ValidationError::InstructionRequiresSpirvVersion {
                opcode: rspirv::spirv::Op::ExecutionModeId,
                required_version: SpirvVersion::new(1, 2),
                target_version: SpirvVersion::new(1, 1),
            },
        );
    }

    #[test]
    fn shader_clock_capability_requires_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability ShaderClockKHR",
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
            .validate(TargetEnv::Vulkan1_3)
            .expect_err("ShaderClockKHR requires declaring the extension");
        assert_eq!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension {
                capability: rspirv::spirv::Capability::ShaderClockKHR,
                required_extension: "SPV_KHR_shader_clock".to_string(),
            }
        );

        let with_extension = [
            "OpCapability Shader",
            "OpCapability ShaderClockKHR",
            "OpExtension \"SPV_KHR_shader_clock\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_extension
            .as_str()
            .validate(TargetEnv::Vulkan1_3)
            .expect("extension declared should satisfy capability");
    }

    #[test]
    fn device_group_capability_requires_spirv_1_3() {
        let text = [
            "OpCapability Shader",
            "OpCapability DeviceGroup",
            "OpExtension \"SPV_KHR_device_group\"",
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
            .expect_err("DeviceGroup requires SPIR-V 1.3");
        match error {
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(extension, ExtensionName::from("SPV_KHR_device_group"));
                assert_eq!(required_version, SpirvVersion::new(1, 3));
                assert_eq!(target_version, TargetEnv::Vulkan1_0.spirv_version());
            }
            ValidationError::CapabilityRequiresSpirvVersion {
                capability,
                required_version,
                target_version,
            } => {
                assert_eq!(capability, rspirv::spirv::Capability::DeviceGroup);
                assert_eq!(required_version, SpirvVersion::new(1, 3));
                assert_eq!(target_version, TargetEnv::Vulkan1_0.spirv_version());
            }
            other => panic!("unexpected error: {other:?}"),
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("DeviceGroup accepted on SPIR-V 1.3+");
    }

    #[test]
    fn device_group_extension_is_vulkan_only() {
        assert_vulkan_only_extension("SPV_KHR_device_group");
    }

    #[test]
    fn device_group_capability_rejected_outside_vulkan_even_with_extension() {
        let text = [
            "OpCapability Shader",
            "OpCapability DeviceGroup",
            "OpExtension \"SPV_KHR_device_group\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        for env in [
            TargetEnv::Universal1_5,
            TargetEnv::OpenCl2_2,
            TargetEnv::OpenGl4_5,
        ] {
            let error = text
                .as_str()
                .validate(env)
                .expect_err("DeviceGroup should be rejected when its extension is disallowed");
            assert_eq!(
                error,
                ValidationError::DisallowedExtension {
                    extension: ExtensionName::from("SPV_KHR_device_group"),
                    env
                }
            );
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_1)
            .expect("DeviceGroup should be accepted for Vulkan targets");
    }

    #[test]
    fn variable_pointers_requires_storage_buffer_capability() {
        let text = [
            "OpCapability Shader",
            "OpCapability VariablePointers",
            "OpExtension \"SPV_KHR_variable_pointers\"",
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
            .expect_err("VariablePointers requires VariablePointersStorageBuffer");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::VariablePointersStorageBuffer,
                capability: rspirv::spirv::Capability::VariablePointers,
            }
        );

        let with_dependency = [
            "OpCapability Shader",
            "OpCapability VariablePointersStorageBuffer",
            "OpCapability VariablePointers",
            "OpExtension \"SPV_KHR_variable_pointers\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_dependency
            .as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("dependency declared should satisfy requirement");
    }

    #[test]
    fn shader_capability_does_not_require_matrix_soft_dependency() {
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
        text.as_str()
            .validate(TargetEnv::Universal1_6)
            .expect("Shader should not require Matrix due to soft dependency");
    }

    #[test]
    fn image_buffer_requires_sampled_buffer_capability() {
        let text = [
            "OpCapability Shader",
            "OpCapability ImageBuffer",
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
            .expect_err("ImageBuffer requires SampledBuffer capability");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::SampledBuffer,
                capability: rspirv::spirv::Capability::ImageBuffer
            }
        );

        let with_dependency = [
            "OpCapability Shader",
            "OpCapability SampledBuffer",
            "OpCapability ImageBuffer",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_dependency
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("dependency declared should satisfy requirement");
    }

    #[test]
    fn sampled_cube_array_requires_shader_capability() {
        let text = [
            "OpCapability SampledCubeArray",
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
            .validate(TargetEnv::Universal1_2)
            .expect_err("SampledCubeArray requires Shader capability");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::Shader,
                capability: rspirv::spirv::Capability::SampledCubeArray
            }
        );

        let with_shader = [
            "OpCapability Shader",
            "OpCapability SampledCubeArray",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_shader
            .as_str()
            .validate(TargetEnv::Universal1_2)
            .expect("Shader capability declared should satisfy dependency");
    }

    #[test]
    fn image_ms_array_requires_shader_capability() {
        let text = [
            "OpCapability ImageMSArray",
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
            .validate(TargetEnv::Universal1_2)
            .expect_err("ImageMSArray requires Shader capability");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::Shader,
                capability: rspirv::spirv::Capability::ImageMSArray
            }
        );

        let with_shader = [
            "OpCapability Shader",
            "OpCapability ImageMSArray",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_shader
            .as_str()
            .validate(TargetEnv::Universal1_2)
            .expect("Shader capability declared should satisfy dependency");
    }

    #[test]
    fn ray_tracing_requires_shader_capability() {
        let text = [
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
        let error = text
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("RayTracingKHR requires Shader capability");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::Shader,
                capability: rspirv::spirv::Capability::RayTracingKHR
            }
        );

        let with_shader = [
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
        with_shader
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("Shader capability declared should satisfy dependency");
    }

    #[test]
    fn group_non_uniform_arithmetic_requires_group_non_uniform() {
        let text = [
            "OpCapability Shader",
            "OpCapability GroupNonUniformArithmetic",
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
            .expect_err("GroupNonUniformArithmetic requires GroupNonUniform");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::GroupNonUniform,
                capability: rspirv::spirv::Capability::GroupNonUniformArithmetic
            }
        );

        let with_base = [
            "OpCapability Shader",
            "OpCapability GroupNonUniform",
            "OpCapability GroupNonUniformArithmetic",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_base
            .as_str()
            .validate(TargetEnv::Vulkan1_2)
            .expect("base capability declared should satisfy dependency");
    }

    #[test]
    fn device_enqueue_requires_kernel() {
        let text = [
            "OpCapability DeviceEnqueue",
            "OpCapability Addresses",
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
            .expect_err("DeviceEnqueue requires Kernel capability");
        assert_eq!(
            error,
            ValidationError::MissingRequiredCapability {
                required_capability: rspirv::spirv::Capability::Kernel,
                capability: rspirv::spirv::Capability::DeviceEnqueue
            }
        );

        let with_kernel = [
            "OpCapability Kernel",
            "OpCapability Addresses",
            "OpCapability DeviceEnqueue",
            "OpMemoryModel Physical32 OpenCL",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        with_kernel
            .as_str()
            .validate(TargetEnv::OpenCl2_0)
            .expect("kernel capability enables device enqueue");
    }

    #[test]
    fn physical_storage_buffer_capability_requires_spirv_1_4() {
        let text = [
            "OpCapability Shader",
            "OpCapability PhysicalStorageBufferAddresses",
            "OpExtension \"SPV_KHR_physical_storage_buffer\"",
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
            .expect_err("requires SPIR-V 1.4");
        match error {
            ValidationError::CapabilityRequiresSpirvVersion {
                capability,
                required_version,
                target_version,
            } => {
                assert_eq!(
                    capability,
                    rspirv::spirv::Capability::PhysicalStorageBufferAddresses
                );
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 3));
            }
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(
                    extension,
                    ExtensionName::from("SPV_KHR_physical_storage_buffer")
                );
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 3));
            }
            ValidationError::DisallowedExtension { extension, env } => {
                assert_eq!(
                    extension,
                    ExtensionName::from("SPV_KHR_physical_storage_buffer")
                );
                assert_eq!(env, TargetEnv::Universal1_3);
            }
            other => panic!("unexpected error: {other:?}"),
        }

        text.as_str()
            .validate(TargetEnv::Vulkan1_2)
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
        match error {
            ValidationError::InstructionRequiresSpirvVersion {
                opcode,
                required_version,
                target_version,
            } => {
                assert_eq!(opcode, rspirv::spirv::Op::TypeRayQueryKHR);
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 3));
            }
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(extension, ExtensionName::from("SPV_KHR_ray_query"));
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 3));
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn instruction_version_clamps_to_env_when_module_is_newer() {
        use rspirv::{binary::Assemble, dr::Builder};
        let mut builder = Builder::new();
        builder.set_version(1, 6);
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
            .validate(TargetEnv::Vulkan1_0)
            .expect_err("Ray query requires SPIR-V 1.4+, env should clamp module version to 1.0");
        match error {
            ValidationError::InstructionRequiresSpirvVersion {
                opcode,
                required_version,
                target_version,
            } => {
                assert_eq!(opcode, rspirv::spirv::Op::TypeRayQueryKHR);
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 0));
            }
            ValidationError::ExtensionRequiresSpirvVersion {
                extension,
                required_version,
                target_version,
            } => {
                assert_eq!(extension, ExtensionName::from("SPV_KHR_ray_query"));
                assert_eq!(required_version, SpirvVersion::new(1, 4));
                assert_eq!(target_version, SpirvVersion::new(1, 0));
            }
            other => panic!("unexpected error {other:?}"),
        }
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
    fn offset_requires_member_decorate() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%struct = OpTypeStruct %void",
            "OpDecorate %struct Offset 0",
        ]
        .join("\n");
        let error = MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::MemberOnlyDecorationUsedWithDecorate {
                decoration: rspirv::spirv::Decoration::Offset
            }
        );
    }

    #[test]
    fn matrix_stride_requires_member_decorate() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%mat2 = OpTypeMatrix %vec2 2",
            "OpDecorate %mat2 MatrixStride 8",
        ]
        .join("\n");
        let error = MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::MemberOnlyDecorationUsedWithDecorate {
                decoration: rspirv::spirv::Decoration::MatrixStride
            }
        );
    }

    #[test]
    fn row_major_requires_member_decorate() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%mat2 = OpTypeMatrix %vec2 2",
            "OpDecorate %mat2 RowMajor",
        ]
        .join("\n");
        let error = MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::MemberOnlyDecorationUsedWithDecorate {
                decoration: rspirv::spirv::Decoration::RowMajor
            }
        );
    }

    #[test]
    fn col_major_requires_member_decorate() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%mat2 = OpTypeMatrix %vec2 2",
            "OpDecorate %mat2 ColMajor",
        ]
        .join("\n");
        let error = MaybeValidModule::Text(&text)
            .validate(TargetEnv::Universal1_6)
            .unwrap_err();
        assert_eq!(
            error,
            ValidationError::MemberOnlyDecorationUsedWithDecorate {
                decoration: rspirv::spirv::Decoration::ColMajor
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
