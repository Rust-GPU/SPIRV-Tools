use std::collections::{HashMap, HashSet};
use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

use rspirv::dr::Module;
use thiserror::Error;

use crate::target_env::TargetEnv;

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
    fn insert(&mut self, capability: rspirv::spirv::Capability) -> Result<(), ValidationError> {
        if !self.values.insert(capability) {
            return Err(ValidationError::DuplicateCapability { capability });
        }
        Ok(())
    }
}

/// A set of declared extensions for a module.
#[derive(Debug, Default)]
struct ExtensionSet {
    values: HashSet<String>,
}

impl ExtensionSet {
    fn insert(&mut self, extension: &str) -> Result<(), ValidationError> {
        if !self.values.insert(extension.to_string()) {
            return Err(ValidationError::DuplicateExtension {
                extension: extension.to_string(),
            });
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
    /// Duplicate extension declarations were found.
    #[error("extension {extension} is declared more than once")]
    DuplicateExtension {
        /// The extension name that was duplicated.
        extension: String,
    },
    /// A decoration group reference did not point to a declared group.
    #[error("decoration group {group} is not declared")]
    UnknownDecorationGroup {
        /// The group id that was missing.
        group: Id,
    },
    /// A decoration targeted an id that does not exist in the module.
    #[error("decoration targets unknown id {target}")]
    UnknownDecorationTarget {
        /// The missing target id.
        target: Id,
    },
    /// A decoration targeted an id that is not valid for that opcode.
    #[error("decoration target {target} is not defined")]
    MissingDecorationTarget {
        /// The target id that is missing.
        target: Id,
    },
    /// A member decoration targeted an id that is not a struct type.
    #[error("member decorations must target struct types (found {target})")]
    MemberDecorationTargetNotStruct {
        /// The target id that was not a struct.
        target: DecorationTargetId,
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Section {
    Capabilities,
    MemoryModel,
    EntryAndModes,
    Debug,
    Names,
    Annotations,
    TypesGlobals,
    Functions,
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

fn validate_words(words: ModuleWords, env: TargetEnv) -> Result<ValidModule, ValidationError> {
    if let Some(&schema) = words.as_slice().get(4) {
        Schema::validate(schema)?;
    }
    run_layout_check(words.as_slice())?;
    let mut loader = rspirv::dr::Loader::new();
    if let Err(error) = rspirv::binary::parse_words(words.as_slice(), &mut loader) {
        return Err(ValidationError::Parse(error.to_string()));
    }
    let module = loader.module();
    let header = ValidatedHeader::from_module(&module)?;
    let defined_ids = validate_id_bound(&module, header)?;
    validate_memory_model(&module)?;
    validate_member_decorations(&module, &defined_ids)?;
    validate_decoration_groups(&module, &defined_ids)?;
    validate_decorations(&module, &defined_ids)?;
    Ok(ValidModule {
        words,
        module,
        env,
        header,
    })
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

fn run_layout_check(words: &[u32]) -> Result<(), ValidationError> {
    struct LayoutChecker {
        memory_model_state: MemoryModelState,
        current_section: Section,
        function_state: FunctionState,
        capabilities: CapabilitySet,
        extensions: ExtensionSet,
    }

    impl LayoutChecker {
        fn new() -> Self {
            Self {
                memory_model_state: MemoryModelState::new(),
                current_section: Section::Capabilities,
                function_state: FunctionState::Outside,
                capabilities: CapabilitySet::default(),
                extensions: ExtensionSet::default(),
            }
        }
    }

    impl rspirv::binary::Consumer for LayoutChecker {
        fn initialize(&mut self) -> rspirv::binary::ParseAction {
            rspirv::binary::ParseAction::Continue
        }

        fn finalize(&mut self) -> rspirv::binary::ParseAction {
            match self.memory_model_state.finalize() {
                Ok(()) => rspirv::binary::ParseAction::Continue,
                Err(err) => rspirv::binary::ParseAction::Error(Box::new(err)),
            }
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

            match inst.class.opcode {
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
                    self.current_section = self.current_section.max(Section::MemoryModel);
                }
                rspirv::spirv::Op::Capability => {
                    if let Some(cap) = capability_operand(&inst) {
                        if let Err(err) = self.capabilities.insert(cap) {
                            return rspirv::binary::ParseAction::Error(Box::new(err));
                        }
                    }
                }
                rspirv::spirv::Op::Extension => {
                    if let Some(extension) = extension_operand(&inst) {
                        if let Err(err) = self.extensions.insert(extension) {
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
                    self.current_section = self.current_section.max(Section::Functions);
                }
                opcode => {
                    let section = section_index(opcode);
                    if section < self.current_section {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::LayoutOutOfOrder { opcode },
                        ));
                    }
                    self.current_section = self.current_section.max(section);
                    if section > Section::MemoryModel && !self.memory_model_state.is_seen() {
                        self.memory_model_state.record_violation(opcode);
                        return rspirv::binary::ParseAction::Continue;
                    }
                }
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

fn section_index(opcode: rspirv::spirv::Op) -> Section {
    use rspirv::spirv::Op::*;
    match opcode {
        Capability | Extension | ExtInstImport => Section::Capabilities,
        MemoryModel => Section::MemoryModel,
        EntryPoint | ExecutionMode | ExecutionModeId => Section::EntryAndModes,
        String | SourceExtension | Source | SourceContinued | ModuleProcessed => Section::Debug,
        Name | MemberName => Section::Names,
        Decorate | DecorateId | MemberDecorate | DecorateString | MemberDecorateString
        | DecorationGroup | GroupDecorate | GroupMemberDecorate => Section::Annotations,
        Function => Section::Functions,
        _ => Section::TypesGlobals,
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

fn extension_operand(inst: &rspirv::dr::Instruction) -> Option<&str> {
    inst.operands.iter().find_map(|operand| {
        if let rspirv::dr::Operand::LiteralString(extension) = operand {
            Some(extension.as_str())
        } else {
            None
        }
    })
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
) -> Result<(), ValidationError> {
    let mut types: HashMap<ResultId, rspirv::spirv::Op> = HashMap::new();
    for inst in &module.types_global_values {
        if let Some(result_id) = inst.result_id {
            let id = ResultId::try_from(result_id).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::Result,
                opcode: inst.class.opcode,
            })?;
            types.insert(id, inst.class.opcode);
        }
    }

    for inst in &module.annotations {
        match inst.class.opcode {
            rspirv::spirv::Op::MemberDecorate | rspirv::spirv::Op::MemberDecorateString => {
                let mut operands = inst.operands.iter();
                let target = operands.find_map(|op| {
                    if let rspirv::dr::Operand::IdRef(id) = op {
                        DecorationTargetId::try_from(*id).ok()
                    } else {
                        None
                    }
                });
                if let Some(target) = target {
                    let target_id =
                        ResultId::try_from(u32::from(Id::from(target))).map_err(|_| {
                            ValidationError::ZeroId {
                                kind: IdKind::Operand,
                                opcode: inst.class.opcode,
                            }
                        })?;
                    if !defined_ids.contains(&target_id) {
                        return Err(ValidationError::UnknownDecorationTarget {
                            target: target.into(),
                        });
                    }
                    let target_id =
                        ResultId::try_from(u32::from(Id::from(target))).map_err(|_| {
                            ValidationError::ZeroId {
                                kind: IdKind::Operand,
                                opcode: inst.class.opcode,
                            }
                        })?;
                    let op = types.get(&target_id).copied();
                    if op != Some(rspirv::spirv::Op::TypeStruct) {
                        return Err(ValidationError::MemberDecorationTargetNotStruct { target });
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn validate_decoration_groups(
    module: &Module,
    defined_ids: &HashSet<ResultId>,
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
                for operand in inst.operands.iter().skip(1) {
                    if let rspirv::dr::Operand::IdRef(id) = operand {
                        let target =
                            ResultId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                                kind: IdKind::Operand,
                                opcode: inst.class.opcode,
                            })?;
                        if !defined_ids.contains(&target) {
                            return Err(ValidationError::UnknownDecorationTarget {
                                target: target.into_inner(),
                            });
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
        validate_module, CheckedBound, DeclaredBound, DecorationTargetId, Id, IdKind,
        MaybeValidModule, ModuleWords, Schema, ValidatableModule, ValidationError,
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
        // Manually build a module where OpMemoryModel appears after the function body.
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
        // Build a minimal module with two memory model instructions.
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
    fn validate_module_rejects_duplicate_capability() {
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
                extension: "SPV_KHR_ray_tracing".to_string()
            }
        );
    }

    #[test]
    fn validate_module_rejects_zero_result_id() {
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
                target: DecorationTargetId::try_from(1).unwrap()
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
