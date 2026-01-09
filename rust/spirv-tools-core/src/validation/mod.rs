use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::Arc,
};

use rspirv::dr::Module;

use crate::{target_env::TargetEnv, version::SpirvVersion};

// Core infrastructure modules
pub(crate) mod capability_info;
pub mod error;
pub use error::ValidationError;
mod instruction_classes;
use instruction_classes::{instruction_class, InstructionClass};
mod instruction_layout;
use instruction_layout::{is_capability_opcode, is_extension_opcode, mode_stage, ModeStage};
mod instruction_versions;
use instruction_versions::grammar_required_spirv_version_for_opcode;
mod operand_requirements;
use operand_requirements::{
    grammar_required_capabilities_for_operand, grammar_required_extensions_for_operand,
};
mod operand_versions;
use operand_versions::grammar_required_spirv_version_for_operand;

// Type definitions
pub mod types;
pub use types::{
    CheckedBound, DecorationTargetId, DecorationTargetKind, DeclaredBound, ExtensionName, Id,
    IdBound, IdKind, MemberDecorationTargetId, MemberIndex, MergeTargetKind, ModuleWords,
    OperandId, ResultId, Schema, TypeId, ZeroIdError,
};

// Shared helper utilities
pub mod helpers;

// Type extension traits for rspirv types
pub mod type_ext;
pub use type_ext::{TypeInstructionExt, TypeResolver, DefaultTypeResolver};

// Validation context and rule trait
pub mod context;
pub use context::{ValidationContext, ValidationRule, run_rules, TestContextData};

// Validation rules organized by category
pub mod rules;
use rules::limits::all_limit_rules;
use rules::extensions::{
    extension_operand, extension_satisfied, validate_extension_allowlist, validate_extensions,
    ExtensionSet,
};
use rules::capabilities::{
    capability_operand, capability_satisfied, required_extension_for_capability,
    validate_capabilities,
};
use helpers::{
    build_decoration_lookup, collect_declared_capabilities, collect_execution_models,
    collect_result_instructions, collect_result_opcodes, collect_result_types,
    constant_u32_from_defs, is_memory_object_declaration, is_vulkan_env, matrix_info, vector_info,
};

/// A validated module header with a checked bound and schema.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ValidatedHeader {
    version: SpirvVersion,
    bound: CheckedBound,
    schema: Schema,
}

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
        (
            ValidationError::EntryPointInterfaceStorageClassInvalid {
                entry_point,
                interface,
                ..
            },
            Some(names),
        ) => format!(
            "{} interface {}",
            names.format_id((*entry_point).into()),
            names.format_id((*interface).into())
        ),
        (
            ValidationError::EntryPointInterfaceStorageClassDuplicate {
                entry_point,
                storage_class,
            },
            Some(names),
        ) => format!(
            "{} has duplicate {storage_class:?} interfaces",
            names.format_id((*entry_point).into())
        ),
        (
            ValidationError::EntryPointInterfaceLocationConflict {
                entry_point,
                storage_class,
                location,
                component,
            },
            Some(names),
        ) => format!(
            "{} {storage_class:?} location {location} component {component}",
            names.format_id((*entry_point).into())
        ),
        (
            ValidationError::ExecutionModeRequiresExecutionModel {
                entry_point,
                mode,
                execution_model,
                allowed_models,
            },
            Some(names),
        ) => format!(
            "{} uses {execution_model:?} with mode {mode:?} (allowed: {:?})",
            names.format_id((*entry_point).into()),
            allowed_models
        ),
        (
            ValidationError::InvalidExecutionModeValue {
                entry_point,
                mode,
                value,
            },
            Some(names),
        ) => format!(
            "{} has {mode:?} value {value}",
            names.format_id((*entry_point).into())
        ),
        (
            ValidationError::EntryPointInterfaceFloatEncodingInvalid {
                interface,
                storage_class,
                encoding,
            },
            Some(names),
        ) => format!(
            "{} in {storage_class:?} uses {encoding:?}",
            names.format_id((*interface).into())
        ),
        (
            ValidationError::DuplicateEntryPoint {
                function,
                execution_model,
            },
            Some(names),
        ) => format!(
            "{} ({execution_model:?})",
            names.format_id((*function).into())
        ),
        (
            ValidationError::DuplicateEntryPointInterface {
                entry_point,
                interface,
            },
            Some(names),
        ) => format!(
            "{} duplicate interface {}",
            names.format_id((*entry_point).into()),
            names.format_id((*interface).into())
        ),
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
        (
            ValidationError::FunctionCallTargetNotFunction {
                function,
                target,
                ..
            },
            Some(names),
        ) => format!(
            "{} calls non-function {}",
            names.format_id((*function).into()),
            names.format_id((*target).into())
        ),
        (
            ValidationError::MergeTargetMissing {
                function, target, ..
            },
            Some(names),
        ) => format!(
            "{} missing block {}",
            names.format_id((*function).into()),
            names.format_id((*target).into())
        ),
        (
            ValidationError::MergeInstructionNotBeforeTerminator {
                function, block, ..
            },
            Some(names),
        )
        | (
            ValidationError::InvalidMergeTerminator {
                function, block, ..
            },
            Some(names),
        )
        | (ValidationError::DuplicateMergeInstruction { function, block }, Some(names))
        | (
            ValidationError::ContinueTargetMatchesMerge {
                function, block, ..
            },
            Some(names),
        )
        | (
            ValidationError::MissingSelectionMerge {
                function, block, ..
            },
            Some(names),
        )
        | (
            ValidationError::PhiIncomingTypeMismatch {
                function, block, ..
            },
            Some(names),
        )
        | (
            ValidationError::ValueDefinedInAnotherFunction {
                function,
                value: block,
            },
            Some(names),
        )
        | (
            ValidationError::FunctionVariableStorageClassMismatch {
                function,
                variable: block,
                ..
            },
            Some(names),
        )
        | (
            ValidationError::FunctionVariableNotInEntryBlock {
                function,
                variable: block,
            },
            Some(names),
        ) => format!(
            "{} in block {}",
            names.format_id((*function).into()),
            names.format_id((*block).into())
        ),
        (
            ValidationError::MissingBlockLabel {
                function,
                block_index,
            },
            Some(names),
        ) => format!(
            "{} missing OpLabel in block {}",
            names.format_id((*function).into()),
            block_index
        ),
        (
            ValidationError::MergeTargetIsBlock {
                function,
                block,
                kind,
                target,
            },
            Some(names),
        ) => format!(
            "{} uses {:?} target {} equal to its block {}",
            names.format_id((*function).into()),
            kind,
            names.format_id((*target).into()),
            names.format_id((*block).into()),
        ),
        (
            ValidationError::ValueNotDominated {
                function,
                block,
                value,
            },
            Some(names),
        ) => format!(
            "{} uses value {} in block {} before its definition dominates",
            names.format_id((*function).into()),
            names.format_id((*value).into()),
            names.format_id((*block).into()),
        ),
        (
            ValidationError::PhiIncomingNotDominated {
                function,
                block,
                incoming,
                value,
            },
            Some(names),
        ) => format!(
            "{} uses incoming value {} for predecessor {} in block {} before its definition dominates",
            names.format_id((*function).into()),
            names.format_id((*value).into()),
            names.format_id((*incoming).into()),
            names.format_id((*block).into()),
        ),
        (
            ValidationError::UndefinedId { function, id },
            Some(names),
        ) => {
            let func = function
                .map(|f| format!(" in function {}", names.format_id(f.into())))
                .unwrap_or_default();
            format!("use of undefined id {}{}", names.format_id((*id).into()), func)
        }
        (
            ValidationError::ResultTypeNotType {
                instruction,
                result_type,
                found,
            },
            Some(names),
        ) => format!(
            "{:?} uses result type {} defined by non-type opcode {:?}",
            instruction,
            names.format_id((*result_type).into()),
            found
        ),
        (ValidationError::PhiAfterNonPhi { function, block }, Some(names)) => format!(
            "{} in block {}",
            names.format_id((*function).into()),
            names.format_id((*block).into())
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
    run_layout_check(words.as_slice(), env)?;
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
    let defined_result_ids = validate_id_bound(&module, header)?;
    let defined_ids: HashSet<Id> = defined_result_ids.iter().copied().map(Id::from).collect();
    let opcodes = collect_result_opcodes(&module);
    let definitions = collect_result_instructions(&module);
    validate_result_types_are_types(&definitions, &opcodes)?;
    let capabilities = collect_declared_capabilities(&module);
    let extensions = validate_extensions(&module, env, target_version)?;
    let declared_caps = validate_capabilities(&module, env, target_version, &extensions)?;
    validate_instruction_requirements(&module, target_version, &declared_caps, &extensions)?;
    validate_sampler_image_addressing_mode(&module, &capabilities)?;
    validate_memory_model(&module)?;
    validate_type_functions(&module, &opcodes)?;
    let struct_member_counts = validate_member_decorations(&module, &defined_result_ids)?;
    enforce_logical_pointer_rules(&module, &definitions, &capabilities, &options)?;
    validate_decoration_groups(
        &module,
        &defined_result_ids,
        &opcodes,
        &struct_member_counts,
    )?;
    validate_decorations(&module, &defined_result_ids)?;
    enforce_decoration_versions(&module, target_version)?;
    enforce_block_storage_classes(&module, target_version)?;
    enforce_descriptor_storage_classes(&module)?;
    enforce_descriptor_requirements(&module, env)?;
    let entry_models = collect_execution_models(&module);
    enforce_struct_block_requirements(&module, target_version)?;
    enforce_location_storage_classes(&module)?;
    enforce_builtin_location_exclusivity(&module)?;
    enforce_builtin_storage_classes(&module, &definitions, &entry_models, &capabilities, env)?;
    enforce_interpolation_exclusivity(&module, &definitions)?;
    enforce_interpolation_storage_classes(&module, &definitions, &entry_models, &capabilities)?;
    enforce_interpolation_entry_point_compatibility(&module, &definitions, env)?;
    validate_decoration_target_categories(&module, &opcodes, &definitions, &capabilities)?;
    enforce_store_type_compatibility(&module, &definitions, &options)?;
    let entry_points = validate_entry_points(&module, &defined_result_ids, &opcodes, &definitions)?;
    validate_entry_point_interface_storage_classes(&module, &definitions, &capabilities, env)?;
    validate_entry_point_locations(&module, &definitions, env)?;
    validate_execution_modes(&module, &entry_points, env, &options, &capabilities)?;
    validate_functions(&module, &capabilities)?;
    validate_operand_definitions(&module, &defined_ids)?;

    // Build validation context and run limit rules
    let result_types = collect_result_types(&module)?;
    let validation_ctx = ValidationContext {
        module: &module,
        env,
        options: &options,
        target_version,
        defined_result_ids: &defined_result_ids,
        defined_ids: &defined_ids,
        definitions: &definitions,
        opcodes: &opcodes,
        result_types: &result_types,
        declared_capabilities: &capabilities,
        extensions: &extensions,
        entry_models: &entry_models,
        struct_member_counts: &struct_member_counts,
    };
    run_rules(&validation_ctx, &all_limit_rules())?;

    enforce_offset_texture_operand_rule(&module, env, &options)?;
    enforce_vulkan_bitwise_widths(&module, env, &definitions, &options)?;
    enforce_small_type_storage_capabilities(&module, &definitions, &capabilities)?;
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

fn validate_functions(
    module: &Module,
    capabilities: &HashSet<rspirv::spirv::Capability>,
) -> Result<(), ValidationError> {
    let definitions = collect_result_instructions(module);
    let result_types = collect_result_types(module)?;
    let mut signatures: HashMap<Id, FunctionSignature> = HashMap::new();
    for function in &module.functions {
        let function_id = function
            .def
            .as_ref()
            .and_then(|inst| inst.result_id)
            .and_then(|raw| Id::try_from(raw).ok())
            .unwrap_or(Id::try_from(1).expect("non-zero literal"));
        let signature = validate_function_signature(function_id, function, &definitions)?;
        signatures.insert(function_id, signature);
    }
    let mut recorded_merges: Vec<(Id, Id, MergeTargetKind, Id)> = Vec::new(); // (function, header, kind, target)
    let mut definition_blocks: HashMap<ResultId, Option<Id>> = HashMap::new();
    for inst in module.all_inst_iter() {
        if let Some(result_id) = inst.result_id {
            if let Ok(id) = ResultId::try_from(result_id) {
                definition_blocks.entry(id).or_insert(None);
            }
        }
    }
    let mut seen_definition = false;
    for function in &module.functions {
        let function_id = function
            .def
            .as_ref()
            .and_then(|inst| inst.result_id)
            .and_then(|raw| Id::try_from(raw).ok())
            .unwrap_or(Id::try_from(1).expect("non-zero literal"));

        let is_declaration = function.blocks.is_empty() && function.parameters.is_empty();
        let signature = signatures
            .get(&function_id)
            .expect("signatures precomputed");
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
        let mut successors: std::collections::HashMap<Id, std::collections::HashSet<Id>> =
            block_ids
                .iter()
                .copied()
                .map(|id| (id, Default::default()))
                .collect();

        for param in &function.parameters {
            if let Some(result_id) = param.result_id {
                if let Ok(result_id) = ResultId::try_from(result_id) {
                    definition_blocks.insert(result_id, Some(entry_label_id));
                }
            }
        }

        for (block_index, block) in function.blocks.iter().enumerate() {
            let label_inst = block
                .label
                .as_ref()
                .ok_or(ValidationError::MissingBlockLabel {
                    function: function_id,
                    block_index,
                })?;
            if label_inst.class.opcode != rspirv::spirv::Op::Label {
                return Err(ValidationError::MissingBlockLabel {
                    function: function_id,
                    block_index,
                });
            }
            let block_label_id = block
                .label
                .as_ref()
                .and_then(|inst| inst.result_id)
                .and_then(|raw| Id::try_from(raw).ok())
                .unwrap_or(entry_label_id);
            let mut first_terminator_index = None;
            let mut merge_instruction: Option<(usize, &rspirv::dr::Instruction)> = None;
            let mut seen_non_phi = false;
            for (index, inst) in block.instructions.iter().enumerate() {
                if let Some(result_id) = inst.result_id {
                    if let Ok(result_id) = ResultId::try_from(result_id) {
                        definition_blocks.insert(result_id, Some(block_label_id));
                    }
                }
                if rspirv::grammar::reflect::is_block_terminator(inst.class.opcode) {
                    first_terminator_index = Some(index);
                    break;
                }
                if inst.class.opcode == rspirv::spirv::Op::Phi {
                    if seen_non_phi {
                        return Err(ValidationError::PhiAfterNonPhi {
                            function: function_id,
                            block: block_label_id,
                        });
                    }
                } else {
                    seen_non_phi = true;
                }
                if inst.class.opcode == rspirv::spirv::Op::SelectionMerge
                    || inst.class.opcode == rspirv::spirv::Op::LoopMerge
                {
                    if merge_instruction.is_some() {
                        return Err(ValidationError::DuplicateMergeInstruction {
                            function: function_id,
                            block: block_label_id,
                        });
                    }
                    merge_instruction = Some((index, inst));
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
            if let Some((merge_index, merge_inst)) = merge_instruction {
                if merge_index + 1 != terminator_index {
                    return Err(ValidationError::MergeInstructionNotBeforeTerminator {
                        function: function_id,
                        block: block_label_id,
                    });
                }
                match merge_inst.class.opcode {
                    rspirv::spirv::Op::SelectionMerge => {
                        match terminator_inst.class.opcode {
                            rspirv::spirv::Op::BranchConditional | rspirv::spirv::Op::Switch => {}
                            other => {
                                return Err(ValidationError::InvalidMergeTerminator {
                                    function: function_id,
                                    block: block_label_id,
                                    terminator: other,
                                });
                            }
                        }
                        if let Some(rspirv::dr::Operand::IdRef(raw_merge)) =
                            merge_inst.operands.first()
                        {
                            if let Ok(target) = Id::try_from(*raw_merge) {
                                if target == block_label_id {
                                    return Err(ValidationError::MergeTargetIsBlock {
                                        function: function_id,
                                        block: block_label_id,
                                        kind: MergeTargetKind::Merge,
                                        target,
                                    });
                                }
                                if !block_ids.contains(&target) {
                                    return Err(ValidationError::MergeTargetMissing {
                                        function: function_id,
                                        block: block_label_id,
                                        kind: MergeTargetKind::Merge,
                                        target,
                                    });
                                }
                                recorded_merges.push((
                                    function_id,
                                    block_label_id,
                                    MergeTargetKind::Merge,
                                    target,
                                ));
                            }
                        }
                    }
                    rspirv::spirv::Op::LoopMerge => {
                        match terminator_inst.class.opcode {
                            rspirv::spirv::Op::Branch | rspirv::spirv::Op::BranchConditional => {}
                            other => {
                                return Err(ValidationError::InvalidMergeTerminator {
                                    function: function_id,
                                    block: block_label_id,
                                    terminator: other,
                                });
                            }
                        }
                        let merge_target = merge_inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::IdRef(raw) => Id::try_from(*raw).ok(),
                            _ => None,
                        });
                        let continue_target = merge_inst.operands.get(1).and_then(|op| match op {
                            rspirv::dr::Operand::IdRef(raw) => Id::try_from(*raw).ok(),
                            _ => None,
                        });
                        if let Some(target) = merge_target {
                            if target == block_label_id {
                                return Err(ValidationError::MergeTargetIsBlock {
                                    function: function_id,
                                    block: block_label_id,
                                    kind: MergeTargetKind::Merge,
                                    target,
                                });
                            }
                            if !block_ids.contains(&target) {
                                return Err(ValidationError::MergeTargetMissing {
                                    function: function_id,
                                    block: block_label_id,
                                    kind: MergeTargetKind::Merge,
                                    target,
                                });
                            }
                            recorded_merges.push((
                                function_id,
                                block_label_id,
                                MergeTargetKind::Merge,
                                target,
                            ));
                        }
                        if let Some(target) = continue_target {
                            if target == block_label_id {
                                return Err(ValidationError::MergeTargetIsBlock {
                                    function: function_id,
                                    block: block_label_id,
                                    kind: MergeTargetKind::Continue,
                                    target,
                                });
                            }
                            if !block_ids.contains(&target) {
                                return Err(ValidationError::MergeTargetMissing {
                                    function: function_id,
                                    block: block_label_id,
                                    kind: MergeTargetKind::Continue,
                                    target,
                                });
                            }
                            recorded_merges.push((
                                function_id,
                                block_label_id,
                                MergeTargetKind::Continue,
                                target,
                            ));
                        }
                        if let (Some(merge_target), Some(continue_target)) =
                            (merge_target, continue_target)
                        {
                            if merge_target == continue_target {
                                return Err(ValidationError::ContinueTargetMatchesMerge {
                                    function: function_id,
                                    block: block_label_id,
                                    target: merge_target,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
            // NOTE: The C++ spirv-val uses a "seen" set to track which blocks have
            // been visited during structured control flow validation. BranchConditional
            // without SelectionMerge is allowed if one of its targets has already been
            // seen. This is a relaxed check that allows certain patterns like loop
            // back-edges. For now, we skip this check to match C++ behavior.
            // TODO: Implement proper "seen" tracking like C++ ValidateStructuredSelections.
            //
            // OpSwitch still requires SelectionMerge.
            if terminator_inst.class.opcode == rspirv::spirv::Op::Switch {
                let requires_selection_merge = !matches!(
                    merge_instruction,
                    Some((_, inst)) if inst.class.opcode == rspirv::spirv::Op::SelectionMerge
                );
                if requires_selection_merge {
                    return Err(ValidationError::MissingSelectionMerge {
                        function: function_id,
                        block: block_label_id,
                        terminator: terminator_inst.class.opcode,
                    });
                }
            }
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
                                if let Some(succs) = successors.get_mut(&block_label_id) {
                                    succs.insert(target);
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
                                if let Some(succs) = successors.get_mut(&block_label_id) {
                                    succs.insert(target);
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
                                    if let Some(succs) = successors.get_mut(&block_label_id) {
                                        succs.insert(target);
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

        for (block_index, block) in function.blocks.iter().enumerate() {
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
                if inst.class.opcode == rspirv::spirv::Op::FunctionCall {
                    validate_function_call(
                        function_id,
                        inst,
                        &signatures,
                        &definitions,
                        &result_types,
                    )?;
                }
                if inst.class.opcode == rspirv::spirv::Op::Variable {
                    if let Some(rspirv::dr::Operand::StorageClass(storage)) = inst.operands.first()
                    {
                        if *storage != rspirv::spirv::StorageClass::Function {
                            let variable = inst
                                .result_id
                                .and_then(|raw| Id::try_from(raw).ok())
                                .unwrap_or(function_id);
                            return Err(ValidationError::FunctionVariableStorageClassMismatch {
                                function: function_id,
                                variable,
                                storage_class: *storage,
                            });
                        }
                        if block_index != 0 {
                            let variable = inst
                                .result_id
                                .and_then(|raw| Id::try_from(raw).ok())
                                .unwrap_or(function_id);
                            return Err(ValidationError::FunctionVariableNotInEntryBlock {
                                function: function_id,
                                variable,
                            });
                        }
                    }
                }
                if inst.class.opcode == rspirv::spirv::Op::Phi {
                    let phi_result_type = inst
                        .result_id
                        .and_then(|raw| ResultId::try_from(raw).ok())
                        .and_then(|rid| result_types.get(&rid).copied());
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
                    if let Some(expected_type) = phi_result_type {
                        for pair in inst.operands.chunks(2) {
                            if let Some(rspirv::dr::Operand::IdRef(raw_value)) = pair.first() {
                                if let Ok(value_id) = ResultId::try_from(*raw_value) {
                                    if let Some(Some(def_block)) = definition_blocks.get(&value_id)
                                    {
                                        if !block_ids.contains(def_block) {
                                            return Err(
                                                ValidationError::ValueDefinedInAnotherFunction {
                                                    function: function_id,
                                                    value: Id::from(value_id),
                                                },
                                            );
                                        }
                                    }
                                    if let Some(found_type) = result_types.get(&value_id).copied() {
                                        if found_type != expected_type {
                                            return Err(ValidationError::PhiIncomingTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                incoming: Id::from(value_id),
                                                expected: expected_type,
                                                found: found_type,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if inst.class.opcode == rspirv::spirv::Op::VectorExtractDynamic {
                    let Some(block) = &block.label else {
                        continue;
                    };
                    let Some(block_label_id) =
                        block.result_id.and_then(|raw| Id::try_from(raw).ok())
                    else {
                        continue;
                    };
                    let Some(result_type_id) =
                        inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                    else {
                        continue;
                    };

                    let vector_operand = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(vector_operand) = vector_operand else {
                        continue;
                    };
                    let Some(vector_type_id) = result_types.get(&vector_operand).copied() else {
                        continue;
                    };

                    let Some(vector_type_inst) = ResultId::try_from(u32::from(vector_type_id))
                        .ok()
                        .and_then(|rid| definitions.get(&rid))
                    else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };
                    if vector_type_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    }
                    let (component_type, _) = vector_info(vector_type_inst);
                    let Some(component_type) = component_type else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };

                    if component_type != result_type_id {
                        return Err(ValidationError::InstructionResultTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            expected: component_type,
                            found: result_type_id,
                        });
                    }

                    let index_operand = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(index_operand) = index_operand else {
                        continue;
                    };
                    let Some(index_type_id) = result_types.get(&index_operand).copied() else {
                        continue;
                    };
                    let Some(index_type_inst) = ResultId::try_from(u32::from(index_type_id))
                        .ok()
                        .and_then(|rid| definitions.get(&rid))
                    else {
                        continue;
                    };
                    if index_type_inst.class.opcode != rspirv::spirv::Op::TypeInt {
                        return Err(ValidationError::VectorIndexTypeInvalid {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand_index: 1,
                            found: index_type_id,
                        });
                    }
                }
                if inst.class.opcode == rspirv::spirv::Op::VectorInsertDynamic {
                    let Some(block) = &block.label else {
                        continue;
                    };
                    let Some(block_label_id) =
                        block.result_id.and_then(|raw| Id::try_from(raw).ok())
                    else {
                        continue;
                    };
                    let Some(result_type_id) =
                        inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                    else {
                        continue;
                    };

                    let vector_operand = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(vector_operand) = vector_operand else {
                        continue;
                    };
                    let Some(vector_type_id) = result_types.get(&vector_operand).copied() else {
                        continue;
                    };
                    let Some(vector_type_inst) = ResultId::try_from(u32::from(vector_type_id))
                        .ok()
                        .and_then(|rid| definitions.get(&rid))
                    else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };
                    if vector_type_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    }
                    let (component_type, _) = vector_info(vector_type_inst);
                    let Some(component_type) = component_type else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };

                    if result_type_id != vector_type_id {
                        return Err(ValidationError::InstructionResultTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            expected: vector_type_id,
                            found: result_type_id,
                        });
                    }

                    let component_operand = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(component_operand) = component_operand else {
                        continue;
                    };
                    let Some(component_operand_type) =
                        result_types.get(&component_operand).copied()
                    else {
                        continue;
                    };
                    if component_operand_type != component_type {
                        return Err(ValidationError::OperandTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand_index: 1,
                            expected: component_type,
                            found: component_operand_type,
                        });
                    }

                    let index_operand = inst.operands.get(2).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(index_operand) = index_operand else {
                        continue;
                    };
                    let Some(index_type_id) = result_types.get(&index_operand).copied() else {
                        continue;
                    };
                    let Some(index_type_inst) = ResultId::try_from(u32::from(index_type_id))
                        .ok()
                        .and_then(|rid| definitions.get(&rid))
                    else {
                        continue;
                    };
                    if index_type_inst.class.opcode != rspirv::spirv::Op::TypeInt {
                        return Err(ValidationError::VectorIndexTypeInvalid {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand_index: 2,
                            found: index_type_id,
                        });
                    }
                }
                if inst.class.opcode == rspirv::spirv::Op::VectorTimesScalar {
                    let Some(block) = &block.label else {
                        continue;
                    };
                    let Some(block_label_id) =
                        block.result_id.and_then(|raw| Id::try_from(raw).ok())
                    else {
                        continue;
                    };

                    let vector_operand = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(vector_operand) = vector_operand else {
                        continue;
                    };
                    let Some(vector_type_id) = result_types.get(&vector_operand).copied() else {
                        continue;
                    };
                    let Some(vector_type_inst) = ResultId::try_from(u32::from(vector_type_id))
                        .ok()
                        .and_then(|rid| definitions.get(&rid))
                    else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };
                    if vector_type_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    }
                    let (component_type, _) = vector_info(vector_type_inst);
                    let Some(component_type) = component_type else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };

                    let scalar_operand = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(scalar_operand) = scalar_operand else {
                        continue;
                    };
                    let Some(scalar_type_id) = result_types.get(&scalar_operand).copied() else {
                        continue;
                    };
                    if scalar_type_id != component_type {
                        return Err(ValidationError::VectorTimesScalarTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            vector_type: vector_type_id,
                            scalar_type: scalar_type_id,
                        });
                    }

                    let Some(result_type_id) =
                        inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                    else {
                        continue;
                    };
                    if result_type_id != vector_type_id {
                        return Err(ValidationError::InstructionResultTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            expected: vector_type_id,
                            found: result_type_id,
                        });
                    }
                }
                if inst.class.opcode == rspirv::spirv::Op::MatrixTimesVector {
                    let Some(block) = &block.label else {
                        continue;
                    };
                    let Some(block_label_id) =
                        block.result_id.and_then(|raw| Id::try_from(raw).ok())
                    else {
                        continue;
                    };
                    let Some(result_type_id) =
                        inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                    else {
                        continue;
                    };

                    let matrix_operand = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(matrix_operand) = matrix_operand else {
                        continue;
                    };
                    let Some(matrix_type_id) = result_types.get(&matrix_operand).copied() else {
                        continue;
                    };
                    let Some((matrix_component, matrix_rows, matrix_columns, matrix_column_type)) =
                        matrix_details(matrix_type_id, &definitions)
                    else {
                        return Err(ValidationError::MatrixOperandNotMatrix {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: matrix_type_id,
                        });
                    };

                    let vector_operand = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(vector_operand) = vector_operand else {
                        continue;
                    };
                    let Some(vector_type_id) = result_types.get(&vector_operand).copied() else {
                        continue;
                    };
                    let Some(vector_type_inst) = type_instruction(vector_type_id, &definitions)
                    else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 1,
                            found: vector_type_id,
                        });
                    };
                    if vector_type_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 1,
                            found: vector_type_id,
                        });
                    }
                    let (vector_component, vector_len) = vector_info(vector_type_inst);
                    let Some(vector_component) = vector_component else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 1,
                            found: vector_type_id,
                        });
                    };
                    let Some(vector_len) = vector_len else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 1,
                            found: vector_type_id,
                        });
                    };

                    if matrix_component != vector_component {
                        return Err(ValidationError::MatrixTimesVectorComponentTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            matrix_component,
                            vector_component,
                        });
                    }
                    if matrix_columns != vector_len {
                        return Err(ValidationError::MatrixTimesVectorDimensionMismatch {
                            function: function_id,
                            block: block_label_id,
                            matrix_columns,
                            vector_components: vector_len,
                        });
                    }

                    let Some(expected_result_type) =
                        TypeId::try_from(u32::from(matrix_column_type)).ok()
                    else {
                        continue;
                    };
                    if result_type_id != expected_result_type {
                        return Err(ValidationError::InstructionResultTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            expected: expected_result_type,
                            found: result_type_id,
                        });
                    }
                    if let Some(result_inst) = type_instruction(result_type_id, &definitions) {
                        if result_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                            return Err(ValidationError::InstructionResultTypeMismatch {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                                expected: expected_result_type,
                                found: result_type_id,
                            });
                        }
                        let (result_component, result_len) = vector_info(result_inst);
                        match result_component {
                            Some(result_component) if result_component == matrix_component => {}
                            _ => {
                                return Err(ValidationError::InstructionResultTypeMismatch {
                                    function: function_id,
                                    block: block_label_id,
                                    instruction: inst.class.opcode,
                                    expected: expected_result_type,
                                    found: result_type_id,
                                });
                            }
                        }
                        match result_len {
                            Some(result_len) if result_len == matrix_rows => {}
                            _ => {
                                return Err(ValidationError::InstructionResultTypeMismatch {
                                    function: function_id,
                                    block: block_label_id,
                                    instruction: inst.class.opcode,
                                    expected: expected_result_type,
                                    found: result_type_id,
                                });
                            }
                        }
                    }
                }
                if inst.class.opcode == rspirv::spirv::Op::VectorTimesMatrix {
                    let Some(block) = &block.label else {
                        continue;
                    };
                    let Some(block_label_id) =
                        block.result_id.and_then(|raw| Id::try_from(raw).ok())
                    else {
                        continue;
                    };
                    let Some(result_type_id) =
                        inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                    else {
                        continue;
                    };

                    let vector_operand = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(vector_operand) = vector_operand else {
                        continue;
                    };
                    let Some(vector_type_id) = result_types.get(&vector_operand).copied() else {
                        continue;
                    };
                    let Some(vector_type_inst) = type_instruction(vector_type_id, &definitions)
                    else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };
                    if vector_type_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    }
                    let (vector_component, vector_len) = vector_info(vector_type_inst);
                    let Some(vector_component) = vector_component else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };
                    let Some(vector_len) = vector_len else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: vector_type_id,
                        });
                    };

                    let matrix_operand = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(matrix_operand) = matrix_operand else {
                        continue;
                    };
                    let Some(matrix_type_id) = result_types.get(&matrix_operand).copied() else {
                        continue;
                    };
                    let Some((matrix_component, matrix_rows, matrix_columns, _)) =
                        matrix_details(matrix_type_id, &definitions)
                    else {
                        return Err(ValidationError::MatrixOperandNotMatrix {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 1,
                            found: matrix_type_id,
                        });
                    };

                    if vector_component != matrix_component {
                        return Err(ValidationError::VectorTimesMatrixComponentTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            vector_component,
                            matrix_component,
                        });
                    }
                    if vector_len != matrix_rows {
                        return Err(ValidationError::VectorTimesMatrixDimensionMismatch {
                            function: function_id,
                            block: block_label_id,
                            vector_components: vector_len,
                            matrix_rows,
                        });
                    }

                    if let Some(result_vector_inst) = type_instruction(result_type_id, &definitions)
                    {
                        if result_vector_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                            return Err(
                                ValidationError::VectorTimesMatrixResultDimensionMismatch {
                                    function: function_id,
                                    block: block_label_id,
                                    expected_components: matrix_columns,
                                    found_components: 0,
                                },
                            );
                        }
                        let (result_component, result_len) = vector_info(result_vector_inst);
                        let Some(result_component) = result_component else {
                            return Err(
                                ValidationError::VectorTimesMatrixResultComponentTypeMismatch {
                                    function: function_id,
                                    block: block_label_id,
                                    expected: matrix_component,
                                    found: matrix_component,
                                },
                            );
                        };
                        if result_component != matrix_component {
                            return Err(
                                ValidationError::VectorTimesMatrixResultComponentTypeMismatch {
                                    function: function_id,
                                    block: block_label_id,
                                    expected: matrix_component,
                                    found: result_component,
                                },
                            );
                        }
                        match result_len {
                            Some(result_len) if result_len == matrix_columns => {}
                            Some(result_len) => {
                                return Err(
                                    ValidationError::VectorTimesMatrixResultDimensionMismatch {
                                        function: function_id,
                                        block: block_label_id,
                                        expected_components: matrix_columns,
                                        found_components: result_len,
                                    },
                                );
                            }
                            None => {
                                return Err(
                                    ValidationError::VectorTimesMatrixResultDimensionMismatch {
                                        function: function_id,
                                        block: block_label_id,
                                        expected_components: matrix_columns,
                                        found_components: 0,
                                    },
                                );
                            }
                        }
                    }
                }
                if inst.class.opcode == rspirv::spirv::Op::MatrixTimesMatrix {
                    let Some(block) = &block.label else {
                        continue;
                    };
                    let Some(block_label_id) =
                        block.result_id.and_then(|raw| Id::try_from(raw).ok())
                    else {
                        continue;
                    };
                    let Some(result_type_id) =
                        inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                    else {
                        continue;
                    };

                    let left_operand = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(left_operand) = left_operand else {
                        continue;
                    };
                    let Some(left_type_id) = result_types.get(&left_operand).copied() else {
                        continue;
                    };
                    let Some((left_component, left_rows, left_columns, _)) =
                        matrix_details(left_type_id, &definitions)
                    else {
                        return Err(ValidationError::MatrixOperandNotMatrix {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: left_type_id,
                        });
                    };

                    let right_operand = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(right_operand) = right_operand else {
                        continue;
                    };
                    let Some(right_type_id) = result_types.get(&right_operand).copied() else {
                        continue;
                    };
                    let Some((right_component, right_rows, right_columns, _)) =
                        matrix_details(right_type_id, &definitions)
                    else {
                        return Err(ValidationError::MatrixOperandNotMatrix {
                            function: function_id,
                            block: block_label_id,
                            instruction: inst.class.opcode,
                            operand: 1,
                            found: right_type_id,
                        });
                    };

                    if left_component != right_component {
                        return Err(ValidationError::MatrixTimesMatrixComponentTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            left_component,
                            right_component,
                        });
                    }
                    if left_columns != right_rows {
                        return Err(ValidationError::MatrixTimesMatrixDimensionMismatch {
                            function: function_id,
                            block: block_label_id,
                            left_columns,
                            right_rows,
                        });
                    }

                    let Some(result_inst) = type_instruction(result_type_id, &definitions) else {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    };
                    if result_inst.class.opcode != rspirv::spirv::Op::TypeMatrix {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    }
                    let (result_column_type, result_columns) = matrix_info(result_inst);
                    let Some(result_columns) = result_columns else {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    };
                    if result_columns != right_columns {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    }
                    let Some(result_column_type) = result_column_type else {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    };
                    let Some(result_column_inst) =
                        type_instruction(result_column_type, &definitions)
                    else {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    };
                    if result_column_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    }
                    let (result_component, result_rows) = vector_info(result_column_inst);
                    let Some(result_rows) = result_rows else {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    };
                    if result_rows != left_rows {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: function_id,
                            block: block_label_id,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    }
                    if let Some(result_component) = result_component {
                        if result_component != left_component {
                            return Err(
                                ValidationError::MatrixTimesMatrixResultComponentTypeMismatch {
                                    function: function_id,
                                    block: block_label_id,
                                    expected: left_component,
                                    found: result_component,
                                },
                            );
                        }
                    } else {
                        return Err(
                            ValidationError::MatrixTimesMatrixResultComponentTypeMismatch {
                                function: function_id,
                                block: block_label_id,
                                expected: left_component,
                                found: left_component,
                            },
                        );
                    }
                }
                if inst.class.opcode == rspirv::spirv::Op::VectorShuffle {
                    let Some(block) = &block.label else {
                        continue;
                    };
                    let Some(block_label_id) =
                        block.result_id.and_then(|raw| Id::try_from(raw).ok())
                    else {
                        continue;
                    };

                    let Some(result_type_id) =
                        inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                    else {
                        continue;
                    };

                    let vector_type_info = |ty: TypeId| -> Result<(TypeId, u32), ValidationError> {
                        let type_inst = ResultId::try_from(u32::from(ty))
                            .ok()
                            .and_then(|rid| definitions.get(&rid));
                        let Some(type_inst) = type_inst else {
                            return Err(ValidationError::VectorShuffleOperandNotVector {
                                function: function_id,
                                block: block_label_id,
                                operand: 0,
                                found: ty,
                            });
                        };
                        if type_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                            return Err(ValidationError::VectorShuffleOperandNotVector {
                                function: function_id,
                                block: block_label_id,
                                operand: 0,
                                found: ty,
                            });
                        }
                        let (elem, count) = vector_info(type_inst);
                        match (elem, count) {
                            (Some(elem), Some(count)) => Ok((elem, count)),
                            _ => Err(ValidationError::VectorShuffleOperandNotVector {
                                function: function_id,
                                block: block_label_id,
                                operand: 0,
                                found: ty,
                            }),
                        }
                    };

                    let Some(vec1_id) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    }) else {
                        continue;
                    };
                    let Some(vec2_id) = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    }) else {
                        continue;
                    };
                    let Some(vec1_type_id) = result_types.get(&vec1_id).copied() else {
                        continue;
                    };
                    let Some(vec2_type_id) = result_types.get(&vec2_id).copied() else {
                        continue;
                    };

                    let (vec1_component, vec1_len) = match vector_type_info(vec1_type_id) {
                        Ok(info) => info,
                        Err(mut err) => {
                            if let ValidationError::VectorShuffleOperandNotVector {
                                ref mut operand,
                                ..
                            } = err
                            {
                                *operand = 0;
                            }
                            return Err(err);
                        }
                    };
                    let (vec2_component, vec2_len) = match vector_type_info(vec2_type_id) {
                        Ok(info) => info,
                        Err(mut err) => {
                            if let ValidationError::VectorShuffleOperandNotVector {
                                ref mut operand,
                                ..
                            } = err
                            {
                                *operand = 1;
                            }
                            return Err(err);
                        }
                    };

                    if vec1_component != vec2_component {
                        return Err(ValidationError::VectorShuffleComponentTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            first: vec1_component,
                            second: vec2_component,
                        });
                    }

                    let result_vector_inst = ResultId::try_from(u32::from(result_type_id))
                        .ok()
                        .and_then(|rid| definitions.get(&rid));
                    let Some(result_vector_inst) = result_vector_inst else {
                        continue;
                    };
                    if result_vector_inst.class.opcode != rspirv::spirv::Op::TypeVector {
                        return Err(ValidationError::VectorShuffleResultTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            result_type: result_type_id,
                            component_type: vec1_component,
                        });
                    }
                    let (result_component, result_len) = vector_info(result_vector_inst);
                    let Some(result_component) = result_component else {
                        return Err(ValidationError::VectorShuffleResultTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            result_type: result_type_id,
                            component_type: vec1_component,
                        });
                    };
                    if result_component != vec1_component {
                        return Err(ValidationError::VectorShuffleResultTypeMismatch {
                            function: function_id,
                            block: block_label_id,
                            result_type: result_type_id,
                            component_type: vec1_component,
                        });
                    }

                    let literal_components: Vec<u32> = inst
                        .operands
                        .iter()
                        .skip(2)
                        .filter_map(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            rspirv::dr::Operand::LiteralBit64(v) => u32::try_from(*v).ok(),
                            _ => None,
                        })
                        .collect();
                    let operand_component_count = literal_components.len() as u32;
                    let Some(result_component_len) = result_len else {
                        return Err(ValidationError::VectorShuffleComponentCountMismatch {
                            function: function_id,
                            block: block_label_id,
                            operand_components: operand_component_count,
                            result_components: 0,
                        });
                    };
                    if operand_component_count != result_component_len {
                        return Err(ValidationError::VectorShuffleComponentCountMismatch {
                            function: function_id,
                            block: block_label_id,
                            operand_components: operand_component_count,
                            result_components: result_component_len,
                        });
                    }

                    let max_index = vec1_len + vec2_len;
                    for value in literal_components {
                        if value == u32::MAX {
                            continue;
                        }
                        if value >= max_index {
                            return Err(ValidationError::VectorShuffleComponentOutOfRange {
                                function: function_id,
                                block: block_label_id,
                                value,
                                max: max_index.saturating_sub(1),
                            });
                        }
                    }
                }
            }
        }

        // Compute dominators.
        // NOTE: Unreachable blocks are allowed by the SPIR-V spec. The C++ validator
        // skips unreachable blocks during structured control flow validation. We simply
        // skip blocks with no predecessors (other than entry) in dominator computation.
        let mut dominators: HashMap<Id, std::collections::HashSet<Id>> = HashMap::new();
        for id in &block_ids {
            let mut set: std::collections::HashSet<Id> = if *id == entry_label_id {
                Default::default()
            } else {
                block_ids.clone()
            };
            set.insert(*id);
            dominators.insert(*id, set);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for block in &block_ids {
                if *block == entry_label_id {
                    continue;
                }
                let preds = predecessors.get(block).cloned().unwrap_or_default();
                if preds.is_empty() {
                    continue;
                }
                let mut new_dom: std::collections::HashSet<Id> = block_ids.clone();
                for pred in preds {
                    if let Some(pred_dom) = dominators.get(&pred) {
                        new_dom = new_dom
                            .intersection(pred_dom)
                            .copied()
                            .collect::<std::collections::HashSet<_>>();
                    }
                }
                new_dom.insert(*block);
                if new_dom != *dominators.get(block).unwrap_or(&Default::default()) {
                    dominators.insert(*block, new_dom);
                    changed = true;
                }
            }
        }

        for (function, header, kind, target) in recorded_merges
            .iter()
            .copied()
            .filter(|(func, _, _, _)| *func == function_id)
        {
            let Some(target_doms) = dominators.get(&target) else {
                continue;
            };
            if !target_doms.contains(&header) && header != target {
                return Err(ValidationError::MergeTargetNotDominated {
                    function,
                    block: header,
                    kind,
                    target,
                });
            }
        }

        let integer_shape = |ty_inst: &rspirv::dr::Instruction| -> Option<(usize, u32)> {
            match ty_inst.class.opcode {
                rspirv::spirv::Op::TypeInt => ty_inst
                    .operands
                    .first()
                    .and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .map(|width| (1, width)),
                rspirv::spirv::Op::TypeVector => {
                    let elem_ty = ty_inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(raw) => ResultId::try_from(*raw).ok(),
                        _ => None,
                    });
                    let elem_inst = elem_ty.and_then(|rid| definitions.get(&rid));
                    let width = elem_inst.and_then(|inst| match inst.class.opcode {
                        rspirv::spirv::Op::TypeInt => {
                            inst.operands.first().and_then(|op| match op {
                                rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                                _ => None,
                            })
                        }
                        _ => None,
                    });
                    let count = ty_inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v as usize),
                        _ => None,
                    });
                    match (count, width) {
                        (Some(c), Some(w)) => Some((c, w)),
                        _ => None,
                    }
                }
                _ => None,
            }
        };

        // Validate dominance: general operands must be dominated by their definition block.
        for block in &function.blocks {
            let block_label_id = block
                .label
                .as_ref()
                .and_then(|inst| inst.result_id)
                .and_then(|raw| Id::try_from(raw).ok())
                .unwrap_or(entry_label_id);
            let empty_set: std::collections::HashSet<Id> = Default::default();
            let block_dominators = dominators.get(&block_label_id).unwrap_or(&empty_set);

            for inst in &block.instructions {
                let result_type = inst.result_type.and_then(|raw| TypeId::try_from(raw).ok());
                let result_type_inst = result_type
                    .and_then(|ty| ResultId::try_from(u32::from(Id::from(ty))).ok())
                    .and_then(|rid| definitions.get(&rid));

                if inst.class.opcode == rspirv::spirv::Op::Phi {
                    for pair in inst.operands.chunks(2) {
                        if let (
                            Some(rspirv::dr::Operand::IdRef(raw_value)),
                            Some(rspirv::dr::Operand::IdRef(raw_incoming)),
                        ) = (pair.first(), pair.get(1))
                        {
                            if let (Ok(value_id), Ok(incoming_block)) =
                                (Id::try_from(*raw_value), Id::try_from(*raw_incoming))
                            {
                                if let Ok(result_id) = ResultId::try_from(*raw_value) {
                                    if let Some(Some(def_block)) = definition_blocks.get(&result_id)
                                    {
                                        let incoming_empty: std::collections::HashSet<Id> =
                                            Default::default();
                                        let incoming_dominators = dominators
                                            .get(&incoming_block)
                                            .unwrap_or(&incoming_empty);
                                        if !incoming_dominators.contains(def_block) {
                                            return Err(ValidationError::PhiIncomingNotDominated {
                                                function: function_id,
                                                block: block_label_id,
                                                incoming: incoming_block,
                                                value: value_id,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    continue;
                }

                match inst.class.opcode {
                    rspirv::spirv::Op::AccessChain
                    | rspirv::spirv::Op::InBoundsAccessChain
                    | rspirv::spirv::Op::PtrAccessChain
                    | rspirv::spirv::Op::InBoundsPtrAccessChain
                    | rspirv::spirv::Op::UntypedPtrAccessChainKHR
                    | rspirv::spirv::Op::UntypedInBoundsPtrAccessChainKHR => {
                        let Some(rspirv::dr::Operand::IdRef(base_raw)) = inst.operands.first()
                        else {
                            continue;
                        };
                        let Some(base_id) = ResultId::try_from(*base_raw).ok() else {
                            continue;
                        };
                        let Some(base_type) = result_types.get(&base_id).copied() else {
                            continue;
                        };
                        let Some(base_ptr_inst) = type_instruction(base_type, &definitions) else {
                            continue;
                        };
                        let (base_storage_class, mut current_type) =
                            match base_ptr_inst.class.opcode {
                                rspirv::spirv::Op::TypePointer
                                | rspirv::spirv::Op::TypeUntypedPointerKHR => {
                                    let storage = base_ptr_inst
                                        .operands
                                        .first()
                                        .and_then(|op| match op {
                                            rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                                            _ => None,
                                        })
                                        .unwrap_or(rspirv::spirv::StorageClass::Function);
                                    let pointee = base_ptr_inst
                                        .operands
                                        .get(1)
                                        .and_then(|op| match op {
                                            rspirv::dr::Operand::IdRef(raw) => {
                                                TypeId::try_from(*raw).ok()
                                            }
                                            _ => None,
                                        })
                                        .ok_or(ValidationError::AccessChainBaseNotPointer {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            base_type,
                                        })?;
                                    (storage, pointee)
                                }
                                _ => {
                                    return Err(ValidationError::AccessChainBaseNotPointer {
                                        function: function_id,
                                        block: block_label_id,
                                        instruction: inst.class.opcode,
                                        base_type,
                                    })
                                }
                            };

                        // All index operands must be integer scalars when they are ids.
                        for (operand_index, operand) in inst.operands.iter().enumerate().skip(1) {
                            if let rspirv::dr::Operand::IdRef(raw) = operand {
                                let Some(index_id) = ResultId::try_from(*raw).ok() else {
                                    continue;
                                };
                                let Some(index_type) = result_types.get(&index_id).copied() else {
                                    continue;
                                };
                                let Some(index_type_inst) =
                                    type_instruction(index_type, &definitions)
                                else {
                                    continue;
                                };
                                let is_int_scalar = matches!(
                                    index_type_inst.class.opcode,
                                    rspirv::spirv::Op::TypeInt
                                );
                                if !is_int_scalar {
                                    return Err(ValidationError::AccessChainIndexTypeInvalid {
                                        function: function_id,
                                        block: block_label_id,
                                        instruction: inst.class.opcode,
                                        operand_index,
                                        found: index_type,
                                    });
                                }
                            }
                        }

                        for (operand_index, operand) in inst.operands.iter().enumerate().skip(1) {
                            let (literal_index, _is_literal_operand) = match operand {
                                rspirv::dr::Operand::LiteralBit32(v) => (Some(*v), true),
                                rspirv::dr::Operand::LiteralBit64(v) => (Some(*v as u32), true),
                                rspirv::dr::Operand::IdRef(raw) => {
                                    let const_inst = ResultId::try_from(*raw)
                                        .ok()
                                        .and_then(|rid| definitions.get(&rid));
                                    let value =
                                        const_inst.and_then(|inst| match inst.class.opcode {
                                            rspirv::spirv::Op::Constant => {
                                                inst.operands.first().and_then(|op| match op {
                                                    rspirv::dr::Operand::LiteralBit32(v) => {
                                                        Some(*v)
                                                    }
                                                    rspirv::dr::Operand::LiteralBit64(v) => {
                                                        Some(*v as u32)
                                                    }
                                                    _ => None,
                                                })
                                            }
                                            _ => None,
                                        });
                                    (value, false)
                                }
                                _ => (None, false),
                            };
                            let Some(current_inst) = type_instruction(current_type, &definitions)
                            else {
                                return Err(ValidationError::AccessChainNonCompositeTarget {
                                    function: function_id,
                                    block: block_label_id,
                                    instruction: inst.class.opcode,
                                    composite_type: current_type,
                                });
                            };
                            match current_inst.class.opcode {
                                rspirv::spirv::Op::TypeStruct => {
                                    let Some(index_val) = literal_index else {
                                        return Err(
                                            ValidationError::AccessChainStructIndexNotLiteral {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                composite_type: current_type,
                                            },
                                        );
                                    };
                                    let bound = current_inst.operands.len() as u32;
                                    if index_val >= bound {
                                        return Err(
                                            ValidationError::AccessChainStructIndexOutOfBounds {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                composite_type: current_type,
                                                index: index_val,
                                                bound,
                                            },
                                        );
                                    }
                                    let member_type = current_inst
                                        .operands
                                        .get(index_val as usize)
                                        .and_then(|op| match op {
                                            rspirv::dr::Operand::IdRef(raw) => {
                                                TypeId::try_from(*raw).ok()
                                            }
                                            _ => None,
                                        });
                                    current_type = member_type.ok_or(
                                        ValidationError::AccessChainNonCompositeTarget {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            composite_type: current_type,
                                        },
                                    )?;
                                }
                                rspirv::spirv::Op::TypeArray
                                | rspirv::spirv::Op::TypeRuntimeArray => {
                                    let element_type =
                                        current_inst.operands.first().and_then(|op| match op {
                                            rspirv::dr::Operand::IdRef(raw) => {
                                                TypeId::try_from(*raw).ok()
                                            }
                                            _ => None,
                                        });
                                    if let (Some(len), Some(idx)) =
                                        (array_length(current_inst, &definitions), literal_index)
                                    {
                                        if idx >= len {
                                            return Err(
                                                ValidationError::AccessChainStructIndexOutOfBounds {
                                                    function: function_id,
                                                    block: block_label_id,
                                                    instruction: inst.class.opcode,
                                                    composite_type: current_type,
                                                    index: idx,
                                                    bound: len,
                                                },
                                            );
                                        }
                                    }
                                    current_type = element_type.ok_or(
                                        ValidationError::AccessChainNonCompositeTarget {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            composite_type: current_type,
                                        },
                                    )?;
                                }
                                rspirv::spirv::Op::TypeVector => {
                                    let element_type =
                                        current_inst.operands.first().and_then(|op| match op {
                                            rspirv::dr::Operand::IdRef(raw) => {
                                                TypeId::try_from(*raw).ok()
                                            }
                                            _ => None,
                                        });
                                    if let (Some(idx), Some(bound)) = (
                                        literal_index,
                                        current_inst.operands.get(1).and_then(|op| match op {
                                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                                            _ => None,
                                        }),
                                    ) {
                                        if idx >= bound {
                                            return Err(
                                                ValidationError::AccessChainStructIndexOutOfBounds {
                                                    function: function_id,
                                                    block: block_label_id,
                                                    instruction: inst.class.opcode,
                                                    composite_type: current_type,
                                                    index: idx,
                                                    bound,
                                                },
                                            );
                                        }
                                    }
                                    current_type = element_type.ok_or(
                                        ValidationError::AccessChainNonCompositeTarget {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            composite_type: current_type,
                                        },
                                    )?;
                                }
                                rspirv::spirv::Op::TypeMatrix => {
                                    let column_type =
                                        current_inst.operands.first().and_then(|op| match op {
                                            rspirv::dr::Operand::IdRef(raw) => {
                                                TypeId::try_from(*raw).ok()
                                            }
                                            _ => None,
                                        });
                                    if let (Some(idx), Some(bound)) = (
                                        literal_index,
                                        current_inst.operands.get(1).and_then(|op| match op {
                                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                                            _ => None,
                                        }),
                                    ) {
                                        if idx >= bound {
                                            return Err(
                                                ValidationError::AccessChainStructIndexOutOfBounds {
                                                    function: function_id,
                                                    block: block_label_id,
                                                    instruction: inst.class.opcode,
                                                    composite_type: current_type,
                                                    index: idx,
                                                    bound,
                                                },
                                            );
                                        }
                                    }
                                    current_type = column_type.ok_or(
                                        ValidationError::AccessChainNonCompositeTarget {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            composite_type: current_type,
                                        },
                                    )?;
                                }
                                rspirv::spirv::Op::TypePointer => {
                                    let pointee =
                                        current_inst.operands.get(1).and_then(|op| match op {
                                            rspirv::dr::Operand::IdRef(raw) => {
                                                TypeId::try_from(*raw).ok()
                                            }
                                            _ => None,
                                        });
                                    current_type = pointee.ok_or(
                                        ValidationError::AccessChainNonCompositeTarget {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            composite_type: current_type,
                                        },
                                    )?;
                                }
                                _ => {
                                    return Err(ValidationError::AccessChainNonCompositeTarget {
                                        function: function_id,
                                        block: block_label_id,
                                        instruction: inst.class.opcode,
                                        composite_type: current_type,
                                    });
                                }
                            }
                            // Subsequent indexes apply to the new composite type.
                            let _ = operand_index;
                        }

                        let Some(result_type) = result_type else {
                            continue;
                        };
                        let Some(result_inst) = type_instruction(result_type, &definitions) else {
                            continue;
                        };
                        let (result_storage, result_pointee) = match result_inst.class.opcode {
                            rspirv::spirv::Op::TypePointer
                            | rspirv::spirv::Op::TypeUntypedPointerKHR => {
                                let storage =
                                    result_inst.operands.first().and_then(|op| match op {
                                        rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                                        _ => None,
                                    });
                                let pointee = result_inst.operands.get(1).and_then(|op| match op {
                                    rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw).ok(),
                                    _ => None,
                                });
                                let (storage, pointee) = match (storage, pointee) {
                                    (Some(storage), Some(pointee)) => (storage, pointee),
                                    _ => {
                                        return Err(
                                            ValidationError::AccessChainResultTypeNotPointer {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                result_type,
                                            },
                                        )
                                    }
                                };
                                (storage, pointee)
                            }
                            _ => {
                                return Err(ValidationError::AccessChainResultTypeNotPointer {
                                    function: function_id,
                                    block: block_label_id,
                                    instruction: inst.class.opcode,
                                    result_type,
                                })
                            }
                        };

                        if base_storage_class != result_storage {
                            return Err(ValidationError::AccessChainStorageClassMismatch {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                                base_storage_class,
                                result_storage_class: result_storage,
                            });
                        }
                        if current_type != result_pointee {
                            return Err(ValidationError::AccessChainResultTypeMismatch {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                                expected: current_type,
                                found: result_pointee,
                            });
                        }
                    }
                    rspirv::spirv::Op::CopyObject => {
                        if let (Some(result_type), Some(rspirv::dr::Operand::IdRef(source_raw))) =
                            (result_type, inst.operands.first())
                        {
                            if let Ok(source_id) = ResultId::try_from(*source_raw) {
                                if let Some(source_type) = result_types.get(&source_id).copied() {
                                    if source_type != result_type {
                                        return Err(
                                            ValidationError::InstructionResultTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                expected: source_type,
                                                found: result_type,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                    rspirv::spirv::Op::Load => {
                        if let (Some(result_type), Some(rspirv::dr::Operand::IdRef(ptr_raw))) =
                            (result_type, inst.operands.first())
                        {
                            if let Ok(ptr_id) = ResultId::try_from(*ptr_raw) {
                                if let Some(ptr_inst) = definitions.get(&ptr_id) {
                                    if let Some(ptr_type_raw) = ptr_inst.result_type {
                                        if let Ok(ptr_type) = TypeId::try_from(ptr_type_raw) {
                                            if let Some(ptr_type_inst) =
                                                type_instruction(ptr_type, &definitions)
                                            {
                                                if ptr_type_inst.class.opcode
                                                    == rspirv::spirv::Op::TypePointer
                                                {
                                                    if let Some(rspirv::dr::Operand::IdRef(
                                                        pointee_raw,
                                                    )) = ptr_type_inst.operands.get(1)
                                                    {
                                                        if let Ok(pointee_type) =
                                                            TypeId::try_from(*pointee_raw)
                                                        {
                                                            if pointee_type != result_type {
                                                                return Err(
                                                                    ValidationError::InstructionResultTypeMismatch {
                                                                        function: function_id,
                                                                        block: block_label_id,
                                                                        instruction: inst.class.opcode,
                                                                        expected: pointee_type,
                                                                        found: result_type,
                                                                    },
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    rspirv::spirv::Op::CompositeExtract | rspirv::spirv::Op::CompositeInsert => {
                        let index_start = if inst.class.opcode == rspirv::spirv::Op::CompositeInsert
                        {
                            2
                        } else {
                            1
                        };
                        let literal_indexes: Option<Vec<u32>> = inst
                            .operands
                            .iter()
                            .skip(index_start)
                            .map(|op| match op {
                                rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                                rspirv::dr::Operand::LiteralBit64(v) => Some(*v as u32),
                                _ => None,
                            })
                            .collect();
                        if literal_indexes.is_none() {
                            return Err(ValidationError::CompositeInstructionMissingIndexes {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                            });
                        }
                        let literal_indexes = literal_indexes.unwrap();
                        if literal_indexes.is_empty() {
                            return Err(ValidationError::CompositeInstructionMissingIndexes {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                            });
                        }
                        let composite_operand = inst
                            .operands
                            .get(if inst.class.opcode == rspirv::spirv::Op::CompositeInsert {
                                1
                            } else {
                                0
                            })
                            .and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(raw) => ResultId::try_from(*raw).ok(),
                                _ => None,
                            });
                        if let (Some(result_type), Some(composite_id)) =
                            (result_type, composite_operand)
                        {
                            if let Some(composite_type) = result_types.get(&composite_id).copied() {
                                let component_type = composite_member_type(
                                    composite_type,
                                    &literal_indexes,
                                    &definitions,
                                )
                                .map_err(|err| match err {
                                    CompositeWalkError::NotComposite => {
                                        ValidationError::CompositeOperandNotComposite {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            composite_type,
                                        }
                                    }
                                    CompositeWalkError::OutOfBounds {
                                        composite_type,
                                        index_position,
                                        index,
                                        bound,
                                    } => ValidationError::CompositeIndexOutOfBounds {
                                        function: function_id,
                                        block: block_label_id,
                                        instruction: inst.class.opcode,
                                        composite_type,
                                        index_position,
                                        index,
                                        bound,
                                    },
                                })?;

                                if inst.class.opcode == rspirv::spirv::Op::CompositeExtract {
                                    if component_type != result_type {
                                        return Err(
                                            ValidationError::InstructionResultTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                expected: component_type,
                                                found: result_type,
                                            },
                                        );
                                    }
                                } else {
                                    if composite_type != result_type {
                                        return Err(
                                            ValidationError::InstructionResultTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                expected: composite_type,
                                                found: result_type,
                                            },
                                        );
                                    }
                                    let object_type = inst
                                        .operands
                                        .first()
                                        .and_then(|op| match op {
                                            rspirv::dr::Operand::IdRef(raw) => {
                                                ResultId::try_from(*raw).ok()
                                            }
                                            _ => None,
                                        })
                                        .and_then(|rid| result_types.get(&rid).copied());
                                    if let Some(object_type) = object_type {
                                        if object_type != component_type {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index: 0,
                                                expected: component_type,
                                                found: object_type,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }

                // Result-type expectations for logical/bitwise/shift ops.
                if let Some(result_type) = result_type {
                    // Logical ops require bool result.
                    if matches!(
                        inst.class.opcode,
                        rspirv::spirv::Op::LogicalAnd
                            | rspirv::spirv::Op::LogicalOr
                            | rspirv::spirv::Op::LogicalNot
                    ) {
                        let is_bool = result_type_inst
                            .map(|inst| inst.class.opcode == rspirv::spirv::Op::TypeBool)
                            .unwrap_or(false);
                        if !is_bool {
                            return Err(ValidationError::InstructionResultTypeMismatch {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                                expected: result_type,
                                found: result_type,
                            });
                        }
                    }

                    // Pointer comparisons require a boolean result for equality, integer result
                    // for pointer difference, pointer operands, matching pointer types, and
                    // appropriate storage-class capabilities.
                    if matches!(
                        inst.class.opcode,
                        rspirv::spirv::Op::PtrEqual
                            | rspirv::spirv::Op::PtrNotEqual
                            | rspirv::spirv::Op::PtrDiff
                    ) {
                        let result_ok = match inst.class.opcode {
                            rspirv::spirv::Op::PtrEqual | rspirv::spirv::Op::PtrNotEqual => {
                                result_type_inst.is_some_and(|ty| {
                                    ty.class.opcode == rspirv::spirv::Op::TypeBool
                                })
                            }
                            rspirv::spirv::Op::PtrDiff => result_type_inst
                                .is_some_and(|ty| ty.class.opcode == rspirv::spirv::Op::TypeInt),
                            _ => false,
                        };
                        if !result_ok {
                            return Err(ValidationError::InstructionResultTypeMismatch {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                                expected: result_type,
                                found: result_type,
                            });
                        }

                        let operand_type = |operand: &rspirv::dr::Operand| -> Option<TypeId> {
                            match operand {
                                rspirv::dr::Operand::IdRef(raw) => ResultId::try_from(*raw)
                                    .ok()
                                    .and_then(|rid| result_types.get(&rid).copied()),
                                _ => None,
                            }
                        };

                        let op0_type = inst.operands.first().and_then(operand_type);
                        if let Some(op0_type) = op0_type {
                            let Some((op0_opcode, op0_storage_class)) =
                                pointer_info(op0_type, &definitions)
                            else {
                                return Err(ValidationError::PointerComparisonOperandNotPointer {
                                    function: function_id,
                                    block: block_label_id,
                                    instruction: inst.class.opcode,
                                    operand_index: 0,
                                    found: op0_type,
                                });
                            };

                            if let Some(op1_type) = inst.operands.get(1).and_then(operand_type) {
                                let Some((op1_opcode, op1_storage_class)) =
                                    pointer_info(op1_type, &definitions)
                                else {
                                    return Err(
                                        ValidationError::PointerComparisonOperandNotPointer {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            operand_index: 1,
                                            found: op1_type,
                                        },
                                    );
                                };

                                match inst.class.opcode {
                                    rspirv::spirv::Op::PtrDiff => {
                                        if op0_type != op1_type {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index: 1,
                                                expected: op0_type,
                                                found: op1_type,
                                            });
                                        }
                                    }
                                    _ => {
                                        let storage_classes_match =
                                            op0_storage_class == op1_storage_class;
                                        let allows_untyped_mismatch = storage_classes_match
                                            && (op0_opcode
                                                == rspirv::spirv::Op::TypeUntypedPointerKHR
                                                || op1_opcode
                                                    == rspirv::spirv::Op::TypeUntypedPointerKHR);

                                        if op0_type != op1_type && !allows_untyped_mismatch {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index: 1,
                                                expected: op0_type,
                                                found: op1_type,
                                            });
                                        }

                                        if !storage_classes_match {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index: 1,
                                                expected: op0_type,
                                                found: op1_type,
                                            });
                                        }
                                    }
                                }

                                match op0_storage_class {
                                    rspirv::spirv::StorageClass::StorageBuffer => {
                                        if !capabilities.contains(
                                            &rspirv::spirv::Capability::VariablePointersStorageBuffer,
                                        ) {
                                            return Err(
                                                ValidationError::PointerComparisonMissingCapability {
                                                    function: function_id,
                                                    block: block_label_id,
                                                    instruction: inst.class.opcode,
                                                    storage_class: op0_storage_class,
                                                    required_capability: rspirv::spirv::Capability::VariablePointersStorageBuffer,
                                                },
                                            );
                                        }
                                    }
                                    rspirv::spirv::StorageClass::Workgroup => {
                                        if !capabilities
                                            .contains(&rspirv::spirv::Capability::VariablePointers)
                                        {
                                            return Err(
                                                ValidationError::PointerComparisonMissingCapability {
                                                    function: function_id,
                                                    block: block_label_id,
                                                    instruction: inst.class.opcode,
                                                    storage_class: op0_storage_class,
                                                    required_capability: rspirv::spirv::Capability::VariablePointers,
                                                },
                                            );
                                        }
                                    }
                                    rspirv::spirv::StorageClass::PhysicalStorageBuffer => {}
                                    _ => {
                                        return Err(
                                            ValidationError::PointerComparisonInvalidStorageClass {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                storage_class: op0_storage_class,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Bitwise ops require integer scalar or vector results.
                    if matches!(
                        inst.class.opcode,
                        rspirv::spirv::Op::BitwiseAnd
                            | rspirv::spirv::Op::BitwiseOr
                            | rspirv::spirv::Op::BitwiseXor
                            | rspirv::spirv::Op::Not
                    ) {
                        let ok = result_type_inst.is_some_and(|ty| match ty.class.opcode {
                            rspirv::spirv::Op::TypeInt => true,
                            rspirv::spirv::Op::TypeVector => {
                                let Some(rspirv::dr::Operand::IdRef(elem)) = ty.operands.first()
                                else {
                                    return false;
                                };
                                ResultId::try_from(*elem)
                                    .ok()
                                    .and_then(|id| definitions.get(&id))
                                    .is_some_and(|elem_inst| {
                                        elem_inst.class.opcode == rspirv::spirv::Op::TypeInt
                                    })
                            }
                            _ => false,
                        });
                        if !ok {
                            return Err(ValidationError::InstructionResultTypeMismatch {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                                expected: result_type,
                                found: result_type,
                            });
                        }
                    }

                    // Integer and float comparisons: result must be bool (scalar or vector of
                    // bool matching operand shape); operands must match each other's type.
                    let int_cmp = matches!(
                        inst.class.opcode,
                        rspirv::spirv::Op::IEqual
                            | rspirv::spirv::Op::INotEqual
                            | rspirv::spirv::Op::UGreaterThan
                            | rspirv::spirv::Op::SGreaterThan
                            | rspirv::spirv::Op::UGreaterThanEqual
                            | rspirv::spirv::Op::SGreaterThanEqual
                            | rspirv::spirv::Op::ULessThan
                            | rspirv::spirv::Op::SLessThan
                            | rspirv::spirv::Op::ULessThanEqual
                            | rspirv::spirv::Op::SLessThanEqual
                    );
                    let float_cmp = matches!(
                        inst.class.opcode,
                        rspirv::spirv::Op::FOrdEqual
                            | rspirv::spirv::Op::FUnordEqual
                            | rspirv::spirv::Op::FOrdNotEqual
                            | rspirv::spirv::Op::FUnordNotEqual
                            | rspirv::spirv::Op::FOrdLessThan
                            | rspirv::spirv::Op::FUnordLessThan
                            | rspirv::spirv::Op::FOrdGreaterThan
                            | rspirv::spirv::Op::FUnordGreaterThan
                            | rspirv::spirv::Op::FOrdLessThanEqual
                            | rspirv::spirv::Op::FUnordLessThanEqual
                            | rspirv::spirv::Op::FOrdGreaterThanEqual
                            | rspirv::spirv::Op::FUnordGreaterThanEqual
                    );
                    if int_cmp || float_cmp {
                        let op_type = inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::IdRef(raw) => ResultId::try_from(*raw).ok(),
                            _ => None,
                        });
                        let op_type_id = op_type.and_then(|rid| result_types.get(&rid).copied());
                        let op_type_inst = op_type_id
                            .and_then(|tid| ResultId::try_from(u32::from(Id::from(tid))).ok())
                            .and_then(|rid| definitions.get(&rid));

                        let (component_count, is_int, is_float) =
                            op_type_inst.map_or((None, false, false), |ty| match ty.class.opcode {
                                rspirv::spirv::Op::TypeInt => (Some(1), true, false),
                                rspirv::spirv::Op::TypeFloat => (Some(1), false, true),
                                rspirv::spirv::Op::TypeVector => {
                                    let count = ty
                                        .operands
                                        .get(1)
                                        .and_then(|op| match op {
                                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                                            _ => None,
                                        })
                                        .map(|v| v as usize);
                                    let elem_ty = ty.operands.first().and_then(|op| match op {
                                        rspirv::dr::Operand::IdRef(raw) => {
                                            ResultId::try_from(*raw).ok()
                                        }
                                        _ => None,
                                    });
                                    let elem_inst = elem_ty
                                        .and_then(|rid| result_types.get(&rid).copied())
                                        .and_then(|tid| {
                                            ResultId::try_from(u32::from(Id::from(tid))).ok()
                                        })
                                        .and_then(|rid| definitions.get(&rid));
                                    let elem_int = elem_inst
                                        .map(|i| i.class.opcode == rspirv::spirv::Op::TypeInt)
                                        .unwrap_or(false);
                                    let elem_float = elem_inst
                                        .map(|i| i.class.opcode == rspirv::spirv::Op::TypeFloat)
                                        .unwrap_or(false);
                                    (count, elem_int, elem_float)
                                }
                                _ => (None, false, false),
                            });

                        // Result type must be bool or vector<bool> matching operand count.
                        let result_ok = result_type_inst.is_some_and(|ty| match ty.class.opcode {
                            rspirv::spirv::Op::TypeBool => component_count == Some(1),
                            rspirv::spirv::Op::TypeVector => {
                                let count = ty.operands.get(1).and_then(|op| match op {
                                    rspirv::dr::Operand::LiteralBit32(v) => Some(*v as usize),
                                    _ => None,
                                });
                                let elem_ty = ty.operands.first().and_then(|op| match op {
                                    rspirv::dr::Operand::IdRef(raw) => {
                                        ResultId::try_from(*raw).ok()
                                    }
                                    _ => None,
                                });
                                let elem_inst = elem_ty
                                    .and_then(|rid| result_types.get(&rid).copied())
                                    .and_then(|tid| {
                                        ResultId::try_from(u32::from(Id::from(tid))).ok()
                                    })
                                    .and_then(|rid| definitions.get(&rid));
                                elem_inst
                                    .map(|i| i.class.opcode == rspirv::spirv::Op::TypeBool)
                                    .unwrap_or(false)
                                    && count == component_count
                            }
                            _ => false,
                        });
                        if !result_ok {
                            return Err(ValidationError::InstructionResultTypeMismatch {
                                function: function_id,
                                block: block_label_id,
                                instruction: inst.class.opcode,
                                expected: result_type,
                                found: result_type,
                            });
                        }

                        // Operand kinds must match and be integer for int cmp, float for float cmp.
                        if int_cmp && !is_int {
                            if let Some(op_type_id) =
                                op_type.and_then(|rid| result_types.get(&rid).copied())
                            {
                                return Err(ValidationError::OperandTypeMismatch {
                                    function: function_id,
                                    block: block_label_id,
                                    instruction: inst.class.opcode,
                                    operand_index: 0,
                                    expected: result_type,
                                    found: op_type_id,
                                });
                            }
                        }
                        if float_cmp && !is_float {
                            if let Some(op_type_id) =
                                op_type.and_then(|rid| result_types.get(&rid).copied())
                            {
                                return Err(ValidationError::OperandTypeMismatch {
                                    function: function_id,
                                    block: block_label_id,
                                    instruction: inst.class.opcode,
                                    operand_index: 0,
                                    expected: result_type,
                                    found: op_type_id,
                                });
                            }
                        }
                        if let Some(op1) = inst.operands.get(1).and_then(|op| match op {
                            rspirv::dr::Operand::IdRef(raw) => ResultId::try_from(*raw).ok(),
                            _ => None,
                        }) {
                            if let Some(op1_type) = result_types.get(&op1).copied() {
                                if let Some(expected) = op_type_id {
                                    if expected != op1_type {
                                        return Err(ValidationError::OperandTypeMismatch {
                                            function: function_id,
                                            block: block_label_id,
                                            instruction: inst.class.opcode,
                                            operand_index: 1,
                                            expected,
                                            found: op1_type,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }

                for (operand_index, operand) in inst.operands.iter().enumerate() {
                    if let rspirv::dr::Operand::IdRef(raw) = operand {
                        if let Ok(result_id) = ResultId::try_from(*raw) {
                            if let Some(Some(def_block)) = definition_blocks.get(&result_id) {
                                if !block_ids.contains(def_block) {
                                    return Err(ValidationError::ValueDefinedInAnotherFunction {
                                        function: function_id,
                                        value: Id::from(result_id),
                                    });
                                }
                                if !block_dominators.contains(def_block) {
                                    return Err(ValidationError::ValueNotDominated {
                                        function: function_id,
                                        block: block_label_id,
                                        value: Id::from(result_id),
                                    });
                                }
                            }
                            if matches!(
                                inst.class.opcode,
                                rspirv::spirv::Op::ShiftLeftLogical
                                    | rspirv::spirv::Op::ShiftRightLogical
                                    | rspirv::spirv::Op::ShiftRightArithmetic
                            ) {
                                let Some(result_type) =
                                    inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                                else {
                                    continue;
                                };
                                let Some(result_inst) = result_type_inst else {
                                    continue;
                                };
                                let Some((result_components, _result_width)) =
                                    integer_shape(result_inst)
                                else {
                                    return Err(ValidationError::InstructionResultTypeMismatch {
                                        function: function_id,
                                        block: block_label_id,
                                        instruction: inst.class.opcode,
                                        expected: result_type,
                                        found: result_type,
                                    });
                                };

                                if let Some(op_type_id) = result_types.get(&result_id).copied() {
                                    if operand_index == 0 {
                                        // Base operand must match result type exactly
                                        if op_type_id != result_type {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index,
                                                expected: result_type,
                                                found: op_type_id,
                                            });
                                        }
                                    } else if operand_index == 1 {
                                        // Shift operand must be int scalar/vector with same
                                        // dimension as result, but bit width can differ
                                        let op_inst =
                                            ResultId::try_from(u32::from(Id::from(op_type_id)))
                                                .ok()
                                                .and_then(|rid| definitions.get(&rid));
                                        let Some((op_components, _op_width)) =
                                            op_inst.and_then(integer_shape)
                                        else {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index,
                                                expected: result_type,
                                                found: op_type_id,
                                            });
                                        };
                                        // Only check dimension matches, not bit width
                                        if op_components != result_components {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index,
                                                expected: result_type,
                                                found: op_type_id,
                                            });
                                        }
                                    }
                                }
                                continue;
                            }
                            // Operand type checks for simple arithmetic/logical ops: operands must
                            // match the result type.
                            if let Some(result_type) =
                                inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                            {
                                let requires_operand_type_match = matches!(
                                    inst.class.opcode,
                                    rspirv::spirv::Op::IAdd
                                        | rspirv::spirv::Op::ISub
                                        | rspirv::spirv::Op::IMul
                                        | rspirv::spirv::Op::SDiv
                                        | rspirv::spirv::Op::UDiv
                                        | rspirv::spirv::Op::SRem
                                        | rspirv::spirv::Op::UMod
                                        | rspirv::spirv::Op::BitwiseAnd
                                        | rspirv::spirv::Op::BitwiseOr
                                        | rspirv::spirv::Op::BitwiseXor
                                        | rspirv::spirv::Op::Not
                                        | rspirv::spirv::Op::LogicalAnd
                                        | rspirv::spirv::Op::LogicalOr
                                        | rspirv::spirv::Op::LogicalNot
                                );
                                if requires_operand_type_match {
                                    if let Some(found_type) = result_types.get(&result_id).copied()
                                    {
                                        if found_type != result_type {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index,
                                                expected: result_type,
                                                found: found_type,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Basic operand type checks for arithmetic instructions: operands must
                // match the instruction's result type.
                if let Some(result_type) =
                    inst.result_type.and_then(|raw| TypeId::try_from(raw).ok())
                {
                    let requires_operand_type_match = matches!(
                        inst.class.opcode,
                        rspirv::spirv::Op::IAdd
                            | rspirv::spirv::Op::ISub
                            | rspirv::spirv::Op::IMul
                            | rspirv::spirv::Op::SDiv
                            | rspirv::spirv::Op::UDiv
                            | rspirv::spirv::Op::SRem
                            | rspirv::spirv::Op::UMod
                    );
                    if requires_operand_type_match {
                        for (operand_index, operand) in inst.operands.iter().enumerate() {
                            if let rspirv::dr::Operand::IdRef(raw) = operand {
                                if let Ok(op_id) = ResultId::try_from(*raw) {
                                    if let Some(found_type) = result_types.get(&op_id).copied() {
                                        if found_type != result_type {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: function_id,
                                                block: block_label_id,
                                                instruction: inst.class.opcode,
                                                operand_index,
                                                expected: result_type,
                                                found: found_type,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
enum CompositeWalkError {
    NotComposite,
    OutOfBounds {
        composite_type: TypeId,
        index_position: usize,
        index: u32,
        bound: u32,
    },
}

fn type_instruction(
    type_id: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<&rspirv::dr::Instruction> {
    let result_id = ResultId::try_from(u32::from(type_id)).ok()?;
    definitions.get(&result_id)
}

fn composite_member_type(
    composite_type: TypeId,
    indexes: &[u32],
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Result<TypeId, CompositeWalkError> {
    if indexes.is_empty() {
        return Err(CompositeWalkError::NotComposite);
    }
    let mut current_type = composite_type;
    for (position, &index) in indexes.iter().enumerate() {
        let Some(inst) = type_instruction(current_type, definitions) else {
            return Err(CompositeWalkError::NotComposite);
        };
        match inst.class.opcode {
            rspirv::spirv::Op::TypeVector | rspirv::spirv::Op::TypeMatrix => {
                let element_type = inst
                    .operands
                    .first()
                    .and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw).ok(),
                        _ => None,
                    })
                    .ok_or(CompositeWalkError::NotComposite)?;
                let bound = inst
                    .operands
                    .get(1)
                    .and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .unwrap_or(0);
                if bound != 0 && index >= bound {
                    return Err(CompositeWalkError::OutOfBounds {
                        composite_type: current_type,
                        index_position: position,
                        index,
                        bound,
                    });
                }
                current_type = element_type;
            }
            rspirv::spirv::Op::TypeArray | rspirv::spirv::Op::TypeRuntimeArray => {
                let element_type = inst
                    .operands
                    .first()
                    .and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw).ok(),
                        _ => None,
                    })
                    .ok_or(CompositeWalkError::NotComposite)?;
                if inst.class.opcode == rspirv::spirv::Op::TypeArray {
                    if let Some(bound) = array_length(inst, definitions) {
                        if index >= bound {
                            return Err(CompositeWalkError::OutOfBounds {
                                composite_type: current_type,
                                index_position: position,
                                index,
                                bound,
                            });
                        }
                    }
                }
                current_type = element_type;
            }
            rspirv::spirv::Op::TypeStruct => {
                let bound = inst.operands.len() as u32;
                if index >= bound {
                    return Err(CompositeWalkError::OutOfBounds {
                        composite_type: current_type,
                        index_position: position,
                        index,
                        bound,
                    });
                }
                let member_type = inst.operands.get(index as usize).and_then(|op| match op {
                    rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw).ok(),
                    _ => None,
                });
                current_type = member_type.ok_or(CompositeWalkError::NotComposite)?;
            }
            _ => return Err(CompositeWalkError::NotComposite),
        }
    }
    Ok(current_type)
}

fn consumed_components_for_type(
    ty: ResultId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    seen: &mut HashSet<ResultId>,
) -> Option<u32> {
    if !seen.insert(ty) {
        return Some(0);
    }
    let inst = definitions.get(&ty)?;
    match inst.class.opcode {
        rspirv::spirv::Op::TypeInt | rspirv::spirv::Op::TypeFloat => Some(1),
        rspirv::spirv::Op::TypeVector => inst.operands.get(1).and_then(|op| match op {
            rspirv::dr::Operand::LiteralBit32(width) => Some(*width),
            _ => None,
        }),
        rspirv::spirv::Op::TypeMatrix => {
            let column_type = inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            })?;
            let columns = inst.operands.get(1).and_then(|op| match op {
                rspirv::dr::Operand::LiteralBit32(count) => Some(*count),
                _ => None,
            })?;
            let mut seen = seen.clone();
            consumed_components_for_type(column_type, definitions, &mut seen)
                .map(|per_column| per_column.saturating_mul(columns))
        }
        rspirv::spirv::Op::TypeArray => {
            let element = inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            })?;
            let length_id = inst.operands.get(1).and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            })?;
            let length = constant_u32_from_defs(definitions, length_id)?;
            let mut seen = seen.clone();
            consumed_components_for_type(element, definitions, &mut seen)
                .map(|per_element| per_element.saturating_mul(length))
        }
        rspirv::spirv::Op::TypeRuntimeArray => None,
        rspirv::spirv::Op::TypeStruct => {
            let mut total: u32 = 0;
            for op in &inst.operands {
                if let rspirv::dr::Operand::IdRef(member) = op {
                    if let Ok(member_id) = ResultId::try_from(*member) {
                        let mut seen = seen.clone();
                        if let Some(components) =
                            consumed_components_for_type(member_id, definitions, &mut seen)
                        {
                            total = total.saturating_add(components);
                        }
                    }
                }
            }
            Some(total)
        }
        _ => Some(1),
    }
}

fn has_patch_decoration(module: &Module, target: ResultId) -> bool {
    let target_raw = target.into_inner().get();
    module.annotations.iter().any(|dec| {
        dec.class.opcode == rspirv::spirv::Op::Decorate
            && matches!(dec.operands.first(), Some(rspirv::dr::Operand::IdRef(id)) if *id == target_raw)
            && matches!(dec.operands.get(1), Some(rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Patch)))
    })
}

fn location_and_component(module: &Module, target: ResultId) -> Option<(u32, u32)> {
    let mut location = None;
    let mut component = 0;
    let target_raw = target.into_inner().get();
    for dec in &module.annotations {
        if dec.class.opcode == rspirv::spirv::Op::Decorate {
            if let (
                Some(rspirv::dr::Operand::IdRef(id)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
            ) = (dec.operands.first(), dec.operands.get(1))
            {
                if *id == target_raw {
                    match decoration {
                        rspirv::spirv::Decoration::Location => {
                            if let Some(rspirv::dr::Operand::LiteralBit32(loc)) =
                                dec.operands.get(2)
                            {
                                location = Some(*loc);
                            }
                        }
                        rspirv::spirv::Decoration::Component => {
                            if let Some(rspirv::dr::Operand::LiteralBit32(comp)) =
                                dec.operands.get(2)
                            {
                                component = *comp;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    location.map(|loc| (loc, component))
}

fn validate_entry_point_locations(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    env: TargetEnv,
) -> Result<(), ValidationError> {
    if !is_vulkan_env(env) {
        return Ok(());
    }

    for ep in &module.entry_points {
        let mut operands = ep.operands.iter();
        if ep.class.opcode == rspirv::spirv::Op::ConditionalEntryPointINTEL {
            let _ = operands.next();
        }
        let execution_model = operands.next().and_then(|op| match op {
            rspirv::dr::Operand::ExecutionModel(model) => Some(*model),
            _ => None,
        });
        match execution_model {
            Some(
                rspirv::spirv::ExecutionModel::Vertex
                | rspirv::spirv::ExecutionModel::TessellationControl
                | rspirv::spirv::ExecutionModel::TessellationEvaluation
                | rspirv::spirv::ExecutionModel::Geometry
                | rspirv::spirv::ExecutionModel::Fragment,
            ) => {}
            _ => continue,
        }
        let entry_point_id = operands
            .next()
            .and_then(|op| match op {
                rspirv::dr::Operand::IdRef(ep_id) => Id::try_from(*ep_id).ok(),
                _ => None,
            })
            .ok_or(ValidationError::InvalidEntryPointOperand)?;
        // Skip name.
        let operands = operands.skip(1);
        let mut input_locs = HashSet::new();
        let mut output_locs = HashSet::new();
        let mut patch_input_locs = HashSet::new();
        let mut patch_output_locs = HashSet::new();
        for operand in operands {
            let interface_id = match operand {
                rspirv::dr::Operand::IdRef(id) => *id,
                _ => continue,
            };
            let interface_id = match ResultId::try_from(interface_id) {
                Ok(id) => id,
                Err(_) => continue,
            };
            let storage_class = definitions
                .get(&interface_id)
                .and_then(|inst| inst.operands.first())
                .and_then(|op| match op {
                    rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                    _ => None,
                });
            let storage_class = match storage_class {
                Some(sc)
                    if sc == rspirv::spirv::StorageClass::Input
                        || sc == rspirv::spirv::StorageClass::Output =>
                {
                    sc
                }
                _ => continue,
            };
            let (location, component) = match location_and_component(module, interface_id) {
                Some(loc) => loc,
                None => continue,
            };
            let pointer_type = definitions
                .get(&interface_id)
                .and_then(|inst| inst.result_type)
                .and_then(|ty| ResultId::try_from(ty).ok());
            let pointee_type = pointer_type
                .and_then(|ptr| definitions.get(&ptr))
                .and_then(|ptr_inst| ptr_inst.operands.get(1))
                .and_then(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                    _ => None,
                });
            let consumed = pointee_type
                .and_then(|ty| consumed_components_for_type(ty, definitions, &mut HashSet::new()))
                .unwrap_or(1);
            let is_patch = has_patch_decoration(module, interface_id);
            let loc_set = match (storage_class, is_patch) {
                (rspirv::spirv::StorageClass::Input, true) => &mut patch_input_locs,
                (rspirv::spirv::StorageClass::Input, false) => &mut input_locs,
                (rspirv::spirv::StorageClass::Output, true) => &mut patch_output_locs,
                (rspirv::spirv::StorageClass::Output, false) => &mut output_locs,
                _ => unreachable!(),
            };
            let start_index = location.saturating_mul(4).saturating_add(component);
            for offset in 0..consumed {
                let linear = start_index.saturating_add(offset);
                let loc_component = (linear / 4, linear % 4);
                if !loc_set.insert(loc_component) {
                    return Err(ValidationError::EntryPointInterfaceLocationConflict {
                        entry_point: entry_point_id,
                        storage_class,
                        location: loc_component.0,
                        component: loc_component.1,
                    });
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
struct FunctionSignature {
    return_type: TypeId,
    parameter_types: Vec<TypeId>,
}

fn validate_function_call(
    function_id: Id,
    call: &rspirv::dr::Instruction,
    signatures: &HashMap<Id, FunctionSignature>,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    result_types: &HashMap<ResultId, TypeId>,
) -> Result<(), ValidationError> {
    let Some(rspirv::dr::Operand::IdRef(raw_callee)) = call.operands.first() else {
        return Err(ValidationError::ZeroId {
            kind: IdKind::Operand,
            opcode: call.class.opcode,
        });
    };
    let callee_id = Id::try_from(*raw_callee).map_err(|_| ValidationError::ZeroId {
        kind: IdKind::Operand,
        opcode: call.class.opcode,
    })?;
    let callee_result_id =
        ResultId::try_from(*raw_callee).map_err(|_| ValidationError::ZeroId {
            kind: IdKind::Operand,
            opcode: call.class.opcode,
        })?;
    let callee_inst = definitions
        .get(&callee_result_id)
        .ok_or(ValidationError::UndefinedId {
            function: Some(function_id),
            id: callee_id,
        })?;
    if callee_inst.class.opcode != rspirv::spirv::Op::Function {
        return Err(ValidationError::FunctionCallTargetNotFunction {
            function: function_id,
            target: callee_id,
            found: callee_inst.class.opcode,
        });
    }
    let signature =
        signatures
            .get(&callee_id)
            .ok_or(ValidationError::FunctionCallTargetNotFunction {
                function: function_id,
                target: callee_id,
                found: callee_inst.class.opcode,
            })?;

    let call_result_type = call
        .result_type
        .and_then(|raw| TypeId::try_from(raw).ok())
        .ok_or(ValidationError::ZeroId {
            kind: IdKind::ResultType,
            opcode: call.class.opcode,
        })?;
    if call_result_type != signature.return_type {
        return Err(ValidationError::FunctionCallResultTypeMismatch {
            function: function_id,
            expected: signature.return_type,
            found: call_result_type,
        });
    }

    let provided_args = call.operands.iter().skip(1).collect::<Vec<_>>();
    if provided_args.len() != signature.parameter_types.len() {
        return Err(ValidationError::FunctionCallArgumentCountMismatch {
            function: function_id,
            expected: signature.parameter_types.len(),
            found: provided_args.len(),
        });
    }

    for (expected_type, operand) in signature.parameter_types.iter().zip(provided_args) {
        let raw_id = match operand {
            rspirv::dr::Operand::IdRef(raw) => *raw,
            _ => {
                return Err(ValidationError::ZeroId {
                    kind: IdKind::Operand,
                    opcode: call.class.opcode,
                });
            }
        };
        let arg_id = ResultId::try_from(raw_id).map_err(|_| ValidationError::ZeroId {
            kind: IdKind::Operand,
            opcode: call.class.opcode,
        })?;
        let arg_type = result_types
            .get(&arg_id)
            .ok_or(ValidationError::UndefinedId {
                function: Some(function_id),
                id: Id::from(arg_id),
            })?;
        if arg_type != expected_type {
            return Err(ValidationError::FunctionCallArgumentTypeMismatch {
                function: function_id,
                argument: Id::from(arg_id),
                expected: *expected_type,
                found: *arg_type,
            });
        }
    }

    Ok(())
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

    Ok(FunctionSignature {
        return_type,
        parameter_types: expected_params,
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
    match opcode {
        Capability | ConditionalCapabilityINTEL => return Section::Capabilities,
        Extension | ConditionalExtensionINTEL => return Section::Extensions,
        _ => {}
    }
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
        // Instruction capabilities are disjunctive - you need AT LEAST ONE from the list
        if !inst.class.capabilities.is_empty() {
            let has_any_capability = inst.class.capabilities.iter().any(|&required_cap| {
                capability_satisfied(required_cap, capabilities)
            });
            // Special case for PtrDiff which has alternative capabilities
            let ptrdiff_alternative = inst.class.opcode == rspirv::spirv::Op::PtrDiff
                && (capabilities.contains(&rspirv::spirv::Capability::UntypedPointersKHR)
                    || capabilities.contains(&rspirv::spirv::Capability::PhysicalStorageBufferAddresses));
            if !has_any_capability && !ptrdiff_alternative {
                // Report the first required capability for the error message
                return Err(ValidationError::MissingInstructionCapability {
                    opcode: inst.class.opcode,
                    required_capability: inst.class.capabilities[0],
                });
            }
        }
        // Instruction extensions are disjunctive - you need AT LEAST ONE from the list
        if !inst.class.extensions.is_empty() {
            let has_any_extension = inst.class.extensions.iter().any(|&required_ext| {
                extension_satisfied(required_ext, extensions, target_version)
            });
            if !has_any_extension {
                return Err(ValidationError::MissingInstructionExtension {
                    opcode: inst.class.opcode,
                    required_extension: ExtensionName::from(inst.class.extensions[0]),
                });
            }
        }
        if let Some(required_version) = required_spirv_version_for_opcode(inst.class.opcode) {
            if target_version < required_version {
                // Check if an enabling extension is available and declared
                let has_extension_from_inst = !inst.class.extensions.is_empty()
                    && inst.class.extensions.iter().any(|&ext| {
                        extension_satisfied(ext, extensions, target_version)
                    });
                // Also check if any of the instruction's required capabilities have enabling extensions
                let has_extension_from_cap = !inst.class.capabilities.is_empty()
                    && inst.class.capabilities.iter().any(|&cap| {
                        if let Some(ext) = required_extension_for_capability(cap) {
                            extension_satisfied(ext, extensions, target_version)
                        } else {
                            false
                        }
                    });
                // Only error if no enabling extension is declared
                if !has_extension_from_inst && !has_extension_from_cap {
                    return Err(ValidationError::InstructionRequiresSpirvVersion {
                        opcode: inst.class.opcode,
                        required_version,
                        target_version,
                    });
                }
            }
        }
        for (index, operand) in inst.operands.iter().enumerate() {
            let resolved_operand = resolve_id_operand(module, operand);
            let operand = resolved_operand.as_ref().unwrap_or(operand);
            if (inst.class.opcode == rspirv::spirv::Op::ExecutionMode
                || inst.class.opcode == rspirv::spirv::Op::ExecutionModeId)
                && matches!(
                    operand,
                    rspirv::dr::Operand::ExecutionMode(
                        rspirv::spirv::ExecutionMode::OutputVertices
                            | rspirv::spirv::ExecutionMode::OutputPrimitivesEXT
                            | rspirv::spirv::ExecutionMode::OutputLinesEXT
                            | rspirv::spirv::ExecutionMode::OutputTrianglesEXT
                    )
                )
            {
                // Output* execution modes are validated against the entry point execution model;
                // skipping operand capability checks avoids over-constraining mesh/geometry/tess.
                continue;
            }
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
            // Collect ALL capabilities from all sources and check DISJUNCTIVELY
            // (like C++ spirv-val's HasAnyOfCapabilities).
            // The grammar lists alternatives like [Kernel, GroupNonUniformArithmetic, GroupNonUniformBallot]
            // and you need AT LEAST ONE from the combined set.
            let mut all_required_caps: Vec<rspirv::spirv::Capability> = Vec::new();
            all_required_caps.extend(operand.required_capabilities());
            all_required_caps.extend(manual_required_capabilities_for_operand(operand).iter().copied());
            all_required_caps.extend(grammar_required_capabilities_for_operand(operand));

            if !all_required_caps.is_empty() {
                let has_any = all_required_caps.iter().any(|&cap| capability_satisfied(cap, capabilities));
                if !has_any {
                    return Err(ValidationError::MissingOperandCapability {
                        opcode: inst.class.opcode,
                        operand_index: index,
                        required_capability: all_required_caps[0],
                    });
                }
            }
            for required_ext in operand.required_extensions() {
                if !extension_satisfied(required_ext, extensions, target_version) {
                    return Err(ValidationError::MissingOperandExtension {
                        opcode: inst.class.opcode,
                        operand_index: index,
                        required_extension: ExtensionName::from(required_ext),
                    });
                }
            }
            for required_ext in grammar_required_extensions_for_operand(operand) {
                if !extension_satisfied(required_ext, extensions, target_version) {
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
        rspirv::dr::Operand::ImageOperands(operands)
            if operands.intersects(
                rspirv::spirv::ImageOperands::MAKE_TEXEL_VISIBLE
                    | rspirv::spirv::ImageOperands::MAKE_TEXEL_AVAILABLE,
            ) =>
        {
            &[rspirv::spirv::Capability::VulkanMemoryModel]
        }
        rspirv::dr::Operand::ImageOperands(operands)
            if operands.contains(rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL) =>
        {
            &[rspirv::spirv::Capability::VulkanMemoryModel]
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

fn matrix_details(
    type_id: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<(TypeId, u32, u32, ResultId)> {
    let matrix_result = ResultId::try_from(u32::from(type_id)).ok()?;
    let inst = definitions.get(&matrix_result)?;
    if inst.class.opcode != rspirv::spirv::Op::TypeMatrix {
        return None;
    }
    let (column_type, columns) = matrix_info(inst);
    let column_type = column_type?;
    let columns = columns?;
    let column_result = ResultId::try_from(u32::from(column_type)).ok()?;
    let column_inst = definitions.get(&column_result)?;
    if column_inst.class.opcode != rspirv::spirv::Op::TypeVector {
        return None;
    }
    let (component_type, rows) = vector_info(column_inst);
    Some((component_type?, rows?, columns, column_result))
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
    for (struct_id, block_info) in block_structs {
        // Only validate block layout for structs used in storage classes that require it.
        // Per C++ spirv-val (validate_decorations.cpp lines 1365-1369):
        // - Block + Uniform → block rules
        // - BufferBlock + Uniform → buffer rules
        // - Block + (PushConstant | StorageBuffer | PhysicalStorageBuffer | Workgroup) → buffer rules
        // Structs in other storage classes (Output, Input, etc.) don't require offset decorations.
        if !block_info.requires_block_layout() {
            continue;
        }

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
        if block_info
            .storage_classes
            .contains(&rspirv::spirv::StorageClass::Workgroup)
            && !options.workgroup_scalar_block_layout
        {
            // No additional action; reserved for future stricter checks.
        }
    }

    Ok(())
}

/// Which block decoration a struct has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockDecoration {
    Block,
    BufferBlock,
}

/// Information about a struct that may require block layout validation.
#[derive(Debug, Clone)]
struct BlockStructInfo {
    decoration: BlockDecoration,
    storage_classes: HashSet<rspirv::spirv::StorageClass>,
}

impl BlockStructInfo {
    fn new(decoration: BlockDecoration) -> Self {
        Self {
            decoration,
            storage_classes: HashSet::new(),
        }
    }

    /// Returns true if block layout rules apply to this struct based on its
    /// decoration and the storage classes where it's used.
    ///
    /// From the C++ spirv-val (validate_decorations.cpp lines 1365-1369):
    /// - Block rules: Uniform storage class + Block decoration
    /// - Buffer rules: (Uniform + BufferBlock) OR
    ///                 ((PushConstant | StorageBuffer | PhysicalStorageBuffer | Workgroup) + Block)
    fn requires_block_layout(&self) -> bool {
        use rspirv::spirv::StorageClass;

        let has_uniform = self.storage_classes.contains(&StorageClass::Uniform);
        let has_push_constant = self.storage_classes.contains(&StorageClass::PushConstant);
        let has_storage_buffer = self.storage_classes.contains(&StorageClass::StorageBuffer);
        let has_phys_storage_buffer = self
            .storage_classes
            .contains(&StorageClass::PhysicalStorageBuffer);
        let has_workgroup = self.storage_classes.contains(&StorageClass::Workgroup);

        match self.decoration {
            BlockDecoration::Block => {
                // Block rules apply for Uniform
                // Buffer rules apply for PushConstant, StorageBuffer, PhysicalStorageBuffer, Workgroup
                has_uniform
                    || has_push_constant
                    || has_storage_buffer
                    || has_phys_storage_buffer
                    || has_workgroup
            }
            BlockDecoration::BufferBlock => {
                // BufferBlock rules only apply with Uniform storage class
                has_uniform
            }
        }
    }
}

fn collect_block_structs(module: &Module) -> HashMap<ResultId, BlockStructInfo> {
    let mut structs: HashMap<ResultId, BlockStructInfo> = HashMap::new();

    // First pass: collect all Block/BufferBlock decorated structs
    for inst in &module.annotations {
        if inst.class.opcode == rspirv::spirv::Op::Decorate {
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
            ) = (inst.operands.first(), inst.operands.get(1))
            {
                let block_deco = match *decoration {
                    rspirv::spirv::Decoration::Block => Some(BlockDecoration::Block),
                    rspirv::spirv::Decoration::BufferBlock => Some(BlockDecoration::BufferBlock),
                    _ => None,
                };
                if let Some(deco) = block_deco {
                    if let Ok(struct_id) = ResultId::try_from(*target) {
                        structs.entry(struct_id).or_insert_with(|| BlockStructInfo::new(deco));
                    }
                }
            }
        }
    }

    // Second pass: map struct ids to storage classes where they are used
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
            if let Some(info) = structs.get_mut(&struct_id) {
                info.storage_classes.insert(*sc);
            }
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

    // Only bit field and bit count operations have the 32-bit restriction in Vulkan.
    // Shift operations and basic bitwise operations (Or, Xor, And, Not) do NOT have
    // this restriction.
    let restricted_opcodes = [
        rspirv::spirv::Op::BitFieldInsert,
        rspirv::spirv::Op::BitFieldSExtract,
        rspirv::spirv::Op::BitFieldUExtract,
        rspirv::spirv::Op::BitReverse,
        rspirv::spirv::Op::BitCount,
    ];

    for inst in module.all_inst_iter() {
        if !restricted_opcodes.contains(&inst.class.opcode) {
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

fn has_decoration(module: &Module, target: u32, decoration: rspirv::spirv::Decoration) -> bool {
    module.annotations.iter().any(|inst| {
        inst.class.opcode == rspirv::spirv::Op::Decorate
            && matches!(
                (inst.operands.first(), inst.operands.get(1)),
                (
                    Some(rspirv::dr::Operand::IdRef(id)),
                    Some(rspirv::dr::Operand::Decoration(dec))
                ) if *id == target && *dec == decoration
            )
    })
}

fn contains_sized_int_or_float(
    type_id: TypeId,
    target_opcode: rspirv::spirv::Op,
    width: u32,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visited: &mut HashSet<TypeId>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }
    let Ok(result_id) = ResultId::try_from(u32::from(type_id)) else {
        return false;
    };
    let Some(inst) = definitions.get(&result_id) else {
        return false;
    };
    match inst.class.opcode {
        rspirv::spirv::Op::TypeInt if target_opcode == rspirv::spirv::Op::TypeInt => {
            inst.operands.iter().any(|op| match op {
                rspirv::dr::Operand::LiteralBit32(bits) => *bits == width,
                rspirv::dr::Operand::LiteralBit64(bits) => *bits as u32 == width,
                _ => false,
            })
        }
        rspirv::spirv::Op::TypeFloat if target_opcode == rspirv::spirv::Op::TypeFloat => {
            inst.operands.iter().any(|op| match op {
                rspirv::dr::Operand::LiteralBit32(bits) => *bits == width,
                rspirv::dr::Operand::LiteralBit64(bits) => *bits as u32 == width,
                _ => false,
            })
        }
        rspirv::spirv::Op::TypeVector
        | rspirv::spirv::Op::TypeMatrix
        | rspirv::spirv::Op::TypeArray
        | rspirv::spirv::Op::TypeRuntimeArray
        | rspirv::spirv::Op::TypeStruct
        | rspirv::spirv::Op::TypePointer => inst.operands.iter().any(|op| {
            if let rspirv::dr::Operand::IdRef(raw) = op {
                if let Ok(child) = TypeId::try_from(*raw) {
                    return contains_sized_int_or_float(
                        child,
                        target_opcode,
                        width,
                        definitions,
                        visited,
                    );
                }
            }
            false
        }),
        _ => false,
    }
}

fn enforce_small_type_storage_capabilities(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    capabilities: &HashSet<rspirv::spirv::Capability>,
) -> Result<(), ValidationError> {
    use rspirv::spirv::{Capability, Decoration, Op, StorageClass};

    if !capabilities.contains(&Capability::Shader) {
        return Ok(());
    }

    for inst in &module.types_global_values {
        if inst.class.opcode != Op::Variable && inst.class.opcode != Op::UntypedVariableKHR {
            continue;
        }
        let Some(raw_type) = inst.result_type else {
            continue;
        };
        let Ok(ptr_type_id) = TypeId::try_from(raw_type) else {
            continue;
        };
        let Some(ptr_type_inst) = ResultId::try_from(u32::from(ptr_type_id))
            .ok()
            .and_then(|rid| definitions.get(&rid))
        else {
            continue;
        };
        if ptr_type_inst.class.opcode != Op::TypePointer {
            continue;
        }
        let storage_class = match ptr_type_inst.operands.first() {
            Some(rspirv::dr::Operand::StorageClass(class)) => *class,
            _ => continue,
        };
        let pointee = match ptr_type_inst.operands.get(1) {
            Some(rspirv::dr::Operand::IdRef(raw)) => match TypeId::try_from(*raw) {
                Ok(id) => id,
                Err(_) => continue,
            },
            _ => continue,
        };

        let contains_int = |width: u32| -> bool {
            let mut visited = HashSet::new();
            contains_sized_int_or_float(pointee, Op::TypeInt, width, definitions, &mut visited)
        };
        let contains_float = |width: u32| -> bool {
            let mut visited = HashSet::new();
            contains_sized_int_or_float(pointee, Op::TypeFloat, width, definitions, &mut visited)
        };

        for bit_width in [8u32, 16u32] {
            // The storage class restrictions only apply when the module does NOT
            // declare the base capability for the type width. If Int8 is declared,
            // 8-bit integers can be used in any storage class. Similarly for Int16/Float16.
            // See C++ validator: validate_memory.cpp
            if bit_width == 8 {
                // If Int8 is declared, skip all 8-bit storage class checks
                if capabilities.contains(&Capability::Int8) {
                    continue;
                }
            } else {
                // bit_width == 16
                // If Int16 is declared, skip 16-bit int storage class checks
                // If Float16 is declared, skip 16-bit float storage class checks
                let has_int16_cap = capabilities.contains(&Capability::Int16);
                let has_float16_cap = capabilities.contains(&Capability::Float16);
                let has_int16 = contains_int(16);
                let has_float16 = contains_float(16);

                // Skip if all 16-bit types in this variable are covered by capabilities
                let int16_ok = !has_int16 || has_int16_cap;
                let float16_ok = !has_float16 || has_float16_cap;
                if int16_ok && float16_ok {
                    continue;
                }
            }

            let has_width =
                contains_int(bit_width) || (bit_width == 16 && contains_float(bit_width));
            if !has_width {
                continue;
            }

            let require_capability = |cap: Capability| -> Result<(), ValidationError> {
                if capabilities.contains(&cap) {
                    Ok(())
                } else {
                    Err(ValidationError::SmallTypeMissingCapability {
                        bit_width,
                        storage_class,
                        required_capability: cap,
                    })
                }
            };

            match storage_class {
                StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer => {
                    let required = if bit_width == 8 {
                        Capability::StorageBuffer8BitAccess
                    } else {
                        Capability::StorageBuffer16BitAccess
                    };
                    require_capability(required)?
                }
                StorageClass::Uniform => {
                    let (primary, fallback) = if bit_width == 8 {
                        (
                            Capability::UniformAndStorageBuffer8BitAccess,
                            Capability::StorageBuffer8BitAccess,
                        )
                    } else {
                        (
                            Capability::UniformAndStorageBuffer16BitAccess,
                            Capability::StorageBuffer16BitAccess,
                        )
                    };
                    if capabilities.contains(&primary) {
                        continue;
                    }
                    if capabilities.contains(&fallback)
                        && has_decoration(module, u32::from(pointee), Decoration::BufferBlock)
                    {
                        continue;
                    }
                    return Err(ValidationError::SmallTypeMissingCapability {
                        bit_width,
                        storage_class,
                        required_capability: primary,
                    });
                }
                StorageClass::PushConstant => {
                    let required = if bit_width == 8 {
                        Capability::StoragePushConstant8
                    } else {
                        Capability::StoragePushConstant16
                    };
                    require_capability(required)?
                }
                StorageClass::Input | StorageClass::Output => {
                    if bit_width == 16 {
                        require_capability(Capability::StorageInputOutput16)?
                    } else {
                        return Err(ValidationError::SmallTypeDisallowedInStorageClass {
                            bit_width,
                            storage_class,
                        });
                    }
                }
                StorageClass::Workgroup => {
                    let required = if bit_width == 8 {
                        Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR
                    } else {
                        Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR
                    };
                    require_capability(required)?
                }
                _ => {
                    return Err(ValidationError::SmallTypeDisallowedInStorageClass {
                        bit_width,
                        storage_class,
                    })
                }
            }
        }
    }

    Ok(())
}

fn enforce_decoration_versions(
    module: &Module,
    target_version: SpirvVersion,
) -> Result<(), ValidationError> {
    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        if let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) {
            if *decoration == rspirv::spirv::Decoration::BufferBlock
                && target_version > SpirvVersion::new(1, 3)
            {
                return Err(ValidationError::DecorationRequiresSpirvVersion {
                    decoration: *decoration,
                    required_version: SpirvVersion::new(1, 3),
                    target_version,
                });
            }
        }
    }
    Ok(())
}

fn enforce_block_storage_classes(
    module: &Module,
    target_version: SpirvVersion,
) -> Result<(), ValidationError> {
    use rspirv::spirv::{Decoration, StorageClass};

    let mut blocks: HashMap<ResultId, (Decoration, HashSet<StorageClass>)> = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        if let (
            Some(rspirv::dr::Operand::IdRef(target)),
            Some(rspirv::dr::Operand::Decoration(decoration)),
        ) = (inst.operands.first(), inst.operands.get(1))
        {
            if *decoration == Decoration::Block || *decoration == Decoration::BufferBlock {
                if let Ok(id) = ResultId::try_from(*target) {
                    blocks.entry(id).or_insert((*decoration, HashSet::new()));
                }
            }
        }
    }

    if blocks.is_empty() {
        return Ok(());
    }

    for var in &module.types_global_values {
        if var.class.opcode != rspirv::spirv::Op::Variable {
            continue;
        }
        let Some(rspirv::dr::Operand::StorageClass(storage_class)) = var.operands.first() else {
            continue;
        };
        let Some(result_type) = var.result_type else {
            continue;
        };
        let Ok(ptr_type) = ResultId::try_from(result_type) else {
            continue;
        };
        let Some(ptr_inst) = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(u32::from(ptr_type)))
        else {
            continue;
        };
        if ptr_inst.class.opcode != rspirv::spirv::Op::TypePointer {
            continue;
        }
        let pointee = match ptr_inst.operands.get(1) {
            Some(rspirv::dr::Operand::IdRef(raw)) => match ResultId::try_from(*raw) {
                Ok(id) => id,
                Err(_) => continue,
            },
            _ => continue,
        };
        if let Some((decoration, classes)) = blocks.get_mut(&pointee) {
            classes.insert(*storage_class);
            // Early version gate: BufferBlock was replaced after 1.3.
            if *decoration == Decoration::BufferBlock && target_version > SpirvVersion::new(1, 3) {
                return Err(ValidationError::DecorationRequiresSpirvVersion {
                    decoration: *decoration,
                    required_version: SpirvVersion::new(1, 3),
                    target_version,
                });
            }
        }
    }

    for (block_id, (decoration, storage_classes)) in blocks {
        if storage_classes.is_empty() {
            continue;
        }
        for storage_class in storage_classes {
            let allowed = match decoration {
                Decoration::Block => matches!(
                    storage_class,
                    StorageClass::Uniform
                        | StorageClass::StorageBuffer
                        | StorageClass::PhysicalStorageBuffer
                        | StorageClass::PushConstant
                ),
                Decoration::BufferBlock => matches!(
                    storage_class,
                    StorageClass::Uniform
                        | StorageClass::StorageBuffer
                        | StorageClass::PhysicalStorageBuffer
                ),
                _ => true,
            };
            if !allowed {
                return Err(ValidationError::InvalidBlockDecorationStorageClass {
                    decoration,
                    storage_class,
                });
            }

            // PushConstant requires Block, never BufferBlock.
            if storage_class == StorageClass::PushConstant && decoration == Decoration::BufferBlock
            {
                return Err(ValidationError::InvalidBlockDecorationStorageClass {
                    decoration,
                    storage_class,
                });
            }
        }
        // Silence unused warning when block_id isn't used otherwise.
        let _ = block_id;
    }

    Ok(())
}

fn enforce_descriptor_storage_classes(module: &Module) -> Result<(), ValidationError> {
    use rspirv::spirv::Decoration;
    let mut decorated_vars: HashMap<ResultId, rspirv::spirv::StorageClass> = HashMap::new();

    for inst in &module.types_global_values {
        if inst.class.opcode != rspirv::spirv::Op::Variable
            && inst.class.opcode != rspirv::spirv::Op::UntypedVariableKHR
        {
            continue;
        }
        let Some(rspirv::dr::Operand::StorageClass(sc)) = inst.operands.first() else {
            continue;
        };
        if let Some(result_id) = inst.result_id {
            if let Ok(id) = ResultId::try_from(result_id) {
                decorated_vars.insert(id, *sc);
            }
        }
    }

    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
            continue;
        };
        let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
            continue;
        };
        if *decoration != Decoration::Binding && *decoration != Decoration::DescriptorSet {
            continue;
        }
        let Ok(var_id) = ResultId::try_from(*target) else {
            continue;
        };
        let Some(storage_class) = decorated_vars.get(&var_id) else {
            continue;
        };

        let allowed = matches!(
            storage_class,
            rspirv::spirv::StorageClass::UniformConstant
                | rspirv::spirv::StorageClass::Uniform
                | rspirv::spirv::StorageClass::StorageBuffer
                | rspirv::spirv::StorageClass::PhysicalStorageBuffer
        );
        if !allowed {
            return Err(ValidationError::InvalidDescriptorStorageClass {
                storage_class: *storage_class,
            });
        }
    }

    Ok(())
}

fn enforce_descriptor_requirements(module: &Module, env: TargetEnv) -> Result<(), ValidationError> {
    use rspirv::spirv::Decoration;

    if !is_vulkan_env(env) {
        return Ok(());
    }

    let interface_vars: HashSet<ResultId> = module
        .entry_points
        .iter()
        .flat_map(|ep| ep.operands.iter().skip(2))
        .filter_map(|op| match op {
            rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
            _ => None,
        })
        .collect();
    let decoration_lookup = build_decoration_lookup(module);
    for var in &module.types_global_values {
        if var.class.opcode != rspirv::spirv::Op::Variable
            && var.class.opcode != rspirv::spirv::Op::UntypedVariableKHR
        {
            continue;
        }
        let Some(raw_id) = var.result_id else {
            continue;
        };
        let Some(rid) = ResultId::try_from(raw_id).ok() else {
            continue;
        };
        if !interface_vars.contains(&rid) {
            continue;
        }
        let Some(storage_class) = var.operands.first().and_then(|op| match op {
            rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
            _ => None,
        }) else {
            continue;
        };
        if !matches!(
            storage_class,
            rspirv::spirv::StorageClass::UniformConstant
                | rspirv::spirv::StorageClass::Uniform
                | rspirv::spirv::StorageClass::StorageBuffer
                | rspirv::spirv::StorageClass::PhysicalStorageBuffer
        ) {
            continue;
        }
        let decos = decoration_lookup.get(&rid).cloned().unwrap_or_default();
        if decos.contains(&Decoration::BuiltIn) {
            continue;
        }
        let has_descriptor_set = decos.contains(&Decoration::DescriptorSet);
        let has_binding = decos.contains(&Decoration::Binding);
        if !has_descriptor_set {
            return Err(ValidationError::MissingDescriptorSetDecoration {
                variable: Id::from(rid),
            });
        }
        if !has_binding {
            return Err(ValidationError::MissingBindingDecoration {
                variable: Id::from(rid),
            });
        }
    }

    Ok(())
}

fn has_block_decoration(module: &Module, type_id: ResultId) -> bool {
    module.annotations.iter().any(|inst| {
        inst.class.opcode == rspirv::spirv::Op::Decorate
            && matches!(
                (inst.operands.first(), inst.operands.get(1)),
                (
                    Some(rspirv::dr::Operand::IdRef(target)),
                    Some(rspirv::dr::Operand::Decoration(dec))
                ) if *target == u32::from(type_id)
                    && (*dec == rspirv::spirv::Decoration::Block
                        || *dec == rspirv::spirv::Decoration::BufferBlock)
            )
    })
}

fn enforce_struct_block_requirements(
    module: &Module,
    target_version: SpirvVersion,
) -> Result<(), ValidationError> {
    for var in &module.types_global_values {
        if var.class.opcode != rspirv::spirv::Op::Variable {
            continue;
        }
        let Some(storage_class) = var.operands.first().and_then(|op| match op {
            rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
            _ => None,
        }) else {
            continue;
        };
        if !matches!(
            storage_class,
            rspirv::spirv::StorageClass::Uniform
                | rspirv::spirv::StorageClass::StorageBuffer
                | rspirv::spirv::StorageClass::PhysicalStorageBuffer
                | rspirv::spirv::StorageClass::PushConstant
        ) {
            continue;
        }
        let Some(ptr_type) = var.result_type else {
            continue;
        };
        let Ok(ptr_id) = ResultId::try_from(ptr_type) else {
            continue;
        };
        let Some(ptr_inst) = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(u32::from(ptr_id)))
        else {
            continue;
        };
        if ptr_inst.class.opcode != rspirv::spirv::Op::TypePointer {
            continue;
        }
        let pointee = match ptr_inst.operands.get(1) {
            Some(rspirv::dr::Operand::IdRef(id)) => match ResultId::try_from(*id) {
                Ok(id) => id,
                Err(_) => continue,
            },
            _ => continue,
        };
        let Some(type_inst) = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(u32::from(pointee)))
        else {
            continue;
        };
        if type_inst.class.opcode != rspirv::spirv::Op::TypeStruct {
            continue;
        }

        let has_block = has_block_decoration(module, pointee);
        if has_block {
            if storage_class == rspirv::spirv::StorageClass::PushConstant {
                // Must be Block, not BufferBlock.
                let block_only = module.annotations.iter().any(|inst| {
                    inst.class.opcode == rspirv::spirv::Op::Decorate
                        && matches!(
                        (inst.operands.first(), inst.operands.get(1)),
                                (
                                    Some(rspirv::dr::Operand::IdRef(target)),
                                    Some(rspirv::dr::Operand::Decoration(dec))
                                ) if *target == u32::from(pointee)
                                    && *dec == rspirv::spirv::Decoration::BufferBlock
                            )
                });
                if block_only {
                    return Err(ValidationError::InvalidBlockDecorationStorageClass {
                        decoration: rspirv::spirv::Decoration::BufferBlock,
                        storage_class,
                    });
                }
            }
            if target_version > SpirvVersion::new(1, 3)
                && storage_class != rspirv::spirv::StorageClass::PushConstant
            {
                let buffer_block = module.annotations.iter().any(|inst| {
                    inst.class.opcode == rspirv::spirv::Op::Decorate
                        && matches!(
                        (inst.operands.first(), inst.operands.get(1)),
                                (
                                    Some(rspirv::dr::Operand::IdRef(target)),
                                    Some(rspirv::dr::Operand::Decoration(dec))
                                ) if *target == u32::from(pointee)
                                    && *dec == rspirv::spirv::Decoration::BufferBlock
                            )
                });
                if buffer_block {
                    return Err(ValidationError::DecorationRequiresSpirvVersion {
                        decoration: rspirv::spirv::Decoration::BufferBlock,
                        required_version: SpirvVersion::new(1, 3),
                        target_version,
                    });
                }
            }
            continue;
        }

        return Err(ValidationError::MissingBlockDecoration { storage_class });
    }
    Ok(())
}

fn enforce_location_storage_classes(module: &Module) -> Result<(), ValidationError> {
    use rspirv::spirv::{Decoration, StorageClass};

    // Map variables to their storage class.
    let mut var_storage: HashMap<ResultId, StorageClass> = HashMap::new();
    let mut has_location: HashSet<ResultId> = HashSet::new();
    for inst in &module.types_global_values {
        if inst.class.opcode != rspirv::spirv::Op::Variable {
            continue;
        }
        if let (Some(result_id), Some(rspirv::dr::Operand::StorageClass(sc))) =
            (inst.result_id, inst.operands.first())
        {
            if let Ok(id) = ResultId::try_from(result_id) {
                var_storage.insert(id, *sc);
            }
        }
    }
    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
            continue;
        };
        let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
            continue;
        };
        if *decoration == Decoration::Location {
            if let Ok(var_id) = ResultId::try_from(*target) {
                has_location.insert(var_id);
            }
        }
    }

    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
            continue;
        };
        let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
            continue;
        };
        if *decoration != Decoration::Location && *decoration != Decoration::Component {
            continue;
        }
        let Ok(var_id) = ResultId::try_from(*target) else {
            continue;
        };
        let Some(storage_class) = var_storage.get(&var_id) else {
            continue;
        };
        if !matches!(storage_class, StorageClass::Input | StorageClass::Output) {
            return Err(ValidationError::InvalidLocationStorageClass {
                storage_class: *storage_class,
            });
        }
        if *decoration == Decoration::Component {
            let Some(rspirv::dr::Operand::LiteralBit32(component)) = inst.operands.get(2) else {
                continue;
            };
            if *component > 3 {
                return Err(ValidationError::ComponentOutOfRange {
                    component: *component,
                });
            }
            if !has_location.contains(&var_id) {
                return Err(ValidationError::ComponentMissingLocation);
            }
        }
    }

    Ok(())
}

fn enforce_builtin_location_exclusivity(module: &Module) -> Result<(), ValidationError> {
    use rspirv::spirv::Decoration;
    let mut built_ins: HashSet<ResultId> = HashSet::new();
    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        if let (
            Some(rspirv::dr::Operand::IdRef(target)),
            Some(rspirv::dr::Operand::Decoration(decoration)),
        ) = (inst.operands.first(), inst.operands.get(1))
        {
            if *decoration == Decoration::BuiltIn {
                if let Ok(id) = ResultId::try_from(*target) {
                    built_ins.insert(id);
                }
            }
        }
    }
    if built_ins.is_empty() {
        return Ok(());
    }

    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
            continue;
        };
        let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
            continue;
        };
        if *decoration != Decoration::Location && *decoration != Decoration::Component {
            continue;
        }
        let Ok(id) = ResultId::try_from(*target) else {
            continue;
        };
        if built_ins.contains(&id) {
            return Err(ValidationError::LocationConflictsWithBuiltIn);
        }
    }

    Ok(())
}

fn enforce_builtin_storage_classes(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    entry_models: &HashSet<rspirv::spirv::ExecutionModel>,
    capabilities: &HashSet<rspirv::spirv::Capability>,
    env: TargetEnv,
) -> Result<(), ValidationError> {
    use rspirv::spirv::{BuiltIn, Decoration, ExecutionModel, Op, StorageClass};

    // Map id -> (built-in, storage class)
    for inst in &module.annotations {
        if inst.class.opcode != Op::Decorate {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
            continue;
        };
        let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
            continue;
        };
        if *decoration != Decoration::BuiltIn {
            continue;
        }
        let builtin = inst
            .operands
            .get(2)
            .and_then(|op| match op {
                rspirv::dr::Operand::BuiltIn(b) => Some(*b),
                rspirv::dr::Operand::LiteralBit32(raw) => BuiltIn::from_u32(*raw),
                _ => None,
            })
            .unwrap_or(BuiltIn::Position);

        let Ok(id) = ResultId::try_from(*target) else {
            continue;
        };
        // Look up storage class of the variable.
        let storage_class = module.types_global_values.iter().find_map(|var| {
            if var.class.opcode != Op::Variable {
                return None;
            }
            if var.result_id != Some(u32::from(id)) {
                return None;
            }
            match var.operands.first() {
                Some(rspirv::dr::Operand::StorageClass(sc)) => Some(*sc),
                _ => None,
            }
        });
        let Some(storage_class) = storage_class else {
            continue;
        };

        if builtin == BuiltIn::WorkgroupSize {
            // Target-kind and type checks handle WorkgroupSize; storage class is validated elsewhere.
            continue;
        }

        if env.is_vulkan() && (builtin == BuiltIn::VertexId || builtin == BuiltIn::InstanceId) {
            return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env });
        }
        if !env.is_vulkan()
            && matches!(
                builtin,
                BuiltIn::ShadingRateKHR | BuiltIn::PrimitiveShadingRateKHR
            )
        {
            return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env });
        }
        if !env.is_vulkan()
            && matches!(
                builtin,
                BuiltIn::PrimitivePointIndicesEXT
                    | BuiltIn::PrimitiveLineIndicesEXT
                    | BuiltIn::PrimitiveTriangleIndicesEXT
                    | BuiltIn::CullPrimitiveEXT
            )
        {
            return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env });
        }

        let allowed = matches!(storage_class, StorageClass::Input | StorageClass::Output);
        if !allowed {
            return Err(ValidationError::InvalidBuiltInStorageClass {
                builtin,
                storage_class,
            });
        }

        if env.is_vulkan()
            && builtin == BuiltIn::ViewIndex
            && entry_models.contains(&ExecutionModel::GLCompute)
        {
            return Err(ValidationError::BuiltInRequiresExecutionModel {
                builtin,
                allowed: vec![
                    ExecutionModel::Vertex,
                    ExecutionModel::Geometry,
                    ExecutionModel::TessellationEvaluation,
                    ExecutionModel::MeshEXT,
                    ExecutionModel::MeshNV,
                ],
            });
        }

        let required_capability: Option<&[rspirv::spirv::Capability]> = match builtin {
            BuiltIn::ShadingRateKHR | BuiltIn::PrimitiveShadingRateKHR => {
                Some(&[rspirv::spirv::Capability::FragmentShadingRateKHR])
            }
            BuiltIn::ViewIndex => Some(&[rspirv::spirv::Capability::MultiView]),
            BuiltIn::DeviceIndex => Some(&[rspirv::spirv::Capability::DeviceGroup]),
            BuiltIn::WorkDim
            | BuiltIn::GlobalSize
            | BuiltIn::GlobalOffset
            | BuiltIn::EnqueuedWorkgroupSize
            | BuiltIn::GlobalLinearId
            | BuiltIn::SubgroupMaxSize => Some(&[rspirv::spirv::Capability::Kernel]),
            BuiltIn::NumEnqueuedSubgroups => Some(&[rspirv::spirv::Capability::DeviceEnqueue]),
            BuiltIn::WarpIDNV | BuiltIn::SMIDNV | BuiltIn::SMCountNV | BuiltIn::WarpsPerSMNV => {
                Some(&[rspirv::spirv::Capability::ShaderSMBuiltinsNV])
            }
            BuiltIn::CoreIDARM
            | BuiltIn::CoreCountARM
            | BuiltIn::CoreMaxIDARM
            | BuiltIn::WarpIDARM
            | BuiltIn::WarpMaxIDARM => Some(&[rspirv::spirv::Capability::CoreBuiltinsARM]),
            BuiltIn::SubgroupId
            | BuiltIn::NumSubgroups
            | BuiltIn::SubgroupLocalInvocationId
            | BuiltIn::SubgroupSize => Some(&[rspirv::spirv::Capability::GroupNonUniform]),
            BuiltIn::SubgroupEqMask
            | BuiltIn::SubgroupGeMask
            | BuiltIn::SubgroupGtMask
            | BuiltIn::SubgroupLeMask
            | BuiltIn::SubgroupLtMask => Some(&[
                rspirv::spirv::Capability::GroupNonUniformBallot,
                rspirv::spirv::Capability::SubgroupBallotKHR,
            ]),
            BuiltIn::BaryCoordKHR
            | BuiltIn::BaryCoordNoPerspKHR
            | BuiltIn::BaryCoordSmoothAMD
            | BuiltIn::BaryCoordSmoothCentroidAMD
            | BuiltIn::BaryCoordSmoothSampleAMD
            | BuiltIn::BaryCoordNoPerspAMD
            | BuiltIn::BaryCoordNoPerspCentroidAMD
            | BuiltIn::BaryCoordNoPerspSampleAMD
            | BuiltIn::BaryCoordPullModelAMD => {
                Some(&[rspirv::spirv::Capability::FragmentBarycentricKHR])
            }
            BuiltIn::CullPrimitiveEXT
            | BuiltIn::PrimitivePointIndicesEXT
            | BuiltIn::PrimitiveLineIndicesEXT
            | BuiltIn::PrimitiveTriangleIndicesEXT => Some(&[
                rspirv::spirv::Capability::MeshShadingEXT,
                rspirv::spirv::Capability::MeshShadingNV,
            ]),
            BuiltIn::LaunchIdKHR
            | BuiltIn::LaunchSizeKHR
            | BuiltIn::RayTminKHR
            | BuiltIn::RayTmaxKHR
            | BuiltIn::WorldRayOriginKHR
            | BuiltIn::WorldRayDirectionKHR
            | BuiltIn::ObjectRayOriginKHR
            | BuiltIn::ObjectRayDirectionKHR
            | BuiltIn::ObjectToWorldKHR
            | BuiltIn::WorldToObjectKHR
            | BuiltIn::InstanceCustomIndexKHR
            | BuiltIn::InstanceId
            | BuiltIn::RayGeometryIndexKHR
            | BuiltIn::IncomingRayFlagsKHR
            | BuiltIn::CullMaskKHR
            | BuiltIn::HitKindKHR
            | BuiltIn::HitTNV => Some(&[
                rspirv::spirv::Capability::RayTracingKHR,
                rspirv::spirv::Capability::RayTracingNV,
            ]),
            _ => None,
        };
        if let Some(required) = required_capability {
            if !required.iter().any(|cap| capabilities.contains(cap)) {
                let capability = required[0];
                return Err(ValidationError::BuiltInRequiresCapability {
                    builtin,
                    capability,
                });
            }
        }

        let fragment_only = matches!(
            builtin,
            BuiltIn::FragCoord
                | BuiltIn::PointCoord
                | BuiltIn::FrontFacing
                | BuiltIn::SampleId
                | BuiltIn::SamplePosition
                | BuiltIn::SampleMask
                | BuiltIn::FragDepth
                | BuiltIn::HelperInvocation
                | BuiltIn::FragInvocationCountEXT
                | BuiltIn::FragSizeEXT
                | BuiltIn::FragStencilRefEXT
                | BuiltIn::FullyCoveredEXT
                | BuiltIn::BaryCoordKHR
                | BuiltIn::BaryCoordNoPerspKHR
                | BuiltIn::BaryCoordSmoothAMD
                | BuiltIn::BaryCoordSmoothCentroidAMD
                | BuiltIn::BaryCoordSmoothSampleAMD
                | BuiltIn::BaryCoordNoPerspAMD
                | BuiltIn::BaryCoordNoPerspCentroidAMD
                | BuiltIn::BaryCoordNoPerspSampleAMD
                | BuiltIn::BaryCoordPullModelAMD
        );
        if fragment_only && !entry_models.contains(&ExecutionModel::Fragment) {
            return Err(ValidationError::BuiltInRequiresFragment { builtin });
        }

        let barycentric_only_input = matches!(
            builtin,
            BuiltIn::BaryCoordKHR
                | BuiltIn::BaryCoordNoPerspKHR
                | BuiltIn::BaryCoordSmoothAMD
                | BuiltIn::BaryCoordSmoothCentroidAMD
                | BuiltIn::BaryCoordSmoothSampleAMD
                | BuiltIn::BaryCoordNoPerspAMD
                | BuiltIn::BaryCoordNoPerspCentroidAMD
                | BuiltIn::BaryCoordNoPerspSampleAMD
                | BuiltIn::BaryCoordPullModelAMD
        );
        if barycentric_only_input && storage_class != StorageClass::Input {
            return Err(ValidationError::InvalidBuiltInStorageClass {
                builtin,
                storage_class,
            });
        }

        if builtin == BuiltIn::ShadingRateKHR && storage_class != StorageClass::Input {
            return Err(ValidationError::InvalidBuiltInStorageClass {
                builtin,
                storage_class,
            });
        }

        let mesh_output_only = matches!(
            builtin,
            BuiltIn::PrimitivePointIndicesEXT
                | BuiltIn::PrimitiveLineIndicesEXT
                | BuiltIn::PrimitiveTriangleIndicesEXT
                | BuiltIn::CullPrimitiveEXT
        );
        if mesh_output_only && storage_class != StorageClass::Output {
            return Err(ValidationError::InvalidBuiltInStorageClass {
                builtin,
                storage_class,
            });
        }

        if builtin == BuiltIn::PrimitiveShadingRateKHR && storage_class != StorageClass::Output {
            return Err(ValidationError::InvalidBuiltInStorageClass {
                builtin,
                storage_class,
            });
        }

        let compute_only = matches!(
            builtin,
            BuiltIn::GlobalInvocationId
                | BuiltIn::LocalInvocationId
                | BuiltIn::LocalInvocationIndex
                | BuiltIn::NumWorkgroups
                | BuiltIn::WorkgroupId
                | BuiltIn::NumSubgroups
                | BuiltIn::SubgroupId
                | BuiltIn::SubgroupLocalInvocationId
        );
        if compute_only
            && !entry_models.contains(&ExecutionModel::GLCompute)
            && !entry_models.contains(&ExecutionModel::Kernel)
        {
            return Err(ValidationError::BuiltInRequiresExecutionModel {
                builtin,
                allowed: vec![ExecutionModel::GLCompute, ExecutionModel::Kernel],
            });
        }

        let kernel_only = matches!(
            builtin,
            BuiltIn::WorkDim
                | BuiltIn::GlobalSize
                | BuiltIn::GlobalOffset
                | BuiltIn::EnqueuedWorkgroupSize
                | BuiltIn::GlobalLinearId
                | BuiltIn::SubgroupMaxSize
                | BuiltIn::NumEnqueuedSubgroups
        );
        if kernel_only && !entry_models.contains(&ExecutionModel::Kernel) {
            return Err(ValidationError::BuiltInRequiresExecutionModel {
                builtin,
                allowed: vec![ExecutionModel::Kernel],
            });
        }

        // Type checks for selected built-ins.
        if let Ok(var_id) = ResultId::try_from(*target) {
            if let Some(pointee) = resolve_builtin_pointee_type(definitions, var_id) {
                if let Some(error) = validate_builtin_type(builtin, pointee, definitions) {
                    return Err(error);
                }
            }

            if matches!(builtin, BuiltIn::TessLevelOuter | BuiltIn::TessLevelInner)
                && !has_patch_decoration(module, var_id)
            {
                return Err(ValidationError::BuiltInRequiresPatchDecoration { builtin });
            }
        }

        // Execution model allowlists for built-ins that are limited to specific pipeline stages.
        let required_models: Option<&[ExecutionModel]> = match builtin {
            BuiltIn::TessCoord | BuiltIn::TessLevelInner | BuiltIn::TessLevelOuter => {
                Some(&[ExecutionModel::TessellationEvaluation])
            }
            BuiltIn::PatchVertices => Some(&[ExecutionModel::TessellationControl]),
            BuiltIn::PrimitiveId => Some(&[
                ExecutionModel::Geometry,
                ExecutionModel::TessellationControl,
                ExecutionModel::TessellationEvaluation,
                ExecutionModel::MeshNV,
                ExecutionModel::MeshEXT,
                ExecutionModel::RayGenerationKHR,
                ExecutionModel::ClosestHitKHR,
                ExecutionModel::AnyHitKHR,
                ExecutionModel::MissKHR,
                ExecutionModel::IntersectionKHR,
                ExecutionModel::CallableKHR,
            ]),
            BuiltIn::LaunchIdKHR
            | BuiltIn::LaunchSizeKHR
            | BuiltIn::RayTminKHR
            | BuiltIn::RayTmaxKHR
            | BuiltIn::WorldRayOriginKHR
            | BuiltIn::WorldRayDirectionKHR
            | BuiltIn::ObjectRayOriginKHR
            | BuiltIn::ObjectRayDirectionKHR
            | BuiltIn::ObjectToWorldKHR
            | BuiltIn::WorldToObjectKHR
            | BuiltIn::InstanceCustomIndexKHR
            | BuiltIn::InstanceId
            | BuiltIn::RayGeometryIndexKHR
            | BuiltIn::IncomingRayFlagsKHR
            | BuiltIn::CullMaskKHR
            | BuiltIn::HitKindKHR
            | BuiltIn::HitTNV => Some(&[
                ExecutionModel::RayGenerationKHR,
                ExecutionModel::IntersectionKHR,
                ExecutionModel::AnyHitKHR,
                ExecutionModel::ClosestHitKHR,
                ExecutionModel::MissKHR,
                ExecutionModel::CallableKHR,
            ]),
            BuiltIn::ShadingRateKHR => Some(&[ExecutionModel::Fragment]),
            BuiltIn::PrimitivePointIndicesEXT
            | BuiltIn::PrimitiveLineIndicesEXT
            | BuiltIn::PrimitiveTriangleIndicesEXT
            | BuiltIn::CullPrimitiveEXT => Some(&[ExecutionModel::MeshEXT, ExecutionModel::MeshNV]),
            BuiltIn::PrimitiveShadingRateKHR => Some(&[
                ExecutionModel::Vertex,
                ExecutionModel::Geometry,
                ExecutionModel::MeshEXT,
                ExecutionModel::MeshNV,
            ]),
            BuiltIn::VertexIndex | BuiltIn::InstanceIndex => Some(&[ExecutionModel::Vertex]),
            _ => None,
        };
        if let Some(models) = required_models {
            if !entry_models.iter().any(|m| models.contains(m)) {
                return Err(ValidationError::BuiltInRequiresExecutionModel {
                    builtin,
                    allowed: models.to_vec(),
                });
            }
        }
    }
    Ok(())
}

fn resolve_builtin_pointee_type(
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    var_id: ResultId,
) -> Option<&rspirv::dr::Instruction> {
    let var_inst = definitions.get(&var_id)?;
    let ptr_type_id = var_inst.result_type?;
    let ptr_type = ResultId::try_from(ptr_type_id)
        .ok()
        .and_then(|id| definitions.get(&id))?;
    if ptr_type.class.opcode != rspirv::spirv::Op::TypePointer {
        return None;
    }
    let pointee_id = match ptr_type.operands.get(1) {
        Some(rspirv::dr::Operand::IdRef(id)) => *id,
        _ => return None,
    };
    ResultId::try_from(pointee_id)
        .ok()
        .and_then(|id| definitions.get(&id))
}

fn fragment_requires_flat(
    var_inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    let Some(var_id) = var_inst
        .result_id
        .and_then(|id| ResultId::try_from(id).ok())
    else {
        return false;
    };
    let Some(pointee) = resolve_builtin_pointee_type(definitions, var_id) else {
        return false;
    };
    is_int_scalar_or_vector(pointee, definitions)
        || is_float_scalar_of_width(pointee, definitions, 64)
}

fn is_int_scalar_or_vector(
    ty: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    match ty.class.opcode {
        rspirv::spirv::Op::TypeInt => true,
        rspirv::spirv::Op::TypeVector => {
            let Some(rspirv::dr::Operand::IdRef(elem)) = ty.operands.first() else {
                return false;
            };
            ResultId::try_from(*elem)
                .ok()
                .and_then(|id| definitions.get(&id))
                .is_some_and(|inst| inst.class.opcode == rspirv::spirv::Op::TypeInt)
        }
        _ => false,
    }
}

fn type_bit_width(ty: &rspirv::dr::Instruction) -> Option<u32> {
    ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(w) => Some(*w),
        _ => None,
    })
}

fn is_float_scalar_of_width(
    ty: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    width: u32,
) -> bool {
    match ty.class.opcode {
        rspirv::spirv::Op::TypeFloat => type_bit_width(ty) == Some(width),
        rspirv::spirv::Op::TypeVector => {
            let Some(rspirv::dr::Operand::IdRef(elem)) = ty.operands.first() else {
                return false;
            };
            ResultId::try_from(*elem)
                .ok()
                .and_then(|id| definitions.get(&id))
                .is_some_and(|inst| {
                    inst.class.opcode == rspirv::spirv::Op::TypeFloat
                        && type_bit_width(inst) == Some(width)
                })
        }
        _ => false,
    }
}

fn literal_u32(op: &rspirv::dr::Operand) -> Option<u32> {
    match op {
        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
        _ => None,
    }
}

fn is_float32(inst: &rspirv::dr::Instruction) -> bool {
    inst.class.opcode == rspirv::spirv::Op::TypeFloat
        && inst.operands.first().and_then(literal_u32) == Some(32)
}

fn is_int32(inst: &rspirv::dr::Instruction) -> bool {
    inst.class.opcode == rspirv::spirv::Op::TypeInt
        && inst.operands.first().and_then(literal_u32) == Some(32)
}

fn is_bool(inst: &rspirv::dr::Instruction) -> bool {
    inst.class.opcode == rspirv::spirv::Op::TypeBool
}

fn is_vector_of(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    len: u32,
    element_predicate: fn(&rspirv::dr::Instruction) -> bool,
) -> bool {
    if inst.class.opcode != rspirv::spirv::Op::TypeVector {
        return false;
    }
    let elem_id = match inst.operands.first() {
        Some(rspirv::dr::Operand::IdRef(id)) => *id,
        _ => return false,
    };
    let count = inst.operands.get(1).and_then(literal_u32);
    if count != Some(len) {
        return false;
    }
    ResultId::try_from(elem_id)
        .ok()
        .and_then(|id| definitions.get(&id))
        .is_some_and(element_predicate)
}

fn is_array_of(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    element_predicate: fn(&rspirv::dr::Instruction) -> bool,
) -> bool {
    if inst.class.opcode != rspirv::spirv::Op::TypeArray
        && inst.class.opcode != rspirv::spirv::Op::TypeRuntimeArray
    {
        return false;
    }
    let elem_id = match inst.operands.first() {
        Some(rspirv::dr::Operand::IdRef(id)) => *id,
        _ => return false,
    };
    ResultId::try_from(elem_id)
        .ok()
        .and_then(|id| definitions.get(&id))
        .is_some_and(element_predicate)
}

fn is_array_of_len(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    len: u32,
    element_predicate: fn(&rspirv::dr::Instruction) -> bool,
) -> bool {
    if inst.class.opcode != rspirv::spirv::Op::TypeArray {
        return false;
    }
    if array_length(inst, definitions) != Some(len) {
        return false;
    }
    is_array_of(inst, definitions, element_predicate)
}

/// Checks if `inst` is either a vector directly or an array of vectors.
/// This matches the C++ validator's `ValidateOptionalArrayedF32Vec` logic.
fn is_optional_array_of_vector(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    len: u32,
    element_predicate: fn(&rspirv::dr::Instruction) -> bool,
) -> bool {
    // First try: direct vector
    if is_vector_of(inst, definitions, len, element_predicate) {
        return true;
    }
    // Second try: array of vectors
    if inst.class.opcode == rspirv::spirv::Op::TypeArray
        || inst.class.opcode == rspirv::spirv::Op::TypeRuntimeArray
    {
        let elem_id = match inst.operands.first() {
            Some(rspirv::dr::Operand::IdRef(id)) => *id,
            _ => return false,
        };
        if let Some(elem_inst) = ResultId::try_from(elem_id)
            .ok()
            .and_then(|id| definitions.get(&id))
        {
            return is_vector_of(elem_inst, definitions, len, element_predicate);
        }
    }
    false
}

fn validate_builtin_type(
    builtin: rspirv::spirv::BuiltIn,
    pointee: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<ValidationError> {
    use rspirv::spirv::BuiltIn::*;
    let ok = match builtin {
        FragCoord => is_vector_of(pointee, definitions, 4, is_float32),
        // Position can be vec4<f32> or array<vec4<f32>> for mesh/geometry/tessellation shaders.
        // See C++ spirv-val's ValidateOptionalArrayedF32Vec.
        Position => is_optional_array_of_vector(pointee, definitions, 4, is_float32),
        FragDepth | PointSize => is_float32(pointee),
        SampleMask => is_array_of(pointee, definitions, is_int32),
        ClipDistance | CullDistance => is_array_of(pointee, definitions, is_float32),
        TessCoord => is_vector_of(pointee, definitions, 3, is_float32),
        TessLevelOuter => is_array_of_len(pointee, definitions, 4, is_float32),
        TessLevelInner => is_array_of_len(pointee, definitions, 2, is_float32),
        PrimitiveId
        | Layer
        | ViewIndex
        | SampleId
        | VertexId
        | InstanceId
        | VertexIndex
        | InstanceIndex
        | BaseVertex
        | BaseInstance
        | DrawIndex
        | DeviceIndex
        | InvocationId
        | LocalInvocationIndex
        | PatchVertices
        | SubgroupId
        | NumSubgroups
        | SubgroupLocalInvocationId
        | SubgroupSize => is_int32(pointee),
        GlobalInvocationId | LocalInvocationId | WorkgroupId | NumWorkgroups => {
            is_vector_of(pointee, definitions, 3, is_int32)
        }
        BaryCoordKHR | BaryCoordNoPerspKHR => is_vector_of(pointee, definitions, 3, is_float32),
        SubgroupEqMask | SubgroupGeMask | SubgroupGtMask | SubgroupLeMask | SubgroupLtMask => {
            is_vector_of(pointee, definitions, 4, is_int32)
        }
        PointCoord | SamplePosition => is_vector_of(pointee, definitions, 2, is_float32),
        FrontFacing | HelperInvocation => is_bool(pointee),
        ShadingRateKHR | PrimitiveShadingRateKHR => is_int32(pointee),
        _ => true,
    };
    if ok {
        return None;
    }
    let expected = match builtin {
        FragCoord => "vec4<f32>",
        Position => "vec4<f32> or array of vec4<f32>",
        FragDepth | PointSize => "f32",
        SampleMask => "array/runtime array of 32-bit ints",
        ClipDistance | CullDistance => "array/runtime array of 32-bit floats",
        ShadingRateKHR | PrimitiveShadingRateKHR => "i32",
        TessCoord => "vec3<f32>",
        TessLevelOuter => "array[4] of f32",
        TessLevelInner => "array[2] of f32",
        PrimitiveId
        | Layer
        | ViewIndex
        | SampleId
        | VertexId
        | InstanceId
        | VertexIndex
        | InstanceIndex
        | BaseVertex
        | BaseInstance
        | DrawIndex
        | DeviceIndex
        | InvocationId
        | LocalInvocationIndex
        | PatchVertices
        | SubgroupId
        | NumSubgroups
        | SubgroupLocalInvocationId
        | SubgroupSize => "i32",
        GlobalInvocationId | LocalInvocationId | WorkgroupId | NumWorkgroups => "vec3<i32>",
        BaryCoordKHR | BaryCoordNoPerspKHR => "vec3<f32>",
        SubgroupEqMask | SubgroupGeMask | SubgroupGtMask | SubgroupLeMask | SubgroupLtMask => {
            "vec4<i32>"
        }
        PointCoord | SamplePosition => "vec2<f32>",
        FrontFacing | HelperInvocation => "bool",
        _ => "",
    };
    Some(ValidationError::InvalidBuiltInType { builtin, expected })
}

fn enforce_interpolation_storage_classes(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    entry_models: &HashSet<rspirv::spirv::ExecutionModel>,
    capabilities: &HashSet<rspirv::spirv::Capability>,
) -> Result<(), ValidationError> {
    use rspirv::spirv::Decoration::{Centroid, Flat, NoPerspective, Patch, Sample};

    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        let mut operands = inst.operands.iter();
        let Some(rspirv::dr::Operand::IdRef(target)) = operands.next() else {
            continue;
        };
        let Some(rspirv::dr::Operand::Decoration(decoration)) = operands.next() else {
            continue;
        };
        let decoration = *decoration;
        let is_interp_base = matches!(decoration, NoPerspective | Flat | Patch | Centroid | Sample);
        if !is_interp_base {
            continue;
        }
        let Ok(id) = ResultId::try_from(*target) else {
            continue;
        };
        let Some(def_inst) = definitions.get(&id) else {
            continue;
        };
        if def_inst.class.opcode != rspirv::spirv::Op::Variable
            && def_inst.class.opcode != rspirv::spirv::Op::UntypedVariableKHR
        {
            continue;
        }
        let storage_class = def_inst
            .operands
            .first()
            .and_then(|op| match op {
                rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                _ => None,
            })
            .unwrap_or(rspirv::spirv::StorageClass::Function);
        if storage_class != rspirv::spirv::StorageClass::Input
            && storage_class != rspirv::spirv::StorageClass::Output
        {
            return Err(
                ValidationError::InterpolationDecorationInvalidStorageClass {
                    decoration,
                    storage_class,
                },
            );
        }

        if decoration == Sample
            && !capabilities.contains(&rspirv::spirv::Capability::SampleRateShading)
        {
            return Err(ValidationError::DecorationRequiresCapability {
                decoration,
                capability: rspirv::spirv::Capability::SampleRateShading,
            });
        }

        if decoration != Patch && !entry_models.contains(&rspirv::spirv::ExecutionModel::Fragment) {
            return Err(ValidationError::InterpolationDecorationRequiresFragment { decoration });
        }
    }

    Ok(())
}

fn enforce_interpolation_exclusivity(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Result<(), ValidationError> {
    use rspirv::spirv::Decoration::{Centroid, Flat, NoPerspective, Patch, Sample};

    #[derive(Default)]
    struct InterpDecorations {
        base: Option<rspirv::spirv::Decoration>,
        centroid_sample_patch: Option<rspirv::spirv::Decoration>,
    }

    let mut seen: HashMap<ResultId, InterpDecorations> = HashMap::new();

    for inst in &module.annotations {
        if inst.class.opcode != rspirv::spirv::Op::Decorate {
            continue;
        }
        let mut operands = inst.operands.iter();
        let Some(rspirv::dr::Operand::IdRef(target)) = operands.next() else {
            continue;
        };
        let Some(rspirv::dr::Operand::Decoration(decoration)) = operands.next() else {
            continue;
        };
        let decoration = *decoration;
        if !matches!(decoration, Flat | NoPerspective | Centroid | Sample | Patch) {
            continue;
        }
        let Ok(id) = ResultId::try_from(*target) else {
            continue;
        };
        let Some(def_inst) = definitions.get(&id) else {
            continue;
        };
        if def_inst.class.opcode != rspirv::spirv::Op::Variable
            && def_inst.class.opcode != rspirv::spirv::Op::UntypedVariableKHR
        {
            continue;
        }

        let entry = seen.entry(id).or_default();
        if matches!(decoration, Flat | NoPerspective) {
            if let Some(existing) = entry.base {
                if existing != decoration {
                    return Err(ValidationError::InterpolationDecorationConflict {
                        decoration,
                        existing,
                    });
                }
            } else {
                entry.base = Some(decoration);
            }
        }

        if matches!(decoration, Centroid | Sample | Patch) {
            if let Some(existing) = entry.base {
                if existing == Flat {
                    return Err(ValidationError::InterpolationDecorationConflict {
                        decoration,
                        existing,
                    });
                }
            }
            if let Some(existing) = entry.centroid_sample_patch {
                if existing != decoration {
                    return Err(ValidationError::InterpolationDecorationConflict {
                        decoration,
                        existing,
                    });
                }
            } else {
                entry.centroid_sample_patch = Some(decoration);
            }
        }

        if matches!(decoration, Flat) {
            if let Some(existing) = entry.centroid_sample_patch {
                return Err(ValidationError::InterpolationDecorationConflict {
                    decoration,
                    existing,
                });
            }
        }
    }

    Ok(())
}

fn enforce_interpolation_entry_point_compatibility(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    env: TargetEnv,
) -> Result<(), ValidationError> {
    use rspirv::spirv::Decoration::{Centroid, Flat, NoPerspective, Sample};
    use rspirv::spirv::{ExecutionModel, StorageClass};

    if !is_vulkan_env(env) {
        return Ok(());
    }

    let decoration_lookup = build_decoration_lookup(module);

    for entry in &module.entry_points {
        let Some(rspirv::dr::Operand::ExecutionModel(model)) = entry.operands.first() else {
            continue;
        };
        let model = *model;
        let interfaces = entry.operands.iter().skip(2).filter_map(|op| match op {
            rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
            _ => None,
        });
        for var_id in interfaces {
            let Some(var_inst) = definitions.get(&var_id) else {
                continue;
            };
            if var_inst.class.opcode != rspirv::spirv::Op::Variable
                && var_inst.class.opcode != rspirv::spirv::Op::UntypedVariableKHR
            {
                continue;
            }
            let storage_class = match var_inst.operands.first() {
                Some(rspirv::dr::Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };
            let decos = decoration_lookup.get(&var_id).cloned().unwrap_or_default();
            let has_interp = decos.contains(&NoPerspective)
                || decos.contains(&Flat)
                || decos.contains(&Sample)
                || decos.contains(&Centroid);
            if has_interp {
                match storage_class {
                    StorageClass::Input if model == ExecutionModel::Vertex => {
                        return Err(
                            ValidationError::InterpolationDecorationInvalidForEntryPoint {
                                decoration: *decos
                                    .iter()
                                    .find(|d| matches!(d, NoPerspective | Flat | Sample | Centroid))
                                    .unwrap(),
                                storage_class,
                                execution_model: model,
                            },
                        );
                    }
                    StorageClass::Output if model == ExecutionModel::Fragment => {
                        return Err(
                            ValidationError::InterpolationDecorationInvalidForEntryPoint {
                                decoration: *decos
                                    .iter()
                                    .find(|d| matches!(d, NoPerspective | Flat | Sample | Centroid))
                                    .unwrap(),
                                storage_class,
                                execution_model: model,
                            },
                        );
                    }
                    _ => {}
                }
            }

            if model == ExecutionModel::Fragment && storage_class == StorageClass::Input {
                let has_flat = decos.contains(&Flat);
                if !has_flat && fragment_requires_flat(var_inst, definitions) {
                    return Err(ValidationError::FragmentInputRequiresFlat);
                }
            }
        }
    }

    Ok(())
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

fn pointer_info(
    pointer_type: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<(rspirv::spirv::Op, rspirv::spirv::StorageClass)> {
    let rid = ResultId::try_from(u32::from(Id::from(pointer_type))).ok()?;
    let inst = definitions.get(&rid)?;
    match inst.class.opcode {
        rspirv::spirv::Op::TypePointer | rspirv::spirv::Op::TypeUntypedPointerKHR => {
            inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::StorageClass(sc) => Some((inst.class.opcode, *sc)),
                _ => None,
            })
        }
        _ => None,
    }
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
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Result<HashSet<ResultId>, ValidationError> {
    let mut entry_points = HashSet::new();
    let mut seen: HashSet<(ResultId, rspirv::spirv::ExecutionModel)> = HashSet::new();
    for ep in &module.entry_points {
        let entry_opcode = ep.class.opcode;
        let mut operands = ep.operands.iter();
        if entry_opcode == rspirv::spirv::Op::ConditionalEntryPointINTEL {
            // Skip the condition operand.
            let _ = operands.next();
        }
        // Next operand is ExecutionModel.
        let execution_model = operands.next().and_then(|op| match op {
            rspirv::dr::Operand::ExecutionModel(model) => Some(*model),
            _ => None,
        });
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
        if let Some(model) = execution_model {
            if !seen.insert((function_id, model)) {
                return Err(ValidationError::DuplicateEntryPoint {
                    function: function_id.into_inner(),
                    execution_model: model,
                });
            }
        }
        entry_points.insert(function_id);
        // Skip the name operand.
        let _ = operands.next();
        let mut interfaces = HashSet::new();
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
                if !interfaces.insert(interface_id) {
                    return Err(ValidationError::DuplicateEntryPointInterface {
                        entry_point: function_id.into_inner(),
                        interface: interface_id.into_inner(),
                    });
                }
                if let Some(opcode) = opcodes.get(&interface_id) {
                    if *opcode != rspirv::spirv::Op::Variable {
                        return Err(ValidationError::InvalidEntryPointTarget {
                            target: interface_id.into_inner(),
                            opcode: *opcode,
                        });
                    }
                    if let Some(var_inst) = definitions.get(&interface_id) {
                        if let Some(rspirv::dr::Operand::StorageClass(storage)) =
                            var_inst.operands.first()
                        {
                            if *storage == rspirv::spirv::StorageClass::Function {
                                return Err(
                                    ValidationError::EntryPointInterfaceStorageClassInvalid {
                                        entry_point: function_id.into_inner(),
                                        interface: interface_id.into_inner(),
                                        storage_class: *storage,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(entry_points)
}

fn validate_entry_point_interface_storage_classes(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    capabilities: &HashSet<rspirv::spirv::Capability>,
    env: TargetEnv,
) -> Result<(), ValidationError> {
    if !is_vulkan_env(env) {
        return Ok(());
    }

    let decoration_lookup = build_decoration_lookup(module);

    fn contains_disallowed_fp_encoding(
        definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
        ty: ResultId,
        seen: &mut HashSet<ResultId>,
    ) -> Option<rspirv::spirv::FPEncoding> {
        if !seen.insert(ty) {
            return None;
        }
        let inst = definitions.get(&ty)?;
        match inst.class.opcode {
            rspirv::spirv::Op::TypeFloat => inst.operands.iter().find_map(|op| match op {
                rspirv::dr::Operand::FPEncoding(encoding)
                    if matches!(
                        encoding,
                        rspirv::spirv::FPEncoding::Float8E4M3EXT
                            | rspirv::spirv::FPEncoding::Float8E5M2EXT
                            | rspirv::spirv::FPEncoding::BFloat16KHR
                    ) =>
                {
                    Some(*encoding)
                }
                _ => None,
            }),
            rspirv::spirv::Op::TypePointer => inst.operands.get(1).and_then(|op| {
                if let rspirv::dr::Operand::IdRef(pointee) = op {
                    ResultId::try_from(*pointee)
                        .ok()
                        .and_then(|id| contains_disallowed_fp_encoding(definitions, id, seen))
                } else {
                    None
                }
            }),
            rspirv::spirv::Op::TypeVector
            | rspirv::spirv::Op::TypeMatrix
            | rspirv::spirv::Op::TypeArray
            | rspirv::spirv::Op::TypeRuntimeArray => inst.operands.first().and_then(|op| {
                if let rspirv::dr::Operand::IdRef(element) = op {
                    ResultId::try_from(*element)
                        .ok()
                        .and_then(|id| contains_disallowed_fp_encoding(definitions, id, seen))
                } else {
                    None
                }
            }),
            rspirv::spirv::Op::TypeStruct => inst.operands.iter().find_map(|op| {
                if let rspirv::dr::Operand::IdRef(member) = op {
                    ResultId::try_from(*member)
                        .ok()
                        .and_then(|id| contains_disallowed_fp_encoding(definitions, id, seen))
                } else {
                    None
                }
            }),
            _ => None,
        }
    }

    for ep in &module.entry_points {
        let mut operands = ep.operands.iter();
        if ep.class.opcode == rspirv::spirv::Op::ConditionalEntryPointINTEL {
            let _ = operands.next();
        }
        // ExecutionModel
        let exec_model = operands
            .next()
            .and_then(|op| match op {
                rspirv::dr::Operand::ExecutionModel(model) => Some(*model),
                _ => None,
            })
            .ok_or(ValidationError::InvalidEntryPointOperand)?;
        let entry_point_id = operands
            .next()
            .and_then(|op| match op {
                rspirv::dr::Operand::IdRef(ep_id) => Some(*ep_id),
                _ => None,
            })
            .and_then(|raw| Id::try_from(raw).ok())
            .ok_or(ValidationError::InvalidEntryPointOperand)?;
        // Skip the name operand.
        let operands = operands.skip(1);
        let mut seen_push_constant = false;
        let mut seen_ray_payload = false;
        let mut seen_hit_attribute = false;
        let mut seen_callable_data = false;
        let mut seen_interface_ids: HashSet<Id> = HashSet::new();
        for operand in operands {
            let interface_id = match operand {
                rspirv::dr::Operand::IdRef(id) => *id,
                _ => continue,
            };
            if let Ok(id) = ResultId::try_from(interface_id) {
                if let Some(inst) = definitions.get(&id) {
                    if let Some(rspirv::dr::Operand::StorageClass(storage)) = inst.operands.first()
                    {
                        if !seen_interface_ids.insert(id.into()) {
                            return Err(ValidationError::DuplicateEntryPointInterface {
                                entry_point: entry_point_id,
                                interface: id.into(),
                            });
                        }
                        let has_patch = decoration_lookup
                            .get(&id)
                            .is_some_and(|decs| decs.contains(&rspirv::spirv::Decoration::Patch));
                        let storage_allowed = matches!(
                            *storage,
                            rspirv::spirv::StorageClass::Input
                                | rspirv::spirv::StorageClass::Output
                                | rspirv::spirv::StorageClass::Uniform
                                | rspirv::spirv::StorageClass::UniformConstant
                                | rspirv::spirv::StorageClass::PushConstant
                                | rspirv::spirv::StorageClass::StorageBuffer
                                | rspirv::spirv::StorageClass::PhysicalStorageBuffer
                                | rspirv::spirv::StorageClass::Workgroup
                                | rspirv::spirv::StorageClass::Private
                                | rspirv::spirv::StorageClass::IncomingRayPayloadKHR
                                | rspirv::spirv::StorageClass::RayPayloadKHR
                                | rspirv::spirv::StorageClass::HitAttributeKHR
                                | rspirv::spirv::StorageClass::IncomingCallableDataKHR
                                | rspirv::spirv::StorageClass::CallableDataKHR
                                | rspirv::spirv::StorageClass::ShaderRecordBufferKHR
                                | rspirv::spirv::StorageClass::TaskPayloadWorkgroupEXT
                        );
                        if !storage_allowed {
                            return Err(ValidationError::EntryPointInterfaceStorageClassInvalid {
                                entry_point: entry_point_id,
                                interface: id.into_inner(),
                                storage_class: *storage,
                            });
                        }
                        if has_patch
                            && !capabilities.contains(&rspirv::spirv::Capability::Tessellation)
                        {
                            return Err(ValidationError::DecorationRequiresCapability {
                                decoration: rspirv::spirv::Decoration::Patch,
                                capability: rspirv::spirv::Capability::Tessellation,
                            });
                        }
                        if has_patch
                            && !matches!(
                                exec_model,
                                rspirv::spirv::ExecutionModel::TessellationControl
                                    | rspirv::spirv::ExecutionModel::TessellationEvaluation
                            )
                        {
                            return Err(ValidationError::PatchDecorationRequiresTessellation {
                                execution_model: exec_model,
                            });
                        }
                        if matches!(
                            exec_model,
                            rspirv::spirv::ExecutionModel::RayGenerationKHR
                                | rspirv::spirv::ExecutionModel::IntersectionKHR
                                | rspirv::spirv::ExecutionModel::AnyHitKHR
                                | rspirv::spirv::ExecutionModel::ClosestHitKHR
                                | rspirv::spirv::ExecutionModel::MissKHR
                                | rspirv::spirv::ExecutionModel::CallableKHR
                        ) {
                            let allowed = matches!(
                                *storage,
                                rspirv::spirv::StorageClass::IncomingRayPayloadKHR
                                    | rspirv::spirv::StorageClass::RayPayloadKHR
                                    | rspirv::spirv::StorageClass::HitAttributeKHR
                                    | rspirv::spirv::StorageClass::IncomingCallableDataKHR
                                    | rspirv::spirv::StorageClass::CallableDataKHR
                                    | rspirv::spirv::StorageClass::PushConstant
                                    | rspirv::spirv::StorageClass::ShaderRecordBufferKHR
                                    | rspirv::spirv::StorageClass::UniformConstant
                                    | rspirv::spirv::StorageClass::Input
                                    | rspirv::spirv::StorageClass::Output
                            );
                            if !allowed {
                                return Err(
                                    ValidationError::EntryPointInterfaceStorageClassInvalid {
                                        entry_point: entry_point_id,
                                        interface: id.into_inner(),
                                        storage_class: *storage,
                                    },
                                );
                            }
                        } else {
                            // Non-ray entry points cannot list ray-specific storage classes.
                            if matches!(
                                *storage,
                                rspirv::spirv::StorageClass::IncomingRayPayloadKHR
                                    | rspirv::spirv::StorageClass::RayPayloadKHR
                                    | rspirv::spirv::StorageClass::HitAttributeKHR
                                    | rspirv::spirv::StorageClass::IncomingCallableDataKHR
                                    | rspirv::spirv::StorageClass::CallableDataKHR
                                    | rspirv::spirv::StorageClass::ShaderRecordBufferKHR
                            ) {
                                return Err(
                                    ValidationError::EntryPointInterfaceStorageClassInvalid {
                                        entry_point: entry_point_id,
                                        interface: id.into_inner(),
                                        storage_class: *storage,
                                    },
                                );
                            }
                        }
                        match storage {
                            rspirv::spirv::StorageClass::PushConstant => {
                                if seen_push_constant {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassDuplicate {
                                            entry_point: entry_point_id,
                                            storage_class: *storage,
                                        },
                                    );
                                }
                                seen_push_constant = true;
                            }
                            rspirv::spirv::StorageClass::IncomingRayPayloadKHR => {
                                if seen_ray_payload {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassDuplicate {
                                            entry_point: entry_point_id,
                                            storage_class: *storage,
                                        },
                                    );
                                }
                                seen_ray_payload = true;
                            }
                            rspirv::spirv::StorageClass::HitAttributeKHR => {
                                if seen_hit_attribute {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassDuplicate {
                                            entry_point: entry_point_id,
                                            storage_class: *storage,
                                        },
                                    );
                                }
                                seen_hit_attribute = true;
                            }
                            rspirv::spirv::StorageClass::IncomingCallableDataKHR => {
                                if seen_callable_data {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassDuplicate {
                                            entry_point: entry_point_id,
                                            storage_class: *storage,
                                        },
                                    );
                                }
                                seen_callable_data = true;
                            }
                            rspirv::spirv::StorageClass::Input => {
                                let allow_input = matches!(
                                    exec_model,
                                    rspirv::spirv::ExecutionModel::Vertex
                                        | rspirv::spirv::ExecutionModel::TessellationControl
                                        | rspirv::spirv::ExecutionModel::TessellationEvaluation
                                        | rspirv::spirv::ExecutionModel::Geometry
                                        | rspirv::spirv::ExecutionModel::Fragment
                                        | rspirv::spirv::ExecutionModel::MeshEXT
                                        | rspirv::spirv::ExecutionModel::TaskEXT
                                        | rspirv::spirv::ExecutionModel::GLCompute
                                        | rspirv::spirv::ExecutionModel::RayGenerationKHR
                                        | rspirv::spirv::ExecutionModel::IntersectionKHR
                                        | rspirv::spirv::ExecutionModel::AnyHitKHR
                                        | rspirv::spirv::ExecutionModel::ClosestHitKHR
                                        | rspirv::spirv::ExecutionModel::MissKHR
                                        | rspirv::spirv::ExecutionModel::CallableKHR
                                );
                                if !allow_input {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassInvalid {
                                            entry_point: entry_point_id,
                                            interface: id.into_inner(),
                                            storage_class: *storage,
                                        },
                                    );
                                }
                                if let Some(pointer_type) =
                                    inst.result_type.and_then(|ty| ResultId::try_from(ty).ok())
                                {
                                    if let Some(pointer_inst) = definitions.get(&pointer_type) {
                                        if let Some(rspirv::dr::Operand::IdRef(pointee)) =
                                            pointer_inst.operands.get(1)
                                        {
                                            if let Ok(pointee_id) = ResultId::try_from(*pointee) {
                                                let mut seen_types = HashSet::new();
                                                if let Some(encoding) =
                                                    contains_disallowed_fp_encoding(
                                                        definitions,
                                                        pointee_id,
                                                        &mut seen_types,
                                                    )
                                                {
                                                    return Err(
                                                        ValidationError::EntryPointInterfaceFloatEncodingInvalid {
                                                            interface: id.into(),
                                                            storage_class: *storage,
                                                            encoding,
                                                        },
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            rspirv::spirv::StorageClass::Output => {
                                let allow_output = matches!(
                                    exec_model,
                                    rspirv::spirv::ExecutionModel::Vertex
                                        | rspirv::spirv::ExecutionModel::TessellationControl
                                        | rspirv::spirv::ExecutionModel::TessellationEvaluation
                                        | rspirv::spirv::ExecutionModel::Geometry
                                        | rspirv::spirv::ExecutionModel::Fragment
                                        | rspirv::spirv::ExecutionModel::MeshEXT
                                        | rspirv::spirv::ExecutionModel::TaskEXT
                                );
                                if !allow_output {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassInvalid {
                                            entry_point: entry_point_id,
                                            interface: id.into_inner(),
                                            storage_class: *storage,
                                        },
                                    );
                                }
                                if let Some(pointer_type) =
                                    inst.result_type.and_then(|ty| ResultId::try_from(ty).ok())
                                {
                                    if let Some(pointer_inst) = definitions.get(&pointer_type) {
                                        if let Some(rspirv::dr::Operand::IdRef(pointee)) =
                                            pointer_inst.operands.get(1)
                                        {
                                            if let Ok(pointee_id) = ResultId::try_from(*pointee) {
                                                let mut seen_types = HashSet::new();
                                                if let Some(encoding) =
                                                    contains_disallowed_fp_encoding(
                                                        definitions,
                                                        pointee_id,
                                                        &mut seen_types,
                                                    )
                                                {
                                                    return Err(
                                                        ValidationError::EntryPointInterfaceFloatEncodingInvalid {
                                                            interface: id.into(),
                                                            storage_class: *storage,
                                                            encoding,
                                                        },
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            rspirv::spirv::StorageClass::Function => {
                                return Err(
                                    ValidationError::EntryPointInterfaceStorageClassInvalid {
                                        entry_point: entry_point_id,
                                        interface: id.into_inner(),
                                        storage_class: *storage,
                                    },
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn validate_execution_modes(
    module: &Module,
    entry_points: &HashSet<ResultId>,
    env: TargetEnv,
    options: &ValidationOptions,
    capabilities: &HashSet<rspirv::spirv::Capability>,
) -> Result<(), ValidationError> {
    let mut entry_point_models: HashMap<ResultId, rspirv::spirv::ExecutionModel> = HashMap::new();
    for ep in &module.entry_points {
        let mut operands = ep.operands.iter();
        if ep.class.opcode == rspirv::spirv::Op::ConditionalEntryPointINTEL {
            let _ = operands.next();
        }
        let execution_model = operands.next().and_then(|op| match op {
            rspirv::dr::Operand::ExecutionModel(model) => Some(*model),
            _ => None,
        });
        let function = operands.next().and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
            _ => None,
        });
        if let (Some(model), Some(function)) = (execution_model, function) {
            entry_point_models.insert(function, model);
        }
    }

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

        let execution_mode = execution_mode_from_operand(mode.operands.get(1));
        if let Some(execution_mode) = execution_mode {
            if execution_mode == rspirv::spirv::ExecutionMode::LocalSizeId
                && !local_size_id_allowed(env, options)
            {
                return Err(ValidationError::LocalSizeIdNotAllowed { env });
            }
            if let Some(model) = entry_point_models.get(&function) {
                match execution_mode {
                    rspirv::spirv::ExecutionMode::OutputVertices => {
                        let allowed = [
                            rspirv::spirv::ExecutionModel::Geometry,
                            rspirv::spirv::ExecutionModel::TessellationControl,
                            rspirv::spirv::ExecutionModel::MeshEXT,
                            rspirv::spirv::ExecutionModel::MeshNV,
                        ];
                        if !allowed.contains(model) {
                            return Err(ValidationError::ExecutionModeRequiresExecutionModel {
                                entry_point: function.into_inner(),
                                mode: execution_mode,
                                execution_model: *model,
                                allowed_models: allowed.to_vec(),
                            });
                        }
                        if env.is_vulkan()
                            && capabilities.contains(&rspirv::spirv::Capability::MeshShadingEXT)
                            && matches!(mode.operands.get(2), Some(rspirv::dr::Operand::LiteralBit32(v)) if *v == 0)
                            && (*model == rspirv::spirv::ExecutionModel::MeshEXT
                                || *model == rspirv::spirv::ExecutionModel::MeshNV)
                        {
                            return Err(ValidationError::InvalidExecutionModeValue {
                                entry_point: function.into_inner(),
                                mode: execution_mode,
                                value: 0,
                            });
                        }
                    }
                    rspirv::spirv::ExecutionMode::OutputLinesEXT
                    | rspirv::spirv::ExecutionMode::OutputTrianglesEXT
                    | rspirv::spirv::ExecutionMode::OutputPrimitivesEXT => {
                        let allowed = [
                            rspirv::spirv::ExecutionModel::MeshEXT,
                            rspirv::spirv::ExecutionModel::MeshNV,
                        ];
                        if !allowed.contains(model) {
                            return Err(ValidationError::ExecutionModeRequiresExecutionModel {
                                entry_point: function.into_inner(),
                                mode: execution_mode,
                                execution_model: *model,
                                allowed_models: allowed.to_vec(),
                            });
                        }
                        if env.is_vulkan()
                            && capabilities.contains(&rspirv::spirv::Capability::MeshShadingEXT)
                            && execution_mode == rspirv::spirv::ExecutionMode::OutputPrimitivesEXT
                            && matches!(mode.operands.get(2), Some(rspirv::dr::Operand::LiteralBit32(v)) if *v == 0)
                        {
                            return Err(ValidationError::InvalidExecutionModeValue {
                                entry_point: function.into_inner(),
                                mode: execution_mode,
                                value: 0,
                            });
                        }
                    }
                    _ => {}
                }
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

fn validate_result_types_are_types(
    instructions: &HashMap<ResultId, rspirv::dr::Instruction>,
    opcodes: &HashMap<ResultId, rspirv::spirv::Op>,
) -> Result<(), ValidationError> {
    for inst in instructions.values() {
        if let Some(result_type_raw) = inst.result_type {
            if let Ok(type_id) = ResultId::try_from(result_type_raw) {
                if let Some(type_opcode) = opcodes.get(&type_id) {
                    if !is_type_opcode(*type_opcode) {
                        return Err(ValidationError::ResultTypeNotType {
                            instruction: inst.class.opcode,
                            result_type: Id::from(type_id),
                            found: *type_opcode,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::manual_is_multiple_of)]
fn is_block_operand(opcode: rspirv::spirv::Op, index: usize) -> bool {
    match opcode {
        rspirv::spirv::Op::Branch => index == 0,
        rspirv::spirv::Op::BranchConditional => index == 1 || index == 2,
        rspirv::spirv::Op::Switch => index == 1 || (index > 1 && index % 2 == 0),
        rspirv::spirv::Op::LoopMerge => index <= 1,
        rspirv::spirv::Op::SelectionMerge => index == 0,
        rspirv::spirv::Op::Phi => index % 2 == 1,
        _ => false,
    }
}

fn check_instruction_ids(
    inst: &rspirv::dr::Instruction,
    defined_ids: &HashSet<Id>,
    function: Option<Id>,
) -> Result<(), ValidationError> {
    if let Some(result_type) = inst.result_type {
        if let Ok(id) = Id::try_from(result_type) {
            if !defined_ids.contains(&id) {
                return Err(ValidationError::UndefinedId { function, id });
            }
        }
    }

    for (idx, operand) in inst.operands.iter().enumerate() {
        if is_block_operand(inst.class.opcode, idx) {
            continue;
        }
        if let rspirv::dr::Operand::IdRef(raw) = operand {
            if let Ok(id) = Id::try_from(*raw) {
                if !defined_ids.contains(&id) {
                    return Err(ValidationError::UndefinedId { function, id });
                }
            }
        }
    }
    Ok(())
}

fn validate_operand_definitions(
    module: &Module,
    defined_ids: &HashSet<Id>,
) -> Result<(), ValidationError> {
    for inst in &module.types_global_values {
        check_instruction_ids(inst, defined_ids, None)?;
    }
    for function in &module.functions {
        let function_id = function
            .def
            .as_ref()
            .and_then(|def| def.result_id)
            .and_then(|raw| Id::try_from(raw).ok());
        if let Some(def) = &function.def {
            check_instruction_ids(def, defined_ids, function_id)?;
        }
        for param in &function.parameters {
            check_instruction_ids(param, defined_ids, function_id)?;
        }
        for block in &function.blocks {
            for inst in &block.instructions {
                check_instruction_ids(inst, defined_ids, function_id)?;
            }
        }
    }
    Ok(())
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
            | rspirv::spirv::Op::TypeUntypedPointerKHR
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
mod tests;
