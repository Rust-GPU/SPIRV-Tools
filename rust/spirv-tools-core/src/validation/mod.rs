use std::{
    collections::{HashMap, HashSet},
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
    CheckedBound, DeclaredBound, DecorationTargetId, DecorationTargetKind, ExtensionName, Id,
    IdBound, IdKind, MemberDecorationTargetId, MemberIndex, MergeTargetKind, ModuleWords,
    OperandId, ResultId, Schema, TypeId, ZeroIdError,
};

// Validator options and limits
pub mod options;
pub use options::{
    ValidationLimits, ValidationOptions, LIMIT_MAX_ACCESS_CHAIN_INDEXES,
    LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH, LIMIT_MAX_FUNCTION_ARGS, LIMIT_MAX_GLOBAL_VARIABLES,
    LIMIT_MAX_ID_BOUND, LIMIT_MAX_LOCAL_VARIABLES, LIMIT_MAX_STRUCT_DEPTH,
    LIMIT_MAX_STRUCT_MEMBERS, LIMIT_MAX_SWITCH_BRANCHES,
};

// Validated header
pub mod header;
pub use header::ValidatedHeader;

// Friendly names for error messages
pub mod friendly_names;
pub use friendly_names::{
    build_friendly_name_table, format_validation_error, format_validation_error_from_words,
    FriendlyNames,
};

// ValidModule and related types
pub mod valid_module;
pub use valid_module::{MaybeValidModule, ValidModule, ValidModuleCache, ValidatableModule};

// Shared helper utilities
pub mod helpers;

// Type extension traits for rspirv types
pub mod type_ext;
pub use type_ext::{DefaultTypeResolver, TypeInstructionExt, TypeResolver};

// Opcode classification extension traits
pub mod op_ext;
pub use op_ext::{BuiltInExt, DecorationExt, OpExt};

// Validation context and rule trait
pub mod context;
pub use context::{run_boxed_rules, run_rules, TestContextData, ValidationContext, ValidationRule};

// CFG analysis utilities
pub mod cfg_analysis;
pub use cfg_analysis::{
    get_block_label, get_merge_info, get_terminator, ControlFlowGraph, MergeInfo,
};

// Source span information for rich error reporting
pub mod span;
pub use span::{
    extract_source_snippet, spanned_err, LabelKind, LabeledSpan, SourceLocation, SourceSnippet,
    SourceSpan, SpanLabel, SpanMap, SpannedError, SpannedResult, SpannedValidationError,
    ValidationErrorExt, ValidationResult, WithSpan,
};

// Validation rules organized by category
pub mod rules;
use helpers::{
    collect_declared_capabilities, collect_execution_models, collect_result_instructions,
    collect_result_opcodes, collect_result_types, is_memory_object_declaration,
};
use rules::capabilities::{
    capability_operand, capability_satisfied, required_extension_for_capability,
    validate_capabilities,
};
use rules::extensions::{
    extension_operand, extension_satisfied, has_extension, validate_extension_allowlist,
    validate_extensions, ExtensionSet,
};
use rules::limits::all_limit_rules;

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
    validate_words_internal(ModuleWords::from(Arc::from(words)), env, options, None)
        .map(|_| ())
        .map_err(|e| e.into())
}

/// Validates a SPIR-V module with a span map for rich error reporting.
///
/// When validation fails, the returned error will contain source spans
/// pointing to the definition sites of the offending IDs.
///
/// Use `assemble_text_with_spans` to get both the SPIR-V binary and span map.
pub fn validate_module_with_spans(
    words: &[u32],
    env: TargetEnv,
    options: ValidationOptions,
    span_map: &span::SpanMap,
) -> Result<(), SpannedValidationError> {
    validate_words_internal(
        ModuleWords::from(Arc::from(words)),
        env,
        options,
        Some(span_map),
    )
    .map(|_| ())
}

pub(crate) fn validate_words_internal(
    words: ModuleWords,
    env: TargetEnv,
    options: ValidationOptions,
    span_map: Option<&span::SpanMap>,
) -> Result<ValidModule, SpannedValidationError> {
    if let Some(&schema) = words.as_slice().get(4) {
        Schema::validate(schema)?;
    }
    run_layout_check(words.as_slice(), env)?;
    let module = parse_module(words.as_slice())?;
    validate_extension_allowlist(&module, env)?;
    let header = ValidatedHeader::from_module(&module)?;
    if let Some(&limit) = options.limits.get(&LIMIT_MAX_ID_BOUND) {
        if header.bound().declared().0 > limit {
            return Err(ValidationError::IdBoundExceedsLimit {
                declared: header.bound().declared(),
                limit,
            }
            .into());
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
    validate_decoration_groups(
        &module,
        &defined_result_ids,
        &opcodes,
        &struct_member_counts,
    )?;
    validate_decorations(&module, &defined_result_ids)?;
    let entry_models = collect_execution_models(&module);
    validate_decoration_target_categories(&module, &opcodes, &definitions, &capabilities)?;
    let _entry_points =
        validate_entry_points(&module, &defined_result_ids, &opcodes, &definitions)?;
    // Entry point interface storage class validation is handled by EntryPointInterfaceStorageClassesRule in entry_points.rs
    // Entry point location conflict validation is handled by LocationConflictRule in interfaces.rs
    // Execution mode validation is handled by ExecutionModesRule in execution_modes.rs
    // Function validation is handled by modular rules in cfg.rs, functions.rs, memory.rs, arithmetics.rs, etc.
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
        span_map,
    };
    run_rules(&validation_ctx, &all_limit_rules())?;
    run_rules(
        &validation_ctx,
        &rules::block_layout::all_block_layout_rules(),
    )?;
    run_rules(&validation_ctx, &rules::vulkan::all_vulkan_rules())?;
    run_rules(&validation_ctx, &rules::opengl::all_opengl_rules())?;
    run_rules(&validation_ctx, &rules::pointers::all_pointer_rules())?;
    run_rules(&validation_ctx, &rules::decorations::all_decoration_rules())?;
    run_rules(
        &validation_ctx,
        &rules::storage_classes::all_storage_class_rules(),
    )?;
    run_rules(&validation_ctx, &rules::builtins::all_builtin_rules())?;
    run_rules(
        &validation_ctx,
        &rules::interpolation::all_interpolation_rules(),
    )?;
    run_rules(&validation_ctx, &rules::composites::all_composite_rules())?;
    run_rules(&validation_ctx, &rules::cfg::all_cfg_rules())?;
    run_rules(
        &validation_ctx,
        &rules::entry_points::all_entry_point_rules(),
    )?;
    run_rules(
        &validation_ctx,
        &rules::execution_modes::all_execution_mode_rules(),
    )?;
    run_rules(&validation_ctx, &rules::functions::all_function_rules())?;
    run_rules(&validation_ctx, &rules::adjacency::all_adjacency_rules())?;
    run_rules(&validation_ctx, &rules::arithmetics::all_arithmetic_rules())?;
    run_rules(&validation_ctx, &rules::atomics::all_atomic_rules())?;
    run_rules(&validation_ctx, &rules::barriers::all_barrier_rules())?;
    run_rules(&validation_ctx, &rules::bitwise::all_bitwise_rules())?;
    run_rules(&validation_ctx, &rules::constants::all_constant_rules())?;
    run_rules(&validation_ctx, &rules::conversion::all_conversion_rules())?;
    run_rules(&validation_ctx, &rules::derivatives::all_derivative_rules())?;
    run_rules(&validation_ctx, &rules::image::all_image_rules())?;
    run_rules(&validation_ctx, &rules::literals::all_literal_rules())?;
    run_rules(&validation_ctx, &rules::logicals::all_logical_rules())?;
    run_rules(&validation_ctx, &rules::memory::all_memory_rules())?;
    run_rules(&validation_ctx, &rules::types::all_type_rules())?;
    run_rules(&validation_ctx, &rules::group::all_group_rules())?;
    run_rules(
        &validation_ctx,
        &rules::dot_product::all_dot_product_rules(),
    )?;

    // Box-based rules (return Vec<Box<dyn ValidationRule>>)
    run_boxed_rules(&validation_ctx, &rules::annotation::all_annotation_rules())?;
    run_boxed_rules(&validation_ctx, &rules::debug::all_debug_rules())?;
    run_boxed_rules(&validation_ctx, &rules::debug_info::all_debug_info_rules())?;
    run_boxed_rules(&validation_ctx, &rules::graph::all_graph_rules())?;
    run_boxed_rules(&validation_ctx, &rules::hit_object::all_hit_object_rules())?;
    run_boxed_rules(&validation_ctx, &rules::interfaces::all_interface_rules())?;
    run_boxed_rules(
        &validation_ctx,
        &rules::invalid_type::all_invalid_type_rules(),
    )?;
    run_boxed_rules(
        &validation_ctx,
        &rules::memory_semantics::all_memory_semantics_rules(),
    )?;
    run_boxed_rules(
        &validation_ctx,
        &rules::mesh_shading::all_mesh_shading_rules(),
    )?;
    run_boxed_rules(&validation_ctx, &rules::misc::all_misc_rules())?;
    run_boxed_rules(
        &validation_ctx,
        &rules::mode_setting::all_mode_setting_rules(),
    )?;
    run_boxed_rules(
        &validation_ctx,
        &rules::non_uniform::all_non_uniform_rules(),
    )?;
    run_boxed_rules(&validation_ctx, &rules::primitives::all_primitive_rules())?;
    run_boxed_rules(
        &validation_ctx,
        &rules::ray_tracing::all_ray_tracing_rules(),
    )?;
    run_boxed_rules(&validation_ctx, &rules::scopes::all_scope_rules())?;
    run_boxed_rules(
        &validation_ctx,
        &rules::small_type_uses::all_small_type_uses_rules(),
    )?;
    run_boxed_rules(&validation_ctx, &rules::tensor::all_tensor_rules())?;
    run_boxed_rules(
        &validation_ctx,
        &rules::tensor_layout::all_tensor_layout_rules(),
    )?;

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
            let has_any_capability = inst
                .class
                .capabilities
                .iter()
                .any(|&required_cap| capability_satisfied(required_cap, capabilities));
            // Special case for PtrDiff which has alternative capabilities
            let ptrdiff_alternative = inst.class.opcode == rspirv::spirv::Op::PtrDiff
                && (capabilities.contains(&rspirv::spirv::Capability::UntypedPointersKHR)
                    || capabilities
                        .contains(&rspirv::spirv::Capability::PhysicalStorageBufferAddresses));
            // Special case for AMD Shader Ballot: OpGroup*NonUniformAMD opcodes normally require
            // Group capability, but when SPV_AMD_shader_ballot extension is present, the capability
            // requirement is waived.
            let amd_shader_ballot_alternative = matches!(
                inst.class.opcode,
                rspirv::spirv::Op::GroupIAddNonUniformAMD
                    | rspirv::spirv::Op::GroupFAddNonUniformAMD
                    | rspirv::spirv::Op::GroupFMinNonUniformAMD
                    | rspirv::spirv::Op::GroupUMinNonUniformAMD
                    | rspirv::spirv::Op::GroupSMinNonUniformAMD
                    | rspirv::spirv::Op::GroupFMaxNonUniformAMD
                    | rspirv::spirv::Op::GroupUMaxNonUniformAMD
                    | rspirv::spirv::Op::GroupSMaxNonUniformAMD
            ) && extensions
                .values
                .iter()
                .any(|ext| ext.as_str() == "SPV_AMD_shader_ballot");
            if !has_any_capability && !ptrdiff_alternative && !amd_shader_ballot_alternative {
                // Report the first required capability for the error message
                return Err(ValidationError::MissingInstructionCapability {
                    opcode: inst.class.opcode,
                    required_capability: inst.class.capabilities[0],
                });
            }
        }
        // Instruction extensions are disjunctive - you need AT LEAST ONE from the list
        if !inst.class.extensions.is_empty() {
            let has_any_extension =
                inst.class.extensions.iter().any(|&required_ext| {
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
                    && inst
                        .class
                        .extensions
                        .iter()
                        .any(|&ext| extension_satisfied(ext, extensions, target_version));
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
                    // Check if an enabling extension is declared that can
                    // relax the version requirement (matching C++ spirv-val's
                    // OperandVersionExtensionCheck).
                    let has_enabling_extension = operand
                        .required_extensions()
                        .iter()
                        .any(|ext| has_extension(extensions, ext))
                        || grammar_required_extensions_for_operand(operand)
                            .iter()
                            .any(|ext| has_extension(extensions, ext));

                    if !has_enabling_extension {
                        return Err(ValidationError::OperandRequiresSpirvVersion {
                            opcode: inst.class.opcode,
                            operand_index: index,
                            required_version,
                            target_version,
                        });
                    }
                }
            }
            // Collect ALL capabilities from all sources and check DISJUNCTIVELY
            // (like C++ spirv-val's HasAnyOfCapabilities).
            // The grammar lists alternatives like [Kernel, GroupNonUniformArithmetic, GroupNonUniformBallot]
            // and you need AT LEAST ONE from the combined set.
            let mut all_required_caps: Vec<rspirv::spirv::Capability> = Vec::new();
            all_required_caps.extend(operand.required_capabilities());
            all_required_caps.extend(
                manual_required_capabilities_for_operand(operand)
                    .iter()
                    .copied(),
            );
            all_required_caps.extend(grammar_required_capabilities_for_operand(operand));

            if !all_required_caps.is_empty() {
                let has_any = all_required_caps
                    .iter()
                    .any(|&cap| capability_satisfied(cap, capabilities));
                if !has_any {
                    return Err(ValidationError::MissingOperandCapability {
                        opcode: inst.class.opcode,
                        operand_index: index,
                        required_capability: all_required_caps[0],
                    });
                }
            }
            // Collect ALL extensions from all sources and check DISJUNCTIVELY
            // (like C++ spirv-val's HasAnyOfExtensions).
            // The grammar lists alternatives and you need AT LEAST ONE
            // from the combined set.
            let mut all_required_exts: Vec<&str> = Vec::new();
            all_required_exts.extend(operand.required_extensions());
            all_required_exts.extend(grammar_required_extensions_for_operand(operand));

            if !all_required_exts.is_empty() {
                let has_any = all_required_exts
                    .iter()
                    .any(|&ext| extension_satisfied(ext, extensions, target_version));
                if !has_any {
                    return Err(ValidationError::MissingOperandExtension {
                        opcode: inst.class.opcode,
                        operand_index: index,
                        required_extension: ExtensionName::from(all_required_exts[0]),
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

// Entry point interface storage class validation has been moved to
// EntryPointInterfaceStorageClassesRule in rules/entry_points.rs

// Execution mode validation has been moved to
// ExecutionModesRule in rules/execution_modes.rs

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
