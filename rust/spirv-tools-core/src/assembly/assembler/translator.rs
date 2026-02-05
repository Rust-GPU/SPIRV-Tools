use core::convert::TryFrom;
use std::str::FromStr;

use rspirv::binary::Assemble;
use rspirv::dr::{self, Error as BuildError, InsertPoint};
use rspirv::grammar::{LogicalOperand, OperandKind, OperandQuantifier};
use rspirv::spirv;

use crate::assembly::decoration::decoration_operand_descriptors;
use crate::assembly::ext_inst::{lookup_custom_ext_inst_opcode, ExtInstImportInfo, ResolvedExtInst};
use crate::assembly::instruction::LiteralNumber;
use crate::assembly::options::TextToBinaryOptions;
use crate::assembly::parser::{
    parse_instruction_with_origin, OperandValue, ParseError, ParsedInstruction, ParsedOperand,
};
use crate::diagnostic::{DiagnosticMessage, MessagePosition};
use crate::string_literal::parse_string_literal;
use crate::target_env::TargetEnv;
use crate::validation::span::SpanMap;

use super::module_builder::ModuleBuilder;
use super::types::{
    ArrayTypeInfo, CompositeTypeInfo, MatrixMajorness, MatrixTypeInfo, StructTypeInfo,
    VectorTypeInfo,
};
use super::{AssemblyError, AssemblyWithSpans};

/// Drives translation from parsed instructions to SPIR-V DR form.
pub struct AssemblyTranslator<'a> {
    module_builder: ModuleBuilder<'a>,
    builder: dr::Builder,
}

impl<'a> Default for AssemblyTranslator<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> AssemblyTranslator<'a> {
    /// Creates a new translator with an empty module.
    pub fn new() -> Self {
        Self::with_target_env(TargetEnv::Universal1_6)
    }

    /// Creates a new translator configured for the requested target environment.
    pub fn with_target_env(env: TargetEnv) -> Self {
        Self::with_target_env_and_options(env, TextToBinaryOptions::NONE)
    }

    /// Creates a translator configured for the provided environment and options.
    pub fn with_target_env_and_options(env: TargetEnv, options: TextToBinaryOptions) -> Self {
        Self::with_full_options(env, options, false)
    }

    /// Creates a translator with full control over all options including span tracking.
    pub fn with_full_options(
        env: TargetEnv,
        options: TextToBinaryOptions,
        track_spans: bool,
    ) -> Self {
        let mut builder = dr::Builder::new();
        configure_builder_for_env(&mut builder, env);
        let preserve_numeric_ids = options.contains(TextToBinaryOptions::PRESERVE_NUMERIC_IDS);
        Self {
            module_builder: ModuleBuilder::with_options(preserve_numeric_ids, track_spans),
            builder,
        }
    }

    fn reserve_numeric_result_ids(&mut self, instructions: &[ParsedInstruction<'a>]) {
        if !self.module_builder.preserve_numeric_ids() {
            return;
        }
        for instruction in instructions {
            if let Some(result) = instruction.result_id() {
                if let Some(numeric) = result.as_spirv_id().as_numeric() {
                    self.module_builder.reserve_numeric_id(numeric.get());
                }
            }
        }
    }

    /// Translates a parsed instruction.
    pub fn translate(&mut self, instruction: &ParsedInstruction<'a>) {
        match instruction.opcode() {
            spirv::Op::Capability => self.translate_capability(instruction),
            spirv::Op::Extension => self.translate_extension(instruction),
            spirv::Op::ConditionalExtensionINTEL => {
                self.translate_conditional_extension(instruction)
            }
            spirv::Op::ExtInstImport => self.translate_ext_inst_import(instruction),
            spirv::Op::TypeVoid => self.translate_type_void(instruction),
            spirv::Op::TypeBool => self.translate_type_bool(instruction),
            spirv::Op::TypeInt => self.translate_type_int(instruction),
            spirv::Op::TypeFloat => self.translate_type_float(instruction),
            spirv::Op::TypeFunction => self.translate_type_function(instruction),
            spirv::Op::TypePointer => self.translate_type_pointer(instruction),
            spirv::Op::TypeVector => self.translate_type_vector(instruction),
            spirv::Op::TypeArray => self.translate_type_array(instruction),
            spirv::Op::TypeRuntimeArray => self.translate_type_runtime_array(instruction),
            spirv::Op::TypeStruct => self.translate_type_struct(instruction),
            spirv::Op::TypeMatrix => self.translate_type_matrix(instruction),
            spirv::Op::TypeImage => self.translate_type_image(instruction),
            spirv::Op::TypeSampledImage => self.translate_type_sampled_image(instruction),
            spirv::Op::TypeSampler => self.translate_type_sampler(instruction),
            spirv::Op::MemoryModel => self.translate_memory_model(instruction),
            spirv::Op::EntryPoint => self.translate_entry_point(instruction),
            spirv::Op::ExecutionMode => self.translate_execution_mode(instruction),
            spirv::Op::Name => self.translate_name(instruction),
            spirv::Op::MemberName => self.translate_member_name(instruction),
            spirv::Op::AccessChain => self.translate_access_chain(instruction, false),
            spirv::Op::InBoundsAccessChain => self.translate_access_chain(instruction, true),
            spirv::Op::CopyMemory => self.translate_copy_memory(instruction, false),
            spirv::Op::CopyMemorySized => self.translate_copy_memory(instruction, true),
            spirv::Op::SelectionMerge => self.translate_selection_merge(instruction),
            spirv::Op::LoopMerge => self.translate_loop_merge(instruction),
            spirv::Op::Decorate => self.translate_decorate(instruction),
            spirv::Op::DecorateId => self.translate_decorate_id(instruction),
            spirv::Op::DecorateString => self.translate_decorate_string(instruction),
            spirv::Op::MemberDecorate => self.translate_member_decorate(instruction),
            spirv::Op::MemberDecorateString => self.translate_member_decorate_string(instruction),
            spirv::Op::CompositeConstruct => self.translate_composite_construct(instruction),
            spirv::Op::VectorShuffle => self.translate_vector_shuffle(instruction),
            spirv::Op::CompositeExtract => self.translate_composite_extract(instruction),
            spirv::Op::CompositeInsert => self.translate_composite_insert(instruction),
            spirv::Op::ExtInst => self.translate_ext_inst(instruction),
            spirv::Op::Phi => self.translate_phi(instruction),
            spirv::Op::ConstantTrue => self.translate_boolean_constant(instruction, true),
            spirv::Op::ConstantFalse => self.translate_boolean_constant(instruction, false),
            spirv::Op::ConstantComposite => self.translate_constant_composite(instruction),
            spirv::Op::ConstantNull => self.translate_constant_null(instruction),
            spirv::Op::Function => self.translate_function(instruction),
            spirv::Op::FunctionParameter => self.translate_function_parameter(instruction),
            spirv::Op::Label => self.translate_label(instruction),
            spirv::Op::Branch => self.translate_branch(instruction),
            spirv::Op::BranchConditional => self.translate_branch_conditional(instruction),
            spirv::Op::Return => self.translate_return(instruction),
            spirv::Op::ReturnValue => self.translate_return_value(instruction),
            spirv::Op::FunctionEnd => self.translate_function_end(instruction),
            spirv::Op::Constant => self.translate_constant(instruction),
            spirv::Op::Variable => self.translate_variable(instruction),
            spirv::Op::Load => self.translate_load(instruction),
            spirv::Op::Store => self.translate_store(instruction),
            spirv::Op::IAdd
            | spirv::Op::ISub
            | spirv::Op::IMul
            | spirv::Op::FAdd
            | spirv::Op::FSub
            | spirv::Op::FMul => self.translate_binary_arithmetic(instruction),
            spirv::Op::Bitcast
            | spirv::Op::ConvertFToU
            | spirv::Op::ConvertFToS
            | spirv::Op::ConvertSToF
            | spirv::Op::ConvertUToF
            | spirv::Op::UConvert
            | spirv::Op::SConvert
            | spirv::Op::FConvert => self.translate_unary_op(instruction),
            // Image operations: typed result + 1 IdRef (no image operands)
            spirv::Op::Image
            | spirv::Op::ImageQueryFormat
            | spirv::Op::ImageQueryOrder
            | spirv::Op::ImageQuerySize
            | spirv::Op::ImageQueryLevels
            | spirv::Op::ImageQuerySamples
            | spirv::Op::ImageSparseTexelsResident => self.translate_image_op(instruction, 1),
            // Image operations: typed result + 2 IdRef (no image operands)
            spirv::Op::SampledImage
            | spirv::Op::ImageQuerySizeLod
            | spirv::Op::ImageQueryLod => self.translate_image_op(instruction, 2),
            // Image operations: typed result + 3 IdRef (no image operands)
            spirv::Op::ImageTexelPointer => self.translate_image_op(instruction, 3),
            // Image operations: typed result + 2 IdRef + optional/required ImageOperands
            spirv::Op::ImageSampleImplicitLod
            | spirv::Op::ImageSampleExplicitLod
            | spirv::Op::ImageSampleProjImplicitLod
            | spirv::Op::ImageSampleProjExplicitLod
            | spirv::Op::ImageFetch
            | spirv::Op::ImageRead
            | spirv::Op::ImageSparseSampleImplicitLod
            | spirv::Op::ImageSparseSampleExplicitLod
            | spirv::Op::ImageSparseSampleProjImplicitLod
            | spirv::Op::ImageSparseSampleProjExplicitLod
            | spirv::Op::ImageSparseFetch
            | spirv::Op::ImageSparseRead => self.translate_image_op(instruction, 2),
            // Image operations: typed result + 3 IdRef + optional/required ImageOperands
            spirv::Op::ImageSampleDrefImplicitLod
            | spirv::Op::ImageSampleDrefExplicitLod
            | spirv::Op::ImageSampleProjDrefImplicitLod
            | spirv::Op::ImageSampleProjDrefExplicitLod
            | spirv::Op::ImageGather
            | spirv::Op::ImageDrefGather
            | spirv::Op::ImageSparseSampleDrefImplicitLod
            | spirv::Op::ImageSparseSampleDrefExplicitLod
            | spirv::Op::ImageSparseSampleProjDrefImplicitLod
            | spirv::Op::ImageSparseSampleProjDrefExplicitLod
            | spirv::Op::ImageSparseGather
            | spirv::Op::ImageSparseDrefGather => self.translate_image_op(instruction, 3),
            // ImageWrite: no result, 3 IdRef + optional ImageOperands
            spirv::Op::ImageWrite => self.translate_image_write(instruction),
            _ => self
                .module_builder
                .emit_error(instruction.opcode_position(), "unsupported opcode"),
        }
    }

    fn record_instruction(&mut self, _inst: dr::Instruction) {}

    fn record_from_module<F>(&mut self, fetch: F)
    where
        F: FnOnce(&dr::Module) -> Option<dr::Instruction>,
    {
        if let Some(inst) = fetch(self.builder.module_ref()) {
            self.record_instruction(inst);
        }
    }

    fn record_from_current_block(&mut self) {
        if let (Some(function), Some(block)) = (
            self.builder.selected_function(),
            self.builder.selected_block(),
        ) {
            self.record_from_module(|module| {
                module
                    .functions
                    .get(function)
                    .and_then(|f| f.blocks.get(block))
                    .and_then(|bb| bb.instructions.last())
                    .cloned()
            });
        }
    }

    fn record_function_def(&mut self) {}

    fn record_function_param(&mut self) {
        if let Some(function) = self.builder.selected_function() {
            self.record_from_module(|module| {
                module
                    .functions
                    .get(function)
                    .and_then(|f| f.parameters.last())
                    .cloned()
            });
        }
    }

    fn translate_capability(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCapability missing enumerant");
            return;
        };
        let Some(capability) =
            self.parse_enum_operand::<spirv::Capability>(Some(operand), "capability", opcode_pos)
        else {
            return;
        };
        self.builder.capability(capability);
        self.record_from_module(|module| module.capabilities.last().cloned());
    }

    fn translate_extension(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        if instruction.result_id().is_some() {
            self.module_builder
                .emit_error(opcode_pos, "OpExtension does not produce a result id");
            return;
        }
        let Some(name_operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtension missing extension name");
            return;
        };
        let name = match name_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "OpExtension operand must be a literal string",
                );
                return;
            }
        };
        if let Some(extra) = instruction.operands().get(1) {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpExtension received unexpected operands",
            );
            return;
        }
        self.builder.extension(name);
        self.record_from_module(|module| module.extensions.last().cloned());
    }

    fn translate_conditional_extension(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        if instruction.result_id().is_some() {
            self.module_builder.emit_error(
                opcode_pos,
                "OpConditionalExtensionINTEL does not produce a result id",
            );
            return;
        }
        let mut operands = instruction.operands().iter();
        let Some(condition_operand) = operands.next() else {
            self.module_builder.emit_error(
                opcode_pos,
                "OpConditionalExtensionINTEL missing condition id",
            );
            return;
        };
        let Some(extension_operand) = operands.next() else {
            self.module_builder.emit_error(
                opcode_pos,
                "OpConditionalExtensionINTEL missing extension name",
            );
            return;
        };
        let Some(condition_id) =
            self.operand_as_id(condition_operand, "OpConditionalExtensionINTEL condition")
        else {
            return;
        };
        let extension = match extension_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    extension_operand.span().start(),
                    "OpConditionalExtensionINTEL extension must be a literal string",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpConditionalExtensionINTEL received unexpected operands",
            );
            return;
        }
        let inst = dr::Instruction::new(
            spirv::Op::ConditionalExtensionINTEL,
            None,
            None,
            vec![
                dr::Operand::IdRef(condition_id),
                dr::Operand::LiteralString(extension),
            ],
        );
        self.builder.module_mut().extensions.push(inst);
        self.record_from_module(|module| module.extensions.last().cloned());
    }

    fn translate_ext_inst_import(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInstImport missing result id");
            return;
        };
        let Some(name_operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInstImport missing import name");
            return;
        };
        let name = match name_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "OpExtInstImport operand must be a literal string",
                );
                return;
            }
        };
        if instruction.operands().len() > 1 {
            if let Some(extra) = instruction.operands().get(1) {
                self.module_builder.emit_error(
                    extra.span().start(),
                    "OpExtInstImport only accepts a single operand",
                );
            }
            return;
        }
        let numeric_id = self.module_builder.resolve_result_id(result_id);
        let info = ExtInstImportInfo::new(&name);
        let inst = dr::Instruction::new(
            spirv::Op::ExtInstImport,
            None,
            Some(numeric_id),
            vec![dr::Operand::LiteralString(name)],
        );
        self.builder.module_mut().ext_inst_imports.push(inst);
        self.module_builder.note_ext_inst_import(numeric_id, info);
        self.record_from_module(|module| module.ext_inst_imports.last().cloned());
    }

    fn translate_type_void(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeVoid requires a result id");
            return;
        };
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_void_id(Some(result_id));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_bool(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeBool requires a result id");
            return;
        };
        if !instruction.operands().is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeBool does not take operands");
            return;
        }
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_bool_id(Some(result_id));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_int(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeInt requires a result identifier");
            return;
        };

        let mut operands = instruction.operands().iter();
        let Some(width_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeInt missing width literal");
            return;
        };
        let width_literal = match width_operand.value() {
            OperandValue::Literal(lit) => lit,
            _ => {
                self.module_builder.emit_error(
                    width_operand.span().start(),
                    "Width operand must be literal",
                );
                return;
            }
        };
        let Some(signed_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeInt missing signedness literal");
            return;
        };
        let signed_literal = match signed_operand.value() {
            OperandValue::Literal(lit) => lit,
            _ => {
                self.module_builder.emit_error(
                    signed_operand.span().start(),
                    "Signedness operand must be literal",
                );
                return;
            }
        };

        let width = literal_to_u32(width_literal);
        let signedness = literal_to_u32(signed_literal);
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_int_id(Some(result_id), width, signedness);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_float(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeFloat requires a result identifier");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(width_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeFloat missing width literal");
            return;
        };
        let width_literal = match width_operand.value() {
            OperandValue::Literal(lit) => lit,
            _ => {
                self.module_builder.emit_error(
                    width_operand.span().start(),
                    "Width operand must be literal",
                );
                return;
            }
        };
        let encoding = match operands.next() {
            Some(operand) => match self.parse_fp_encoding_operand(operand) {
                Some(value) => Some(value),
                None => return,
            },
            None => None,
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeFloat received unexpected operands",
            );
            return;
        }
        let width = literal_to_u32(width_literal);
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_float_id(Some(result_id), width, encoding);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_constant(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstant missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstant requires a result identifier");
            return;
        };

        let literal_operand = match instruction.operands().first() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpConstant missing literal operand");
                return;
            }
        };

        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);

        // Look up the result type to determine how to encode the literal.
        // The operand is a LiteralContextDependentNumber: its encoding depends
        // on the result type, matching the C++ spirv-as behavior.
        let float_width = self.lookup_float_type_width(type_id);

        let operands = match literal_operand.value() {
            OperandValue::Literal(literal) => {
                // Integer text (e.g. "42", "-1"). For float types, convert the
                // integer value to float then encode as IEEE 754 bit pattern.
                match float_width {
                    Some(32) => {
                        let float_val = match literal {
                            LiteralNumber::Unsigned(v) => *v as f32,
                            LiteralNumber::Signed(v) => *v as f32,
                        };
                        vec![dr::Operand::LiteralBit32(float_val.to_bits())]
                    }
                    Some(64) => {
                        let float_val = match literal {
                            LiteralNumber::Unsigned(v) => *v as f64,
                            LiteralNumber::Signed(v) => *v as f64,
                        };
                        vec![dr::Operand::LiteralBit64(float_val.to_bits())]
                    }
                    Some(_) => {
                        // 16-bit or other widths: encode as raw bits in a 32-bit word
                        vec![dr::Operand::LiteralBit32(literal_to_u32(literal))]
                    }
                    None => {
                        // Integer type: encode as raw integer bits.
                        self.module_builder
                            .note_integer_constant(result_id, literal_to_u64(literal));
                        vec![encode_literal_operand(literal)]
                    }
                }
            }
            OperandValue::Word(word) => {
                // Non-integer text (e.g. "42.5", "0x1.8p+1"). Must be a float type.
                let text = word.as_str();
                if float_width.is_none() {
                    self.module_builder.emit_error(
                        literal_operand.span().start(),
                        "integer type requires an integer literal",
                    );
                    return;
                }
                match float_width {
                    Some(32) => match text.parse::<f32>() {
                        Ok(val) => vec![dr::Operand::LiteralBit32(val.to_bits())],
                        Err(_) => {
                            self.module_builder.emit_error(
                                literal_operand.span().start(),
                                "invalid 32-bit float literal",
                            );
                            return;
                        }
                    },
                    Some(64) => match text.parse::<f64>() {
                        Ok(val) => vec![dr::Operand::LiteralBit64(val.to_bits())],
                        Err(_) => {
                            self.module_builder.emit_error(
                                literal_operand.span().start(),
                                "invalid 64-bit float literal",
                            );
                            return;
                        }
                    },
                    _ => {
                        self.module_builder.emit_error(
                            literal_operand.span().start(),
                            "unsupported float width for literal",
                        );
                        return;
                    }
                }
            }
            _ => {
                self.module_builder.emit_error(
                    literal_operand.span().start(),
                    "OpConstant literal operand must be numeric",
                );
                return;
            }
        };

        let inst = dr::Instruction::new(
            spirv::Op::Constant,
            Some(type_id),
            Some(result_id),
            operands,
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    /// Look up whether `type_id` refers to an `OpTypeFloat` instruction
    /// and return its bit width if so.
    fn lookup_float_type_width(&self, type_id: u32) -> Option<u32> {
        self.builder
            .module_ref()
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == spirv::Op::TypeFloat && inst.result_id == Some(type_id)
            })
            .and_then(|inst| inst.operands.first())
            .and_then(|op| match op {
                dr::Operand::LiteralBit32(w) => Some(*w),
                _ => None,
            })
    }

    fn translate_type_function(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeFunction missing result id");
            return;
        };

        let mut operands = instruction.operands().iter();
        let return_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpTypeFunction missing return type");
                return;
            }
        };
        let return_type = match return_operand.value() {
            OperandValue::Id(id) => self.module_builder.resolve_id_ref(*id),
            _ => {
                self.module_builder.emit_error(
                    return_operand.span().start(),
                    "Return type must be an id reference",
                );
                return;
            }
        };

        let mut parameter_types = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Id(id) => {
                    parameter_types.push(self.module_builder.resolve_id_ref(*id))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Parameter type must be an id reference",
                    );
                    return;
                }
            }
        }

        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder
            .type_function_id(Some(result_id), return_type, parameter_types);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_function(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpFunction missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpFunction missing result id");
            return;
        };

        let mut operands = instruction.operands().iter();
        let control_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpFunction missing control operand");
                return;
            }
        };
        let control = match self.parse_function_control(control_operand) {
            Some(value) => value,
            None => return,
        };
        let function_type_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpFunction missing function type");
                return;
            }
        };
        let function_type = match function_type_operand.value() {
            OperandValue::Id(id) => self.module_builder.resolve_id_ref(*id),
            _ => {
                self.module_builder.emit_error(
                    function_type_operand.span().start(),
                    "Function type must be an id reference",
                );
                return;
            }
        };

        let (result_type, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        match self
            .builder
            .begin_function(result_type, Some(result_id), control, function_type)
        {
            Ok(_) => self.record_function_def(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_function_parameter(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpFunctionParameter missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpFunctionParameter missing result id");
            return;
        };

        let Some(selected_function) = self.builder.selected_function() else {
            self.emit_builder_error(BuildError::DetachedFunctionParameter, opcode_pos);
            return;
        };

        // Use bind_typed_result to allocate IDs from the module_builder's counter,
        // rather than Builder::function_parameter() which uses Builder's separate
        // next_id counter and can produce duplicate IDs.
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            spirv::Op::FunctionParameter,
            Some(type_id),
            Some(result_id),
            vec![],
        );
        self.builder.module_mut().functions[selected_function]
            .parameters
            .push(inst);
        self.record_function_param();
    }

    fn translate_label(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLabel missing result id");
            return;
        };
        let label_id = self.module_builder.resolve_result_id(result_id);
        match self.builder.begin_block(Some(label_id)) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_return(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        match self.builder.ret() {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_return_value(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let operand = match instruction.operands().first() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpReturnValue missing result id");
                return;
            }
        };
        let Some(value_id) = self.operand_as_id(operand, "return value") else {
            return;
        };
        match self.builder.ret_value(value_id) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, operand.span().start()),
        }
    }

    fn translate_function_end(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let current_function = self.builder.selected_function();
        match self.builder.end_function() {
            Ok(_) => {
                if let Some(function) = current_function {
                    self.record_from_module(|module| {
                        module
                            .functions
                            .get(function)
                            .and_then(|f| f.end.as_ref())
                            .cloned()
                    });
                }
            }
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_type_pointer(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypePointer missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(storage_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypePointer missing storage class");
            return;
        };
        let Some(storage_class) = self.parse_enum_operand::<spirv::StorageClass>(
            Some(storage_operand),
            "storage class",
            opcode_pos,
        ) else {
            return;
        };
        let Some(pointee_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypePointer missing pointee type");
            return;
        };
        let Some(pointee_id) = self.operand_as_id(pointee_operand, "pointee type") else {
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypePointer received unexpected operands",
            );
            return;
        }
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder
            .type_pointer(Some(result_id), storage_class, pointee_id);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_vector(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeVector missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(component_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeVector missing component type");
            return;
        };
        let Some(component_type) = self.operand_as_id(component_operand, "component type") else {
            return;
        };
        let Some(count_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeVector missing component count");
            return;
        };
        let count_literal = match count_operand.value() {
            OperandValue::Literal(literal) => literal,
            _ => {
                self.module_builder.emit_error(
                    count_operand.span().start(),
                    "Component count must be a literal",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeVector received unexpected operands",
            );
            return;
        }
        let component_count = literal_to_u32(count_literal);
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder
            .type_vector_id(Some(result_id), component_type, component_count);
        self.module_builder.note_vector_type(
            result_id,
            VectorTypeInfo::new(component_type, component_count),
        );
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_array(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeArray missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(element_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeArray missing element type");
            return;
        };
        let Some(element_type) = self.operand_as_id(element_operand, "element type") else {
            return;
        };
        let Some(length_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeArray missing length id");
            return;
        };
        let Some(length_id) = self.operand_as_id(length_operand, "length id") else {
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeArray received unexpected operands",
            );
            return;
        }
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst = dr::Instruction::new(
            spirv::Op::TypeArray,
            None,
            Some(result_id),
            vec![
                dr::Operand::IdRef(element_type),
                dr::Operand::IdRef(length_id),
            ],
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.module_builder
            .note_array_type(result_id, ArrayTypeInfo::new(element_type, length_id));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_runtime_array(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeRuntimeArray missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(element_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeRuntimeArray missing element type");
            return;
        };
        let Some(element_type) = self.operand_as_id(element_operand, "element type") else {
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeRuntimeArray received unexpected operands",
            );
            return;
        }
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst = dr::Instruction::new(
            spirv::Op::TypeRuntimeArray,
            None,
            Some(result_id),
            vec![dr::Operand::IdRef(element_type)],
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_struct(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeStruct missing result id");
            return;
        };
        let result_id = self.module_builder.resolve_result_id(result_id);
        let mut field_types = Vec::new();
        for operand in instruction.operands() {
            match operand.value() {
                OperandValue::Id(id) => field_types.push(self.module_builder.resolve_id_ref(*id)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Struct members must be id references",
                    );
                    return;
                }
            }
        }
        let operands = field_types
            .iter()
            .copied()
            .map(dr::Operand::IdRef)
            .collect();
        let inst = dr::Instruction::new(spirv::Op::TypeStruct, None, Some(result_id), operands);
        self.builder.module_mut().types_global_values.push(inst);
        self.module_builder
            .note_struct_type(result_id, StructTypeInfo::new(field_types));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_matrix(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeMatrix missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(column_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeMatrix missing column type");
            return;
        };
        let Some(column_type) = self.operand_as_id(column_operand, "column type") else {
            return;
        };
        if self.module_builder.vector_type(column_type).is_none() {
            self.module_builder.emit_error(
                column_operand.span().start(),
                "Matrix column type must be a previously defined vector",
            );
            return;
        }
        let Some(count_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeMatrix missing column count");
            return;
        };
        let column_count_literal = match count_operand.value() {
            OperandValue::Literal(literal) => literal,
            _ => {
                self.module_builder.emit_error(
                    count_operand.span().start(),
                    "Matrix column count must be a literal",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeMatrix received unexpected operands",
            );
            return;
        }
        let column_count = literal_to_u32(column_count_literal);
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst = dr::Instruction::new(
            spirv::Op::TypeMatrix,
            None,
            Some(result_id),
            vec![
                dr::Operand::IdRef(column_type),
                dr::Operand::LiteralBit32(column_count),
            ],
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.module_builder
            .note_matrix_type(result_id, MatrixTypeInfo::new(column_type, column_count));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_image(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeImage missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();

        // 1. Sampled Type (IdRef)
        let Some(sampled_type_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeImage missing sampled type");
            return;
        };
        let Some(sampled_type) =
            self.operand_as_id(sampled_type_operand, "sampled type")
        else {
            return;
        };

        // 2. Dim (Enum) — SPIR-V assembly uses short names (1D, 2D, Cube, etc.)
        let Some(dim_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeImage missing Dim");
            return;
        };
        let dim = match dim_operand.value() {
            OperandValue::Word(word) => match word.as_str() {
                "1D" | "Dim1D" => spirv::Dim::Dim1D,
                "2D" | "Dim2D" => spirv::Dim::Dim2D,
                "3D" | "Dim3D" => spirv::Dim::Dim3D,
                "Cube" | "DimCube" => spirv::Dim::DimCube,
                "Rect" | "DimRect" => spirv::Dim::DimRect,
                "Buffer" | "DimBuffer" => spirv::Dim::DimBuffer,
                "SubpassData" | "DimSubpassData" => spirv::Dim::DimSubpassData,
                "TileImageDataEXT" | "DimTileImageDataEXT" => {
                    spirv::Dim::DimTileImageDataEXT
                }
                _ => {
                    self.module_builder
                        .emit_error(dim_operand.span().start(), "Invalid Dim");
                    return;
                }
            },
            _ => {
                self.module_builder
                    .emit_error(dim_operand.span().start(), "Dim must be an enumerant");
                return;
            }
        };

        // 3. Depth (Literal)
        let Some(depth_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeImage missing Depth");
            return;
        };
        let depth = match depth_operand.value() {
            OperandValue::Literal(lit) => literal_to_u32(lit),
            _ => {
                self.module_builder
                    .emit_error(depth_operand.span().start(), "Depth must be a literal");
                return;
            }
        };

        // 4. Arrayed (Literal)
        let Some(arrayed_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeImage missing Arrayed");
            return;
        };
        let arrayed = match arrayed_operand.value() {
            OperandValue::Literal(lit) => literal_to_u32(lit),
            _ => {
                self.module_builder.emit_error(
                    arrayed_operand.span().start(),
                    "Arrayed must be a literal",
                );
                return;
            }
        };

        // 5. MS (Literal)
        let Some(ms_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeImage missing MS");
            return;
        };
        let ms = match ms_operand.value() {
            OperandValue::Literal(lit) => literal_to_u32(lit),
            _ => {
                self.module_builder
                    .emit_error(ms_operand.span().start(), "MS must be a literal");
                return;
            }
        };

        // 6. Sampled (Literal)
        let Some(sampled_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeImage missing Sampled");
            return;
        };
        let sampled = match sampled_operand.value() {
            OperandValue::Literal(lit) => literal_to_u32(lit),
            _ => {
                self.module_builder.emit_error(
                    sampled_operand.span().start(),
                    "Sampled must be a literal",
                );
                return;
            }
        };

        // 7. Image Format (Enum)
        let Some(format_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeImage missing Image Format");
            return;
        };
        let Some(format) = self.parse_enum_operand::<spirv::ImageFormat>(
            Some(format_operand),
            "Image Format",
            opcode_pos,
        ) else {
            return;
        };

        // 8. Optional: Access Qualifier (Enum)
        let access_qualifier = match operands.next() {
            Some(operand) => {
                match self.parse_enum_operand::<spirv::AccessQualifier>(
                    Some(operand),
                    "Access Qualifier",
                    opcode_pos,
                ) {
                    Some(aq) => Some(aq),
                    None => return,
                }
            }
            None => None,
        };

        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeImage received unexpected operands",
            );
            return;
        }

        let result_id = self.module_builder.resolve_result_id(result_id);
        let mut inst_operands = vec![
            dr::Operand::IdRef(sampled_type),
            dr::Operand::from(dim),
            dr::Operand::LiteralBit32(depth),
            dr::Operand::LiteralBit32(arrayed),
            dr::Operand::LiteralBit32(ms),
            dr::Operand::LiteralBit32(sampled),
            dr::Operand::from(format),
        ];
        if let Some(aq) = access_qualifier {
            inst_operands.push(dr::Operand::from(aq));
        }

        let inst = dr::Instruction::new(
            spirv::Op::TypeImage,
            None,
            Some(result_id),
            inst_operands,
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_sampled_image(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeSampledImage missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(image_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeSampledImage missing image type");
            return;
        };
        let Some(image_type) = self.operand_as_id(image_operand, "image type") else {
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeSampledImage received unexpected operands",
            );
            return;
        }
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst = dr::Instruction::new(
            spirv::Op::TypeSampledImage,
            None,
            Some(result_id),
            vec![dr::Operand::IdRef(image_type)],
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_sampler(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeSampler requires a result id");
            return;
        };
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_sampler_id(Some(result_id));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_memory_model(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let addressing = match self.parse_enum_operand::<spirv::AddressingModel>(
            operands.next(),
            "addressing model",
            opcode_pos,
        ) {
            Some(value) => value,
            None => return,
        };
        let memory = match self.parse_enum_operand::<spirv::MemoryModel>(
            operands.next(),
            "memory model",
            opcode_pos,
        ) {
            Some(value) => value,
            None => return,
        };
        self.builder.memory_model(addressing, memory);
        self.record_from_module(|module| module.memory_model.clone());
    }

    fn translate_entry_point(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let execution_model = match self.parse_enum_operand::<spirv::ExecutionModel>(
            operands.next(),
            "execution model",
            opcode_pos,
        ) {
            Some(value) => value,
            None => return,
        };

        let function_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpEntryPoint missing function identifier");
                return;
            }
        };
        let function_id = match function_operand.value() {
            OperandValue::Id(id) => self.module_builder.resolve_id_ref(*id),
            _ => {
                self.module_builder.emit_error(
                    function_operand.span().start(),
                    "Function operand must be an id reference",
                );
                return;
            }
        };

        let name_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpEntryPoint missing name literal");
                return;
            }
        };
        let entry_name = match name_operand.value() {
            OperandValue::String(name) => parse_string_literal(name),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "Entry point name must be a literal string",
                );
                return;
            }
        };

        let mut interfaces = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Id(id) => interfaces.push(self.module_builder.resolve_id_ref(*id)),
                _ => self.module_builder.emit_error(
                    operand.span().start(),
                    "Interface operand must be an id reference",
                ),
            }
        }

        self.builder
            .entry_point(execution_model, function_id, &entry_name, interfaces);
        self.record_from_module(|module| module.entry_points.last().cloned());
    }

    fn translate_name(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(target_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpName missing target id");
            return;
        };
        let Some(name_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpName missing name operand");
            return;
        };
        let Some(target_id) = self.operand_as_id(target_operand, "target id") else {
            return;
        };
        let name = match name_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "OpName operand must be a literal string",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder
                .emit_error(extra.span().start(), "OpName received unexpected operands");
            return;
        }
        self.builder.name(target_id, name);
        self.record_from_module(|module| module.debug_names.last().cloned());
    }

    fn translate_member_name(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(target_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberName missing target id");
            return;
        };
        let Some(member_index_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberName missing member index operand");
            return;
        };
        let Some(name_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberName missing name operand");
            return;
        };
        let Some(target_id) = self.operand_as_id(target_operand, "target id") else {
            return;
        };
        let member_index = match member_index_operand.value() {
            OperandValue::Literal(literal) => literal_to_u32(literal),
            _ => {
                self.module_builder.emit_error(
                    member_index_operand.span().start(),
                    "OpMemberName member index must be a literal number",
                );
                return;
            }
        };
        let name = match name_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "OpMemberName operand must be a literal string",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpMemberName received unexpected operands",
            );
            return;
        }
        self.builder.member_name(target_id, member_index, name);
        self.record_from_module(|module| module.debug_names.last().cloned());
    }

    fn translate_access_chain(&mut self, instruction: &ParsedInstruction<'a>, in_bounds: bool) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Access chain missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Access chain missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(base_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "Access chain missing base pointer");
            return;
        };
        let Some(base_id) = self.operand_as_id(base_operand, "base pointer") else {
            return;
        };
        let mut indexes = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Id(id) => {
                    indexes.push(self.module_builder.resolve_id_ref(*id));
                }
                _ => {
                    self.module_builder
                        .emit_error(operand.span().start(), "Access chain indexes must be ids");
                    return;
                }
            }
        }
        let (result_type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let result = if in_bounds {
            self.builder
                .in_bounds_access_chain(result_type_id, Some(result_id), base_id, indexes)
        } else {
            self.builder
                .access_chain(result_type_id, Some(result_id), base_id, indexes)
        };
        match result {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_execution_mode(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let entry_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpExecutionMode missing entry point");
                return;
            }
        };
        let Some(entry_point) = self.operand_as_id(entry_operand, "entry point") else {
            return;
        };
        let Some(execution_mode) = self.parse_enum_operand::<spirv::ExecutionMode>(
            operands.next(),
            "execution mode",
            opcode_pos,
        ) else {
            return;
        };
        let mut parameters = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => parameters.push(literal_to_u32(literal)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Execution mode parameters must be literals",
                    );
                    return;
                }
            }
        }
        self.builder
            .execution_mode(entry_point, execution_mode, parameters);
        self.record_from_module(|module| module.execution_modes.last().cloned());
    }

    fn translate_selection_merge(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(merge_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpSelectionMerge missing merge block");
            return;
        };
        let Some(merge_id) = self.operand_as_id(merge_operand, "merge block") else {
            return;
        };
        let Some(control_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpSelectionMerge missing control mask");
            return;
        };
        let Some(selection_control) = self.parse_selection_control(control_operand) else {
            return;
        };
        match self.builder.selection_merge(merge_id, selection_control) {
            Ok(()) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, merge_operand.span().start()),
        }
    }

    fn translate_loop_merge(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(merge_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoopMerge missing merge block");
            return;
        };
        let Some(continue_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoopMerge missing continue target");
            return;
        };
        let Some(control_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoopMerge missing control mask");
            return;
        };
        let Some(merge_id) = self.operand_as_id(merge_operand, "merge block") else {
            return;
        };
        let Some(continue_id) = self.operand_as_id(continue_operand, "continue target") else {
            return;
        };
        let Some((loop_control, control_operands)) =
            self.parse_loop_control_operand(control_operand, &mut operands)
        else {
            return;
        };
        match self
            .builder
            .loop_merge(merge_id, continue_id, loop_control, control_operands)
        {
            Ok(()) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, merge_operand.span().start()),
        }
    }

    fn translate_composite_construct(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeConstruct missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeConstruct missing result id");
            return;
        };
        if instruction.operands().is_empty() {
            self.module_builder.emit_error(
                opcode_pos,
                "OpCompositeConstruct requires at least one constituent",
            );
            return;
        }
        let mut constituents = Vec::new();
        for operand in instruction.operands() {
            match operand.value() {
                OperandValue::Id(id) => constituents.push(self.module_builder.resolve_id_ref(*id)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Constituent must be an id reference",
                    );
                    return;
                }
            }
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        match self
            .builder
            .composite_construct(type_id, Some(result_id), constituents)
        {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_vector_shuffle(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(vector1_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle missing first vector");
            return;
        };
        let Some(vector2_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle missing second vector");
            return;
        };
        let Some(vector1_id) = self.operand_as_id(vector1_operand, "first vector") else {
            return;
        };
        let Some(vector2_id) = self.operand_as_id(vector2_operand, "second vector") else {
            return;
        };
        let mut components = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => {
                    components.push((literal_to_u32(literal), operand.span().start()))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Shuffle components must be literals",
                    );
                    return;
                }
            }
        }
        let component_position = components
            .first()
            .map(|(_, span)| *span)
            .unwrap_or_else(|| vector2_operand.span().start());
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let Some(result_vector) = self.module_builder.vector_type(type_id) else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle result type must be a vector");
            return;
        };
        let expected_components = result_vector.component_count() as usize;
        if components.len() != expected_components {
            self.module_builder.emit_error(
                component_position,
                format!(
                    "OpVectorShuffle expects {expected_components} component literals but received {}",
                    components.len()
                ),
            );
            return;
        }

        let Some(vector1_type_id) = self.module_builder.value_type(vector1_id) else {
            self.module_builder.emit_error(
                vector1_operand.span().start(),
                "First vector operand has no known type",
            );
            return;
        };
        let Some(vector1_info) = self.module_builder.vector_type(vector1_type_id) else {
            self.module_builder.emit_error(
                vector1_operand.span().start(),
                "First vector operand must be a vector",
            );
            return;
        };
        let Some(vector2_type_id) = self.module_builder.value_type(vector2_id) else {
            self.module_builder.emit_error(
                vector2_operand.span().start(),
                "Second vector operand has no known type",
            );
            return;
        };
        let Some(vector2_info) = self.module_builder.vector_type(vector2_type_id) else {
            self.module_builder.emit_error(
                vector2_operand.span().start(),
                "Second vector operand must be a vector",
            );
            return;
        };
        if vector1_info.component_type() != result_vector.component_type() {
            self.module_builder.emit_error(
                vector1_operand.span().start(),
                "First vector operand type does not match the result vector component type",
            );
            return;
        }
        if vector2_info.component_type() != result_vector.component_type() {
            self.module_builder.emit_error(
                vector2_operand.span().start(),
                "Second vector operand type does not match the result vector component type",
            );
            return;
        }

        let total_components =
            u64::from(vector1_info.component_count()) + u64::from(vector2_info.component_count());
        for (value, position) in &components {
            if *value != u32::MAX && u64::from(*value) >= total_components {
                self.module_builder.emit_error(
                    *position,
                    format!(
                        "Shuffle component {value} exceeds the available inputs ({total_components})",
                    ),
                );
                return;
            }
        }

        let literal_components: Vec<u32> = components.iter().map(|(value, _)| *value).collect();
        if let Err(error) = self.builder.vector_shuffle(
            type_id,
            Some(result_id),
            vector1_id,
            vector2_id,
            literal_components,
        ) {
            self.emit_builder_error(error, opcode_pos);
        } else {
            self.record_from_current_block();
        }
    }

    fn translate_boolean_constant(&mut self, instruction: &ParsedInstruction<'a>, value: bool) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Boolean constant missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Boolean constant missing result id");
            return;
        };
        if !instruction.operands().is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "Boolean constants do not take operands");
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let opcode = if value {
            spirv::Op::ConstantTrue
        } else {
            spirv::Op::ConstantFalse
        };
        let inst = dr::Instruction::new(opcode, Some(type_id), Some(result_id), vec![]);
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_constant_composite(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantComposite missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantComposite missing result id");
            return;
        };
        if instruction.operands().is_empty() {
            self.module_builder.emit_error(
                opcode_pos,
                "OpConstantComposite requires at least one constituent",
            );
            return;
        }
        let mut constituents = Vec::new();
        for operand in instruction.operands() {
            match operand.value() {
                OperandValue::Id(id) => constituents.push(self.module_builder.resolve_id_ref(*id)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Constituent must be an id reference",
                    );
                    return;
                }
            }
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let operands = constituents.into_iter().map(dr::Operand::IdRef).collect();
        let inst = dr::Instruction::new(
            spirv::Op::ConstantComposite,
            Some(type_id),
            Some(result_id),
            operands,
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_constant_null(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantNull missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantNull missing result id");
            return;
        };
        if !instruction.operands().is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantNull does not take operands");
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            spirv::Op::ConstantNull,
            Some(type_id),
            Some(result_id),
            vec![],
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_composite_extract(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeExtract missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeExtract missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(composite_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeExtract missing composite value");
            return;
        };
        let Some(composite_id) = self.operand_as_id(composite_operand, "composite value") else {
            return;
        };
        let mut indexes = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => {
                    indexes.push((literal_to_u32(literal), operand.span().start()))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Composite extract indexes must be literals",
                    );
                    return;
                }
            }
        }
        if indexes.is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeExtract requires at least one index");
            return;
        }
        let mismatch_position = indexes
            .first()
            .map(|(_, span)| *span)
            .unwrap_or_else(|| composite_operand.span().start());
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let Some(composite_type_id) = self.module_builder.value_type(composite_id) else {
            self.module_builder.emit_error(
                composite_operand.span().start(),
                "Composite operand has no known type",
            );
            return;
        };
        let Some(expected_type) = self.resolve_composite_access(composite_type_id, &indexes) else {
            return;
        };
        if expected_type != type_id {
            self.module_builder.emit_error(
                mismatch_position,
                "Result type does not match the selected component type",
            );
            return;
        }
        let literal_indexes: Vec<u32> = indexes.iter().map(|(value, _)| *value).collect();
        match self.builder.composite_extract(
            type_id,
            Some(result_id),
            composite_id,
            literal_indexes,
        ) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_composite_insert(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(object_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert missing object operand");
            return;
        };
        let Some(object_id) = self.operand_as_id(object_operand, "object operand") else {
            return;
        };
        let Some(composite_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert missing composite operand");
            return;
        };
        let Some(composite_id) = self.operand_as_id(composite_operand, "composite operand") else {
            return;
        };
        let mut indexes = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => {
                    indexes.push((literal_to_u32(literal), operand.span().start()))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Composite insert indexes must be literals",
                    );
                    return;
                }
            }
        }
        if indexes.is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert requires at least one index");
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let Some(composite_type_id) = self.module_builder.value_type(composite_id) else {
            self.module_builder.emit_error(
                composite_operand.span().start(),
                "Composite operand has no known type",
            );
            return;
        };
        if composite_type_id != type_id {
            self.module_builder.emit_error(
                composite_operand.span().start(),
                "Result type must match the composite operand type",
            );
            return;
        }
        let Some(target_type) = self.resolve_composite_access(composite_type_id, &indexes) else {
            return;
        };
        let Some(object_type_id) = self.module_builder.value_type(object_id) else {
            self.module_builder.emit_error(
                object_operand.span().start(),
                "Object operand has no known type",
            );
            return;
        };
        if object_type_id != target_type {
            self.module_builder.emit_error(
                object_operand.span().start(),
                "Object operand type must match the selected component type",
            );
            return;
        }
        let literal_indexes: Vec<u32> = indexes.iter().map(|(value, _)| *value).collect();
        match self.builder.composite_insert(
            type_id,
            Some(result_id),
            object_id,
            composite_id,
            literal_indexes,
        ) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_ext_inst(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInst missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInst missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(set_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInst missing instruction set");
            return;
        };
        let Some(set_id) = self.operand_as_id(set_operand, "instruction set") else {
            return;
        };
        let Some(import_info) = self.module_builder.ext_inst_import(set_id).cloned() else {
            self.module_builder.emit_error(
                set_operand.span().start(),
                "OpExtInst references an unknown instruction set",
            );
            return;
        };
        let Some(opcode_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInst missing opcode operand");
            return;
        };
        let Some(resolved) = self.resolve_ext_inst_opcode(&import_info, opcode_operand) else {
            return;
        };
        let mut encoded_operands = vec![
            dr::Operand::IdRef(set_id),
            dr::Operand::LiteralExtInstInteger(resolved.opcode),
        ];
        if let Some(descriptors) = resolved.operands {
            for descriptor in descriptors {
                match descriptor.quantifier {
                    OperandQuantifier::One => {
                        let Some(next_operand) = operands.next() else {
                            self.module_builder
                                .emit_error(opcode_pos, "Extended instruction missing operands");
                            return;
                        };
                        let Some(encoded) =
                            self.encode_ext_inst_operand(descriptor, next_operand, opcode_pos)
                        else {
                            return;
                        };
                        encoded_operands.push(encoded);
                    }
                    OperandQuantifier::ZeroOrOne => {
                        if operands.as_slice().is_empty() {
                            continue;
                        }
                        let next_operand = operands.next().expect("peeked operand");
                        let Some(encoded) =
                            self.encode_ext_inst_operand(descriptor, next_operand, opcode_pos)
                        else {
                            return;
                        };
                        encoded_operands.push(encoded);
                    }
                    OperandQuantifier::ZeroOrMore => {
                        for next_operand in operands.by_ref() {
                            let Some(encoded) =
                                self.encode_ext_inst_operand(descriptor, next_operand, opcode_pos)
                            else {
                                return;
                            };
                            encoded_operands.push(encoded);
                        }
                    }
                }
            }
            if let Some(extra) = operands.next() {
                self.module_builder.emit_error(
                    extra.span().start(),
                    "Extended instruction received unexpected operands",
                );
                return;
            }
        } else {
            for operand in operands {
                let Some(encoded) = self.encode_generic_ext_inst_operand(operand) else {
                    return;
                };
                encoded_operands.push(encoded);
            }
        }

        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            spirv::Op::ExtInst,
            Some(type_id),
            Some(result_id),
            encoded_operands,
        );
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_decorate(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(target_operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpDecorate missing target id");
            return;
        };
        let Some(decoration_operand) = instruction.operands().get(1) else {
            self.module_builder
                .emit_error(opcode_pos, "OpDecorate missing decoration enumerant");
            return;
        };
        let Some(target_id) = self.operand_as_id(target_operand, "decorated id") else {
            return;
        };
        let Some(decoration) = self.parse_enum_operand::<spirv::Decoration>(
            Some(decoration_operand),
            "decoration",
            opcode_pos,
        ) else {
            return;
        };
        let mut operands = vec![
            dr::Operand::IdRef(target_id),
            dr::Operand::Decoration(decoration),
        ];
        if let Some(extra) =
            self.encode_decoration_operands(decoration, instruction.operands(), 2, opcode_pos)
        {
            operands.extend(extra);
        } else {
            return;
        }
        self.push_annotation_instruction(dr::Instruction::new(
            spirv::Op::Decorate,
            None,
            None,
            operands,
        ));
    }

    fn translate_decorate_id(&mut self, instruction: &ParsedInstruction<'a>) {
        self.translate_decorate(instruction);
    }

    fn translate_decorate_string(&mut self, instruction: &ParsedInstruction<'a>) {
        self.translate_decorate(instruction);
    }

    fn translate_member_decorate(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(struct_operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberDecorate missing struct id");
            return;
        };
        let Some(member_operand) = instruction.operands().get(1) else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberDecorate missing member index");
            return;
        };
        let Some(decoration_operand) = instruction.operands().get(2) else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberDecorate missing decoration enumerant");
            return;
        };
        let Some(struct_id) = self.operand_as_id(struct_operand, "struct id") else {
            return;
        };
        let Some(member_index_value) = (match member_operand.value() {
            OperandValue::Literal(literal) => Some(literal_to_u32(literal)),
            _ => {
                self.module_builder.emit_error(
                    member_operand.span().start(),
                    "Member index must be a literal",
                );
                None
            }
        }) else {
            return;
        };
        let Ok(member_index) = usize::try_from(member_index_value) else {
            self.module_builder
                .emit_error(member_operand.span().start(), "Member index is too large");
            return;
        };
        let Some(decoration) = self.parse_enum_operand::<spirv::Decoration>(
            Some(decoration_operand),
            "decoration",
            opcode_pos,
        ) else {
            return;
        };
        let mut operands = vec![
            dr::Operand::IdRef(struct_id),
            dr::Operand::LiteralBit32(member_index_value),
            dr::Operand::Decoration(decoration),
        ];
        if let Some(extra) =
            self.encode_decoration_operands(decoration, instruction.operands(), 3, opcode_pos)
        {
            operands.extend(extra);
        } else {
            return;
        }
        self.apply_member_decorate_metadata(struct_id, member_index, decoration, instruction);
        self.push_annotation_instruction(dr::Instruction::new(
            spirv::Op::MemberDecorate,
            None,
            None,
            operands,
        ));
    }

    fn translate_member_decorate_string(&mut self, instruction: &ParsedInstruction<'a>) {
        self.translate_member_decorate(instruction);
    }

    fn apply_member_decorate_metadata(
        &mut self,
        struct_id: u32,
        member_index: usize,
        decoration: spirv::Decoration,
        instruction: &ParsedInstruction<'a>,
    ) {
        match decoration {
            spirv::Decoration::RowMajor => {
                let position = instruction
                    .operands()
                    .get(2)
                    .map(|operand| operand.span().start())
                    .unwrap_or_default();
                self.module_builder.apply_member_majorness(
                    struct_id,
                    member_index,
                    MatrixMajorness::RowMajor,
                    position,
                );
            }
            spirv::Decoration::ColMajor => {
                let position = instruction
                    .operands()
                    .get(2)
                    .map(|operand| operand.span().start())
                    .unwrap_or_default();
                self.module_builder.apply_member_majorness(
                    struct_id,
                    member_index,
                    MatrixMajorness::ColumnMajor,
                    position,
                );
            }
            spirv::Decoration::MatrixStride => {
                let Some(stride_operand) = instruction.operands().get(3) else {
                    return;
                };
                let stride = match stride_operand.value() {
                    OperandValue::Literal(literal) => literal_to_u32(literal),
                    _ => {
                        self.module_builder.emit_error(
                            stride_operand.span().start(),
                            "MatrixStride requires an integer literal",
                        );
                        return;
                    }
                };
                self.module_builder.apply_member_matrix_stride(
                    struct_id,
                    member_index,
                    stride,
                    stride_operand.span().start(),
                );
            }
            _ => {}
        }
    }

    fn translate_phi(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpPhi missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpPhi missing result id");
            return;
        };
        let mut incoming = Vec::new();
        for operand in instruction.operands() {
            match operand.value() {
                OperandValue::IdPair(value, parent) => {
                    let value_id = self.module_builder.resolve_id_ref(*value);
                    let parent_id = self.module_builder.resolve_id_ref(*parent);
                    incoming.push((value_id, parent_id));
                }
                _ => {
                    self.module_builder
                        .emit_error(operand.span().start(), "OpPhi operands must be id pairs");
                    return;
                }
            }
        }
        if incoming.is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpPhi requires incoming edges");
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        match self.builder.phi(type_id, Some(result_id), incoming) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_copy_memory(&mut self, instruction: &ParsedInstruction<'a>, sized: bool) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(target_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCopyMemory missing target");
            return;
        };
        let Some(source_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCopyMemory missing source");
            return;
        };
        let Some(target_id) = self.operand_as_id(target_operand, "target pointer") else {
            return;
        };
        let Some(source_id) = self.operand_as_id(source_operand, "source pointer") else {
            return;
        };
        let mut dr_operands = vec![dr::Operand::IdRef(target_id), dr::Operand::IdRef(source_id)];
        if sized {
            let Some(size_operand) = operands.next() else {
                self.module_builder
                    .emit_error(opcode_pos, "OpCopyMemorySized missing size operand");
                return;
            };
            let Some(size_id) = self.operand_as_id(size_operand, "size operand") else {
                return;
            };
            dr_operands.push(dr::Operand::IdRef(size_id));
        }
        self.take_memory_access_operand(&mut operands, &mut dr_operands);
        self.take_memory_access_operand(&mut operands, &mut dr_operands);
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Unexpected operands after memory access masks",
            );
            return;
        }
        let opcode = if sized {
            spirv::Op::CopyMemorySized
        } else {
            spirv::Op::CopyMemory
        };
        let inst = dr::Instruction::new(opcode, None, None, dr_operands);
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_branch(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let operand = match instruction.operands().first() {
            Some(op) => op,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpBranch missing target");
                return;
            }
        };
        let Some(target) = self.operand_as_id(operand, "branch target") else {
            return;
        };
        match self.builder.branch(target) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, operand.span().start()),
        }
    }

    fn translate_branch_conditional(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(condition_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpBranchConditional missing condition");
            return;
        };
        let Some(true_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpBranchConditional missing true label");
            return;
        };
        let Some(false_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpBranchConditional missing false label");
            return;
        };
        let Some(condition_id) = self.operand_as_id(condition_operand, "condition id") else {
            return;
        };
        let Some(true_label) = self.operand_as_id(true_operand, "true label") else {
            return;
        };
        let Some(false_label) = self.operand_as_id(false_operand, "false label") else {
            return;
        };
        let mut branch_weights = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => branch_weights.push(literal_to_u32(literal)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Branch weights must be literal integers",
                    );
                    return;
                }
            }
        }
        match self
            .builder
            .branch_conditional(condition_id, true_label, false_label, branch_weights)
        {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_variable(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVariable missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVariable missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(storage_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVariable missing storage class");
            return;
        };
        let Some(storage_class) = self.parse_enum_operand::<spirv::StorageClass>(
            Some(storage_operand),
            "storage class",
            opcode_pos,
        ) else {
            return;
        };
        let initializer = match operands.next() {
            Some(operand) => match self.operand_as_id(operand, "initializer") {
                Some(id) => Some(id),
                None => return,
            },
            None => None,
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpVariable received unexpected operands",
            );
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let initializer_id = initializer;
        self.builder
            .variable(type_id, Some(result_id), storage_class, initializer_id);
        match (
            self.builder.selected_function(),
            self.builder.selected_block(),
        ) {
            (Some(_), Some(_)) => self.record_from_current_block(),
            _ => self.record_from_module(|module| module.types_global_values.last().cloned()),
        }
    }

    fn translate_load(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoad missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoad missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(pointer_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoad missing pointer operand");
            return;
        };
        let Some(pointer_id) = self.operand_as_id(pointer_operand, "pointer operand") else {
            return;
        };
        let mut dr_operands = vec![dr::Operand::IdRef(pointer_id)];
        self.take_memory_access_operand(&mut operands, &mut dr_operands);
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Unexpected operands after memory access mask",
            );
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst =
            dr::Instruction::new(spirv::Op::Load, Some(type_id), Some(result_id), dr_operands);
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_store(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(pointer_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpStore missing pointer operand");
            return;
        };
        let Some(object_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpStore missing object operand");
            return;
        };
        let Some(pointer_id) = self.operand_as_id(pointer_operand, "pointer operand") else {
            return;
        };
        let Some(object_id) = self.operand_as_id(object_operand, "object operand") else {
            return;
        };
        let mut dr_operands = vec![
            dr::Operand::IdRef(pointer_id),
            dr::Operand::IdRef(object_id),
        ];
        self.take_memory_access_operand(&mut operands, &mut dr_operands);
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Unexpected operands after memory access mask",
            );
            return;
        }
        let inst = dr::Instruction::new(spirv::Op::Store, None, None, dr_operands);
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_binary_arithmetic(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Binary operation missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Binary operation missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(lhs_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "Binary operation missing operands");
            return;
        };
        let Some(rhs_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "Binary operation requires two operands");
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Binary operation received unexpected operands",
            );
            return;
        }
        let Some(lhs_id) = self.operand_as_id(lhs_operand, "left operand") else {
            return;
        };
        let Some(rhs_id) = self.operand_as_id(rhs_operand, "right operand") else {
            return;
        };
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            instruction.opcode(),
            Some(type_id),
            Some(result_id),
            vec![dr::Operand::IdRef(lhs_id), dr::Operand::IdRef(rhs_id)],
        );
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_unary_op(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Unary operation missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Unary operation missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "Unary operation missing operand");
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Unary operation received unexpected operands",
            );
            return;
        }
        let Some(operand_id) = self.operand_as_id(operand, "operand") else {
            return;
        };
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            instruction.opcode(),
            Some(type_id),
            Some(result_id),
            vec![dr::Operand::IdRef(operand_id)],
        );
        self.push_block_instruction(inst, opcode_pos);
    }

    fn operand_as_id(&mut self, operand: &ParsedOperand<'a>, label: &str) -> Option<u32> {
        match operand.value() {
            OperandValue::Id(id) => Some(self.module_builder.resolve_id_ref(*id)),
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!("{label} must be an id reference"),
                );
                None
            }
        }
    }

    fn push_block_instruction(&mut self, instruction: dr::Instruction, position: MessagePosition) {
        match self
            .builder
            .insert_into_block(InsertPoint::End, instruction)
        {
            Ok(()) => self.record_from_current_block(),
            Err(BuildError::DetachedInstruction(Some(inst))) => {
                self.builder.module_mut().types_global_values.push(inst);
                self.record_from_module(|module| module.types_global_values.last().cloned());
            }
            Err(error) => self.emit_builder_error(error, position),
        }
    }

    fn push_annotation_instruction(&mut self, instruction: dr::Instruction) {
        self.builder.module_mut().annotations.push(instruction);
        self.record_from_module(|module| module.annotations.last().cloned());
    }

    fn take_memory_access_operand(
        &mut self,
        operands: &mut std::slice::Iter<'_, ParsedOperand<'a>>,
        target: &mut Vec<dr::Operand>,
    ) {
        if let Some(next) = operands.as_slice().first() {
            if next.descriptor().kind() == OperandKind::MemoryAccess {
                let operand = operands.next().expect("peeked operand");
                self.encode_memory_access_operand(operand, target);
            }
        }
    }

    fn encode_memory_access_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
        target: &mut Vec<dr::Operand>,
    ) {
        let OperandValue::MemoryAccess(memory) = operand.value() else {
            self.module_builder
                .emit_error(operand.span().start(), "Invalid memory access operand");
            return;
        };
        target.push(dr::Operand::MemoryAccess(memory.mask()));
        if let Some(alignment) = memory.alignment() {
            target.push(dr::Operand::LiteralBit32(literal_to_u32(alignment)));
        }
        if let Some(scope) = memory.make_pointer_available_scope() {
            target.push(dr::Operand::IdRef(
                self.module_builder.resolve_id_ref(scope),
            ));
        }
        if let Some(scope) = memory.make_pointer_visible_scope() {
            target.push(dr::Operand::IdRef(
                self.module_builder.resolve_id_ref(scope),
            ));
        }
    }

    fn take_image_operands(
        &mut self,
        operands: &mut std::slice::Iter<'_, ParsedOperand<'a>>,
        target: &mut Vec<dr::Operand>,
    ) {
        if let Some(next) = operands.as_slice().first() {
            if next.descriptor().kind() == OperandKind::ImageOperands {
                let operand = operands.next().expect("peeked operand");
                self.encode_image_operands_operand(operand, target);
            }
        }
    }

    fn encode_image_operands_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
        target: &mut Vec<dr::Operand>,
    ) {
        let OperandValue::ImageOperands(img_ops) = operand.value() else {
            self.module_builder
                .emit_error(operand.span().start(), "Invalid image operands");
            return;
        };
        target.push(dr::Operand::ImageOperands(img_ops.mask()));
        for id_ref in img_ops.dependent_ids() {
            target.push(dr::Operand::IdRef(
                self.module_builder.resolve_id_ref(*id_ref),
            ));
        }
    }

    /// Translate a typed-result image operation with N IdRef operands followed
    /// by optional ImageOperands. Covers the majority of image sampling/fetch/
    /// gather/read/query instructions.
    fn translate_image_op(&mut self, instruction: &ParsedInstruction<'a>, id_count: usize) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Image operation missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Image operation missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let mut dr_operands = Vec::new();
        for i in 0..id_count {
            let Some(op) = operands.next() else {
                self.module_builder.emit_error(
                    opcode_pos,
                    format!("Image operation missing operand {}", i + 1),
                );
                return;
            };
            let Some(id) = self.operand_as_id(op, &format!("operand {}", i + 1)) else {
                return;
            };
            dr_operands.push(dr::Operand::IdRef(id));
        }
        self.take_image_operands(&mut operands, &mut dr_operands);
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            instruction.opcode(),
            Some(type_id),
            Some(result_id),
            dr_operands,
        );
        self.push_block_instruction(inst, opcode_pos);
    }

    /// Translate OpImageWrite which has no result type/id: image, coord, texel + optional ImageOperands.
    fn translate_image_write(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let mut dr_operands = Vec::new();
        for label in ["image", "coordinate", "texel"] {
            let Some(op) = operands.next() else {
                self.module_builder.emit_error(
                    opcode_pos,
                    format!("OpImageWrite missing {label} operand"),
                );
                return;
            };
            let Some(id) = self.operand_as_id(op, label) else {
                return;
            };
            dr_operands.push(dr::Operand::IdRef(id));
        }
        self.take_image_operands(&mut operands, &mut dr_operands);
        let inst = dr::Instruction::new(spirv::Op::ImageWrite, None, None, dr_operands);
        self.push_block_instruction(inst, opcode_pos);
    }

    fn encode_decoration_operands(
        &mut self,
        decoration: spirv::Decoration,
        operands: &[ParsedOperand<'a>],
        start_index: usize,
        opcode_pos: MessagePosition,
    ) -> Option<Vec<dr::Operand>> {
        let mut encoded = Vec::new();
        let mut index = start_index;
        let descriptors = decoration_operand_descriptors(decoration);
        for descriptor in descriptors {
            match descriptor.quantifier() {
                OperandQuantifier::One => {
                    let Some(next) = operands.get(index) else {
                        self.module_builder
                            .emit_error(opcode_pos, "Decoration missing required operand");
                        return None;
                    };
                    let value = self.encode_operand_for_kind(next, descriptor.kind())?;
                    encoded.push(value);
                    index += 1;
                }
                OperandQuantifier::ZeroOrOne => {
                    if let Some(next) = operands.get(index) {
                        let value = self.encode_operand_for_kind(next, descriptor.kind())?;
                        encoded.push(value);
                        index += 1;
                    }
                }
                OperandQuantifier::ZeroOrMore => {
                    while let Some(next) = operands.get(index) {
                        let value = self.encode_operand_for_kind(next, descriptor.kind())?;
                        encoded.push(value);
                        index += 1;
                    }
                }
            }
        }
        if index != operands.len() {
            if let Some(extra) = operands.get(index) {
                self.module_builder.emit_error(
                    extra.span().start(),
                    "Decoration received unexpected operands",
                );
            }
            return None;
        }
        Some(encoded)
    }

    fn encode_operand_for_kind(
        &mut self,
        operand: &ParsedOperand<'a>,
        kind: OperandKind,
    ) -> Option<dr::Operand> {
        use rspirv::grammar::OperandKind::*;
        match kind {
            IdRef => self
                .operand_as_id(operand, "decoration id")
                .map(dr::Operand::IdRef),
            IdScope => self
                .operand_as_id(operand, "decoration scope id")
                .map(dr::Operand::IdScope),
            IdMemorySemantics => self
                .operand_as_id(operand, "decoration memory semantics id")
                .map(dr::Operand::IdMemorySemantics),
            LiteralInteger | LiteralContextDependentNumber => match operand.value() {
                OperandValue::Literal(literal) => Some(encode_literal_operand(literal)),
                _ => {
                    self.module_builder
                        .emit_error(operand.span().start(), "Expected literal integer operand");
                    None
                }
            },
            LiteralString => match operand.value() {
                OperandValue::String(value) => {
                    Some(dr::Operand::LiteralString(parse_string_literal(value)))
                }
                _ => {
                    self.module_builder
                        .emit_error(operand.span().start(), "Expected string literal operand");
                    None
                }
            },
            LiteralFloat => self.encode_float_operand(operand),
            BuiltIn => self.encode_enumerant_operand::<spirv::BuiltIn>(operand, "built-in"),
            FunctionParameterAttribute => self
                .encode_enumerant_operand::<spirv::FunctionParameterAttribute>(
                    operand,
                    "function parameter attribute",
                ),
            FPRoundingMode => {
                self.encode_enumerant_operand::<spirv::FPRoundingMode>(operand, "FP rounding mode")
            }
            FPFastMathMode => self.encode_fp_fast_math_mode(operand),
            FPDenormMode => {
                self.encode_enumerant_operand::<spirv::FPDenormMode>(operand, "FP denorm mode")
            }
            FPOperationMode => self
                .encode_enumerant_operand::<spirv::FPOperationMode>(operand, "FP operation mode"),
            LinkageType => {
                self.encode_enumerant_operand::<spirv::LinkageType>(operand, "linkage type")
            }
            AccessQualifier => {
                self.encode_enumerant_operand::<spirv::AccessQualifier>(operand, "access qualifier")
            }
            HostAccessQualifier => self.encode_enumerant_operand::<spirv::HostAccessQualifier>(
                operand,
                "host access qualifier",
            ),
            InitializationModeQualifier => self
                .encode_enumerant_operand::<spirv::InitializationModeQualifier>(
                    operand,
                    "initialization mode",
                ),
            LoadCacheControl => {
                self.encode_enumerant_operand::<spirv::LoadCacheControl>(operand, "cache control")
            }
            StoreCacheControl => {
                self.encode_enumerant_operand::<spirv::StoreCacheControl>(operand, "cache control")
            }
            _ => {
                // Decoration operands should only reference the operand kinds enumerated in the
                // grammar subset we generated. If we reach here, the grammar introduced a new
                // operand kind that needs encoding support.
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!("Unsupported decoration operand kind: {:?}", kind),
                );
                None
            }
        }
    }

    fn encode_float_operand(&mut self, operand: &ParsedOperand<'a>) -> Option<dr::Operand> {
        match operand.value() {
            OperandValue::Literal(literal) => Some(encode_literal_operand(literal)),
            OperandValue::Word(word) => {
                let text = word.as_str();
                match text.parse::<f64>() {
                    Ok(value64) => {
                        if value64.is_nan() {
                            Some(dr::Operand::LiteralBit32(f32::NAN.to_bits()))
                        } else {
                            let value32 = value64 as f32;
                            if value64 == (value32 as f64) {
                                Some(dr::Operand::LiteralBit32(value32.to_bits()))
                            } else {
                                Some(dr::Operand::LiteralBit64(value64.to_bits()))
                            }
                        }
                    }
                    Err(_) => {
                        self.module_builder
                            .emit_error(operand.span().start(), "Failed to parse float literal");
                        None
                    }
                }
            }
            _ => {
                self.module_builder
                    .emit_error(operand.span().start(), "Float literal expected");
                None
            }
        }
    }

    fn encode_fp_fast_math_mode(&mut self, operand: &ParsedOperand<'a>) -> Option<dr::Operand> {
        match operand.value() {
            OperandValue::Literal(literal) => Some(dr::Operand::FPFastMathMode(
                spirv::FPFastMathMode::from_bits_truncate(literal_to_u32(literal)),
            )),
            OperandValue::Word(word) => {
                let mut mode = spirv::FPFastMathMode::empty();
                for part in word.as_str().split('|').map(str::trim) {
                    if part.is_empty() || part == "None" {
                        continue;
                    }
                    let flag = match part {
                        "NotNaN" => spirv::FPFastMathMode::NOT_NAN,
                        "NotInf" => spirv::FPFastMathMode::NOT_INF,
                        "NSZ" => spirv::FPFastMathMode::NSZ,
                        "AllowRecip" => spirv::FPFastMathMode::ALLOW_RECIP,
                        "Fast" => spirv::FPFastMathMode::FAST,
                        "AllowContract" | "AllowContractFastINTEL" => {
                            spirv::FPFastMathMode::ALLOW_CONTRACT
                        }
                        "AllowReassoc" | "AllowReassocINTEL" => {
                            spirv::FPFastMathMode::ALLOW_REASSOC
                        }
                        "AllowTransform" | "AllowTransformINTEL" => {
                            spirv::FPFastMathMode::ALLOW_TRANSFORM
                        }
                        other => {
                            self.module_builder.emit_error(
                                operand.span().start(),
                                format!("Unknown FPFastMathMode flag '{other}'"),
                            );
                            return None;
                        }
                    };
                    mode |= flag;
                }
                Some(dr::Operand::FPFastMathMode(mode))
            }
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "FPFastMathMode operand must be literal or enumerant",
                );
                None
            }
        }
    }

    fn parse_fp_encoding_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
    ) -> Option<spirv::FPEncoding> {
        match operand.value() {
            OperandValue::Literal(literal) => {
                let value = literal_to_u32(literal);
                match spirv::FPEncoding::from_u32(value) {
                    Some(encoding) => Some(encoding),
                    None => {
                        self.module_builder.emit_error(
                            operand.span().start(),
                            format!("Unknown FPEncoding literal {value}"),
                        );
                        None
                    }
                }
            }
            OperandValue::Word(word) => match word.as_str().parse::<spirv::FPEncoding>() {
                Ok(encoding) => Some(encoding),
                Err(_) => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!("Unknown FPEncoding '{}'", word.as_str()),
                    );
                    None
                }
            },
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "FPEncoding operand must be literal or enumerant",
                );
                None
            }
        }
    }

    fn encode_enumerant_operand<E>(
        &mut self,
        operand: &ParsedOperand<'a>,
        label: &str,
    ) -> Option<dr::Operand>
    where
        E: FromStr,
        dr::Operand: From<E>,
    {
        self.parse_enum_operand_value::<E>(operand, label)
            .map(dr::Operand::from)
    }

    fn parse_enum_operand_value<E>(&mut self, operand: &ParsedOperand<'a>, label: &str) -> Option<E>
    where
        E: FromStr,
    {
        self.parse_enum_operand::<E>(Some(operand), label, operand.span().start())
    }

    fn resolve_composite_access(
        &mut self,
        mut type_id: u32,
        indexes: &[(u32, MessagePosition)],
    ) -> Option<u32> {
        for (depth, (index, position)) in indexes.iter().enumerate() {
            let Some(info) = self.module_builder.composite_type(type_id) else {
                self.module_builder
                    .emit_error(*position, "Operand type is not a composite value");
                return None;
            };
            match info {
                CompositeTypeInfo::Vector(vector) => {
                    let vector = *vector;
                    if *index >= vector.component_count() {
                        self.module_builder.emit_error(
                            *position,
                            format!(
                                "Composite extract index {index} exceeds vector width {}",
                                vector.component_count()
                            ),
                        );
                        return None;
                    }
                    if depth + 1 != indexes.len() {
                        self.module_builder
                            .emit_error(*position, "Cannot descend past a vector component");
                        return None;
                    }
                    type_id = vector.component_type();
                }
                CompositeTypeInfo::Array(array) => {
                    let array = *array;
                    let Some(length) = self.module_builder.array_length(&array) else {
                        self.module_builder.emit_error(
                            *position,
                            "Array length must be defined by an integer constant",
                        );
                        return None;
                    };
                    if *index >= length {
                        self.module_builder.emit_error(
                            *position,
                            format!("Array index {index} exceeds array length {length}",),
                        );
                        return None;
                    }
                    type_id = array.element_type();
                }
                CompositeTypeInfo::Struct(struct_info) => {
                    let Ok(field_index) = usize::try_from(*index) else {
                        self.module_builder
                            .emit_error(*position, "Struct index exceeds implementation limits");
                        return None;
                    };
                    let Some(field_type) = struct_info.field_type(field_index) else {
                        self.module_builder.emit_error(
                            *position,
                            format!(
                                "Struct index {index} exceeds field count {}",
                                struct_info.field_count()
                            ),
                        );
                        return None;
                    };
                    type_id = field_type;
                }
                CompositeTypeInfo::Matrix(matrix) => {
                    let matrix = *matrix;
                    if *index >= matrix.column_count() {
                        self.module_builder.emit_error(
                            *position,
                            format!(
                                "Matrix column index {index} exceeds column count {}",
                                matrix.column_count()
                            ),
                        );
                        return None;
                    }
                    type_id = matrix.column_type();
                }
            }
        }
        Some(type_id)
    }

    fn resolve_ext_inst_opcode(
        &mut self,
        info: &ExtInstImportInfo,
        operand: &ParsedOperand<'a>,
    ) -> Option<ResolvedExtInst<'static>> {
        match operand.value() {
            OperandValue::Literal(literal) => Some(ResolvedExtInst {
                opcode: literal_to_u32(literal),
                operands: None,
            }),
            OperandValue::Word(word) => {
                if let Some(inst) = info.kind.lookup(word.as_str()) {
                    return Some(inst.into());
                }
                if let Some(mapped) = lookup_custom_ext_inst_opcode(&info.name, word.as_str()) {
                    return Some(mapped);
                }
                if let Ok(value) = word.as_str().parse::<u32>() {
                    return Some(ResolvedExtInst {
                        opcode: value,
                        operands: None,
                    });
                }
                if info.kind.has_grammar() {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!(
                            "Unknown {} extended instruction '{}'",
                            info.name,
                            word.as_str()
                        ),
                    );
                } else {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!(
                            "Instruction set '{}' does not support named opcodes; use numeric identifiers",
                            info.name
                        ),
                    );
                }
                None
            }
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "Extended instruction opcode must be a literal or enumerant name",
                );
                None
            }
        }
    }

    fn encode_ext_inst_operand(
        &mut self,
        descriptor: &LogicalOperand,
        operand: &ParsedOperand<'a>,
        opcode_pos: MessagePosition,
    ) -> Option<dr::Operand> {
        match descriptor.kind {
            OperandKind::IdRef => self
                .operand_as_id(operand, "extended instruction operand")
                .map(dr::Operand::IdRef),
            OperandKind::LiteralInteger
            | OperandKind::LiteralContextDependentNumber
            | OperandKind::LiteralExtInstInteger => match operand.value() {
                OperandValue::Literal(literal) => {
                    Some(dr::Operand::LiteralBit32(literal_to_u32(literal)))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Extended instruction literal operand expected",
                    );
                    None
                }
            },
            OperandKind::FPRoundingMode => {
                let mode = self.parse_enum_operand::<spirv::FPRoundingMode>(
                    Some(operand),
                    "rounding mode",
                    opcode_pos,
                )?;
                Some(dr::Operand::FPRoundingMode(mode))
            }
            OperandKind::LiteralString => match operand.value() {
                OperandValue::String(value) => {
                    Some(dr::Operand::LiteralString(parse_string_literal(value)))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Extended instruction string operand expected",
                    );
                    None
                }
            },
            _ => {
                self.module_builder.emit_error(
                    opcode_pos,
                    format!(
                        "Extended instruction operand kind {:?} is not supported",
                        descriptor.kind
                    ),
                );
                None
            }
        }
    }

    fn encode_generic_ext_inst_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
    ) -> Option<dr::Operand> {
        match operand.value() {
            OperandValue::Id(id) => {
                Some(dr::Operand::IdRef(self.module_builder.resolve_id_ref(*id)))
            }
            OperandValue::Literal(literal) => {
                Some(dr::Operand::LiteralBit32(literal_to_u32(literal)))
            }
            OperandValue::String(value) => {
                Some(dr::Operand::LiteralString(parse_string_literal(value)))
            }
            OperandValue::Word(word) => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!(
                        "Named opcode '{}' is not supported for this instruction set",
                        word.as_str()
                    ),
                );
                None
            }
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "Extended instruction operand must be an id or literal",
                );
                None
            }
        }
    }

    fn parse_enum_operand<E>(
        &mut self,
        operand: Option<&ParsedOperand<'a>>,
        label: &str,
        missing_position: MessagePosition,
    ) -> Option<E>
    where
        E: FromStr,
    {
        let operand = match operand {
            Some(value) => value,
            None => {
                self.module_builder
                    .emit_error(missing_position, format!("Missing {label}"));
                return None;
            }
        };
        let word = match operand.value() {
            OperandValue::Word(word) => word,
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!("{label} must be an enumerant"),
                );
                return None;
            }
        };
        match word.as_str().parse::<E>() {
            Ok(value) => Some(value),
            Err(_err) => {
                self.module_builder
                    .emit_error(operand.span().start(), format!("Invalid {label}"));
                None
            }
        }
    }

    fn parse_loop_control_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
        remaining: &mut std::slice::Iter<'_, ParsedOperand<'a>>,
    ) -> Option<(spirv::LoopControl, Vec<dr::Operand>)> {
        let word = match operand.value() {
            OperandValue::Word(word) => word,
            _ => {
                self.module_builder
                    .emit_error(operand.span().start(), "Loop control must be an enumerant");
                return None;
            }
        };
        let mut control = spirv::LoopControl::empty();
        let mut additional_operands = Vec::new();
        for part in word.as_str().split('|').map(str::trim) {
            if part.is_empty() || part == "None" {
                continue;
            }
            match part {
                "Unroll" => control |= spirv::LoopControl::UNROLL,
                "DontUnroll" => control |= spirv::LoopControl::DONT_UNROLL,
                "DependencyInfinite" => control |= spirv::LoopControl::DEPENDENCY_INFINITE,
                "DependencyLength" => {
                    control |= spirv::LoopControl::DEPENDENCY_LENGTH;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "MinIterations" => {
                    control |= spirv::LoopControl::MIN_ITERATIONS;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "MaxIterations" => {
                    control |= spirv::LoopControl::MAX_ITERATIONS;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "IterationMultiple" => {
                    control |= spirv::LoopControl::ITERATION_MULTIPLE;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "PeelCount" => {
                    control |= spirv::LoopControl::PEEL_COUNT;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "PartialCount" => {
                    control |= spirv::LoopControl::PARTIAL_COUNT;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                other => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!("Unknown loop control flag '{other}'"),
                    );
                    return None;
                }
            }
        }
        Some((control, additional_operands))
    }

    fn expect_loop_control_literal(
        &mut self,
        remaining: &mut std::slice::Iter<'_, ParsedOperand<'a>>,
        flag_position: MessagePosition,
        label: &str,
    ) -> Option<dr::Operand> {
        let Some(operand) = remaining.next() else {
            self.module_builder.emit_error(
                flag_position,
                format!("Loop control flag {label} requires a literal operand"),
            );
            return None;
        };
        let literal = match operand.value() {
            OperandValue::Literal(lit) => lit,
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!("Loop control flag {label} requires a literal operand"),
                );
                return None;
            }
        };
        Some(dr::Operand::LiteralBit32(literal_to_u32(literal)))
    }

    fn parse_function_control(
        &mut self,
        operand: &ParsedOperand<'a>,
    ) -> Option<spirv::FunctionControl> {
        let word = match operand.value() {
            OperandValue::Word(word) => word,
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "Function control must be an enumerant",
                );
                return None;
            }
        };
        let text = word.as_str();
        if text == "None" {
            return Some(spirv::FunctionControl::empty());
        }

        let mut control = spirv::FunctionControl::empty();
        for part in text.split('|').map(str::trim) {
            if part.is_empty() {
                continue;
            }
            let flag = match part {
                "Inline" => Some(spirv::FunctionControl::INLINE),
                "DontInline" => Some(spirv::FunctionControl::DONT_INLINE),
                "Pure" => Some(spirv::FunctionControl::PURE),
                "Const" => Some(spirv::FunctionControl::CONST),
                _ => None,
            };
            match flag {
                Some(value) => control |= value,
                None => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!("Unknown function control flag '{part}'"),
                    );
                    return None;
                }
            }
        }
        Some(control)
    }

    fn parse_selection_control(
        &mut self,
        operand: &ParsedOperand<'a>,
    ) -> Option<spirv::SelectionControl> {
        let word = match operand.value() {
            OperandValue::Word(word) => word,
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "Selection control must be an enumerant",
                );
                return None;
            }
        };
        let mut control = spirv::SelectionControl::empty();
        for part in word.as_str().split('|').map(str::trim) {
            if part.is_empty() || part == "None" {
                continue;
            }
            match part {
                "Flatten" => control |= spirv::SelectionControl::FLATTEN,
                "DontFlatten" => control |= spirv::SelectionControl::DONT_FLATTEN,
                other => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!("Unknown selection control flag '{other}'"),
                    );
                    return None;
                }
            }
        }
        Some(control)
    }

    fn emit_builder_error(&mut self, error: BuildError, position: MessagePosition) {
        self.module_builder
            .emit_error(position, format!("Assembler builder error: {error}"));
    }

    /// Finalizes the translation and returns the constructed module plus diagnostics.
    pub fn finish(self) -> (dr::Module, Vec<DiagnosticMessage<'static>>) {
        let output = self.into_parts();
        (output.module, output.diagnostics)
    }

    /// Finalizes the translation and returns the module, diagnostics, and span map.
    ///
    /// The span map is only populated if the translator was created with `track_spans: true`.
    pub fn finish_with_spans(
        self,
    ) -> (dr::Module, Vec<DiagnosticMessage<'static>>, Option<SpanMap>) {
        let output = self.into_parts();
        (output.module, output.diagnostics, output.span_map)
    }

    fn into_parts(self) -> AssemblyOutput {
        let (diagnostics, next_id, span_map) = self.module_builder.finish_with_spans();
        let mut module = self.builder.module();
        match module.header.as_mut() {
            Some(header) => header.bound = next_id,
            None => module.header = Some(dr::ModuleHeader::new(next_id)),
        }
        AssemblyOutput {
            module,
            diagnostics,
            span_map,
        }
    }
}

pub(super) struct AssemblyOutput {
    pub(super) module: dr::Module,
    pub(super) diagnostics: Vec<DiagnosticMessage<'static>>,
    pub(super) span_map: Option<SpanMap>,
}

pub(super) fn configure_builder_for_env(builder: &mut dr::Builder, env: TargetEnv) {
    let version = env.spirv_version();
    builder.set_version(version.major(), version.minor());
}

/// Assembles a sequence of parsed instructions into a SPIR-V module, returning both the module and
/// any diagnostics emitted along the way.
pub fn assemble_instructions<'a>(
    instructions: &[&'a ParsedInstruction<'a>],
) -> Result<dr::Module, AssemblyError> {
    let mut translator = AssemblyTranslator::new();
    for instruction in instructions {
        translator.translate(instruction);
    }
    let AssemblyOutput {
        module,
        diagnostics,
        ..
    } = translator.into_parts();
    finalize_result(module, diagnostics)
}

pub(super) fn literal_to_u32(literal: &LiteralNumber) -> u32 {
    match literal {
        LiteralNumber::Unsigned(value) => *value as u32,
        LiteralNumber::Signed(value) => *value as u32,
    }
}

pub(super) fn literal_to_u64(literal: &LiteralNumber) -> u64 {
    match literal {
        LiteralNumber::Unsigned(value) => *value,
        LiteralNumber::Signed(value) => *value as u64,
    }
}

pub(super) fn encode_literal_operand(literal: &LiteralNumber) -> dr::Operand {
    match literal {
        LiteralNumber::Unsigned(value) if *value <= u32::MAX as u64 => {
            dr::Operand::LiteralBit32(*value as u32)
        }
        LiteralNumber::Unsigned(value) => dr::Operand::LiteralBit64(*value),
        LiteralNumber::Signed(value) if *value >= i32::MIN as i64 && *value <= i32::MAX as i64 => {
            dr::Operand::LiteralBit32(*value as u32)
        }
        LiteralNumber::Signed(value) => dr::Operand::LiteralBit64(*value as u64),
    }
}

pub(super) fn finalize_result<T>(
    value: T,
    diagnostics: Vec<DiagnosticMessage<'static>>,
) -> Result<T, AssemblyError> {
    if diagnostics.is_empty() {
        Ok(value)
    } else {
        Err(AssemblyError::new(diagnostics))
    }
}

pub(super) fn assemble_text_with_translator<'a>(
    text: &'a str,
    mut translator: AssemblyTranslator<'a>,
) -> Result<Vec<u32>, AssemblyError> {
    let mut diagnostics = Vec::new();
    let mut instructions = Vec::new();

    let mut line_bounds = Vec::new();
    let mut line_start = 0usize;
    for (idx, byte) in text.as_bytes().iter().enumerate() {
        if *byte == b'\n' {
            let mut line_end = idx;
            if line_end > line_start && text.as_bytes()[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            line_bounds.push((line_start, line_end));
            line_start = idx + 1;
        }
    }
    if line_start < text.len() {
        line_bounds.push((line_start, text.len()));
    }

    let mut line_index = 0usize;
    while line_index < line_bounds.len() {
        let (start, _) = line_bounds[line_index];
        let mut last_line = line_index;
        let mut span_end = line_bounds[last_line].1;
        loop {
            match process_line(text, start, span_end, line_index) {
                Ok(Some(parsed)) => {
                    instructions.push(parsed);
                    line_index = last_line + 1;
                    break;
                }
                Ok(None) => {
                    line_index = last_line + 1;
                    break;
                }
                Err(error) => {
                    let is_unterminated =
                        error.diagnostic().message() == "unterminated string literal";
                    if is_unterminated && last_line + 1 < line_bounds.len() {
                        last_line += 1;
                        span_end = line_bounds[last_line].1;
                        continue;
                    }
                    diagnostics.push(error.into_diagnostic());
                    line_index = last_line + 1;
                    break;
                }
            }
        }
    }

    translator.reserve_numeric_result_ids(&instructions);
    for instruction in &instructions {
        translator.translate(instruction);
    }

    let AssemblyOutput {
        module,
        diagnostics: mut translator_diagnostics,
        ..
    } = translator.into_parts();
    diagnostics.append(&mut translator_diagnostics);
    let words = match finalize_result((), diagnostics) {
        Ok(_) => module.assemble(),
        Err(error) => return Err(error),
    };
    Ok(words)
}

pub(super) fn assemble_text_with_translator_for_spans<'a>(
    text: &'a str,
    mut translator: AssemblyTranslator<'a>,
) -> Result<AssemblyWithSpans, AssemblyError> {
    let mut diagnostics = Vec::new();
    let mut instructions = Vec::new();

    let mut line_bounds = Vec::new();
    let mut line_start = 0usize;
    for (idx, byte) in text.as_bytes().iter().enumerate() {
        if *byte == b'\n' {
            let mut line_end = idx;
            if line_end > line_start && text.as_bytes()[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            line_bounds.push((line_start, line_end));
            line_start = idx + 1;
        }
    }
    if line_start < text.len() {
        line_bounds.push((line_start, text.len()));
    }

    let mut line_index = 0usize;
    while line_index < line_bounds.len() {
        let (start, _) = line_bounds[line_index];
        let mut last_line = line_index;
        let mut span_end = line_bounds[last_line].1;
        loop {
            match process_line(text, start, span_end, line_index) {
                Ok(Some(parsed)) => {
                    instructions.push(parsed);
                    line_index = last_line + 1;
                    break;
                }
                Ok(None) => {
                    line_index = last_line + 1;
                    break;
                }
                Err(error) => {
                    let is_unterminated =
                        error.diagnostic().message() == "unterminated string literal";
                    if is_unterminated && last_line + 1 < line_bounds.len() {
                        last_line += 1;
                        span_end = line_bounds[last_line].1;
                        continue;
                    }
                    diagnostics.push(error.into_diagnostic());
                    line_index = last_line + 1;
                    break;
                }
            }
        }
    }

    translator.reserve_numeric_result_ids(&instructions);
    for instruction in &instructions {
        translator.translate(instruction);
    }

    let AssemblyOutput {
        module,
        diagnostics: mut translator_diagnostics,
        span_map,
    } = translator.into_parts();
    diagnostics.append(&mut translator_diagnostics);

    match finalize_result((), diagnostics) {
        Ok(_) => Ok(AssemblyWithSpans {
            words: module.assemble(),
            span_map: span_map.unwrap_or_default(),
        }),
        Err(error) => Err(error),
    }
}

pub(super) fn process_line<'a>(
    source: &'a str,
    line_start: usize,
    line_end: usize,
    line_index: usize,
) -> Result<Option<ParsedInstruction<'a>>, ParseError> {
    if line_start >= line_end {
        return Ok(None);
    }
    let line_slice = &source[line_start..line_end];
    let leading_ws = line_slice.len() - line_slice.trim_start().len();
    let trailing_ws = line_slice.len() - line_slice.trim_end().len();
    if leading_ws >= line_slice.len() - trailing_ws {
        return Ok(None);
    }
    let content_start = line_start + leading_ws;
    let content_end = line_end - trailing_ws;
    let line = &source[content_start..content_end];
    if line.is_empty() || line.starts_with(';') {
        return Ok(None);
    }
    let line_number = u32::try_from(line_index).unwrap_or(u32::MAX);
    let column_offset = u32::try_from(leading_ws).unwrap_or(u32::MAX);
    let index_offset = u32::try_from(content_start).unwrap_or(u32::MAX);
    let origin = MessagePosition::new(line_number, column_offset, index_offset);

    parse_instruction_with_origin(line, origin).map(Some)
}
