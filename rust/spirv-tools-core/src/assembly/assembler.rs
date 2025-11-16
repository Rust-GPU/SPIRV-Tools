use std::borrow::Cow;
use std::collections::{btree_map::Entry, BTreeMap};
use std::str::FromStr;

use rspirv::binary::Assemble;
use rspirv::dr::{self, Error as BuildError, InsertPoint};
use rspirv::grammar::OperandKind;
use rspirv::spirv;

use super::instruction::{IdRef, LiteralNumber, ResultId, SpirvId, TypeId};
use super::parser::{parse_instruction, OperandValue, ParsedInstruction, ParsedOperand};
use crate::diagnostic::{DiagnosticMessage, MessagePosition};
use crate::message::MessageLevel;

/// Tracks textual identifiers and diagnostics while constructing a module.
#[derive(Debug)]
pub struct ModuleBuilder<'a> {
    named_ids: BTreeMap<&'a str, u32>,
    next_numeric_id: u32,
    diagnostics: Vec<DiagnosticMessage<'static>>,
}

impl<'a> Default for ModuleBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ModuleBuilder<'a> {
    /// Creates a new builder that assigns numeric IDs starting at 1.
    pub fn new() -> Self {
        Self {
            named_ids: BTreeMap::new(),
            next_numeric_id: 1,
            diagnostics: Vec::new(),
        }
    }

    /// Resolves a result identifier to a numeric ID.
    pub fn resolve_result_id(&mut self, id: ResultId<'a>) -> u32 {
        self.resolve_spirv_id(id.as_spirv_id())
    }

    /// Resolves a type identifier to a numeric ID.
    pub fn resolve_type_id(&mut self, id: TypeId<'a>) -> u32 {
        self.resolve_spirv_id(id.as_spirv_id())
    }

    /// Resolves an ID reference to a numeric ID.
    pub fn resolve_id_ref(&mut self, id: IdRef<'a>) -> u32 {
        self.resolve_spirv_id(id.as_spirv_id())
    }

    /// Emits an assembler diagnostic.
    pub fn emit_error(&mut self, position: MessagePosition, message: impl Into<Cow<'static, str>>) {
        self.diagnostics.push(
            DiagnosticMessage::new(MessageLevel::Error, position, message).with_source("assembler"),
        );
    }

    /// Returns the collected diagnostics.
    pub fn diagnostics(&self) -> &[DiagnosticMessage<'static>] {
        &self.diagnostics
    }

    /// Consumes the builder and returns the diagnostics alongside the next ID bound.
    pub fn finish(self) -> (Vec<DiagnosticMessage<'static>>, u32) {
        (self.diagnostics, self.next_numeric_id)
    }

    fn resolve_spirv_id(&mut self, id: SpirvId<'a>) -> u32 {
        match id {
            SpirvId::Named(named) => match self.named_ids.entry(named.name()) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let allocated = self.next_numeric_id;
                    self.next_numeric_id += 1;
                    entry.insert(allocated);
                    allocated
                }
            },
            SpirvId::Numeric(raw) => {
                let value = raw.get();
                self.next_numeric_id = self.next_numeric_id.max(value + 1);
                value
            }
        }
    }

    fn bind_result_id(&mut self, result_id: ResultId<'a>, numeric: u32) {
        self.bind_spirv_id(result_id.as_spirv_id(), numeric);
    }

    fn bind_spirv_id(&mut self, id: SpirvId<'a>, numeric: u32) {
        if let SpirvId::Named(named) = id {
            self.named_ids.insert(named.name(), numeric);
        }
        self.next_numeric_id = self.next_numeric_id.max(numeric + 1);
    }
}

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
        Self {
            module_builder: ModuleBuilder::new(),
            builder: dr::Builder::new(),
        }
    }

    /// Translates a parsed instruction.
    pub fn translate(&mut self, instruction: &ParsedInstruction<'a>) {
        match instruction.opcode() {
            spirv::Op::Capability => self.translate_capability(instruction),
            spirv::Op::TypeVoid => self.translate_type_void(instruction),
            spirv::Op::TypeInt => self.translate_type_int(instruction),
            spirv::Op::TypeFunction => self.translate_type_function(instruction),
            spirv::Op::TypePointer => self.translate_type_pointer(instruction),
            spirv::Op::TypeVector => self.translate_type_vector(instruction),
            spirv::Op::MemoryModel => self.translate_memory_model(instruction),
            spirv::Op::EntryPoint => self.translate_entry_point(instruction),
            spirv::Op::ExecutionMode => self.translate_execution_mode(instruction),
            spirv::Op::AccessChain => self.translate_access_chain(instruction, false),
            spirv::Op::InBoundsAccessChain => self.translate_access_chain(instruction, true),
            spirv::Op::CopyMemory => self.translate_copy_memory(instruction, false),
            spirv::Op::CopyMemorySized => self.translate_copy_memory(instruction, true),
            spirv::Op::SelectionMerge => self.translate_selection_merge(instruction),
            spirv::Op::LoopMerge => self.translate_loop_merge(instruction),
            spirv::Op::CompositeConstruct => self.translate_composite_construct(instruction),
            spirv::Op::VectorShuffle => self.translate_vector_shuffle(instruction),
            spirv::Op::CompositeExtract => self.translate_composite_extract(instruction),
            spirv::Op::CompositeInsert => self.translate_composite_insert(instruction),
            spirv::Op::Function => self.translate_function(instruction),
            spirv::Op::FunctionParameter => self.translate_function_parameter(instruction),
            spirv::Op::Label => self.translate_label(instruction),
            spirv::Op::Branch => self.translate_branch(instruction),
            spirv::Op::BranchConditional => self.translate_branch_conditional(instruction),
            spirv::Op::Return => self.translate_return(),
            spirv::Op::ReturnValue => self.translate_return_value(instruction),
            spirv::Op::FunctionEnd => self.translate_function_end(),
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
            _ => self
                .module_builder
                .emit_error(MessagePosition::default(), "unsupported opcode"),
        }
    }

    fn translate_capability(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpCapability missing enumerant");
            return;
        };
        let Some(capability) =
            self.parse_enum_operand::<spirv::Capability>(Some(operand), "capability")
        else {
            return;
        };
        self.builder.capability(capability);
    }

    fn translate_type_void(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypeVoid requires a result id",
            );
            return;
        };
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_void_id(Some(result_id));
    }

    fn translate_type_int(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypeInt requires a result identifier",
            );
            return;
        };

        let mut operands = instruction.operands().iter();
        let Some(width_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypeInt missing width literal",
            );
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
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypeInt missing signedness literal",
            );
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
    }

    fn translate_constant(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpConstant missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpConstant requires a result identifier",
            );
            return;
        };

        let literal_operand = match instruction.operands().first() {
            Some(operand) => operand,
            None => {
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpConstant missing literal operand",
                );
                return;
            }
        };
        let literal = match literal_operand.value() {
            OperandValue::Literal(value) => value,
            _ => {
                self.module_builder.emit_error(
                    literal_operand.span().start(),
                    "OpConstant literal operand must be numeric",
                );
                return;
            }
        };

        let type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst = dr::Instruction::new(
            spirv::Op::Constant,
            Some(type_id),
            Some(result_id),
            vec![dr::Operand::LiteralBit32(literal_to_u32(literal))],
        );
        self.builder.module_mut().types_global_values.push(inst);
    }

    fn translate_type_function(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypeFunction missing result id",
            );
            return;
        };

        let mut operands = instruction.operands().iter();
        let return_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpTypeFunction missing return type",
                );
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
    }

    fn translate_function(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpFunction missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpFunction missing result id");
            return;
        };

        let mut operands = instruction.operands().iter();
        let control_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpFunction missing control operand",
                );
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
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpFunction missing function type",
                );
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

        let result_type = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        if let Err(error) =
            self.builder
                .begin_function(result_type, Some(result_id), control, function_type)
        {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_function_parameter(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpFunctionParameter missing result type",
            );
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpFunctionParameter missing result id",
            );
            return;
        };

        let type_id = self.module_builder.resolve_type_id(result_type);
        match self.builder.function_parameter(type_id) {
            Ok(parameter_id) => self.module_builder.bind_result_id(result_id, parameter_id),
            Err(error) => self.emit_builder_error(error, MessagePosition::default()),
        }
    }

    fn translate_label(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpLabel missing result id");
            return;
        };
        let label_id = self.module_builder.resolve_result_id(result_id);
        if let Err(error) = self.builder.begin_block(Some(label_id)) {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_return(&mut self) {
        if let Err(error) = self.builder.ret() {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_return_value(&mut self, instruction: &ParsedInstruction<'a>) {
        let operand = match instruction.operands().first() {
            Some(operand) => operand,
            None => {
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpReturnValue missing result id",
                );
                return;
            }
        };
        let Some(value_id) = self.operand_as_id(operand, "return value") else {
            return;
        };
        if let Err(error) = self.builder.ret_value(value_id) {
            self.emit_builder_error(error, operand.span().start());
        }
    }

    fn translate_function_end(&mut self) {
        if let Err(error) = self.builder.end_function() {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_type_pointer(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypePointer missing result id",
            );
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(storage_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypePointer missing storage class",
            );
            return;
        };
        let Some(storage_class) =
            self.parse_enum_operand::<spirv::StorageClass>(Some(storage_operand), "storage class")
        else {
            return;
        };
        let Some(pointee_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypePointer missing pointee type",
            );
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
    }

    fn translate_type_vector(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpTypeVector missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(component_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypeVector missing component type",
            );
            return;
        };
        let Some(component_type) = self.operand_as_id(component_operand, "component type") else {
            return;
        };
        let Some(count_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpTypeVector missing component count",
            );
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
    }

    fn translate_memory_model(&mut self, instruction: &ParsedInstruction<'a>) {
        let mut operands = instruction.operands().iter();
        let addressing = match self
            .parse_enum_operand::<spirv::AddressingModel>(operands.next(), "addressing model")
        {
            Some(value) => value,
            None => return,
        };
        let memory =
            match self.parse_enum_operand::<spirv::MemoryModel>(operands.next(), "memory model") {
                Some(value) => value,
                None => return,
            };
        self.builder.memory_model(addressing, memory);
    }

    fn translate_entry_point(&mut self, instruction: &ParsedInstruction<'a>) {
        let mut operands = instruction.operands().iter();
        let execution_model = match self
            .parse_enum_operand::<spirv::ExecutionModel>(operands.next(), "execution model")
        {
            Some(value) => value,
            None => return,
        };

        let function_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpEntryPoint missing function identifier",
                );
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
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpEntryPoint missing name literal",
                );
                return;
            }
        };
        let entry_name = match name_operand.value() {
            OperandValue::String(name) => *name,
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
            .entry_point(execution_model, function_id, entry_name, interfaces);
    }

    fn translate_access_chain(&mut self, instruction: &ParsedInstruction<'a>, in_bounds: bool) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "Access chain missing result type",
            );
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "Access chain missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(base_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "Access chain missing base pointer",
            );
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
        let result_type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        let result = if in_bounds {
            self.builder
                .in_bounds_access_chain(result_type_id, Some(result_id), base_id, indexes)
        } else {
            self.builder
                .access_chain(result_type_id, Some(result_id), base_id, indexes)
        };
        if let Err(error) = result {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_execution_mode(&mut self, instruction: &ParsedInstruction<'a>) {
        let mut operands = instruction.operands().iter();
        let entry_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpExecutionMode missing entry point",
                );
                return;
            }
        };
        let Some(entry_point) = self.operand_as_id(entry_operand, "entry point") else {
            return;
        };
        let Some(execution_mode) =
            self.parse_enum_operand::<spirv::ExecutionMode>(operands.next(), "execution mode")
        else {
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
    }

    fn translate_selection_merge(&mut self, instruction: &ParsedInstruction<'a>) {
        let mut operands = instruction.operands().iter();
        let Some(merge_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpSelectionMerge missing merge block",
            );
            return;
        };
        let Some(merge_id) = self.operand_as_id(merge_operand, "merge block") else {
            return;
        };
        let Some(control_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpSelectionMerge missing control mask",
            );
            return;
        };
        let Some(selection_control) = self.parse_selection_control(control_operand) else {
            return;
        };
        if let Err(error) = self.builder.selection_merge(merge_id, selection_control) {
            self.emit_builder_error(error, merge_operand.span().start());
        }
    }

    fn translate_loop_merge(&mut self, instruction: &ParsedInstruction<'a>) {
        let mut operands = instruction.operands().iter();
        let Some(merge_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpLoopMerge missing merge block",
            );
            return;
        };
        let Some(continue_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpLoopMerge missing continue target",
            );
            return;
        };
        let Some(control_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpLoopMerge missing control mask",
            );
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
        if let Err(error) =
            self.builder
                .loop_merge(merge_id, continue_id, loop_control, control_operands)
        {
            self.emit_builder_error(error, merge_operand.span().start());
        }
    }

    fn translate_composite_construct(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeConstruct missing result type",
            );
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeConstruct missing result id",
            );
            return;
        };
        if instruction.operands().is_empty() {
            self.module_builder.emit_error(
                MessagePosition::default(),
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
        let type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        if let Err(error) = self
            .builder
            .composite_construct(type_id, Some(result_id), constituents)
        {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_vector_shuffle(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpVectorShuffle missing result type",
            );
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpVectorShuffle missing result id",
            );
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(vector1_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpVectorShuffle missing first vector",
            );
            return;
        };
        let Some(vector2_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpVectorShuffle missing second vector",
            );
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
                OperandValue::Literal(literal) => components.push(literal_to_u32(literal)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Shuffle components must be literals",
                    );
                    return;
                }
            }
        }
        let type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        if let Err(error) = self.builder.vector_shuffle(
            type_id,
            Some(result_id),
            vector1_id,
            vector2_id,
            components,
        ) {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_composite_extract(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeExtract missing result type",
            );
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeExtract missing result id",
            );
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(composite_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeExtract missing composite value",
            );
            return;
        };
        let Some(composite_id) = self.operand_as_id(composite_operand, "composite value") else {
            return;
        };
        let mut indexes = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => indexes.push(literal_to_u32(literal)),
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
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeExtract requires at least one index",
            );
            return;
        }
        let type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        if let Err(error) =
            self.builder
                .composite_extract(type_id, Some(result_id), composite_id, indexes)
        {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_composite_insert(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeInsert missing result type",
            );
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeInsert missing result id",
            );
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(object_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeInsert missing object operand",
            );
            return;
        };
        let Some(object_id) = self.operand_as_id(object_operand, "object operand") else {
            return;
        };
        let Some(composite_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeInsert missing composite operand",
            );
            return;
        };
        let Some(composite_id) = self.operand_as_id(composite_operand, "composite operand") else {
            return;
        };
        let mut indexes = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => indexes.push(literal_to_u32(literal)),
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
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpCompositeInsert requires at least one index",
            );
            return;
        }
        let type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        if let Err(error) = self.builder.composite_insert(
            type_id,
            Some(result_id),
            object_id,
            composite_id,
            indexes,
        ) {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_copy_memory(&mut self, instruction: &ParsedInstruction<'a>, sized: bool) {
        let mut operands = instruction.operands().iter();
        let Some(target_operand) = operands.next() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpCopyMemory missing target");
            return;
        };
        let Some(source_operand) = operands.next() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpCopyMemory missing source");
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
                self.module_builder.emit_error(
                    MessagePosition::default(),
                    "OpCopyMemorySized missing size operand",
                );
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
        self.push_block_instruction(inst);
    }

    fn translate_branch(&mut self, instruction: &ParsedInstruction<'a>) {
        let operand = match instruction.operands().first() {
            Some(op) => op,
            None => {
                self.module_builder
                    .emit_error(MessagePosition::default(), "OpBranch missing target");
                return;
            }
        };
        let Some(target) = self.operand_as_id(operand, "branch target") else {
            return;
        };
        if let Err(error) = self.builder.branch(target) {
            self.emit_builder_error(error, operand.span().start());
        }
    }

    fn translate_branch_conditional(&mut self, instruction: &ParsedInstruction<'a>) {
        let mut operands = instruction.operands().iter();
        let Some(condition_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpBranchConditional missing condition",
            );
            return;
        };
        let Some(true_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpBranchConditional missing true label",
            );
            return;
        };
        let Some(false_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpBranchConditional missing false label",
            );
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
        if let Err(error) =
            self.builder
                .branch_conditional(condition_id, true_label, false_label, branch_weights)
        {
            self.emit_builder_error(error, MessagePosition::default());
        }
    }

    fn translate_variable(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpVariable missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpVariable missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(storage_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpVariable missing storage class",
            );
            return;
        };
        let Some(storage_class) =
            self.parse_enum_operand::<spirv::StorageClass>(Some(storage_operand), "storage class")
        else {
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
        let type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        let initializer_id = initializer;
        self.builder
            .variable(type_id, Some(result_id), storage_class, initializer_id);
    }

    fn translate_load(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpLoad missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpLoad missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(pointer_operand) = operands.next() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpLoad missing pointer operand");
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
        let type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst =
            dr::Instruction::new(spirv::Op::Load, Some(type_id), Some(result_id), dr_operands);
        self.push_block_instruction(inst);
    }

    fn translate_store(&mut self, instruction: &ParsedInstruction<'a>) {
        let mut operands = instruction.operands().iter();
        let Some(pointer_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "OpStore missing pointer operand",
            );
            return;
        };
        let Some(object_operand) = operands.next() else {
            self.module_builder
                .emit_error(MessagePosition::default(), "OpStore missing object operand");
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
        self.push_block_instruction(inst);
    }

    fn translate_binary_arithmetic(&mut self, instruction: &ParsedInstruction<'a>) {
        let Some(result_type) = instruction.result_type() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "Binary operation missing result type",
            );
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "Binary operation missing result id",
            );
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(lhs_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "Binary operation missing operands",
            );
            return;
        };
        let Some(rhs_operand) = operands.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
                "Binary operation requires two operands",
            );
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
        let type_id = self.module_builder.resolve_type_id(result_type);
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst = dr::Instruction::new(
            instruction.opcode(),
            Some(type_id),
            Some(result_id),
            vec![dr::Operand::IdRef(lhs_id), dr::Operand::IdRef(rhs_id)],
        );
        self.push_block_instruction(inst);
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

    fn push_block_instruction(&mut self, instruction: dr::Instruction) {
        if let Err(error) = self
            .builder
            .insert_into_block(InsertPoint::End, instruction)
        {
            self.emit_builder_error(error, MessagePosition::default());
        }
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

    fn parse_enum_operand<E>(
        &mut self,
        operand: Option<&ParsedOperand<'a>>,
        label: &str,
    ) -> Option<E>
    where
        E: FromStr,
    {
        let operand = match operand {
            Some(value) => value,
            None => {
                self.module_builder
                    .emit_error(MessagePosition::default(), format!("Missing {label}"));
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
                    additional_operands.push(self.expect_loop_control_literal(remaining, part)?);
                }
                "MinIterations" => {
                    control |= spirv::LoopControl::MIN_ITERATIONS;
                    additional_operands.push(self.expect_loop_control_literal(remaining, part)?);
                }
                "MaxIterations" => {
                    control |= spirv::LoopControl::MAX_ITERATIONS;
                    additional_operands.push(self.expect_loop_control_literal(remaining, part)?);
                }
                "IterationMultiple" => {
                    control |= spirv::LoopControl::ITERATION_MULTIPLE;
                    additional_operands.push(self.expect_loop_control_literal(remaining, part)?);
                }
                "PeelCount" => {
                    control |= spirv::LoopControl::PEEL_COUNT;
                    additional_operands.push(self.expect_loop_control_literal(remaining, part)?);
                }
                "PartialCount" => {
                    control |= spirv::LoopControl::PARTIAL_COUNT;
                    additional_operands.push(self.expect_loop_control_literal(remaining, part)?);
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
        label: &str,
    ) -> Option<dr::Operand> {
        let Some(operand) = remaining.next() else {
            self.module_builder.emit_error(
                MessagePosition::default(),
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
        let module = self.builder.module();
        let (diagnostics, _) = self.module_builder.finish();
        (module, diagnostics)
    }
}

/// Assembles a sequence of parsed instructions into a SPIR-V module, returning both the module and
/// any diagnostics emitted along the way.
pub fn assemble_instructions<'a>(
    instructions: &[&'a ParsedInstruction<'a>],
) -> (dr::Module, Vec<DiagnosticMessage<'static>>) {
    let mut translator = AssemblyTranslator::new();
    for instruction in instructions {
        translator.translate(instruction);
    }
    translator.finish()
}

fn literal_to_u32(literal: &LiteralNumber) -> u32 {
    match literal {
        LiteralNumber::Unsigned(value) => *value as u32,
        LiteralNumber::Signed(value) => *value as u32,
    }
}

/// Assembles a block of textual SPIR-V instructions separated by newlines into a binary module.
/// Returns the assembled words on success along with any diagnostics emitted along the way.
pub fn assemble_text(text: &str) -> (Option<Vec<u32>>, Vec<DiagnosticMessage<'static>>) {
    let mut translator = AssemblyTranslator::new();
    let mut diagnostics = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }

        match parse_instruction(line) {
            Ok(parsed) => translator.translate(&parsed),
            Err(error) => diagnostics.push(error.into_diagnostic()),
        }
    }

    let (module, translator_diagnostics) = translator.finish();
    diagnostics.extend(translator_diagnostics);

    if diagnostics.is_empty() {
        (Some(module.assemble()), diagnostics)
    } else {
        (None, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::{assemble_instructions, assemble_text, AssemblyTranslator};
    use crate::assembly::parser::parse_instruction;
    use rspirv::{dr, spirv};

    #[test]
    fn translator_emits_type_int_instruction() {
        let parsed = parse_instruction("%uint = OpTypeInt 32 0").expect("parse");
        let mut translator = AssemblyTranslator::new();
        translator.translate(&parsed);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        let inst = module
            .types_global_values
            .first()
            .expect("type instruction");
        assert_eq!(inst.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(inst.result_id, Some(1));
        assert_eq!(inst.operands.len(), 2);
    }

    #[test]
    fn translator_sets_memory_model() {
        let parsed = parse_instruction("OpMemoryModel Logical GLSL450").expect("parse");
        let mut translator = AssemblyTranslator::new();
        translator.translate(&parsed);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        let inst = module.memory_model.as_ref().expect("memory model");
        assert_eq!(inst.class.opcode, rspirv::spirv::Op::MemoryModel);
    }

    #[test]
    fn translator_emits_entry_point_instruction() {
        let parsed =
            parse_instruction("OpEntryPoint GLCompute %main \"main\" %a %b").expect("parse");
        let mut translator = AssemblyTranslator::new();
        translator.translate(&parsed);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        let inst = module.entry_points.first().expect("entry point");
        assert_eq!(inst.class.opcode, rspirv::spirv::Op::EntryPoint);
    }

    #[test]
    fn translator_emits_constant_instruction() {
        let type_inst = parse_instruction("%uint = OpTypeInt 32 0").unwrap();
        let const_inst = parse_instruction("%c32 = OpConstant %uint 32").unwrap();
        let mut translator = AssemblyTranslator::new();
        translator.translate(&type_inst);
        translator.translate(&const_inst);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        assert!(module
            .types_global_values
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::Constant));
    }

    #[test]
    fn assemble_instructions_streams_sequence() {
        let type_inst = parse_instruction("%uint = OpTypeInt 32 0").unwrap();
        let mem_model = parse_instruction("OpMemoryModel Logical GLSL450").unwrap();
        let (module, diagnostics) = assemble_instructions(&[&type_inst, &mem_model]);
        assert!(diagnostics.is_empty());
        assert!(module.memory_model.is_some());
    }

    #[test]
    fn assemble_text_parses_multiple_lines() {
        let text = "%uint = OpTypeInt 32 0\nOpMemoryModel Logical GLSL450";
        let (binary, diagnostics) = assemble_text(text);
        assert!(diagnostics.is_empty());
        assert!(binary.is_some());
    }

    #[test]
    fn assemble_text_emits_simple_function() {
        let text = "\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n\
OpMemoryModel Logical GLSL450\n\
%main = OpFunction %void None %void_fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd";
        let (binary, diagnostics) = assemble_text(text);
        assert!(diagnostics.is_empty());
        assert!(binary.is_some());
    }

    #[test]
    fn translator_handles_execution_mode_and_memory_ops() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint GLCompute %main \"main\" %buffer",
            "OpExecutionMode %main LocalSize 1 1 1",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%ptr = OpTypePointer StorageBuffer %uint",
            "%one = OpConstant %uint 1",
            "%buffer = OpVariable %ptr StorageBuffer",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%value = OpLoad %uint %buffer",
            "%sum = OpIAdd %uint %value %one",
            "OpStore %buffer %sum",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| {
                parse_instruction(line)
                    .unwrap_or_else(|err| panic!("failed to parse '{line}': {err:?}"))
            })
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        assert_eq!(module.capabilities.len(), 1);
        assert_eq!(module.execution_modes.len(), 1);
        assert_eq!(module.entry_points.len(), 1);
        let function = module.functions.first().expect("function");
        assert_eq!(function.blocks.len(), 1);
        let block = function.blocks.first().expect("entry block");
        assert!(block
            .instructions
            .iter()
            .any(|inst| inst.class.opcode == spirv::Op::IAdd));
    }

    #[test]
    fn translator_emits_memory_operands_for_load_store() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%ptr_ty = OpTypePointer StorageBuffer %uint",
            "%zero = OpConstant %uint 0",
            "%buffer = OpVariable %ptr_ty StorageBuffer",
            "%scope = OpConstant %uint 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%val = OpLoad %uint %buffer Aligned|MakePointerVisible 8 %scope",
            "OpStore %buffer %val Aligned|MakePointerAvailable 8 %scope",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("entry block");
        let load = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Load)
            .expect("load inst");
        assert!(matches!(
            load.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::MemoryAccess(mask),
                dr::Operand::LiteralBit32(8),
                dr::Operand::IdRef(_)
            ] if mask.contains(spirv::MemoryAccess::ALIGNED)
                && mask.contains(spirv::MemoryAccess::MAKE_POINTER_VISIBLE)
        ));
        let store = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Store)
            .expect("store inst");
        assert!(matches!(
            store.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::MemoryAccess(mask),
                dr::Operand::LiteralBit32(8),
                dr::Operand::IdRef(_)
            ] if mask.contains(spirv::MemoryAccess::ALIGNED)
                && mask.contains(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE)
        ));
    }

    #[test]
    fn translator_emits_access_chain_instruction() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%ptr_uint = OpTypePointer StorageBuffer %uint",
            "%ptr_ptr_uint = OpTypePointer StorageBuffer %ptr_uint",
            "%zero = OpConstant %uint 0",
            "%var = OpVariable %ptr_ptr_uint StorageBuffer",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%elem_ptr = OpAccessChain %ptr_uint %var %zero",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let access_chain = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::AccessChain)
            .expect("access chain instruction");
        assert!(matches!(
            access_chain.operands.as_slice(),
            [dr::Operand::IdRef(_), dr::Operand::IdRef(_)]
        ));
    }

    #[test]
    fn translator_handles_branch_instructions() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%one = OpConstant %uint 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpBranch %mid",
            "%mid = OpLabel",
            "OpBranchConditional %one %then %exit 1 2",
            "%then = OpLabel",
            "OpReturn",
            "%exit = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let all_insts: Vec<_> = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .collect();
        assert!(all_insts
            .iter()
            .any(|inst| inst.class.opcode == spirv::Op::Branch));
        let branch_cond = all_insts
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::BranchConditional)
            .expect("branch conditional inst");
        assert!(branch_cond.operands.len() >= 3);
    }

    #[test]
    fn translator_handles_copy_memory_operands() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%size = OpConstant %uint 4",
            "%ptr_fn = OpTypePointer Function %uint",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%dst = OpVariable %ptr_fn Function",
            "%src = OpVariable %ptr_fn Function",
            "OpCopyMemory %dst %src Aligned 4 Aligned 8",
            "OpCopyMemorySized %dst %src %size Aligned 4 Aligned 8",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let mut copies = block.instructions.iter().filter(|inst| {
            matches!(
                inst.class.opcode,
                spirv::Op::CopyMemory | spirv::Op::CopyMemorySized
            )
        });
        let copy = copies.next().expect("OpCopyMemory");
        assert!(matches!(
            copy.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::MemoryAccess(first),
                dr::Operand::LiteralBit32(4),
                dr::Operand::MemoryAccess(second),
                dr::Operand::LiteralBit32(8)
            ] if first.contains(spirv::MemoryAccess::ALIGNED)
                && second.contains(spirv::MemoryAccess::ALIGNED)
        ));
        let copy_sized = copies.next().expect("OpCopyMemorySized");
        assert!(matches!(
            copy_sized.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::MemoryAccess(first),
                dr::Operand::LiteralBit32(4),
                dr::Operand::MemoryAccess(second),
                dr::Operand::LiteralBit32(8)
            ] if first.contains(spirv::MemoryAccess::ALIGNED)
                && second.contains(spirv::MemoryAccess::ALIGNED)
        ));
    }

    #[test]
    fn translator_emits_selection_merge() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%one = OpConstant %uint 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpSelectionMerge %merge None",
            "OpBranchConditional %one %then %else",
            "%then = OpLabel",
            "OpBranch %merge",
            "%else = OpLabel",
            "OpBranch %merge",
            "%merge = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let selection_merge = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .find(|inst| inst.class.opcode == spirv::Op::SelectionMerge)
            .expect("selection merge");
        assert!(matches!(
            selection_merge.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::SelectionControl(control)
            ] if control.is_empty()
        ));
    }

    #[test]
    fn translator_emits_loop_merge_with_operands() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%one = OpConstant %uint 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpBranch %loop",
            "%loop = OpLabel",
            "OpLoopMerge %merge %continue MinIterations|PartialCount 4 2",
            "OpBranch %continue",
            "%continue = OpLabel",
            "OpBranchConditional %one %loop %merge",
            "%merge = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let loop_merge = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .find(|inst| inst.class.opcode == spirv::Op::LoopMerge)
            .expect("loop merge");
        assert!(matches!(
            loop_merge.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::LoopControl(control),
                dr::Operand::LiteralBit32(4),
                dr::Operand::LiteralBit32(2)
            ] if control.contains(spirv::LoopControl::MIN_ITERATIONS)
                && control.contains(spirv::LoopControl::PARTIAL_COUNT)
        ));
    }

    #[test]
    fn translator_emits_composite_construct() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%uint_0 = OpConstant %int 0",
            "%uint_1 = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%vec = OpCompositeConstruct %vec2 %uint_0 %uint_1",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let inst = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::CompositeConstruct)
            .expect("composite construct");
        assert_eq!(inst.operands.len(), 2);
    }

    #[test]
    fn translator_emits_vector_shuffle() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%vec4 = OpTypeVector %int 4",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%two = OpConstant %int 2",
            "%three = OpConstant %int 3",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%v1 = OpCompositeConstruct %vec2 %zero %one",
            "%v2 = OpCompositeConstruct %vec2 %two %three",
            "%shuffle = OpVectorShuffle %vec4 %v1 %v2 0 1 2 3",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let shuffle = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .find(|inst| inst.class.opcode == spirv::Op::VectorShuffle)
            .expect("vector shuffle");
        assert_eq!(shuffle.operands.len(), 6);
    }

    #[test]
    fn translator_emits_composite_extract_and_insert() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%v = OpCompositeConstruct %vec2 %zero %one",
            "%elem = OpCompositeExtract %int %v 1",
            "%result = OpCompositeInsert %vec2 %elem %v 0",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let (module, diagnostics) = assemble_instructions(&refs);
        assert!(diagnostics.is_empty());
        let function = module.functions.first().expect("function");
        let mut extract_seen = false;
        let mut insert_seen = false;
        for inst in function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
        {
            if inst.class.opcode == spirv::Op::CompositeExtract {
                extract_seen = true;
                assert!(matches!(
                    inst.operands.as_slice(),
                    [dr::Operand::IdRef(_), dr::Operand::LiteralBit32(1)]
                ));
            }
            if inst.class.opcode == spirv::Op::CompositeInsert {
                insert_seen = true;
                assert!(matches!(
                    inst.operands.as_slice(),
                    [
                        dr::Operand::IdRef(_),
                        dr::Operand::IdRef(_),
                        dr::Operand::LiteralBit32(0)
                    ]
                ));
            }
        }
        assert!(extract_seen && insert_seen);
    }
}
