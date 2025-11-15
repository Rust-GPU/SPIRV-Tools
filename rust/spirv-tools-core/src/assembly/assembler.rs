use std::borrow::Cow;
use std::collections::{btree_map::Entry, BTreeMap};
use std::str::FromStr;

use rspirv::binary::Assemble;
use rspirv::dr::{self, Error as BuildError};
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
            spirv::Op::TypeVoid => self.translate_type_void(instruction),
            spirv::Op::TypeInt => self.translate_type_int(instruction),
            spirv::Op::TypeFunction => self.translate_type_function(instruction),
            spirv::Op::MemoryModel => self.translate_memory_model(instruction),
            spirv::Op::EntryPoint => self.translate_entry_point(instruction),
            spirv::Op::Function => self.translate_function(instruction),
            spirv::Op::FunctionParameter => self.translate_function_parameter(instruction),
            spirv::Op::Label => self.translate_label(instruction),
            spirv::Op::Return => self.translate_return(),
            spirv::Op::FunctionEnd => self.translate_function_end(),
            spirv::Op::Constant => self.translate_constant(instruction),
            _ => self
                .module_builder
                .emit_error(MessagePosition::default(), "unsupported opcode"),
        }
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

    fn translate_function_end(&mut self) {
        if let Err(error) = self.builder.end_function() {
            self.emit_builder_error(error, MessagePosition::default());
        }
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
}
