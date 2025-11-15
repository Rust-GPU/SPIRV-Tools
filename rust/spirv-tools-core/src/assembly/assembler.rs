use std::borrow::Cow;
use std::collections::{btree_map::Entry, BTreeMap};
use std::str::FromStr;

use rspirv::dr;
use rspirv::spirv;

use super::instruction::{IdRef, LiteralNumber, ResultId, SpirvId, TypeId};
use super::parser::{OperandValue, ParsedInstruction, ParsedOperand};
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
            spirv::Op::TypeInt => self.translate_type_int(instruction),
            spirv::Op::MemoryModel => self.translate_memory_model(instruction),
            spirv::Op::EntryPoint => self.translate_entry_point(instruction),
            _ => self
                .module_builder
                .emit_error(MessagePosition::default(), "unsupported opcode"),
        }
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

    /// Finalizes the translation and returns the constructed module plus diagnostics.
    pub fn finish(self) -> (dr::Module, Vec<DiagnosticMessage<'static>>) {
        let module = self.builder.module();
        let (diagnostics, _) = self.module_builder.finish();
        (module, diagnostics)
    }
}

fn literal_to_u32(literal: &LiteralNumber) -> u32 {
    match literal {
        LiteralNumber::Unsigned(value) => *value as u32,
        LiteralNumber::Signed(value) => *value as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::AssemblyTranslator;
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
}
