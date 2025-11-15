use std::borrow::Cow;
use std::collections::{btree_map::Entry, BTreeMap};

use rspirv::dr;
use rspirv::spirv;

use super::instruction::{IdRef, LiteralNumber, ResultId, SpirvId, TypeId};
use super::parser::{OperandValue, ParsedInstruction};
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
                    let id = self.next_numeric_id;
                    self.next_numeric_id += 1;
                    entry.insert(id);
                    id
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
}
