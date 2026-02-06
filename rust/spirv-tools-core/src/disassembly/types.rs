use rspirv::binary::ParseAction;
use rspirv::dr::{self, Block, Function, Instruction, Module, ModuleHeader, Operand};
use rspirv::grammar::{GlslStd450InstructionTable, OpenCLStd100InstructionTable};
use rspirv::spirv;
use std::collections::HashMap;

use super::names::visit_module_instructions;
use super::unsupported_options;
use crate::assembly::{
    lookup_custom_ext_inst_name, BinaryToTextOptions, ExtInstImportInfo, ExtInstSetKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LiteralFormat {
    Decimal,
    Hexadecimal,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FormattingOptions {
    pub(super) suppress_header: bool,
    pub(super) show_byte_offsets: bool,
    pub(super) indent: bool,
    pub(super) friendly_names: bool,
    pub(super) nested_indent: bool,
    pub(super) comments: bool,
    pub(super) reorder_blocks: bool,
    pub(super) colorize: bool,
    pub(super) print_to_stdout: bool,
    pub(super) literal_format: LiteralFormat,
}

impl TryFrom<BinaryToTextOptions> for FormattingOptions {
    type Error = BinaryToTextOptions;

    fn try_from(bits: BinaryToTextOptions) -> Result<Self, Self::Error> {
        let unsupported = unsupported_options(bits);
        if !unsupported.is_empty() {
            return Err(unsupported);
        }
        let options = Self {
            suppress_header: bits.contains(BinaryToTextOptions::NO_HEADER),
            show_byte_offsets: bits.contains(BinaryToTextOptions::SHOW_BYTE_OFFSET),
            indent: bits.contains(BinaryToTextOptions::INDENT),
            friendly_names: bits.contains(BinaryToTextOptions::FRIENDLY_NAMES),
            nested_indent: bits.contains(BinaryToTextOptions::NESTED_INDENT),
            comments: bits.contains(BinaryToTextOptions::COMMENT),
            reorder_blocks: bits.contains(BinaryToTextOptions::REORDER_BLOCKS),
            colorize: bits.contains(BinaryToTextOptions::COLOR),
            print_to_stdout: bits.contains(BinaryToTextOptions::PRINT),
            literal_format: if bits.contains(BinaryToTextOptions::HEX) {
                LiteralFormat::Hexadecimal
            } else {
                LiteralFormat::Decimal
            },
        };
        Ok(options)
    }
}

#[derive(Default)]
pub(super) struct TypeTable {
    pub(super) entries: HashMap<u32, TypeInfo>,
}

impl TypeTable {
    pub(super) fn from_module(module: &dr::Module) -> Self {
        let mut entries = HashMap::new();
        for instruction in &module.types_global_values {
            let Some(result_id) = instruction.result_id else {
                continue;
            };
            match instruction.class.opcode {
                spirv::Op::TypeInt => {
                    if let (Some(width), Some(signed)) = (
                        instruction
                            .operands
                            .first()
                            .and_then(literal_operand_to_u32),
                        instruction.operands.get(1).and_then(literal_operand_to_u32),
                    ) {
                        entries.insert(
                            result_id,
                            TypeInfo::Int {
                                width,
                                signed: signed != 0,
                            },
                        );
                    }
                }
                spirv::Op::TypeFloat => {
                    if let Some(width) = instruction
                        .operands
                        .first()
                        .and_then(literal_operand_to_u32)
                    {
                        entries.insert(result_id, TypeInfo::Float { width });
                    }
                }
                _ => {}
            }
        }
        Self { entries }
    }

    pub(super) fn get(&self, id: u32) -> Option<&TypeInfo> {
        self.entries.get(&id)
    }
}

pub(super) struct ValueTypeTable {
    pub(super) entries: HashMap<u32, TypeInfo>,
}

impl ValueTypeTable {
    pub(super) fn from_module(module: &dr::Module, type_table: &TypeTable) -> Self {
        let mut entries = HashMap::new();
        visit_module_instructions(module, |instruction| {
            if let (Some(result_id), Some(result_type)) =
                (instruction.result_id, instruction.result_type)
            {
                if let Some(info) = type_table.get(result_type) {
                    entries.insert(result_id, *info);
                }
            }
        });
        Self { entries }
    }

    pub(super) fn get(&self, id: u32) -> Option<&TypeInfo> {
        self.entries.get(&id)
    }
}

pub(super) struct ExtendedLoader {
    pub(super) module: Module,
    pub(super) function: Option<Function>,
    pub(super) block: Option<Block>,
}

impl ExtendedLoader {
    pub(super) fn new() -> Self {
        Self {
            module: Module::new(),
            function: None,
            block: None,
        }
    }

    pub(super) fn into_module(self) -> Module {
        self.module
    }
}

macro_rules! if_ret_err {
    ($condition: expr, $error: expr) => {
        if $condition {
            return ParseAction::Error(Box::new($error));
        }
    };
}

impl rspirv::binary::Consumer for ExtendedLoader {
    fn initialize(&mut self) -> ParseAction {
        ParseAction::Continue
    }

    fn finalize(&mut self) -> ParseAction {
        if_ret_err!(self.block.is_some(), rspirv::dr::Error::UnclosedBlock);
        if_ret_err!(self.function.is_some(), rspirv::dr::Error::UnclosedFunction);
        ParseAction::Continue
    }

    fn consume_header(&mut self, header: ModuleHeader) -> ParseAction {
        self.module.header = Some(header);
        ParseAction::Continue
    }

    fn consume_instruction(&mut self, inst: Instruction) -> ParseAction {
        let opcode = inst.class.opcode;
        match opcode {
            spirv::Op::Capability => self.module.capabilities.push(inst),
            spirv::Op::Extension | spirv::Op::ConditionalExtensionINTEL => {
                self.module.extensions.push(inst)
            }
            spirv::Op::ExtInstImport => self.module.ext_inst_imports.push(inst),
            spirv::Op::MemoryModel => self.module.memory_model = Some(inst),
            spirv::Op::EntryPoint => self.module.entry_points.push(inst),
            spirv::Op::ExecutionMode => self.module.execution_modes.push(inst),
            spirv::Op::String
            | spirv::Op::SourceExtension
            | spirv::Op::Source
            | spirv::Op::SourceContinued => self.module.debug_string_source.push(inst),
            spirv::Op::Name | spirv::Op::MemberName => self.module.debug_names.push(inst),
            spirv::Op::ModuleProcessed => self.module.debug_module_processed.push(inst),
            opcode if rspirv::grammar::reflect::is_location_debug(opcode) => {
                match &mut self.block {
                    Some(block) => block.instructions.push(inst),
                    None => self.module.types_global_values.push(inst),
                }
            }
            opcode if opcode.is_annotation() => self.module.annotations.push(inst),
            opcode if opcode.is_type() || opcode.is_constant() => {
                self.module.types_global_values.push(inst)
            }
            spirv::Op::Variable if self.function.is_none() => {
                self.module.types_global_values.push(inst)
            }
            spirv::Op::Undef if self.function.is_none() => {
                self.module.types_global_values.push(inst)
            }
            spirv::Op::Function => {
                if_ret_err!(self.function.is_some(), rspirv::dr::Error::NestedFunction);
                let mut func = Function::new();
                func.def = Some(inst);
                self.function = Some(func);
            }
            spirv::Op::FunctionEnd => {
                if_ret_err!(
                    self.function.is_none(),
                    rspirv::dr::Error::MismatchedFunctionEnd
                );
                if_ret_err!(self.block.is_some(), rspirv::dr::Error::UnclosedBlock);
                self.function.as_mut().unwrap().end = Some(inst);
                self.module.functions.push(self.function.take().unwrap());
            }
            spirv::Op::FunctionParameter => {
                if_ret_err!(
                    self.function.is_none(),
                    rspirv::dr::Error::DetachedFunctionParameter
                );
                self.function.as_mut().unwrap().parameters.push(inst);
            }
            spirv::Op::Label => {
                if_ret_err!(self.function.is_none(), rspirv::dr::Error::DetachedBlock);
                if_ret_err!(self.block.is_some(), rspirv::dr::Error::NestedBlock);
                let mut block = Block::new();
                block.label = Some(inst);
                self.block = Some(block);
            }
            opcode if rspirv::grammar::reflect::is_block_terminator(opcode) => {
                if_ret_err!(
                    self.block.is_none(),
                    rspirv::dr::Error::MismatchedTerminator
                );
                self.block.as_mut().unwrap().instructions.push(inst);
                self.function
                    .as_mut()
                    .unwrap()
                    .blocks
                    .push(self.block.take().unwrap());
            }
            _ => {
                if let Some(block) = self.block.as_mut() {
                    block.instructions.push(inst);
                } else {
                    self.module.types_global_values.push(inst);
                }
            }
        }
        ParseAction::Continue
    }
}

pub(super) struct ExtInstTable {
    pub(super) imports: HashMap<u32, ExtInstImportInfo>,
}

impl ExtInstTable {
    pub(super) fn from_module(module: &dr::Module) -> Self {
        let mut imports = HashMap::new();
        for instruction in &module.ext_inst_imports {
            if let Some(id) = instruction.result_id {
                let name = instruction
                    .operands
                    .iter()
                    .find_map(|operand| match operand {
                        Operand::LiteralString(value) => Some(value.as_str()),
                        _ => None,
                    })
                    .unwrap_or("");
                imports.insert(id, ExtInstImportInfo::new(name));
            }
        }
        Self { imports }
    }

    pub(super) fn lookup_name(&self, set_id: u32, opcode: u32) -> Option<&'static str> {
        let inst = match self.imports.get(&set_id).map(|info| info.kind) {
            Some(ExtInstSetKind::GlslStd450) => {
                GlslStd450InstructionTable::iter().find(|inst| inst.opcode == opcode)
            }
            Some(ExtInstSetKind::OpenClStd100) => {
                OpenCLStd100InstructionTable::iter().find(|inst| inst.opcode == opcode)
            }
            Some(ExtInstSetKind::ArmMotionEngine100) => {
                return self
                    .imports
                    .get(&set_id)
                    .and_then(|info| lookup_custom_ext_inst_name(&info.name, opcode));
            }
            _ => None,
        }?;
        Some(inst.opname)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum TypeInfo {
    Int { width: u32, signed: bool },
    Float { width: u32 },
}

pub(super) fn literal_operand_to_u32(operand: &Operand) -> Option<u32> {
    match operand {
        Operand::LiteralBit32(value) => Some(*value),
        Operand::FPEncoding(value) => Some(*value as u32),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BlockPosition {
    Global,
    Label,
    Body,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ModuleSection {
    Header,
    Debug,
    Annotations,
    Types,
    Functions,
}

pub(super) struct InstructionRecord<'a> {
    pub(super) instruction: &'a Instruction,
    pub(super) depth: u32,
    pub(super) position: BlockPosition,
    pub(super) section: ModuleSection,
}

impl<'a> InstructionRecord<'a> {
    pub(super) fn new(
        instruction: &'a Instruction,
        depth: u32,
        position: BlockPosition,
        section: ModuleSection,
    ) -> Self {
        Self {
            instruction,
            depth,
            position,
            section,
        }
    }

    pub(super) fn instruction(&self) -> &'a Instruction {
        self.instruction
    }

    pub(super) fn is_block_body(&self) -> bool {
        matches!(self.position, BlockPosition::Body)
    }
}
