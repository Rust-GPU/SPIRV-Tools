use rspirv::binary::{Disassemble, ParseAction, Parser};
use rspirv::dr::{self, Block, Function, Instruction, Module, ModuleHeader, Operand};
use rspirv::grammar::{GlslStd450InstructionTable, OpenCLStd100InstructionTable};
use rspirv::spirv;
use std::collections::{HashMap, HashSet};
use std::mem::size_of_val;
use std::num::FpCategory;
use std::slice;
use thiserror::Error;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

use crate::assembly::{
    lookup_custom_ext_inst_name, BinaryToTextOptions, ExtInstImportInfo, ExtInstSetKind,
};
use crate::diagnostic::{DiagnosticMessage, MessagePosition};
use crate::message::MessageLevel;
use crate::string_literal::render_string_literal;

const SUPPORTED_OPTION_BITS: u32 = BinaryToTextOptions::NO_HEADER.bits()
    | BinaryToTextOptions::PRINT.bits()
    | BinaryToTextOptions::SHOW_BYTE_OFFSET.bits()
    | BinaryToTextOptions::INDENT.bits()
    | BinaryToTextOptions::FRIENDLY_NAMES.bits()
    | BinaryToTextOptions::NESTED_INDENT.bits()
    | BinaryToTextOptions::COMMENT.bits()
    | BinaryToTextOptions::COLOR.bits()
    | BinaryToTextOptions::HEX.bits()
    | BinaryToTextOptions::REORDER_BLOCKS.bits();
const HEADER_WORD_COUNT: usize = 5;
const INDEX_MAGIC_NUMBER: usize = 0;
const INDEX_VERSION: usize = 1;
const INDEX_GENERATOR: usize = 2;
const INDEX_BOUND: usize = 3;
const INDEX_SCHEMA: usize = 4;
const COMMENT_COLUMN: usize = 50;
const MAX_COMMENT_ALIGN: usize = 256;
const STANDARD_INDENT_COLUMN: usize = 15;
const BLOCK_NEST_INDENT: usize = 2;
const BLOCK_BODY_INDENT_OFFSET: usize = 2;
const COLOR_BLUE: &str = "\x1b[34m";
const COLOR_GREY: &str = "\x1b[90m";
const COLOR_RESET: &str = "\x1b[0m";

#[cfg(test)]
static PRINT_LOG: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));

#[cfg(test)]
const FRIENDLY_NAME_SAMPLE_BINARY: &[u32] = &[
    0x07230203, 0x00010600, 0x00070000, 0x0000000c, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
    0x00000000, 0x00000001, 0x00040015, 0x00000001, 0x00000020, 0x00000000, 0x000b001e, 0x00000002,
    0x00000001, 0x00000003, 0x00000004, 0x00000005, 0x00000006, 0x00000007, 0x00000008, 0x00000009,
    0x0000000a, 0x0004002b, 0x00000001, 0x0000000b, 0x0000002a,
];

/// Errors that can be produced while disassembling SPIR-V binaries.
#[derive(Debug, Error)]
pub enum DisassemblyError {
    /// The binary failed to parse into a valid module.
    #[error("failed to parse SPIR-V binary: {message}")]
    Parse {
        /// Human-readable summary describing the parse failure.
        message: String,
        /// Structured diagnostics suitable for forwarding to message consumers.
        diagnostics: Vec<DiagnosticMessage<'static>>,
    },
    /// The requested disassembly options are not supported by the Rust backend yet.
    #[error("unsupported binary-to-text options: {0:?}")]
    Unsupported(BinaryToTextOptions),
}

impl DisassemblyError {
    fn parse_error(message: String) -> Self {
        let diagnostic = DiagnosticMessage::new(
            MessageLevel::Error,
            MessagePosition::default(),
            message.clone(),
        )
        .with_source("disassembler");
        Self::Parse {
            message,
            diagnostics: vec![diagnostic],
        }
    }

    /// Returns the diagnostics describing this error, if any.
    pub fn diagnostics(&self) -> &[DiagnosticMessage<'static>] {
        match self {
            DisassemblyError::Parse { diagnostics, .. } => diagnostics,
            DisassemblyError::Unsupported(_) => &[],
        }
    }
}

/// Disassembles the provided SPIR-V binary words into textual assembly.
pub fn disassemble_binary(
    words: &[u32],
    options: BinaryToTextOptions,
) -> Result<String, DisassemblyError> {
    let effective_bits = options.bits() & !BinaryToTextOptions::NONE.bits();
    let requested = BinaryToTextOptions::from_bits_truncate(effective_bits);
    let formatting =
        FormattingOptions::try_from(requested).map_err(DisassemblyError::Unsupported)?;

    let mut loader = ExtendedLoader::new();
    match Parser::new(words_as_bytes(words), &mut loader).parse() {
        Ok(()) => {}
        Err(rspirv::binary::ParseState::OpcodeUnknown(_, _, _)) => {
            return Err(DisassemblyError::Unsupported(BinaryToTextOptions::empty()))
        }
        Err(error) => return Err(DisassemblyError::parse_error(error.to_string())),
    }
    let mut module = loader.into_module();
    update_module_header(&mut module, words);
    let offsets = collect_instruction_offsets(words);
    let mut text = render_module_text(&module, words, &offsets, &formatting);
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    if formatting.print_to_stdout {
        emit_disassembly_text(&text);
        text.clear();
    }
    Ok(text)
}

fn words_as_bytes(words: &[u32]) -> &[u8] {
    unsafe { slice::from_raw_parts(words.as_ptr() as *const u8, size_of_val(words)) }
}

fn update_module_header(module: &mut rspirv::dr::Module, words: &[u32]) {
    if words.len() < HEADER_WORD_COUNT {
        return;
    }
    let header = module
        .header
        .get_or_insert_with(|| ModuleHeader::new(words[INDEX_BOUND]));
    header.magic_number = words[INDEX_MAGIC_NUMBER];
    header.version = words[INDEX_VERSION];
    header.generator = words[INDEX_GENERATOR];
    header.bound = words[INDEX_BOUND];
    header.reserved_word = words[INDEX_SCHEMA];
}

fn unsupported_options(options: BinaryToTextOptions) -> BinaryToTextOptions {
    BinaryToTextOptions::from_bits_truncate(options.bits() & !SUPPORTED_OPTION_BITS)
}

/// Returns true if the requested options can be handled by the Rust disassembler.
pub fn supports_options(options: BinaryToTextOptions) -> bool {
    let effective = BinaryToTextOptions::from_bits_truncate(options.bits());
    unsupported_options(effective).is_empty()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiteralFormat {
    Decimal,
    Hexadecimal,
}

#[derive(Clone, Copy, Debug)]
struct FormattingOptions {
    suppress_header: bool,
    show_byte_offsets: bool,
    indent: bool,
    friendly_names: bool,
    nested_indent: bool,
    comments: bool,
    reorder_blocks: bool,
    colorize: bool,
    print_to_stdout: bool,
    literal_format: LiteralFormat,
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
struct TypeTable {
    entries: HashMap<u32, TypeInfo>,
}

impl TypeTable {
    fn from_module(module: &dr::Module) -> Self {
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

    fn get(&self, id: u32) -> Option<&TypeInfo> {
        self.entries.get(&id)
    }
}

struct ValueTypeTable {
    entries: HashMap<u32, TypeInfo>,
}

impl ValueTypeTable {
    fn from_module(module: &dr::Module, type_table: &TypeTable) -> Self {
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

    fn get(&self, id: u32) -> Option<&TypeInfo> {
        self.entries.get(&id)
    }
}

struct ExtendedLoader {
    module: Module,
    function: Option<Function>,
    block: Option<Block>,
}

impl ExtendedLoader {
    fn new() -> Self {
        Self {
            module: Module::new(),
            function: None,
            block: None,
        }
    }

    fn into_module(self) -> Module {
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

struct ExtInstTable {
    imports: HashMap<u32, ExtInstImportInfo>,
}

impl ExtInstTable {
    fn from_module(module: &dr::Module) -> Self {
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

    fn lookup_name(&self, set_id: u32, opcode: u32) -> Option<&'static str> {
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
enum TypeInfo {
    Int { width: u32, signed: bool },
    Float { width: u32 },
}

fn literal_operand_to_u32(operand: &Operand) -> Option<u32> {
    match operand {
        Operand::LiteralBit32(value) => Some(*value),
        Operand::FPEncoding(value) => Some(*value as u32),
        _ => None,
    }
}

fn emit_disassembly_text(text: &str) {
    #[cfg(test)]
    {
        PRINT_LOG.lock().unwrap().push(text.to_string());
    }
    #[cfg(not(test))]
    {
        use std::io::{self, Write};
        let mut stdout = io::stdout();
        let _ = stdout.write_all(text.as_bytes());
        let _ = stdout.flush();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockPosition {
    Global,
    Label,
    Body,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModuleSection {
    Header,
    Debug,
    Annotations,
    Types,
    Functions,
}

struct InstructionRecord<'a> {
    instruction: &'a Instruction,
    depth: u32,
    position: BlockPosition,
    section: ModuleSection,
}

impl<'a> InstructionRecord<'a> {
    fn new(
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

    fn instruction(&self) -> &'a Instruction {
        self.instruction
    }

    fn is_block_body(&self) -> bool {
        matches!(self.position, BlockPosition::Body)
    }
}

fn render_module_text(
    module: &dr::Module,
    words: &[u32],
    offsets: &[u32],
    options: &FormattingOptions,
) -> String {
    let instructions = collect_module_instructions(module, words, options.reorder_blocks);
    let type_table = TypeTable::from_module(module);
    let value_types = ValueTypeTable::from_module(module, &type_table);
    let ext_inst_table = ExtInstTable::from_module(module);
    let friendly_names = if options.friendly_names {
        let table = FriendlyNameTable::from_module(module, &type_table);
        if table.is_empty() {
            None
        } else {
            Some(table)
        }
    } else {
        None
    };
    let mut comment_collector = CommentCollector::new(options.comments);
    let mut text = String::new();
    if !options.suppress_header {
        text.push_str(&render_header(module));
    }
    text.push_str(&render_instructions(
        &instructions,
        offsets,
        options,
        &type_table,
        &ext_inst_table,
        &value_types,
        friendly_names.as_ref(),
        &mut comment_collector,
    ));
    text
}

fn render_header(module: &dr::Module) -> String {
    let header = module
        .header
        .as_ref()
        .cloned()
        .unwrap_or_else(|| ModuleHeader::new(0));
    let (major, minor) = header.version();
    let (vendor, version) = header.generator();
    let mut text = String::new();
    text.push_str("; SPIR-V\n");
    text.push_str(&format!("; Version: {}.{}\n", major, minor));
    text.push_str(&format!(
        "; Generator: {}{}; {}\n",
        vendor_prefix(vendor),
        vendor,
        version
    ));
    text.push_str(&format!("; Bound: {}\n", header.bound));
    text.push_str(&format!("; Schema: {}\n", header.reserved_word));
    text
}

fn vendor_prefix(name: &str) -> &str {
    match name {
        "SPIR-V Tools Assembler" => "Khronos ",
        "SPIR-V Tools Linker" => "Khronos ",
        _ => "",
    }
}

#[allow(clippy::too_many_arguments)]
fn render_instructions(
    instructions: &[InstructionRecord],
    offsets: &[u32],
    options: &FormattingOptions,
    type_table: &TypeTable,
    ext_inst_table: &ExtInstTable,
    value_types: &ValueTypeTable,
    friendly_names: Option<&FriendlyNameTable>,
    comment_collector: &mut CommentCollector,
) -> String {
    let mut aligner = CommentAligner::new();
    let mut text = String::new();
    let mut current_section = ModuleSection::Header;
    for (index, record) in instructions.iter().enumerate() {
        let offset = offsets.get(index).copied().unwrap_or(0);
        let instruction = record.instruction();
        if options.comments {
            comment_collector.observe(instruction);
            if record.section != current_section {
                current_section = record.section;
                if let Some(heading) = section_heading(record.section) {
                    append_section_heading(&mut text, heading, options.indent);
                    aligner.reset();
                }
            }
        }
        if options.nested_indent
            && matches!(record.position, BlockPosition::Label)
            && !text.is_empty()
        {
            text.push('\n');
        }
        let mut line = sanitize_line(disassemble_with_format(
            instruction,
            options.literal_format,
            type_table,
            ext_inst_table,
            value_types,
        ));
        if options.comments && instruction.class.opcode == spirv::Op::Function {
            append_function_heading(
                &mut text,
                instruction.result_id,
                options.indent,
                options.nested_indent,
            );
            aligner.reset();
        }
        apply_friendly_names(&mut line, friendly_names);

        if options.indent {
            apply_indent(&mut line, 0);
        }

        if options.nested_indent {
            let mut block_indent = (record.depth as usize) * BLOCK_NEST_INDENT;
            if record.is_block_body() {
                block_indent += BLOCK_BODY_INDENT_OFFSET;
            }
            insert_block_indent(&mut line, block_indent);
        }
        let mut comment_parts = Vec::new();
        if options.show_byte_offsets {
            comment_parts.push(format!("0x{offset:08x}"));
        }
        if options.comments {
            if let Some(comment) = comment_collector.inline_comment(instruction) {
                comment_parts.push(comment);
            }
            if let Some(comment) = comment_collector.result_comment(instruction) {
                comment_parts.push(comment);
            }
        }

        if comment_parts.is_empty() {
            aligner.reset();
        } else {
            let joined = comment_parts.join(", ");
            aligner.append_comment(&mut line, &joined);
        }

        apply_color_formatting(&mut line, options.colorize);

        text.push_str(&line);
        if index + 1 < instructions.len() || !line.is_empty() {
            text.push('\n');
        }
    }
    text
}

fn sanitize_line(mut line: String) -> String {
    if line.ends_with('\n') {
        line.pop();
    }
    line
}

fn disassemble_with_format(
    instruction: &Instruction,
    literal_format: LiteralFormat,
    type_table: &TypeTable,
    ext_inst_table: &ExtInstTable,
    value_types: &ValueTypeTable,
) -> String {
    let operands = if instruction.class.opcode == spirv::Op::Switch {
        format_switch_operands(instruction, literal_format, value_types)
    } else {
        format_ext_inst_operands(instruction, literal_format, ext_inst_table)
            .or_else(|| format_constant_operands(instruction, literal_format, type_table))
            .unwrap_or_else(|| disassemble_operands(&instruction.operands, literal_format))
    };
    let mut line = String::new();
    if let Some(result_id) = instruction.result_id {
        line.push('%');
        line.push_str(&result_id.to_string());
        line.push_str(" = ");
    }
    line.push_str("Op");
    line.push_str(instruction.class.opname);
    if let Some(result_type) = instruction.result_type {
        line.push(' ');
        line.push('%');
        line.push_str(&result_type.to_string());
    }
    if !operands.is_empty() {
        line.push(' ');
        line.push_str(&operands);
    }
    line
}

fn disassemble_operands(operands: &[Operand], literal_format: LiteralFormat) -> String {
    operands
        .iter()
        .map(|operand| format_operand(operand, literal_format))
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_constant_operands(
    instruction: &Instruction,
    literal_format: LiteralFormat,
    type_table: &TypeTable,
) -> Option<String> {
    if literal_format == LiteralFormat::Hexadecimal {
        return None;
    }
    match instruction.class.opcode {
        spirv::Op::Constant | spirv::Op::SpecConstant => {}
        _ => return None,
    }
    let type_id = instruction.result_type?;
    let literal = instruction.operands.first()?;
    let type_info = type_table.get(type_id)?;
    match type_info {
        TypeInfo::Int { width, signed } => format_integer_literal(literal, *width, *signed),
        TypeInfo::Float { width } => format_float_literal(literal, *width),
    }
}

fn format_switch_operands(
    instruction: &Instruction,
    literal_format: LiteralFormat,
    value_types: &ValueTypeTable,
) -> String {
    if instruction.operands.len() < 2 {
        return disassemble_operands(&instruction.operands, literal_format);
    }
    let selector = &instruction.operands[0];
    let default_label = &instruction.operands[1];
    let selector_text = format_operand(selector, literal_format);
    let default_text = format_operand(default_label, literal_format);
    let selector_type = extract_id_ref(selector)
        .and_then(|id| value_types.get(id))
        .copied();
    let mut parts = vec![selector_text, default_text];
    let mut index = 2;
    while index + 1 < instruction.operands.len() {
        let literal = &instruction.operands[index];
        let label = &instruction.operands[index + 1];
        let literal_text = if literal_format == LiteralFormat::Hexadecimal {
            format_operand(literal, literal_format)
        } else if let Some(type_info) = selector_type {
            format_integer_from_type(literal, type_info)
                .unwrap_or_else(|| format_operand(literal, literal_format))
        } else {
            format_operand(literal, literal_format)
        };
        parts.push(literal_text);
        parts.push(format_operand(label, literal_format));
        index += 2;
    }
    parts.join(" ")
}

fn format_integer_from_type(operand: &Operand, info: TypeInfo) -> Option<String> {
    match info {
        TypeInfo::Int { width, signed } => {
            literal_operand_bits(operand).map(|bits| format_integer_bits(bits, width, signed))
        }
        _ => None,
    }
}

fn format_ext_inst_operands(
    instruction: &Instruction,
    literal_format: LiteralFormat,
    ext_inst_table: &ExtInstTable,
) -> Option<String> {
    if instruction.class.opcode != spirv::Op::ExtInst {
        return None;
    }
    if instruction.operands.len() < 2 {
        return None;
    }
    let mut parts = Vec::with_capacity(instruction.operands.len());
    let set_operand = &instruction.operands[0];
    parts.push(format_operand(set_operand, literal_format));
    let opcode_operand = &instruction.operands[1];
    let opcode_text = match (extract_id_ref(set_operand), opcode_operand) {
        (Some(set_id), Operand::LiteralExtInstInteger(value)) => {
            if let Some(name) = ext_inst_table.lookup_name(set_id, *value) {
                name.to_string()
            } else {
                format_operand(opcode_operand, literal_format)
            }
        }
        _ => format_operand(opcode_operand, literal_format),
    };
    parts.push(opcode_text);
    for operand in instruction.operands.iter().skip(2) {
        parts.push(format_operand(operand, literal_format));
    }
    Some(parts.join(" "))
}

fn format_integer_literal(operand: &Operand, width: u32, signed: bool) -> Option<String> {
    let bits = match operand {
        Operand::LiteralBit32(value) => u64::from(*value),
        Operand::LiteralBit64(value) => *value,
        _ => return None,
    };
    Some(format_integer_bits(bits, width, signed))
}

fn format_integer_bits(bits: u64, width: u32, signed: bool) -> String {
    if signed {
        if width >= 64 {
            (bits as i64).to_string()
        } else {
            let shift = 64 - width;
            let value = ((bits << shift) as i64) >> shift;
            value.to_string()
        }
    } else if width >= 64 {
        bits.to_string()
    } else {
        let mask = (1u64 << width) - 1;
        (bits & mask).to_string()
    }
}

fn format_float_literal(operand: &Operand, width: u32) -> Option<String> {
    match width {
        16 => {
            let bits = match operand {
                Operand::LiteralBit32(value) => u64::from(*value & 0xffff),
                _ => return None,
            };
            Some(format_hex_float(bits, &HEX_FLOAT_F16))
        }
        32 => {
            let bits = match operand {
                Operand::LiteralBit32(value) => *value,
                _ => return None,
            };
            Some(format_f32_literal(bits))
        }
        64 => {
            let bits = match operand {
                Operand::LiteralBit64(value) => *value,
                _ => return None,
            };
            Some(format_f64_literal(bits))
        }
        _ => None,
    }
}

fn format_f32_literal(bits: u32) -> String {
    let value = f32::from_bits(bits);
    match value.classify() {
        FpCategory::Zero | FpCategory::Normal => {
            format_decimal_float(f64::from(value), F32_DECIMAL_DIGITS)
        }
        _ => format_hex_float(u64::from(bits), &HEX_FLOAT_F32),
    }
}

fn format_f64_literal(bits: u64) -> String {
    let value = f64::from_bits(bits);
    match value.classify() {
        FpCategory::Zero | FpCategory::Normal => format_decimal_float(value, F64_DECIMAL_DIGITS),
        _ => format_hex_float(bits, &HEX_FLOAT_F64),
    }
}

fn format_decimal_float(value: f64, digits: usize) -> String {
    // Use pure Rust formatting - format with specified precision then trim trailing zeros
    // This mimics C's %g format specifier behavior
    let formatted = format!("{:.prec$}", value, prec = digits);

    // If it contains a decimal point, trim trailing zeros (like %g does)
    if formatted.contains('.') {
        let trimmed = formatted.trim_end_matches('0');
        // Don't leave a trailing decimal point
        let trimmed = trimmed.trim_end_matches('.');
        trimmed.to_string()
    } else {
        formatted
    }
}

struct HexFloatConfig {
    total_bits: u32,
    exponent_bits: u32,
    fraction_bits: u32,
    exponent_bias: i32,
}

impl HexFloatConfig {
    const fn fraction_nibbles(&self) -> u32 {
        self.fraction_bits.div_ceil(4)
    }

    const fn overflow_bits(&self) -> u32 {
        self.fraction_nibbles() * 4 - self.fraction_bits
    }

    const fn fraction_mask(&self) -> u64 {
        if self.fraction_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << self.fraction_bits) - 1
        }
    }
}

const HEX_FLOAT_F16: HexFloatConfig = HexFloatConfig {
    total_bits: 16,
    exponent_bits: 5,
    fraction_bits: 10,
    exponent_bias: 15,
};
const HEX_FLOAT_F32: HexFloatConfig = HexFloatConfig {
    total_bits: 32,
    exponent_bits: 8,
    fraction_bits: 23,
    exponent_bias: 127,
};
const HEX_FLOAT_F64: HexFloatConfig = HexFloatConfig {
    total_bits: 64,
    exponent_bits: 11,
    fraction_bits: 52,
    exponent_bias: 1023,
};
const F32_DECIMAL_DIGITS: usize = 9;
const F64_DECIMAL_DIGITS: usize = 17;

fn format_hex_float(bits: u64, config: &HexFloatConfig) -> String {
    let sign_mask = 1u64 << (config.total_bits - 1);
    let exponent_mask = if config.exponent_bits >= 64 {
        u64::MAX
    } else {
        ((1u64 << config.exponent_bits) - 1) << config.fraction_bits
    };
    let fraction_mask = config.fraction_mask();
    let sign = (bits & sign_mask) != 0;
    let raw_exponent = (bits & exponent_mask) >> config.fraction_bits;
    let mut fraction = (bits & fraction_mask) << config.overflow_bits();
    let is_zero = raw_exponent == 0 && fraction == 0;
    let is_denorm = raw_exponent == 0 && !is_zero;
    let mut exponent = if is_zero {
        0
    } else {
        raw_exponent as i64 - config.exponent_bias as i64
    };
    if is_denorm && config.fraction_bits + config.overflow_bits() > 0 {
        let top_bit = 1u64 << (config.fraction_bits + config.overflow_bits() - 1);
        while (fraction & top_bit) == 0 {
            fraction <<= 1;
            exponent -= 1;
        }
        fraction <<= 1;
        let mask = (1u64 << (config.fraction_bits + config.overflow_bits())) - 1;
        fraction &= mask;
    }
    let mut fraction_nibbles = config.fraction_nibbles() as usize;
    while fraction_nibbles > 0 && (fraction & 0xF) == 0 {
        fraction >>= 4;
        fraction_nibbles -= 1;
    }
    let mut text = String::new();
    if sign {
        text.push('-');
    }
    text.push_str("0x");
    text.push(if is_zero { '0' } else { '1' });
    if fraction_nibbles > 0 {
        text.push('.');
        let frac_str = format!("{:x}", fraction);
        if frac_str.len() < fraction_nibbles {
            let padding = fraction_nibbles - frac_str.len();
            for _ in 0..padding {
                text.push('0');
            }
        }
        text.push_str(&frac_str);
    }
    text.push('p');
    if exponent >= 0 {
        text.push('+');
    }
    text.push_str(&exponent.to_string());
    text
}

fn format_operand(operand: &Operand, literal_format: LiteralFormat) -> String {
    match (literal_format, operand) {
        (LiteralFormat::Hexadecimal, Operand::LiteralBit32(value)) => {
            format!("0x{value:08x}")
        }
        (LiteralFormat::Hexadecimal, Operand::LiteralBit64(value)) => {
            format!("0x{value:016x}")
        }
        (LiteralFormat::Hexadecimal, Operand::LiteralExtInstInteger(value)) => {
            format!("0x{value:x}")
        }
        (_, Operand::ExecutionModel(model)) => {
            if let Some(name) = canonical_execution_model(*model) {
                name.to_string()
            } else {
                operand.disassemble()
            }
        }
        (_, Operand::LiteralString(value)) => format_literal_string(value),
        (_, Operand::StorageClass(class)) => {
            if let Some(name) = canonical_storage_class(*class) {
                name.to_string()
            } else {
                operand.disassemble()
            }
        }
        (_, Operand::MemoryAccess(_)) | (_, Operand::MemorySemantics(_)) => {
            normalize_mask_string(&operand.disassemble())
        }
        _ => operand.disassemble(),
    }
}

fn normalize_mask_string(raw: &str) -> String {
    let mut seen = HashSet::new();
    let mut parts = Vec::new();
    for token in raw.split('|') {
        let trimmed = token.trim();
        let canonical = match trimmed {
            "MakePointerVisibleKHR" => "MakePointerVisible",
            "MakePointerAvailableKHR" => "MakePointerAvailable",
            "NonPrivatePointerKHR" => "NonPrivatePointer",
            "AcquireReleaseKHR" => "AcquireRelease",
            "AcquireKHR" => "Acquire",
            "ReleaseKHR" => "Release",
            other => other,
        };
        if seen.insert(canonical) {
            if canonical == trimmed {
                parts.push(trimmed.to_string());
            } else {
                parts.push(canonical.to_string());
            }
        }
    }
    parts.join("|")
}

fn format_literal_string(value: &str) -> String {
    render_string_literal(value)
}

fn canonical_execution_model(model: spirv::ExecutionModel) -> Option<&'static str> {
    use spirv::ExecutionModel;
    if model == ExecutionModel::RayGenerationNV || model == ExecutionModel::RayGenerationKHR {
        Some("RayGenerationKHR")
    } else if model == ExecutionModel::IntersectionNV || model == ExecutionModel::IntersectionKHR {
        Some("IntersectionKHR")
    } else if model == ExecutionModel::AnyHitNV || model == ExecutionModel::AnyHitKHR {
        Some("AnyHitKHR")
    } else if model == ExecutionModel::ClosestHitNV || model == ExecutionModel::ClosestHitKHR {
        Some("ClosestHitKHR")
    } else if model == ExecutionModel::MissNV || model == ExecutionModel::MissKHR {
        Some("MissKHR")
    } else if model == ExecutionModel::CallableNV || model == ExecutionModel::CallableKHR {
        Some("CallableKHR")
    } else {
        None
    }
}

fn canonical_storage_class(class: spirv::StorageClass) -> Option<&'static str> {
    use spirv::StorageClass;
    if class == StorageClass::CallableDataNV || class == StorageClass::CallableDataKHR {
        Some("CallableDataKHR")
    } else if class == StorageClass::IncomingCallableDataNV
        || class == StorageClass::IncomingCallableDataKHR
    {
        Some("IncomingCallableDataKHR")
    } else if class == StorageClass::RayPayloadNV || class == StorageClass::RayPayloadKHR {
        Some("RayPayloadKHR")
    } else if class == StorageClass::HitAttributeNV || class == StorageClass::HitAttributeKHR {
        Some("HitAttributeKHR")
    } else if class == StorageClass::IncomingRayPayloadNV
        || class == StorageClass::IncomingRayPayloadKHR
    {
        Some("IncomingRayPayloadKHR")
    } else if class == StorageClass::ShaderRecordBufferNV
        || class == StorageClass::ShaderRecordBufferKHR
    {
        Some("ShaderRecordBufferKHR")
    } else if class == StorageClass::PhysicalStorageBuffer
        || class == StorageClass::PhysicalStorageBufferEXT
    {
        Some("PhysicalStorageBuffer")
    } else {
        None
    }
}

fn apply_friendly_names(line: &mut String, names: Option<&FriendlyNameTable>) {
    let Some(table) = names else {
        return;
    };
    if table.is_empty() {
        return;
    }

    let mut rewritten = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_string = false;

    while let Some(ch) = chars.next() {
        if ch == '"' {
            in_string = !in_string;
            rewritten.push(ch);
            continue;
        }

        if in_string {
            rewritten.push(ch);
            if ch == '\\' {
                if let Some(next) = chars.next() {
                    rewritten.push(next);
                }
            }
            continue;
        }

        if ch == '%' {
            let mut digits = String::new();
            while let Some(&next) = chars.peek() {
                if next.is_ascii_digit() {
                    digits.push(next);
                    chars.next();
                } else {
                    break;
                }
            }

            if !digits.is_empty() {
                if let Ok(id) = digits.parse::<u32>() {
                    if let Some(name) = table.lookup(id) {
                        rewritten.push('%');
                        rewritten.push_str(name);
                        continue;
                    }
                }
                rewritten.push('%');
                rewritten.push_str(&digits);
                continue;
            }
        }

        rewritten.push(ch);
    }

    *line = rewritten;
}

fn apply_indent(line: &mut String, base_indent: usize) {
    if line.is_empty() {
        if base_indent > 0 {
            *line = " ".repeat(base_indent);
        }
        return;
    }

    if let Some(eq_index) = line.find(" = ") {
        if line.starts_with('%') {
            let head = &line[..eq_index];
            let trimmed_head = head.trim();
            let head_chars = trimmed_head.chars().count();
            let id_len = head_chars.saturating_sub(1);
            let mut indented =
                String::with_capacity(base_indent + STANDARD_INDENT_COLUMN + line.len());
            if base_indent > 0 {
                indented.push_str(&" ".repeat(base_indent));
            }
            let padding = STANDARD_INDENT_COLUMN.saturating_sub(4 + id_len);
            if padding > 0 {
                indented.push_str(&" ".repeat(padding));
            }
            indented.push_str(trimmed_head);
            indented.push_str(&line[eq_index..]);
            *line = indented;
            return;
        }
    }

    let mut indented = String::with_capacity(base_indent + STANDARD_INDENT_COLUMN + line.len());
    if base_indent > 0 {
        indented.push_str(&" ".repeat(base_indent));
    }
    indented.push_str(&" ".repeat(STANDARD_INDENT_COLUMN));
    indented.push_str(line.trim_start());
    *line = indented;
}

fn insert_block_indent(line: &mut String, block_indent: usize) {
    if block_indent == 0 || line.is_empty() {
        return;
    }
    if let Some(eq_index) = line.find(" = ") {
        let mut indented = String::with_capacity(line.len() + block_indent);
        indented.push_str(&line[..eq_index + 3]);
        indented.push_str(&" ".repeat(block_indent));
        indented.push_str(&line[eq_index + 3..]);
        *line = indented;
    } else {
        let mut prefixed = String::with_capacity(block_indent + line.len());
        prefixed.push_str(&" ".repeat(block_indent));
        prefixed.push_str(line);
        *line = prefixed;
    }
}

fn apply_color_formatting(line: &mut String, colorize: bool) {
    if !colorize || line.is_empty() {
        return;
    }

    color_result_identifier(line);
    color_comment_section(line);
}

fn color_result_identifier(line: &mut String) {
    if let Some(eq_index) = line.find(" = ") {
        if let Some(percent_index) = line[..eq_index].rfind('%') {
            let mut colored =
                String::with_capacity(line.len() + COLOR_BLUE.len() + COLOR_RESET.len());
            colored.push_str(&line[..percent_index]);
            colored.push_str(COLOR_BLUE);
            colored.push_str(&line[percent_index..eq_index]);
            colored.push_str(COLOR_RESET);
            colored.push_str(&line[eq_index..]);
            *line = colored;
        }
    }
}

fn color_comment_section(line: &mut String) {
    if let Some(comment_index) = line.find("; ") {
        let mut colored = String::with_capacity(line.len() + COLOR_GREY.len() + COLOR_RESET.len());
        colored.push_str(&line[..comment_index]);
        colored.push_str(COLOR_GREY);
        colored.push_str(&line[comment_index..]);
        colored.push_str(COLOR_RESET);
        *line = colored;
    }
}

fn collect_instruction_offsets(words: &[u32]) -> Vec<u32> {
    if words.len() <= HEADER_WORD_COUNT {
        return Vec::new();
    }
    let mut offsets = Vec::new();
    let mut index = HEADER_WORD_COUNT;
    let mut byte_offset = (HEADER_WORD_COUNT * 4) as u32;
    while index < words.len() {
        let word = words[index];
        let word_count = (word >> 16) as usize;
        if word_count == 0 || index + word_count > words.len() {
            break;
        }
        offsets.push(byte_offset);
        index += word_count;
        byte_offset += (word_count * 4) as u32;
    }
    offsets
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocationPlacement {
    BeforeLabel {
        function_index: usize,
        label_index: usize,
    },
    ModuleTail,
}

fn classify_location_debugs(words: &[u32], function_count: usize) -> Vec<LocationPlacement> {
    let mut placements = Vec::new();
    if words.len() <= HEADER_WORD_COUNT {
        return placements;
    }
    let mut index = HEADER_WORD_COUNT;
    let mut current_function = 0usize;
    let mut inside_function = false;
    let mut label_index = 0usize;
    while index < words.len() {
        let word = words[index];
        let word_count = (word >> 16) as usize;
        if word_count == 0 || index + word_count > words.len() {
            break;
        }
        let opcode = word & 0xFFFF;
        if let Some(op) = spirv::Op::from_u32(opcode) {
            if rspirv::grammar::reflect::is_location_debug(op) {
                if inside_function {
                    placements.push(LocationPlacement::BeforeLabel {
                        function_index: current_function,
                        label_index,
                    });
                } else if current_function >= function_count {
                    placements.push(LocationPlacement::ModuleTail);
                }
            } else {
                match op {
                    spirv::Op::Function => {
                        inside_function = true;
                        label_index = 0;
                    }
                    spirv::Op::Label => {
                        label_index += 1;
                    }
                    spirv::Op::FunctionEnd => {
                        inside_function = false;
                        if current_function < function_count {
                            current_function += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        index += word_count;
    }
    placements
}

fn collect_module_instructions<'a>(
    module: &'a dr::Module,
    words: &[u32],
    reorder_blocks: bool,
) -> Vec<InstructionRecord<'a>> {
    let mut records = Vec::new();
    for instruction in &module.capabilities {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.extensions {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.ext_inst_imports {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    if let Some(inst) = module.memory_model.as_ref() {
        records.push(InstructionRecord::new(
            inst,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.entry_points {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.execution_modes {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.debug_string_source {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Debug,
        ));
    }
    for instruction in &module.debug_names {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Debug,
        ));
    }
    for instruction in &module.debug_module_processed {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Debug,
        ));
    }
    for instruction in &module.annotations {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Annotations,
        ));
    }
    let trailing_location_start = module
        .types_global_values
        .iter()
        .rposition(|inst| !rspirv::grammar::reflect::is_location_debug(inst.class.opcode))
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let (type_values, trailing_location_insts) =
        module.types_global_values.split_at(trailing_location_start);
    let location_placements = classify_location_debugs(words, module.functions.len());
    let mut placement_iter = location_placements.into_iter();
    let mut label_locations: Vec<Vec<Vec<&Instruction>>> = vec![Vec::new(); module.functions.len()];
    let mut function_tail_locations: Vec<Vec<&Instruction>> =
        vec![Vec::new(); module.functions.len()];
    let mut module_tail_locations: Vec<&Instruction> = Vec::new();

    for instruction in trailing_location_insts {
        match placement_iter
            .next()
            .unwrap_or(LocationPlacement::ModuleTail)
        {
            LocationPlacement::BeforeLabel {
                function_index,
                label_index,
            } => {
                if let Some(buckets) = label_locations.get_mut(function_index) {
                    if buckets.len() <= label_index {
                        buckets.resize_with(label_index + 1, Vec::new);
                    }
                    buckets[label_index].push(instruction);
                } else if let Some(tails) = function_tail_locations.get_mut(function_index) {
                    tails.push(instruction);
                } else {
                    module_tail_locations.push(instruction);
                }
            }
            LocationPlacement::ModuleTail => module_tail_locations.push(instruction),
        }
    }

    for instruction in type_values {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Types,
        ));
    }
    for (function_index, function) in module.functions.iter().enumerate() {
        if let Some(ref def) = function.def {
            records.push(InstructionRecord::new(
                def,
                0,
                BlockPosition::Global,
                ModuleSection::Functions,
            ));
        }
        for parameter in &function.parameters {
            records.push(InstructionRecord::new(
                parameter,
                0,
                BlockPosition::Global,
                ModuleSection::Functions,
            ));
        }
        let nest_levels = compute_block_nest_levels(function);
        let block_indices = if reorder_blocks {
            reorder_function_blocks(function)
        } else {
            (0..function.blocks.len()).collect()
        };
        let mut label_index = 0usize;
        for index in block_indices {
            let block = &function.blocks[index];
            let block_depth = nest_levels.get(index).copied().unwrap_or(0);
            if let Some(ref label) = block.label {
                if let Some(buckets) = label_locations.get(function_index) {
                    if let Some(locations) = buckets.get(label_index) {
                        for instruction in locations {
                            records.push(InstructionRecord::new(
                                instruction,
                                block_depth,
                                BlockPosition::Global,
                                ModuleSection::Functions,
                            ));
                        }
                    }
                }
                records.push(InstructionRecord::new(
                    label,
                    block_depth,
                    BlockPosition::Label,
                    ModuleSection::Functions,
                ));
                label_index += 1;
            }
            for instruction in &block.instructions {
                records.push(InstructionRecord::new(
                    instruction,
                    block_depth,
                    BlockPosition::Body,
                    ModuleSection::Functions,
                ));
            }
        }

        if let Some(buckets) = label_locations.get(function_index) {
            if label_index < buckets.len() {
                if let Some(tails) = function_tail_locations.get_mut(function_index) {
                    for locations in &buckets[label_index..] {
                        tails.extend(locations.iter().copied());
                    }
                }
            }
        }

        if let Some(ref end) = function.end {
            if let Some(tails) = function_tail_locations.get(function_index) {
                for instruction in tails {
                    records.push(InstructionRecord::new(
                        instruction,
                        0,
                        BlockPosition::Global,
                        ModuleSection::Functions,
                    ));
                }
            }
            records.push(InstructionRecord::new(
                end,
                0,
                BlockPosition::Global,
                ModuleSection::Functions,
            ));
        }
    }

    for instruction in module_tail_locations {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Functions,
        ));
    }

    records
}

struct CommentAligner {
    last_alignment: usize,
}

impl CommentAligner {
    fn new() -> Self {
        Self { last_alignment: 0 }
    }

    fn append_comment(&mut self, line: &mut String, comment: &str) {
        let line_length = line.chars().count();
        let mut align = line_length + 2;
        align = align.max(self.last_alignment).max(COMMENT_COLUMN);
        align = (align + 3) & !0x3;
        self.last_alignment = align.min(MAX_COMMENT_ALIGN);
        if line_length < align {
            line.push_str(&" ".repeat(align - line_length));
        }
        line.push_str("; ");
        line.push_str(comment);
    }

    fn reset(&mut self) {
        self.last_alignment = 0;
    }
}

struct CommentCollector {
    enabled: bool,
    decorations: HashMap<u32, String>,
}

impl CommentCollector {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            decorations: HashMap::new(),
        }
    }

    fn observe(&mut self, instruction: &Instruction) {
        if !self.enabled {
            return;
        }
        if instruction.class.opcode == spirv::Op::Decorate {
            self.record_decorate(instruction);
        }
    }

    fn inline_comment(&self, instruction: &Instruction) -> Option<String> {
        if !self.enabled {
            return None;
        }
        if instruction.class.opcode == spirv::Op::Name {
            if let Some(id) = instruction.operands.first().and_then(extract_id_ref) {
                return Some(format!("id %{}", id));
            }
        }
        None
    }

    fn result_comment(&self, instruction: &Instruction) -> Option<String> {
        if !self.enabled {
            return None;
        }
        instruction
            .result_id
            .and_then(|id| self.decorations.get(&id).cloned())
    }

    fn record_decorate(&mut self, instruction: &Instruction) {
        let Some(target) = instruction.operands.first().and_then(extract_id_ref) else {
            return;
        };
        let mut parts = Vec::new();
        for operand in instruction.operands.iter().skip(1) {
            parts.push(operand.to_string());
        }
        if parts.is_empty() {
            return;
        }
        let entry = self.decorations.entry(target).or_default();
        if !entry.is_empty() {
            entry.push_str(", ");
        }
        entry.push_str(&parts.join(" "));
    }
}

#[derive(Clone, Copy)]
struct StackEntry {
    index: usize,
    post_visit: bool,
}

fn reorder_function_blocks(function: &dr::Function) -> Vec<usize> {
    if function.blocks.is_empty() {
        return Vec::new();
    }

    let (mut infos, id_to_index) = build_block_infos(function);
    let mut stack = Vec::new();
    let mut post_order = Vec::with_capacity(function.blocks.len());
    let mut visited = vec![false; infos.len()];

    if let Some(info) = infos.first_mut() {
        info.nest_level = Some(0);
        info.reachable = true;
    }
    stack.push(StackEntry {
        index: 0,
        post_visit: false,
    });

    while let Some(entry) = stack.pop() {
        if entry.post_visit {
            post_order.push(entry.index);
            continue;
        }

        if visited.get(entry.index).copied().unwrap_or(false) {
            continue;
        }
        if let Some(flag) = visited.get_mut(entry.index) {
            *flag = true;
        }
        stack.push(StackEntry {
            index: entry.index,
            post_visit: true,
        });

        nest_successors(&mut infos, entry.index, &id_to_index);
        infos[entry.index].reachable = true;

        let block = &infos[entry.index];
        // Push higher-priority successors first; reverse post-order traversal will then
        // print structured bodies before their merges.
        push_successor(&mut stack, &id_to_index, block.true_block_id);
        push_successor(&mut stack, &id_to_index, block.false_block_id);
        push_successor(&mut stack, &id_to_index, block.body_block_id);
        push_successor(&mut stack, &id_to_index, block.next_block_id);
        for &case in &block.case_block_ids {
            push_successor(&mut stack, &id_to_index, case);
        }
        push_successor(&mut stack, &id_to_index, block.continue_block_id);
        push_successor(&mut stack, &id_to_index, block.merge_block_id);
    }

    let mut order: Vec<usize> = post_order.into_iter().rev().collect();
    for (index, info) in infos.iter_mut().enumerate() {
        if !info.reachable {
            info.nest_level = Some(0);
            order.push(index);
        }
    }
    order
}

fn push_successor(stack: &mut Vec<StackEntry>, id_to_index: &HashMap<u32, usize>, block_id: u32) {
    if block_id == 0 {
        return;
    }
    if let Some(&index) = id_to_index.get(&block_id) {
        stack.push(StackEntry {
            index,
            post_visit: false,
        });
    }
}

fn nest_successors(infos: &mut [BlockInfo], index: usize, id_to_index: &HashMap<u32, usize>) {
    let level = infos[index].nest_level.unwrap_or(0);
    let merge_block_id = infos[index].merge_block_id;
    let continue_block_id = infos[index].continue_block_id;
    let true_block_id = infos[index].true_block_id;
    let false_block_id = infos[index].false_block_id;
    let body_block_id = infos[index].body_block_id;
    let next_block_id = infos[index].next_block_id;
    let case_block_ids = infos[index].case_block_ids.clone();

    let mut assign = |target: u32, new_level: u32| {
        if target == 0 {
            return;
        }
        if let Some(&succ_index) = id_to_index.get(&target) {
            if infos[succ_index].nest_level.is_none() {
                infos[succ_index].nest_level = Some(new_level);
            }
        }
    };

    assign(merge_block_id, level);
    assign(continue_block_id, level + 1);
    assign(true_block_id, level + 2);
    assign(false_block_id, level + 2);
    assign(body_block_id, level + 2);
    assign(next_block_id, level);
    for case in &case_block_ids {
        assign(*case, level + 2);
    }
}

#[derive(Default)]
struct BlockInfo {
    label_id: u32,
    merge_block_id: u32,
    continue_block_id: u32,
    true_block_id: u32,
    false_block_id: u32,
    body_block_id: u32,
    next_block_id: u32,
    case_block_ids: Vec<u32>,
    nest_level: Option<u32>,
    reachable: bool,
}

fn compute_block_nest_levels(function: &dr::Function) -> Vec<u32> {
    let block_count = function.blocks.len();
    if block_count == 0 {
        return Vec::new();
    }

    let (mut infos, id_to_index) = build_block_infos(function);
    let mut stack = Vec::new();

    infos[0].nest_level = Some(0);
    stack.push(0usize);

    while let Some(index) = stack.pop() {
        let level = infos[index].nest_level.unwrap_or(0);
        let merge_block_id = infos[index].merge_block_id;
        let continue_block_id = infos[index].continue_block_id;
        let true_block_id = infos[index].true_block_id;
        let false_block_id = infos[index].false_block_id;
        let body_block_id = infos[index].body_block_id;
        let next_block_id = infos[index].next_block_id;
        let case_block_ids = infos[index].case_block_ids.clone();
        let mut assign = |target: u32, new_level: u32| {
            if target == 0 {
                return;
            }
            if let Some(&succ_index) = id_to_index.get(&target) {
                if infos[succ_index].nest_level.is_none() {
                    infos[succ_index].nest_level = Some(new_level);
                    stack.push(succ_index);
                }
            }
        };

        assign(merge_block_id, level);
        assign(continue_block_id, level + 1);
        assign(true_block_id, level + 2);
        assign(false_block_id, level + 2);
        assign(body_block_id, level + 2);
        assign(next_block_id, level);
        for case_id in case_block_ids {
            assign(case_id, level + 2);
        }
    }

    infos
        .into_iter()
        .map(|info| info.nest_level.unwrap_or(0))
        .collect()
}

fn build_block_infos(function: &dr::Function) -> (Vec<BlockInfo>, HashMap<u32, usize>) {
    let mut infos = Vec::with_capacity(function.blocks.len());
    let mut id_to_index = HashMap::new();
    for (index, block) in function.blocks.iter().enumerate() {
        let info = build_block_info(block);
        if info.label_id != 0 {
            id_to_index.insert(info.label_id, index);
        }
        infos.push(info);
    }
    (infos, id_to_index)
}

fn build_block_info(block: &dr::Block) -> BlockInfo {
    let mut info = BlockInfo {
        label_id: block
            .label
            .as_ref()
            .and_then(|inst| inst.result_id)
            .unwrap_or(0),
        ..BlockInfo::default()
    };
    for instruction in &block.instructions {
        match instruction.class.opcode {
            spirv::Op::LoopMerge => {
                info.merge_block_id = instruction
                    .operands
                    .first()
                    .and_then(extract_id_ref)
                    .unwrap_or(0);
                info.continue_block_id = instruction
                    .operands
                    .get(1)
                    .and_then(extract_id_ref)
                    .unwrap_or(0);
            }
            spirv::Op::SelectionMerge => {
                info.merge_block_id = instruction
                    .operands
                    .first()
                    .and_then(extract_id_ref)
                    .unwrap_or(0);
            }
            _ => {}
        }
    }

    if let Some(terminator) = block.instructions.last() {
        match terminator.class.opcode {
            spirv::Op::Branch => {
                let target = terminator
                    .operands
                    .last()
                    .and_then(extract_id_ref)
                    .unwrap_or(0);
                if info.merge_block_id != 0 {
                    info.body_block_id = target;
                } else {
                    info.next_block_id = target;
                }
            }
            spirv::Op::BranchConditional => {
                if terminator.operands.len() >= 3 {
                    info.true_block_id = terminator
                        .operands
                        .get(1)
                        .and_then(extract_id_ref)
                        .unwrap_or(0);
                    info.false_block_id = terminator
                        .operands
                        .get(2)
                        .and_then(extract_id_ref)
                        .unwrap_or(0);
                }
            }
            spirv::Op::Switch => {
                for (index, operand) in terminator.operands.iter().enumerate().skip(1) {
                    if index % 2 == 1 {
                        if let Some(id) = extract_id_ref(operand) {
                            info.case_block_ids.push(id);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    info
}

#[cfg(test)]
mod tests {
    use super::{disassemble_binary, FRIENDLY_NAME_SAMPLE_BINARY};
    use crate::assembly::{
        assemble_text, assemble_text_with_options, BinaryToTextOptions, TextToBinaryOptions,
    };
    use crate::target_env::TargetEnv;
    use rspirv::binary::Assemble;
    use rspirv::dr::{self, Builder};
    use rspirv::spirv::{
        AccessQualifier, AddressingModel, BuiltIn, Capability, Decoration, ExecutionModel,
        FunctionControl, MemoryModel, SelectionControl, StorageClass,
    };

    const CONDITIONAL_EXTENSION_SAMPLE_BINARY: &[u32] = &[
        0x07230203, 0x00010600, 0x00000000, 0x00000003, 0x00000000, 0x00020011, 0x00000001,
        0x0003000e, 0x00000000, 0x00000001, 0x00020014, 0x00000001, 0x00030031, 0x00000001,
        0x00000002, 0x00091868, 0x00000002, 0x5f565053, 0x45544e49, 0x75665f4c, 0x6974636e,
        0x765f6e6f, 0x61697261, 0x0073746e,
    ];

    fn disassemble_with_options(words: &[u32], options: BinaryToTextOptions) -> String {
        disassemble_binary(words, options).expect("disassemble")
    }

    fn entry_point_module(name_literal: &str) -> String {
        format!(
            "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %3 {name_literal}
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd
"
        )
    }

    fn encode_and_decode_fixture(
        text: &str,
        disassemble_options: BinaryToTextOptions,
        assemble_options: TextToBinaryOptions,
    ) -> String {
        let binary = assemble_text_with_options(text, TargetEnv::Universal1_0, assemble_options)
            .expect("assemble");
        disassemble_binary(
            &binary,
            disassemble_options | BinaryToTextOptions::NO_HEADER,
        )
        .expect("disassemble")
    }

    fn round_trip_entry_point_literal(name_literal: &str, expected_literal: &str) {
        let before = entry_point_module(name_literal);
        let binary = assemble_text(&before).expect("assemble");
        let text =
            disassemble_binary(&binary, BinaryToTextOptions::NO_HEADER).expect("disassemble");
        assert!(
            text.contains("OpEntryPoint Vertex"),
            "disassembly missing entry point: {text:?}"
        );
        assert!(
            text.contains(expected_literal),
            "expected literal {expected_literal:?} in {text:?}"
        );
        assert!(
            !text.contains(name_literal),
            "disassembly retained escape prefix: {text:?}"
        );
    }

    #[test]
    fn disassembles_simple_module() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble text");
        let options = BinaryToTextOptions::NO_HEADER
            | BinaryToTextOptions::COMMENT
            | BinaryToTextOptions::INDENT;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.trim_start().starts_with("OpCapability Shader"));
        assert!(disassembled.contains("OpFunctionEnd"));
    }

    #[test]
    fn disassembly_respects_no_header_option() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble text");
        let options = BinaryToTextOptions::NONE | BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(!disassembled.starts_with(";"));
        assert!(disassembled.starts_with("OpCapability"));
    }

    #[test]
    fn invalid_binary_reports_parse_error() {
        let binary = vec![0xDEAD_BEEFu32];
        let error = disassemble_binary(&binary, BinaryToTextOptions::NO_HEADER)
            .expect_err("expected parse error");
        match error {
            super::DisassemblyError::Parse { message, .. } => assert!(!message.is_empty()),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn disassembly_accepts_print_option() {
        let text = "OpCapability Shader";
        let binary = assemble_text(text).expect("assemble text");
        let options = BinaryToTextOptions::PRINT | BinaryToTextOptions::NO_HEADER;
        // PRINT is handled by the caller; the disassembler should still succeed.
        let _ = disassemble_binary(&binary, options).expect("disassemble");
    }

    #[test]
    fn supports_options_covers_zero_and_no_header() {
        assert!(super::supports_options(BinaryToTextOptions::empty()));
        assert!(super::supports_options(BinaryToTextOptions::NO_HEADER));
        assert!(super::supports_options(BinaryToTextOptions::INDENT));
        assert!(super::supports_options(BinaryToTextOptions::FRIENDLY_NAMES));
        assert!(super::supports_options(BinaryToTextOptions::COMMENT));
        assert!(super::supports_options(BinaryToTextOptions::REORDER_BLOCKS));
        assert!(super::supports_options(BinaryToTextOptions::COLOR));
        assert!(super::supports_options(BinaryToTextOptions::HEX));
    }

    #[test]
    fn disassembly_appends_byte_offsets() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical Simple",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble text");
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::SHOW_BYTE_OFFSET;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        let expected = [
            "OpCapability Shader                                 ; 0x00000014",
            "OpMemoryModel Logical Simple                        ; 0x0000001c",
            "%1 = OpTypeVoid                                     ; 0x00000028",
            "%2 = OpTypeFunction %1                              ; 0x00000030",
        ]
        .join("\n")
            + "\n";
        assert_eq!(disassembled, expected);
    }

    #[test]
    fn disassembly_applies_indent_formatting() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical Simple",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble text");
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::INDENT;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");

        let expected = [
            "               OpCapability Shader",
            "               OpMemoryModel Logical Simple",
            "          %1 = OpTypeVoid",
            "          %2 = OpTypeFunction %1",
            "          %3 = OpFunction %1 None %2",
            "          %4 = OpLabel",
            "               OpReturn",
            "               OpFunctionEnd",
        ]
        .join("\n")
            + "\n";
        assert_eq!(disassembled, expected);
    }

    #[test]
    fn disassembly_uses_friendly_names_when_available() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let void_fn = builder.type_function(void, vec![]);
        let main = builder
            .begin_function(void, None, FunctionControl::NONE, void_fn)
            .expect("begin function");
        builder.name(main, "my_main");
        builder.entry_point(ExecutionModel::Vertex, main, "main", Vec::new());
        builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end function");
        let module = builder.module();
        let binary = module.assemble();
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.contains("%my_main = OpFunction"));
        assert!(disassembled.contains("OpEntryPoint Vertex %my_main \"main\""));
    }

    #[test]
    fn disassembly_nested_indent_tracks_block_depth() {
        let binary = build_selection_module(false);
        let options = BinaryToTextOptions::NO_HEADER
            | BinaryToTextOptions::INDENT
            | BinaryToTextOptions::NESTED_INDENT
            | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        let lines: Vec<&str> = disassembled.lines().collect();
        let entry_idx = lines
            .iter()
            .position(|line| line.contains("%entry ="))
            .expect("entry label");
        let then_idx = lines
            .iter()
            .position(|line| line.contains("%then ="))
            .expect("then label");
        let entry_spaces = spaces_after_equals(lines[entry_idx]).expect("entry spaces");
        let then_spaces = spaces_after_equals(lines[then_idx]).expect("then spaces");
        assert!(then_spaces > entry_spaces);
        let body_idx = lines
            .iter()
            .enumerate()
            .skip(then_idx + 1)
            .find(|(_, line)| {
                !line.trim().is_empty() && !line.contains("OpLabel") && line.contains("Op")
            })
            .map(|(idx, _)| idx)
            .expect("body instruction");
        assert!(leading_spaces(lines[body_idx]) > leading_spaces(lines[then_idx]));
    }

    #[test]
    fn disassembly_emits_decoration_comments() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let float = builder.type_float(32, None);
        let ptr = builder.type_pointer(None, StorageClass::UniformConstant, float);
        let var = builder.variable(ptr, None, StorageClass::UniformConstant, None);
        builder.decorate(
            var,
            Decoration::DescriptorSet,
            [dr::Operand::LiteralBit32(0)],
        );
        builder.decorate(var, Decoration::Binding, [dr::Operand::LiteralBit32(1)]);
        let void = builder.type_void();
        let void_fn = builder.type_function(void, vec![]);
        builder
            .begin_function(void, None, FunctionControl::NONE, void_fn)
            .expect("function");
        builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end");
        let module = builder.module();
        let binary = module.assemble();
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::COMMENT;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.contains("DescriptorSet 0"));
        assert!(disassembled.contains("Binding 1"));
    }

    #[test]
    fn disassembly_applies_color_formatting() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let void_fn = builder.type_function(void, vec![]);
        builder
            .begin_function(void, None, FunctionControl::NONE, void_fn)
            .expect("function");
        builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end");
        let module = builder.module();
        let binary = module.assemble();
        let options = BinaryToTextOptions::NO_HEADER
            | BinaryToTextOptions::INDENT
            | BinaryToTextOptions::COLOR
            | BinaryToTextOptions::COMMENT
            | BinaryToTextOptions::SHOW_BYTE_OFFSET;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.contains(super::COLOR_BLUE));
        assert!(disassembled.contains(super::COLOR_GREY));
    }

    #[test]
    fn disassembly_reorders_blocks_when_requested() {
        let binary = build_selection_module(false);
        let base_options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let unreordered =
            disassemble_binary(&binary, base_options).expect("disassemble without reorder");
        let reordered =
            disassemble_binary(&binary, base_options | BinaryToTextOptions::REORDER_BLOCKS)
                .expect("disassemble with reorder");

        let label_index = |text: &str, needle: &str| -> usize {
            text.lines()
                .position(|line| line.contains(needle) && line.contains("OpLabel"))
                .unwrap_or(usize::MAX)
        };

        let unreordered_then = label_index(&unreordered, "%then");
        let unreordered_merge = label_index(&unreordered, "%merge");
        assert!(unreordered_merge < unreordered_then);

        let reordered_entry = label_index(&reordered, "%entry");
        let reordered_then = label_index(&reordered, "%then");
        let reordered_merge = label_index(&reordered, "%merge");
        assert!(reordered_then < reordered_merge);
        assert!(reordered_entry < reordered_then);
    }

    #[test]
    fn disassembly_matches_indent_fixture_sample() {
        let input = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeInt 32 0",
            "%2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10 ; force IDs into double digits",
            "%11 = OpConstant %1 42",
            "OpStore %2 %3 Aligned|Volatile 4 ; bogus, but not indented",
        ]
        .join("\n");
        let expected = [
            "               OpCapability Shader",
            "               OpMemoryModel Logical GLSL450",
            "          %1 = OpTypeInt 32 0",
            "          %2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10",
            "         %11 = OpConstant %1 42",
            "               OpStore %2 %3 Volatile|Aligned 4",
        ]
        .join("\n")
            + "\n";
        let output = encode_and_decode_fixture(
            &input,
            BinaryToTextOptions::INDENT,
            TextToBinaryOptions::NONE,
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn disassembly_matches_indent_fixture_nested_if() {
        let input = [
            "OpCapability Shader",
            "OpMemoryModel Logical Simple",
            "OpEntryPoint Fragment %100 \"main\"",
            "OpExecutionMode %100 OriginUpperLeft",
            "OpName %var \"var\"",
            "%void = OpTypeVoid",
            "%3 = OpTypeFunction %void",
            "%bool = OpTypeBool",
            "%5 = OpConstantNull %bool",
            "%true = OpConstantTrue %bool",
            "%false = OpConstantFalse %bool",
            "%uint = OpTypeInt 32 0",
            "%int = OpTypeInt 32 1",
            "%uint_42 = OpConstant %uint 42",
            "%int_42 = OpConstant %int 42",
            "%13 = OpTypeFunction %uint",
            "%uint_0 = OpConstant %uint 0",
            "%uint_1 = OpConstant %uint 1",
            "%uint_2 = OpConstant %uint 2",
            "%uint_3 = OpConstant %uint 3",
            "%uint_4 = OpConstant %uint 4",
            "%uint_5 = OpConstant %uint 5",
            "%uint_6 = OpConstant %uint 6",
            "%uint_7 = OpConstant %uint 7",
            "%uint_8 = OpConstant %uint 8",
            "%uint_10 = OpConstant %uint 10",
            "%uint_20 = OpConstant %uint 20",
            "%uint_30 = OpConstant %uint 30",
            "%uint_40 = OpConstant %uint 40",
            "%uint_50 = OpConstant %uint 50",
            "%uint_90 = OpConstant %uint 90",
            "%uint_99 = OpConstant %uint 99",
            "%_ptr_Private_uint = OpTypePointer Private %uint",
            "%var = OpVariable %_ptr_Private_uint Private",
            "%uint_999 = OpConstant %uint 999",
            "%100 = OpFunction %void None %3",
            "%10 = OpLabel",
            "OpStore %var %uint_0",
            "OpSelectionMerge %99 None",
            "OpBranchConditional %5 %30 %40",
            "%30 = OpLabel",
            "OpStore %var %uint_1",
            "OpBranch %99",
            "%40 = OpLabel",
            "OpStore %var %uint_2",
            "OpBranch %99",
            "%99 = OpLabel",
            "OpStore %var %uint_999",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let expected = [
            "               OpCapability Shader",
            "               OpMemoryModel Logical Simple",
            "               OpEntryPoint Fragment %100 \"main\"",
            "               OpExecutionMode %100 OriginUpperLeft",
            "               OpName %1 \"var\"",
            "          %2 = OpTypeVoid",
            "          %3 = OpTypeFunction %2",
            "          %4 = OpTypeBool",
            "          %5 = OpConstantNull %4",
            "          %6 = OpConstantTrue %4",
            "          %7 = OpConstantFalse %4",
            "          %8 = OpTypeInt 32 0",
            "          %9 = OpTypeInt 32 1",
            "         %11 = OpConstant %8 42",
            "         %12 = OpConstant %9 42",
            "         %13 = OpTypeFunction %8",
            "         %14 = OpConstant %8 0",
            "         %15 = OpConstant %8 1",
            "         %16 = OpConstant %8 2",
            "         %17 = OpConstant %8 3",
            "         %18 = OpConstant %8 4",
            "         %19 = OpConstant %8 5",
            "         %20 = OpConstant %8 6",
            "         %21 = OpConstant %8 7",
            "         %22 = OpConstant %8 8",
            "         %23 = OpConstant %8 10",
            "         %24 = OpConstant %8 20",
            "         %25 = OpConstant %8 30",
            "         %26 = OpConstant %8 40",
            "         %27 = OpConstant %8 50",
            "         %28 = OpConstant %8 90",
            "         %29 = OpConstant %8 99",
            "         %31 = OpTypePointer Private %8",
            "          %1 = OpVariable %31 Private",
            "         %32 = OpConstant %8 999",
            "        %100 = OpFunction %2 None %3",
            "",
            "         %10 = OpLabel",
            "                 OpStore %1 %14",
            "                 OpSelectionMerge %99 None",
            "                 OpBranchConditional %5 %30 %40",
            "",
            "         %30 =     OpLabel",
            "                     OpStore %1 %15",
            "                     OpBranch %99",
            "",
            "         %40 =     OpLabel",
            "                     OpStore %1 %16",
            "                     OpBranch %99",
            "",
            "         %99 = OpLabel",
            "                 OpStore %1 %32",
            "                 OpReturn",
            "               OpFunctionEnd",
        ]
        .join("\n")
            + "\n";
        let output = encode_and_decode_fixture(
            &input,
            BinaryToTextOptions::INDENT | BinaryToTextOptions::NESTED_INDENT,
            TextToBinaryOptions::PRESERVE_NUMERIC_IDS,
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn disassembly_matches_indent_fixture_reordered_if() {
        let input = [
            "               OpCapability Shader",
            "               OpMemoryModel Logical Simple",
            "               OpEntryPoint Fragment %100 \"main\"",
            "               OpExecutionMode %100 OriginUpperLeft",
            "               OpName %1 \"var\"",
            "          %2 = OpTypeVoid",
            "          %3 = OpTypeFunction %2",
            "          %4 = OpTypeBool",
            "          %5 = OpConstantNull %4",
            "          %6 = OpConstantTrue %4",
            "          %7 = OpConstantFalse %4",
            "          %8 = OpTypeInt 32 0",
            "          %9 = OpTypeInt 32 1",
            "         %11 = OpConstant %8 42",
            "         %12 = OpConstant %9 42",
            "         %13 = OpTypeFunction %8",
            "         %14 = OpConstant %8 0",
            "         %15 = OpConstant %8 1",
            "         %16 = OpConstant %8 2",
            "         %17 = OpConstant %8 3",
            "         %18 = OpConstant %8 4",
            "         %19 = OpConstant %8 5",
            "         %21 = OpConstant %8 6",
            "         %22 = OpConstant %8 7",
            "         %23 = OpConstant %8 8",
            "         %24 = OpConstant %8 10",
            "         %25 = OpConstant %8 20",
            "         %26 = OpConstant %8 30",
            "         %27 = OpConstant %8 40",
            "         %28 = OpConstant %8 50",
            "         %29 = OpConstant %8 90",
            "         %31 = OpConstant %8 99",
            "         %32 = OpTypePointer Private %8",
            "          %1 = OpVariable %32 Private",
            "         %33 = OpConstant %8 999",
            "        %100 = OpFunction %2 None %3",
            "         %10 = OpLabel",
            "               OpSelectionMerge %99 None",
            "               OpBranchConditional %5 %20 %50",
            "         %99 = OpLabel",
            "               OpReturn",
            "         %20 = OpLabel",
            "               OpSelectionMerge %49 None",
            "               OpBranchConditional %5 %30 %40",
            "         %49 = OpLabel",
            "               OpBranch %99",
            "         %40 = OpLabel",
            "               OpBranch %49",
            "         %30 = OpLabel",
            "               OpBranch %49",
            "         %50 = OpLabel",
            "               OpSelectionMerge %79 None",
            "               OpBranchConditional %5 %60 %70",
            "         %79 = OpLabel",
            "               OpBranch %99",
            "         %60 = OpLabel",
            "               OpBranch %79",
            "         %70 = OpLabel",
            "               OpBranch %79",
            "               OpFunctionEnd",
        ]
        .join("\n");
        let expected = [
            "               OpCapability Shader",
            "               OpMemoryModel Logical Simple",
            "               OpEntryPoint Fragment %100 \"main\"",
            "               OpExecutionMode %100 OriginUpperLeft",
            "               OpName %1 \"var\"",
            "          %2 = OpTypeVoid",
            "          %3 = OpTypeFunction %2",
            "          %4 = OpTypeBool",
            "          %5 = OpConstantNull %4",
            "          %6 = OpConstantTrue %4",
            "          %7 = OpConstantFalse %4",
            "          %8 = OpTypeInt 32 0",
            "          %9 = OpTypeInt 32 1",
            "         %11 = OpConstant %8 42",
            "         %12 = OpConstant %9 42",
            "         %13 = OpTypeFunction %8",
            "         %14 = OpConstant %8 0",
            "         %15 = OpConstant %8 1",
            "         %16 = OpConstant %8 2",
            "         %17 = OpConstant %8 3",
            "         %18 = OpConstant %8 4",
            "         %19 = OpConstant %8 5",
            "         %21 = OpConstant %8 6",
            "         %22 = OpConstant %8 7",
            "         %23 = OpConstant %8 8",
            "         %24 = OpConstant %8 10",
            "         %25 = OpConstant %8 20",
            "         %26 = OpConstant %8 30",
            "         %27 = OpConstant %8 40",
            "         %28 = OpConstant %8 50",
            "         %29 = OpConstant %8 90",
            "         %31 = OpConstant %8 99",
            "         %32 = OpTypePointer Private %8",
            "          %1 = OpVariable %32 Private",
            "         %33 = OpConstant %8 999",
            "        %100 = OpFunction %2 None %3",
            "         %10 = OpLabel",
            "               OpSelectionMerge %99 None",
            "               OpBranchConditional %5 %20 %50",
            "         %20 = OpLabel",
            "               OpSelectionMerge %49 None",
            "               OpBranchConditional %5 %30 %40",
            "         %30 = OpLabel",
            "               OpBranch %49",
            "         %40 = OpLabel",
            "               OpBranch %49",
            "         %49 = OpLabel",
            "               OpBranch %99",
            "         %50 = OpLabel",
            "               OpSelectionMerge %79 None",
            "               OpBranchConditional %5 %60 %70",
            "         %60 = OpLabel",
            "               OpBranch %79",
            "         %70 = OpLabel",
            "               OpBranch %79",
            "         %79 = OpLabel",
            "               OpBranch %99",
            "         %99 = OpLabel",
            "               OpReturn",
            "               OpFunctionEnd",
        ]
        .join("\n")
            + "\n";
        let output = encode_and_decode_fixture(
            &input,
            BinaryToTextOptions::INDENT | BinaryToTextOptions::REORDER_BLOCKS,
            TextToBinaryOptions::PRESERVE_NUMERIC_IDS,
        );
        assert_eq!(output, expected);
    }

    const REORDER_FIXTURE_BINARY: &[u32] = &[
        0x07230203, 0x00010600, 0x00070000, 0x0000002a, 0x00000000, 0x00020011, 0x00000001,
        0x0003000e, 0x00000000, 0x00000000, 0x0005000f, 0x00000004, 0x00000001, 0x6e69616d,
        0x00000000, 0x00030010, 0x00000001, 0x00000007, 0x00030005, 0x00000002, 0x00007261,
        0x00020013, 0x00000003, 0x00030021, 0x00000004, 0x00000003, 0x00020014, 0x00000005,
        0x0003002e, 0x00000005, 0x00000006, 0x00030029, 0x00000005, 0x00000007, 0x0003002a,
        0x00000005, 0x00000008, 0x00040015, 0x00000009, 0x00000020, 0x00000000, 0x00040015,
        0x0000000a, 0x00000020, 0x00000001, 0x0004002b, 0x00000009, 0x0000000b, 0x0000002a,
        0x0004002b, 0x0000000a, 0x0000000c, 0x0000002a, 0x00030021, 0x0000000d, 0x00000009,
        0x0004002b, 0x00000009, 0x0000000e, 0x00000000, 0x0004002b, 0x00000009, 0x0000000f,
        0x00000001, 0x0004002b, 0x00000009, 0x00000010, 0x00000002, 0x0004002b, 0x00000009,
        0x00000011, 0x00000003, 0x0004002b, 0x00000009, 0x00000012, 0x00000004, 0x0004002b,
        0x00000009, 0x00000013, 0x00000005, 0x0004002b, 0x00000009, 0x00000014, 0x00000006,
        0x0004002b, 0x00000009, 0x00000015, 0x00000007, 0x0004002b, 0x00000009, 0x00000016,
        0x00000008, 0x0004002b, 0x00000009, 0x00000017, 0x0000000a, 0x0004002b, 0x00000009,
        0x00000018, 0x00000014, 0x0004002b, 0x00000009, 0x00000019, 0x0000001e, 0x0004002b,
        0x00000009, 0x0000001a, 0x00000028, 0x0004002b, 0x00000009, 0x0000001b, 0x00000032,
        0x0004002b, 0x00000009, 0x0000001c, 0x0000005a, 0x0004002b, 0x00000009, 0x0000001d,
        0x00000063, 0x00040020, 0x0000001e, 0x00000006, 0x00000009, 0x0004003b, 0x0000001e,
        0x00000002, 0x00000006, 0x0004002b, 0x00000009, 0x0000001f, 0x000003e7, 0x00050036,
        0x00000003, 0x00000064, 0x00000000, 0x00000004, 0x000200f8, 0x00000020, 0x000300f7,
        0x00000021, 0x00000000, 0x000400fa, 0x00000006, 0x00000022, 0x00000023, 0x000200f8,
        0x00000022, 0x000300f7, 0x00000024, 0x00000000, 0x000400fa, 0x00000006, 0x00000025,
        0x00000026, 0x000200f8, 0x00000025, 0x000200f9, 0x00000024, 0x000200f8, 0x00000026,
        0x000200f9, 0x00000024, 0x000200f8, 0x00000024, 0x000200f9, 0x00000021, 0x000200f8,
        0x00000023, 0x000300f7, 0x00000027, 0x00000000, 0x000400fa, 0x00000006, 0x00000028,
        0x00000029, 0x000200f8, 0x00000028, 0x000200f9, 0x00000027, 0x000200f8, 0x00000029,
        0x000200f9, 0x00000027, 0x000200f8, 0x00000027, 0x000200f9, 0x00000021, 0x000200f8,
        0x00000021, 0x000100fd, 0x00010038,
    ];

    #[test]
    fn reorder_blocks_matches_indent_fixture() {
        let module = rspirv::dr::load_words(REORDER_FIXTURE_BINARY).expect("load module");
        let function = module.functions.first().expect("function");
        let order = super::reorder_function_blocks(function);
        let expected: Vec<usize> = (0..function.blocks.len()).collect();
        assert_eq!(order, expected);
    }

    #[test]
    fn disassembly_print_option_writes_stdout() {
        let _ = take_print_log();
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let void_fn = builder.type_function(void, vec![]);
        builder
            .begin_function(void, None, FunctionControl::NONE, void_fn)
            .expect("function");
        builder.begin_block(None).expect("entry block");
        builder.ret().expect("return");
        builder.end_function().expect("end function");
        let module = builder.module();
        let binary = module.assemble();
        let options = BinaryToTextOptions::NO_HEADER
            | BinaryToTextOptions::PRINT
            | BinaryToTextOptions::FRIENDLY_NAMES;
        let text = disassemble_binary(&binary, options).expect("disassemble");
        assert!(text.is_empty());
        let printed = take_print_log();
        assert!(!printed.is_empty());
        assert!(printed.iter().any(|entry| entry.contains("OpFunction")));
    }

    #[test]
    fn disassembly_formats_literals_as_hex() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%uint = OpTypeInt 32 0",
            "%val = OpConstant %uint 42",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::HEX;
        let output = disassemble_binary(&binary, options).expect("disassemble");
        assert!(output.contains("0x0000002a"), "{output}");
        assert!(!output.contains(" 42"));
    }

    #[test]
    fn format_literal_string_preserves_escape_sequences() {
        let formatted = super::format_literal_string("foo\\nbar");
        assert_eq!(formatted, r#""foo\\nbar""#);
    }

    #[test]
    fn format_literal_string_reescapes_quotes() {
        let formatted = super::format_literal_string("say \"hi\"");
        assert_eq!(formatted, r#""say \"hi\"""#);
    }

    #[test]
    fn round_trips_string_literal_stripping_escape_prefix() {
        round_trip_entry_point_literal("\"\\foo\"", "\"foo\"");
    }

    #[test]
    fn round_trips_string_literal_with_leading_newline() {
        round_trip_entry_point_literal("\"\\\nfoo\"", "\"\nfoo\"");
    }

    #[test]
    fn round_trips_string_literal_with_utf8_escape_prefix() {
        round_trip_entry_point_literal("\"\\亲\"", "\"亲\"");
    }

    #[test]
    fn disassembly_formats_literal_strings_with_embedded_newlines() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let fn_ty = builder.type_function(void, vec![]);
        let func = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .expect("function");
        builder.begin_block(None).expect("entry block");
        builder.ret().expect("return");
        builder.end_function().expect("end function");
        builder.name(func, "foo\nbar");
        let module = builder.module();
        let binary = module.assemble();
        let text =
            disassemble_binary(&binary, BinaryToTextOptions::NO_HEADER).expect("disassemble");
        assert!(text.contains("\"foo\nbar\""), "{text:?}");
    }

    #[test]
    fn friendly_name_builder_assigns_type_names() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let int = builder.type_int(32, 1);
        builder.constant_bit32(int, 1);
        let module = builder.module();
        let type_table = super::TypeTable::from_module(&module);
        let names = super::FriendlyNameTable::from_module(&module, &type_table);
        assert_eq!(names.lookup(void), Some("void"));
        assert_eq!(names.lookup(int), Some("int"));
    }

    #[test]
    fn friendly_names_match_opt_fixture() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let int = builder.type_int(32, 1);
        let ptr = builder.type_pointer(None, StorageClass::Function, int);
        let fn_type = builder.type_function(void, vec![ptr]);
        let main = builder
            .begin_function(void, None, FunctionControl::NONE, fn_type)
            .expect("function");
        builder.name(main, "main");
        builder.function_parameter(ptr).expect("param");
        builder.begin_block(None).expect("block");
        builder.ret().expect("ret");
        builder.end_function().expect("end");
        let binary = builder.module().assemble();
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(
            disassembled.contains("%int = OpTypeInt 32 1"),
            "{disassembled}"
        );
        assert!(
            disassembled.contains("%_ptr_Function_int = OpTypePointer Function %int"),
            "{disassembled}"
        );
    }

    #[test]
    fn friendly_names_match_binary_to_text_fixture() {
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled =
            disassemble_binary(FRIENDLY_NAME_SAMPLE_BINARY, options).expect("disassemble");
        assert!(
            disassembled.contains("%uint = OpTypeInt 32 0"),
            "{disassembled}"
        );
        assert!(
            disassembled.contains("%uint_42 = OpConstant %uint 42"),
            "{disassembled}"
        );
    }

    #[test]
    fn disassembly_respects_raw_id_option() {
        let options = BinaryToTextOptions::NO_HEADER;
        let disassembled =
            disassemble_binary(FRIENDLY_NAME_SAMPLE_BINARY, options).expect("disassemble");
        assert!(
            disassembled.contains("%1 = OpTypeInt 32 0"),
            "{disassembled}"
        );
    }

    #[test]
    fn disassembly_omits_newline_when_output_empty() {
        let binary = vec![0x0723_0203, 0x0001_0000, 0, 1, 0];
        let options = BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert_eq!(disassembled, "");
    }

    #[test]
    fn indent_alignment_matches_legacy_formatter() {
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::INDENT;
        let disassembled =
            disassemble_binary(FRIENDLY_NAME_SAMPLE_BINARY, options).expect("disassemble");
        let type_line = disassembled
            .lines()
            .find(|line| line.contains("OpTypeInt 32 0"))
            .expect("type line present");
        assert!(type_line.starts_with("          %1 ="), "{type_line}");
    }

    #[test]
    fn friendly_names_match_pass_fixture_sample() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        builder.name(void, "void_t");
        let fn_ty = builder.type_function(void, vec![]);
        builder.name(fn_ty, "fn_t");
        let bool_ty = builder.type_bool();
        let out_ptr = builder.type_pointer(None, StorageClass::Output, bool_ty);
        let flag = builder.variable(out_ptr, None, StorageClass::Output, None);
        builder.name(flag, "flag");
        builder.decorate(flag, Decoration::RelaxedPrecision, []);

        let main_fn = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .expect("function");
        builder.name(main_fn, "main");
        builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end");
        builder.entry_point(ExecutionModel::Fragment, main_fn, "main", vec![flag]);
        builder.decorate(main_fn, Decoration::RelaxedPrecision, []);

        let module = builder.module();
        let binary = module.assemble();
        let options = BinaryToTextOptions::NO_HEADER
            | BinaryToTextOptions::COMMENT
            | BinaryToTextOptions::INDENT
            | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        let expected = r#"               OpCapability Shader
               OpMemoryModel Logical Simple
               OpEntryPoint Fragment %main "main" %flag

               ; Debug Information
               OpName %void_t "void_t"              ; id %1
               OpName %fn_t "fn_t"                  ; id %2
               OpName %flag "flag"                  ; id %5
               OpName %main "main"                  ; id %6

               ; Annotations
               OpDecorate %flag RelaxedPrecision
               OpDecorate %main RelaxedPrecision

               ; Types, variables and constants
     %void_t = OpTypeVoid
       %fn_t = OpTypeFunction %void_t
       %bool = OpTypeBool
%_ptr_Output_bool = OpTypePointer Output %bool
       %flag = OpVariable %_ptr_Output_bool Output  ; RelaxedPrecision

               ; Function 6
       %main = OpFunction %void_t None %fn_t        ; RelaxedPrecision
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
"#;
        assert_eq!(disassembled, expected);
    }

    #[test]
    fn friendly_names_include_builtin_decorations() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let fn_type = builder.type_function(void, vec![]);
        let uint = builder.type_int(32, 0);
        let vec3 = builder.type_vector(uint, 3);
        let input_ptr = builder.type_pointer(None, StorageClass::Input, vec3);
        let builtin = builder.variable(input_ptr, None, StorageClass::Input, None);
        builder.decorate(
            builtin,
            Decoration::BuiltIn,
            [dr::Operand::BuiltIn(BuiltIn::GlobalInvocationId)],
        );
        builder
            .begin_function(void, None, FunctionControl::NONE, fn_type)
            .expect("function");
        builder.begin_block(None).expect("block");
        builder.ret().expect("ret");
        builder.end_function().expect("end");

        let binary = builder.module().assemble();
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(
            disassembled.contains("%gl_GlobalInvocationID = OpVariable"),
            "{disassembled}"
        );
    }

    #[test]
    fn disassembly_normalizes_execution_model_aliases() {
        let mut builder = Builder::new();
        builder.capability(Capability::RayTracingNV);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let fn_type = builder.type_function(void, vec![]);
        let main = builder
            .begin_function(void, None, FunctionControl::NONE, fn_type)
            .expect("function");
        builder.begin_block(None).expect("block");
        builder.ret().expect("ret");
        builder.end_function().expect("end");
        builder.entry_point(ExecutionModel::RayGenerationNV, main, "main", vec![]);
        let binary = builder.module().assemble();
        let options = BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(
            disassembled.contains("OpEntryPoint RayGenerationKHR"),
            "{disassembled}"
        );
    }

    #[test]
    fn disassembly_normalizes_storage_class_aliases() {
        let mut builder = Builder::new();
        builder.capability(Capability::RayTracingNV);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let float = builder.type_float(32, None);
        let ptr = builder.type_pointer(None, StorageClass::CallableDataNV, float);
        let payload = builder.variable(ptr, None, StorageClass::CallableDataNV, None);
        builder.name(payload, "payload");
        let binary = builder.module().assemble();
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(
            disassembled
                .contains("%_ptr_CallableDataKHR_float = OpTypePointer CallableDataKHR %float"),
            "{disassembled}"
        );
        assert!(
            disassembled
                .contains("%payload = OpVariable %_ptr_CallableDataKHR_float CallableDataKHR"),
            "{disassembled}"
        );
    }

    #[test]
    fn disassembly_preserves_trailing_opline_order() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let file = builder.string("file.ext");
        let void = builder.type_void();
        let fn_ty = builder.type_function(void, vec![]);
        let func = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .expect("function");
        let label = builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end");
        builder.line(file, 1, 0);
        let binary = builder.module().assemble();
        let options = BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        let expected = format!(
            "OpCapability Shader\nOpMemoryModel Logical Simple\n%{} = OpString \"file.ext\"\n%{} = OpTypeVoid\n%{} = OpTypeFunction %{}\n%{} = OpFunction %{} None %{}\n%{} = OpLabel\nOpReturn\nOpFunctionEnd\nOpLine %{} 1 0\n",
            file, void, fn_ty, void, func, void, fn_ty, label, file
        );
        assert_eq!(disassembled, expected);
    }

    #[test]
    fn disassembly_preserves_prefunction_opline_order() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let file = builder.string("file.ext");
        let void = builder.type_void();
        let fn_ty = builder.type_function(void, vec![]);
        let func = builder
            .begin_function(void, None, FunctionControl::NONE, fn_ty)
            .expect("function");
        builder.line(file, 10, 10);
        builder.line(file, 20, 20);
        let label = builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end");
        let module = builder.module();
        let mut binary = module.assemble();
        let mut prelude_words = Vec::new();
        let mut scan = super::HEADER_WORD_COUNT;
        while scan < binary.len() {
            let word = binary[scan];
            let word_count = (word >> 16) as usize;
            let opcode = word & 0xFFFF;
            if opcode == rspirv::spirv::Op::Line as u32 {
                prelude_words.extend_from_slice(&binary[scan..scan + word_count]);
                binary.drain(scan..scan + word_count);
            } else {
                scan += word_count;
            }
        }
        let mut insert = super::HEADER_WORD_COUNT;
        while insert < binary.len() {
            let word = binary[insert];
            let word_count = (word >> 16) as usize;
            let opcode = word & 0xFFFF;
            insert += word_count;
            if opcode == rspirv::spirv::Op::Function as u32 {
                break;
            }
        }
        binary.splice(insert..insert, prelude_words.iter().cloned());
        let options = BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        let expected = format!(
            "OpCapability Shader\nOpMemoryModel Logical Simple\n%{} = OpString \"file.ext\"\n%{} = OpTypeVoid\n%{} = OpTypeFunction %{}\n%{} = OpFunction %{} None %{}\nOpLine %{} 10 10\nOpLine %{} 20 20\n%{} = OpLabel\nOpReturn\nOpFunctionEnd\n",
            file, void, fn_ty, void, func, void, fn_ty, file, file, label
        );
        assert_eq!(disassembled, expected);
    }

    #[test]
    fn disassembly_formats_integer_constants_using_type() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let int = builder.type_int(32, 1);
        let uint = builder.type_int(32, 0);
        builder.constant_bit32(int, (-1867i32) as u32);
        builder.constant_bit32(uint, 1867);
        let binary = builder.module().assemble();
        let options = BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.contains("-1867"));
        assert!(disassembled.contains(" 1867"));
    }

    #[test]
    fn friendly_names_rename_type_ids() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let void_fn = builder.type_function(void, vec![]);
        builder
            .begin_function(void, None, FunctionControl::NONE, void_fn)
            .expect("function");
        builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end");
        let module = builder.module();
        let binary = module.assemble();
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(
            disassembled.contains("%void = OpTypeVoid"),
            "{disassembled}"
        );
    }

    #[test]
    fn friendly_names_deduplicate_opname_collisions() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let fn_type = builder.type_function(void, vec![]);

        let first = builder
            .begin_function(void, None, FunctionControl::NONE, fn_type)
            .expect("function");
        builder.name(first, "foo");
        builder.begin_block(None).expect("block");
        builder.ret().expect("ret");
        builder.end_function().expect("end");

        let second = builder
            .begin_function(void, None, FunctionControl::NONE, fn_type)
            .expect("function");
        builder.name(second, "foo");
        builder.begin_block(None).expect("block");
        builder.ret().expect("ret");
        builder.end_function().expect("end");

        let module = builder.module();
        let binary = module.assemble();
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.contains("%foo = OpFunction"), "{disassembled}");
        assert!(
            disassembled.contains("%foo_0 = OpFunction"),
            "{disassembled}"
        );
    }

    #[test]
    fn friendly_names_include_pipe_access_qualifier() {
        let mut builder = Builder::new();
        builder.capability(Capability::Pipes);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        builder.type_pipe(AccessQualifier::ReadOnly);
        let module = builder.module();
        let binary = module.assemble();
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(
            disassembled.contains("%PipeReadOnly = OpTypePipe ReadOnly"),
            "{disassembled}"
        );
    }

    #[test]
    fn disassembly_formats_float_constants() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let f16 = builder.type_float(16, None);
        let f32 = builder.type_float(32, None);
        builder.constant_bit32(f32, (-3.125f32).to_bits());
        builder.constant_bit32(f16, 0x7e00);
        let binary = builder.module().assemble();
        let options = BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.contains("-3.125"), "{disassembled}");
        assert!(
            disassembled
                .lines()
                .any(|line| line.contains("OpConstant") && line.contains("0x")),
            "{disassembled}"
        );
    }

    #[test]
    fn disassembly_formats_special_float_values_as_hex() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let float = builder.type_float(32, None);
        builder.constant_bit32(float, 0x7fc00000); // NaN
        builder.constant_bit32(float, 0x7f800000); // +Inf
        builder.constant_bit32(float, 0xff800000); // -Inf
        let binary = builder.module().assemble();
        let disassembled =
            disassemble_binary(&binary, BinaryToTextOptions::NO_HEADER).expect("disassemble");
        assert!(disassembled.contains("0x1.8p+128"));
        assert!(disassembled.contains("0x1p+128"));
        assert!(disassembled.contains("-0x1p+128"));
    }

    #[test]
    fn hex_option_overrides_constant_formatting() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%int = OpTypeInt 32 1",
            "%val = OpConstant %int -42",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble text");
        let options = BinaryToTextOptions::HEX | BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.contains("0xffffffd6"), "{disassembled}");
    }

    #[test]
    fn nested_indent_inserts_blank_line_before_labels() {
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
        let binary = assemble_text(&text).expect("assemble text");
        let options = BinaryToTextOptions::NO_HEADER
            | BinaryToTextOptions::INDENT
            | BinaryToTextOptions::NESTED_INDENT;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        let lines: Vec<&str> = disassembled.lines().collect();
        let mut has_blank = false;
        for pair in lines.windows(2) {
            if pair[1].contains("OpLabel") && pair[0].trim().is_empty() {
                has_blank = true;
                break;
            }
        }
        assert!(
            has_blank,
            "expected blank line before label:\n{disassembled}"
        );
    }

    #[test]
    fn disassembly_handles_conditional_extension_intel() {
        let text = disassemble_with_options(
            CONDITIONAL_EXTENSION_SAMPLE_BINARY,
            BinaryToTextOptions::INDENT,
        );
        assert!(
            text.contains("OpConditionalExtensionINTEL %2 \"SPV_INTEL_function_variants\""),
            "{text}"
        );
    }

    fn spaces_after_equals(line: &str) -> Option<usize> {
        let (_, rest) = line.split_once('=')?;
        Some(rest.chars().take_while(|ch| *ch == ' ').count())
    }

    fn leading_spaces(line: &str) -> usize {
        line.chars().take_while(|ch| ch.is_whitespace()).count()
    }

    fn build_selection_module(permuted: bool) -> Vec<u32> {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = builder.type_void();
        let bool_type = builder.type_bool();
        let void_fn = builder.type_function(void, vec![]);
        let true_id = builder.constant_true(bool_type);
        builder
            .begin_function(void, None, FunctionControl::NONE, void_fn)
            .expect("function");
        let entry_label = builder.begin_block(None).expect("entry block");
        builder.name(entry_label, "entry");
        let merge_label = builder.id();
        let then_label = builder.id();
        builder.name(merge_label, "merge");
        builder.name(then_label, "then");
        builder
            .selection_merge(merge_label, SelectionControl::NONE)
            .expect("merge");
        builder
            .branch_conditional(true_id, then_label, merge_label, std::iter::empty())
            .expect("branch conditional");
        builder.begin_block(Some(merge_label)).expect("merge block");
        builder.ret().expect("return");
        builder.begin_block(Some(then_label)).expect("then block");
        builder.branch(merge_label).expect("branch");
        builder.end_function().expect("end function");

        let mut module = builder.module();
        if permuted {
            if let Some(function) = module.functions.get_mut(0) {
                if function.blocks.len() >= 3 {
                    let merge_block = function.blocks.remove(function.blocks.len() - 1);
                    function.blocks.insert(1, merge_block);
                }
            }
        }
        module.assemble()
    }

    #[cfg(test)]
    fn take_print_log() -> Vec<String> {
        super::PRINT_LOG.lock().unwrap().drain(..).collect()
    }
}

#[derive(Default)]
struct FriendlyNameTable {
    names: HashMap<u32, String>,
}

impl FriendlyNameTable {
    fn from_module(module: &dr::Module, type_table: &TypeTable) -> Self {
        let mut builder = FriendlyNameBuilder::new(type_table);
        visit_module_instructions(module, |instruction| builder.observe(instruction));
        Self {
            names: builder.finish(),
        }
    }

    fn lookup(&self, id: u32) -> Option<&str> {
        self.names.get(&id).map(|name| name.as_str())
    }

    fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

fn visit_module_instructions<'a>(module: &'a dr::Module, mut visit: impl FnMut(&'a Instruction)) {
    for instruction in &module.capabilities {
        visit(instruction);
    }
    for instruction in &module.extensions {
        visit(instruction);
    }
    for instruction in &module.ext_inst_imports {
        visit(instruction);
    }
    if let Some(inst) = module.memory_model.as_ref() {
        visit(inst);
    }
    for instruction in &module.entry_points {
        visit(instruction);
    }
    for instruction in &module.execution_modes {
        visit(instruction);
    }
    for instruction in &module.debug_string_source {
        visit(instruction);
    }
    for instruction in &module.debug_names {
        visit(instruction);
    }
    for instruction in &module.debug_module_processed {
        visit(instruction);
    }
    for instruction in &module.annotations {
        visit(instruction);
    }
    for instruction in &module.types_global_values {
        visit(instruction);
    }
    for function in &module.functions {
        if let Some(def) = function.def.as_ref() {
            visit(def);
        }
        for parameter in &function.parameters {
            visit(parameter);
        }
        for block in &function.blocks {
            if let Some(label) = block.label.as_ref() {
                visit(label);
            }
            for instruction in &block.instructions {
                visit(instruction);
            }
        }
        if let Some(end) = function.end.as_ref() {
            visit(end);
        }
    }
}

const FP_ENCODING_BFLOAT16_KHR: u32 = 0;
const FP_ENCODING_FLOAT8_E4M3_EXT: u32 = 4214;
const FP_ENCODING_FLOAT8_E5M2_EXT: u32 = 4215;

struct FriendlyNameBuilder<'a> {
    names: HashMap<u32, String>,
    used: HashMap<String, u32>,
    type_table: &'a TypeTable,
}

impl<'a> FriendlyNameBuilder<'a> {
    fn new(type_table: &'a TypeTable) -> Self {
        Self {
            names: HashMap::new(),
            used: HashMap::new(),
            type_table,
        }
    }

    fn observe(&mut self, instruction: &Instruction) {
        use spirv::Op;
        match instruction.class.opcode {
            Op::Name => self.handle_name(instruction),
            Op::Decorate => self.handle_decorate(instruction),
            Op::TypeVoid => self.assign_result_name(instruction, "void"),
            Op::TypeBool => self.assign_result_name(instruction, "bool"),
            Op::TypeInt => self.handle_type_int(instruction),
            Op::TypeFloat => self.handle_type_float(instruction),
            Op::TypeVector => self.handle_type_vector(instruction),
            Op::TypeMatrix => self.handle_type_matrix(instruction),
            Op::TypeArray => self.handle_type_array(instruction, "_arr_"),
            Op::TypeRuntimeArray => self.handle_runtime_array(instruction, "_runtimearr_"),
            Op::TypePointer => self.handle_type_pointer(instruction),
            Op::TypePipe => self.handle_type_pipe(instruction),
            Op::TypeEvent => self.assign_result_name(instruction, "Event"),
            Op::TypeDeviceEvent => self.assign_result_name(instruction, "DeviceEvent"),
            Op::TypeReserveId => self.assign_result_name(instruction, "ReserveId"),
            Op::TypeQueue => self.assign_result_name(instruction, "Queue"),
            Op::TypeOpaque => self.handle_type_opaque(instruction),
            Op::TypePipeStorage => self.assign_result_name(instruction, "PipeStorage"),
            Op::TypeNamedBarrier => self.assign_result_name(instruction, "NamedBarrier"),
            Op::TypeStruct => self.handle_type_struct(instruction),
            Op::ConstantTrue => self.assign_result_name(instruction, "true"),
            Op::ConstantFalse => self.assign_result_name(instruction, "false"),
            Op::Constant => self.handle_constant(instruction),
            _ => {
                if let Some(id) = instruction.result_id {
                    self.ensure_name(id);
                }
            }
        }
    }

    fn handle_name(&mut self, instruction: &Instruction) {
        if let (Some(Operand::IdRef(target)), Some(Operand::LiteralString(raw_name))) =
            (instruction.operands.first(), instruction.operands.get(1))
        {
            self.assign_name(*target, raw_name);
        }
    }

    fn handle_decorate(&mut self, instruction: &Instruction) {
        if instruction.operands.len() < 2 {
            return;
        }
        let Operand::IdRef(target) = instruction.operands[0] else {
            return;
        };
        if let Operand::Decoration(decoration) = instruction.operands[1] {
            if decoration == spirv::Decoration::BuiltIn {
                let built_in_operand = instruction.operands.get(2);
                if let Some(built_in) = built_in_operand.and_then(extract_built_in) {
                    self.assign_builtin_name(target, built_in);
                }
            }
        }
    }

    fn handle_type_int(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        let Some(width) = instruction
            .operands
            .first()
            .and_then(literal_operand_to_u32)
        else {
            return;
        };
        let is_signed = instruction
            .operands
            .get(1)
            .and_then(literal_operand_to_u32)
            .map(|value| value != 0)
            .unwrap_or(true);
        match width {
            8 => self.assign_name(result_id, if is_signed { "char" } else { "uchar" }),
            16 => self.assign_name(result_id, if is_signed { "short" } else { "ushort" }),
            32 => self.assign_name(result_id, if is_signed { "int" } else { "uint" }),
            64 => self.assign_name(result_id, if is_signed { "long" } else { "ulong" }),
            _ => {
                let name = if is_signed {
                    format!("i{width}")
                } else {
                    format!("u{width}")
                };
                self.assign_name(result_id, &name);
            }
        }
    }

    fn handle_type_float(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        let Some(width) = instruction
            .operands
            .first()
            .and_then(literal_operand_to_u32)
        else {
            return;
        };
        if let Some(encoded) = instruction.operands.get(1).and_then(literal_operand_to_u32) {
            if let Some(name) = fp_encoding_name(encoded) {
                self.assign_name(result_id, name);
                return;
            }
        }
        match width {
            16 => self.assign_name(result_id, "half"),
            32 => self.assign_name(result_id, "float"),
            64 => self.assign_name(result_id, "double"),
            _ => {
                let name = format!("fp{width}");
                self.assign_name(result_id, &name);
            }
        }
    }

    fn handle_type_vector(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let (Some(component), Some(count)) = (
            instruction.operands.first().and_then(extract_id_ref),
            instruction.operands.get(1).and_then(literal_operand_to_u32),
        ) {
            let element_name = self.lookup_name(component);
            let name = format!("v{count}{element_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_type_matrix(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let (Some(column_type), Some(count)) = (
            instruction.operands.first().and_then(extract_id_ref),
            instruction.operands.get(1).and_then(literal_operand_to_u32),
        ) {
            let column_name = self.lookup_name(column_type);
            let name = format!("mat{count}{column_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_type_array(&mut self, instruction: &Instruction, prefix: &str) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let (Some(element), Some(length)) = (
            instruction.operands.first().and_then(extract_id_ref),
            instruction.operands.get(1).and_then(extract_id_ref),
        ) {
            let element_name = self.lookup_name(element);
            let length_name = self.lookup_name(length);
            let name = format!("{prefix}{element_name}_{length_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_runtime_array(&mut self, instruction: &Instruction, prefix: &str) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let Some(element) = instruction.operands.first().and_then(extract_id_ref) {
            let element_name = self.lookup_name(element);
            let name = format!("{prefix}{element_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_type_pointer(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if instruction.operands.len() < 2 {
            return;
        }
        let storage_class = instruction.operands.first().and_then(extract_storage_class);
        let pointee = instruction.operands.get(1).and_then(extract_id_ref);
        if let (Some(class), Some(pointee_id)) = (storage_class, pointee) {
            let pointee_name = self.lookup_name(pointee_id);
            let name = format!("_ptr_{class}_{pointee_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_type_pipe(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        let Some(access) = instruction
            .operands
            .first()
            .and_then(extract_access_qualifier)
        else {
            return;
        };
        let name = format!("Pipe{access}");
        self.assign_name(result_id, &name);
    }

    fn handle_type_opaque(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let Some(name) = instruction
            .operands
            .first()
            .and_then(extract_literal_string)
        {
            let formatted = format!("Opaque_{}", sanitize_identifier(name));
            self.assign_name(result_id, &formatted);
        }
    }

    fn handle_type_struct(&mut self, instruction: &Instruction) {
        if let Some(result_id) = instruction.result_id {
            let name = format!("_struct_{result_id}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_constant(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        let Some(type_id) = instruction.result_type else {
            return;
        };
        if let Some(value) = self.constant_literal(instruction) {
            let base = self.lookup_name(type_id);
            let sanitized = value.replace('-', "n");
            let name = format!("{base}_{sanitized}");
            self.assign_name(result_id, &name);
        }
    }

    fn constant_literal(&self, instruction: &Instruction) -> Option<String> {
        let type_id = instruction.result_type?;
        let operand = instruction.operands.first()?;
        let type_info = self.type_table.get(type_id)?;
        match type_info {
            TypeInfo::Int { width, signed } => {
                literal_operand_bits(operand).map(|bits| format_integer_bits(bits, *width, *signed))
            }
            TypeInfo::Float { width } => match width {
                16 => literal_operand_bits(operand)
                    .map(|bits| format_hex_float(bits & 0xffff, &HEX_FLOAT_F16)),
                32 => literal_operand_bits(operand).map(|bits| format_f32_literal(bits as u32)),
                64 => literal_operand_bits(operand).map(format_f64_literal),
                _ => None,
            },
        }
    }

    fn assign_result_name(&mut self, instruction: &Instruction, name: &str) {
        if let Some(result_id) = instruction.result_id {
            self.assign_name(result_id, name);
        }
    }

    fn assign_builtin_name(&mut self, target: u32, built_in: spirv::BuiltIn) {
        if let Some(name) = builtin_name(built_in) {
            self.assign_name(target, name);
        }
    }

    fn assign_name(&mut self, id: u32, raw: &str) {
        if self.names.contains_key(&id) {
            return;
        }
        let base = sanitize_identifier(raw);
        let name = self.unique_name(base);
        self.names.insert(id, name);
    }

    fn ensure_name(&mut self, id: u32) {
        if self.names.contains_key(&id) {
            return;
        }
        let name = id.to_string();
        self.used.entry(name.clone()).or_insert(1);
        self.names.insert(id, name);
    }

    fn lookup_name(&mut self, id: u32) -> String {
        if !self.names.contains_key(&id) {
            self.ensure_name(id);
        }
        self.names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| id.to_string())
    }

    fn unique_name(&mut self, base: String) -> String {
        let normalized = if base.is_empty() {
            "_".to_owned()
        } else {
            base
        };
        let counter = self.used.entry(normalized.clone()).or_insert(0);
        let result = if *counter == 0 {
            normalized.clone()
        } else {
            format!("{}_{}", normalized, *counter - 1)
        };
        *counter += 1;
        result
    }

    fn finish(self) -> HashMap<u32, String> {
        self.names
    }
}

fn sanitize_identifier(raw: &str) -> String {
    if raw.is_empty() {
        return "_".to_string();
    }
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn extract_id_ref(operand: &Operand) -> Option<u32> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

fn extract_literal_string(operand: &Operand) -> Option<&str> {
    if let Operand::LiteralString(value) = operand {
        Some(value.as_str())
    } else {
        None
    }
}

fn extract_built_in(operand: &Operand) -> Option<spirv::BuiltIn> {
    match operand {
        Operand::BuiltIn(value) => Some(*value),
        _ => literal_operand_to_u32(operand).and_then(spirv::BuiltIn::from_u32),
    }
}

fn extract_storage_class(operand: &Operand) -> Option<String> {
    match operand {
        Operand::StorageClass(class) => Some(
            canonical_storage_class(*class)
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("{:?}", class)),
        ),
        _ => literal_operand_to_u32(operand).map(|value| {
            if let Some(class) = spirv::StorageClass::from_u32(value) {
                canonical_storage_class(class)
                    .map(|name| name.to_string())
                    .unwrap_or_else(|| format!("{:?}", class))
            } else {
                format!("StorageClass{value}")
            }
        }),
    }
}

fn extract_access_qualifier(operand: &Operand) -> Option<String> {
    match operand {
        Operand::AccessQualifier(qualifier) => Some(format!("{:?}", qualifier)),
        _ => literal_operand_to_u32(operand).map(|value| {
            spirv::AccessQualifier::from_u32(value)
                .map(|qualifier| format!("{:?}", qualifier))
                .unwrap_or_else(|| format!("AccessQualifier{value}"))
        }),
    }
}

fn literal_operand_bits(operand: &Operand) -> Option<u64> {
    match operand {
        Operand::LiteralBit32(value) => Some(u64::from(*value)),
        Operand::LiteralBit64(value) => Some(*value),
        _ => None,
    }
}

fn section_heading(section: ModuleSection) -> Option<&'static str> {
    match section {
        ModuleSection::Debug => Some("Debug Information"),
        ModuleSection::Annotations => Some("Annotations"),
        ModuleSection::Types => Some("Types, variables and constants"),
        _ => None,
    }
}

fn append_section_heading(text: &mut String, heading: &str, indent: bool) {
    if !text.is_empty() {
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push('\n');
    }
    if indent {
        text.push_str(&" ".repeat(STANDARD_INDENT_COLUMN));
    }
    text.push_str("; ");
    text.push_str(heading);
    text.push('\n');
}

fn append_function_heading(
    text: &mut String,
    result_id: Option<u32>,
    indent: bool,
    extra_spacing: bool,
) {
    let Some(id) = result_id else {
        return;
    };
    if !text.is_empty() {
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push('\n');
        if extra_spacing {
            text.push('\n');
        }
    }
    if indent {
        text.push_str(&" ".repeat(STANDARD_INDENT_COLUMN));
    }
    text.push_str("; Function ");
    text.push_str(&id.to_string());
    text.push('\n');
}

fn fp_encoding_name(value: u32) -> Option<&'static str> {
    match value {
        FP_ENCODING_BFLOAT16_KHR => Some("bfloat16"),
        FP_ENCODING_FLOAT8_E4M3_EXT => Some("fp8e4m3"),
        FP_ENCODING_FLOAT8_E5M2_EXT => Some("fp8e5m2"),
        _ => None,
    }
}

fn builtin_name(built_in: spirv::BuiltIn) -> Option<&'static str> {
    match built_in {
        spirv::BuiltIn::Position => Some("gl_Position"),
        spirv::BuiltIn::PointSize => Some("gl_PointSize"),
        spirv::BuiltIn::ClipDistance => Some("gl_ClipDistance"),
        spirv::BuiltIn::CullDistance => Some("gl_CullDistance"),
        spirv::BuiltIn::VertexId => Some("gl_VertexID"),
        spirv::BuiltIn::InstanceId => Some("gl_InstanceID"),
        spirv::BuiltIn::PrimitiveId => Some("gl_PrimitiveID"),
        spirv::BuiltIn::InvocationId => Some("gl_InvocationID"),
        spirv::BuiltIn::Layer => Some("gl_Layer"),
        spirv::BuiltIn::ViewportIndex => Some("gl_ViewportIndex"),
        spirv::BuiltIn::TessLevelOuter => Some("gl_TessLevelOuter"),
        spirv::BuiltIn::TessLevelInner => Some("gl_TessLevelInner"),
        spirv::BuiltIn::TessCoord => Some("gl_TessCoord"),
        spirv::BuiltIn::PatchVertices => Some("gl_PatchVertices"),
        spirv::BuiltIn::FragCoord => Some("gl_FragCoord"),
        spirv::BuiltIn::PointCoord => Some("gl_PointCoord"),
        spirv::BuiltIn::FrontFacing => Some("gl_FrontFacing"),
        spirv::BuiltIn::SampleId => Some("gl_SampleID"),
        spirv::BuiltIn::SamplePosition => Some("gl_SamplePosition"),
        spirv::BuiltIn::SampleMask => Some("gl_SampleMask"),
        spirv::BuiltIn::FragDepth => Some("gl_FragDepth"),
        spirv::BuiltIn::HelperInvocation => Some("gl_HelperInvocation"),
        spirv::BuiltIn::NumWorkgroups => Some("gl_NumWorkGroups"),
        spirv::BuiltIn::WorkgroupSize => Some("gl_WorkGroupSize"),
        spirv::BuiltIn::WorkgroupId => Some("gl_WorkGroupID"),
        spirv::BuiltIn::LocalInvocationId => Some("gl_LocalInvocationID"),
        spirv::BuiltIn::GlobalInvocationId => Some("gl_GlobalInvocationID"),
        spirv::BuiltIn::LocalInvocationIndex => Some("gl_LocalInvocationIndex"),
        spirv::BuiltIn::VertexIndex => Some("gl_VertexIndex"),
        spirv::BuiltIn::InstanceIndex => Some("gl_InstanceIndex"),
        spirv::BuiltIn::BaseVertex => Some("gl_BaseVertex"),
        spirv::BuiltIn::BaseInstance => Some("gl_BaseInstance"),
        spirv::BuiltIn::WorkDim => Some("WorkDim"),
        spirv::BuiltIn::GlobalSize => Some("GlobalSize"),
        spirv::BuiltIn::EnqueuedWorkgroupSize => Some("EnqueuedWorkgroupSize"),
        spirv::BuiltIn::GlobalOffset => Some("GlobalOffset"),
        spirv::BuiltIn::GlobalLinearId => Some("GlobalLinearId"),
        spirv::BuiltIn::SubgroupSize => Some("SubgroupSize"),
        spirv::BuiltIn::SubgroupMaxSize => Some("SubgroupMaxSize"),
        spirv::BuiltIn::NumSubgroups => Some("NumSubgroups"),
        spirv::BuiltIn::NumEnqueuedSubgroups => Some("NumEnqueuedSubgroups"),
        spirv::BuiltIn::SubgroupId => Some("SubgroupId"),
        spirv::BuiltIn::SubgroupLocalInvocationId => Some("SubgroupLocalInvocationId"),
        spirv::BuiltIn::SubgroupEqMask => Some("SubgroupEqMaskKHR"),
        spirv::BuiltIn::SubgroupGeMask => Some("SubgroupGeMaskKHR"),
        spirv::BuiltIn::SubgroupGtMask => Some("SubgroupGtMaskKHR"),
        spirv::BuiltIn::SubgroupLeMask => Some("SubgroupLeMaskKHR"),
        spirv::BuiltIn::SubgroupLtMask => Some("SubgroupLtMaskKHR"),
        _ => None,
    }
}
