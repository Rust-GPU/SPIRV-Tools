use rspirv::binary::{Disassemble, Parser};
use rspirv::dr::{self, Instruction, Loader, ModuleHeader, Operand};
use rspirv::grammar::{GlslStd450InstructionTable, OpenCLStd100InstructionTable};
use rspirv::spirv;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::mem::size_of_val;
use std::num::FpCategory;
use std::slice;
use thiserror::Error;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

use crate::assembly::{BinaryToTextOptions, ExtInstImportInfo, ExtInstSetKind};
use crate::diagnostic::{DiagnosticMessage, MessagePosition};
use crate::message::MessageLevel;

const SUPPORTED_OPTION_BITS: u32 = BinaryToTextOptions::NO_HEADER.bits()
    | BinaryToTextOptions::PRINT.bits()
    | BinaryToTextOptions::SHOW_BYTE_OFFSET.bits()
    | BinaryToTextOptions::INDENT.bits()
    | BinaryToTextOptions::FRIENDLY_NAMES.bits()
    | BinaryToTextOptions::NESTED_INDENT.bits()
    | BinaryToTextOptions::COMMENT.bits()
    | BinaryToTextOptions::REORDER_BLOCKS.bits()
    | BinaryToTextOptions::COLOR.bits()
    | BinaryToTextOptions::HEX.bits();
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

    let mut loader = Loader::new();
    Parser::new(words_as_bytes(words), &mut loader)
        .parse()
        .map_err(|error| DisassemblyError::parse_error(error.to_string()))?;
    let mut module = loader.module();
    update_module_header(&mut module, words);
    let offsets = collect_instruction_offsets(words);
    let mut text = render_module_text(&module, &offsets, &formatting);
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
        Ok(Self {
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
        })
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

fn render_module_text(module: &dr::Module, offsets: &[u32], options: &FormattingOptions) -> String {
    let instructions = collect_module_instructions(module, options.reorder_blocks);
    let type_table = TypeTable::from_module(module);
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
    if instructions.len() != offsets.len() {
        return module.disassemble();
    }

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

fn render_instructions(
    instructions: &[InstructionRecord],
    offsets: &[u32],
    options: &FormattingOptions,
    type_table: &TypeTable,
    ext_inst_table: &ExtInstTable,
    friendly_names: Option<&FriendlyNameTable>,
    comment_collector: &mut CommentCollector,
) -> String {
    let mut aligner = CommentAligner::new();
    let mut text = String::new();
    let mut current_section = ModuleSection::Header;
    for (index, (record, &offset)) in instructions.iter().zip(offsets).enumerate() {
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

        let mut block_indent = 0usize;
        if options.nested_indent {
            let nest_level = (record.depth as usize) * 2;
            block_indent += nest_level * BLOCK_NEST_INDENT;
            if record.is_block_body() {
                block_indent += BLOCK_BODY_INDENT_OFFSET;
            }
        }

        if options.indent {
            apply_indent(&mut line, 0);
        }
        insert_block_indent(&mut line, block_indent);
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
) -> String {
    let operands = format_ext_inst_operands(instruction, literal_format, ext_inst_table)
        .or_else(|| format_constant_operands(instruction, literal_format, type_table))
        .unwrap_or_else(|| disassemble_operands(&instruction.operands, literal_format));
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
    let format = match CString::new(format!("%.{digits}g")) {
        Ok(fmt) => fmt,
        Err(_) => return value.to_string(),
    };
    let mut stack = [0u8; 128];
    unsafe {
        let len = libc::snprintf(
            stack.as_mut_ptr() as *mut libc::c_char,
            stack.len(),
            format.as_ptr(),
            value as libc::c_double,
        );
        if len < 0 {
            return value.to_string();
        }
        let len = len as usize;
        if len < stack.len() {
            return String::from_utf8_lossy(&stack[..len]).into_owned();
        }
        let mut heap = vec![0u8; len + 1];
        let retry = libc::snprintf(
            heap.as_mut_ptr() as *mut libc::c_char,
            heap.len(),
            format.as_ptr(),
            value as libc::c_double,
        );
        if retry < 0 {
            value.to_string()
        } else {
            String::from_utf8_lossy(&heap[..retry as usize]).into_owned()
        }
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

fn collect_module_instructions(
    module: &dr::Module,
    reorder_blocks: bool,
) -> Vec<InstructionRecord<'_>> {
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
    for instruction in &module.types_global_values {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Types,
        ));
    }
    for function in &module.functions {
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

        let mut tracker = MergeTracker::new();
        let block_indices = if reorder_blocks {
            reorder_function_blocks(function)
        } else {
            (0..function.blocks.len()).collect()
        };
        for index in block_indices {
            let block = &function.blocks[index];
            let mut block_depth = tracker.current_depth();
            if let Some(ref label) = block.label {
                tracker.enter_block(label.result_id);
                block_depth = tracker.current_depth();
                records.push(InstructionRecord::new(
                    label,
                    block_depth,
                    BlockPosition::Label,
                    ModuleSection::Functions,
                ));
            }
            for instruction in &block.instructions {
                records.push(InstructionRecord::new(
                    instruction,
                    block_depth,
                    BlockPosition::Body,
                    ModuleSection::Functions,
                ));
                tracker.observe(instruction);
            }
        }

        if let Some(ref end) = function.end {
            records.push(InstructionRecord::new(
                end,
                0,
                BlockPosition::Global,
                ModuleSection::Functions,
            ));
        }
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

struct MergeTracker {
    stack: Vec<u32>,
}

impl MergeTracker {
    fn new() -> Self {
        Self { stack: Vec::new() }
    }

    fn enter_block(&mut self, label_id: Option<u32>) {
        if let Some(id) = label_id {
            while let Some(&merge_id) = self.stack.last() {
                if merge_id == id {
                    self.stack.pop();
                } else {
                    break;
                }
            }
        }
    }

    fn current_depth(&self) -> u32 {
        self.stack.len() as u32
    }

    fn observe(&mut self, instruction: &Instruction) {
        match instruction.class.opcode {
            spirv::Op::SelectionMerge | spirv::Op::LoopMerge => {
                if let Some(target) = instruction.operands.first().and_then(extract_id_ref) {
                    self.stack.push(target);
                }
            }
            _ => {}
        }
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

fn reorder_function_blocks(function: &dr::Function) -> Vec<usize> {
    if function.blocks.is_empty() {
        return Vec::new();
    }

    let mut id_to_index = HashMap::new();
    for (index, block) in function.blocks.iter().enumerate() {
        if let Some(id) = block.label.as_ref().and_then(|inst| inst.result_id) {
            id_to_index.insert(id, index);
        }
    }

    let successors: Vec<Vec<usize>> = function
        .blocks
        .iter()
        .map(|block| {
            block_successor_ids(block)
                .into_iter()
                .filter_map(|id| id_to_index.get(&id).copied())
                .collect()
        })
        .collect();

    let mut order = Vec::with_capacity(function.blocks.len());
    let mut visited = vec![false; function.blocks.len()];
    fn dfs(
        index: usize,
        successors: &Vec<Vec<usize>>,
        visited: &mut [bool],
        order: &mut Vec<usize>,
    ) {
        if visited[index] {
            return;
        }
        visited[index] = true;
        order.push(index);
        for &succ in &successors[index] {
            dfs(succ, successors, visited, order);
        }
    }

    dfs(0, &successors, &mut visited, &mut order);
    for (index, flag) in visited.iter().enumerate() {
        if !*flag {
            order.push(index);
        }
    }
    order
}

fn block_successor_ids(block: &dr::Block) -> Vec<u32> {
    let mut successors = Vec::new();
    let Some(terminator) = block.instructions.last() else {
        return successors;
    };
    match terminator.class.opcode {
        spirv::Op::Branch => {
            if let Some(id) = terminator.operands.last().and_then(extract_id_ref) {
                successors.push(id);
            }
        }
        spirv::Op::BranchConditional => {
            for operand in terminator.operands.iter().skip(1).take(2) {
                if let Some(id) = extract_id_ref(operand) {
                    successors.push(id);
                }
            }
        }
        spirv::Op::Switch => {
            for operand in terminator.operands.iter().skip(1) {
                if let Some(id) = extract_id_ref(operand) {
                    successors.push(id);
                }
            }
        }
        _ => {}
    }
    successors
}

#[cfg(test)]
mod tests {
    use super::{disassemble_binary, FRIENDLY_NAME_SAMPLE_BINARY};
    use crate::assembly::{assemble_text, BinaryToTextOptions};
    use rspirv::binary::Assemble;
    use rspirv::dr::{self, Builder};
    use rspirv::spirv::{
        AccessQualifier, AddressingModel, BuiltIn, Capability, Decoration, ExecutionModel,
        FunctionControl, MemoryModel, SelectionControl, StorageClass,
    };

    #[test]
    fn disassembles_simple_module() {
        let text = "\
OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n\
%main = OpFunction %void None %void_fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd";
        let binary = assemble_text(text).expect("assemble text");
        let options = BinaryToTextOptions::NO_HEADER
            | BinaryToTextOptions::COMMENT
            | BinaryToTextOptions::INDENT;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.trim_start().starts_with("OpCapability Shader"));
        assert!(disassembled.contains("OpFunctionEnd"));
    }

    #[test]
    fn disassembly_respects_no_header_option() {
        let text = "\
OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%void = OpTypeVoid";
        let binary = assemble_text(text).expect("assemble text");
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
        let text = "\
OpCapability Shader\n\
OpMemoryModel Logical Simple\n\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n";
        let binary = assemble_text(text).expect("assemble text");
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::SHOW_BYTE_OFFSET;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        let expected = "OpCapability Shader                                 ; 0x00000014\n\
OpMemoryModel Logical Simple                        ; 0x0000001c\n\
%1 = OpTypeVoid                                     ; 0x00000028\n\
%2 = OpTypeFunction %1                              ; 0x00000030\n";
        assert_eq!(disassembled, expected);
    }

    #[test]
    fn disassembly_applies_indent_formatting() {
        let text = "\
OpCapability Shader\n\
OpMemoryModel Logical Simple\n\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n\
%main = OpFunction %void None %void_fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd";
        let binary = assemble_text(text).expect("assemble text");
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::INDENT;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");

        let indent = " ".repeat(super::STANDARD_INDENT_COLUMN);
        let pad = |name: &str| -> String {
            let head_chars = name.chars().count();
            let id_len = head_chars.saturating_sub(1);
            let spaces = super::STANDARD_INDENT_COLUMN.saturating_sub(4 + id_len);
            format!("{}{} = ", " ".repeat(spaces), name)
        };

        let expected = format!(
            "{indent}OpCapability Shader\n\
{indent}OpMemoryModel Logical Simple\n\
{id1}OpTypeVoid\n\
{id2}OpTypeFunction %1\n\
{id3}OpFunction %1 None %2\n\
{id4}OpLabel\n\
{indent}OpReturn\n\
{indent}OpFunctionEnd\n",
            indent = indent,
            id1 = pad("%1"),
            id2 = pad("%2"),
            id3 = pad("%3"),
            id4 = pad("%4"),
        );
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
            .position(|line| line.contains("%entry = OpLabel"))
            .expect("entry label");
        let then_idx = lines
            .iter()
            .position(|line| line.contains("%then = OpLabel"))
            .expect("then label");
        assert!(leading_spaces(lines[then_idx]) > leading_spaces(lines[entry_idx]));
        let then_branch_idx = lines
            .iter()
            .enumerate()
            .skip(then_idx)
            .find(|(_, line)| line.contains("OpBranch"))
            .map(|(idx, _)| idx)
            .expect("then branch");
        assert!(leading_spaces(lines[then_branch_idx]) > leading_spaces(lines[entry_idx]));
    }

    #[test]
    fn disassembly_emits_decoration_comments() {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let float = builder.type_float(32);
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
        let binary = build_selection_module(true);
        let options = BinaryToTextOptions::NO_HEADER
            | BinaryToTextOptions::REORDER_BLOCKS
            | BinaryToTextOptions::FRIENDLY_NAMES;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        let lines: Vec<&str> = disassembled.lines().collect();
        let entry_idx = lines
            .iter()
            .position(|line| line.contains("%entry = OpLabel"))
            .expect("entry");
        let merge_idx = lines
            .iter()
            .position(|line| line.contains("%merge = OpLabel"))
            .expect("merge");
        assert!(merge_idx > entry_idx);
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
        let text = "\
OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%uint = OpTypeInt 32 0\n\
%val = OpConstant %uint 42\n";
        let binary = assemble_text(text).expect("assemble");
        let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::HEX;
        let output = disassemble_binary(&binary, options).expect("disassemble");
        assert!(output.contains("0x0000002a"), "{output}");
        assert!(!output.contains(" 42"));
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
        let float = builder.type_float(32);
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
        let f16 = builder.type_float(16);
        let f32 = builder.type_float(32);
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
        let float = builder.type_float(32);
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
        let text = "\
OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%int = OpTypeInt 32 1\n\
%val = OpConstant %int -42\n";
        let binary = assemble_text(text).expect("assemble text");
        let options = BinaryToTextOptions::HEX | BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.contains("0xffffffd6"), "{disassembled}");
    }

    #[test]
    fn nested_indent_inserts_blank_line_before_labels() {
        let text = "\
OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%void = OpTypeVoid\n\
%fn = OpTypeFunction %void\n\
%main = OpFunction %void None %fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd\n";
        let binary = assemble_text(text).expect("assemble text");
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
