use rspirv::binary::{Disassemble, Parser};
use rspirv::dr::{self, Instruction, Loader, ModuleHeader, Operand};
use rspirv::spirv;
use std::collections::HashMap;
use std::mem::size_of_val;
use std::slice;
use thiserror::Error;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

use crate::assembly::BinaryToTextOptions;
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
    if !text.ends_with('\n') {
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

struct InstructionRecord<'a> {
    instruction: &'a Instruction,
    depth: u32,
    position: BlockPosition,
}

impl<'a> InstructionRecord<'a> {
    fn new(instruction: &'a Instruction, depth: u32, position: BlockPosition) -> Self {
        Self {
            instruction,
            depth,
            position,
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
    let friendly_names = if options.friendly_names {
        let table = FriendlyNameTable::from_module(module);
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
    friendly_names: Option<&FriendlyNameTable>,
    comment_collector: &mut CommentCollector,
) -> String {
    let mut aligner = CommentAligner::new();
    let mut text = String::new();
    for (index, (record, &offset)) in instructions.iter().zip(offsets).enumerate() {
        let instruction = record.instruction();
        if options.comments {
            comment_collector.observe(instruction);
        }
        let mut line = sanitize_line(disassemble_with_format(instruction, options.literal_format));
        apply_friendly_names(&mut line, friendly_names);

        let mut block_indent = 0usize;
        if options.nested_indent {
            block_indent += (record.depth as usize) * BLOCK_NEST_INDENT;
            if record.is_block_body() {
                block_indent += BLOCK_BODY_INDENT_OFFSET;
            }
        }

        if options.indent {
            apply_indent(&mut line, block_indent);
        } else if block_indent > 0 {
            prepend_spaces(&mut line, block_indent);
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

fn disassemble_with_format(instruction: &Instruction, literal_format: LiteralFormat) -> String {
    let space = if instruction.operands.is_empty() {
        ""
    } else {
        " "
    };
    let operands = disassemble_operands(&instruction.operands, literal_format);
    format!(
        "{rid}Op{opcode}{rtype}{space}{operands}",
        rid = instruction
            .result_id
            .map_or(String::new(), |w| format!("%{} = ", w)),
        opcode = instruction.class.opname,
        rtype = instruction
            .result_type
            .map_or(String::new(), |w| format!("  %{}{}", w, space)),
        space = space,
        operands = operands,
    )
}

fn disassemble_operands(operands: &[Operand], literal_format: LiteralFormat) -> String {
    operands
        .iter()
        .map(|operand| format_operand(operand, literal_format))
        .collect::<Vec<_>>()
        .join(" ")
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
        _ => operand.disassemble(),
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
            let head_width = trimmed_head.chars().count();
            let mut indented =
                String::with_capacity(base_indent + STANDARD_INDENT_COLUMN + line.len());
            if base_indent > 0 {
                indented.push_str(&" ".repeat(base_indent));
            }
            if head_width < STANDARD_INDENT_COLUMN {
                indented.push_str(&" ".repeat(STANDARD_INDENT_COLUMN - head_width));
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

fn prepend_spaces(line: &mut String, count: usize) {
    if count == 0 {
        return;
    }
    let mut prefixed = String::with_capacity(count + line.len());
    prefixed.push_str(&" ".repeat(count));
    prefixed.push_str(line);
    *line = prefixed;
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
        ));
    }
    for instruction in &module.extensions {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }
    for instruction in &module.ext_inst_imports {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }
    if let Some(ref inst) = module.memory_model {
        records.push(InstructionRecord::new(inst, 0, BlockPosition::Global));
    }
    for instruction in &module.entry_points {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }
    for instruction in &module.execution_modes {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }
    for instruction in &module.debug_string_source {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }
    for instruction in &module.debug_names {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }
    for instruction in &module.debug_module_processed {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }
    for instruction in &module.annotations {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }
    for instruction in &module.types_global_values {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
        ));
    }

    for function in &module.functions {
        if let Some(ref def) = function.def {
            records.push(InstructionRecord::new(def, 0, BlockPosition::Global));
        }
        for parameter in &function.parameters {
            records.push(InstructionRecord::new(parameter, 0, BlockPosition::Global));
        }

        let mut tracker = MergeTracker::new();
        let block_indices = if reorder_blocks {
            reorder_function_blocks(function)
        } else {
            (0..function.blocks.len()).collect()
        };
        for index in block_indices {
            let block = &function.blocks[index];
            if let Some(ref label) = block.label {
                tracker.enter_block(label.result_id);
                records.push(InstructionRecord::new(
                    label,
                    tracker.current_depth(),
                    BlockPosition::Label,
                ));
            }
            for instruction in &block.instructions {
                records.push(InstructionRecord::new(
                    instruction,
                    tracker.current_depth(),
                    BlockPosition::Body,
                ));
                tracker.observe(instruction);
            }
        }

        if let Some(ref end) = function.end {
            records.push(InstructionRecord::new(end, 0, BlockPosition::Global));
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
    use super::disassemble_binary;
    use crate::assembly::{assemble_text, BinaryToTextOptions};
    use rspirv::binary::Assemble;
    use rspirv::dr::{self, Builder};
    use rspirv::spirv::{
        AddressingModel, Capability, Decoration, ExecutionModel, FunctionControl, MemoryModel,
        SelectionControl, StorageClass,
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
        let options = BinaryToTextOptions::NO_HEADER;
        let disassembled = disassemble_binary(&binary, options).expect("disassemble");
        assert!(disassembled.starts_with("OpCapability Shader"));
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
            let spaces = super::STANDARD_INDENT_COLUMN.saturating_sub(name.chars().count());
            format!("{}{} = ", " ".repeat(spaces), name)
        };

        let expected = format!(
            "{indent}OpCapability Shader\n\
{indent}OpMemoryModel Logical Simple\n\
{id1}OpTypeVoid\n\
{id2}OpTypeFunction %1\n\
{id3}OpFunction  %1  None %2\n\
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
    fn from_module(module: &dr::Module) -> Self {
        let mut builder = FriendlyNameBuilder::new();
        for instruction in &module.debug_names {
            if instruction.class.opcode == spirv::Op::Name {
                if let Some(id) = instruction.operands.first().and_then(extract_id_ref) {
                    if let Some(name) = instruction
                        .operands
                        .get(1)
                        .and_then(|operand| extract_literal_string(operand))
                    {
                        builder.assign_name(id, name);
                    }
                }
            }
        }

        for instruction in module.all_inst_iter() {
            if let Some(id) = instruction.result_id {
                builder.ensure_name(id);
            }
        }

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

struct FriendlyNameBuilder {
    names: HashMap<u32, String>,
    used: HashMap<String, u32>,
    fallback: u32,
}

impl FriendlyNameBuilder {
    fn new() -> Self {
        Self {
            names: HashMap::new(),
            used: HashMap::new(),
            fallback: 1,
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
        let base = format!("_{}", self.fallback);
        self.fallback += 1;
        let name = self.unique_name(base);
        self.names.insert(id, name);
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
            format!("{}_{}", normalized, counter)
        };
        *counter += 1;
        result
    }

    fn finish(self) -> HashMap<u32, String> {
        self.names
    }
}

fn sanitize_identifier(raw: &str) -> String {
    let mut sanitized = String::new();
    let mut last_was_sep = false;
    for ch in raw.chars() {
        let mapped = if ch.is_ascii_alphanumeric() || ch == '_' {
            ch
        } else {
            '_'
        };
        if mapped == '_' {
            if last_was_sep {
                continue;
            }
            last_was_sep = true;
        } else {
            last_was_sep = false;
        }
        sanitized.push(mapped);
    }

    let mut trimmed = sanitized.trim_matches('_').to_string();
    if trimmed.is_empty() {
        trimmed.push('_');
    }

    if trimmed
        .chars()
        .next()
        .map(|ch| ch.is_ascii_digit())
        .unwrap_or(false)
    {
        trimmed.insert(0, '_');
    }

    trimmed
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
