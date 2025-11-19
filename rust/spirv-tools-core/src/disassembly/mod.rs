use rspirv::binary::{Disassemble, Parser};
use rspirv::dr::{self, Instruction, Loader, ModuleHeader, Operand};
use rspirv::spirv;
use std::collections::HashMap;
use std::mem::size_of_val;
use std::slice;
use thiserror::Error;

use crate::assembly::BinaryToTextOptions;

const SUPPORTED_OPTION_BITS: u32 = BinaryToTextOptions::NO_HEADER.bits()
    | BinaryToTextOptions::PRINT.bits()
    | BinaryToTextOptions::SHOW_BYTE_OFFSET.bits()
    | BinaryToTextOptions::INDENT.bits()
    | BinaryToTextOptions::FRIENDLY_NAMES.bits();
const HEADER_WORD_COUNT: usize = 5;
const INDEX_MAGIC_NUMBER: usize = 0;
const INDEX_VERSION: usize = 1;
const INDEX_GENERATOR: usize = 2;
const INDEX_BOUND: usize = 3;
const INDEX_SCHEMA: usize = 4;
const COMMENT_COLUMN: usize = 50;
const MAX_COMMENT_ALIGN: usize = 256;
const STANDARD_INDENT_COLUMN: usize = 15;

/// Errors that can be produced while disassembling SPIR-V binaries.
#[derive(Debug, Error)]
pub enum DisassemblyError {
    /// The binary failed to parse into a valid module.
    #[error("failed to parse SPIR-V binary: {0}")]
    Parse(String),
    /// The requested disassembly options are not supported by the Rust backend yet.
    #[error("unsupported binary-to-text options: {0:?}")]
    Unsupported(BinaryToTextOptions),
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
        .map_err(|error| DisassemblyError::Parse(error.to_string()))?;
    let mut module = loader.module();
    update_module_header(&mut module, words);
    let offsets = collect_instruction_offsets(words);
    let mut text = render_module_text(&module, &offsets, &formatting);
    if !text.ends_with('\n') {
        text.push('\n');
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

#[derive(Clone, Copy, Debug)]
struct FormattingOptions {
    suppress_header: bool,
    show_byte_offsets: bool,
    indent: bool,
    friendly_names: bool,
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
        })
    }
}

fn render_module_text(module: &dr::Module, offsets: &[u32], options: &FormattingOptions) -> String {
    let instructions = collect_module_instructions(module);
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
    instructions: &[&Instruction],
    offsets: &[u32],
    options: &FormattingOptions,
    friendly_names: Option<&FriendlyNameTable>,
) -> String {
    let mut aligner = CommentAligner::new();
    let mut text = String::new();
    for (index, (instruction, &offset)) in instructions.iter().zip(offsets).enumerate() {
        let mut line = sanitize_line(instruction.disassemble());
        apply_friendly_names(&mut line, friendly_names);
        if options.indent {
            apply_indent(&mut line);
        }
        if options.show_byte_offsets {
            let comment = format!("0x{offset:08x}");
            aligner.append_comment(&mut line, &comment);
        } else {
            aligner.reset();
        }
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

fn apply_indent(line: &mut String) {
    if line.is_empty() {
        return;
    }

    if let Some(eq_index) = line.find(" = ") {
        if line.starts_with('%') {
            let (head, tail) = line.split_at(eq_index);
            let trimmed_head = head.trim();
            let head_width = trimmed_head.chars().count();
            if head_width < STANDARD_INDENT_COLUMN {
                let padding = STANDARD_INDENT_COLUMN - head_width;
                let mut indented = String::with_capacity(line.len() + padding);
                indented.push_str(&" ".repeat(padding));
                indented.push_str(trimmed_head);
                indented.push_str(tail);
                *line = indented;
            } else if head != trimmed_head {
                let mut normalized = String::with_capacity(line.len());
                normalized.push_str(trimmed_head);
                normalized.push_str(tail);
                *line = normalized;
            }
            return;
        }
    }

    let mut indented = String::with_capacity(STANDARD_INDENT_COLUMN + line.len());
    indented.push_str(&" ".repeat(STANDARD_INDENT_COLUMN));
    indented.push_str(line.trim_start());
    *line = indented;
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

fn collect_module_instructions(module: &dr::Module) -> Vec<&Instruction> {
    let mut instructions = Vec::new();
    instructions.extend(module.capabilities.iter());
    instructions.extend(module.extensions.iter());
    instructions.extend(module.ext_inst_imports.iter());
    if let Some(ref inst) = module.memory_model {
        instructions.push(inst);
    }
    instructions.extend(module.entry_points.iter());
    instructions.extend(module.execution_modes.iter());
    instructions.extend(module.debug_string_source.iter());
    instructions.extend(module.debug_names.iter());
    instructions.extend(module.debug_module_processed.iter());
    instructions.extend(module.annotations.iter());
    instructions.extend(module.types_global_values.iter());
    for function in &module.functions {
        if let Some(ref def) = function.def {
            instructions.push(def);
        }
        instructions.extend(function.parameters.iter());
        for block in &function.blocks {
            if let Some(ref label) = block.label {
                instructions.push(label);
            }
            instructions.extend(block.instructions.iter());
        }
        if let Some(ref end) = function.end {
            instructions.push(end);
        }
    }
    instructions
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

#[cfg(test)]
mod tests {
    use super::disassemble_binary;
    use crate::assembly::{assemble_text, BinaryToTextOptions};
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{
        AddressingModel, Capability, ExecutionModel, FunctionControl, MemoryModel,
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
    fn unsupported_options_request_fallback() {
        let text = "OpCapability Shader";
        let binary = assemble_text(text).expect("assemble text");
        let options = BinaryToTextOptions::COLOR;
        let error = disassemble_binary(&binary, options).expect_err("expected error");
        match error {
            super::DisassemblyError::Unsupported(bits) => {
                assert!(bits.contains(BinaryToTextOptions::COLOR));
            }
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
        assert!(!super::supports_options(BinaryToTextOptions::COLOR));
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
