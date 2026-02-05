use rspirv::dr::{self, ModuleHeader};
use rspirv::spirv;

use super::{
    COLOR_BLUE, COLOR_GREY, COLOR_RESET,
    STANDARD_INDENT_COLUMN, BLOCK_NEST_INDENT, BLOCK_BODY_INDENT_OFFSET,
};
use super::types::*;
use super::formatting::disassemble_with_format;
use super::names::{
    FriendlyNameTable, section_heading, append_section_heading,
    append_function_heading,
};
use super::block_analysis::*;

pub(super) fn emit_disassembly_text(text: &str) {
    #[cfg(test)]
    {
        super::PRINT_LOG.lock().unwrap().push(text.to_string());
    }
    #[cfg(not(test))]
    {
        use std::io::{self, Write};
        let mut stdout = io::stdout();
        let _ = stdout.write_all(text.as_bytes());
        let _ = stdout.flush();
    }
}

pub(super) fn render_module_text(
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

pub(super) fn render_header(module: &dr::Module) -> String {
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

pub(super) fn vendor_prefix(name: &str) -> &str {
    match name {
        "SPIR-V Tools Assembler" => "Khronos ",
        "SPIR-V Tools Linker" => "Khronos ",
        _ => "",
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_instructions(
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

pub(super) fn sanitize_line(mut line: String) -> String {
    if line.ends_with('\n') {
        line.pop();
    }
    line
}

pub(super) fn apply_friendly_names(line: &mut String, names: Option<&FriendlyNameTable>) {
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

pub(super) fn apply_indent(line: &mut String, base_indent: usize) {
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

pub(super) fn insert_block_indent(line: &mut String, block_indent: usize) {
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

pub(super) fn apply_color_formatting(line: &mut String, colorize: bool) {
    if !colorize || line.is_empty() {
        return;
    }

    color_result_identifier(line);
    color_comment_section(line);
}

pub(super) fn color_result_identifier(line: &mut String) {
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

pub(super) fn color_comment_section(line: &mut String) {
    if let Some(comment_index) = line.find("; ") {
        let mut colored = String::with_capacity(line.len() + COLOR_GREY.len() + COLOR_RESET.len());
        colored.push_str(&line[..comment_index]);
        colored.push_str(COLOR_GREY);
        colored.push_str(&line[comment_index..]);
        colored.push_str(COLOR_RESET);
        *line = colored;
    }
}
