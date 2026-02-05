mod names;
mod types;
mod formatting;
mod block_analysis;
mod text_rendering;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
use types::*;
#[allow(unused_imports)]
use formatting::*;
#[allow(unused_imports)]
use block_analysis::*;
#[allow(unused_imports)]
use text_rendering::*;
#[allow(unused_imports)]
use names::*;

use rspirv::binary::Parser;
use rspirv::dr::ModuleHeader;
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
