use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use spirv_tools_core::assembly::BinaryToTextOptions;
use spirv_tools_core::disassembly::{disassemble_binary, DisassemblyError};
use thiserror::Error;

/// Error returned while running the disassembler CLI.
#[derive(Debug, Error)]
pub enum DisassembleCliError {
    /// Failed to read the input stream.
    #[error("failed to read SPIR-V module: {0}")]
    Input(#[from] io::Error),
    /// Input length is not a multiple of four bytes.
    #[error("input size must be a multiple of 4 bytes")]
    MisalignedInput,
    /// The underlying disassembler emitted an error.
    #[error("{0}")]
    Disassembly(#[from] DisassemblyError),
}

/// Describes which stream should be disassembled.
#[derive(Clone, Debug, Default)]
pub enum InputSource {
    /// Read bytes from standard input.
    #[default]
    Stdin,
    /// Read bytes from the given file path.
    Path(PathBuf),
}

/// Configuration for the disassembler.
#[derive(Clone, Debug)]
pub struct DisassembleConfig {
    /// Where to read the binary words from.
    pub input: InputSource,
    /// When set, the textual header is omitted from the output.
    pub suppress_header: bool,
    /// Whether to include byte offsets in the resulting text.
    pub show_byte_offsets: bool,
    /// Whether to align operands using indentation.
    pub indent: bool,
    /// Use friendly names when available instead of raw numeric ids.
    pub friendly_names: bool,
    /// Emits nested indentation markers for structured control flow.
    pub nested_indent: bool,
    /// Reorders blocks to match structured control flow.
    pub reorder_blocks: bool,
    /// Includes decoration comments alongside instructions.
    pub comments: bool,
    /// Enables ANSI color escapes in the output.
    pub colorize: bool,
    /// Emits literal numbers in hexadecimal form.
    pub hex_literals: bool,
}

impl Default for DisassembleConfig {
    fn default() -> Self {
        Self {
            input: InputSource::default(),
            suppress_header: false,
            show_byte_offsets: false,
            indent: true,
            friendly_names: true,
            nested_indent: false,
            reorder_blocks: false,
            comments: false,
            colorize: false,
            hex_literals: false,
        }
    }
}

/// Runs the disassembler against the requested stream, returning the textual form.
pub fn run_disassemble(config: &DisassembleConfig) -> Result<String, DisassembleCliError> {
    let bytes = match &config.input {
        InputSource::Stdin => read_stdin()?,
        InputSource::Path(path) => fs::read(path)?,
    };
    let words = bytes_to_words(&bytes)?;
    disassemble_words(&words, config)
}

fn disassemble_words(
    words: &[u32],
    config: &DisassembleConfig,
) -> Result<String, DisassembleCliError> {
    let mut options = BinaryToTextOptions::NONE;
    if config.suppress_header {
        options |= BinaryToTextOptions::NO_HEADER;
    }
    if config.show_byte_offsets {
        options |= BinaryToTextOptions::SHOW_BYTE_OFFSET;
    }
    if config.indent {
        options |= BinaryToTextOptions::INDENT;
    }
    if config.friendly_names {
        options |= BinaryToTextOptions::FRIENDLY_NAMES;
    }
    if config.nested_indent {
        options |= BinaryToTextOptions::NESTED_INDENT;
    }
    if config.reorder_blocks {
        options |= BinaryToTextOptions::REORDER_BLOCKS;
    }
    if config.comments {
        options |= BinaryToTextOptions::COMMENT;
    }
    if config.colorize {
        options |= BinaryToTextOptions::COLOR;
    }
    if config.hex_literals {
        options |= BinaryToTextOptions::HEX;
    }

    disassemble_binary(words, options).map_err(Into::into)
}

fn read_stdin() -> io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, DisassembleCliError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(DisassembleCliError::MisalignedInput);
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        words.push(value);
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::{
        bytes_to_words, disassemble_words, run_disassemble, DisassembleConfig, InputSource,
    };
    use spirv_tools_core::assembly::assemble_text;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn converts_bytes_into_words() {
        let input = [0x01u8, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD];
        let words = bytes_to_words(&input).expect("bytes to words");
        assert_eq!(words, vec![0x0403_0201, 0xDDCC_BBAA]);
    }

    #[test]
    fn disassembler_runs_from_file_source() {
        let text = "%void = OpTypeVoid";
        let binary = assemble_text(text).expect("assemble text");
        let mut file = NamedTempFile::new().expect("temp file");
        write_words(&mut file, &binary);

        let config = DisassembleConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            suppress_header: true,
            show_byte_offsets: false,
            ..DisassembleConfig::default()
        };

        let output = run_disassemble(&config).expect("disassemble");
        assert!(output.contains("OpTypeVoid"));
    }

    #[test]
    fn header_is_present_by_default() {
        let text = "%void = OpTypeVoid";
        let binary = assemble_text(text).expect("assemble text");
        let mut file = NamedTempFile::new().expect("temp file");
        write_words(&mut file, &binary);

        let config = DisassembleConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            suppress_header: false,
            show_byte_offsets: false,
            ..DisassembleConfig::default()
        };
        let output = run_disassemble(&config).expect("disassemble");
        assert!(output.starts_with("; SPIR-V"));
    }

    #[test]
    fn disassembler_honors_raw_id_flag() {
        let binary = build_named_function_binary();

        let mut config = DisassembleConfig {
            friendly_names: true,
            ..DisassembleConfig::default()
        };
        let friendly = disassemble_words(&binary, &config).expect("disassemble");
        assert!(friendly.contains("%main_fn"));

        config.friendly_names = false;
        let raw = disassemble_words(&binary, &config).expect("disassemble");
        assert!(!raw.contains("%main_fn"));
    }

    #[test]
    fn disassembler_emits_comments_when_requested() {
        let binary = build_decorated_function_binary();

        let config = DisassembleConfig {
            comments: true,
            ..DisassembleConfig::default()
        };
        let text = disassemble_words(&binary, &config).expect("disassemble");
        assert!(text.contains("DescriptorSet 0"));
    }

    #[test]
    fn disassembler_emits_color_codes() {
        let binary = build_named_function_binary();

        let config = DisassembleConfig {
            colorize: true,
            ..DisassembleConfig::default()
        };
        let text = disassemble_words(&binary, &config).expect("disassemble");
        assert!(text.contains("\x1b["));
    }

    #[test]
    fn disassembler_emits_hex_literals() {
        let text = "\
OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%uint = OpTypeInt 32 0\n\
%val = OpConstant %uint 255\n";
        let binary = assemble_text(text).expect("assemble text");
        let config = DisassembleConfig {
            hex_literals: true,
            suppress_header: true,
            ..DisassembleConfig::default()
        };
        let output = disassemble_words(&binary, &config).expect("disassemble");
        assert!(output.contains("0x000000ff"), "{output}");
    }

    fn write_words(file: &mut NamedTempFile, words: &[u32]) {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        file.write_all(&bytes).expect("write words");
    }

    fn build_named_function_binary() -> Vec<u32> {
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel};

        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void = builder.type_void();
        let void_fn = builder.type_function(void, vec![]);
        let function = builder
            .begin_function(void, None, FunctionControl::NONE, void_fn)
            .expect("function");
        builder.name(function, "main_fn");
        builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end");
        builder.module().assemble()
    }

    fn build_decorated_function_binary() -> Vec<u32> {
        use rspirv::binary::Assemble;
        use rspirv::dr::{Builder, Operand};
        use rspirv::spirv::{
            AddressingModel, Capability, Decoration, FunctionControl, MemoryModel,
        };

        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void = builder.type_void();
        let void_fn = builder.type_function(void, vec![]);
        let function = builder
            .begin_function(void, None, FunctionControl::NONE, void_fn)
            .expect("function");
        builder.begin_block(None).expect("block");
        builder.ret().expect("return");
        builder.end_function().expect("end");
        builder.decorate(
            function,
            Decoration::DescriptorSet,
            [Operand::LiteralBit32(0)],
        );
        builder.module().assemble()
    }
}
