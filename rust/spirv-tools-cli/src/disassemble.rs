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
#[derive(Clone, Debug, Default)]
pub struct DisassembleConfig {
    /// Where to read the binary words from.
    pub input: InputSource,
    /// When set, the textual header is omitted from the output.
    pub suppress_header: bool,
    /// Whether to include byte offsets in the resulting text.
    pub show_byte_offsets: bool,
}

/// Runs the disassembler against the requested stream, returning the textual form.
pub fn run_disassemble(config: &DisassembleConfig) -> Result<String, DisassembleCliError> {
    let bytes = match &config.input {
        InputSource::Stdin => read_stdin()?,
        InputSource::Path(path) => fs::read(path)?,
    };
    let words = bytes_to_words(&bytes)?;

    let mut options = BinaryToTextOptions::NONE;
    if config.suppress_header {
        options |= BinaryToTextOptions::NO_HEADER;
    }
    if config.show_byte_offsets {
        options |= BinaryToTextOptions::SHOW_BYTE_OFFSET;
    }

    disassemble_binary(&words, options).map_err(Into::into)
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
    use super::{bytes_to_words, run_disassemble, DisassembleConfig, InputSource};
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
        };
        let output = run_disassemble(&config).expect("disassemble");
        assert!(output.starts_with("; SPIR-V"));
    }

    fn write_words(file: &mut NamedTempFile, words: &[u32]) {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        file.write_all(&bytes).expect("write words");
    }
}
