use std::fmt;
use std::fs;
use std::io::{self, Read};

use crate::disassemble::InputSource;
use rspirv::binary::ParseState;
use rspirv::dr::{load_words, Module};
use thiserror::Error;

/// Input selection for the size tool.
#[derive(Clone, Debug, Default)]
pub struct SizeConfig {
    /// Where to read the module from.
    pub input: InputSource,
}

/// Summary statistics for a SPIR-V module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleSize {
    /// Number of 32-bit words in the binary module.
    pub words: usize,
    /// Number of bytes in the binary module.
    pub bytes: usize,
    /// Total instruction count (including global sections).
    pub instructions: usize,
    /// Number of functions defined in the module.
    pub functions: usize,
}

impl ModuleSize {
    fn from_module(module: &Module, word_count: usize) -> Self {
        let instructions = module.all_inst_iter().count();
        let functions = module.functions.len();
        Self {
            words: word_count,
            bytes: word_count * 4,
            instructions,
            functions,
        }
    }
}

impl fmt::Display for ModuleSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "words: {}", self.words)?;
        writeln!(f, "bytes: {}", self.bytes)?;
        writeln!(f, "instructions: {}", self.instructions)?;
        writeln!(f, "functions: {}", self.functions)
    }
}

/// Errors surfaced by the size CLI entry point.
#[derive(Debug, Error)]
pub enum SizeCliError {
    /// Failure while reading the input bytes.
    #[error("failed to read SPIR-V module: {0}")]
    Input(#[from] io::Error),
    /// The byte stream length was not a multiple of 4.
    #[error("input size must be a multiple of 4 bytes")]
    MisalignedInput,
    /// The module could not be decoded.
    #[error("failed to decode SPIR-V module: {0}")]
    Decode(#[from] ParseState),
}

/// Run the size tool against the requested input.
pub fn run_size(config: &SizeConfig) -> Result<ModuleSize, SizeCliError> {
    let bytes = match &config.input {
        InputSource::Stdin => read_stdin()?,
        InputSource::Path(path) => fs::read(path)?,
    };
    let words = bytes_to_words(&bytes)?;
    let module = load_words(&words)?;
    Ok(ModuleSize::from_module(&module, words.len()))
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, SizeCliError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(SizeCliError::MisalignedInput);
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(words)
}

fn read_stdin() -> io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::words_to_bytes;
    use spirv_tools_core::assembly::assemble_text;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn reports_basic_module_stats() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical Simple",
            "OpEntryPoint GLCompute %main \"main\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let words = assemble_text(&text).expect("assemble module");
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&words_to_bytes(&words))
            .expect("write module");

        let stats = run_size(&SizeConfig {
            input: InputSource::Path(file.path().to_path_buf()),
        })
        .expect("compute stats");

        assert_eq!(stats.words, words.len());
        assert_eq!(stats.bytes, words.len() * 4);
        assert_eq!(stats.functions, 1);
        assert!(
            stats.instructions >= 9,
            "expected at least 9 instructions, got {}",
            stats.instructions
        );
    }

    #[test]
    fn rejects_misaligned_input() {
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&[1u8, 2, 3]).expect("write misaligned");

        let error = run_size(&SizeConfig {
            input: InputSource::Path(file.path().to_path_buf()),
        })
        .expect_err("expected misalignment failure");
        assert!(matches!(error, SizeCliError::MisalignedInput));
    }
}
