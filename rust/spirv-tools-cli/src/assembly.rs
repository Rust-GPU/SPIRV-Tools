use std::fs;
use std::io::{self, Read};

use crate::disassemble::InputSource;
use spirv_tools_core::assembly::{assemble_text_with_env, AssemblyError, TextToBinaryOptions};
use spirv_tools_core::target_env::TargetEnv;
use thiserror::Error;

/// Errors that can occur while assembling SPIR-V text.
#[derive(Debug, Error)]
pub enum AssembleCliError {
    /// Input stream failed.
    #[error("failed to read SPIR-V text: {0}")]
    Input(#[from] io::Error),
    /// Input was not valid UTF-8.
    #[error("input must be valid UTF-8")]
    Utf8,
    /// User provided an unknown target environment name.
    #[error("unknown target environment '{0}'")]
    UnknownTargetEnv(String),
    /// The assembler reported diagnostics.
    #[error("failed to assemble module: {0:?}")]
    Assembly(AssemblyError),
}

/// Configuration for the assembler CLI.
#[derive(Clone, Debug, Default)]
pub struct AssembleConfig {
    /// Where to read textual SPIR-V from.
    pub input: InputSource,
    /// Optional target environment name.
    pub target_env: Option<String>,
    /// Whether numeric IDs in the source should be preserved.
    pub preserve_numeric_ids: bool,
}

/// Runs the assembler for the provided configuration, returning SPIR-V words.
pub fn run_assemble(config: &AssembleConfig) -> Result<Vec<u32>, AssembleCliError> {
    let source = match &config.input {
        InputSource::Stdin => read_stdin_string()?,
        InputSource::Path(path) => fs::read_to_string(path).map_err(|err| match err.kind() {
            io::ErrorKind::InvalidData => AssembleCliError::Utf8,
            _ => AssembleCliError::Input(err),
        })?,
    };

    let env = parse_target_env(config.target_env.as_deref())?;
    let mut options = TextToBinaryOptions::NONE;
    if config.preserve_numeric_ids {
        options |= TextToBinaryOptions::PRESERVE_NUMERIC_IDS;
    }

    assemble_text_with_env(&source, env).map_err(AssembleCliError::Assembly)
}

fn read_stdin_string() -> Result<String, AssembleCliError> {
    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .map_err(|err| match err.kind() {
            io::ErrorKind::InvalidData => AssembleCliError::Utf8,
            _ => AssembleCliError::Input(err),
        })?;
    Ok(buffer)
}

/// Parses an optional target environment string into a `TargetEnv` value.
pub fn parse_target_env(spec: Option<&str>) -> Result<TargetEnv, AssembleCliError> {
    match spec {
        Some(name) => TargetEnv::parse_name(name)
            .ok_or_else(|| AssembleCliError::UnknownTargetEnv(name.to_string())),
        None => Ok(TargetEnv::Universal1_6),
    }
}

/// Converts SPIR-V words into their little-endian byte representation.
pub fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::{run_assemble, words_to_bytes, AssembleCliError, AssembleConfig, InputSource};
    use spirv_tools_core::assembly::assemble_text;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn assembles_simple_module_from_stdin() {
        let text = "%void = OpTypeVoid";
        let binary = assemble_text(text).expect("assemble text");
        let mut temp = NamedTempFile::new().expect("temp file");
        temp.write_all(text.as_bytes()).expect("write text");

        let config = AssembleConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            target_env: None,
            preserve_numeric_ids: false,
        };
        let output = run_assemble(&config).expect("assemble");
        assert_eq!(output, binary);
    }

    #[test]
    fn rejects_unknown_target_env() {
        let config = AssembleConfig {
            input: InputSource::Stdin,
            target_env: Some("unknown+env".to_string()),
            preserve_numeric_ids: false,
        };
        match run_assemble(&config) {
            Err(AssembleCliError::UnknownTargetEnv(env)) => assert_eq!(env, "unknown+env"),
            other => panic!("expected unknown env error, got {other:?}"),
        }
    }

    #[test]
    fn words_are_serialized_as_little_endian_bytes() {
        let words = vec![0x0102_0304, 0x0506_0708];
        let bytes = words_to_bytes(&words);
        assert_eq!(bytes, vec![0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05]);
    }
}
