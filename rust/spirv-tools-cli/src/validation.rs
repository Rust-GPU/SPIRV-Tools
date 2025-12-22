use std::fs;
use std::io::{self, Read};
use std::sync::{Mutex, OnceLock};

use crate::assembly::parse_target_env;
use crate::disassemble::InputSource;
use spirv_tools_core::validation::{
    format_validation_error_from_words, ModuleWords, ValidModuleCache, ValidationOptions,
};
use spirv_tools_core::TargetEnv;
use thiserror::Error;

/// Errors that can occur while validating SPIR-V modules.
#[derive(Debug, Error)]
pub enum ValidateCliError {
    /// The binary stream could not be read.
    #[error("failed to read SPIR-V binary: {0}")]
    Input(#[from] io::Error),
    /// The binary length is not a multiple of 4 bytes.
    #[error("input size must be a multiple of 4 bytes")]
    MisalignedInput,
    /// The requested target environment is unknown.
    #[error("unknown target environment '{0}'")]
    UnknownTargetEnv(String),
    /// The validator reported diagnostics.
    #[error("validation failed:\n{0}")]
    Failed(String),
}

/// Validator configuration shared with the CLI binary.
#[derive(Clone, Debug, Default)]
pub struct ValidateConfig {
    /// Binary input source.
    pub input: InputSource,
    /// Optional target environment string.
    pub target_env: Option<String>,
}

static VALIDATION_CACHE: OnceLock<Mutex<ValidModuleCache>> = OnceLock::new();

fn validation_cache() -> &'static Mutex<ValidModuleCache> {
    VALIDATION_CACHE.get_or_init(Default::default)
}

/// Runs validation against the provided configuration.
pub fn run_validate(config: &ValidateConfig) -> Result<(), ValidateCliError> {
    let bytes = match &config.input {
        InputSource::Stdin => read_stdin_bytes()?,
        InputSource::Path(path) => fs::read(path)?,
    };
    let words: ModuleWords = bytes_to_words(&bytes)?.into_boxed_slice().into();
    let env = parse_env(config.target_env.as_deref())?;
    let options = ValidationOptions::default();
    let mut cache = validation_cache()
        .lock()
        .expect("validation cache mutex should not be poisoned");
    cache
        .validate_words_with_options(words.as_slice(), env, options.clone())
        .map(|_| ())
        .map_err(|err| {
            ValidateCliError::Failed(format_validation_error_from_words(
                words.as_slice(),
                &options,
                &err,
            ))
        })
}

fn read_stdin_bytes() -> io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, ValidateCliError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ValidateCliError::MisalignedInput);
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(words)
}

fn parse_env(spec: Option<&str>) -> Result<TargetEnv, ValidateCliError> {
    match spec {
        Some(name) => parse_target_env(Some(name))
            .map_err(|_| ValidateCliError::UnknownTargetEnv(name.to_string())),
        None => Ok(TargetEnv::Universal1_6),
    }
}

#[cfg(test)]
mod tests {
    use super::{bytes_to_words, run_validate, InputSource, ValidateCliError, ValidateConfig};
    use crate::assembly::words_to_bytes;
    use spirv_tools_core::assembly::assemble_text;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn bytes_convert_to_words() {
        let bytes = [0x01, 0x02, 0x03, 0x04];
        let words = bytes_to_words(&bytes).expect("convert words");
        assert_eq!(words, vec![0x0403_0201]);
    }

    #[test]
    fn validation_success() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical Simple",
            "OpEntryPoint Vertex %main \"main\"",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(text.as_str()).expect("assemble text");
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&words_to_bytes(&binary)).expect("write");
        let config = ValidateConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            target_env: None,
        };
        run_validate(&config).expect("validate");
    }

    #[test]
    fn validation_errors_include_friendly_names() {
        // ExecutionMode without an entry point should surface the function's friendly name.
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpName %main \"friendly_main\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "OpExecutionMode %main LocalSize 1 1 1",
        ]
        .join("\n");
        let binary = assemble_text(text.as_str()).expect("assemble text");
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&words_to_bytes(&binary)).expect("write");
        let config = ValidateConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            target_env: None,
        };

        let error = run_validate(&config).expect_err("expected validation failure");
        let ValidateCliError::Failed(message) = error else {
            panic!("unexpected error: {error:?}");
        };
        assert!(
            message.contains("friendly_main"),
            "expected friendly name in error message: {message}"
        );
    }

    #[test]
    fn validation_reports_failure() {
        // Invalid module: missing OpMemoryModel.
        let text = "%void = OpTypeVoid";
        let binary = assemble_text(text).expect("assemble text");
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&words_to_bytes(&binary)).expect("write");
        let config = ValidateConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            target_env: Some("spv1.6".to_string()),
        };
        let error = run_validate(&config).expect_err("expected failure");
        let ValidateCliError::Failed(message) = error else {
            panic!("unexpected error: {error:?}");
        };
        assert!(!message.is_empty());
    }

    #[test]
    fn validation_reuses_cache_for_repeated_inputs() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical Simple",
            "OpEntryPoint Vertex %main \"main\"",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text(text.as_str()).expect("assemble text");
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&words_to_bytes(&binary)).expect("write");
        let config = ValidateConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            target_env: None,
        };
        run_validate(&config).expect("first validation");
        // Second call should hit the cache and still succeed.
        run_validate(&config).expect("second validation");
    }
}
