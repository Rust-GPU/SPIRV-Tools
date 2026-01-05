use crate::assembly::words_to_bytes;
use crate::disassemble::InputSource;
use spirv_tools_ffi::optimize_basic_block;
use spirv_tools_ffi::OptimizeError as FfiOptimizeError;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use thiserror::Error;

/// Errors surfaced by the optimizer CLI.
#[derive(Debug, Error)]
pub enum OptimizeCliError {
    /// Failed to read the input stream.
    #[error("failed to read SPIR-V module: {0}")]
    Input(#[from] io::Error),
    /// Input length is not a multiple of four bytes.
    #[error("input size must be a multiple of 4 bytes")]
    MisalignedInput,
    /// The Rust optimizer reported an error.
    #[error("optimization failed: {0}")]
    Optimize(String),
    /// The optimizer could not parse the input module.
    #[error("failed to parse SPIR-V module: {0}")]
    Parse(String),
    /// The C++ spirv-opt fallback failed with status/stderr.
    #[error("cpp spirv-opt failed with status {status}: {stderr}")]
    CppFailure {
        /// Exit status returned by the C++ optimizer.
        status: std::process::ExitStatus,
        /// Stderr from the failing C++ optimizer process.
        stderr: String,
    },
}

/// Configuration for the optimizer CLI.
#[derive(Clone, Debug)]
pub struct OptimizeConfig {
    /// Where to read the binary words from.
    pub input: InputSource,
    /// Where to write the optimized module. If `None`, stdout is used.
    pub output: Option<PathBuf>,
    /// When true, uses the Rust arithmetic optimizer; otherwise passthrough.
    pub rust_arith_pass: bool,
    /// Optional path to C++ spirv-opt for fallback/benchmarking.
    pub cpp_opt_path: Option<std::ffi::OsString>,
    /// Force-enable the Rust optimizer regardless of env disable flags.
    pub force_rust_opt: bool,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            input: InputSource::default(),
            output: None,
            rust_arith_pass: true,
            cpp_opt_path: None,
            force_rust_opt: false,
        }
    }
}

/// Run the arithmetic optimizer on the chosen input.
pub fn run_optimize(config: &OptimizeConfig) -> Result<Vec<u32>, OptimizeCliError> {
    let bytes = match &config.input {
        InputSource::Stdin => read_stdin()?,
        InputSource::Path(path) => fs::read(path)?,
    };
    let words = bytes_to_words(&bytes)?;
    if config.rust_arith_pass {
        let _override_guard = config.force_rust_opt.then(OptimizerOverrideGuard::enable);
        let result = optimize_basic_block(&words);
        if result.success {
            return Ok(result.words);
        } else {
            return Err(match result.error {
                FfiOptimizeError::Parse => OptimizeCliError::Parse(result.message),
                FfiOptimizeError::Optimize => OptimizeCliError::Optimize(result.message),
                // Disabled is unexpected when rust_arith_pass is true, but fall back to passthrough.
                FfiOptimizeError::Disabled | FfiOptimizeError::None => {
                    OptimizeCliError::Optimize(result.message)
                }
                _ => OptimizeCliError::Optimize(result.message),
            });
        }
    }

    if let Some(path) = &config.cpp_opt_path {
        return run_cpp_opt(path, &words);
    }

    Ok(words)
}

/// Writes the optimized words to the configured sink.
pub fn write_output(words: &[u32], output: &Option<PathBuf>) -> Result<(), OptimizeCliError> {
    let bytes = words_to_bytes(words);
    match output {
        Some(path) => fs::write(path, bytes).map_err(OptimizeCliError::Input),
        None => {
            let mut stdout = io::stdout();
            stdout.write_all(&bytes)?;
            Ok(())
        }
    }
}

fn read_stdin() -> io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn run_cpp_opt(path: &std::ffi::OsStr, words: &[u32]) -> Result<Vec<u32>, OptimizeCliError> {
    use std::process::{Command, Stdio};

    let mut child = Command::new(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(OptimizeCliError::Input)?;

    {
        let stdin = child.stdin.as_mut().ok_or_else(|| {
            OptimizeCliError::Input(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "failed to open spirv-opt stdin",
            ))
        })?;
        stdin.write_all(&words_to_bytes(words))?;
    }

    let output = child.wait_with_output().map_err(OptimizeCliError::Input)?;
    if !output.status.success() {
        return Err(OptimizeCliError::CppFailure {
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    bytes_to_words(&output.stdout)
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, OptimizeCliError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(OptimizeCliError::MisalignedInput);
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        words.push(value);
    }
    Ok(words)
}

struct OptimizerOverrideGuard;

impl OptimizerOverrideGuard {
    fn enable() -> Self {
        spirv_tools_ffi::set_rust_optimizer_override(true);
        Self
    }
}

impl Drop for OptimizerOverrideGuard {
    fn drop(&mut self) {
        spirv_tools_ffi::clear_rust_optimizer_override();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::binary::{parse_words, Assemble};
    use rspirv::dr::Builder;
    use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
    use tempfile::NamedTempFile;
    #[test]
    fn folds_sub_self_via_cli() {
        let mut b = Builder::new();
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        let _ = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .expect("function");
        let _ = b.begin_block(None).expect("block");
        let c7 = b.constant_bit32(int, 7);
        let _sub = b.i_sub(int, None, c7, c7).expect("isub");
        b.ret().expect("ret");
        b.end_function().expect("end");
        let module = b.module().assemble();

        let mut temp = NamedTempFile::new().expect("temp file");
        temp.write_all(&words_to_bytes(&module))
            .expect("write words");
        let config = OptimizeConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            output: None,
            rust_arith_pass: true,
            cpp_opt_path: None,
            force_rust_opt: true,
        };
        let optimized = run_optimize(&config).expect("optimize");
        let mut loader = rspirv::dr::Loader::new();
        parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();
        // Verify that ISub was folded away - the optimizer should recognize x - x = 0
        for inst in module.all_inst_iter() {
            assert_ne!(inst.class.opcode, Op::ISub, "sub should be folded away");
        }
    }

    #[test]
    fn passthrough_when_rust_optimizer_disabled() {
        let mut b = Builder::new();
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        let _ = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .expect("function");
        let _ = b.begin_block(None).expect("block");
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let _ = b.i_add(int, None, c2, c3);
        b.ret().expect("ret");
        b.end_function().expect("end");
        let module = b.module().assemble();

        let mut temp = NamedTempFile::new().expect("temp file");
        temp.write_all(&words_to_bytes(&module))
            .expect("write words");
        let config = OptimizeConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            output: None,
            rust_arith_pass: false,
            cpp_opt_path: None,
            force_rust_opt: false,
        };
        let optimized = run_optimize(&config).expect("optimize passthrough");
        assert_eq!(optimized, module);
    }

    #[test]
    fn cpp_fallback_uses_cli_when_requested() {
        let mut b = Builder::new();
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        let _ = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .expect("function");
        let _ = b.begin_block(None).expect("block");
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let add = b.i_add(int, None, c2, c3).expect("add");
        b.ret().expect("ret");
        b.end_function().expect("end");
        let module = b.module().assemble();

        let mut temp = NamedTempFile::new().expect("temp file");
        temp.write_all(&words_to_bytes(&module))
            .expect("write words");

        let Some(cpp_path) = std::env::var_os("CARGO_BIN_EXE_spirv-opt") else {
            // Binary not built in this test configuration; skip.
            return;
        };
        let config = OptimizeConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            output: None,
            rust_arith_pass: false,
            cpp_opt_path: Some(cpp_path),
            force_rust_opt: false,
        };
        let optimized = run_optimize(&config).expect("optimize via cpp fallback");
        let mut loader = rspirv::dr::Loader::new();
        parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut found_const_five = false;
        for inst in module.all_inst_iter() {
            match inst.class.opcode {
                Op::IAdd => panic!("addition should be folded by CLI fallback"),
                Op::Constant => {
                    if inst.result_id == Some(add)
                        && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5u32)]
                    {
                        found_const_five = true;
                    }
                }
                _ => {}
            }
        }
        assert!(found_const_five, "cpp fallback should fold to const 5");
    }

    #[test]
    fn rejects_misaligned_input() {
        let temp = NamedTempFile::new().expect("temp file");
        std::fs::write(temp.path(), [1u8, 2, 3]).expect("write bytes");
        let config = OptimizeConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            output: None,
            rust_arith_pass: true,
            cpp_opt_path: None,
            force_rust_opt: false,
        };
        match run_optimize(&config) {
            Err(OptimizeCliError::MisalignedInput) => {}
            other => panic!("expected misaligned input error, got {other:?}"),
        }
    }

    #[test]
    fn reports_parse_error() {
        let temp = NamedTempFile::new().expect("temp file");
        // Four bytes is a single word but not a valid SPIR-V header.
        std::fs::write(temp.path(), [0u8, 0, 0, 0]).expect("write bytes");
        let config = OptimizeConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            output: None,
            rust_arith_pass: true,
            cpp_opt_path: None,
            force_rust_opt: false,
        };
        match run_optimize(&config) {
            Err(OptimizeCliError::Parse(_)) => {}
            other => panic!("expected parse error, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn cpp_fallback_reports_failure() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        // Build a trivial module.
        let mut b = Builder::new();
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        let _ = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .expect("function");
        let _ = b.begin_block(None).expect("block");
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let _ = b.i_add(int, None, c2, c3);
        b.ret().expect("ret");
        b.end_function().expect("end");
        let module = b.module().assemble();

        // Create a failing shim executable to stand in for spirv-opt.
        let mut shim = NamedTempFile::new().expect("temp shim");
        shim.write_all(b"#!/bin/sh\nexit 17\n").expect("write shim");
        let mut perms = shim.as_file().metadata().expect("meta").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(shim.path(), perms).expect("chmod shim");

        let mut temp = NamedTempFile::new().expect("temp file");
        temp.write_all(&words_to_bytes(&module))
            .expect("write words");

        let config = OptimizeConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            output: None,
            rust_arith_pass: false,
            cpp_opt_path: Some(shim.path().as_os_str().to_os_string()),
            force_rust_opt: false,
        };
        match run_optimize(&config) {
            Err(OptimizeCliError::CppFailure { status, stderr }) => {
                assert!(!status.success());
                assert!(
                    status.code() == Some(17),
                    "expected exit code 17, got {status:?}"
                );
                assert!(
                    stderr.is_empty(),
                    "shim should not produce stderr, got: {stderr}"
                );
            }
            other => panic!("expected cpp failure, got {other:?}"),
        }
    }
}
