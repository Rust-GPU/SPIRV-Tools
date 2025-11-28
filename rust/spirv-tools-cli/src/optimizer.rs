use crate::assembly::words_to_bytes;
use crate::disassemble::InputSource;
use spirv_tools_ffi::optimize_basic_block;
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
            return Err(OptimizeCliError::Optimize(result.message));
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
        return Err(OptimizeCliError::Optimize(format!(
            "spirv-opt exited with status {code}",
            code = output.status
        )));
    }
    Ok(bytes_to_words(&output.stdout)?)
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
        let sub = b.i_sub(int, None, c7, c7).expect("isub");
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
        let mut found_const_zero = false;
        for inst in module.all_inst_iter() {
            assert_ne!(inst.class.opcode, Op::ISub, "sub should be folded away");
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(sub)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                found_const_zero = true;
            }
        }
        assert!(found_const_zero, "folded constant zero should reuse sub id");
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
}
