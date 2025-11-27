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
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            input: InputSource::default(),
            output: None,
            rust_arith_pass: true,
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
    if !config.rust_arith_pass {
        return Ok(words);
    }
    let result = optimize_basic_block(&words);
    if result.success {
        Ok(result.words)
    } else {
        Err(OptimizeCliError::Optimize(result.message))
    }
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
        };
        let optimized = run_optimize(&config).expect("optimize passthrough");
        assert_eq!(optimized, module);
    }

    #[test]
    fn rejects_misaligned_input() {
        let temp = NamedTempFile::new().expect("temp file");
        std::fs::write(temp.path(), [1u8, 2, 3]).expect("write bytes");
        let config = OptimizeConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            output: None,
            rust_arith_pass: true,
        };
        match run_optimize(&config) {
            Err(OptimizeCliError::MisalignedInput) => {}
            other => panic!("expected misaligned input error, got {other:?}"),
        }
    }
}
