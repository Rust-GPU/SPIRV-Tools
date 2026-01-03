//! SPIR-V module optimization using egglog.
//!
//! This module provides the main entry points for optimizing SPIR-V modules.
//! All optimization is done in a SINGLE egglog pass for the ENTIRE module,
//! enabling cross-function optimizations.

use crate::direct::optimize_module_direct;
use crate::egglog_opt::EgglogOptError;
use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Module;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;
use thiserror::Error;

/// Error type for module-level optimization.
#[derive(Debug, Error)]
pub enum ModuleError {
    /// Failed to parse SPIR-V binary.
    #[error("failed to parse SPIR-V module: {0}")]
    ParseError(String),
    /// Failed to optimize the module.
    #[error("optimization error: {0}")]
    OptimizeError(#[from] EgglogOptError),
}

/// Optimize a SPIR-V module given as raw bytes.
///
/// This performs whole-module optimization in a single egglog pass.
pub fn optimize_bytes(bytes: &[u8]) -> Result<Vec<u8>, ModuleError> {
    if bytes.len() % 4 != 0 {
        return Err(ModuleError::ParseError(
            "input length is not a multiple of 4".to_string(),
        ));
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let optimized = optimize_words(&words)?;
    Ok(optimized.iter().flat_map(|w| w.to_le_bytes()).collect())
}

/// Optimize a SPIR-V module given as words.
///
/// This performs whole-module optimization in a single egglog pass.
pub fn optimize_words(words: &[u32]) -> Result<Vec<u32>, ModuleError> {
    let mut loader = rspirv::dr::Loader::new();
    parse_words(words, &mut loader).map_err(|e| ModuleError::ParseError(e.to_string()))?;
    let module = loader.module();
    let optimized = optimize_module(&module)?;
    Ok(optimized.assemble())
}

/// Optimize all functions in a SPIR-V module using ONE unified e-graph pass.
///
/// This collects all optimizable instructions from all functions and runs
/// a single egglog optimization pass over the entire module, enabling:
/// - Cross-function constant propagation
/// - Global common subexpression elimination
/// - Inter-procedural algebraic simplifications
pub fn optimize_module(module: &Module) -> Result<Module, ModuleError> {
    optimize_module_direct(module).map_err(ModuleError::OptimizeError)
}

/// Extract integer type widths from a parsed module.
pub fn type_widths_from_module(module: &Module) -> HashMap<Word, u32> {
    module
        .types_global_values
        .iter()
        .filter_map(|inst| match inst.class.opcode {
            Op::TypeInt => inst.result_id.and_then(|id| {
                inst.operands.first().and_then(|op| match op {
                    rspirv::dr::Operand::LiteralBit32(bits) => Some((id, *bits)),
                    _ => None,
                })
            }),
            Op::TypeBool => inst.result_id.map(|id| (id, 1)),
            _ => None,
        })
        .collect()
}
