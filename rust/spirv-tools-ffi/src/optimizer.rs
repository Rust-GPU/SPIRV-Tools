use spirv_tools_opt::translate::{optimize_words, ModuleError};
use thiserror::Error;

/// Errors produced by the arithmetic optimizer.
#[derive(Debug, Error)]
pub enum OptimizeError {
    /// The input failed to parse as SPIR-V.
    #[error("failed to parse module: {0}")]
    Parse(String),
    /// The arithmetic optimizer reported a failure.
    #[error("optimization failed: {0}")]
    Rewrite(String),
}

impl From<ModuleError> for OptimizeError {
    fn from(err: ModuleError) -> Self {
        match err {
            ModuleError::ParseError(s) => OptimizeError::Parse(s),
            ModuleError::OptimizeError(e) => OptimizeError::Rewrite(e.to_string()),
        }
    }
}

/// Optimize a SPIR-V module (given as words) using the egglog-based optimizer.
///
/// This performs whole-module optimization in a single egglog pass,
/// enabling cross-function optimizations.
pub fn optimize_basic_block(insts: &[u32]) -> Result<Vec<u32>, OptimizeError> {
    optimize_words(insts).map_err(OptimizeError::from)
}
