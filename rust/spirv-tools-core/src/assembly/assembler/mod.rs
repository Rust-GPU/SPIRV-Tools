mod module_builder;
mod translator;
mod types;

#[cfg(test)]
mod tests;

use thiserror::Error;

use crate::assembly::options::TextToBinaryOptions;
use crate::diagnostic::DiagnosticMessage;
use crate::target_env::TargetEnv;
use crate::validation::span::SpanMap;

pub use module_builder::ModuleBuilder;
pub use translator::{assemble_instructions, AssemblyTranslator};
pub use types::{
    ArrayTypeInfo, CompositeTypeInfo, MatrixTypeInfo, MemberLayout, StructTypeInfo, VectorTypeInfo,
};

use translator::{
    assemble_text_with_translator, assemble_text_with_translator_for_spans,
};

/// Error emitted when the assembler produces diagnostics instead of a finished module.
#[derive(Debug, Error)]
#[error("assembly failed with diagnostics")]
pub struct AssemblyError {
    diagnostics: Vec<DiagnosticMessage<'static>>,
}

impl AssemblyError {
    pub(crate) fn new(diagnostics: Vec<DiagnosticMessage<'static>>) -> Self {
        Self { diagnostics }
    }

    /// Borrows the underlying diagnostics describing the failure.
    pub fn diagnostics(&self) -> &[DiagnosticMessage<'static>] {
        &self.diagnostics
    }

    /// Consumes this error and returns the owned diagnostics.
    pub fn into_diagnostics(self) -> Vec<DiagnosticMessage<'static>> {
        self.diagnostics
    }
}

/// Result of assembling SPIR-V text with span tracking enabled.
#[derive(Debug)]
pub struct AssemblyWithSpans {
    /// The assembled SPIR-V binary words.
    pub words: Vec<u32>,
    /// Map from result IDs to their source locations.
    pub span_map: SpanMap,
}

/// Assembles a block of textual SPIR-V instructions separated by newlines into a binary module.
/// Returns the assembled words on success along with any diagnostics emitted along the way.
pub fn assemble_text(text: &str) -> Result<Vec<u32>, AssemblyError> {
    assemble_text_with_translator(text, AssemblyTranslator::new())
}

/// Assembles SPIR-V text using the provided target environment to configure the module header.
pub fn assemble_text_with_env(text: &str, env: TargetEnv) -> Result<Vec<u32>, AssemblyError> {
    assemble_text_with_options(text, env, TextToBinaryOptions::NONE)
}

/// Assembles SPIR-V text with the provided options and target environment.
pub fn assemble_text_with_options(
    text: &str,
    env: TargetEnv,
    options: TextToBinaryOptions,
) -> Result<Vec<u32>, AssemblyError> {
    assemble_text_with_translator(
        text,
        AssemblyTranslator::with_target_env_and_options(env, options),
    )
}

/// Assembles SPIR-V text and tracks source locations for all result IDs.
///
/// This is useful for validation error reporting, as the span map can be
/// passed to the validator to provide precise source locations in errors.
///
/// # Example
///
/// ```ignore
/// use spirv_tools_core::assembly::assemble_text_with_spans;
///
/// let text = r#"
///     OpCapability Shader
///     OpMemoryModel Logical GLSL450
///     %void = OpTypeVoid
/// "#;
///
/// let result = assemble_text_with_spans(text)?;
/// // result.span_map now contains the source location for %void
/// ```
pub fn assemble_text_with_spans(text: &str) -> Result<AssemblyWithSpans, AssemblyError> {
    assemble_text_with_spans_and_env(text, TargetEnv::Universal1_6)
}

/// Assembles SPIR-V text with span tracking and a specific target environment.
pub fn assemble_text_with_spans_and_env(
    text: &str,
    env: TargetEnv,
) -> Result<AssemblyWithSpans, AssemblyError> {
    assemble_text_with_spans_full(text, env, TextToBinaryOptions::NONE)
}

/// Assembles SPIR-V text with span tracking, environment, and options.
pub fn assemble_text_with_spans_full(
    text: &str,
    env: TargetEnv,
    options: TextToBinaryOptions,
) -> Result<AssemblyWithSpans, AssemblyError> {
    let translator = AssemblyTranslator::with_full_options(env, options, true);
    assemble_text_with_translator_for_spans(text, translator)
}
