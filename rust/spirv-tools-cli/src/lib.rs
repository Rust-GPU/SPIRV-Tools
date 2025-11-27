#![warn(missing_docs)]

//! Command-line helpers for the SPIRV-Tools Rust port.

/// Assembly helpers shared by the CLI binaries.
pub mod assembly;
/// Disassembler-specific utilities reused by the binaries.
pub mod disassemble;
/// Optimizer helpers reused by CLI binaries.
pub mod optimizer;
/// Validator helpers reused by CLI binaries.
pub mod validation;
