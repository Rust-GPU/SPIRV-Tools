#![warn(missing_docs)]

//! Command-line helpers for the SPIRV-Tools Rust port.

/// Assembly helpers shared by the CLI binaries.
pub mod assembly;
/// Disassembler-specific utilities reused by the binaries.
pub mod disassemble;
/// Objdump utilities for CLI parity.
pub mod objdump;
/// Optimizer helpers reused by CLI binaries.
pub mod optimizer;
/// Module size reporting helpers.
pub mod size;
/// Validator helpers reused by CLI binaries.
pub mod validation;
