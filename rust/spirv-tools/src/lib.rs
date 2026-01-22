//! API-compatible wrapper around our Rust SPIR-V tools implementation.
//!
//! This crate provides the same API as the `spirv-tools` crate on crates.io,
//! but backed by our pure Rust implementation instead of C++ FFI.

pub mod assembler;
pub mod binary;
pub mod error;
pub mod opt;
pub mod val;

pub use error::{Error, SpirvResult};

// Re-export TargetEnv from our implementation
pub use crate::error::TargetEnv;

// Re-export validation types for consumers who want structured error data
pub use val::{ValidationError, ValidatorError};
