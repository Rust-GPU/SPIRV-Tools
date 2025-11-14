#![warn(missing_docs)]

//! Core, type-safe building blocks for the SPIRV-Tools Rust port.

/// Endianness modeling utilities.
pub mod endian;
/// Message severity abstractions.
pub mod message;
/// Result codes and helpers.
pub mod result;
/// Target environment modeling.
pub mod target_env;
/// Version helpers for SPIR-V and client APIs.
pub mod version;

pub use endian::Endianness;
pub use message::MessageLevel;
pub use result::{InvalidSpvResult, SpvResult};
pub use target_env::TargetEnv;
pub use version::{SpirvVersion, VulkanVersion};
