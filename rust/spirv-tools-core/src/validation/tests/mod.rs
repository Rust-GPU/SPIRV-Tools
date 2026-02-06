//! Validation tests for SPIR-V modules.
//!
//! This module contains comprehensive tests for the SPIR-V validation logic.
//! Tests are organized by validation category and can be run with:
//! `cargo test --package spirv-tools-core validation::tests`

mod prelude;
use prelude::*;

// Import private validation functions used in tests
use super::rules::block_layout::array_stride;
use super::{parse_module, validate_words_internal};

// Re-export internal modules and functions so submodule tests can reach them
// via `super::` (these are `pub(crate)` in the validation module).
use super::{effective_spirv_version, extension_operand, instruction_layout};

mod access_chain;
mod arithmetic;
mod block_layout;
mod builtins;
mod capabilities;
mod cfg;
mod composites;
mod decorations;
mod entry_points;
mod functions;
mod image;
mod layout_ordering;
mod misc;
mod module_basics;
mod storage_classes;
mod types;
mod versions;
