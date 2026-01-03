#![recursion_limit = "256"]

//! E-graph driven optimizer for SPIR-V using egglog.
//!
//! This crate provides whole-module optimization for SPIR-V modules using
//! equality saturation. All optimization happens in a SINGLE egglog pass
//! for the ENTIRE module, enabling cross-function optimizations.
//!
//! # Quick Start
//!
//! ```no_run
//! use spirv_tools_opt::optimize_words;
//!
//! fn optimize(words: &[u32]) -> Vec<u32> {
//!     optimize_words(words).unwrap()
//! }
//! ```

pub mod direct;
pub mod egglog_opt;
pub mod translate;

// Re-export the primary API - whole module optimization using rspirv
pub use translate::{optimize_bytes, optimize_module, optimize_words, ModuleError};

// Re-export key types from egglog_opt
pub use egglog_opt::{create_spirv_egraph, EgglogOptError};
