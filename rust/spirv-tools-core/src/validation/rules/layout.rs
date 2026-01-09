//! Block and memory layout validation rules.
//!
//! This module validates SPIR-V memory layout requirements for Block and
//! BufferBlock decorated structs, including offset alignment, stride
//! requirements, and overlap detection.
//!
//! # Layout Modes
//!
//! The validator supports several layout modes configured via `ValidationOptions`:
//!
//! - **Standard layout**: Strict SPIR-V layout rules (no relaxations)
//! - **Relaxed block layout** (`relax_block_layout`): Allows vectors to align
//!   to scalar element size
//! - **Uniform buffer standard layout** (`uniform_buffer_standard_layout`):
//!   VK_KHR_uniform_buffer_standard_layout semantics
//! - **Scalar block layout** (`scalar_block_layout`): Full scalar alignment
//! - **Workgroup scalar layout** (`workgroup_scalar_block_layout`): Scalar
//!   alignment for Workgroup storage class
//!
//! # Adding New Layout Rules
//!
//! Layout rules are typically added to `enforce_block_layout_rules()`. Consider:
//!
//! 1. Which storage classes the rule applies to
//! 2. Whether it respects layout relaxation options
//! 3. Adding appropriate error messages to `ValidationError`

// Layout validation is complex and tightly integrated with the main validator.
// The core logic remains in mod.rs but this module provides documentation
// for the layout validation subsystem. Future refactoring may move more code here.
