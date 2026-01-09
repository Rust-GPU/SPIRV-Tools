//! Control flow graph validation rules.
//!
//! This module validates SPIR-V control flow requirements including:
//!
//! - Function structure (entry blocks, terminators)
//! - Merge instruction placement
//! - Loop and selection construct validity
//! - Dominator relationships
//! - Block reachability
//!
//! # Control Flow Validation
//!
//! SPIR-V has strict structured control flow requirements. Key rules include:
//!
//! - Every function must start with an OpLabel (entry block)
//! - Every block must end with a terminator instruction
//! - OpSelectionMerge/OpLoopMerge must immediately precede branch instructions
//! - Merge blocks must be dominated by their header blocks
//! - Continue targets must be dominated by loop headers
//!
//! # Adding New CFG Rules
//!
//! CFG validation typically requires:
//!
//! 1. Building a CFG representation from the function blocks
//! 2. Computing dominator trees when needed
//! 3. Validating structural requirements
//!
//! Example pattern:
//!
//! ```ignore
//! pub fn validate_my_cfg_rule(module: &Module) -> Result<(), ValidationError> {
//!     for function in &module.functions {
//!         // Build CFG for this function
//!         // Validate structural requirements
//!     }
//!     Ok(())
//! }
//! ```

// CFG validation is complex and tightly integrated with validate_functions().
// The core logic remains in mod.rs but can be incrementally moved here.
