//! Memory model validation rules.
//!
//! This module validates SPIR-V memory model requirements.

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;

// ============================================================================
// Memory Model Rule
// ============================================================================

/// Validates that the module contains a memory model instruction.
pub struct MemoryModelRule;

impl ValidationRule for MemoryModelRule {
    fn name(&self) -> &'static str {
        "memory-model"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        if ctx.module.memory_model.is_none() {
            return Err(ValidationError::MissingMemoryModel);
        }
        Ok(())
    }
}

// ============================================================================
// All memory rules
// ============================================================================

/// Returns all memory validation rules.
pub fn all_memory_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&MemoryModelRule]
}
