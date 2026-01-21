//! Type and ID validation rules.
//!
//! This module validates SPIR-V type and ID requirements including:
//!
//! - Result types must be type opcodes
//! - OpTypeFunction parameter validation
//! - Operand definitions
//! - OpTypeInt capability requirements (Int8, Int16, Int64)
//! - OpTypeFloat capability requirements (Float16, Float64)
//! - OpTypeVector component count and capability requirements
//! - OpTypeMatrix column type and count requirements
//! - OpTypeArray/OpTypeRuntimeArray element type requirements
//! - OpTypeStruct member requirements
//! - OpTypePointer requirements
//! - OpTypeCooperativeMatrix requirements
//! - Tensor type requirements
//! - Type uniqueness
//!
//! # Module Organization
//!
//! - [`scalars`]: Scalar type validation (int, float)
//! - [`composites`]: Composite type validation (vector, matrix, array, struct)
//! - [`pointers`]: Pointer type validation
//! - [`cooperative`]: Cooperative matrix/vector type validation
//! - [`tensor`]: Tensor type validation
//! - [`general`]: General type validation (function types, uniqueness, ID pass)
//! - [`helpers`]: Shared helper functions

pub mod composites;
pub mod cooperative;
pub mod general;
pub mod helpers;
pub mod pointers;
pub mod scalars;
pub mod tensor;

// Re-export all rule types for convenience
pub use composites::{
    TypeArrayRule, TypeMatrixRule, TypeRuntimeArrayRule, TypeStructRule, TypeVectorRule,
};
pub use cooperative::{TypeCooperativeMatrixRule, TypeCooperativeVectorNVRule};
pub use general::{
    IdPassRule, OperandDefinitionsRule, ReservedOpcodeRule, ResultTypesAreTypesRule,
    TypeFunctionsRule, TypeUniquenessRule,
};
pub use pointers::{TypeForwardPointerRule, TypePointerRule, TypeUntypedPointerKHRRule};
pub use scalars::{TypeFloatRule, TypeIntRule};
pub use tensor::{TypeTensorARMRule, TypeTensorLayoutNVRule, TypeTensorViewNVRule};

use crate::validation::context::ValidationRule;

/// Returns all type validation rules.
pub fn all_type_rules() -> Vec<&'static dyn ValidationRule> {
    let mut rules = Vec::new();
    rules.extend(general::all_general_rules());
    rules.extend(scalars::all_scalar_rules());
    rules.extend(composites::all_composite_rules());
    rules.extend(pointers::all_pointer_rules());
    rules.extend(cooperative::all_cooperative_rules());
    rules.extend(tensor::all_tensor_rules());
    rules
}
