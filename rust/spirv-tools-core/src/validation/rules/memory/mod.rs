//! Memory operation validation rules.
//!
//! This module validates SPIR-V memory operations including:
//!
//! - Variables (`OpVariable`)
//! - Load/Store operations (`OpLoad`, `OpStore`)
//! - Access chains (`OpAccessChain`, `OpInBoundsAccessChain`, `OpPtrAccessChain`, etc.)
//! - Cooperative matrix/vector operations
//! - Array length and copy operations
//! - Pointer comparisons
//! - Memory model validation

mod access_chain;
mod cooperative;
pub mod helpers;
mod load_store;
mod misc;
mod variables;

// Re-export all rules
pub use access_chain::{AccessChainRule, PtrAccessChainRule, RawAccessChainRule};
pub use cooperative::{
    CooperativeMatrixLengthRule, CooperativeMatrixLoadStoreKHRRule,
    CooperativeMatrixLoadStoreNVRule, CooperativeMatrixMulAddKHRRule,
    CooperativeMatrixMulAddNVRule, CooperativeMatrixPerElementOpNVRule,
    CooperativeVectorLoadStoreNVRule,
};
pub use load_store::{LoadRule, StoreRule};
pub use misc::{ArrayLengthRule, CopyMemoryRule, MemoryModelRule, PointerComparisonRule};
pub use variables::VariableRule;

use crate::validation::context::ValidationRule;

/// Returns all memory validation rules.
pub fn all_memory_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &MemoryModelRule,
        &VariableRule,
        &LoadRule,
        &StoreRule,
        &AccessChainRule,
        &ArrayLengthRule,
        &CopyMemoryRule,
        &PointerComparisonRule,
        &RawAccessChainRule,
        &PtrAccessChainRule,
        // Cooperative matrix rules
        &CooperativeMatrixLengthRule,
        &CooperativeMatrixLoadStoreNVRule,
        &CooperativeMatrixLoadStoreKHRRule,
        &CooperativeMatrixMulAddNVRule,
        &CooperativeMatrixMulAddKHRRule,
        &CooperativeMatrixPerElementOpNVRule,
        // Cooperative vector rules
        &CooperativeVectorLoadStoreNVRule,
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use rspirv::dr::Operand;
    use rspirv::spirv::{Op, StorageClass};

    use crate::validation::context::{TestContextData, ValidationRule};
    use crate::validation::error::ValidationError;

    use super::helpers::{allows_non_private_pointer, is_logical_pointer_producer, is_readonly_storage_class};
    use super::MemoryModelRule;

    #[test]
    fn test_memory_model_present() {
        let mut data = TestContextData::default();
        data.module.memory_model = Some(rspirv::dr::Instruction::new(
            Op::MemoryModel,
            None,
            None,
            vec![
                Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
            ],
        ));
        let ctx = data.as_context();
        assert!(MemoryModelRule.validate(&ctx).is_ok());
    }

    #[test]
    fn test_memory_model_missing() {
        let data = TestContextData::default();
        let ctx = data.as_context();
        let result = MemoryModelRule.validate(&ctx);
        assert!(result.is_err());
        if let Err(spanned) = result {
            assert!(matches!(spanned.error, ValidationError::MissingMemoryModel));
        }
    }

    #[test]
    fn test_is_logical_pointer_producer() {
        assert!(is_logical_pointer_producer(Op::Variable));
        assert!(is_logical_pointer_producer(Op::AccessChain));
        assert!(is_logical_pointer_producer(Op::FunctionParameter));
        assert!(!is_logical_pointer_producer(Op::IAdd));
        assert!(!is_logical_pointer_producer(Op::FAdd));
    }

    #[test]
    fn test_is_readonly_storage_class() {
        assert!(is_readonly_storage_class(StorageClass::UniformConstant));
        assert!(is_readonly_storage_class(StorageClass::Input));
        assert!(is_readonly_storage_class(StorageClass::PushConstant));
        assert!(!is_readonly_storage_class(StorageClass::Private));
        assert!(!is_readonly_storage_class(StorageClass::Function));
    }

    #[test]
    fn test_allows_non_private_pointer() {
        assert!(allows_non_private_pointer(StorageClass::Uniform));
        assert!(allows_non_private_pointer(StorageClass::Workgroup));
        assert!(allows_non_private_pointer(StorageClass::StorageBuffer));
        assert!(!allows_non_private_pointer(StorageClass::Private));
        assert!(!allows_non_private_pointer(StorageClass::Function));
    }
}
