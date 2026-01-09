//! Atomic instruction validation rules.
//!
//! This module validates SPIR-V atomic instructions including:
//!
//! - OpAtomicLoad, OpAtomicStore, OpAtomicExchange
//! - OpAtomicCompareExchange, OpAtomicCompareExchangeWeak
//! - OpAtomicIIncrement, OpAtomicIDecrement
//! - OpAtomicIAdd, OpAtomicISub
//! - OpAtomicSMin, OpAtomicUMin, OpAtomicSMax, OpAtomicUMax
//! - OpAtomicAnd, OpAtomicOr, OpAtomicXor
//! - OpAtomicFlagTestAndSet, OpAtomicFlagClear
//! - OpAtomicFAddEXT, OpAtomicFMinEXT, OpAtomicFMaxEXT
//!
//! Atomic operations have strict requirements for:
//! - Result type (integer, float, or bool depending on operation)
//! - Pointer storage class
//! - Capability requirements (Int64Atomics, AtomicFloat*EXT)

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::is_vulkan_env;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId};

// ============================================================================
// Atomic Opcode Categories
// ============================================================================

/// All atomic opcodes.
const ATOMIC_OPS: &[Op] = &[
    Op::AtomicLoad,
    Op::AtomicStore,
    Op::AtomicExchange,
    Op::AtomicCompareExchange,
    Op::AtomicCompareExchangeWeak,
    Op::AtomicIIncrement,
    Op::AtomicIDecrement,
    Op::AtomicIAdd,
    Op::AtomicISub,
    Op::AtomicSMin,
    Op::AtomicUMin,
    Op::AtomicSMax,
    Op::AtomicUMax,
    Op::AtomicAnd,
    Op::AtomicOr,
    Op::AtomicXor,
    Op::AtomicFlagTestAndSet,
    Op::AtomicFlagClear,
    Op::AtomicFAddEXT,
    Op::AtomicFMinEXT,
    Op::AtomicFMaxEXT,
];

/// Returns true if the opcode is an atomic instruction.
fn is_atomic_op(op: Op) -> bool {
    ATOMIC_OPS.contains(&op)
}

/// Returns true if the atomic opcode returns a result.
fn has_return_type(op: Op) -> bool {
    !matches!(op, Op::AtomicStore | Op::AtomicFlagClear)
}

/// Returns true if the atomic opcode requires a float result type.
fn requires_float_result(op: Op) -> bool {
    matches!(op, Op::AtomicFAddEXT | Op::AtomicFMinEXT | Op::AtomicFMaxEXT)
}

/// Returns true if the atomic opcode requires an integer result type.
fn requires_int_result(op: Op) -> bool {
    matches!(
        op,
        Op::AtomicCompareExchange
            | Op::AtomicCompareExchangeWeak
            | Op::AtomicIIncrement
            | Op::AtomicIDecrement
            | Op::AtomicIAdd
            | Op::AtomicISub
            | Op::AtomicSMin
            | Op::AtomicUMin
            | Op::AtomicSMax
            | Op::AtomicUMax
            | Op::AtomicAnd
            | Op::AtomicOr
            | Op::AtomicXor
    )
}

/// Returns true if the atomic opcode can have int or float result.
fn allows_int_or_float_result(op: Op) -> bool {
    matches!(op, Op::AtomicLoad | Op::AtomicExchange)
}

/// Returns true if the atomic opcode requires a bool result type.
fn requires_bool_result(op: Op) -> bool {
    matches!(op, Op::AtomicFlagTestAndSet)
}

// ============================================================================
// Storage Class Validation
// ============================================================================

/// Storage classes allowed by universal atomic rules.
const UNIVERSAL_ALLOWED_STORAGE_CLASSES: &[StorageClass] = &[
    StorageClass::Uniform,
    StorageClass::StorageBuffer,
    StorageClass::Workgroup,
    StorageClass::CrossWorkgroup,
    StorageClass::Generic,
    StorageClass::AtomicCounter,
    StorageClass::Image,
    StorageClass::Function,
    StorageClass::PhysicalStorageBuffer,
    StorageClass::TaskPayloadWorkgroupEXT,
];

/// Storage classes allowed in Vulkan for atomics.
const VULKAN_ALLOWED_STORAGE_CLASSES: &[StorageClass] = &[
    StorageClass::Uniform,
    StorageClass::StorageBuffer,
    StorageClass::Workgroup,
    StorageClass::Image,
    StorageClass::PhysicalStorageBuffer,
    StorageClass::TaskPayloadWorkgroupEXT,
];

/// Returns true if the storage class is allowed for atomics universally.
fn is_storage_class_allowed_universal(sc: StorageClass) -> bool {
    UNIVERSAL_ALLOWED_STORAGE_CLASSES.contains(&sc)
}

/// Returns true if the storage class is allowed for atomics in Vulkan.
fn is_storage_class_allowed_vulkan(sc: StorageClass) -> bool {
    VULKAN_ALLOWED_STORAGE_CLASSES.contains(&sc)
}

// ============================================================================
// Atomic Result Type Rule
// ============================================================================

/// Validates that atomic instructions have correct result types.
///
/// Ensures that:
/// - Float atomics (FAdd, FMin, FMax) return float scalars
/// - Integer atomics return integer scalars
/// - Load/Exchange can return int or float
/// - FlagTestAndSet returns bool
pub struct AtomicResultTypeRule;

impl ValidationRule for AtomicResultTypeRule {
    fn name(&self) -> &'static str {
        "atomic-result-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !is_atomic_op(inst.class.opcode) {
                        continue;
                    }

                    if !has_return_type(inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    let op = inst.class.opcode;

                    // Check result type requirements based on opcode
                    if requires_float_result(op) {
                        if !resolver.is_float_scalar(result_type_id, ctx.definitions) {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::AtomicResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: op,
                                    result_type,
                                    expected: "float scalar",
                                });
                            }
                        }
                    } else if requires_int_result(op) {
                        if !resolver.is_int_scalar(result_type_id, ctx.definitions) {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::AtomicResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: op,
                                    result_type,
                                    expected: "integer scalar",
                                });
                            }
                        }
                    } else if allows_int_or_float_result(op) {
                        let is_int = resolver.is_int_scalar(result_type_id, ctx.definitions);
                        let is_float = resolver.is_float_scalar(result_type_id, ctx.definitions);
                        if !is_int && !is_float {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::AtomicResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: op,
                                    result_type,
                                    expected: "integer or float scalar",
                                });
                            }
                        }
                    } else if requires_bool_result(op) {
                        if !resolver.is_bool_scalar(result_type_id, ctx.definitions) {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::AtomicResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: op,
                                    result_type,
                                    expected: "bool scalar",
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Storage Class Rule
// ============================================================================

/// Validates that atomic instructions use valid storage classes.
///
/// Different environments have different allowed storage classes:
/// - Universal: Uniform, StorageBuffer, Workgroup, CrossWorkgroup, Generic,
///   AtomicCounter, Image, Function, PhysicalStorageBuffer, TaskPayloadWorkgroupEXT
/// - Vulkan: Uniform, StorageBuffer, Workgroup, Image, PhysicalStorageBuffer,
///   TaskPayloadWorkgroupEXT
/// - Shader capability: Function storage class is forbidden
pub struct AtomicStorageClassRule;

impl ValidationRule for AtomicStorageClassRule {
    fn name(&self) -> &'static str {
        "atomic-storage-class"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let is_vulkan = is_vulkan_env(ctx.env);
        let has_shader = ctx.has_capability(Capability::Shader);

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !is_atomic_op(inst.class.opcode) {
                        continue;
                    }

                    // Get the pointer operand (first operand for store/clear, after result for others)
                    let pointer_operand_idx = if has_return_type(inst.class.opcode) {
                        0
                    } else {
                        0
                    };

                    let pointer_id = inst.operands.get(pointer_operand_idx).and_then(|op| {
                        match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        }
                    });

                    let Some(pointer_id) = pointer_id else {
                        continue;
                    };

                    // Get the type of the pointer
                    let pointer_type_id = ResultId::try_from(pointer_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type);

                    let Some(pointer_type_id) = pointer_type_id else {
                        continue;
                    };

                    // Look up the pointer type to get storage class
                    let pointer_type_inst = ResultId::try_from(pointer_type_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(pointer_type_inst) = pointer_type_inst else {
                        continue;
                    };

                    if pointer_type_inst.class.opcode != Op::TypePointer {
                        continue;
                    }

                    // Storage class is the first operand of OpTypePointer
                    let storage_class = pointer_type_inst.operands.first().and_then(|op| {
                        match op {
                            Operand::StorageClass(sc) => Some(*sc),
                            _ => None,
                        }
                    });

                    let Some(storage_class) = storage_class else {
                        continue;
                    };

                    // Check universal rules
                    if !is_storage_class_allowed_universal(storage_class) {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicStorageClassForbidden {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                storage_class,
                                reason: "forbidden by universal validation rules",
                            });
                        }
                    }

                    // Check Vulkan rules
                    if is_vulkan && !is_storage_class_allowed_vulkan(storage_class) {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicStorageClassForbidden {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                storage_class,
                                reason: "forbidden in Vulkan environment",
                            });
                        }
                    }

                    // Shader capability forbids Function storage class (except Vulkan)
                    if has_shader && !is_vulkan && storage_class == StorageClass::Function {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicStorageClassForbidden {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                storage_class,
                                reason: "Function storage class forbidden when Shader capability is declared",
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Int64 Capability Rule
// ============================================================================

/// Validates that 64-bit integer atomics require Int64Atomics capability.
pub struct AtomicInt64CapabilityRule;

impl ValidationRule for AtomicInt64CapabilityRule {
    fn name(&self) -> &'static str {
        "atomic-int64-capability"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // If we already have Int64Atomics, everything is fine
        if ctx.has_capability(Capability::Int64Atomics) {
            return Ok(());
        }

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !is_atomic_op(inst.class.opcode) {
                        continue;
                    }

                    // Check if this is a 64-bit integer atomic
                    let data_type_id = if has_return_type(inst.class.opcode) {
                        inst.result_type
                    } else {
                        // For store operations, get the value type
                        inst.operands.get(3).and_then(|op| match op {
                            Operand::IdRef(id) => ResultId::try_from(*id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))
                                .and_then(|inst| inst.result_type),
                            _ => None,
                        })
                    };

                    let Some(data_type_id) = data_type_id else {
                        continue;
                    };

                    // Check if it's a 64-bit integer
                    if resolver.is_int_scalar(data_type_id, ctx.definitions)
                        && resolver.get_bit_width(data_type_id, ctx.definitions) == Some(64)
                    {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicMissingCapability {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                required_capability: Capability::Int64Atomics,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// All atomic rules
// ============================================================================

/// Returns all atomic validation rules.
pub fn all_atomic_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &AtomicResultTypeRule,
        &AtomicStorageClassRule,
        &AtomicInt64CapabilityRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_atomic_op() {
        assert!(is_atomic_op(Op::AtomicLoad));
        assert!(is_atomic_op(Op::AtomicStore));
        assert!(is_atomic_op(Op::AtomicExchange));
        assert!(is_atomic_op(Op::AtomicIAdd));
        assert!(is_atomic_op(Op::AtomicFAddEXT));

        assert!(!is_atomic_op(Op::Load));
        assert!(!is_atomic_op(Op::Store));
        assert!(!is_atomic_op(Op::IAdd));
    }

    #[test]
    fn test_has_return_type() {
        assert!(has_return_type(Op::AtomicLoad));
        assert!(has_return_type(Op::AtomicExchange));
        assert!(has_return_type(Op::AtomicIAdd));

        assert!(!has_return_type(Op::AtomicStore));
        assert!(!has_return_type(Op::AtomicFlagClear));
    }

    #[test]
    fn test_requires_float_result() {
        assert!(requires_float_result(Op::AtomicFAddEXT));
        assert!(requires_float_result(Op::AtomicFMinEXT));
        assert!(requires_float_result(Op::AtomicFMaxEXT));

        assert!(!requires_float_result(Op::AtomicIAdd));
        assert!(!requires_float_result(Op::AtomicLoad));
    }

    #[test]
    fn test_requires_int_result() {
        assert!(requires_int_result(Op::AtomicIAdd));
        assert!(requires_int_result(Op::AtomicISub));
        assert!(requires_int_result(Op::AtomicCompareExchange));
        assert!(requires_int_result(Op::AtomicAnd));

        assert!(!requires_int_result(Op::AtomicLoad));
        assert!(!requires_int_result(Op::AtomicFAddEXT));
    }

    #[test]
    fn test_allows_int_or_float_result() {
        assert!(allows_int_or_float_result(Op::AtomicLoad));
        assert!(allows_int_or_float_result(Op::AtomicExchange));

        assert!(!allows_int_or_float_result(Op::AtomicIAdd));
        assert!(!allows_int_or_float_result(Op::AtomicFAddEXT));
    }

    #[test]
    fn test_requires_bool_result() {
        assert!(requires_bool_result(Op::AtomicFlagTestAndSet));

        assert!(!requires_bool_result(Op::AtomicLoad));
        assert!(!requires_bool_result(Op::AtomicIAdd));
    }

    #[test]
    fn test_universal_storage_classes() {
        assert!(is_storage_class_allowed_universal(StorageClass::Uniform));
        assert!(is_storage_class_allowed_universal(StorageClass::Workgroup));
        assert!(is_storage_class_allowed_universal(StorageClass::Function));
        assert!(is_storage_class_allowed_universal(StorageClass::Image));

        assert!(!is_storage_class_allowed_universal(StorageClass::Private));
        assert!(!is_storage_class_allowed_universal(StorageClass::Input));
        assert!(!is_storage_class_allowed_universal(StorageClass::Output));
    }

    #[test]
    fn test_vulkan_storage_classes() {
        assert!(is_storage_class_allowed_vulkan(StorageClass::Uniform));
        assert!(is_storage_class_allowed_vulkan(StorageClass::Workgroup));
        assert!(is_storage_class_allowed_vulkan(StorageClass::Image));

        // Function is NOT allowed in Vulkan
        assert!(!is_storage_class_allowed_vulkan(StorageClass::Function));
        assert!(!is_storage_class_allowed_vulkan(StorageClass::Generic));
    }
}
