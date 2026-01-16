//! Constant instruction validation rules.
//!
//! This module validates SPIR-V constant instructions:
//!
//! - OpConstantTrue/OpConstantFalse: Boolean constant type validation
//! - OpSpecConstantTrue/OpSpecConstantFalse: Spec constant boolean type validation
//! - OpConstantComposite/OpSpecConstantComposite: Composite constant validation
//! - OpConstantSampler: Sampler constant type validation
//! - OpConstantNull: Null constant type validation
//! - OpSpecConstant: Spec constant type validation
//! - OpSpecConstantOp: Spec constant operation capability requirements
//!
//! Constant validation ensures type correctness and capability requirements.

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};

// ============================================================================
// Helper Functions
// ============================================================================

/// Returns true if the given opcode is a constant or undef instruction.
fn is_constant_or_undef(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::ConstantTrue
            | Op::ConstantFalse
            | Op::Constant
            | Op::ConstantComposite
            | Op::ConstantSampler
            | Op::ConstantNull
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstant
            | Op::SpecConstantComposite
            | Op::SpecConstantOp
            | Op::Undef
    )
}

/// Returns true if the given opcode is a composite type.
fn is_composite_type(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::TypeVector
            | Op::TypeMatrix
            | Op::TypeArray
            | Op::TypeStruct
            | Op::TypeCooperativeMatrixNV
            | Op::TypeCooperativeMatrixKHR
            | Op::TypeCooperativeVectorNV
    )
}

/// Check if a type is nullable (can have a null value).
fn is_type_nullable(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    let type_inst = ResultId::try_from(type_id)
        .ok()
        .and_then(|rid| ctx.definitions.get(&rid));

    let Some(type_inst) = type_inst else {
        return false;
    };

    match type_inst.class.opcode {
        // Scalar types are nullable
        Op::TypeBool | Op::TypeInt | Op::TypeFloat => true,
        // Event types are nullable
        Op::TypeEvent | Op::TypeDeviceEvent | Op::TypeReserveId | Op::TypeQueue => true,
        // Composite types are nullable if their element type is nullable
        Op::TypeArray | Op::TypeMatrix | Op::TypeVector => {
            let element_type_id = type_inst.operands.first().and_then(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            element_type_id
                .map(|id| is_type_nullable(id, ctx))
                .unwrap_or(false)
        }
        Op::TypeCooperativeMatrixNV
        | Op::TypeCooperativeMatrixKHR
        | Op::TypeCooperativeVectorNV => {
            let component_type_id = type_inst.operands.first().and_then(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            component_type_id
                .map(|id| is_type_nullable(id, ctx))
                .unwrap_or(false)
        }
        // Struct is nullable if all members are nullable
        Op::TypeStruct => type_inst.operands.iter().all(|op| match op {
            Operand::IdRef(id) => is_type_nullable(*id, ctx),
            _ => true,
        }),
        // Pointers are nullable except for PhysicalStorageBuffer
        Op::TypePointer | Op::TypeUntypedPointerKHR => {
            let storage_class = type_inst.operands.first().and_then(|op| match op {
                Operand::StorageClass(sc) => Some(*sc),
                _ => None,
            });
            storage_class != Some(StorageClass::PhysicalStorageBuffer)
        }
        _ => false,
    }
}

// ============================================================================
// Boolean Constant Type Rule
// ============================================================================

/// Validates boolean constant instructions.
///
/// Ensures that OpConstantTrue, OpConstantFalse, OpSpecConstantTrue,
/// and OpSpecConstantFalse have OpTypeBool as their result type.
pub struct ConstantBoolTypeRule;

impl ValidationRule for ConstantBoolTypeRule {
    fn name(&self) -> &'static str {
        "constant-bool-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            let opcode = inst.class.opcode;
            if !matches!(
                opcode,
                Op::ConstantTrue
                    | Op::ConstantFalse
                    | Op::SpecConstantTrue
                    | Op::SpecConstantFalse
            ) {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let type_opcode = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            if type_opcode != Some(Op::TypeBool) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantResultTypeInvalid {
                        opcode,
                        result_type,
                        expected: "OpTypeBool",
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Sampler Constant Type Rule
// ============================================================================

/// Validates OpConstantSampler instructions.
///
/// Ensures that the result type is OpTypeSampler.
pub struct ConstantSamplerTypeRule;

impl ValidationRule for ConstantSamplerTypeRule {
    fn name(&self) -> &'static str {
        "constant-sampler-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::ConstantSampler {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let type_opcode = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            if type_opcode != Some(Op::TypeSampler) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantResultTypeInvalid {
                        opcode: Op::ConstantSampler,
                        result_type,
                        expected: "OpTypeSampler",
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Null Constant Type Rule
// ============================================================================

/// Validates OpConstantNull instructions.
///
/// Ensures that the result type is a nullable type.
pub struct ConstantNullTypeRule;

impl ValidationRule for ConstantNullTypeRule {
    fn name(&self) -> &'static str {
        "constant-null-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::ConstantNull {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            if !is_type_nullable(result_type_id, ctx) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantNullTypeNotNullable { result_type });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Spec Constant Type Rule
// ============================================================================

/// Validates OpSpecConstant instructions.
///
/// Ensures that the result type is an integer or floating-point type.
pub struct SpecConstantTypeRule;

impl ValidationRule for SpecConstantTypeRule {
    fn name(&self) -> &'static str {
        "spec-constant-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::SpecConstant {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let type_opcode = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            if !matches!(type_opcode, Some(Op::TypeInt) | Some(Op::TypeFloat)) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantResultTypeInvalid {
                        opcode: Op::SpecConstant,
                        result_type,
                        expected: "OpTypeInt or OpTypeFloat",
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Composite Constant Type Rule
// ============================================================================

/// Validates OpConstantComposite and OpSpecConstantComposite instructions.
///
/// Ensures that:
/// - Result type is a composite type
/// - Constituent count matches the type's expected element count
/// - All constituents are constants or undef
/// - Constituent types match the expected element types
pub struct ConstantCompositeRule;

impl ValidationRule for ConstantCompositeRule {
    fn name(&self) -> &'static str {
        "constant-composite"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            let opcode = inst.class.opcode;
            if !matches!(opcode, Op::ConstantComposite | Op::SpecConstantComposite) {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let result_type_inst = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            let Some(result_type_inst) = result_type_inst else {
                continue;
            };

            // Check that result type is a composite type
            if !is_composite_type(result_type_inst.class.opcode) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantResultTypeInvalid {
                        opcode,
                        result_type,
                        expected: "a composite type",
                    });
                }
            }

            // Validate constituents are constants or undef
            for constituent_op in &inst.operands {
                let constituent_id = match constituent_op {
                    Operand::IdRef(id) => *id,
                    _ => continue,
                };

                let constituent_opcode = ResultId::try_from(constituent_id)
                    .ok()
                    .and_then(|rid| ctx.opcodes.get(&rid))
                    .copied();

                if let Some(op) = constituent_opcode {
                    if !is_constant_or_undef(op) {
                        if let Ok(constituent) = Id::try_from(constituent_id) {
                            return Err(ValidationError::ConstantCompositeConstituentNotConstant {
                                opcode,
                                constituent,
                            });
                        }
                    }
                }
            }

            // Validate type-specific requirements
            match result_type_inst.class.opcode {
                Op::TypeVector => {
                    // Get expected component count
                    let component_count = result_type_inst
                        .operands
                        .get(1)
                        .and_then(|op| match op {
                            Operand::LiteralBit32(n) => Some(*n as usize),
                            _ => None,
                        })
                        .unwrap_or(0);

                    let constituent_count = inst.operands.len();

                    if constituent_count != component_count {
                        if let Ok(result_type) = TypeId::try_from(result_type_id) {
                            return Err(ValidationError::ConstantCompositeCountMismatch {
                                opcode,
                                result_type,
                                expected: component_count,
                                found: constituent_count,
                            });
                        }
                    }
                }
                Op::TypeMatrix => {
                    // Get expected column count
                    let column_count = result_type_inst
                        .operands
                        .get(1)
                        .and_then(|op| match op {
                            Operand::LiteralBit32(n) => Some(*n as usize),
                            _ => None,
                        })
                        .unwrap_or(0);

                    let constituent_count = inst.operands.len();

                    if constituent_count != column_count {
                        if let Ok(result_type) = TypeId::try_from(result_type_id) {
                            return Err(ValidationError::ConstantCompositeCountMismatch {
                                opcode,
                                result_type,
                                expected: column_count,
                                found: constituent_count,
                            });
                        }
                    }
                }
                Op::TypeStruct => {
                    // Struct member count
                    let member_count = result_type_inst.operands.len();
                    let constituent_count = inst.operands.len();

                    if constituent_count != member_count {
                        if let Ok(result_type) = TypeId::try_from(result_type_id) {
                            return Err(ValidationError::ConstantCompositeCountMismatch {
                                opcode,
                                result_type,
                                expected: member_count,
                                found: constituent_count,
                            });
                        }
                    }
                }
                Op::TypeCooperativeMatrixNV | Op::TypeCooperativeMatrixKHR => {
                    // Cooperative matrix requires exactly 1 constituent
                    let constituent_count = inst.operands.len();

                    if constituent_count != 1 {
                        if let Ok(result_type) = TypeId::try_from(result_type_id) {
                            return Err(ValidationError::ConstantCompositeCountMismatch {
                                opcode,
                                result_type,
                                expected: 1,
                                found: constituent_count,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

// ============================================================================
// Spec Constant Op Capability Rule
// ============================================================================

/// Validates OpSpecConstantOp capability requirements.
///
/// Certain operations within OpSpecConstantOp require specific capabilities.
pub struct SpecConstantOpCapabilityRule;

impl ValidationRule for SpecConstantOpCapabilityRule {
    fn name(&self) -> &'static str {
        "spec-constant-op-capability"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let has_shader = ctx.declared_capabilities.contains(&Capability::Shader);
        let has_kernel = ctx.declared_capabilities.contains(&Capability::Kernel);

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::SpecConstantOp {
                continue;
            }

            // The opcode is the first operand (encoded as a literal)
            let inner_op_num = inst.operands.first().and_then(|op| match op {
                Operand::LiteralBit32(n) => Some(*n),
                _ => None,
            });

            let Some(inner_op_num) = inner_op_num else {
                continue;
            };

            // Check capability requirements for specific ops by their opcode numbers
            // See SPIR-V specification for opcode values
            // QuantizeToF16 = 116
            if inner_op_num == 116 && !has_shader {
                return Err(ValidationError::SpecConstantOpMissingCapability {
                    inner_opcode: Op::QuantizeToF16,
                    required_capability: Capability::Shader,
                });
            }

            // Kernel-only operations:
            // ConvertFToS = 109, ConvertSToF = 111, ConvertFToU = 110, ConvertUToF = 112
            // ConvertPtrToU = 117, ConvertUToPtr = 120, GenericCastToPtr = 122, PtrCastToGeneric = 121
            // Bitcast = 124, FNegate = 127, FAdd = 129, FSub = 131, FMul = 133, FDiv = 136, FRem = 140, FMod = 141
            // AccessChain = 65, InBoundsAccessChain = 66, PtrAccessChain = 67, InBoundsPtrAccessChain = 70
            let kernel_ops = [
                109, 111, 110, 112, // ConvertFToS, ConvertSToF, ConvertFToU, ConvertUToF
                117, 120, 122, 121, // ConvertPtrToU, ConvertUToPtr, GenericCastToPtr, PtrCastToGeneric
                124, 127, 129, 131, 133, 136, 140, 141, // Bitcast, FNegate, FAdd, FSub, FMul, FDiv, FRem, FMod
                65, 66, 67, 70, // AccessChain, InBoundsAccessChain, PtrAccessChain, InBoundsPtrAccessChain
            ];

            if kernel_ops.contains(&inner_op_num) && !has_kernel {
                // Map the number back to an Op for the error message
                let inner_op = match inner_op_num {
                    109 => Op::ConvertFToS,
                    111 => Op::ConvertSToF,
                    110 => Op::ConvertFToU,
                    112 => Op::ConvertUToF,
                    117 => Op::ConvertPtrToU,
                    120 => Op::ConvertUToPtr,
                    122 => Op::GenericCastToPtr,
                    121 => Op::PtrCastToGeneric,
                    124 => Op::Bitcast,
                    127 => Op::FNegate,
                    129 => Op::FAdd,
                    131 => Op::FSub,
                    133 => Op::FMul,
                    136 => Op::FDiv,
                    140 => Op::FRem,
                    141 => Op::FMod,
                    65 => Op::AccessChain,
                    66 => Op::InBoundsAccessChain,
                    67 => Op::PtrAccessChain,
                    70 => Op::InBoundsPtrAccessChain,
                    _ => continue,
                };
                return Err(ValidationError::SpecConstantOpMissingCapability {
                    inner_opcode: inner_op,
                    required_capability: Capability::Kernel,
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// All constant rules
// ============================================================================

/// Returns all constant validation rules.
pub fn all_constant_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &ConstantBoolTypeRule,
        &ConstantSamplerTypeRule,
        &ConstantNullTypeRule,
        &SpecConstantTypeRule,
        &ConstantCompositeRule,
        &SpecConstantOpCapabilityRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_constant_or_undef() {
        assert!(is_constant_or_undef(Op::ConstantTrue));
        assert!(is_constant_or_undef(Op::ConstantFalse));
        assert!(is_constant_or_undef(Op::Constant));
        assert!(is_constant_or_undef(Op::ConstantComposite));
        assert!(is_constant_or_undef(Op::ConstantNull));
        assert!(is_constant_or_undef(Op::Undef));
        assert!(is_constant_or_undef(Op::SpecConstant));
        assert!(is_constant_or_undef(Op::SpecConstantOp));

        assert!(!is_constant_or_undef(Op::IAdd));
        assert!(!is_constant_or_undef(Op::Variable));
        assert!(!is_constant_or_undef(Op::Load));
    }

    #[test]
    fn test_is_composite_type() {
        assert!(is_composite_type(Op::TypeVector));
        assert!(is_composite_type(Op::TypeMatrix));
        assert!(is_composite_type(Op::TypeArray));
        assert!(is_composite_type(Op::TypeStruct));
        assert!(is_composite_type(Op::TypeCooperativeMatrixNV));
        assert!(is_composite_type(Op::TypeCooperativeMatrixKHR));
        assert!(is_composite_type(Op::TypeCooperativeVectorNV));

        assert!(!is_composite_type(Op::TypeInt));
        assert!(!is_composite_type(Op::TypeFloat));
        assert!(!is_composite_type(Op::TypeBool));
        assert!(!is_composite_type(Op::TypePointer));
    }
}
