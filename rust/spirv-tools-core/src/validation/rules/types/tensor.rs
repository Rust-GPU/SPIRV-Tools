//! Tensor type validation rules.
//!
//! This module validates SPIR-V tensor type requirements:
//! - OpTypeTensorLayoutNV requirements
//! - OpTypeTensorViewNV requirements
//! - OpTypeTensorARM requirements

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};

use super::helpers::{get_constant_int_value, is_constant_opcode};

// ============================================================================
// OpTypeTensorLayoutNV Validation Rule
// ============================================================================

/// Validates OpTypeTensorLayoutNV requirements.
pub struct TypeTensorLayoutNVRule;

impl ValidationRule for TypeTensorLayoutNVRule {
    fn name(&self) -> &'static str {
        "type-tensor-layout-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeTensorLayoutNV {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Validate Dim (operand 0) - must be 32-bit integer between 1 and 5
            validate_tensor_dim(inst, ctx, type_id, Op::TypeTensorLayoutNV)?;

            // Validate ClampMode (operand 1) - must be 32-bit integer with valid TensorClampMode
            let clamp_id_raw = match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(clamp_result_id) = ResultId::try_from(clamp_id_raw) {
                if let Some(clamp_inst) = ctx.definitions.get(&clamp_result_id) {
                    // Check type is 32-bit integer
                    if let Some(clamp_type_raw) = clamp_inst.result_type {
                        if let Ok(clamp_type_result_id) = ResultId::try_from(clamp_type_raw) {
                            if let Some(clamp_type_inst) = ctx.definitions.get(&clamp_type_result_id)
                            {
                                if clamp_type_inst.class.opcode != Op::TypeInt {
                                    let clamp_id = Id::try_from(clamp_id_raw)
                                        .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                    return Err(
                                        ValidationError::TypeTensorLayoutClampNot32BitInteger {
                                            type_id,
                                            clamp_id,
                                        }.into(),
                        );
                                }
                                // Check width is 32
                                if let Some(Operand::LiteralBit32(width)) =
                                    clamp_type_inst.operands.first()
                                {
                                    if *width != 32 {
                                        let clamp_id = Id::try_from(clamp_id_raw)
                                            .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                        return Err(
                                            ValidationError::TypeTensorLayoutClampNot32BitInteger {
                                                type_id,
                                                clamp_id,
                                            }.into(),
                        );
                                    }
                                }
                            }
                        }
                    }

                    // Check value is a valid TensorClampMode (0-3 based on C++ code)
                    if let Some(clamp_value) = get_constant_int_value(clamp_inst, ctx) {
                        // TensorClampMode::RepeatMirrored is the max value (3)
                        if clamp_value < 0 || clamp_value > 3 {
                            return Err(ValidationError::TypeTensorLayoutClampInvalid { type_id }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeTensorViewNV Validation Rule
// ============================================================================

/// Validates OpTypeTensorViewNV requirements.
pub struct TypeTensorViewNVRule;

impl ValidationRule for TypeTensorViewNVRule {
    fn name(&self) -> &'static str {
        "type-tensor-view-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeTensorViewNV {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Validate Dim (operand 0) - must be 32-bit integer between 1 and 5
            let dim_value = validate_tensor_dim(inst, ctx, type_id, Op::TypeTensorViewNV)?;

            // Validate HasDimensions (operand 1) - must be boolean
            let has_dim_id_raw = match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(has_dim_result_id) = ResultId::try_from(has_dim_id_raw) {
                if let Some(has_dim_inst) = ctx.definitions.get(&has_dim_result_id) {
                    if let Some(has_dim_type_raw) = has_dim_inst.result_type {
                        if let Ok(has_dim_type_result_id) = ResultId::try_from(has_dim_type_raw) {
                            if let Some(has_dim_type_opcode) =
                                ctx.opcodes.get(&has_dim_type_result_id)
                            {
                                if *has_dim_type_opcode != Op::TypeBool {
                                    let has_dim_id = Id::try_from(has_dim_id_raw)
                                        .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                    return Err(ValidationError::TypeTensorViewHasDimNotBool {
                                        type_id,
                                        has_dim_id,
                                    }.into());
                                }
                            }
                        }
                    }
                }
            }

            // Validate permutation values (operands 2+)
            let num_dim = inst.operands.len() - 2; // Subtract Dim and HasDimensions
            let mut permutation_mask: u32 = 0;
            let mut all_constant = true;

            for p_index in 2..inst.operands.len() {
                let p_id_raw = match inst.operands.get(p_index) {
                    Some(Operand::IdRef(id)) => *id,
                    _ => continue,
                };

                if let Ok(p_result_id) = ResultId::try_from(p_id_raw) {
                    if let Some(p_inst) = ctx.definitions.get(&p_result_id) {
                        // Check type is 32-bit integer
                        if let Some(p_type_raw) = p_inst.result_type {
                            if let Ok(p_type_result_id) = ResultId::try_from(p_type_raw) {
                                if let Some(p_type_inst) = ctx.definitions.get(&p_type_result_id) {
                                    if p_type_inst.class.opcode != Op::TypeInt {
                                        let permutation_id = Id::try_from(p_id_raw)
                                            .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                        return Err(
                                            ValidationError::TypeTensorViewPermutationNot32BitInteger {
                                                type_id,
                                                permutation_id,
                                            }.into(),
                        );
                                    }
                                    // Check width is 32
                                    if let Some(Operand::LiteralBit32(width)) =
                                        p_type_inst.operands.first()
                                    {
                                        if *width != 32 {
                                            let permutation_id = Id::try_from(p_id_raw)
                                                .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                            return Err(
                                                ValidationError::TypeTensorViewPermutationNot32BitInteger {
                                                    type_id,
                                                    permutation_id,
                                                }.into(),
                        );
                                        }
                                    }
                                }
                            }
                        }

                        // Check permutation value
                        if let Some(p_value) = get_constant_int_value(p_inst, ctx) {
                            if p_value < 0 || p_value as usize >= num_dim {
                                let permutation_id = Id::try_from(p_id_raw)
                                    .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                return Err(
                                    ValidationError::TypeTensorViewPermutationOutOfRange {
                                        type_id,
                                        permutation_id,
                                    }.into(),
                        );
                            }
                            permutation_mask |= 1 << p_value;
                        } else {
                            all_constant = false;
                        }
                    }
                }
            }

            // Check permutation validity
            if all_constant && permutation_mask != (1u32 << num_dim) - 1 {
                return Err(ValidationError::TypeTensorViewPermutationInvalid { type_id }.into());
            }

            // Check permutation count matches Dim
            if let Some(dim) = dim_value {
                if dim as usize != num_dim {
                    return Err(ValidationError::TypeTensorViewPermutationCountMismatch {
                        type_id,
                    }.into());
                }
            }
        }

        Ok(())
    }
}

/// Validates the Dim operand for tensor types.
fn validate_tensor_dim(
    inst: &rspirv::dr::Instruction,
    ctx: &ValidationContext<'_>,
    type_id: TypeId,
    opcode: Op,
) -> Result<Option<u64>, ValidationError> {
    let dim_id_raw = match inst.operands.first() {
        Some(Operand::IdRef(id)) => *id,
        _ => return Ok(None),
    };

    if let Ok(dim_result_id) = ResultId::try_from(dim_id_raw) {
        if let Some(dim_inst) = ctx.definitions.get(&dim_result_id) {
            // Check type is 32-bit integer
            if let Some(dim_type_raw) = dim_inst.result_type {
                if let Ok(dim_type_result_id) = ResultId::try_from(dim_type_raw) {
                    if let Some(dim_type_inst) = ctx.definitions.get(&dim_type_result_id) {
                        if dim_type_inst.class.opcode != Op::TypeInt {
                            let dim_id = Id::try_from(dim_id_raw)
                                .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                            return Err(ValidationError::TypeTensorDimNot32BitInteger {
                                type_id,
                                opcode,
                                dim_id,
                            }.into());
                        }
                        // Check width is 32
                        if let Some(Operand::LiteralBit32(width)) = dim_type_inst.operands.first() {
                            if *width != 32 {
                                let dim_id = Id::try_from(dim_id_raw)
                                    .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                return Err(ValidationError::TypeTensorDimNot32BitInteger {
                                    type_id,
                                    opcode,
                                    dim_id,
                                }.into());
                            }
                        }
                    }
                }
            }

            // Check value is between 1 and 5
            if let Some(dim_value) = get_constant_int_value(dim_inst, ctx) {
                let dim_u64 = dim_value as u64;
                if dim_u64 == 0 || dim_u64 > 5 {
                    return Err(ValidationError::TypeTensorDimOutOfRange {
                        type_id,
                        opcode,
                        value: dim_u64,
                    }.into());
                }
                return Ok(Some(dim_u64));
            }
        }
    }

    Ok(None)
}

// ============================================================================
// OpTypeTensorARM Validation Rule
// ============================================================================

/// Validates OpTypeTensorARM requirements.
pub struct TypeTensorARMRule;

impl ValidationRule for TypeTensorARMRule {
    fn name(&self) -> &'static str {
        "type-tensor-arm"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeTensorARM {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Validate Element Type (operand 0) - must be scalar (int, float, or bool)
            let element_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(element_result_id) = ResultId::try_from(element_type_raw) {
                if let Some(element_opcode) = ctx.opcodes.get(&element_result_id) {
                    if !matches!(
                        element_opcode,
                        Op::TypeInt | Op::TypeFloat | Op::TypeBool
                    ) {
                        let element_type = TypeId::try_from(element_type_raw)
                            .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                        return Err(ValidationError::TypeTensorARMElementNotScalar {
                            type_id,
                            element_type,
                        }.into());
                    }
                }
            }

            // If we have Rank (operand 1), validate it
            if inst.operands.len() < 2 {
                continue;
            }

            let rank_id_raw = match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            let mut rank_value: Option<u64> = None;

            if let Ok(rank_result_id) = ResultId::try_from(rank_id_raw) {
                if let Some(rank_inst) = ctx.definitions.get(&rank_result_id) {
                    // Must be a constant
                    if !is_constant_opcode(rank_inst.class.opcode) {
                        let rank_id = Id::try_from(rank_id_raw)
                            .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                        return Err(ValidationError::TypeTensorARMRankNotConstant {
                            type_id,
                            rank_id,
                        }.into());
                    }

                    // Must have integer type
                    if let Some(rank_type_raw) = rank_inst.result_type {
                        if let Ok(rank_type_result_id) = ResultId::try_from(rank_type_raw) {
                            if let Some(rank_type_opcode) = ctx.opcodes.get(&rank_type_result_id) {
                                if *rank_type_opcode != Op::TypeInt {
                                    let rank_id = Id::try_from(rank_id_raw)
                                        .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                    return Err(ValidationError::TypeTensorARMRankNotInteger {
                                        type_id,
                                        rank_id,
                                    }.into());
                                }
                            }
                        }
                    }

                    // Check value > 0
                    if let Some(value) = get_constant_int_value(rank_inst, ctx) {
                        if value <= 0 {
                            return Err(ValidationError::TypeTensorARMRankZero { type_id }.into());
                        }
                        rank_value = Some(value as u64);
                    }
                }
            }

            // If we have Shape (operand 2), validate it
            if inst.operands.len() < 3 {
                continue;
            }

            let shape_id_raw = match inst.operands.get(2) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(shape_result_id) = ResultId::try_from(shape_id_raw) {
                if let Some(shape_inst) = ctx.definitions.get(&shape_result_id) {
                    // Must be a constant
                    if !is_constant_opcode(shape_inst.class.opcode) {
                        let shape_id = Id::try_from(shape_id_raw)
                            .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                        return Err(ValidationError::TypeTensorARMShapeNotConstant {
                            type_id,
                            shape_id,
                        }.into());
                    }

                    // Shape must be array of integers with length equal to Rank
                    // This is a simplified check - full validation would need to verify
                    // the array type and constituents
                    if let Some(shape_type_raw) = shape_inst.result_type {
                        if let Ok(shape_type_result_id) = ResultId::try_from(shape_type_raw) {
                            if let Some(shape_type_opcode) = ctx.opcodes.get(&shape_type_result_id) {
                                // Should be OpTypeArray
                                if *shape_type_opcode != Op::TypeArray {
                                    let shape_id = Id::try_from(shape_id_raw)
                                        .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                    return Err(ValidationError::TypeTensorARMShapeNotIntegerArray {
                                        type_id,
                                        shape_id,
                                    }.into());
                                }
                            }
                        }
                    }

                    // Check shape constituents are > 0 (for OpConstantComposite)
                    if shape_inst.class.opcode == Op::ConstantComposite {
                        for (i, operand) in shape_inst.operands.iter().enumerate() {
                            if let Operand::IdRef(constituent_id) = operand {
                                if let Ok(constituent_result_id) =
                                    ResultId::try_from(*constituent_id)
                                {
                                    if let Some(constituent_inst) =
                                        ctx.definitions.get(&constituent_result_id)
                                    {
                                        if let Some(value) =
                                            get_constant_int_value(constituent_inst, ctx)
                                        {
                                            if value <= 0 {
                                                return Err(
                                                    ValidationError::TypeTensorARMShapeConstituentZero {
                                                        type_id,
                                                        index: i,
                                                    }.into(),
                        );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Verify length matches Rank if both are known
                    if let Some(rank) = rank_value {
                        if shape_inst.class.opcode == Op::ConstantComposite
                            && shape_inst.operands.len() != rank as usize
                        {
                            let shape_id = Id::try_from(shape_id_raw)
                                .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                            return Err(ValidationError::TypeTensorARMShapeNotIntegerArray {
                                type_id,
                                shape_id,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Returns all tensor type validation rules.
pub fn all_tensor_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &TypeTensorLayoutNVRule,
        &TypeTensorViewNVRule,
        &TypeTensorARMRule,
    ]
}
