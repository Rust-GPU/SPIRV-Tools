//! Dot product instruction validation rules.
//!
//! This module validates SPIR-V integer dot product instructions including:
//!
//! - OpSDot, OpUDot, OpSUDot (dot product)
//! - OpSDotAccSat, OpUDotAccSat, OpSUDotAccSat (dot product with accumulate and saturate)
//!
//! These instructions are from SPV_KHR_integer_dot_product and require validation of:
//! - Result type (integer scalar)
//! - Vector operand types (matching integer vectors or packed 32-bit scalars)
//! - Accumulator type (must match result for AccSat variants)
//! - Packed vector format presence for scalar operands

use std::collections::HashMap;

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::TypeInstructionExt;
use crate::validation::types::ResultId;
use crate::validation::ValidationResult;

// ============================================================================
// Helpers
// ============================================================================

/// Gets the type ID (result_type) of the value referenced by an operand.
fn get_operand_type_id(
    inst: &Instruction,
    operand_idx: usize,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<u32> {
    let operand_id = match inst.operands.get(operand_idx)? {
        Operand::IdRef(id) => *id,
        _ => return None,
    };
    let result_id = ResultId::try_from(operand_id).ok()?;
    definitions.get(&result_id)?.result_type
}

/// Looks up a type instruction by its ID.
fn get_type_inst<'a>(
    type_id: u32,
    definitions: &'a HashMap<ResultId, Instruction>,
) -> Option<&'a Instruction> {
    let type_rid = ResultId::try_from(type_id).ok()?;
    definitions.get(&type_rid)
}

/// Checks if a type is an int scalar type.
fn is_int_scalar(type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
    get_type_inst(type_id, definitions)
        .map(|inst| inst.is_int_type())
        .unwrap_or(false)
}

/// Checks if a type is an int scalar with a specific bit width.
fn is_int_scalar_with_width(
    type_id: u32,
    width: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> bool {
    get_type_inst(type_id, definitions)
        .map(|inst| inst.is_int_type() && inst.numeric_bit_width() == Some(width))
        .unwrap_or(false)
}

/// Checks if a type is an unsigned int scalar.
fn is_unsigned_int_scalar(type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
    get_type_inst(type_id, definitions)
        .map(|inst| inst.is_unsigned_int_type())
        .unwrap_or(false)
}

/// Checks if a type is a vector type.
fn is_vector_type(type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
    get_type_inst(type_id, definitions)
        .map(|inst| inst.is_vector_type())
        .unwrap_or(false)
}

/// Gets the vector component count (dimension) for a type.
fn get_dimension(type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> u32 {
    get_type_inst(type_id, definitions)
        .and_then(|inst| inst.vector_component_count())
        .unwrap_or(0)
}

/// Gets the vector component type ID.
fn get_component_type(type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> Option<u32> {
    get_type_inst(type_id, definitions).and_then(|inst| inst.vector_component_type_id())
}

/// Gets the bit width of a numeric type.
fn get_bit_width(type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> u32 {
    get_type_inst(type_id, definitions)
        .and_then(|inst| inst.numeric_bit_width())
        .unwrap_or(0)
}

// ============================================================================
// Dot Product Rule
// ============================================================================

/// Validates integer dot product instructions.
pub struct DotProductRule;

impl ValidationRule for DotProductRule {
    fn name(&self) -> &'static str {
        "dot-product"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    match inst.class.opcode {
                        Op::SDot
                        | Op::UDot
                        | Op::SUDot
                        | Op::SDotAccSat
                        | Op::UDotAccSat
                        | Op::SUDotAccSat => {
                            validate_same_signed_dot(inst, ctx)?;
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates OpSDot, OpUDot, OpSUDot, OpSDotAccSat, OpUDotAccSat, OpSUDotAccSat.
fn validate_same_signed_dot(inst: &Instruction, ctx: &ValidationContext<'_>) -> ValidationResult {
    let opcode = inst.class.opcode;
    let defs = ctx.definitions;

    // Result type must be int scalar
    let Some(result_type_id) = inst.result_type else {
        return Ok(());
    };
    if !is_int_scalar(result_type_id, defs) {
        return Err(ValidationError::DotProductResultNotIntScalar { opcode }.into());
    }

    let has_accumulator = matches!(opcode, Op::SDotAccSat | Op::UDotAccSat | Op::SUDotAccSat);

    // If AccSat variant, accumulator type must match result type
    // Accumulator is at rspirv operand 2 (Vector1=0, Vector2=1, Accumulator=2)
    if has_accumulator {
        if let Some(accumulator_type) = get_operand_type_id(inst, 2, defs) {
            if accumulator_type != result_type_id {
                return Err(ValidationError::DotProductAccumulatorTypeMismatch { opcode }.into());
            }
        }
    }

    // For OpUDot/OpUDotAccSat, result must be unsigned
    if matches!(opcode, Op::UDot | Op::UDotAccSat) {
        if !is_unsigned_int_scalar(result_type_id, defs) {
            return Err(ValidationError::DotProductResultNotUnsignedIntScalar { opcode }.into());
        }
    }

    // Get Vector 1 and Vector 2 type IDs
    // rspirv operands: Vector1=0, Vector2=1
    let Some(vec_1_id) = get_operand_type_id(inst, 0, defs) else {
        return Ok(());
    };
    let Some(vec_2_id) = get_operand_type_id(inst, 1, defs) else {
        return Ok(());
    };

    let is_vec_1_scalar = is_int_scalar_with_width(vec_1_id, 32, defs);
    let is_vec_2_scalar = is_int_scalar_with_width(vec_2_id, 32, defs);

    if is_vec_1_scalar != is_vec_2_scalar {
        // One is scalar, other is not
        return Err(ValidationError::DotProductVectorTypeMismatch { opcode }.into());
    } else if is_vec_1_scalar && is_vec_2_scalar {
        // Both are 32-bit int scalars (packed vector mode)
        let vec_1_width = get_bit_width(vec_1_id, defs);
        let vec_2_width = get_bit_width(vec_2_id, defs);

        if vec_1_width != 32 {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: "Expected 'Vector 1' to be 32-bit when a scalar.".to_string(),
            }
            .into());
        } else if vec_2_width != 32 {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: "Expected 'Vector 2' to be 32-bit when a scalar.".to_string(),
            }
            .into());
        }

        // When packed, result width must be >= 8
        let result_width = get_bit_width(result_type_id, defs);
        if result_width < 8 {
            return Err(ValidationError::DotProductPackedResultWidthTooSmall {
                opcode,
                width: result_width,
            }
            .into());
        }

        // PackedVectorFormat must be present
        // For AccSat: operands = [Vector1, Vector2, Accumulator, PackedVectorFormat?]
        // For non-AccSat: operands = [Vector1, Vector2, PackedVectorFormat?]
        let expected_with_packed = if has_accumulator { 4 } else { 3 };
        let has_packed_vec_format = inst.operands.len() == expected_with_packed;
        if !has_packed_vec_format {
            return Err(ValidationError::DotProductPackedMissingFormat { opcode }.into());
        }
    } else {
        // Both should be vectors
        if !is_vector_type(vec_1_id, defs) {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: "Expected 'Vector 1' to be an int scalar or vector.".to_string(),
            }
            .into());
        } else if !is_vector_type(vec_2_id, defs) {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: "Expected 'Vector 2' to be an int scalar or vector.".to_string(),
            }
            .into());
        }

        let vec_1_length = get_dimension(vec_1_id, defs);
        let vec_2_length = get_dimension(vec_2_id, defs);

        // If both dimensions are known and don't match, error
        if vec_1_length != 0 && vec_2_length != 0 && vec_1_length != vec_2_length {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: format!(
                    "'Vector 1' is {} components but 'Vector 2' is {} components",
                    vec_1_length, vec_2_length
                ),
            }
            .into());
        }

        // Check component types are integer
        let Some(vec_1_comp_type) = get_component_type(vec_1_id, defs) else {
            return Ok(());
        };
        let Some(vec_2_comp_type) = get_component_type(vec_2_id, defs) else {
            return Ok(());
        };

        if !is_int_scalar(vec_1_comp_type, defs) {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: "Expected 'Vector 1' to be a vector of integers.".to_string(),
            }
            .into());
        } else if !is_int_scalar(vec_2_comp_type, defs) {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: "Expected 'Vector 2' to be a vector of integers.".to_string(),
            }
            .into());
        }

        // Check component widths match
        let vec_1_width = get_bit_width(vec_1_comp_type, defs);
        let vec_2_width = get_bit_width(vec_2_comp_type, defs);
        if vec_1_width != vec_2_width {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: format!(
                    "'Vector 1' component is {}-bit but 'Vector 2' component is {}-bit",
                    vec_1_width, vec_2_width
                ),
            }
            .into());
        }

        // Result width must be >= component width
        let result_width = get_bit_width(result_type_id, defs);
        if result_width < vec_1_width {
            return Err(ValidationError::DotProductVectorInvalid {
                opcode,
                message: format!(
                    "Result width ({}) must be greater than or equal to the vectors width ({}).",
                    result_width, vec_1_width
                ),
            }
            .into());
        }

        // For OpUDot/OpUDotAccSat, vectors must be unsigned
        if matches!(opcode, Op::UDot | Op::UDotAccSat) {
            if !is_unsigned_int_scalar(vec_1_comp_type, defs) {
                return Err(ValidationError::DotProductVectorInvalid {
                    opcode,
                    message: "Expected 'Vector 1' to be an vector of unsigned integers."
                        .to_string(),
                }
                .into());
            } else if !is_unsigned_int_scalar(vec_2_comp_type, defs) {
                return Err(ValidationError::DotProductVectorInvalid {
                    opcode,
                    message: "Expected 'Vector 2' to be an vector of unsigned integers."
                        .to_string(),
                }
                .into());
            }
        } else if matches!(opcode, Op::SUDot | Op::SUDotAccSat) {
            // For OpSUDot/OpSUDotAccSat, Vector 2 must be unsigned
            if !is_unsigned_int_scalar(vec_2_comp_type, defs) {
                return Err(ValidationError::DotProductVectorInvalid {
                    opcode,
                    message: "Expected 'Vector 2' to be an vector of unsigned integers."
                        .to_string(),
                }
                .into());
            }
        }
    }

    Ok(())
}

// ============================================================================
// All dot product rules
// ============================================================================

static DOT_PRODUCT_RULE: DotProductRule = DotProductRule;

/// Returns all dot product validation rules.
pub fn all_dot_product_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&DOT_PRODUCT_RULE]
}
