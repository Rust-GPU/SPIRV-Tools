//! Invalid type usage validation rules.
//!
//! This module validates that certain types (BFloat16, FP8 E4M3/E5M2) are not
//! used with operations that don't support them.
//!
//! Many SPIR-V operations do not support these special floating-point types,
//! and using them with unsupported operations is an error.

use rspirv::dr::Operand;
use rspirv::spirv::Op;
use std::collections::HashMap;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId};
use crate::validation::ValidationResult;

fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Check if a type is BFloat16 (16-bit brain floating point).
fn is_bfloat16_type(
    type_id: u32,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(inst) = definitions.get(&result_id) {
            // BFloat16 is typically represented as a float with specific width
            // In SPIR-V, it might be OpTypeFloat with width 16 and specific encoding
            // For now, check for TypeFloat with specific FP encoding decoration
            if inst.class.opcode == Op::TypeFloat {
                // BFloat16 uses FPEncoding decoration with value 1
                // This is a simplification - actual implementation needs to check decorations
                if let Some(Operand::LiteralBit32(width)) = inst.operands.first() {
                    // 16-bit float could be bfloat16 if it has the FPEncoding decoration
                    // For now, we don't have direct access to check this in a simple way
                    // The actual check would need to look at decorations on this type
                    return *width == 16 && has_bfloat16_encoding(result_id, definitions);
                }
            }
            // Also check vectors of BFloat16
            if inst.class.opcode == Op::TypeVector {
                if let Some(Operand::IdRef(comp_type)) = inst.operands.first() {
                    return is_bfloat16_type(*comp_type, definitions);
                }
            }
        }
    }
    false
}

/// Check if a type has BFloat16 encoding (FPEncoding = BFloat16KHR).
///
/// BFloat16 is indicated by the FPEncoding operand on OpTypeFloat, not by a decoration.
/// The encoding is stored as the second operand of OpTypeFloat (after the width).
fn has_bfloat16_encoding(
    type_id: ResultId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    if let Some(inst) = definitions.get(&type_id) {
        // BFloat16 encoding is specified as the second operand of OpTypeFloat
        // (first operand is the width)
        return inst
            .operands
            .get(1)
            .map(|op| {
                matches!(
                    op,
                    Operand::FPEncoding(rspirv::spirv::FPEncoding::BFloat16KHR)
                )
            })
            .unwrap_or(false);
    }
    false
}

/// Check if a type is FP8 E4M3 or E5M2.
fn is_fp8_type(type_id: u32, definitions: &HashMap<ResultId, rspirv::dr::Instruction>) -> bool {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(inst) = definitions.get(&result_id) {
            if inst.class.opcode == Op::TypeFloat {
                if let Some(Operand::LiteralBit32(width)) = inst.operands.first() {
                    // 8-bit float is FP8
                    if *width == 8 {
                        return true;
                    }
                }
            }
            // Also check vectors of FP8
            if inst.class.opcode == Op::TypeVector {
                if let Some(Operand::IdRef(comp_type)) = inst.operands.first() {
                    return is_fp8_type(*comp_type, definitions);
                }
            }
        }
    }
    false
}

/// Check if a type contains BFloat16 (including matrix component types).
fn contains_bfloat16(
    type_id: u32,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    if is_bfloat16_type(type_id, definitions) {
        return true;
    }

    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(inst) = definitions.get(&result_id) {
            // Check matrix column type
            if inst.class.opcode == Op::TypeMatrix {
                if let Some(Operand::IdRef(col_type)) = inst.operands.first() {
                    return is_bfloat16_type(*col_type, definitions);
                }
            }
        }
    }
    false
}

/// Check if a type contains FP8 (including matrix component types).
fn contains_fp8(type_id: u32, definitions: &HashMap<ResultId, rspirv::dr::Instruction>) -> bool {
    if is_fp8_type(type_id, definitions) {
        return true;
    }

    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(inst) = definitions.get(&result_id) {
            if inst.class.opcode == Op::TypeMatrix {
                if let Some(Operand::IdRef(col_type)) = inst.operands.first() {
                    return is_fp8_type(*col_type, definitions);
                }
            }
        }
    }
    false
}

/// Get the operand's type ID for a given instruction operand index.
fn get_operand_type_id(
    inst: &rspirv::dr::Instruction,
    operand_idx: usize,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let operand_id = inst.operands.get(operand_idx).and_then(|op| match op {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    })?;

    let operand_inst = ResultId::try_from(operand_id)
        .ok()
        .and_then(|rid| definitions.get(&rid))?;

    operand_inst.result_type
}

/// Operations that don't support BFloat16 or FP8 types on their result.
const UNSUPPORTED_RESULT_TYPE_OPS: &[Op] = &[
    // Arithmetic operations
    Op::FAdd,
    Op::FSub,
    Op::FMul,
    Op::FDiv,
    Op::FRem,
    Op::FMod,
    Op::FNegate,
    // Derivative operations
    Op::DPdx,
    Op::DPdy,
    Op::Fwidth,
    Op::DPdxFine,
    Op::DPdyFine,
    Op::FwidthFine,
    Op::DPdxCoarse,
    Op::DPdyCoarse,
    Op::FwidthCoarse,
    // Atomic operations
    Op::AtomicFAddEXT,
    Op::AtomicFMinEXT,
    Op::AtomicFMaxEXT,
    Op::AtomicLoad,
    Op::AtomicExchange,
    // Group operations
    Op::GroupNonUniformRotateKHR,
    Op::GroupNonUniformBroadcast,
    Op::GroupNonUniformShuffle,
    Op::GroupNonUniformShuffleXor,
    Op::GroupNonUniformShuffleUp,
    Op::GroupNonUniformShuffleDown,
    Op::GroupNonUniformQuadBroadcast,
    Op::GroupNonUniformQuadSwap,
    Op::GroupNonUniformBroadcastFirst,
    Op::GroupNonUniformFAdd,
    Op::GroupNonUniformFMul,
    Op::GroupNonUniformFMin,
    // Extended instructions
    Op::ExtInst,
];

/// Operations that check operand types instead of result types.
const UNSUPPORTED_OPERAND_TYPE_OPS: &[Op] = &[
    Op::IsNan,
    Op::IsInf,
    Op::IsFinite,
    Op::IsNormal,
    Op::SignBitSet,
];

/// Validates that BFloat16 and FP8 types are not used with unsupported operations.
pub struct InvalidTypeRule;

impl ValidationRule for InvalidTypeRule {
    fn name(&self) -> &'static str {
        "invalid-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    // Check result type for most operations
                    if UNSUPPORTED_RESULT_TYPE_OPS.contains(&inst.class.opcode) {
                        if let Some(result_type_id) = inst.result_type {
                            if contains_bfloat16(result_type_id, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeBFloat16 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                            if contains_fp8(result_type_id, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeFP8 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                        }
                    }

                    // Check operand types for specific operations
                    if UNSUPPORTED_OPERAND_TYPE_OPS.contains(&inst.class.opcode) {
                        // These ops have the operand at index 0 (after result type/id)
                        if let Some(operand_type) = get_operand_type_id(inst, 0, ctx.definitions) {
                            if contains_bfloat16(operand_type, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeBFloat16 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                            if contains_fp8(operand_type, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeFP8 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                        }
                    }

                    // Special handling for OpAtomicStore (check data operand)
                    if inst.class.opcode == Op::AtomicStore {
                        // Data is operand index 3
                        if let Some(data_type) = get_operand_type_id(inst, 3, ctx.definitions) {
                            if contains_bfloat16(data_type, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeBFloat16 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                            if contains_fp8(data_type, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeFP8 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                        }
                    }

                    // Special handling for OpGroupNonUniformAllEqual (check value operand)
                    if inst.class.opcode == Op::GroupNonUniformAllEqual {
                        // Value is operand index 1 (after Execution scope)
                        if let Some(value_type) = get_operand_type_id(inst, 1, ctx.definitions) {
                            if contains_bfloat16(value_type, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeBFloat16 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                            if contains_fp8(value_type, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeFP8 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                        }
                    }

                    // Special handling for OpMatrixTimesMatrix
                    if inst.class.opcode == Op::MatrixTimesMatrix {
                        if let Some(result_type_id) = inst.result_type {
                            if contains_bfloat16(result_type_id, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeBFloat16 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                            if contains_fp8(result_type_id, ctx.definitions) {
                                return Err(ValidationError::InvalidTypeFP8 {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Returns all invalid type validation rules.
pub fn all_invalid_type_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![Box::new(InvalidTypeRule)]
}
