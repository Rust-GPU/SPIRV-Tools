//! Tensor instruction validation rules.
//!
//! This module validates SPIR-V tensor instructions including:
//!
//! - ARM tensor operations (OpTensorReadARM, OpTensorWriteARM, OpTensorQuerySizeARM)
//! - NVIDIA tensor layout/view operations
//!
//! Note: These are extension-specific instructions that may not be present
//! in all SPIR-V environments.

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::helpers::{get_type_structure, is_constant_opcode};
use crate::validation::types::{Id, ResultId, ScalarKind, TypeId, TypeStructure};

/// Helper to convert a u32 to Id (with fallback to id 1).
fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Check if a type is OpTypeTensorARM with a rank specified.
fn is_ranked_tensor_arm(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    let Some(result_id) = ResultId::try_from(type_id).ok() else {
        return false;
    };
    let Some(inst) = ctx.definitions.get(&result_id) else {
        return false;
    };

    // OpTypeTensorARM has at least 4 words: opcode, result_id, element_type, rank
    inst.class.opcode == Op::TypeTensorARM && inst.operands.len() >= 2
}

/// Get the rank operand ID from OpTypeTensorARM.
/// Returns the ID of the rank operand (operand index 1).
fn get_tensor_arm_rank_id(type_id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let result_id = ResultId::try_from(type_id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;
    if inst.class.opcode != Op::TypeTensorARM {
        return None;
    }
    // OpTypeTensorARM: result_type (none), result_id, element_type (operand 0), rank (operand 1)
    match inst.operands.get(1) {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

/// Evaluate a constant instruction to get its u64 value.
/// Works for OpConstant with 32-bit or 64-bit integer types.
fn eval_constant_u64(id: u32, ctx: &ValidationContext<'_>) -> Option<u64> {
    let result_id = ResultId::try_from(id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;

    // Must be a constant instruction
    if !is_constant_opcode(inst.class.opcode) {
        return None;
    }

    // For OpConstant, get the literal value
    if inst.class.opcode == Op::Constant {
        match inst.operands.first() {
            Some(Operand::LiteralBit32(v)) => return Some(*v as u64),
            Some(Operand::LiteralBit64(v)) => return Some(*v),
            _ => return None,
        }
    }

    // For spec constants, we can't evaluate at validation time
    None
}

/// Check if an ID refers to an instruction with integer scalar type.
fn is_int_scalar_type_id(id: u32, ctx: &ValidationContext<'_>) -> bool {
    let Some(result_id) = ResultId::try_from(id).ok() else {
        return false;
    };
    let Some(inst) = ctx.definitions.get(&result_id) else {
        return false;
    };
    let Some(type_id) = inst.result_type else {
        return false;
    };
    let Ok(tid) = TypeId::try_from(type_id) else {
        return false;
    };
    let ty = get_type_structure(tid, ctx.definitions);
    matches!(
        ty,
        TypeStructure::Scalar(ScalarKind::SignedInt(_) | ScalarKind::UnsignedInt(_))
    )
}

/// Check if a type is OpTypeTensorLayoutNV.
fn is_tensor_layout_nv(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    let Some(result_id) = ResultId::try_from(type_id).ok() else {
        return false;
    };
    let Some(inst) = ctx.definitions.get(&result_id) else {
        return false;
    };

    inst.class.opcode == Op::TypeTensorLayoutNV
}

/// Check if a type is OpTypeTensorViewNV.
fn is_tensor_view_nv(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    let Some(result_id) = ResultId::try_from(type_id).ok() else {
        return false;
    };
    let Some(inst) = ctx.definitions.get(&result_id) else {
        return false;
    };

    inst.class.opcode == Op::TypeTensorViewNV
}

/// Check if a type is a scalar type or array of scalar type.
fn is_scalar_or_array_of_scalar(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    let ty = get_type_structure(type_id, ctx.definitions);
    match ty {
        TypeStructure::Scalar(_) => true,
        TypeStructure::Array { element, .. } => {
            let elem_ty = get_type_structure(element, ctx.definitions);
            matches!(elem_ty, TypeStructure::Scalar(_))
        }
        _ => false,
    }
}

/// Validates ARM tensor read operations.
pub struct TensorReadARMRule;

impl ValidationRule for TensorReadARMRule {
    fn name(&self) -> &'static str {
        "tensor-read-arm"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode != Op::TensorReadARM {
                        continue;
                    }

                    // Result Type must be a scalar type or array of scalar type
                    if let Some(result_type) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type) {
                            if !is_scalar_or_array_of_scalar(type_id, ctx) {
                                return Err(ValidationError::TensorReadResultNotScalar {
                                    instruction_id: inst.result_id.map(to_id),
                                }.into());
                            }
                        }
                    }

                    // Tensor must be a ranked tensor
                    if let Some(Operand::IdRef(tensor_id)) = inst.operands.first() {
                        if let Ok(result_id) = ResultId::try_from(*tensor_id) {
                            if let Some(tensor_inst) = ctx.definitions.get(&result_id) {
                                if let Some(tensor_type) = tensor_inst.result_type {
                                    if !is_ranked_tensor_arm(tensor_type, ctx) {
                                        return Err(ValidationError::TensorNotRankedTensor {
                                            instruction_id: inst.result_id.map(to_id),
                                        }.into());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates ARM tensor write operations.
pub struct TensorWriteARMRule;

impl ValidationRule for TensorWriteARMRule {
    fn name(&self) -> &'static str {
        "tensor-write-arm"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode != Op::TensorWriteARM {
                        continue;
                    }

                    // Tensor must be a ranked tensor
                    if let Some(Operand::IdRef(tensor_id)) = inst.operands.first() {
                        if let Ok(result_id) = ResultId::try_from(*tensor_id) {
                            if let Some(tensor_inst) = ctx.definitions.get(&result_id) {
                                if let Some(tensor_type) = tensor_inst.result_type {
                                    if !is_ranked_tensor_arm(tensor_type, ctx) {
                                        return Err(ValidationError::TensorNotRankedTensor {
                                            instruction_id: None,
                                        }.into());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates ARM tensor query size operations.
pub struct TensorQuerySizeARMRule;

impl ValidationRule for TensorQuerySizeARMRule {
    fn name(&self) -> &'static str {
        "tensor-query-size-arm"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode != Op::TensorQuerySizeARM {
                        continue;
                    }

                    // Result Type must be an integer type scalar
                    if let Some(result_type) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            let is_int_scalar = matches!(
                                ty,
                                TypeStructure::Scalar(
                                    crate::validation::types::ScalarKind::SignedInt(_)
                                        | crate::validation::types::ScalarKind::UnsignedInt(_)
                                )
                            );
                            if !is_int_scalar {
                                return Err(ValidationError::TensorQuerySizeResultNotInt {
                                    instruction_id: inst.result_id.map(to_id),
                                }.into());
                            }
                        }
                    }

                    // Get tensor type for validation
                    let tensor_type = inst
                        .operands
                        .first()
                        .and_then(|op| match op {
                            Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                            _ => None,
                        })
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|tensor_inst| tensor_inst.result_type);

                    // Tensor must be a ranked tensor
                    if let Some(tensor_type) = tensor_type {
                        if !is_ranked_tensor_arm(tensor_type, ctx) {
                            return Err(ValidationError::TensorNotRankedTensor {
                                instruction_id: inst.result_id.map(to_id),
                            }.into());
                        }
                    }

                    // Validate Dimension operand (operand index 1)
                    // OpTensorQuerySizeARM: Tensor (operand 0), Dimension (operand 1)
                    if let Some(Operand::IdRef(dim_id)) = inst.operands.get(1) {
                        // Dimension must come from a constant instruction of scalar integer type
                        let dim_result_id = ResultId::try_from(*dim_id).ok();
                        let dim_inst = dim_result_id.and_then(|rid| ctx.definitions.get(&rid));

                        let is_constant = dim_inst
                            .map(|di| is_constant_opcode(di.class.opcode))
                            .unwrap_or(false);
                        let is_int_scalar = is_int_scalar_type_id(*dim_id, ctx);

                        if !is_constant || !is_int_scalar {
                            return Err(ValidationError::TensorQuerySizeDimensionNotConstant {
                                instruction_id: inst.result_id.map(to_id),
                            }.into());
                        }

                        // If we can evaluate both dimension and tensor rank, check dimension < rank
                        if let Some(tensor_type) = tensor_type {
                            if let Some(rank_id) = get_tensor_arm_rank_id(tensor_type, ctx) {
                                if let (Some(dim_val), Some(rank_val)) = (
                                    eval_constant_u64(*dim_id, ctx),
                                    eval_constant_u64(rank_id, ctx),
                                ) {
                                    if dim_val >= rank_val {
                                        return Err(
                                            ValidationError::TensorQuerySizeDimensionOutOfRange {
                                                instruction_id: inst.result_id.map(to_id),
                                                dimension: dim_val,
                                                tensor_rank: rank_val,
                                            }.into(),
                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates NVIDIA tensor layout creation operations.
pub struct CreateTensorLayoutNVRule;

impl ValidationRule for CreateTensorLayoutNVRule {
    fn name(&self) -> &'static str {
        "create-tensor-layout-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode != Op::CreateTensorLayoutNV {
                        continue;
                    }

                    // Result Type must be OpTypeTensorLayoutNV
                    if let Some(result_type) = inst.result_type {
                        if !is_tensor_layout_nv(result_type, ctx) {
                            return Err(ValidationError::TensorLayoutResultNotTensorLayout {
                                instruction_id: inst.result_id.map(to_id),
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates NVIDIA tensor view creation operations.
pub struct CreateTensorViewNVRule;

impl ValidationRule for CreateTensorViewNVRule {
    fn name(&self) -> &'static str {
        "create-tensor-view-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode != Op::CreateTensorViewNV {
                        continue;
                    }

                    // Result Type must be OpTypeTensorViewNV
                    if let Some(result_type) = inst.result_type {
                        if !is_tensor_view_nv(result_type, ctx) {
                            return Err(ValidationError::TensorViewResultNotTensorView {
                                instruction_id: inst.result_id.map(to_id),
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Returns all tensor validation rules.
pub fn all_tensor_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(TensorReadARMRule),
        Box::new(TensorWriteARMRule),
        Box::new(TensorQuerySizeARMRule),
        Box::new(CreateTensorLayoutNVRule),
        Box::new(CreateTensorViewNVRule),
    ]
}
