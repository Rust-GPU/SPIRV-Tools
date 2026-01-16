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
use crate::validation::error::ValidationError;
use crate::validation::helpers::get_type_structure;
use crate::validation::types::{Id, ResultId, TypeId, TypeStructure};

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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                });
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
                                        });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                        });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                });
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
                                        });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                            });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                            });
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
