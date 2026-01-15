//! Tensor layout and view validation rules (NVIDIA extension).
//!
//! This module validates SPIR-V tensor layout and view instructions:
//!
//! - OpCreateTensorLayoutNV requires OpTypeTensorLayoutNV result type
//! - OpCreateTensorViewNV requires OpTypeTensorViewNV result type
//! - Tensor layout operations validate operand counts based on dimension
//! - All value operands must be 32-bit integers

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId};

fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Expected number of value operands for tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedValueCount {
    /// Number equals dimension of tensor type
    Dim,
    /// Number equals dimension * 2 (for slice operations)
    DimX2,
    /// Exactly one value
    One,
    /// Exactly four values
    Four,
}

/// Check if a type is a 32-bit integer scalar.
fn is_int32_scalar(
    type_id: u32,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(inst) = definitions.get(&result_id) {
            if inst.class.opcode == Op::TypeInt {
                if let Some(Operand::LiteralBit32(width)) = inst.operands.first() {
                    return *width == 32;
                }
            }
        }
    }
    false
}

/// Get the dimension value from a tensor layout/view type.
fn get_tensor_dimension(
    type_id: u32,
    ctx: &ValidationContext<'_>,
) -> Option<u32> {
    let result_id = ResultId::try_from(type_id).ok()?;
    let type_inst = ctx.definitions.get(&result_id)?;

    // OpTypeTensorLayoutNV and OpTypeTensorViewNV have Dim as first operand
    let dim_id = match type_inst.operands.first() {
        Some(Operand::IdRef(id)) => *id,
        _ => return None,
    };

    // Look up the constant value for dimension
    let dim_result_id = ResultId::try_from(dim_id).ok()?;
    let dim_inst = ctx.definitions.get(&dim_result_id)?;

    // Must be OpConstant
    if dim_inst.class.opcode != Op::Constant {
        return None;
    }

    // Get the literal value
    match dim_inst.operands.first() {
        Some(Operand::LiteralBit32(val)) => Some(*val),
        _ => None,
    }
}

/// Validates OpCreateTensorLayoutNV result type.
pub struct CreateTensorLayoutRule;

impl ValidationRule for CreateTensorLayoutRule {
    fn name(&self) -> &'static str {
        "create-tensor-layout"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id);

            for block in &func.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::CreateTensorLayoutNV {
                        continue;
                    }

                    // Result type must be OpTypeTensorLayoutNV
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_result_id) = ResultId::try_from(result_type_id) {
                            if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                                if type_inst.class.opcode != Op::TypeTensorLayoutNV {
                                    return Err(ValidationError::TensorLayoutInvalidResultType {
                                        function: func_id,
                                        block: block_id,
                                        opcode: inst.class.opcode,
                                        expected: "OpTypeTensorLayoutNV",
                                    });
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

/// Validates OpCreateTensorViewNV result type.
pub struct CreateTensorViewRule;

impl ValidationRule for CreateTensorViewRule {
    fn name(&self) -> &'static str {
        "create-tensor-view"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id);

            for block in &func.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::CreateTensorViewNV {
                        continue;
                    }

                    // Result type must be OpTypeTensorViewNV
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_result_id) = ResultId::try_from(result_type_id) {
                            if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                                if type_inst.class.opcode != Op::TypeTensorViewNV {
                                    return Err(ValidationError::TensorViewInvalidResultType {
                                        function: func_id,
                                        block: block_id,
                                        opcode: inst.class.opcode,
                                        expected: "OpTypeTensorViewNV",
                                    });
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

/// Validates tensor layout operations with dimension-based operand counts.
pub struct TensorLayoutOperandsRule;

impl TensorLayoutOperandsRule {
    /// Returns (expected_count, is_view) for tensor operations that need validation.
    fn get_operand_expectation(opcode: Op) -> Option<(ExpectedValueCount, bool)> {
        match opcode {
            // Layout operations with DIM values
            Op::TensorLayoutSetBlockSizeNV
            | Op::TensorLayoutSetDimensionNV
            | Op::TensorLayoutSetStrideNV => Some((ExpectedValueCount::Dim, false)),

            // Layout slice with DIM*2 values
            Op::TensorLayoutSliceNV => Some((ExpectedValueCount::DimX2, false)),

            // Layout clamp with ONE value
            Op::TensorLayoutSetClampValueNV => Some((ExpectedValueCount::One, false)),

            // View operations with DIM values
            Op::TensorViewSetDimensionNV | Op::TensorViewSetStrideNV => {
                Some((ExpectedValueCount::Dim, true))
            }

            // View clip with FOUR values
            Op::TensorViewSetClipNV => Some((ExpectedValueCount::Four, true)),

            _ => None,
        }
    }
}

impl ValidationRule for TensorLayoutOperandsRule {
    fn name(&self) -> &'static str {
        "tensor-layout-operands"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id);

            for block in &func.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .map(to_id);

                for inst in &block.instructions {
                    let Some((expected, is_view)) =
                        Self::get_operand_expectation(inst.class.opcode)
                    else {
                        continue;
                    };

                    // Validate result type
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_result_id) = ResultId::try_from(result_type_id) {
                            if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                                let expected_type_op = if is_view {
                                    Op::TypeTensorViewNV
                                } else {
                                    Op::TypeTensorLayoutNV
                                };

                                if type_inst.class.opcode != expected_type_op {
                                    if is_view {
                                        return Err(ValidationError::TensorViewInvalidResultType {
                                            function: func_id,
                                            block: block_id,
                                            opcode: inst.class.opcode,
                                            expected: "OpTypeTensorViewNV",
                                        });
                                    } else {
                                        return Err(ValidationError::TensorLayoutInvalidResultType {
                                            function: func_id,
                                            block: block_id,
                                            opcode: inst.class.opcode,
                                            expected: "OpTypeTensorLayoutNV",
                                        });
                                    }
                                }
                            }
                        }
                    }

                    // Validate tensor operand matches result type
                    // Tensor operand is at index 0 (first operand after result type/id)
                    if let Some(Operand::IdRef(tensor_id)) = inst.operands.first() {
                        if let Ok(tensor_result_id) = ResultId::try_from(*tensor_id) {
                            if let Some(tensor_inst) = ctx.definitions.get(&tensor_result_id) {
                                if tensor_inst.result_type != inst.result_type {
                                    return Err(ValidationError::TensorTypeMismatch {
                                        function: func_id,
                                        block: block_id,
                                        opcode: inst.class.opcode,
                                    });
                                }
                            }
                        }
                    }

                    // Count value operands (everything after tensor operand)
                    let num_values = inst.operands.len().saturating_sub(1);

                    // Get dimension if we can evaluate it
                    if let Some(result_type_id) = inst.result_type {
                        if let Some(dim) = get_tensor_dimension(result_type_id, ctx) {
                            let expected_count = match expected {
                                ExpectedValueCount::Dim => dim as usize,
                                ExpectedValueCount::DimX2 => (dim * 2) as usize,
                                ExpectedValueCount::One => 1,
                                ExpectedValueCount::Four => 4,
                            };

                            if num_values != expected_count {
                                return Err(ValidationError::TensorUnexpectedOperandCount {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                    expected: expected_count,
                                    actual: num_values,
                                });
                            }
                        }
                    }

                    // Validate all value operands are 32-bit integers
                    for operand in inst.operands.iter().skip(1) {
                        if let Operand::IdRef(val_id) = operand {
                            if let Ok(val_result_id) = ResultId::try_from(*val_id) {
                                if let Some(val_inst) = ctx.definitions.get(&val_result_id) {
                                    if let Some(val_type_id) = val_inst.result_type {
                                        if !is_int32_scalar(val_type_id, ctx.definitions) {
                                            return Err(ValidationError::TensorOperandNotInt32 {
                                                function: func_id,
                                                block: block_id,
                                                opcode: inst.class.opcode,
                                                operand_id: to_id(*val_id),
                                            });
                                        }
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

/// Returns all tensor layout validation rules.
pub fn all_tensor_layout_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(CreateTensorLayoutRule),
        Box::new(CreateTensorViewRule),
        Box::new(TensorLayoutOperandsRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_tensor_layout_rules() {
        let rules = all_tensor_layout_rules();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[0].name(), "create-tensor-layout");
        assert_eq!(rules[1].name(), "create-tensor-view");
        assert_eq!(rules[2].name(), "tensor-layout-operands");
    }

    #[test]
    fn test_expected_value_count_debug() {
        // Ensure enum derives work
        let dim = ExpectedValueCount::Dim;
        assert_eq!(format!("{:?}", dim), "Dim");
        assert_eq!(dim, ExpectedValueCount::Dim);
        assert_ne!(dim, ExpectedValueCount::One);

        let cloned = dim;
        assert_eq!(cloned, ExpectedValueCount::Dim);
    }

    #[test]
    fn test_get_operand_expectation_layout_ops() {
        // Layout operations with DIM values
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::TensorLayoutSetBlockSizeNV),
            Some((ExpectedValueCount::Dim, false))
        );
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::TensorLayoutSetDimensionNV),
            Some((ExpectedValueCount::Dim, false))
        );
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::TensorLayoutSetStrideNV),
            Some((ExpectedValueCount::Dim, false))
        );

        // Layout slice with DIM*2
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::TensorLayoutSliceNV),
            Some((ExpectedValueCount::DimX2, false))
        );

        // Layout clamp with ONE
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::TensorLayoutSetClampValueNV),
            Some((ExpectedValueCount::One, false))
        );
    }

    #[test]
    fn test_get_operand_expectation_view_ops() {
        // View operations with DIM values
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::TensorViewSetDimensionNV),
            Some((ExpectedValueCount::Dim, true))
        );
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::TensorViewSetStrideNV),
            Some((ExpectedValueCount::Dim, true))
        );

        // View clip with FOUR
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::TensorViewSetClipNV),
            Some((ExpectedValueCount::Four, true))
        );
    }

    #[test]
    fn test_get_operand_expectation_non_tensor_ops() {
        // Non-tensor operations return None
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::Nop),
            None
        );
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::FAdd),
            None
        );
        assert_eq!(
            TensorLayoutOperandsRule::get_operand_expectation(Op::Load),
            None
        );
    }
}
