//! Derivative instruction validation rules.
//!
//! This module validates SPIR-V derivative instructions:
//!
//! - OpDPdx, OpDPdy, OpFwidth
//! - OpDPdxFine, OpDPdyFine, OpFwidthFine
//! - OpDPdxCoarse, OpDPdyCoarse, OpFwidthCoarse
//!
//! These instructions compute the rate of change of values with respect to
//! screen coordinates and have specific requirements:
//!
//! - Result type must be 32-bit float scalar or vector
//! - Input P must have the same type as result
//! - Must be used in Fragment, GLCompute, MeshEXT, or TaskEXT execution models
//! - GLCompute/MeshEXT/TaskEXT require DerivativeGroupQuadsKHR or DerivativeGroupLinearKHR

use rspirv::dr::Operand;
use rspirv::spirv::{ExecutionModel, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId};

// ============================================================================
// Derivative Opcodes
// ============================================================================

/// All derivative instruction opcodes.
const DERIVATIVE_OPS: &[Op] = &[
    Op::DPdx,
    Op::DPdy,
    Op::Fwidth,
    Op::DPdxFine,
    Op::DPdyFine,
    Op::FwidthFine,
    Op::DPdxCoarse,
    Op::DPdyCoarse,
    Op::FwidthCoarse,
];

/// Returns true if the opcode is a derivative instruction.
fn is_derivative_op(op: Op) -> bool {
    DERIVATIVE_OPS.contains(&op)
}

/// Valid execution models for derivative instructions.
const VALID_DERIVATIVE_EXECUTION_MODELS: &[ExecutionModel] = &[
    ExecutionModel::Fragment,
    ExecutionModel::GLCompute,
    ExecutionModel::MeshEXT,
    ExecutionModel::TaskEXT,
];

// ============================================================================
// Derivative Type Rule
// ============================================================================

/// Validates that derivative instructions have correct types.
///
/// Ensures that:
/// - Result type is a 32-bit float scalar or vector
/// - Input P has the same type as the result
pub struct DerivativeTypeRule;

impl ValidationRule for DerivativeTypeRule {
    fn name(&self) -> &'static str {
        "derivative-type"
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
                    if !is_derivative_op(inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result type must be float scalar or vector
                    if !resolver.is_float_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Ok(result_type)) = (
                            function_id,
                            block_id,
                            TypeId::try_from(result_type_id),
                        ) {
                            return Err(ValidationError::DerivativeResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "float scalar or vector",
                            });
                        }
                    }

                    // Result type component width must be 32 bits
                    let width = resolver.get_bit_width(result_type_id, ctx.definitions);
                    if width != Some(32) {
                        if let (Some(func), Some(block), Ok(result_type)) = (
                            function_id,
                            block_id,
                            TypeId::try_from(result_type_id),
                        ) {
                            return Err(ValidationError::DerivativeResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "32-bit float",
                            });
                        }
                    }

                    // Get P operand type (first operand after result)
                    let p_type = inst
                        .operands
                        .first()
                        .and_then(|op| match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        })
                        .and_then(|p_id| {
                            ResultId::try_from(p_id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))
                        })
                        .and_then(|p_inst| p_inst.result_type);

                    // P type must match result type
                    if let Some(p_type_id) = p_type {
                        if p_type_id != result_type_id {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::DerivativeOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
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
// Derivative Execution Model Rule
// ============================================================================

/// Validates that derivative instructions are used in valid execution models.
///
/// Derivative instructions are only valid in:
/// - Fragment shader
/// - GLCompute (with derivative execution mode)
/// - MeshEXT (with derivative execution mode)
/// - TaskEXT (with derivative execution mode)
pub struct DerivativeExecutionModelRule;

impl ValidationRule for DerivativeExecutionModelRule {
    fn name(&self) -> &'static str {
        "derivative-execution-model"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Check if any derivative instructions exist
        let has_derivatives = ctx.module.functions.iter().any(|f| {
            f.blocks
                .iter()
                .any(|b| b.instructions.iter().any(|i| is_derivative_op(i.class.opcode)))
        });

        if !has_derivatives {
            return Ok(());
        }

        // Check if we have at least one valid execution model
        let has_valid_model = ctx.entry_models.iter().any(|model| {
            VALID_DERIVATIVE_EXECUTION_MODELS.contains(model)
        });

        if !has_valid_model && !ctx.entry_models.is_empty() {
            // Find a derivative instruction to report in the error
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
                        if is_derivative_op(inst.class.opcode) {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                return Err(ValidationError::DerivativeRequiresExecutionModel {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    allowed: VALID_DERIVATIVE_EXECUTION_MODELS.to_vec(),
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
// All derivative rules
// ============================================================================

/// Returns all derivative validation rules.
pub fn all_derivative_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&DerivativeTypeRule, &DerivativeExecutionModelRule]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_derivative_op() {
        assert!(is_derivative_op(Op::DPdx));
        assert!(is_derivative_op(Op::DPdy));
        assert!(is_derivative_op(Op::Fwidth));
        assert!(is_derivative_op(Op::DPdxFine));
        assert!(is_derivative_op(Op::DPdyFine));
        assert!(is_derivative_op(Op::FwidthFine));
        assert!(is_derivative_op(Op::DPdxCoarse));
        assert!(is_derivative_op(Op::DPdyCoarse));
        assert!(is_derivative_op(Op::FwidthCoarse));

        assert!(!is_derivative_op(Op::FAdd));
        assert!(!is_derivative_op(Op::FMul));
        assert!(!is_derivative_op(Op::Nop));
    }

    #[test]
    fn test_valid_execution_models() {
        assert!(VALID_DERIVATIVE_EXECUTION_MODELS.contains(&ExecutionModel::Fragment));
        assert!(VALID_DERIVATIVE_EXECUTION_MODELS.contains(&ExecutionModel::GLCompute));
        assert!(VALID_DERIVATIVE_EXECUTION_MODELS.contains(&ExecutionModel::MeshEXT));
        assert!(VALID_DERIVATIVE_EXECUTION_MODELS.contains(&ExecutionModel::TaskEXT));

        assert!(!VALID_DERIVATIVE_EXECUTION_MODELS.contains(&ExecutionModel::Vertex));
        assert!(!VALID_DERIVATIVE_EXECUTION_MODELS.contains(&ExecutionModel::Geometry));
        assert!(!VALID_DERIVATIVE_EXECUTION_MODELS.contains(&ExecutionModel::TessellationControl));
    }
}
