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
use rspirv::spirv::{ExecutionMode, ExecutionModel};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::op_ext::OpExt;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId};

/// Valid execution models for derivative instructions.
const VALID_DERIVATIVE_EXECUTION_MODELS: &[ExecutionModel] = &[
    ExecutionModel::Fragment,
    ExecutionModel::GLCompute,
    ExecutionModel::MeshEXT,
    ExecutionModel::TaskEXT,
];

/// Execution models that require derivative execution modes.
const MODELS_REQUIRING_DERIVATIVE_MODE: &[ExecutionModel] = &[
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
                    if !inst.class.opcode.is_derivative() {
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
                .any(|b| b.instructions.iter().any(|i| i.class.opcode.is_derivative()))
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
                        if inst.class.opcode.is_derivative() {
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
// Derivative Execution Mode Rule
// ============================================================================

/// Validates that derivative instructions in GLCompute/MeshEXT/TaskEXT have
/// the required execution mode.
///
/// For GLCompute, MeshEXT, and TaskEXT execution models, derivative instructions
/// require either `DerivativeGroupQuadsKHR` or `DerivativeGroupLinearKHR`
/// execution mode to be declared.
pub struct DerivativeExecutionModeRule;

impl ValidationRule for DerivativeExecutionModeRule {
    fn name(&self) -> &'static str {
        "derivative-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Check if we have any execution models that require derivative mode
        let requires_mode_models: Vec<ExecutionModel> = ctx
            .entry_models
            .iter()
            .filter(|model| MODELS_REQUIRING_DERIVATIVE_MODE.contains(model))
            .copied()
            .collect();

        if requires_mode_models.is_empty() {
            // No execution models that require derivative mode
            return Ok(());
        }

        // Check if any derivative instructions exist
        let has_derivatives = ctx.module.functions.iter().any(|f| {
            f.blocks
                .iter()
                .any(|b| b.instructions.iter().any(|i| i.class.opcode.is_derivative()))
        });

        if !has_derivatives {
            return Ok(());
        }

        // Check if the required execution mode is present
        let has_derivative_mode = ctx.module.execution_modes.iter().any(|mode_inst| {
            mode_inst.operands.get(1).map_or(false, |operand| {
                matches!(
                    operand,
                    Operand::ExecutionMode(ExecutionMode::DerivativeGroupQuadsKHR)
                        | Operand::ExecutionMode(ExecutionMode::DerivativeGroupLinearKHR)
                )
            })
        });

        if !has_derivative_mode {
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
                        if inst.class.opcode.is_derivative() {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                // Report error for the first model that requires the mode
                                return Err(ValidationError::DerivativeRequiresExecutionMode {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    execution_model: requires_mode_models[0],
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
    vec![
        &DerivativeTypeRule,
        &DerivativeExecutionModelRule,
        &DerivativeExecutionModeRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::spirv::Op;

    #[test]
    fn test_is_derivative_op() {
        assert!(Op::DPdx.is_derivative());
        assert!(Op::DPdy.is_derivative());
        assert!(Op::Fwidth.is_derivative());
        assert!(Op::DPdxFine.is_derivative());
        assert!(Op::DPdyFine.is_derivative());
        assert!(Op::FwidthFine.is_derivative());
        assert!(Op::DPdxCoarse.is_derivative());
        assert!(Op::DPdyCoarse.is_derivative());
        assert!(Op::FwidthCoarse.is_derivative());

        assert!(!Op::FAdd.is_derivative());
        assert!(!Op::FMul.is_derivative());
        assert!(!Op::Nop.is_derivative());
    }

}
