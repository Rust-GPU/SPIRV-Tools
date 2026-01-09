//! Barrier instruction validation rules.
//!
//! This module validates SPIR-V barrier instructions:
//!
//! - OpControlBarrier: Synchronizes execution within a group
//! - OpMemoryBarrier: Orders memory operations
//! - OpNamedBarrierInitialize: Creates a named barrier
//! - OpMemoryNamedBarrier: Named barrier memory synchronization
//!
//! Barrier instructions have specific execution model requirements,
//! scope constraints, and memory semantic rules.

use rspirv::dr::Operand;
use rspirv::spirv::{ExecutionModel, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId};
use crate::version::SpirvVersion;

// ============================================================================
// Barrier Execution Model Requirements
// ============================================================================

/// Valid execution models for OpControlBarrier in SPIR-V 1.2 and earlier.
const CONTROL_BARRIER_MODELS_V12: &[ExecutionModel] = &[
    ExecutionModel::TessellationControl,
    ExecutionModel::GLCompute,
    ExecutionModel::Kernel,
    ExecutionModel::TaskNV,
    ExecutionModel::MeshNV,
];

// ============================================================================
// Named Barrier Type Rule
// ============================================================================

/// Validates OpNamedBarrierInitialize instructions.
///
/// Ensures that:
/// - Result type is OpTypeNamedBarrier
/// - Subgroup count is a 32-bit integer
pub struct NamedBarrierInitializeRule;

impl ValidationRule for NamedBarrierInitializeRule {
    fn name(&self) -> &'static str {
        "named-barrier-initialize"
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
                    if inst.class.opcode != Op::NamedBarrierInitialize {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result type must be OpTypeNamedBarrier
                    let type_opcode = ResultId::try_from(result_type_id)
                        .ok()
                        .and_then(|rid| ctx.opcodes.get(&rid))
                        .copied();

                    if type_opcode != Some(Op::TypeNamedBarrier) {
                        if let (Some(func), Some(block), Ok(result_type)) = (
                            function_id,
                            block_id,
                            TypeId::try_from(result_type_id),
                        ) {
                            return Err(ValidationError::BarrierResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "OpTypeNamedBarrier",
                            });
                        }
                    }

                    // Subgroup count (first operand) must be 32-bit int
                    let subgroup_type_id = inst
                        .operands
                        .first()
                        .and_then(|op| match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        })
                        .and_then(|id| {
                            ResultId::try_from(id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))
                        })
                        .and_then(|inst| inst.result_type);

                    if let Some(type_id) = subgroup_type_id {
                        let is_int_scalar = resolver.is_int_scalar(type_id, ctx.definitions);
                        let width = resolver.get_bit_width(type_id, ctx.definitions);

                        if !is_int_scalar || width != Some(32) {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                return Err(ValidationError::BarrierOperandTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: 0,
                                    expected: "32-bit integer",
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
// Memory Named Barrier Type Rule
// ============================================================================

/// Validates OpMemoryNamedBarrier instructions.
///
/// Ensures that:
/// - Named barrier operand is of type OpTypeNamedBarrier
pub struct MemoryNamedBarrierRule;

impl ValidationRule for MemoryNamedBarrierRule {
    fn name(&self) -> &'static str {
        "memory-named-barrier"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                    if inst.class.opcode != Op::MemoryNamedBarrier {
                        continue;
                    }

                    // Named barrier operand (first operand) must be of type OpTypeNamedBarrier
                    let barrier_type_id = inst
                        .operands
                        .first()
                        .and_then(|op| match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        })
                        .and_then(|id| {
                            ResultId::try_from(id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))
                        })
                        .and_then(|inst| inst.result_type);

                    if let Some(type_id) = barrier_type_id {
                        let type_opcode = ResultId::try_from(type_id)
                            .ok()
                            .and_then(|rid| ctx.opcodes.get(&rid))
                            .copied();

                        if type_opcode != Some(Op::TypeNamedBarrier) {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                return Err(ValidationError::BarrierOperandTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: 0,
                                    expected: "OpTypeNamedBarrier",
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
// Control Barrier Execution Model Rule
// ============================================================================

/// Validates OpControlBarrier execution model requirements.
///
/// In SPIR-V 1.2 and earlier, OpControlBarrier is only valid in:
/// - TessellationControl
/// - GLCompute
/// - Kernel
/// - TaskNV
/// - MeshNV
pub struct ControlBarrierExecutionModelRule;

impl ValidationRule for ControlBarrierExecutionModelRule {
    fn name(&self) -> &'static str {
        "control-barrier-execution-model"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Only check for SPIR-V 1.2 and earlier
        let v1_3 = SpirvVersion::new(1, 3);
        if ctx.target_version >= v1_3 {
            return Ok(());
        }

        // Check if any control barrier instructions exist
        let has_control_barrier = ctx.module.functions.iter().any(|f| {
            f.blocks.iter().any(|b| {
                b.instructions
                    .iter()
                    .any(|i| i.class.opcode == Op::ControlBarrier)
            })
        });

        if !has_control_barrier {
            return Ok(());
        }

        // Check if we have at least one valid execution model
        let has_valid_model = ctx
            .entry_models
            .iter()
            .any(|model| CONTROL_BARRIER_MODELS_V12.contains(model));

        if !has_valid_model && !ctx.entry_models.is_empty() {
            // Find a control barrier instruction to report in the error
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
                        if inst.class.opcode == Op::ControlBarrier {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                return Err(ValidationError::BarrierRequiresExecutionModel {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    allowed: CONTROL_BARRIER_MODELS_V12.to_vec(),
                                    spirv_version: ctx.target_version,
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
// All barrier rules
// ============================================================================

/// Returns all barrier validation rules.
pub fn all_barrier_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &NamedBarrierInitializeRule,
        &MemoryNamedBarrierRule,
        &ControlBarrierExecutionModelRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_barrier_execution_models() {
        assert!(CONTROL_BARRIER_MODELS_V12.contains(&ExecutionModel::TessellationControl));
        assert!(CONTROL_BARRIER_MODELS_V12.contains(&ExecutionModel::GLCompute));
        assert!(CONTROL_BARRIER_MODELS_V12.contains(&ExecutionModel::Kernel));
        assert!(CONTROL_BARRIER_MODELS_V12.contains(&ExecutionModel::TaskNV));
        assert!(CONTROL_BARRIER_MODELS_V12.contains(&ExecutionModel::MeshNV));

        assert!(!CONTROL_BARRIER_MODELS_V12.contains(&ExecutionModel::Vertex));
        assert!(!CONTROL_BARRIER_MODELS_V12.contains(&ExecutionModel::Fragment));
    }
}
