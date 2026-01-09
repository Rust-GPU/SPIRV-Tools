//! Execution mode validation rules.
//!
//! This module validates SPIR-V execution mode requirements including:
//!
//! - Execution mode target validation
//! - Execution model compatibility
//! - LocalSizeId restrictions

use std::collections::{HashMap, HashSet};

use rspirv::spirv::{Capability, ExecutionMode, ExecutionModel, Op};

use crate::target_env::TargetEnv;
use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{IdKind, ResultId};

// ============================================================================
// Execution Modes Rule
// ============================================================================

/// Validates execution mode requirements.
pub struct ExecutionModesRule;

impl ValidationRule for ExecutionModesRule {
    fn name(&self) -> &'static str {
        "execution-modes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Build entry points set
        let entry_points: HashSet<ResultId> = ctx
            .module
            .entry_points
            .iter()
            .filter_map(|ep| {
                let mut operands = ep.operands.iter();
                if ep.class.opcode == Op::ConditionalEntryPointINTEL {
                    let _ = operands.next();
                }
                let _ = operands.next(); // ExecutionModel
                operands.next().and_then(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                    _ => None,
                })
            })
            .collect();

        let mut entry_point_models: HashMap<ResultId, ExecutionModel> = HashMap::new();
        for ep in &ctx.module.entry_points {
            let mut operands = ep.operands.iter();
            if ep.class.opcode == Op::ConditionalEntryPointINTEL {
                let _ = operands.next();
            }
            let execution_model = operands.next().and_then(|op| match op {
                rspirv::dr::Operand::ExecutionModel(model) => Some(*model),
                _ => None,
            });
            let function = operands.next().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            });
            if let (Some(model), Some(function)) = (execution_model, function) {
                entry_point_models.insert(function, model);
            }
        }

        for mode in &ctx.module.execution_modes {
            let mut operands = mode.operands.iter();
            // First operand is the target entry point function.
            let Some(rspirv::dr::Operand::IdRef(target)) = operands.next() else {
                continue;
            };
            let function = ResultId::try_from(*target).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::Operand,
                opcode: mode.class.opcode,
            })?;
            if !entry_points.contains(&function) {
                return Err(ValidationError::ExecutionModeWithoutEntryPoint {
                    function: function.into_inner(),
                });
            }

            let execution_mode = execution_mode_from_operand(mode.operands.get(1));
            if let Some(execution_mode) = execution_mode {
                if execution_mode == ExecutionMode::LocalSizeId
                    && !local_size_id_allowed(ctx.env, ctx.options)
                {
                    return Err(ValidationError::LocalSizeIdNotAllowed { env: ctx.env });
                }
                if let Some(model) = entry_point_models.get(&function) {
                    match execution_mode {
                        ExecutionMode::OutputVertices => {
                            let allowed = [
                                ExecutionModel::Geometry,
                                ExecutionModel::TessellationControl,
                                ExecutionModel::MeshEXT,
                                ExecutionModel::MeshNV,
                            ];
                            if !allowed.contains(model) {
                                return Err(ValidationError::ExecutionModeRequiresExecutionModel {
                                    entry_point: function.into_inner(),
                                    mode: execution_mode,
                                    execution_model: *model,
                                    allowed_models: allowed.to_vec(),
                                });
                            }
                            if ctx.env.is_vulkan()
                                && ctx
                                    .declared_capabilities
                                    .contains(&Capability::MeshShadingEXT)
                                && matches!(mode.operands.get(2), Some(rspirv::dr::Operand::LiteralBit32(v)) if *v == 0)
                                && (*model == ExecutionModel::MeshEXT
                                    || *model == ExecutionModel::MeshNV)
                            {
                                return Err(ValidationError::InvalidExecutionModeValue {
                                    entry_point: function.into_inner(),
                                    mode: execution_mode,
                                    value: 0,
                                });
                            }
                        }
                        ExecutionMode::OutputLinesEXT
                        | ExecutionMode::OutputTrianglesEXT
                        | ExecutionMode::OutputPrimitivesEXT => {
                            let allowed = [ExecutionModel::MeshEXT, ExecutionModel::MeshNV];
                            if !allowed.contains(model) {
                                return Err(ValidationError::ExecutionModeRequiresExecutionModel {
                                    entry_point: function.into_inner(),
                                    mode: execution_mode,
                                    execution_model: *model,
                                    allowed_models: allowed.to_vec(),
                                });
                            }
                            if ctx.env.is_vulkan()
                                && ctx
                                    .declared_capabilities
                                    .contains(&Capability::MeshShadingEXT)
                                && execution_mode == ExecutionMode::OutputPrimitivesEXT
                                && matches!(mode.operands.get(2), Some(rspirv::dr::Operand::LiteralBit32(v)) if *v == 0)
                            {
                                return Err(ValidationError::InvalidExecutionModeValue {
                                    entry_point: function.into_inner(),
                                    mode: execution_mode,
                                    value: 0,
                                });
                            }
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
// Helper Functions
// ============================================================================

fn execution_mode_from_operand(operand: Option<&rspirv::dr::Operand>) -> Option<ExecutionMode> {
    match operand {
        Some(rspirv::dr::Operand::ExecutionMode(mode)) => Some(*mode),
        Some(rspirv::dr::Operand::LiteralBit32(raw)) => ExecutionMode::from_u32(*raw),
        _ => None,
    }
}

fn local_size_id_allowed(env: TargetEnv, options: &crate::validation::ValidationOptions) -> bool {
    match env {
        TargetEnv::Vulkan1_0
        | TargetEnv::Vulkan1_1
        | TargetEnv::Vulkan1_1Spirv1_4
        | TargetEnv::Vulkan1_2 => options.allow_localsizeid,
        _ => true,
    }
}

// ============================================================================
// All execution mode rules
// ============================================================================

/// Returns all execution mode validation rules.
pub fn all_execution_mode_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&ExecutionModesRule]
}
