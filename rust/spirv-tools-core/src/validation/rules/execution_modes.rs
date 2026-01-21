//! Execution mode validation rules.
//!
//! This module validates SPIR-V execution mode requirements including:
//!
//! - Execution mode target validation
//! - Execution model compatibility
//! - LocalSizeId restrictions

use std::collections::{HashMap, HashSet, VecDeque};

use rspirv::spirv::{Capability, Decoration, ExecutionMode, ExecutionModel, FPFastMathMode, Op};

use crate::target_env::TargetEnv;
use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                }.into());
            }

            let execution_mode = execution_mode_from_operand(mode.operands.get(1));
            if let Some(execution_mode) = execution_mode {
                if execution_mode == ExecutionMode::LocalSizeId
                    && !local_size_id_allowed(ctx.env, ctx.options)
                {
                    return Err(ValidationError::LocalSizeIdNotAllowed { env: ctx.env }.into());
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
                                }.into());
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
                                }.into());
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
                                }.into());
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
                                }.into());
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
// Duplicate Execution Modes Rule
// ============================================================================

/// Returns true if this execution mode can only appear once per entry point.
/// Some execution modes can appear multiple times with different operands.
fn is_per_entry_execution_mode(mode: ExecutionMode) -> bool {
    // These execution modes can be specified multiple times per entry point
    // with different operands.
    !matches!(
        mode,
        ExecutionMode::DenormPreserve
            | ExecutionMode::DenormFlushToZero
            | ExecutionMode::SignedZeroInfNanPreserve
            | ExecutionMode::RoundingModeRTE
            | ExecutionMode::RoundingModeRTZ
            | ExecutionMode::FPFastMathDefault
            | ExecutionMode::RoundingModeRTPINTEL
            | ExecutionMode::RoundingModeRTNINTEL
            | ExecutionMode::FloatingPointModeALTINTEL
            | ExecutionMode::FloatingPointModeIEEEINTEL
    )
}

/// Validates that execution modes are not duplicated incorrectly.
///
/// Most execution modes can only be specified once per entry point.
/// Some (like float control modes) can appear multiple times but only with different operands.
pub struct DuplicateExecutionModesRule;

impl ValidationRule for DuplicateExecutionModesRule {
    fn name(&self) -> &'static str {
        "duplicate-execution-modes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Track seen (mode, entry_point) pairs for per-entry modes
        let mut seen_per_entry: HashSet<(ExecutionMode, u32)> = HashSet::new();
        // Track seen (mode, entry_point, operand) tuples for modes that can repeat with different operands
        let mut seen_per_operand: HashSet<(ExecutionMode, u32, u32)> = HashSet::new();

        for mode_inst in &ctx.module.execution_modes {
            if mode_inst.class.opcode != Op::ExecutionMode
                && mode_inst.class.opcode != Op::ExecutionModeId
            {
                continue;
            }

            let Some(rspirv::dr::Operand::IdRef(entry_point)) = mode_inst.operands.first() else {
                continue;
            };
            let Some(mode) = execution_mode_from_operand(mode_inst.operands.get(1)) else {
                continue;
            };

            if is_per_entry_execution_mode(mode) {
                // This mode can only appear once per entry point
                if !seen_per_entry.insert((mode, *entry_point)) {
                    return Err(ValidationError::DuplicateExecutionModePerEntry {
                        entry_point: *entry_point,
                        execution_mode: mode,
                    }.into());
                }
            } else {
                // This mode can appear multiple times but only with different operands
                // Get the first operand value (these modes all take a single operand)
                let operand = match mode_inst.operands.get(2) {
                    Some(rspirv::dr::Operand::IdRef(id)) => *id,
                    Some(rspirv::dr::Operand::LiteralBit32(val)) => *val,
                    _ => 0,
                };

                if !seen_per_operand.insert((mode, *entry_point, operand)) {
                    return Err(ValidationError::DuplicateExecutionModePerOperand {
                        entry_point: *entry_point,
                        execution_mode: mode,
                        operand,
                    }.into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Float Controls 2 Rule (SPV_KHR_float_controls2)
// ============================================================================

/// Validates that NoContraction and FPFastMathMode::Fast decorations
/// are not used by entry points with FPFastMathDefault execution mode.
///
/// This implements SPV_KHR_float_controls2 validation per the SPIR-V spec:
/// - Instructions decorated with NoContraction cannot be reachable from
///   entry points that specify FPFastMathDefault
/// - Instructions decorated with FPFastMathMode containing Fast cannot be
///   reachable from entry points that specify FPFastMathDefault
pub struct FloatControls2Rule;

impl ValidationRule for FloatControls2Rule {
    fn name(&self) -> &'static str {
        "float-controls2"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;

        // Step 1: Find entry points that have FPFastMathDefault execution mode
        let mut fp_fast_math_default_entry_points: HashSet<u32> = HashSet::new();
        for mode_inst in &module.execution_modes {
            if mode_inst.class.opcode != Op::ExecutionMode
                && mode_inst.class.opcode != Op::ExecutionModeId
            {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(entry_point)) = mode_inst.operands.first() else {
                continue;
            };
            let Some(mode) = execution_mode_from_operand(mode_inst.operands.get(1)) else {
                continue;
            };
            if mode == ExecutionMode::FPFastMathDefault {
                fp_fast_math_default_entry_points.insert(*entry_point);
            }
        }

        // If no entry points have FPFastMathDefault, nothing to validate
        if fp_fast_math_default_entry_points.is_empty() {
            return Ok(());
        }

        // Step 2: Build a call graph - map from function ID to functions it calls
        let mut call_graph: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut function_ids: HashSet<u32> = HashSet::new();

        for func in &module.functions {
            let Some(func_id) = func.def.as_ref().and_then(|d| d.result_id) else {
                continue;
            };
            function_ids.insert(func_id);
            let callees = call_graph.entry(func_id).or_default();

            for block in &func.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode == Op::FunctionCall {
                        if let Some(rspirv::dr::Operand::IdRef(callee)) = inst.operands.first() {
                            callees.insert(*callee);
                        }
                    }
                }
            }
        }

        // Step 3: Build map from function ID to entry points that can reach it
        let mut function_to_entry_points: HashMap<u32, HashSet<u32>> = HashMap::new();

        // For each entry point, do BFS to find all reachable functions
        for entry_point in &fp_fast_math_default_entry_points {
            if !function_ids.contains(entry_point) {
                continue;
            }

            let mut visited: HashSet<u32> = HashSet::new();
            let mut queue: VecDeque<u32> = VecDeque::new();
            queue.push_back(*entry_point);

            while let Some(func_id) = queue.pop_front() {
                if !visited.insert(func_id) {
                    continue;
                }

                function_to_entry_points
                    .entry(func_id)
                    .or_default()
                    .insert(*entry_point);

                if let Some(callees) = call_graph.get(&func_id) {
                    for callee in callees {
                        if !visited.contains(callee) {
                            queue.push_back(*callee);
                        }
                    }
                }
            }
        }

        // Step 4: Find instructions decorated with NoContraction or FPFastMathMode Fast
        // and check if they're in functions reachable from FPFastMathDefault entry points

        // Build map from target ID to its decoration type
        #[derive(Clone, Copy, Debug)]
        enum ProblematicDecoration {
            NoContraction,
            FPFastMathModeFast,
        }

        let mut decorated_ids: HashMap<u32, ProblematicDecoration> = HashMap::new();

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }

            let Some(rspirv::dr::Operand::IdRef(target_id)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };

            if *decoration == Decoration::NoContraction {
                decorated_ids.insert(*target_id, ProblematicDecoration::NoContraction);
            } else if *decoration == Decoration::FPFastMathMode {
                // Check if the Fast bit is set
                if let Some(rspirv::dr::Operand::FPFastMathMode(mode)) = inst.operands.get(2) {
                    if mode.contains(FPFastMathMode::FAST) {
                        decorated_ids.insert(*target_id, ProblematicDecoration::FPFastMathModeFast);
                    }
                }
            }
        }

        // If no problematic decorations, nothing to validate
        if decorated_ids.is_empty() {
            return Ok(());
        }

        // Step 5: For each function, check if any of its instructions are decorated
        // and the function is reachable from an FPFastMathDefault entry point
        for func in &module.functions {
            let Some(func_id) = func.def.as_ref().and_then(|d| d.result_id) else {
                continue;
            };

            // Check if this function is reachable from any FPFastMathDefault entry point
            let Some(reachable_entry_points) = function_to_entry_points.get(&func_id) else {
                continue;
            };

            // Check all instructions in this function
            for block in &func.blocks {
                for inst in &block.instructions {
                    if let Some(result_id) = inst.result_id {
                        if let Some(decoration) = decorated_ids.get(&result_id) {
                            // This instruction has a problematic decoration and is
                            // reachable from an FPFastMathDefault entry point
                            let decoration_name = match decoration {
                                ProblematicDecoration::NoContraction => "NoContraction",
                                ProblematicDecoration::FPFastMathModeFast => "FPFastMathMode Fast",
                            };
                            return Err(
                                ValidationError::DecorationConflictsWithFPFastMathDefault {
                                    result_id,
                                    decoration: decoration_name.to_string(),
                                    entry_points: reachable_entry_points.iter().copied().collect(),
                                }.into(),
                        );
                        }
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
    vec![
        &ExecutionModesRule,
        &DuplicateExecutionModesRule,
        &FloatControls2Rule,
    ]
}
