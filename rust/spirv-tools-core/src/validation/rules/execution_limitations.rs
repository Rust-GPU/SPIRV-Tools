//! Execution limitation validation rules.
//!
//! This module validates that functions in a SPIR-V module are compatible with
//! their entry point's execution model. SPIR-V has restrictions on which
//! instructions can be used in functions called from specific execution models.
//!
//! For example:
//! - Derivative instructions are only valid in fragment shaders
//! - Workgroup memory instructions have compute shader restrictions
//! - Certain control barrier semantics are only valid in specific stages

use std::collections::{HashMap, HashSet};

use rspirv::dr::Operand;
use rspirv::spirv::{ExecutionModel, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::Id;
use crate::validation::ValidationResult;

/// Helper to convert a raw u32 ID to our Id wrapper type.
fn to_id(raw: u32) -> Option<Id> {
    Id::try_from(raw).ok()
}

/// Validates that functions are compatible with their entry point execution models.
///
/// This rule checks that instructions used in functions called from entry points
/// are allowed for the entry point's execution model. It builds a callgraph from
/// entry points and validates each function in the transitive closure.
pub struct ExecutionLimitationsRule;

/// Execution model limitations that can be registered per-function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionModelLimitation {
    /// Function contains derivative instructions (dFdx, dFdy, etc.)
    /// Only valid in Fragment execution model.
    DerivativeInstructions,
    /// Function contains workgroup memory operations.
    /// Only valid in compute-like models (GLCompute, Kernel, etc.)
    WorkgroupMemory,
    /// Function contains subgroup operations requiring specific execution models.
    SubgroupOperations,
    /// Function uses InputAttachment storage class.
    /// Only valid in Fragment execution model.
    InputAttachment,
    /// Function contains geometry-specific instructions.
    /// Only valid in Geometry execution model.
    GeometryInstructions,
    /// Function contains tessellation-specific instructions.
    /// Only valid in tessellation execution models.
    TessellationInstructions,
    /// Function contains mesh shading specific instructions.
    /// Only valid in mesh/task execution models.
    MeshShadingInstructions,
    /// Function contains ray tracing specific instructions.
    /// Only valid in ray tracing execution models.
    RayTracingInstructions,
}

impl ExecutionModelLimitation {
    /// Check if this limitation is compatible with the given execution model.
    fn is_compatible_with(&self, model: ExecutionModel) -> bool {
        match self {
            Self::DerivativeInstructions => {
                matches!(model, ExecutionModel::Fragment)
            }
            Self::WorkgroupMemory => {
                matches!(
                    model,
                    ExecutionModel::GLCompute
                        | ExecutionModel::Kernel
                        | ExecutionModel::TaskNV
                        | ExecutionModel::TaskEXT
                        | ExecutionModel::MeshNV
                        | ExecutionModel::MeshEXT
                )
            }
            Self::SubgroupOperations => {
                // Subgroup operations are generally allowed in most shader stages
                // but have some restrictions in vertex shaders without extensions
                true
            }
            Self::InputAttachment => {
                matches!(model, ExecutionModel::Fragment)
            }
            Self::GeometryInstructions => {
                matches!(model, ExecutionModel::Geometry)
            }
            Self::TessellationInstructions => {
                matches!(
                    model,
                    ExecutionModel::TessellationControl | ExecutionModel::TessellationEvaluation
                )
            }
            Self::MeshShadingInstructions => {
                matches!(
                    model,
                    ExecutionModel::TaskNV
                        | ExecutionModel::TaskEXT
                        | ExecutionModel::MeshNV
                        | ExecutionModel::MeshEXT
                )
            }
            Self::RayTracingInstructions => {
                matches!(
                    model,
                    ExecutionModel::RayGenerationKHR
                        | ExecutionModel::IntersectionKHR
                        | ExecutionModel::AnyHitKHR
                        | ExecutionModel::ClosestHitKHR
                        | ExecutionModel::MissKHR
                        | ExecutionModel::CallableKHR
                )
            }
        }
    }

    /// Get a human-readable description of this limitation.
    fn description(&self) -> &'static str {
        match self {
            Self::DerivativeInstructions => {
                "derivative instructions (dFdx/dFdy/Fwidth) require Fragment execution model"
            }
            Self::WorkgroupMemory => {
                "workgroup memory requires compute-like execution model (GLCompute, Kernel, etc.)"
            }
            Self::SubgroupOperations => "subgroup operations have execution model restrictions",
            Self::InputAttachment => {
                "InputAttachment storage class requires Fragment execution model"
            }
            Self::GeometryInstructions => "geometry instructions require Geometry execution model",
            Self::TessellationInstructions => {
                "tessellation instructions require TessellationControl or TessellationEvaluation"
            }
            Self::MeshShadingInstructions => {
                "mesh shading instructions require Task or Mesh execution model"
            }
            Self::RayTracingInstructions => {
                "ray tracing instructions require ray tracing execution model"
            }
        }
    }
}

/// Collects execution model limitations for a function based on the instructions it contains.
fn collect_function_limitations(
    _ctx: &ValidationContext<'_>,
    func: &rspirv::dr::Function,
) -> Vec<ExecutionModelLimitation> {
    let mut limitations = Vec::new();
    let mut seen_limitations: HashSet<ExecutionModelLimitation> = HashSet::new();

    for block in &func.blocks {
        for inst in &block.instructions {
            let limitation = match inst.class.opcode {
                // Derivative instructions - Fragment only
                Op::DPdx
                | Op::DPdy
                | Op::Fwidth
                | Op::DPdxFine
                | Op::DPdyFine
                | Op::FwidthFine
                | Op::DPdxCoarse
                | Op::DPdyCoarse
                | Op::FwidthCoarse => Some(ExecutionModelLimitation::DerivativeInstructions),

                // Geometry instructions
                Op::EmitVertex
                | Op::EndPrimitive
                | Op::EmitStreamVertex
                | Op::EndStreamPrimitive => Some(ExecutionModelLimitation::GeometryInstructions),

                // Mesh shading instructions
                Op::SetMeshOutputsEXT
                | Op::EmitMeshTasksEXT
                | Op::WritePackedPrimitiveIndices4x8NV => {
                    Some(ExecutionModelLimitation::MeshShadingInstructions)
                }

                // Ray tracing instructions
                Op::TraceRayKHR
                | Op::ExecuteCallableKHR
                | Op::ReportIntersectionKHR
                | Op::IgnoreIntersectionKHR
                | Op::TerminateRayKHR
                | Op::TraceRayMotionNV => Some(ExecutionModelLimitation::RayTracingInstructions),

                // Variable declarations can have storage class limitations
                Op::Variable => {
                    if let Some(Operand::StorageClass(sc)) = inst.operands.first() {
                        match sc {
                            rspirv::spirv::StorageClass::Workgroup => {
                                Some(ExecutionModelLimitation::WorkgroupMemory)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                }

                _ => None,
            };

            if let Some(lim) = limitation {
                if !seen_limitations.contains(&lim) {
                    limitations.push(lim);
                    seen_limitations.insert(lim);
                }
            }
        }
    }

    limitations
}

/// Build a callgraph mapping function IDs to the set of functions they call.
fn build_callgraph(ctx: &ValidationContext<'_>) -> HashMap<Id, HashSet<Id>> {
    let mut callgraph: HashMap<Id, HashSet<Id>> = HashMap::new();

    for func in &ctx.module.functions {
        let func_id = func.def.as_ref().and_then(|d| d.result_id).and_then(to_id);

        let Some(func_id) = func_id else {
            continue;
        };

        let mut callees: HashSet<Id> = HashSet::new();

        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == Op::FunctionCall {
                    if let Some(Operand::IdRef(callee_id)) = inst.operands.first() {
                        if let Some(id) = to_id(*callee_id) {
                            callees.insert(id);
                        }
                    }
                }
            }
        }

        callgraph.insert(func_id, callees);
    }

    callgraph
}

/// Compute the transitive closure of functions reachable from an entry point.
fn get_reachable_functions(entry_func: Id, callgraph: &HashMap<Id, HashSet<Id>>) -> HashSet<Id> {
    let mut reachable = HashSet::new();
    let mut worklist = vec![entry_func];

    while let Some(func_id) = worklist.pop() {
        if !reachable.insert(func_id) {
            continue;
        }

        if let Some(callees) = callgraph.get(&func_id) {
            for callee in callees {
                if !reachable.contains(callee) {
                    worklist.push(*callee);
                }
            }
        }
    }

    reachable
}

impl ValidationRule for ExecutionLimitationsRule {
    fn name(&self) -> &'static str {
        "execution-limitations"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Build callgraph
        let callgraph = build_callgraph(ctx);

        // Collect limitations for each function
        let mut function_limitations: HashMap<Id, Vec<ExecutionModelLimitation>> = HashMap::new();

        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).and_then(to_id);

            if let Some(func_id) = func_id {
                let limitations = collect_function_limitations(ctx, func);
                if !limitations.is_empty() {
                    function_limitations.insert(func_id, limitations);
                }
            }
        }

        // For each entry point, check all reachable functions
        for entry_point in &ctx.module.entry_points {
            // Get entry point execution model
            let execution_model = match entry_point.operands.first() {
                Some(Operand::ExecutionModel(model)) => *model,
                _ => continue,
            };

            // Get entry point function ID
            let entry_func_id = match entry_point.operands.get(1) {
                Some(Operand::IdRef(id)) => match to_id(*id) {
                    Some(id) => id,
                    None => continue,
                },
                _ => continue,
            };

            // Get all functions reachable from this entry point
            let reachable = get_reachable_functions(entry_func_id, &callgraph);

            // Check each reachable function for incompatible limitations
            for func_id in &reachable {
                if let Some(limitations) = function_limitations.get(func_id) {
                    for limitation in limitations {
                        if !limitation.is_compatible_with(execution_model) {
                            return Err(ValidationError::ExecutionModelIncompatible {
                                entry_point: entry_func_id,
                                function: *func_id,
                                execution_model,
                                reason: limitation.description().to_string(),
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Static rule instance.
static EXECUTION_LIMITATIONS_RULE: ExecutionLimitationsRule = ExecutionLimitationsRule;

/// Returns all execution limitations validation rules.
pub fn all_execution_limitations_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&EXECUTION_LIMITATIONS_RULE]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_limitation_compatibility() {
        let lim = ExecutionModelLimitation::DerivativeInstructions;
        assert!(lim.is_compatible_with(ExecutionModel::Fragment));
        assert!(!lim.is_compatible_with(ExecutionModel::Vertex));
        assert!(!lim.is_compatible_with(ExecutionModel::GLCompute));
    }

    #[test]
    fn test_workgroup_limitation_compatibility() {
        let lim = ExecutionModelLimitation::WorkgroupMemory;
        assert!(lim.is_compatible_with(ExecutionModel::GLCompute));
        assert!(lim.is_compatible_with(ExecutionModel::Kernel));
        assert!(!lim.is_compatible_with(ExecutionModel::Fragment));
        assert!(!lim.is_compatible_with(ExecutionModel::Vertex));
    }

    #[test]
    fn test_geometry_limitation_compatibility() {
        let lim = ExecutionModelLimitation::GeometryInstructions;
        assert!(lim.is_compatible_with(ExecutionModel::Geometry));
        assert!(!lim.is_compatible_with(ExecutionModel::Vertex));
        assert!(!lim.is_compatible_with(ExecutionModel::Fragment));
    }

    #[test]
    fn test_ray_tracing_limitation_compatibility() {
        let lim = ExecutionModelLimitation::RayTracingInstructions;
        assert!(lim.is_compatible_with(ExecutionModel::RayGenerationKHR));
        assert!(lim.is_compatible_with(ExecutionModel::ClosestHitKHR));
        assert!(lim.is_compatible_with(ExecutionModel::MissKHR));
        assert!(!lim.is_compatible_with(ExecutionModel::Fragment));
        assert!(!lim.is_compatible_with(ExecutionModel::GLCompute));
    }
}
