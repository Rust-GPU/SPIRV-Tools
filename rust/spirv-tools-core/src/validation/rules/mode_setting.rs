//! Mode setting validation rules.
//!
//! This module validates SPIR-V mode setting requirements including:
//!
//! - Entry point validation (function, return type, parameters)
//! - Execution mode constraints (fragment origin, tessellation, geometry)
//! - Memory model validation
//! - Capability dependencies
//! - Duplicate execution mode detection

use std::collections::{HashMap, HashSet};

use rspirv::dr::Operand;
use rspirv::spirv::{AddressingModel, Capability, ExecutionMode, ExecutionModel, MemoryModel, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::Id;

/// Helper to convert a u32 to Id (with fallback to id 1).
fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Validates entry point requirements.
///
/// - Entry point must be a function
/// - Entry point must return void
/// - Non-Kernel entry points must have zero parameters
pub struct EntryPointValidationRule;

impl ValidationRule for EntryPointValidationRule {
    fn name(&self) -> &'static str {
        "entry-point-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        // Build map of function ID -> function type ID
        let mut function_types: HashMap<u32, u32> = HashMap::new();
        for func in &module.functions {
            if let Some(def) = &func.def {
                if let (Some(func_id), Some(Operand::IdRef(type_id))) =
                    (def.result_id, def.operands.get(1))
                {
                    function_types.insert(func_id, *type_id);
                }
            }
        }

        // Build map of type ID -> type instruction for type functions
        let mut type_functions: HashMap<u32, (Option<u32>, usize)> = HashMap::new();
        for inst in &module.types_global_values {
            if inst.class.opcode == Op::TypeFunction {
                if let Some(type_id) = inst.result_id {
                    // TypeFunction: result_id, return_type, param_types...
                    let return_type = inst.operands.first().and_then(|op| match op {
                        Operand::IdRef(id) => Some(*id),
                        _ => None,
                    });
                    let param_count = inst.operands.len().saturating_sub(1);
                    type_functions.insert(type_id, (return_type, param_count));
                }
            }
        }

        // Build set of void type IDs
        let void_types: HashSet<u32> = module
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == Op::TypeVoid)
            .filter_map(|inst| inst.result_id)
            .collect();

        for ep in &module.entry_points {
            if ep.class.opcode != Op::EntryPoint {
                continue;
            }

            let execution_model = ep.operands.first().and_then(|op| match op {
                Operand::ExecutionModel(model) => Some(*model),
                _ => None,
            });

            let Some(Operand::IdRef(func_id)) = ep.operands.get(1) else {
                continue;
            };

            // Check that entry point is a function
            let Some(func_type_id) = function_types.get(func_id) else {
                return Err(ValidationError::EntryPointNotFunction {
                    entry_point: to_id(*func_id),
                });
            };

            // Get function type info
            let Some((return_type, param_count)) = type_functions.get(func_type_id) else {
                continue;
            };

            // Check return type is void
            if let Some(ret_type) = return_type {
                if !void_types.contains(ret_type) {
                    return Err(ValidationError::EntryPointReturnTypeNotVoid {
                        entry_point: to_id(*func_id),
                    });
                }
            }

            // Non-Kernel entry points must have zero parameters
            if execution_model != Some(ExecutionModel::Kernel) && *param_count > 0 {
                return Err(ValidationError::EntryPointNonZeroParameters {
                    entry_point: to_id(*func_id),
                    param_count: *param_count as u32,
                });
            }
        }

        Ok(())
    }
}

/// Validates fragment shader execution mode requirements.
///
/// - Fragment must have exactly one of OriginUpperLeft/OriginLowerLeft
/// - At most one depth mode (DepthGreater, DepthLess, DepthUnchanged)
/// - At most one interlock mode
pub struct FragmentExecutionModeRule;

impl ValidationRule for FragmentExecutionModeRule {
    fn name(&self) -> &'static str {
        "fragment-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        if !ctx.declared_capabilities.contains(&Capability::Shader) {
            return Ok(());
        }

        let module = ctx.module();

        // Build map of entry point -> execution model
        let mut entry_point_models: HashMap<u32, ExecutionModel> = HashMap::new();
        for ep in &module.entry_points {
            if ep.class.opcode != Op::EntryPoint {
                continue;
            }
            if let (Some(Operand::ExecutionModel(model)), Some(Operand::IdRef(func_id))) =
                (ep.operands.first(), ep.operands.get(1))
            {
                entry_point_models.insert(*func_id, *model);
            }
        }

        // Build map of entry point -> execution modes
        let mut entry_point_modes: HashMap<u32, HashSet<ExecutionMode>> = HashMap::new();
        for mode in &module.execution_modes {
            let Some(Operand::IdRef(func_id)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };
            entry_point_modes
                .entry(*func_id)
                .or_default()
                .insert(*exec_mode);
        }

        // Check fragment entry points
        for (func_id, model) in &entry_point_models {
            if *model != ExecutionModel::Fragment {
                continue;
            }

            let modes = entry_point_modes.get(func_id);

            // Check origin modes
            let has_upper = modes.map_or(false, |m| m.contains(&ExecutionMode::OriginUpperLeft));
            let has_lower = modes.map_or(false, |m| m.contains(&ExecutionMode::OriginLowerLeft));

            if has_upper && has_lower {
                return Err(ValidationError::FragmentMultipleOriginModes {
                    entry_point: to_id(*func_id),
                });
            }

            if !has_upper && !has_lower {
                return Err(ValidationError::FragmentMissingOriginMode {
                    entry_point: to_id(*func_id),
                });
            }

            // Check depth modes
            if let Some(modes) = modes {
                let depth_count = [
                    ExecutionMode::DepthGreater,
                    ExecutionMode::DepthLess,
                    ExecutionMode::DepthUnchanged,
                ]
                .iter()
                .filter(|m| modes.contains(m))
                .count();

                if depth_count > 1 {
                    return Err(ValidationError::FragmentMultipleDepthModes {
                        entry_point: to_id(*func_id),
                    });
                }

                // Check interlock modes
                let interlock_count = [
                    ExecutionMode::PixelInterlockOrderedEXT,
                    ExecutionMode::PixelInterlockUnorderedEXT,
                    ExecutionMode::SampleInterlockOrderedEXT,
                    ExecutionMode::SampleInterlockUnorderedEXT,
                    ExecutionMode::ShadingRateInterlockOrderedEXT,
                    ExecutionMode::ShadingRateInterlockUnorderedEXT,
                ]
                .iter()
                .filter(|m| modes.contains(m))
                .count();

                if interlock_count > 1 {
                    return Err(ValidationError::FragmentMultipleInterlockModes {
                        entry_point: to_id(*func_id),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Validates Vulkan-specific execution mode restrictions.
///
/// - OriginLowerLeft is not allowed in Vulkan
/// - PixelCenterInteger is not allowed in Vulkan
pub struct VulkanExecutionModeRule;

impl ValidationRule for VulkanExecutionModeRule {
    fn name(&self) -> &'static str {
        "vulkan-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        if !ctx.env.is_vulkan() {
            return Ok(());
        }

        let module = ctx.module();

        for mode in &module.execution_modes {
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };

            match exec_mode {
                ExecutionMode::OriginLowerLeft => {
                    return Err(ValidationError::VulkanOriginLowerLeftNotAllowed);
                }
                ExecutionMode::PixelCenterInteger => {
                    return Err(ValidationError::VulkanPixelCenterIntegerNotAllowed);
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Validates memory model requirements.
///
/// - VulkanMemoryModelKHR capability requires VulkanKHR memory model
/// - OpenCL requires Physical32/Physical64 addressing and OpenCL memory model
/// - Vulkan requires Logical or PhysicalStorageBuffer64 addressing
pub struct MemoryModelValidationRule;

impl ValidationRule for MemoryModelValidationRule {
    fn name(&self) -> &'static str {
        "memory-model-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        // Find memory model instruction
        let mut memory_model: Option<MemoryModel> = None;
        let mut addressing_model: Option<AddressingModel> = None;

        if let Some(inst) = &module.memory_model {
            if inst.class.opcode == Op::MemoryModel {
                if let Some(Operand::AddressingModel(addr)) = inst.operands.first() {
                    addressing_model = Some(*addr);
                }
                if let Some(Operand::MemoryModel(mem)) = inst.operands.get(1) {
                    memory_model = Some(*mem);
                }
            }
        }

        // VulkanMemoryModelKHR requires VulkanKHR memory model
        if ctx
            .declared_capabilities
            .contains(&Capability::VulkanMemoryModelKHR)
        {
            if memory_model != Some(MemoryModel::VulkanKHR) {
                return Err(ValidationError::VulkanMemoryModelCapabilityRequiresVulkanKHR);
            }
        }

        // Vulkan environment restrictions
        if ctx.env.is_vulkan() {
            if let Some(addr) = addressing_model {
                if addr != AddressingModel::Logical
                    && addr != AddressingModel::PhysicalStorageBuffer64
                {
                    return Err(ValidationError::VulkanInvalidAddressingModel {
                        addressing_model: addr,
                    });
                }
            }
        }

        // OpenCL environment restrictions
        if ctx.env.is_opencl() {
            if let Some(addr) = addressing_model {
                if addr != AddressingModel::Physical32 && addr != AddressingModel::Physical64 {
                    return Err(ValidationError::OpenCLInvalidAddressingModel {
                        addressing_model: addr,
                    });
                }
            }
            if let Some(mem) = memory_model {
                if mem != MemoryModel::OpenCL {
                    return Err(ValidationError::OpenCLInvalidMemoryModel {
                        memory_model: mem,
                    });
                }
            }
        }

        Ok(())
    }
}

/// Validates capability dependencies.
///
/// - CooperativeMatrixKHR with Shader requires VulkanMemoryModel
pub struct CapabilityDependenciesRule;

impl ValidationRule for CapabilityDependenciesRule {
    fn name(&self) -> &'static str {
        "capability-dependencies"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // CooperativeMatrixKHR + Shader requires VulkanMemoryModel
        if ctx
            .declared_capabilities
            .contains(&Capability::CooperativeMatrixKHR)
            && ctx.declared_capabilities.contains(&Capability::Shader)
            && !ctx
                .declared_capabilities
                .contains(&Capability::VulkanMemoryModel)
        {
            return Err(ValidationError::CooperativeMatrixRequiresVulkanMemoryModel);
        }

        Ok(())
    }
}

/// Validates that execution modes are not duplicated.
///
/// Most execution modes can only appear once per entry point.
pub struct DuplicateExecutionModeRule;

impl ValidationRule for DuplicateExecutionModeRule {
    fn name(&self) -> &'static str {
        "duplicate-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        // Modes that can be specified multiple times per entry point
        let per_operand_modes = [
            ExecutionMode::DenormPreserve,
            ExecutionMode::DenormFlushToZero,
            ExecutionMode::SignedZeroInfNanPreserve,
            ExecutionMode::RoundingModeRTE,
            ExecutionMode::RoundingModeRTZ,
            ExecutionMode::FPFastMathDefault,
            ExecutionMode::RoundingModeRTPINTEL,
            ExecutionMode::RoundingModeRTNINTEL,
            ExecutionMode::FloatingPointModeALTINTEL,
            ExecutionMode::FloatingPointModeIEEEINTEL,
        ];

        // Track seen modes: (entry_point, mode) for per-entry modes
        // (entry_point, mode, operand) for per-operand modes
        let mut seen_per_entry: HashSet<(u32, ExecutionMode)> = HashSet::new();
        let mut seen_per_operand: HashSet<(u32, ExecutionMode, u32)> = HashSet::new();

        for mode in &module.execution_modes {
            let Some(Operand::IdRef(entry_point)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };

            if per_operand_modes.contains(exec_mode) {
                // Per-operand modes - check with operand
                let operand = mode
                    .operands
                    .get(2)
                    .and_then(|op| match op {
                        Operand::IdRef(id) => Some(*id),
                        Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .unwrap_or(0);

                if !seen_per_operand.insert((*entry_point, *exec_mode, operand)) {
                    return Err(ValidationError::DuplicateExecutionMode {
                        entry_point: to_id(*entry_point),
                        mode: *exec_mode,
                    });
                }
            } else {
                // Per-entry modes
                if !seen_per_entry.insert((*entry_point, *exec_mode)) {
                    return Err(ValidationError::DuplicateExecutionMode {
                        entry_point: to_id(*entry_point),
                        mode: *exec_mode,
                    });
                }
            }
        }

        Ok(())
    }
}

/// Returns all mode setting validation rules.
pub fn all_mode_setting_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(EntryPointValidationRule),
        Box::new(FragmentExecutionModeRule),
        Box::new(VulkanExecutionModeRule),
        Box::new(MemoryModelValidationRule),
        Box::new(CapabilityDependenciesRule),
        Box::new(DuplicateExecutionModeRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_mode_setting_rules() {
        let rules = all_mode_setting_rules();
        assert_eq!(rules.len(), 6);
        assert_eq!(rules[0].name(), "entry-point-validation");
        assert_eq!(rules[1].name(), "fragment-execution-mode");
        assert_eq!(rules[2].name(), "vulkan-execution-mode");
        assert_eq!(rules[3].name(), "memory-model-validation");
        assert_eq!(rules[4].name(), "capability-dependencies");
        assert_eq!(rules[5].name(), "duplicate-execution-mode");
    }
}
