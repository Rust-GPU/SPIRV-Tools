//! Scope validation rules.
//!
//! This module validates SPIR-V scope operands used in atomic, barrier, and
//! group operations. Scopes define the range of invocations affected by
//! synchronization and memory operations.
//!
//! Key validations include:
//! - Scope must be a 32-bit integer
//! - Scope values must be valid (CrossDevice, Device, Workgroup, Subgroup, Invocation, etc.)
//! - Vulkan-specific restrictions on execution and memory scopes
//! - Capability requirements for various scope values

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, MemoryModel, Op, Scope};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::is_constant_opcode;
use crate::validation::types::ResultId;

/// Check if a scope value is valid.
fn is_valid_scope(value: u32) -> bool {
    matches!(
        value,
        x if x == Scope::CrossDevice as u32
            || x == Scope::Device as u32
            || x == Scope::Workgroup as u32
            || x == Scope::Subgroup as u32
            || x == Scope::Invocation as u32
            || x == Scope::QueueFamily as u32
            || x == Scope::ShaderCallKHR as u32
    )
}

/// Evaluate a value ID as a constant 32-bit integer if possible.
fn eval_const_u32(id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let result_id = ResultId::try_from(id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;

    if !is_constant_opcode(inst.class.opcode) {
        return None;
    }

    if inst.class.opcode == Op::Constant || inst.class.opcode == Op::SpecConstant {
        if let Some(Operand::LiteralBit32(value)) = inst.operands.first() {
            return Some(*value);
        }
    }

    None
}

/// Check if an ID is a constant or spec constant.
fn is_constant_id(id: u32, ctx: &ValidationContext<'_>) -> bool {
    if let Ok(result_id) = ResultId::try_from(id) {
        if let Some(inst) = ctx.definitions.get(&result_id) {
            return is_constant_opcode(inst.class.opcode);
        }
    }
    false
}

/// Check if an opcode is a non-uniform group operation.
fn is_non_uniform_group_op(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::GroupNonUniformElect
            | Op::GroupNonUniformAll
            | Op::GroupNonUniformAny
            | Op::GroupNonUniformAllEqual
            | Op::GroupNonUniformBroadcast
            | Op::GroupNonUniformBroadcastFirst
            | Op::GroupNonUniformBallot
            | Op::GroupNonUniformInverseBallot
            | Op::GroupNonUniformBallotBitExtract
            | Op::GroupNonUniformBallotBitCount
            | Op::GroupNonUniformBallotFindLSB
            | Op::GroupNonUniformBallotFindMSB
            | Op::GroupNonUniformShuffle
            | Op::GroupNonUniformShuffleXor
            | Op::GroupNonUniformShuffleUp
            | Op::GroupNonUniformShuffleDown
            | Op::GroupNonUniformIAdd
            | Op::GroupNonUniformFAdd
            | Op::GroupNonUniformIMul
            | Op::GroupNonUniformFMul
            | Op::GroupNonUniformSMin
            | Op::GroupNonUniformUMin
            | Op::GroupNonUniformFMin
            | Op::GroupNonUniformSMax
            | Op::GroupNonUniformUMax
            | Op::GroupNonUniformFMax
            | Op::GroupNonUniformBitwiseAnd
            | Op::GroupNonUniformBitwiseOr
            | Op::GroupNonUniformBitwiseXor
            | Op::GroupNonUniformLogicalAnd
            | Op::GroupNonUniformLogicalOr
            | Op::GroupNonUniformLogicalXor
            | Op::GroupNonUniformQuadBroadcast
            | Op::GroupNonUniformQuadSwap
            | Op::GroupNonUniformRotateKHR
            | Op::GroupNonUniformQuadAllKHR
            | Op::GroupNonUniformQuadAnyKHR
    )
}

/// Validates execution scope operands.
///
/// Execution scope defines which invocations participate in a group operation.
pub struct ExecutionScopeRule;

impl ValidationRule for ExecutionScopeRule {
    fn name(&self) -> &'static str {
        "execution-scope"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();
        let is_vulkan = ctx.is_vulkan_env();
        let has_shader_cap = ctx.has_capability(Capability::Shader);
        let has_cooperative_matrix_nv = ctx.has_capability(Capability::CooperativeMatrixNV);

        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Get execution scope operand index based on opcode
                    let exec_scope_idx = get_execution_scope_operand_index(opcode);

                    if let Some(idx) = exec_scope_idx {
                        if let Some(Operand::IdRef(scope_id)) = inst.operands.get(idx) {
                            self.validate_execution_scope(
                                ctx,
                                opcode,
                                *scope_id,
                                is_vulkan,
                                has_shader_cap,
                                has_cooperative_matrix_nv,
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl ExecutionScopeRule {
    fn validate_execution_scope(
        &self,
        ctx: &ValidationContext<'_>,
        opcode: Op,
        scope_id: u32,
        is_vulkan: bool,
        has_shader_cap: bool,
        has_cooperative_matrix_nv: bool,
    ) -> Result<(), ValidationError> {
        let is_const = is_constant_id(scope_id, ctx);

        if !is_const {
            // Must be constant with Shader capability (unless CooperativeMatrixNV)
            if has_shader_cap && !has_cooperative_matrix_nv {
                return Err(ValidationError::ScopeNotConstantWithShader);
            }
            return Ok(());
        }

        let value = match eval_const_u32(scope_id, ctx) {
            Some(v) => v,
            None => return Ok(()),
        };

        // Validate scope value is valid
        if !is_valid_scope(value) {
            return Err(ValidationError::ScopeInvalidValue { value });
        }

        // Vulkan-specific validation
        if is_vulkan {
            // Non-uniform group operations require Subgroup scope in Vulkan
            // (except QuadAll and QuadAny)
            if is_non_uniform_group_op(opcode)
                && opcode != Op::GroupNonUniformQuadAllKHR
                && opcode != Op::GroupNonUniformQuadAnyKHR
                && value != Scope::Subgroup as u32
            {
                return Err(ValidationError::ScopeNonUniformRequiresSubgroup { opcode });
            }

            // Execution scope is limited to Workgroup or Subgroup in Vulkan
            if value != Scope::Workgroup as u32 && value != Scope::Subgroup as u32 {
                return Err(ValidationError::ScopeExecutionLimitedInVulkan { opcode });
            }
        }

        // General SPIR-V: non-uniform operations limited to Subgroup or Workgroup
        if is_non_uniform_group_op(opcode)
            && opcode != Op::GroupNonUniformQuadAllKHR
            && opcode != Op::GroupNonUniformQuadAnyKHR
            && value != Scope::Subgroup as u32
            && value != Scope::Workgroup as u32
        {
            return Err(ValidationError::ScopeNonUniformLimited { opcode });
        }

        Ok(())
    }
}

/// Validates memory scope operands.
///
/// Memory scope defines the range of memory operations that are synchronized.
pub struct MemoryScopeRule;

impl ValidationRule for MemoryScopeRule {
    fn name(&self) -> &'static str {
        "memory-scope"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();
        let is_vulkan = ctx.is_vulkan_env();
        let has_shader_cap = ctx.has_capability(Capability::Shader);
        let has_vulkan_memory_model = ctx.has_capability(Capability::VulkanMemoryModel);
        let has_vulkan_memory_model_device_scope =
            ctx.has_capability(Capability::VulkanMemoryModelDeviceScope);
        let has_subgroup_ballot = ctx.has_capability(Capability::SubgroupBallotKHR);
        let has_subgroup_vote = ctx.has_capability(Capability::SubgroupVoteKHR);
        let has_cooperative_matrix_nv = ctx.has_capability(Capability::CooperativeMatrixNV);

        // Check if using Vulkan memory model
        let uses_vulkan_memory_model = module
            .memory_model
            .as_ref()
            .map(|mm| {
                mm.operands
                    .get(1)
                    .map(|op| matches!(op, Operand::MemoryModel(MemoryModel::Vulkan)))
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Get memory scope operand index based on opcode
                    let mem_scope_idx = get_memory_scope_operand_index(opcode);

                    if let Some(idx) = mem_scope_idx {
                        if let Some(Operand::IdRef(scope_id)) = inst.operands.get(idx) {
                            self.validate_memory_scope(
                                ctx,
                                opcode,
                                *scope_id,
                                is_vulkan,
                                has_shader_cap,
                                has_vulkan_memory_model,
                                has_vulkan_memory_model_device_scope,
                                uses_vulkan_memory_model,
                                has_subgroup_ballot,
                                has_subgroup_vote,
                                has_cooperative_matrix_nv,
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl MemoryScopeRule {
    #[allow(clippy::too_many_arguments)]
    fn validate_memory_scope(
        &self,
        ctx: &ValidationContext<'_>,
        opcode: Op,
        scope_id: u32,
        is_vulkan: bool,
        has_shader_cap: bool,
        has_vulkan_memory_model: bool,
        has_vulkan_memory_model_device_scope: bool,
        uses_vulkan_memory_model: bool,
        _has_subgroup_ballot: bool,
        _has_subgroup_vote: bool,
        has_cooperative_matrix_nv: bool,
    ) -> Result<(), ValidationError> {
        let is_const = is_constant_id(scope_id, ctx);

        if !is_const {
            // Must be constant with Shader capability (unless CooperativeMatrixNV)
            if has_shader_cap && !has_cooperative_matrix_nv {
                return Err(ValidationError::ScopeNotConstantWithShader);
            }
            return Ok(());
        }

        let value = match eval_const_u32(scope_id, ctx) {
            Some(v) => v,
            None => return Ok(()),
        };

        // Validate scope value is valid
        if !is_valid_scope(value) {
            return Err(ValidationError::ScopeInvalidValue { value });
        }

        // QueueFamilyKHR requires VulkanMemoryModelKHR capability
        if value == Scope::QueueFamily as u32 && !has_vulkan_memory_model {
            return Err(ValidationError::ScopeQueueFamilyRequiresVulkanMemoryModel { opcode });
        }

        // Device scope with VulkanMemoryModel requires VulkanMemoryModelDeviceScopeKHR
        if value == Scope::Device as u32
            && uses_vulkan_memory_model
            && !has_vulkan_memory_model_device_scope
        {
            return Err(ValidationError::ScopeDeviceRequiresDeviceScopeCapability);
        }

        // Vulkan-specific validation
        if is_vulkan {
            // Memory scope is limited in Vulkan
            if value != Scope::Device as u32
                && value != Scope::Workgroup as u32
                && value != Scope::Subgroup as u32
                && value != Scope::Invocation as u32
                && value != Scope::ShaderCallKHR as u32
                && value != Scope::QueueFamily as u32
            {
                return Err(ValidationError::ScopeMemoryLimitedInVulkan { opcode });
            }

            // Note: Vulkan 1.0 specific Subgroup scope validation would require
            // target environment version detection which is not yet implemented
        }

        Ok(())
    }
}

/// Get the execution scope operand index for an opcode.
fn get_execution_scope_operand_index(opcode: Op) -> Option<usize> {
    match opcode {
        // ControlBarrier: Execution scope is first operand
        Op::ControlBarrier => Some(0),

        // Non-uniform group operations: Execution scope is first operand
        op if is_non_uniform_group_op(op) => Some(0),

        // Group operations
        Op::GroupAll | Op::GroupAny | Op::GroupBroadcast => Some(0),
        Op::GroupIAdd
        | Op::GroupFAdd
        | Op::GroupFMin
        | Op::GroupUMin
        | Op::GroupSMin
        | Op::GroupFMax
        | Op::GroupUMax
        | Op::GroupSMax => Some(0),

        _ => None,
    }
}

/// Get the memory scope operand index for an opcode.
fn get_memory_scope_operand_index(opcode: Op) -> Option<usize> {
    match opcode {
        // Atomic operations: Memory scope is operand 1 (after pointer)
        Op::AtomicLoad
        | Op::AtomicStore
        | Op::AtomicExchange
        | Op::AtomicCompareExchange
        | Op::AtomicCompareExchangeWeak
        | Op::AtomicIIncrement
        | Op::AtomicIDecrement
        | Op::AtomicIAdd
        | Op::AtomicISub
        | Op::AtomicSMin
        | Op::AtomicUMin
        | Op::AtomicSMax
        | Op::AtomicUMax
        | Op::AtomicAnd
        | Op::AtomicOr
        | Op::AtomicXor
        | Op::AtomicFlagTestAndSet
        | Op::AtomicFlagClear
        | Op::AtomicFMinEXT
        | Op::AtomicFMaxEXT
        | Op::AtomicFAddEXT => Some(1),

        // Barrier operations
        Op::MemoryBarrier => Some(0),
        Op::ControlBarrier => Some(1), // Memory scope is second operand
        Op::MemoryNamedBarrier => Some(1),

        _ => None,
    }
}

/// Returns all scope validation rules.
pub fn all_scope_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![Box::new(ExecutionScopeRule), Box::new(MemoryScopeRule)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_scope() {
        assert!(is_valid_scope(Scope::CrossDevice as u32));
        assert!(is_valid_scope(Scope::Device as u32));
        assert!(is_valid_scope(Scope::Workgroup as u32));
        assert!(is_valid_scope(Scope::Subgroup as u32));
        assert!(is_valid_scope(Scope::Invocation as u32));
        assert!(is_valid_scope(Scope::QueueFamily as u32));
        assert!(is_valid_scope(Scope::ShaderCallKHR as u32));
        assert!(!is_valid_scope(999));
    }

    #[test]
    fn test_is_non_uniform_group_op() {
        assert!(is_non_uniform_group_op(Op::GroupNonUniformElect));
        assert!(is_non_uniform_group_op(Op::GroupNonUniformAll));
        assert!(is_non_uniform_group_op(Op::GroupNonUniformBallot));
        assert!(!is_non_uniform_group_op(Op::AtomicLoad));
        assert!(!is_non_uniform_group_op(Op::MemoryBarrier));
    }

    #[test]
    fn test_get_execution_scope_operand_index() {
        assert_eq!(get_execution_scope_operand_index(Op::ControlBarrier), Some(0));
        assert_eq!(
            get_execution_scope_operand_index(Op::GroupNonUniformElect),
            Some(0)
        );
        assert_eq!(get_execution_scope_operand_index(Op::AtomicLoad), None);
    }

    #[test]
    fn test_get_memory_scope_operand_index() {
        assert_eq!(get_memory_scope_operand_index(Op::AtomicLoad), Some(1));
        assert_eq!(get_memory_scope_operand_index(Op::MemoryBarrier), Some(0));
        assert_eq!(get_memory_scope_operand_index(Op::ControlBarrier), Some(1));
        assert_eq!(get_memory_scope_operand_index(Op::IAdd), None);
    }

    #[test]
    fn test_all_scope_rules() {
        let rules = all_scope_rules();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].name(), "execution-scope");
        assert_eq!(rules[1].name(), "memory-scope");
    }
}
