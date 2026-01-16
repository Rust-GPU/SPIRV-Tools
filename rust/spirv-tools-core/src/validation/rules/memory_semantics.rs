//! Memory semantics validation rules.
//!
//! This module validates SPIR-V memory semantics operands used in atomic and
//! barrier instructions. Memory semantics specify memory ordering constraints
//! and which storage classes are affected by memory operations.
//!
//! Key validations include:
//! - Memory semantics must be 32-bit integers
//! - Only one memory ordering bit may be set (Acquire, Release, AcquireRelease, SequentiallyConsistent)
//! - Vulkan-specific restrictions on memory ordering and storage classes
//! - Capability requirements for various memory semantics flags

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, MemoryModel, MemorySemantics as MemorySemanticsMask, Op, Scope};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::is_constant_opcode;
use crate::validation::types::ResultId;

/// Bit mask for memory ordering flags.
const MEMORY_ORDER_MASK: u32 = MemorySemanticsMask::ACQUIRE.bits()
    | MemorySemanticsMask::RELEASE.bits()
    | MemorySemanticsMask::ACQUIRE_RELEASE.bits()
    | MemorySemanticsMask::SEQUENTIALLY_CONSISTENT.bits();

/// Bit mask for Vulkan-supported storage class semantics.
const VULKAN_STORAGE_CLASS_MASK: u32 = MemorySemanticsMask::UNIFORM_MEMORY.bits()
    | MemorySemanticsMask::WORKGROUP_MEMORY.bits()
    | MemorySemanticsMask::IMAGE_MEMORY.bits()
    | MemorySemanticsMask::OUTPUT_MEMORY.bits();

/// Count the number of set bits in a value.
fn count_bits(value: u32) -> u32 {
    value.count_ones()
}

/// Check if an opcode is an atomic operation.
fn is_atomic_op(opcode: Op) -> bool {
    matches!(
        opcode,
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
            | Op::AtomicFAddEXT
    )
}

/// Evaluate a value ID as a constant 32-bit integer if possible.
fn eval_const_u32(id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let result_id = ResultId::try_from(id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;

    if !is_constant_opcode(inst.class.opcode) {
        return None;
    }

    // For OpConstant, the value is in operands after the type
    if inst.class.opcode == Op::Constant {
        if let Some(Operand::LiteralBit32(value)) = inst.operands.first() {
            return Some(*value);
        }
    }

    // For OpSpecConstant, treat as constant with the default value
    if inst.class.opcode == Op::SpecConstant {
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

/// Validates memory semantics operands.
///
/// This rule validates that memory semantics values used in atomic and barrier
/// instructions follow the SPIR-V and Vulkan specifications.
pub struct MemorySemanticsRule;

impl ValidationRule for MemorySemanticsRule {
    fn name(&self) -> &'static str {
        "memory-semantics"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();
        let is_vulkan = ctx.is_vulkan_env()
            || module
                .memory_model
                .as_ref()
                .map(|mm| {
                    mm.operands
                        .get(1)
                        .map(|op| matches!(op, Operand::MemoryModel(MemoryModel::Vulkan)))
                        .unwrap_or(false)
                })
                .unwrap_or(false);

        let has_shader_cap = ctx.has_capability(Capability::Shader);
        let has_vulkan_memory_model = ctx.has_capability(Capability::VulkanMemoryModel);
        let has_cooperative_matrix_nv = ctx.has_capability(Capability::CooperativeMatrixNV);

        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Find memory semantics operands based on opcode
                    let semantics_operands = get_memory_semantics_operand_indices(opcode);

                    for (operand_idx, is_unequal) in semantics_operands {
                        if let Some(Operand::IdRef(semantics_id)) = inst.operands.get(operand_idx) {
                            // Get memory scope if present (for Invocation scope check)
                            let memory_scope = get_memory_scope_operand(opcode, inst);

                            self.validate_memory_semantics(
                                ctx,
                                opcode,
                                *semantics_id,
                                is_unequal,
                                is_vulkan,
                                has_shader_cap,
                                has_vulkan_memory_model,
                                has_cooperative_matrix_nv,
                                memory_scope,
                                inst,
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl MemorySemanticsRule {
    fn validate_memory_semantics(
        &self,
        ctx: &ValidationContext<'_>,
        opcode: Op,
        semantics_id: u32,
        is_unequal: bool,
        is_vulkan: bool,
        has_shader_cap: bool,
        has_vulkan_memory_model: bool,
        has_cooperative_matrix_nv: bool,
        memory_scope: Option<u32>,
        inst: &rspirv::dr::Instruction,
    ) -> Result<(), ValidationError> {
        // Check if it's a constant
        let is_const = is_constant_id(semantics_id, ctx);

        if !is_const {
            // Must be constant with Shader capability (unless CooperativeMatrixNV)
            if has_shader_cap && !has_cooperative_matrix_nv {
                return Err(ValidationError::MemorySemanticsNotConstantWithShader);
            }
            // Can't validate further without constant value
            return Ok(());
        }

        let value = match eval_const_u32(semantics_id, ctx) {
            Some(v) => v,
            None => return Ok(()), // Can't evaluate, skip further validation
        };

        // Check capability requirements for various flags
        if value & MemorySemanticsMask::UNIFORM_MEMORY.bits() != 0 {
            if !has_shader_cap {
                return Err(ValidationError::MemorySemanticsUniformMemoryRequiresShader { opcode });
            }
        }

        if value & MemorySemanticsMask::OUTPUT_MEMORY.bits() != 0 {
            if !has_vulkan_memory_model {
                return Err(
                    ValidationError::MemorySemanticsOutputMemoryRequiresVulkanMemoryModel {
                        opcode,
                    },
                );
            }
        }

        // Check memory order - at most one bit may be set
        let order_bits = value & MEMORY_ORDER_MASK;
        let num_order_bits = count_bits(order_bits);

        if num_order_bits > 1 {
            return Err(ValidationError::MemorySemanticsMultipleOrderBits { opcode });
        }

        // Vulkan forbids SequentiallyConsistent
        if is_vulkan && (value & MemorySemanticsMask::SEQUENTIALLY_CONSISTENT.bits() != 0) {
            return Err(ValidationError::MemorySemanticsSequentiallyConsistentInVulkan { opcode });
        }

        // AtomicStore/AtomicFlagClear cannot use Acquire or AcquireRelease
        if (opcode == Op::AtomicStore || opcode == Op::AtomicFlagClear)
            && (value
                & (MemorySemanticsMask::ACQUIRE.bits()
                    | MemorySemanticsMask::ACQUIRE_RELEASE.bits())
                != 0)
        {
            return Err(ValidationError::MemorySemanticsInvalidOrderForStore { opcode });
        }

        // AtomicLoad cannot use Release or AcquireRelease
        if opcode == Op::AtomicLoad
            && (value
                & (MemorySemanticsMask::RELEASE.bits()
                    | MemorySemanticsMask::ACQUIRE_RELEASE.bits())
                != 0)
        {
            return Err(ValidationError::MemorySemanticsInvalidOrderForLoad { opcode });
        }

        // In Vulkan, OpMemoryBarrier must not use relaxed ordering
        if is_vulkan && opcode == Op::MemoryBarrier && num_order_bits == 0 {
            return Err(ValidationError::MemorySemanticsRelaxedBarrierInVulkan { opcode });
        }

        // Vulkan storage class and ordering requirements
        if is_vulkan {
            let includes_storage_class = (value & VULKAN_STORAGE_CLASS_MASK) != 0;

            // Non-relaxed order requires storage class
            if num_order_bits > 0 && !includes_storage_class {
                return Err(ValidationError::MemorySemanticsOrderWithoutStorageClass { opcode });
            }

            // Storage class requires non-relaxed order
            if num_order_bits == 0 && includes_storage_class {
                return Err(ValidationError::MemorySemanticsStorageClassWithoutOrder { opcode });
            }
        }

        // MakeAvailableKHR validation
        if value & MemorySemanticsMask::MAKE_AVAILABLE.bits() != 0 {
            if !has_vulkan_memory_model {
                return Err(
                    ValidationError::MemorySemanticsMakeAvailableRequiresVulkanMemoryModel {
                        opcode,
                    },
                );
            }
            if (value
                & (MemorySemanticsMask::RELEASE.bits()
                    | MemorySemanticsMask::ACQUIRE_RELEASE.bits()))
                == 0
            {
                return Err(ValidationError::MemorySemanticsMakeAvailableRequiresRelease { opcode });
            }
        }

        // MakeVisibleKHR validation
        if value & MemorySemanticsMask::MAKE_VISIBLE.bits() != 0 {
            if !has_vulkan_memory_model {
                return Err(
                    ValidationError::MemorySemanticsMakeVisibleRequiresVulkanMemoryModel { opcode },
                );
            }
            if (value
                & (MemorySemanticsMask::ACQUIRE.bits()
                    | MemorySemanticsMask::ACQUIRE_RELEASE.bits()))
                == 0
            {
                return Err(ValidationError::MemorySemanticsMakeVisibleRequiresAcquire { opcode });
            }
        }

        // Volatile validation
        if value & MemorySemanticsMask::VOLATILE.bits() != 0 {
            if !has_vulkan_memory_model {
                return Err(ValidationError::MemorySemanticsVolatileRequiresVulkanMemoryModel {
                    opcode,
                });
            }
            if !is_atomic_op(opcode) {
                return Err(ValidationError::MemorySemanticsVolatileWithBarrier { opcode });
            }
        }

        // Unequal memory semantics validation for compare-exchange
        if is_unequal {
            // Unequal cannot use Release or AcquireRelease
            if value
                & (MemorySemanticsMask::RELEASE.bits()
                    | MemorySemanticsMask::ACQUIRE_RELEASE.bits())
                != 0
            {
                return Err(ValidationError::MemorySemanticsUnequalInvalidOrder { opcode });
            }

            // Get equal semantics to compare
            if let Some(equal_value) = get_equal_semantics(inst, ctx) {
                // Validate unequal is not stronger than equal
                let unequal_seq_cst =
                    (value & MemorySemanticsMask::SEQUENTIALLY_CONSISTENT.bits()) != 0;
                let equal_seq_cst =
                    (equal_value & MemorySemanticsMask::SEQUENTIALLY_CONSISTENT.bits()) != 0;

                let unequal_acquire = (value & MemorySemanticsMask::ACQUIRE.bits()) != 0;
                let equal_has_acquire_or_stronger = (equal_value
                    & (MemorySemanticsMask::SEQUENTIALLY_CONSISTENT.bits()
                        | MemorySemanticsMask::ACQUIRE_RELEASE.bits()
                        | MemorySemanticsMask::RELEASE.bits()
                        | MemorySemanticsMask::ACQUIRE.bits()))
                    != 0;

                if (unequal_seq_cst && !equal_seq_cst)
                    || (unequal_acquire && !equal_has_acquire_or_stronger)
                {
                    return Err(ValidationError::MemorySemanticsUnequalStrongerThanEqual {
                        opcode,
                    });
                }
            }
        }

        // Vulkan: non-relaxed with Invocation scope is invalid
        if is_vulkan && num_order_bits > 0 {
            if let Some(scope_id) = memory_scope {
                if let Some(scope_value) = eval_const_u32(scope_id, ctx) {
                    if scope_value == Scope::Invocation as u32 {
                        return Err(ValidationError::MemorySemanticsRequiresRelaxedWithInvocation {
                            opcode,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

/// Get the operand indices that contain memory semantics for an opcode.
/// Returns (index, is_unequal) pairs.
fn get_memory_semantics_operand_indices(opcode: Op) -> Vec<(usize, bool)> {
    match opcode {
        // Atomic operations with one memory semantics operand
        Op::AtomicLoad => vec![(2, false)],          // Pointer, Scope, Semantics
        Op::AtomicStore => vec![(2, false)],         // Pointer, Scope, Semantics, Value
        Op::AtomicFlagTestAndSet => vec![(2, false)], // Pointer, Scope, Semantics
        Op::AtomicFlagClear => vec![(2, false)],     // Pointer, Scope, Semantics

        // Atomic operations with semantics at position 3
        Op::AtomicExchange
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
        | Op::AtomicFMinEXT
        | Op::AtomicFMaxEXT
        | Op::AtomicFAddEXT => vec![(2, false)],

        // Compare-exchange has equal and unequal semantics
        Op::AtomicCompareExchange | Op::AtomicCompareExchangeWeak => {
            vec![(2, false), (3, true)] // Equal at 2, Unequal at 3
        }

        // Barrier operations
        Op::MemoryBarrier => vec![(1, false)],    // Scope, Semantics
        Op::ControlBarrier => vec![(2, false)],   // Execution, Memory, Semantics
        Op::MemoryNamedBarrier => vec![(2, false)], // Named, Memory, Semantics

        _ => vec![],
    }
}

/// Get the memory scope operand ID for an instruction if applicable.
fn get_memory_scope_operand(opcode: Op, inst: &rspirv::dr::Instruction) -> Option<u32> {
    let idx = match opcode {
        Op::AtomicLoad
        | Op::AtomicStore
        | Op::AtomicFlagTestAndSet
        | Op::AtomicFlagClear
        | Op::AtomicExchange
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
        | Op::AtomicFMinEXT
        | Op::AtomicFMaxEXT
        | Op::AtomicFAddEXT
        | Op::AtomicCompareExchange
        | Op::AtomicCompareExchangeWeak => Some(1),
        Op::MemoryBarrier => Some(0),
        Op::ControlBarrier => Some(1),
        Op::MemoryNamedBarrier => Some(1),
        _ => None,
    };

    idx.and_then(|i| {
        if let Some(Operand::IdRef(id)) = inst.operands.get(i) {
            Some(*id)
        } else {
            None
        }
    })
}

/// Get the equal memory semantics value for compare-exchange operations.
fn get_equal_semantics(inst: &rspirv::dr::Instruction, ctx: &ValidationContext<'_>) -> Option<u32> {
    if inst.class.opcode == Op::AtomicCompareExchange
        || inst.class.opcode == Op::AtomicCompareExchangeWeak
    {
        if let Some(Operand::IdRef(equal_id)) = inst.operands.get(2) {
            return eval_const_u32(*equal_id, ctx);
        }
    }
    None
}

/// Returns all memory semantics validation rules.
pub fn all_memory_semantics_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![Box::new(MemorySemanticsRule)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_bits() {
        assert_eq!(count_bits(0), 0);
        assert_eq!(count_bits(1), 1);
        assert_eq!(count_bits(0b1010), 2);
        assert_eq!(count_bits(0b1111), 4);
        assert_eq!(count_bits(0xFFFFFFFF), 32);
    }

    #[test]
    fn test_is_atomic_op() {
        assert!(is_atomic_op(Op::AtomicLoad));
        assert!(is_atomic_op(Op::AtomicStore));
        assert!(is_atomic_op(Op::AtomicCompareExchange));
        assert!(!is_atomic_op(Op::MemoryBarrier));
        assert!(!is_atomic_op(Op::ControlBarrier));
        assert!(!is_atomic_op(Op::IAdd));
    }

    #[test]
    fn test_get_memory_semantics_operand_indices() {
        assert_eq!(
            get_memory_semantics_operand_indices(Op::AtomicLoad),
            vec![(2, false)]
        );
        assert_eq!(
            get_memory_semantics_operand_indices(Op::AtomicCompareExchange),
            vec![(2, false), (3, true)]
        );
        assert_eq!(
            get_memory_semantics_operand_indices(Op::MemoryBarrier),
            vec![(1, false)]
        );
        assert_eq!(get_memory_semantics_operand_indices(Op::IAdd), vec![]);
    }
}
