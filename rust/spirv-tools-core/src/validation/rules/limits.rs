//! Resource limit enforcement rules.
//!
//! This module validates that SPIR-V modules respect configurable resource
//! limits such as maximum struct members, nesting depth, variable counts, etc.
//!
//! # Adding New Limits
//!
//! To add a new limit:
//!
//! 1. Add a new `LIMIT_*` constant in mod.rs
//! 2. Create a struct implementing [`ValidationRule`]
//! 3. Add it to the validation pipeline
//!
//! All limit enforcement follows the pattern of checking if the limit is set
//! in `options.limits`, then validating against that limit and returning
//! `ValidationError::LimitExceeded` if exceeded.

use std::collections::{HashMap, HashSet};

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::span::ValidationErrorExt;
use crate::validation::types::ResultId;
use crate::validation::ValidationResult;
use crate::validation::{
    LIMIT_MAX_ACCESS_CHAIN_INDEXES, LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH, LIMIT_MAX_FUNCTION_ARGS,
    LIMIT_MAX_GLOBAL_VARIABLES, LIMIT_MAX_LOCAL_VARIABLES, LIMIT_MAX_STRUCT_DEPTH,
    LIMIT_MAX_STRUCT_MEMBERS, LIMIT_MAX_SWITCH_BRANCHES,
};

// ============================================================================
// Struct Member Limit
// ============================================================================

/// Enforces the maximum struct member count limit.
pub struct StructMemberLimitRule;

impl ValidationRule for StructMemberLimitRule {
    fn name(&self) -> &'static str {
        "struct-member-limit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let Some(&limit) = ctx.options.limits.get(&LIMIT_MAX_STRUCT_MEMBERS) else {
            return Ok(());
        };

        for (&struct_id, &member_count) in ctx.struct_member_counts.iter() {
            if member_count as u32 > limit {
                return Err(ValidationError::LimitExceeded {
                    limit_kind: LIMIT_MAX_STRUCT_MEMBERS,
                    limit,
                    found: member_count as u32,
                }
                .at_id_ctx(
                    struct_id,
                    format!("struct has {} members, limit is {}", member_count, limit),
                    ctx,
                ));
            }
        }
        Ok(())
    }
}

// ============================================================================
// Function Argument Limit
// ============================================================================

/// Enforces the maximum function argument count limit.
pub struct FunctionArgLimitRule;

impl ValidationRule for FunctionArgLimitRule {
    fn name(&self) -> &'static str {
        "function-arg-limit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let Some(&limit) = ctx.options.limits.get(&LIMIT_MAX_FUNCTION_ARGS) else {
            return Ok(());
        };

        for function in &ctx.module.functions {
            let arg_count = function.parameters.len() as u32;
            if arg_count > limit {
                let func_id = function.def.as_ref().and_then(|d| d.result_id).unwrap_or(0);
                return Err(ValidationError::LimitExceeded {
                    limit_kind: LIMIT_MAX_FUNCTION_ARGS,
                    limit,
                    found: arg_count,
                }
                .at_id_ctx(
                    func_id,
                    format!("function has {} arguments, limit is {}", arg_count, limit),
                    ctx,
                ));
            }
        }
        Ok(())
    }
}

// ============================================================================
// Struct Depth Limit
// ============================================================================

/// Enforces the maximum struct nesting depth limit.
pub struct StructDepthLimitRule;

impl ValidationRule for StructDepthLimitRule {
    fn name(&self) -> &'static str {
        "struct-depth-limit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let Some(&limit) = ctx.options.limits.get(&LIMIT_MAX_STRUCT_DEPTH) else {
            return Ok(());
        };

        /// Recursively compute the depth of nested structs.
        fn depth_for(
            ty: ResultId,
            defs: &HashMap<ResultId, Instruction>,
            memo: &mut HashMap<ResultId, u32>,
            visiting: &mut HashSet<ResultId>,
        ) -> u32 {
            if let Some(&cached) = memo.get(&ty) {
                return cached;
            }
            if visiting.contains(&ty) {
                // Cycle detected - treat as depth 1
                return 1;
            }
            let Some(inst) = defs.get(&ty) else {
                return 0;
            };
            if inst.class.opcode != Op::TypeStruct {
                memo.insert(ty, 0);
                return 0;
            }
            visiting.insert(ty);
            let mut max_child = 0u32;
            for operand in &inst.operands {
                if let Operand::IdRef(raw) = operand {
                    if let Ok(child) = ResultId::try_from(*raw) {
                        let child_depth = depth_for(child, defs, memo, visiting);
                        max_child = max_child.max(child_depth);
                    }
                }
            }
            visiting.remove(&ty);
            let depth = 1 + max_child;
            memo.insert(ty, depth);
            depth
        }

        let mut memo = HashMap::new();
        let mut visiting = HashSet::new();

        for inst in &ctx.module.types_global_values {
            if let Some(result_id) = inst.result_id {
                if inst.class.opcode == Op::TypeStruct {
                    if let Ok(id) = ResultId::try_from(result_id) {
                        let depth = depth_for(id, ctx.definitions, &mut memo, &mut visiting);
                        if depth > limit {
                            return Err(ValidationError::LimitExceeded {
                                limit_kind: LIMIT_MAX_STRUCT_DEPTH,
                                limit,
                                found: depth,
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

// ============================================================================
// Variable Limits
// ============================================================================

/// Enforces global variable count limit.
pub struct GlobalVariableLimitRule;

impl ValidationRule for GlobalVariableLimitRule {
    fn name(&self) -> &'static str {
        "global-variable-limit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let Some(&limit) = ctx.options.limits.get(&LIMIT_MAX_GLOBAL_VARIABLES) else {
            return Ok(());
        };

        let globals = ctx
            .module
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == Op::Variable)
            .count() as u32;

        if globals > limit {
            return Err(ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_GLOBAL_VARIABLES,
                limit,
                found: globals,
            }
            .into());
        }
        Ok(())
    }
}

/// Enforces local variable count limit.
pub struct LocalVariableLimitRule;

impl ValidationRule for LocalVariableLimitRule {
    fn name(&self) -> &'static str {
        "local-variable-limit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let Some(&limit) = ctx.options.limits.get(&LIMIT_MAX_LOCAL_VARIABLES) else {
            return Ok(());
        };

        let mut locals: u32 = 0;
        for function in &ctx.module.functions {
            for block in &function.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode == Op::Variable {
                        if let Some(Operand::StorageClass(StorageClass::Function)) =
                            inst.operands.first()
                        {
                            locals = locals.saturating_add(1);
                        }
                    }
                }
            }
        }

        if locals > limit {
            return Err(ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_LOCAL_VARIABLES,
                limit,
                found: locals,
            }
            .into());
        }
        Ok(())
    }
}

/// Enforces control flow nesting depth limit.
pub struct ControlFlowNestingLimitRule;

impl ValidationRule for ControlFlowNestingLimitRule {
    fn name(&self) -> &'static str {
        "control-flow-nesting-limit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let Some(&limit) = ctx
            .options
            .limits
            .get(&LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH)
        else {
            return Ok(());
        };

        let mut max_depth = 0u32;
        for function in &ctx.module.functions {
            let mut depth = 0i32;
            for block in &function.blocks {
                for inst in &block.instructions {
                    match inst.class.opcode {
                        Op::SelectionMerge | Op::LoopMerge => {
                            depth = depth.saturating_add(1);
                            max_depth = max_depth.max(depth as u32);
                        }
                        Op::Branch | Op::BranchConditional => {
                            depth = (depth - 1).max(0);
                        }
                        _ => {}
                    }
                }
            }
        }

        if max_depth > limit {
            return Err(ValidationError::LimitExceeded {
                limit_kind: LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH,
                limit,
                found: max_depth,
            }
            .into());
        }
        Ok(())
    }
}

// ============================================================================
// Switch Branch Limit
// ============================================================================

/// Enforces the maximum switch branch count limit.
pub struct SwitchBranchLimitRule;

impl ValidationRule for SwitchBranchLimitRule {
    fn name(&self) -> &'static str {
        "switch-branch-limit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let Some(&limit) = ctx.options.limits.get(&LIMIT_MAX_SWITCH_BRANCHES) else {
            return Ok(());
        };

        for function in &ctx.module.functions {
            for block in &function.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode == Op::Switch {
                        let operands = &inst.operands;
                        if operands.len() < 2 {
                            continue;
                        }
                        let pair_count = (operands.len().saturating_sub(2)) / 2;
                        let branches = 1 + pair_count as u32; // include default target
                        if branches > limit {
                            return Err(ValidationError::LimitExceeded {
                                limit_kind: LIMIT_MAX_SWITCH_BRANCHES,
                                limit,
                                found: branches,
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

// ============================================================================
// Access Chain Limit
// ============================================================================

/// Enforces the maximum access chain index count limit.
pub struct AccessChainLimitRule;

impl ValidationRule for AccessChainLimitRule {
    fn name(&self) -> &'static str {
        "access-chain-limit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let Some(&limit) = ctx.options.limits.get(&LIMIT_MAX_ACCESS_CHAIN_INDEXES) else {
            return Ok(());
        };

        const ACCESS_CHAIN_OPCODES: &[Op] = &[
            Op::AccessChain,
            Op::InBoundsAccessChain,
            Op::PtrAccessChain,
            Op::InBoundsPtrAccessChain,
            Op::UntypedPtrAccessChainKHR,
            Op::UntypedInBoundsPtrAccessChainKHR,
        ];

        let check_inst = |inst: &Instruction| -> ValidationResult {
            if !ACCESS_CHAIN_OPCODES.contains(&inst.class.opcode) {
                return Ok(());
            }
            let num_operands = inst.operands.len();
            let indexes = match inst.class.opcode {
                Op::AccessChain | Op::InBoundsAccessChain => num_operands.saturating_sub(1),
                Op::PtrAccessChain
                | Op::InBoundsPtrAccessChain
                | Op::UntypedPtrAccessChainKHR
                | Op::UntypedInBoundsPtrAccessChainKHR => num_operands.saturating_sub(2),
                _ => 0,
            } as u32;

            if indexes > limit {
                return Err(ValidationError::LimitExceeded {
                    limit_kind: LIMIT_MAX_ACCESS_CHAIN_INDEXES,
                    limit,
                    found: indexes,
                }
                .into());
            }
            Ok(())
        };

        // Check types_global_values section
        for inst in &ctx.module.types_global_values {
            check_inst(inst)?;
        }

        // Check function bodies
        for function in &ctx.module.functions {
            for block in &function.blocks {
                for inst in &block.instructions {
                    check_inst(inst)?;
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// All limit rules
// ============================================================================

/// Returns all limit validation rules.
pub fn all_limit_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &StructMemberLimitRule,
        &FunctionArgLimitRule,
        &StructDepthLimitRule,
        &GlobalVariableLimitRule,
        &LocalVariableLimitRule,
        &ControlFlowNestingLimitRule,
        &SwitchBranchLimitRule,
        &AccessChainLimitRule,
    ]
}
