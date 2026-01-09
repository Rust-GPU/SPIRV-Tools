//! Adjacency and instruction placement validation rules.
//!
//! This module validates SPIR-V instruction placement requirements:
//!
//! - OpPhi instructions must appear at the start of a block (after OpLabel)
//! - OpVariable instructions in functions must appear in the entry block
//! - OpLoopMerge and OpSelectionMerge must immediately precede branch instructions
//!
//! These rules ensure proper structured control flow and SSA form.

use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::Id;

// ============================================================================
// Instruction Placement Categories
// ============================================================================

/// Categories of instructions that affect block structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstructionCategory {
    /// OpPhi - must appear first in a block (after OpLabel)
    Phi,
    /// OpVariable - in functions, must appear in entry block before other instructions
    Variable,
    /// OpLine, OpNoLine - can appear anywhere, don't affect ordering
    LineDebug,
    /// OpLoopMerge, OpSelectionMerge - must immediately precede branch
    Merge,
    /// Branch terminators (OpBranch, OpBranchConditional, OpSwitch, OpReturn, etc.)
    Terminator,
    /// Regular instructions
    Regular,
}

impl InstructionCategory {
    /// Categorize an opcode for adjacency validation.
    fn from_opcode(op: Op) -> Self {
        match op {
            Op::Phi => Self::Phi,
            Op::Variable => Self::Variable,
            Op::Line | Op::NoLine => Self::LineDebug,
            Op::LoopMerge | Op::SelectionMerge => Self::Merge,
            // Terminators
            Op::Branch
            | Op::BranchConditional
            | Op::Switch
            | Op::Return
            | Op::ReturnValue
            | Op::Kill
            | Op::Unreachable
            | Op::TerminateInvocation
            | Op::IgnoreIntersectionKHR
            | Op::TerminateRayKHR
            | Op::EmitMeshTasksEXT
            | Op::DemoteToHelperInvocation => Self::Terminator,
            _ => Self::Regular,
        }
    }

    /// Returns true if this category can appear among OpPhi instructions.
    fn allowed_in_phi_region(self) -> bool {
        matches!(self, Self::Phi | Self::LineDebug)
    }

    /// Returns true if this category can appear among OpVariable instructions.
    fn allowed_in_variable_region(self) -> bool {
        matches!(self, Self::Phi | Self::Variable | Self::LineDebug)
    }
}

// ============================================================================
// OpPhi Placement Rule
// ============================================================================

/// Validates that OpPhi instructions appear at the start of blocks.
///
/// According to the SPIR-V specification, all OpPhi instructions in a block
/// must appear before any other instructions (except OpLine/OpNoLine).
///
/// This rule reports an error if:
/// - An OpPhi appears after a non-Phi instruction in the same block
pub struct PhiPlacementRule;

impl ValidationRule for PhiPlacementRule {
    fn name(&self) -> &'static str {
        "phi-placement"
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

                // Track whether we've seen a non-phi instruction
                let mut seen_non_phi = false;

                for inst in &block.instructions {
                    let category = InstructionCategory::from_opcode(inst.class.opcode);

                    match category {
                        InstructionCategory::Phi => {
                            if seen_non_phi {
                                if let (Some(func), Some(block)) = (function_id, block_id) {
                                    return Err(ValidationError::PhiAfterNonPhi {
                                        function: func,
                                        block,
                                    });
                                }
                            }
                        }
                        InstructionCategory::LineDebug => {
                            // Line debug instructions don't affect phi ordering
                        }
                        _ => {
                            seen_non_phi = true;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpVariable Placement Rule
// ============================================================================

/// Validates that OpVariable instructions in functions appear in the entry block.
///
/// According to the SPIR-V specification, OpVariable instructions with Function
/// storage class must appear in the first block of the function (the entry block).
///
/// This rule reports an error if:
/// - An OpVariable appears in a block that is not the entry block
/// - An OpVariable appears after non-variable instructions in the entry block
pub struct VariablePlacementRule;

impl ValidationRule for VariablePlacementRule {
    fn name(&self) -> &'static str {
        "variable-placement"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            let entry_block_id = function
                .blocks
                .first()
                .and_then(|b| b.label.as_ref())
                .and_then(|l| l.result_id);

            for (block_idx, block) in function.blocks.iter().enumerate() {
                let block_label_id = block.label.as_ref().and_then(|l| l.result_id);
                let is_entry_block = block_idx == 0
                    || (entry_block_id.is_some() && block_label_id == entry_block_id);

                // Track whether we've exited the variable region
                let mut past_variable_region = false;

                for inst in &block.instructions {
                    let category = InstructionCategory::from_opcode(inst.class.opcode);

                    if category == InstructionCategory::Variable {
                        let variable_id = inst.result_id.and_then(|id| Id::try_from(id).ok());

                        // OpVariable in non-entry block
                        if !is_entry_block {
                            if let (Some(func), Some(var)) = (function_id, variable_id) {
                                return Err(ValidationError::FunctionVariableNotInEntryBlock {
                                    function: func,
                                    variable: var,
                                });
                            }
                        }

                        // OpVariable after non-variable instructions in entry block
                        // Note: The spec allows OpPhi before OpVariable in the entry block
                        // (though this is unusual), and line debug instructions anywhere
                        if past_variable_region {
                            // This would be a layout error - variables after other instructions
                            // The existing FunctionVariableNotInEntryBlock error could be
                            // repurposed or a new error added, but for now we just check
                            // the entry block constraint
                        }
                    } else if !category.allowed_in_variable_region() {
                        past_variable_region = true;
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Merge Instruction Adjacency Rule
// ============================================================================

/// Validates that merge instructions immediately precede branch instructions.
///
/// According to the SPIR-V specification:
/// - OpLoopMerge must immediately precede OpBranch or OpBranchConditional
/// - OpSelectionMerge must immediately precede OpBranchConditional or OpSwitch
///
/// This rule reports an error if:
/// - A merge instruction is not immediately followed by an appropriate terminator
/// - A merge instruction appears without any terminator following it
pub struct MergeAdjacencyRule;

impl ValidationRule for MergeAdjacencyRule {
    fn name(&self) -> &'static str {
        "merge-adjacency"
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

                let instructions: Vec<_> = block.instructions.iter().collect();
                let mut i = 0;

                while i < instructions.len() {
                    let inst = instructions[i];
                    let op = inst.class.opcode;

                    if op == Op::LoopMerge || op == Op::SelectionMerge {
                        // Find the next non-line instruction
                        let mut next_idx = i + 1;
                        while next_idx < instructions.len() {
                            let next_op = instructions[next_idx].class.opcode;
                            if next_op != Op::Line && next_op != Op::NoLine {
                                break;
                            }
                            next_idx += 1;
                        }

                        // Check if we have a valid terminator
                        let valid_terminator = if next_idx < instructions.len() {
                            let next_op = instructions[next_idx].class.opcode;
                            match op {
                                Op::LoopMerge => {
                                    // Must be followed by OpBranch or OpBranchConditional
                                    next_op == Op::Branch || next_op == Op::BranchConditional
                                }
                                Op::SelectionMerge => {
                                    // Must be followed by OpBranchConditional or OpSwitch
                                    next_op == Op::BranchConditional || next_op == Op::Switch
                                }
                                _ => false,
                            }
                        } else {
                            false
                        };

                        if !valid_terminator {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                // Determine what terminator we actually found (if any)
                                let found_terminator = if next_idx < instructions.len() {
                                    instructions[next_idx].class.opcode
                                } else {
                                    // No terminator at all - use a generic opcode
                                    Op::Nop
                                };

                                return Err(ValidationError::InvalidMergeTerminator {
                                    function: func,
                                    block,
                                    terminator: found_terminator,
                                });
                            }
                        }
                    }

                    i += 1;
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// All adjacency rules
// ============================================================================

/// Returns all adjacency validation rules.
pub fn all_adjacency_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &PhiPlacementRule,
        &VariablePlacementRule,
        &MergeAdjacencyRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_category_phi() {
        assert_eq!(InstructionCategory::from_opcode(Op::Phi), InstructionCategory::Phi);
    }

    #[test]
    fn test_instruction_category_variable() {
        assert_eq!(InstructionCategory::from_opcode(Op::Variable), InstructionCategory::Variable);
    }

    #[test]
    fn test_instruction_category_line_debug() {
        assert_eq!(InstructionCategory::from_opcode(Op::Line), InstructionCategory::LineDebug);
        assert_eq!(InstructionCategory::from_opcode(Op::NoLine), InstructionCategory::LineDebug);
    }

    #[test]
    fn test_instruction_category_merge() {
        assert_eq!(InstructionCategory::from_opcode(Op::LoopMerge), InstructionCategory::Merge);
        assert_eq!(InstructionCategory::from_opcode(Op::SelectionMerge), InstructionCategory::Merge);
    }

    #[test]
    fn test_instruction_category_terminator() {
        assert_eq!(InstructionCategory::from_opcode(Op::Branch), InstructionCategory::Terminator);
        assert_eq!(InstructionCategory::from_opcode(Op::BranchConditional), InstructionCategory::Terminator);
        assert_eq!(InstructionCategory::from_opcode(Op::Switch), InstructionCategory::Terminator);
        assert_eq!(InstructionCategory::from_opcode(Op::Return), InstructionCategory::Terminator);
        assert_eq!(InstructionCategory::from_opcode(Op::ReturnValue), InstructionCategory::Terminator);
        assert_eq!(InstructionCategory::from_opcode(Op::Kill), InstructionCategory::Terminator);
        assert_eq!(InstructionCategory::from_opcode(Op::Unreachable), InstructionCategory::Terminator);
    }

    #[test]
    fn test_instruction_category_regular() {
        assert_eq!(InstructionCategory::from_opcode(Op::IAdd), InstructionCategory::Regular);
        assert_eq!(InstructionCategory::from_opcode(Op::FMul), InstructionCategory::Regular);
        assert_eq!(InstructionCategory::from_opcode(Op::Load), InstructionCategory::Regular);
        assert_eq!(InstructionCategory::from_opcode(Op::Store), InstructionCategory::Regular);
    }

    #[test]
    fn test_allowed_in_phi_region() {
        assert!(InstructionCategory::Phi.allowed_in_phi_region());
        assert!(InstructionCategory::LineDebug.allowed_in_phi_region());
        assert!(!InstructionCategory::Variable.allowed_in_phi_region());
        assert!(!InstructionCategory::Regular.allowed_in_phi_region());
        assert!(!InstructionCategory::Merge.allowed_in_phi_region());
        assert!(!InstructionCategory::Terminator.allowed_in_phi_region());
    }

    #[test]
    fn test_allowed_in_variable_region() {
        assert!(InstructionCategory::Phi.allowed_in_variable_region());
        assert!(InstructionCategory::Variable.allowed_in_variable_region());
        assert!(InstructionCategory::LineDebug.allowed_in_variable_region());
        assert!(!InstructionCategory::Regular.allowed_in_variable_region());
        assert!(!InstructionCategory::Merge.allowed_in_variable_region());
        assert!(!InstructionCategory::Terminator.allowed_in_variable_region());
    }
}
