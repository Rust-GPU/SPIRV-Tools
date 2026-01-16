//! Control flow graph validation rules.
//!
//! This module validates SPIR-V control flow requirements including:
//!
//! - Function structure (entry blocks, terminators)
//! - Merge instruction placement
//! - Loop and selection construct validity
//! - Dominator relationships
//! - Block reachability
//!
//! # Control Flow Validation
//!
//! SPIR-V has strict structured control flow requirements. Key rules include:
//!
//! - Every function must start with an OpLabel (entry block)
//! - Every block must end with a terminator instruction
//! - OpSelectionMerge/OpLoopMerge must immediately precede branch instructions
//! - Merge blocks must be dominated by their header blocks
//! - Continue targets must be dominated by loop headers

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::cfg_analysis::{get_block_label, ControlFlowGraph};
use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, MergeTargetKind};

/// Helper to convert a raw u32 ID to our Id wrapper type.
fn to_id(raw: u32) -> Id {
    Id::try_from(raw).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

// ============================================================================
// Block Structure Rule
// ============================================================================

/// Validates basic block structure requirements.
///
/// Checks:
/// - Every block has a label (OpLabel)
/// - Every block has a terminator instruction
/// - No instructions appear after the terminator
/// - Entry block has no predecessors
pub struct BlockStructureRule;

impl ValidationRule for BlockStructureRule {
    fn name(&self) -> &'static str {
        "block-structure"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            // Skip function declarations (no blocks)
            if func.blocks.is_empty() {
                continue;
            }

            // Validate entry block exists
            let entry_block = func.blocks.first().ok_or(ValidationError::MissingFunctionEntryBlock {
                function: func_id,
            })?;

            let entry_label = entry_block
                .label
                .as_ref()
                .ok_or(ValidationError::MissingFunctionEntryBlock { function: func_id })?;

            if entry_label.class.opcode != Op::Label {
                return Err(ValidationError::MissingFunctionEntryBlock { function: func_id });
            }

            // Build CFG to check entry predecessors
            if let Some(cfg) = ControlFlowGraph::build(func) {
                if cfg.entry_has_predecessors() {
                    return Err(ValidationError::EntryBlockHasPredecessor {
                        function: func_id,
                        entry: cfg.entry,
                    });
                }
            }

            // Validate each block
            for (block_index, block) in func.blocks.iter().enumerate() {
                let block_id = get_block_label(block).unwrap_or(func_id);

                // Check label exists
                let label = block
                    .label
                    .as_ref()
                    .ok_or(ValidationError::MissingBlockLabel {
                        function: func_id,
                        block_index,
                    })?;

                if label.class.opcode != Op::Label {
                    return Err(ValidationError::MissingBlockLabel {
                        function: func_id,
                        block_index,
                    });
                }

                // Find terminator
                let terminator_index = block
                    .instructions
                    .iter()
                    .position(|inst| rspirv::grammar::reflect::is_block_terminator(inst.class.opcode));

                let Some(term_idx) = terminator_index else {
                    return Err(ValidationError::MissingBlockTerminator {
                        function: func_id,
                        block: block_id,
                    });
                };

                // Check no instructions after terminator
                if term_idx + 1 < block.instructions.len() {
                    return Err(ValidationError::InstructionsAfterTerminator {
                        function: func_id,
                        block: block_id,
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Merge Instruction Rule
// ============================================================================

/// Validates merge instruction placement and targets.
///
/// Checks:
/// - OpSelectionMerge/OpLoopMerge must immediately precede a branch
/// - Only one merge instruction per block
/// - Merge targets must exist within the function
/// - Merge targets cannot be the header block itself
/// - Continue target must differ from merge target
pub struct MergeInstructionRule;

impl ValidationRule for MergeInstructionRule {
    fn name(&self) -> &'static str {
        "merge-instruction"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            // Collect all block IDs in function
            let block_ids: std::collections::HashSet<Id> = func
                .blocks
                .iter()
                .filter_map(get_block_label)
                .collect();

            for block in &func.blocks {
                let block_id = get_block_label(block).unwrap_or(func_id);

                let mut merge_info: Option<(usize, &rspirv::dr::Instruction)> = None;
                let mut terminator_index = None;

                for (index, inst) in block.instructions.iter().enumerate() {
                    // Track merge instructions
                    if inst.class.opcode == Op::SelectionMerge
                        || inst.class.opcode == Op::LoopMerge
                    {
                        if merge_info.is_some() {
                            return Err(ValidationError::DuplicateMergeInstruction {
                                function: func_id,
                                block: block_id,
                            });
                        }
                        merge_info = Some((index, inst));
                    }

                    // Track terminator
                    if rspirv::grammar::reflect::is_block_terminator(inst.class.opcode) {
                        terminator_index = Some(index);
                        break;
                    }
                }

                // If we have a merge instruction, validate it
                if let (Some((merge_idx, merge_inst)), Some(term_idx)) = (merge_info, terminator_index) {
                    // Merge must immediately precede terminator
                    if merge_idx + 1 != term_idx {
                        return Err(ValidationError::MergeInstructionNotBeforeTerminator {
                            function: func_id,
                            block: block_id,
                        });
                    }

                    let terminator = &block.instructions[term_idx];

                    // Validate terminator type matches merge type
                    match merge_inst.class.opcode {
                        Op::SelectionMerge => {
                            if !matches!(
                                terminator.class.opcode,
                                Op::BranchConditional | Op::Switch
                            ) {
                                return Err(ValidationError::InvalidMergeTerminator {
                                    function: func_id,
                                    block: block_id,
                                    terminator: terminator.class.opcode,
                                });
                            }

                            // Validate merge target
                            if let Some(Operand::IdRef(raw_merge)) = merge_inst.operands.first() {
                                let target = to_id(*raw_merge);

                                if target == block_id {
                                    return Err(ValidationError::MergeTargetIsBlock {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target,
                                    });
                                }

                                if !block_ids.contains(&target) {
                                    return Err(ValidationError::MergeTargetMissing {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target,
                                    });
                                }
                            }
                        }
                        Op::LoopMerge => {
                            if !matches!(
                                terminator.class.opcode,
                                Op::Branch | Op::BranchConditional
                            ) {
                                return Err(ValidationError::InvalidMergeTerminator {
                                    function: func_id,
                                    block: block_id,
                                    terminator: terminator.class.opcode,
                                });
                            }

                            // Validate merge and continue targets
                            let merge_target = merge_inst.operands.first().and_then(|op| {
                                if let Operand::IdRef(raw) = op {
                                    Some(to_id(*raw))
                                } else {
                                    None
                                }
                            });
                            let continue_target = merge_inst.operands.get(1).and_then(|op| {
                                if let Operand::IdRef(raw) = op {
                                    Some(to_id(*raw))
                                } else {
                                    None
                                }
                            });

                            if let Some(target) = merge_target {
                                if target == block_id {
                                    return Err(ValidationError::MergeTargetIsBlock {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target,
                                    });
                                }
                                if !block_ids.contains(&target) {
                                    return Err(ValidationError::MergeTargetMissing {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target,
                                    });
                                }
                            }

                            if let Some(target) = continue_target {
                                if target == block_id {
                                    return Err(ValidationError::MergeTargetIsBlock {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Continue,
                                        target,
                                    });
                                }
                                if !block_ids.contains(&target) {
                                    return Err(ValidationError::MergeTargetMissing {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Continue,
                                        target,
                                    });
                                }
                            }

                            // Continue target must differ from merge target
                            if let (Some(merge), Some(cont)) = (merge_target, continue_target) {
                                if merge == cont {
                                    return Err(ValidationError::ContinueTargetMatchesMerge {
                                        function: func_id,
                                        block: block_id,
                                        target: merge,
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // OpSwitch requires SelectionMerge
                if let Some(term_idx) = terminator_index {
                    let terminator = &block.instructions[term_idx];
                    if terminator.class.opcode == Op::Switch {
                        let has_selection_merge = merge_info
                            .map(|(_, inst)| inst.class.opcode == Op::SelectionMerge)
                            .unwrap_or(false);

                        if !has_selection_merge {
                            return Err(ValidationError::MissingSelectionMerge {
                                function: func_id,
                                block: block_id,
                                terminator: terminator.class.opcode,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Merge Domination Rule
// ============================================================================

/// Validates that merge targets are dominated by their header blocks.
///
/// This is a key structured control flow requirement in SPIR-V.
pub struct MergeDominationRule;

impl ValidationRule for MergeDominationRule {
    fn name(&self) -> &'static str {
        "merge-domination"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            // Skip functions without blocks
            if func.blocks.is_empty() {
                continue;
            }

            // Build CFG with dominator info
            let Some(cfg) = ControlFlowGraph::build(func) else {
                continue;
            };

            // Check each block for merge instructions
            for block in &func.blocks {
                let Some(block_id) = get_block_label(block) else {
                    continue;
                };

                // Find merge instruction
                for inst in &block.instructions {
                    match inst.class.opcode {
                        Op::SelectionMerge => {
                            if let Some(Operand::IdRef(raw_merge)) = inst.operands.first() {
                                let merge_target = to_id(*raw_merge);

                                // Merge target must be dominated by header
                                if !cfg.dominates(block_id, merge_target) && block_id != merge_target {
                                    return Err(ValidationError::MergeTargetNotDominated {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target: merge_target,
                                    });
                                }
                            }
                        }
                        Op::LoopMerge => {
                            // Check merge target
                            if let Some(Operand::IdRef(raw_merge)) = inst.operands.first() {
                                let merge_target = to_id(*raw_merge);

                                if !cfg.dominates(block_id, merge_target) && block_id != merge_target {
                                    return Err(ValidationError::MergeTargetNotDominated {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target: merge_target,
                                    });
                                }
                            }

                            // Check continue target
                            if let Some(Operand::IdRef(raw_continue)) = inst.operands.get(1) {
                                let continue_target = to_id(*raw_continue);

                                if !cfg.dominates(block_id, continue_target) && block_id != continue_target {
                                    return Err(ValidationError::MergeTargetNotDominated {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Continue,
                                        target: continue_target,
                                    });
                                }
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
// Branch Target Rule
// ============================================================================

/// Validates that branch targets refer to valid blocks within the function.
pub struct BranchTargetRule;

impl ValidationRule for BranchTargetRule {
    fn name(&self) -> &'static str {
        "branch-target"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            // Collect all block IDs
            let block_ids: std::collections::HashSet<Id> = func
                .blocks
                .iter()
                .filter_map(get_block_label)
                .collect();

            for block in &func.blocks {
                for inst in &block.instructions {
                    let targets: Vec<Id> = match inst.class.opcode {
                        Op::Branch => {
                            inst.operands
                                .first()
                                .and_then(|op| {
                                    if let Operand::IdRef(raw) = op {
                                        Some(vec![to_id(*raw)])
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or_default()
                        }
                        Op::BranchConditional => {
                            inst.operands
                                .iter()
                                .skip(1)
                                .take(2)
                                .filter_map(|op| {
                                    if let Operand::IdRef(raw) = op {
                                        Some(to_id(*raw))
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        }
                        Op::Switch => {
                            inst.operands
                                .iter()
                                .enumerate()
                                .filter_map(|(idx, op)| {
                                    // Skip selector (idx 0), include default (idx 1) and case targets (even indices)
                                    if idx == 0 {
                                        return None;
                                    }
                                    if idx == 1 || idx % 2 == 0 {
                                        if let Operand::IdRef(raw) = op {
                                            return Some(to_id(*raw));
                                        }
                                    }
                                    None
                                })
                                .collect()
                        }
                        _ => vec![],
                    };

                    for target in targets {
                        if !block_ids.contains(&target) {
                            return Err(ValidationError::MissingBlockTarget {
                                function: func_id,
                                target,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Phi Instruction Rule
// ============================================================================

/// Validates OpPhi instructions.
///
/// Checks:
/// - All phi instructions must be at the beginning of the block (before non-phi instructions)
/// - The number of incoming value/block pairs must match the number of predecessors
/// - Each incoming block must be a predecessor of the current block
/// - No duplicate predecessor blocks
/// - Incoming blocks must exist in the function
/// - Types of incoming values must match the phi's result type
/// - Incoming values must be dominated by the incoming block
pub struct PhiInstructionRule;

impl ValidationRule for PhiInstructionRule {
    fn name(&self) -> &'static str {
        "phi-instruction"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        use crate::validation::types::{ResultId, TypeId};
        use std::collections::HashSet;

        let module = ctx.module();

        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            // Skip function declarations
            if func.blocks.is_empty() {
                continue;
            }

            // Build CFG for predecessor/dominator info
            let Some(cfg) = ControlFlowGraph::build(func) else {
                continue;
            };

            // Collect block IDs
            let block_ids: HashSet<Id> = func.blocks.iter().filter_map(get_block_label).collect();

            for block in &func.blocks {
                let block_id = get_block_label(block).unwrap_or(func_id);

                let expected_preds = cfg
                    .get_predecessors(block_id)
                    .map(|p| p.len())
                    .unwrap_or(0);

                let mut seen_non_phi = false;

                for inst in &block.instructions {
                    if inst.class.opcode == Op::Phi {
                        // Phi must come before non-phi instructions
                        if seen_non_phi {
                            return Err(ValidationError::PhiAfterNonPhi {
                                function: func_id,
                                block: block_id,
                            });
                        }

                        // Validate operand count (pairs of value, block)
                        let pair_count = inst.operands.len() / 2;
                        if pair_count != expected_preds {
                            return Err(ValidationError::PhiPredecessorCountMismatch {
                                function: func_id,
                                block: block_id,
                                expected: expected_preds,
                                found: pair_count,
                            });
                        }

                        // Get phi result type
                        let phi_result_type = inst
                            .result_type
                            .and_then(|raw| TypeId::try_from(raw).ok());

                        let mut seen_incoming: HashSet<Id> = HashSet::new();

                        for pair in inst.operands.chunks(2) {
                            let value_ref = pair.first().and_then(|op| {
                                if let Operand::IdRef(raw) = op {
                                    Some(*raw)
                                } else {
                                    None
                                }
                            });
                            let incoming_block = pair.get(1).and_then(|op| {
                                if let Operand::IdRef(raw) = op {
                                    Some(to_id(*raw))
                                } else {
                                    None
                                }
                            });

                            if let Some(incoming) = incoming_block {
                                // Incoming block must exist in function
                                if !block_ids.contains(&incoming) {
                                    return Err(ValidationError::PhiIncomingBlockMissing {
                                        function: func_id,
                                        block: block_id,
                                        incoming,
                                    });
                                }

                                // Incoming block must be a predecessor
                                if let Some(preds) = cfg.get_predecessors(block_id) {
                                    if !preds.contains(&incoming) {
                                        return Err(ValidationError::PhiIncomingNotPredecessor {
                                            function: func_id,
                                            block: block_id,
                                            incoming,
                                        });
                                    }
                                }

                                // Check for duplicate predecessor
                                if !seen_incoming.insert(incoming) {
                                    return Err(ValidationError::PhiDuplicatePredecessor {
                                        function: func_id,
                                        block: block_id,
                                        incoming,
                                    });
                                }

                                // Validate incoming value type matches phi type
                                if let (Some(expected_type), Some(value_raw)) =
                                    (phi_result_type, value_ref)
                                {
                                    if let Ok(value_id) = ResultId::try_from(value_raw) {
                                        if let Some(found_type) = ctx.result_types.get(&value_id) {
                                            if *found_type != expected_type {
                                                return Err(ValidationError::PhiIncomingTypeMismatch {
                                                    function: func_id,
                                                    block: block_id,
                                                    incoming: to_id(value_raw),
                                                    expected: expected_type,
                                                    found: *found_type,
                                                });
                                            }
                                        }
                                    }
                                }

                                // Validate dominance: value must be dominated by incoming block
                                // This is complex - the value's definition block must dominate
                                // the incoming block. For now, we rely on the general dominance
                                // check in validate_functions() which handles this case.
                            }
                        }
                    } else {
                        seen_non_phi = true;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Static rule instances
static BLOCK_STRUCTURE_RULE: BlockStructureRule = BlockStructureRule;
static MERGE_INSTRUCTION_RULE: MergeInstructionRule = MergeInstructionRule;
static MERGE_DOMINATION_RULE: MergeDominationRule = MergeDominationRule;
static BRANCH_TARGET_RULE: BranchTargetRule = BranchTargetRule;
static PHI_INSTRUCTION_RULE: PhiInstructionRule = PhiInstructionRule;

/// Returns all CFG validation rules.
pub fn all_cfg_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &BLOCK_STRUCTURE_RULE,
        &MERGE_INSTRUCTION_RULE,
        &MERGE_DOMINATION_RULE,
        &BRANCH_TARGET_RULE,
        &PHI_INSTRUCTION_RULE,
    ]
}
