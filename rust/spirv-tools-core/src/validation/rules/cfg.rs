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
use rspirv::spirv::{LoopControl, Op, SelectionControl};

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

            // Skip function declarations (no blocks and no parameters).
            // A function with parameters but no blocks is a definition with missing entry block.
            if func.blocks.is_empty() {
                if !func.parameters.is_empty() {
                    return Err(ValidationError::MissingFunctionEntryBlock { function: func_id });
                }
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
// Loop Back-Edge Rule
// ============================================================================

/// Validates that loops have proper back edges and reachable continue blocks.
///
/// SPIR-V structured control flow requires:
/// - The continue target must be reachable from the loop header
/// - There must be a back edge from the continue construct to the loop header
pub struct LoopBackEdgeRule;

impl ValidationRule for LoopBackEdgeRule {
    fn name(&self) -> &'static str {
        "loop-back-edge"
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

            if func.blocks.is_empty() {
                continue;
            }

            let Some(cfg) = ControlFlowGraph::build(func) else {
                continue;
            };

            // Find all loop headers and their continue targets
            for block in &func.blocks {
                let Some(header_id) = get_block_label(block) else {
                    continue;
                };

                for inst in &block.instructions {
                    if inst.class.opcode == Op::LoopMerge {
                        // Get continue target (second operand)
                        if let Some(Operand::IdRef(raw_continue)) = inst.operands.get(1) {
                            let continue_target = to_id(*raw_continue);

                            // Check that continue target is reachable from header
                            if !cfg.is_reachable(continue_target) {
                                return Err(ValidationError::ContinueNotReachable {
                                    function: func_id,
                                    header: header_id,
                                    continue_target,
                                });
                            }

                            // Check for back edge: continue block or its successors must branch to header
                            // The continue block terminates the continue construct, and one of its
                            // successors must be the header block for a proper loop.
                            let has_back_edge = if let Some(successors) =
                                cfg.get_successors(continue_target)
                            {
                                successors.contains(&header_id)
                            } else {
                                false
                            };

                            if !has_back_edge {
                                // Also check if header is a successor of continue target's successors
                                // (continue block might branch to an intermediate block that goes to header)
                                let indirect_back_edge = cfg
                                    .get_successors(continue_target)
                                    .map(|succs| {
                                        succs.iter().any(|succ| {
                                            cfg.get_successors(*succ)
                                                .map(|s| s.contains(&header_id))
                                                .unwrap_or(false)
                                        })
                                    })
                                    .unwrap_or(false);

                                if !indirect_back_edge {
                                    return Err(ValidationError::LoopMissingBackEdge {
                                        function: func_id,
                                        header: header_id,
                                        continue_target,
                                    });
                                }
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

// ============================================================================
// Dominance Validation Rule
// ============================================================================

/// Validates that all operand uses are dominated by their definitions.
///
/// Checks:
/// - Values used in non-Phi instructions must be dominated by their definition block
/// - Phi incoming values must dominate the incoming edge
/// - Values must be defined in the same function
pub struct DominanceValidationRule;

impl ValidationRule for DominanceValidationRule {
    fn name(&self) -> &'static str {
        "dominance-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        use crate::validation::types::ResultId;
        use std::collections::{HashMap, HashSet};

        // Build a set of all IDs defined inside any function (not globally)
        // These are the IDs that cannot be used cross-function
        let mut function_local_ids: HashSet<ResultId> = HashSet::new();
        for func in &ctx.module.functions {
            for param in &func.parameters {
                if let Some(result_id) = param.result_id {
                    if let Ok(rid) = ResultId::try_from(result_id) {
                        function_local_ids.insert(rid);
                    }
                }
            }
            for block in &func.blocks {
                // Block labels are function-local
                if let Some(label) = &block.label {
                    if let Some(result_id) = label.result_id {
                        if let Ok(rid) = ResultId::try_from(result_id) {
                            function_local_ids.insert(rid);
                        }
                    }
                }
                for inst in &block.instructions {
                    if let Some(result_id) = inst.result_id {
                        if let Ok(rid) = ResultId::try_from(result_id) {
                            function_local_ids.insert(rid);
                        }
                    }
                }
            }
        }

        for func in &ctx.module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            if func.blocks.is_empty() {
                continue;
            }

            let Some(cfg) = ControlFlowGraph::build(func) else {
                continue;
            };

            // Build definition block map
            let mut definition_blocks: HashMap<ResultId, Id> = HashMap::new();
            let entry_label = cfg.entry;

            for param in &func.parameters {
                if let Some(result_id) = param.result_id {
                    if let Ok(rid) = ResultId::try_from(result_id) {
                        definition_blocks.insert(rid, entry_label);
                    }
                }
            }

            // Build set of block labels (these don't need domination checking)
            let mut block_labels: HashSet<ResultId> = HashSet::new();

            for block in &func.blocks {
                let block_id = get_block_label(block).unwrap_or(entry_label);
                // Track block labels separately (they don't need domination checking)
                if let Some(label) = &block.label {
                    if let Some(result_id) = label.result_id {
                        if let Ok(rid) = ResultId::try_from(result_id) {
                            block_labels.insert(rid);
                        }
                    }
                }
                for inst in &block.instructions {
                    if let Some(result_id) = inst.result_id {
                        if let Ok(rid) = ResultId::try_from(result_id) {
                            definition_blocks.insert(rid, block_id);
                        }
                    }
                }
            }

            // Validate dominance
            for block in &func.blocks {
                let block_id = get_block_label(block).unwrap_or(entry_label);

                for inst in &block.instructions {
                    if inst.class.opcode == Op::Phi {
                        // For Phi, value must dominate the incoming edge
                        for pair in inst.operands.chunks(2) {
                            let value_ref = pair.first().and_then(|op| {
                                if let Operand::IdRef(raw) = op {
                                    ResultId::try_from(*raw).ok()
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

                            if let (Some(value_id), Some(incoming)) = (value_ref, incoming_block) {
                                if let Some(def_block) = definition_blocks.get(&value_id) {
                                    if !cfg.blocks.contains(def_block) {
                                        return Err(ValidationError::ValueDefinedInAnotherFunction {
                                            function: func_id,
                                            value: Id::from(value_id),
                                        });
                                    }
                                    if !cfg.dominates(*def_block, incoming) {
                                        return Err(ValidationError::PhiIncomingNotDominated {
                                            function: func_id,
                                            block: block_id,
                                            incoming,
                                            value: Id::from(value_id),
                                        });
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    // For non-Phi, definition must dominate use
                    for operand in &inst.operands {
                        if let Operand::IdRef(raw) = operand {
                            let Some(result_id) = ResultId::try_from(*raw).ok() else {
                                continue;
                            };

                            // Block labels are used as branch targets, not values - skip domination check
                            if block_labels.contains(&result_id) {
                                continue;
                            }

                            if let Some(def_block) = definition_blocks.get(&result_id) {
                                if !cfg.blocks.contains(def_block) {
                                    return Err(ValidationError::ValueDefinedInAnotherFunction {
                                        function: func_id,
                                        value: Id::from(result_id),
                                    });
                                }
                                if !cfg.dominates(*def_block, block_id) {
                                    return Err(ValidationError::ValueNotDominated {
                                        function: func_id,
                                        block: block_id,
                                        value: Id::from(result_id),
                                    });
                                }
                            } else if function_local_ids.contains(&result_id) {
                                // ID is defined in some function but not this one
                                return Err(ValidationError::ValueDefinedInAnotherFunction {
                                    function: func_id,
                                    value: Id::from(result_id),
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
// Loop Control Rule
// ============================================================================

/// Validates loop control flags in OpLoopMerge.
///
/// Checks:
/// - Unroll and DontUnroll cannot both be specified
/// - PeelCount and DontUnroll cannot both be specified
/// - PartialCount and DontUnroll cannot both be specified
/// - IterationMultiple operand must be greater than zero
pub struct LoopControlRule;

impl ValidationRule for LoopControlRule {
    fn name(&self) -> &'static str {
        "loop-control"
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

            for block in &func.blocks {
                let block_id = get_block_label(block).unwrap_or(func_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::LoopMerge {
                        continue;
                    }

                    // Operand 2 is the loop control mask
                    let loop_control = match inst.operands.get(2) {
                        Some(Operand::LoopControl(ctrl)) => *ctrl,
                        _ => continue,
                    };

                    // Check Unroll and DontUnroll cannot both be specified
                    if loop_control.contains(LoopControl::UNROLL)
                        && loop_control.contains(LoopControl::DONT_UNROLL)
                    {
                        return Err(ValidationError::LoopControlUnrollAndDontUnroll {
                            function: func_id,
                            block: block_id,
                        });
                    }

                    // Check PeelCount and DontUnroll cannot both be specified
                    if loop_control.contains(LoopControl::PEEL_COUNT)
                        && loop_control.contains(LoopControl::DONT_UNROLL)
                    {
                        return Err(ValidationError::LoopControlPeelCountAndDontUnroll {
                            function: func_id,
                            block: block_id,
                        });
                    }

                    // Check PartialCount and DontUnroll cannot both be specified
                    if loop_control.contains(LoopControl::PARTIAL_COUNT)
                        && loop_control.contains(LoopControl::DONT_UNROLL)
                    {
                        return Err(ValidationError::LoopControlPartialCountAndDontUnroll {
                            function: func_id,
                            block: block_id,
                        });
                    }

                    // Validate IterationMultiple operand if flag is set
                    if loop_control.contains(LoopControl::ITERATION_MULTIPLE) {
                        // Find the IterationMultiple operand
                        // Operands after the loop control mask are in order based on which flags are set:
                        // DependencyLength, MinIterations, MaxIterations, IterationMultiple, PeelCount, PartialCount
                        let mut operand_index = 3; // Start after merge, continue, loop_control

                        if loop_control.contains(LoopControl::DEPENDENCY_LENGTH) {
                            operand_index += 1;
                        }
                        if loop_control.contains(LoopControl::MIN_ITERATIONS) {
                            operand_index += 1;
                        }
                        if loop_control.contains(LoopControl::MAX_ITERATIONS) {
                            operand_index += 1;
                        }

                        // Now operand_index should point to IterationMultiple operand
                        if let Some(Operand::LiteralBit32(value)) = inst.operands.get(operand_index) {
                            if *value == 0 {
                                return Err(ValidationError::LoopControlIterationMultipleZero {
                                    function: func_id,
                                    block: block_id,
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
// Selection Control Rule
// ============================================================================

/// Validates selection control flags in OpSelectionMerge.
///
/// Checks:
/// - Flatten and DontFlatten cannot both be specified
pub struct SelectionControlRule;

impl ValidationRule for SelectionControlRule {
    fn name(&self) -> &'static str {
        "selection-control"
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

            for block in &func.blocks {
                let block_id = get_block_label(block).unwrap_or(func_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::SelectionMerge {
                        continue;
                    }

                    // Operand 1 is the selection control mask
                    let selection_control = match inst.operands.get(1) {
                        Some(Operand::SelectionControl(ctrl)) => *ctrl,
                        _ => continue,
                    };

                    // Check Flatten and DontFlatten cannot both be specified
                    if selection_control.contains(SelectionControl::FLATTEN)
                        && selection_control.contains(SelectionControl::DONT_FLATTEN)
                    {
                        return Err(ValidationError::SelectionControlFlattenAndDontFlatten {
                            function: func_id,
                            block: block_id,
                        });
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
static LOOP_BACK_EDGE_RULE: LoopBackEdgeRule = LoopBackEdgeRule;
static BRANCH_TARGET_RULE: BranchTargetRule = BranchTargetRule;
static PHI_INSTRUCTION_RULE: PhiInstructionRule = PhiInstructionRule;
static DOMINANCE_VALIDATION_RULE: DominanceValidationRule = DominanceValidationRule;
static LOOP_CONTROL_RULE: LoopControlRule = LoopControlRule;
static SELECTION_CONTROL_RULE: SelectionControlRule = SelectionControlRule;

/// Returns all CFG validation rules.
pub fn all_cfg_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &BLOCK_STRUCTURE_RULE,
        &MERGE_INSTRUCTION_RULE,
        &MERGE_DOMINATION_RULE,
        &LOOP_BACK_EDGE_RULE,
        &BRANCH_TARGET_RULE,
        &PHI_INSTRUCTION_RULE,
        &DOMINANCE_VALIDATION_RULE,
        &LOOP_CONTROL_RULE,
        &SELECTION_CONTROL_RULE,
    ]
}
