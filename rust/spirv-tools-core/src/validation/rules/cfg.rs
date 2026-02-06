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

use std::collections::{HashMap, HashSet};

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, ExecutionMode, LoopControl, Op, SelectionControl};

use crate::validation::cfg_analysis::{get_block_label, ControlFlowGraph};
use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::TypeInstructionExt;
use crate::validation::types::{Id, MergeTargetKind, ResultId};
use crate::validation::ValidationResult;
use crate::version::SpirvVersion;

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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                    return Err(
                        ValidationError::MissingFunctionEntryBlock { function: func_id }.into(),
                    );
                }
                continue;
            }

            // Validate entry block exists
            let entry_block = func
                .blocks
                .first()
                .ok_or(ValidationError::MissingFunctionEntryBlock { function: func_id })?;

            let entry_label = entry_block
                .label
                .as_ref()
                .ok_or(ValidationError::MissingFunctionEntryBlock { function: func_id })?;

            if entry_label.class.opcode != Op::Label {
                return Err(
                    ValidationError::MissingFunctionEntryBlock { function: func_id }.into(),
                );
            }

            // Build CFG to check entry predecessors
            if let Some(cfg) = ControlFlowGraph::build(func) {
                if cfg.entry_has_predecessors() {
                    return Err(ValidationError::EntryBlockHasPredecessor {
                        function: func_id,
                        entry: cfg.entry,
                    }
                    .into());
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
                    }
                    .into());
                }

                // Find terminator
                let terminator_index = block.instructions.iter().position(|inst| {
                    rspirv::grammar::reflect::is_block_terminator(inst.class.opcode)
                });

                let Some(term_idx) = terminator_index else {
                    return Err(ValidationError::MissingBlockTerminator {
                        function: func_id,
                        block: block_id,
                    }
                    .into());
                };

                // Check no instructions after terminator
                if term_idx + 1 < block.instructions.len() {
                    return Err(ValidationError::InstructionsAfterTerminator {
                        function: func_id,
                        block: block_id,
                    }
                    .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            // Collect all block IDs in function
            let block_ids: std::collections::HashSet<Id> =
                func.blocks.iter().filter_map(get_block_label).collect();

            for block in &func.blocks {
                let block_id = get_block_label(block).unwrap_or(func_id);

                let mut merge_info: Option<(usize, &rspirv::dr::Instruction)> = None;
                let mut terminator_index = None;

                for (index, inst) in block.instructions.iter().enumerate() {
                    // Track merge instructions
                    if inst.class.opcode == Op::SelectionMerge || inst.class.opcode == Op::LoopMerge
                    {
                        if merge_info.is_some() {
                            return Err(ValidationError::DuplicateMergeInstruction {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
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
                if let (Some((merge_idx, merge_inst)), Some(term_idx)) =
                    (merge_info, terminator_index)
                {
                    // Merge must immediately precede terminator
                    if merge_idx + 1 != term_idx {
                        return Err(ValidationError::MergeInstructionNotBeforeTerminator {
                            function: func_id,
                            block: block_id,
                        }
                        .into());
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
                                }
                                .into());
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
                                    }
                                    .into());
                                }

                                if !block_ids.contains(&target) {
                                    return Err(ValidationError::MergeTargetMissing {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target,
                                    }
                                    .into());
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
                                }
                                .into());
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
                                    }
                                    .into());
                                }
                                if !block_ids.contains(&target) {
                                    return Err(ValidationError::MergeTargetMissing {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target,
                                    }
                                    .into());
                                }
                            }

                            if let Some(target) = continue_target {
                                if target == block_id {
                                    return Err(ValidationError::MergeTargetIsBlock {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Continue,
                                        target,
                                    }
                                    .into());
                                }
                                if !block_ids.contains(&target) {
                                    return Err(ValidationError::MergeTargetMissing {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Continue,
                                        target,
                                    }
                                    .into());
                                }
                            }

                            // Continue target must differ from merge target
                            if let (Some(merge), Some(cont)) = (merge_target, continue_target) {
                                if merge == cont {
                                    return Err(ValidationError::ContinueTargetMatchesMerge {
                                        function: func_id,
                                        block: block_id,
                                        target: merge,
                                    }
                                    .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                                if !cfg.dominates(block_id, merge_target)
                                    && block_id != merge_target
                                {
                                    return Err(ValidationError::MergeTargetNotDominated {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target: merge_target,
                                    }
                                    .into());
                                }
                            }
                        }
                        Op::LoopMerge => {
                            // Check merge target
                            if let Some(Operand::IdRef(raw_merge)) = inst.operands.first() {
                                let merge_target = to_id(*raw_merge);

                                if !cfg.dominates(block_id, merge_target)
                                    && block_id != merge_target
                                {
                                    return Err(ValidationError::MergeTargetNotDominated {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Merge,
                                        target: merge_target,
                                    }
                                    .into());
                                }
                            }

                            // Check continue target
                            if let Some(Operand::IdRef(raw_continue)) = inst.operands.get(1) {
                                let continue_target = to_id(*raw_continue);

                                if !cfg.dominates(block_id, continue_target)
                                    && block_id != continue_target
                                {
                                    return Err(ValidationError::MergeTargetNotDominated {
                                        function: func_id,
                                        block: block_id,
                                        kind: MergeTargetKind::Continue,
                                        target: continue_target,
                                    }
                                    .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                                }
                                .into());
                            }

                            // Check for back edge: continue block or its successors must branch to header
                            // The continue block terminates the continue construct, and one of its
                            // successors must be the header block for a proper loop.
                            let has_back_edge =
                                if let Some(successors) = cfg.get_successors(continue_target) {
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
                                    }
                                    .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            // Collect all block IDs
            let block_ids: std::collections::HashSet<Id> =
                func.blocks.iter().filter_map(get_block_label).collect();

            for block in &func.blocks {
                for inst in &block.instructions {
                    let targets: Vec<Id> = match inst.class.opcode {
                        Op::Branch => inst
                            .operands
                            .first()
                            .and_then(|op| {
                                if let Operand::IdRef(raw) = op {
                                    Some(vec![to_id(*raw)])
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_default(),
                        Op::BranchConditional => inst
                            .operands
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
                            .collect(),
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
// Phi Instruction Rule
// ============================================================================

/// Validates OpPhi instructions.
///
/// Checks:
/// - All phi instructions must be at the beginning of the block (before non-phi instructions)
/// - OpPhi must not have void result type
/// - OpPhi with pointer type requires VariablePointers capability in logical addressing
/// - OpPhi cannot have sampled image/image/sampler result type
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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

                let expected_preds = cfg.get_predecessors(block_id).map(|p| p.len()).unwrap_or(0);

                let mut seen_non_phi = false;

                for inst in &block.instructions {
                    if inst.class.opcode == Op::Phi {
                        // Phi must come before non-phi instructions
                        if seen_non_phi {
                            return Err(ValidationError::PhiAfterNonPhi {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }

                        // Validate result type is not void, pointer (without VariablePointers),
                        // or sampled image/image/sampler
                        if let Some(result_type_id) = inst.result_type {
                            if let Ok(result_type_rid) = ResultId::try_from(result_type_id) {
                                if let Some(type_inst) = ctx.definitions.get(&result_type_rid) {
                                    let type_op = type_inst.class.opcode;

                                    // OpPhi must not have void result type
                                    if type_op == Op::TypeVoid {
                                        return Err(ValidationError::PhiVoidResultType {
                                            function: func_id,
                                            block: block_id,
                                        }
                                        .into());
                                    }

                                    // OpPhi with pointer requires VariablePointers in logical addressing
                                    if type_op == Op::TypePointer {
                                        let is_logical = ctx.is_logical_addressing();
                                        let has_variable_pointers = ctx.declared_capabilities.contains(
                                            &rspirv::spirv::Capability::VariablePointers,
                                        ) || ctx.declared_capabilities.contains(
                                            &rspirv::spirv::Capability::VariablePointersStorageBuffer,
                                        );

                                        if is_logical && !has_variable_pointers {
                                            return Err(
                                                ValidationError::PhiPointerRequiresVariablePointers {
                                                    function: func_id,
                                                    block: block_id,
                                                }.into(),
                        );
                                        }
                                    }

                                    // OpPhi cannot have sampled image, image, or sampler
                                    // (unless BindlessTextureNV is declared or before_hlsl_legalization)
                                    let has_bindless = ctx
                                        .declared_capabilities
                                        .contains(&rspirv::spirv::Capability::BindlessTextureNV);
                                    if !has_bindless
                                        && matches!(
                                            type_op,
                                            Op::TypeSampledImage | Op::TypeImage | Op::TypeSampler
                                        )
                                    {
                                        return Err(ValidationError::PhiInvalidResultType {
                                            function: func_id,
                                            block: block_id,
                                            type_opcode: type_op,
                                        }
                                        .into());
                                    }
                                }
                            }
                        }

                        // Validate operand count (pairs of value, block)
                        let pair_count = inst.operands.len() / 2;
                        if pair_count != expected_preds {
                            return Err(ValidationError::PhiPredecessorCountMismatch {
                                function: func_id,
                                block: block_id,
                                expected: expected_preds,
                                found: pair_count,
                            }
                            .into());
                        }

                        // Get phi result type
                        let phi_result_type =
                            inst.result_type.and_then(|raw| TypeId::try_from(raw).ok());

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
                                    }
                                    .into());
                                }

                                // Incoming block must be a predecessor
                                if let Some(preds) = cfg.get_predecessors(block_id) {
                                    if !preds.contains(&incoming) {
                                        return Err(ValidationError::PhiIncomingNotPredecessor {
                                            function: func_id,
                                            block: block_id,
                                            incoming,
                                        }
                                        .into());
                                    }
                                }

                                // Check for duplicate predecessor
                                if !seen_incoming.insert(incoming) {
                                    return Err(ValidationError::PhiDuplicatePredecessor {
                                        function: func_id,
                                        block: block_id,
                                        incoming,
                                    }
                                    .into());
                                }

                                // Validate incoming value type matches phi type
                                if let (Some(expected_type), Some(value_raw)) =
                                    (phi_result_type, value_ref)
                                {
                                    if let Ok(value_id) = ResultId::try_from(value_raw) {
                                        if let Some(found_type) = ctx.result_types.get(&value_id) {
                                            if *found_type != expected_type {
                                                return Err(
                                                    ValidationError::PhiIncomingTypeMismatch {
                                                        function: func_id,
                                                        block: block_id,
                                                        incoming: to_id(value_raw),
                                                        expected: expected_type,
                                                        found: *found_type,
                                                    }
                                                    .into(),
                                                );
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                                        return Err(
                                            ValidationError::ValueDefinedInAnotherFunction {
                                                function: func_id,
                                                value: Id::from(value_id),
                                            }
                                            .into(),
                                        );
                                    }
                                    if !cfg.dominates(*def_block, incoming) {
                                        return Err(ValidationError::PhiIncomingNotDominated {
                                            function: func_id,
                                            block: block_id,
                                            incoming,
                                            value: Id::from(value_id),
                                        }
                                        .into());
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
                                    }
                                    .into());
                                }
                                if !cfg.dominates(*def_block, block_id) {
                                    return Err(ValidationError::ValueNotDominated {
                                        function: func_id,
                                        block: block_id,
                                        value: Id::from(result_id),
                                    }
                                    .into());
                                }
                            } else if function_local_ids.contains(&result_id) {
                                // ID is defined in some function but not this one
                                return Err(ValidationError::ValueDefinedInAnotherFunction {
                                    function: func_id,
                                    value: Id::from(result_id),
                                }
                                .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                        }
                        .into());
                    }

                    // Check PeelCount and DontUnroll cannot both be specified
                    if loop_control.contains(LoopControl::PEEL_COUNT)
                        && loop_control.contains(LoopControl::DONT_UNROLL)
                    {
                        return Err(ValidationError::LoopControlPeelCountAndDontUnroll {
                            function: func_id,
                            block: block_id,
                        }
                        .into());
                    }

                    // Check PartialCount and DontUnroll cannot both be specified
                    if loop_control.contains(LoopControl::PARTIAL_COUNT)
                        && loop_control.contains(LoopControl::DONT_UNROLL)
                    {
                        return Err(ValidationError::LoopControlPartialCountAndDontUnroll {
                            function: func_id,
                            block: block_id,
                        }
                        .into());
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
                        if let Some(Operand::LiteralBit32(value)) = inst.operands.get(operand_index)
                        {
                            if *value == 0 {
                                return Err(ValidationError::LoopControlIterationMultipleZero {
                                    function: func_id,
                                    block: block_id,
                                }
                                .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// BranchConditional Same Labels Rule
// ============================================================================

/// Validates that BranchConditional true and false labels are different.
///
/// Checks:
/// - In SPIR-V 1.6 or later, True Label and False Label must be different
/// - In MaximallyReconvergesKHR execution mode, True Label and False Label must be different
pub struct BranchConditionalRule;

impl ValidationRule for BranchConditionalRule {
    fn name(&self) -> &'static str {
        "branch-conditional"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();
        let is_spirv_1_6_or_later = ctx.target_version >= SpirvVersion::new(1, 6);

        // Collect functions that use MaximallyReconvergesKHR execution mode
        // This is set per entry point, and we need to track which functions are called
        // from those entry points.
        let mut maximal_reconverge_funcs: HashSet<Id> = HashSet::new();

        // First, collect entry points with MaximallyReconvergesKHR
        for entry_point in &module.entry_points {
            if entry_point.class.opcode != Op::EntryPoint {
                continue;
            }

            // Get the entry point function ID (operand 1)
            let entry_func_id = match entry_point.operands.get(1) {
                Some(Operand::IdRef(id)) => to_id(*id),
                _ => continue,
            };

            // Check if this entry point has MaximallyReconvergesKHR execution mode
            for exec_mode in &module.execution_modes {
                if exec_mode.class.opcode != Op::ExecutionMode
                    && exec_mode.class.opcode != Op::ExecutionModeId
                {
                    continue;
                }

                // Check if this execution mode targets this entry point
                let mode_target = match exec_mode.operands.first() {
                    Some(Operand::IdRef(id)) => to_id(*id),
                    _ => continue,
                };

                if mode_target != entry_func_id {
                    continue;
                }

                // Check if the execution mode is MaximallyReconvergesKHR (value 6023)
                if let Some(Operand::ExecutionMode(mode)) = exec_mode.operands.get(1) {
                    // MaximallyReconvergesKHR = 6023
                    if *mode == ExecutionMode::MaximallyReconvergesKHR {
                        maximal_reconverge_funcs.insert(entry_func_id);
                    }
                }
            }
        }

        // Now validate BranchConditional instructions
        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            let is_maximal_reconverge = maximal_reconverge_funcs.contains(&func_id);

            for block in &func.blocks {
                let block_id = get_block_label(block).unwrap_or(func_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::BranchConditional {
                        continue;
                    }

                    // Get true and false label operands (operands 1 and 2)
                    let true_label = match inst.operands.get(1) {
                        Some(Operand::IdRef(id)) => *id,
                        _ => continue,
                    };
                    let false_label = match inst.operands.get(2) {
                        Some(Operand::IdRef(id)) => *id,
                        _ => continue,
                    };

                    // Check if true and false labels are the same
                    if true_label == false_label {
                        // In SPIR-V 1.6 or later, this is always an error
                        if is_spirv_1_6_or_later {
                            return Err(ValidationError::BranchConditionalSameLabels {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }

                        // In MaximallyReconvergesKHR execution mode, this is also an error
                        if is_maximal_reconverge {
                            return Err(
                                ValidationError::BranchConditionalSameLabelsMaximalReconvergence {
                                    function: func_id,
                                    block: block_id,
                                }
                                .into(),
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
// MaximalReconvergence Multiple Predecessors Rule
// ============================================================================

/// Validates that blocks in MaximallyReconvergesKHR functions don't have multiple predecessors.
///
/// In entry points using MaximallyReconvergesKHR execution mode, basic blocks cannot have
/// multiple unique predecessors, except for:
/// - Loop headers (blocks preceded by OpLoopMerge)
/// - Merge targets (targets of OpSelectionMerge or OpLoopMerge)
/// - Switch targets (targets of OpSwitch)
pub struct MaximalReconvergencePredecessorsRule;

impl ValidationRule for MaximalReconvergencePredecessorsRule {
    fn name(&self) -> &'static str {
        "maximal-reconvergence-predecessors"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Collect entry points with MaximallyReconvergesKHR
        let mut maximal_reconverge_funcs: HashSet<Id> = HashSet::new();

        for entry_point in &module.entry_points {
            if entry_point.class.opcode != Op::EntryPoint {
                continue;
            }

            let entry_func_id = match entry_point.operands.get(1) {
                Some(Operand::IdRef(id)) => to_id(*id),
                _ => continue,
            };

            // Check if this entry point has MaximallyReconvergesKHR execution mode
            for exec_mode in &module.execution_modes {
                if exec_mode.class.opcode != Op::ExecutionMode
                    && exec_mode.class.opcode != Op::ExecutionModeId
                {
                    continue;
                }

                let mode_target = match exec_mode.operands.first() {
                    Some(Operand::IdRef(id)) => to_id(*id),
                    _ => continue,
                };

                if mode_target != entry_func_id {
                    continue;
                }

                if let Some(Operand::ExecutionMode(mode)) = exec_mode.operands.get(1) {
                    if *mode == ExecutionMode::MaximallyReconvergesKHR {
                        maximal_reconverge_funcs.insert(entry_func_id);
                    }
                }
            }
        }

        if maximal_reconverge_funcs.is_empty() {
            return Ok(());
        }

        // Process each function that uses MaximallyReconvergesKHR
        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            let func_id_val = match func_id {
                Some(id) => id,
                None => continue,
            };

            if !maximal_reconverge_funcs.contains(&func_id_val) {
                continue;
            }

            // Build a map of block ID -> set of predecessor block IDs
            let mut predecessors: HashMap<u32, HashSet<u32>> = HashMap::new();

            // Also track which blocks are loop headers (have OpLoopMerge before terminator)
            let mut loop_headers: HashSet<u32> = HashSet::new();

            // Track which blocks are merge targets or switch targets
            let mut allowed_multi_pred_blocks: HashSet<u32> = HashSet::new();

            for block in &func.blocks {
                let block_label = block.label.as_ref().and_then(|l| l.result_id).unwrap_or(0);

                // Check if this block is a loop header (has OpLoopMerge)
                for inst in &block.instructions {
                    match inst.class.opcode {
                        Op::LoopMerge => {
                            loop_headers.insert(block_label);
                            // Merge target is allowed to have multiple predecessors
                            if let Some(Operand::IdRef(merge_id)) = inst.operands.first() {
                                allowed_multi_pred_blocks.insert(*merge_id);
                            }
                            // Continue target is allowed to have multiple predecessors
                            if let Some(Operand::IdRef(continue_id)) = inst.operands.get(1) {
                                allowed_multi_pred_blocks.insert(*continue_id);
                            }
                        }
                        Op::SelectionMerge => {
                            // Merge target is allowed to have multiple predecessors
                            if let Some(Operand::IdRef(merge_id)) = inst.operands.first() {
                                allowed_multi_pred_blocks.insert(*merge_id);
                            }
                        }
                        Op::Switch => {
                            // All switch targets are allowed to have multiple predecessors
                            for (i, operand) in inst.operands.iter().enumerate() {
                                // Switch operands: Selector, Default, then pairs of (Literal, Label)
                                // Default is operand 1, then labels are at odd indices >= 3
                                if i == 1 {
                                    if let Operand::IdRef(default_id) = operand {
                                        allowed_multi_pred_blocks.insert(*default_id);
                                    }
                                } else if i >= 2 && i % 2 == 1 {
                                    if let Operand::IdRef(label_id) = operand {
                                        allowed_multi_pred_blocks.insert(*label_id);
                                    }
                                }
                            }
                        }
                        Op::Branch => {
                            if let Some(Operand::IdRef(target_id)) = inst.operands.first() {
                                predecessors
                                    .entry(*target_id)
                                    .or_default()
                                    .insert(block_label);
                            }
                        }
                        Op::BranchConditional => {
                            if let Some(Operand::IdRef(true_id)) = inst.operands.get(1) {
                                predecessors
                                    .entry(*true_id)
                                    .or_default()
                                    .insert(block_label);
                            }
                            if let Some(Operand::IdRef(false_id)) = inst.operands.get(2) {
                                predecessors
                                    .entry(*false_id)
                                    .or_default()
                                    .insert(block_label);
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Now check each block for invalid multiple predecessors
            for block in &func.blocks {
                let block_label = block.label.as_ref().and_then(|l| l.result_id).unwrap_or(0);

                let preds: &HashSet<u32> = match predecessors.get(&block_label) {
                    Some(p) => p,
                    None => continue,
                };

                // Only check if there are 2+ unique predecessors
                if preds.len() < 2 {
                    continue;
                }

                // Loop headers are allowed
                if loop_headers.contains(&block_label) {
                    continue;
                }

                // Merge targets and switch targets are allowed
                if allowed_multi_pred_blocks.contains(&block_label) {
                    continue;
                }

                // This block has multiple predecessors and is not allowed
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                return Err(ValidationError::MaximalReconvergenceMultiplePredecessors {
                    function: func_id,
                    block: block_id,
                }
                .into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// Lifetime Rule
// ============================================================================

/// Validates OpLifetimeStart and OpLifetimeStop instructions.
///
/// Checks:
/// - Pointer operand type must be OpTypePointer
/// - Pointer must be in Function storage class
/// - If size is non-zero, Addresses capability must be declared
pub struct LifetimeRule;

impl ValidationRule for LifetimeRule {
    fn name(&self) -> &'static str {
        "lifetime"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode != Op::LifetimeStart
                        && inst.class.opcode != Op::LifetimeStop
                    {
                        continue;
                    }

                    let opcode = inst.class.opcode;

                    // Get the pointer operand (operand 0)
                    let pointer_id = match inst.operands.first() {
                        Some(Operand::IdRef(id)) => *id,
                        _ => continue,
                    };

                    // Look up the pointer's type instruction
                    if let Some(type_inst) = ResultId::try_from(pointer_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|def| def.result_type)
                        .and_then(|tid| ResultId::try_from(tid).ok())
                        .and_then(|rid| ctx.definitions.get(&rid))
                    {
                        if !type_inst.is_pointer_type() {
                            return Err(
                                ValidationError::LifetimePointerNotTypePointer { opcode }.into()
                            );
                        }
                        if type_inst.pointer_storage_class()
                            != Some(rspirv::spirv::StorageClass::Function)
                        {
                            return Err(ValidationError::LifetimePointerNotFunctionStorageClass {
                                opcode,
                            }
                            .into());
                        }
                    }

                    // Check size (operand 1) - if non-zero, Addresses must be declared
                    if let Some(Operand::LiteralBit32(size)) = inst.operands.get(1) {
                        if *size != 0 && !ctx.has_capability(Capability::Addresses) {
                            return Err(ValidationError::LifetimeNonZeroSizeRequiresAddresses {
                                opcode,
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
// Switch Case Uniqueness Rule
// ============================================================================

/// Validates that OpSwitch case literals are unique.
pub struct SwitchCaseUniquenessRule;

impl ValidationRule for SwitchCaseUniquenessRule {
    fn name(&self) -> &'static str {
        "switch-case-uniqueness"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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

                for inst in &block.instructions {
                    if inst.class.opcode != Op::Switch {
                        continue;
                    }
                    // OpSwitch operands: selector, default, [literal, target]...
                    // Case literals start at index 2, at even indices
                    let mut seen = HashSet::new();
                    let mut i = 2;
                    while i + 1 < inst.operands.len() {
                        let literal = &inst.operands[i];
                        let key = match literal {
                            Operand::LiteralBit32(v) => *v as u64,
                            Operand::LiteralBit64(v) => *v,
                            _ => {
                                i += 2;
                                continue;
                            }
                        };
                        if !seen.insert(key) {
                            return Err(ValidationError::SwitchDuplicateCaseLiteral {
                                function: function_id,
                                block: block_id,
                                literal: key,
                            }
                            .into());
                        }
                        i += 2;
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Switch Selector Type Rule
// ============================================================================

/// Validates that OpSwitch selector operand is an integer type.
pub struct SwitchSelectorTypeRule;

impl ValidationRule for SwitchSelectorTypeRule {
    fn name(&self) -> &'static str {
        "switch-selector-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            for block in &function.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode != Op::Switch {
                        continue;
                    }
                    // OpSwitch operands[0] = selector id
                    let Some(Operand::IdRef(selector_id)) = inst.operands.first() else {
                        continue;
                    };
                    let selector_rid = match ResultId::try_from(*selector_id) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    // Look up the instruction that defines the selector
                    let Some(selector_def) = ctx.definitions.get(&selector_rid) else {
                        continue;
                    };
                    // Get the selector's result type
                    let Some(type_id) = selector_def.result_type else {
                        continue;
                    };
                    let type_rid = match ResultId::try_from(type_id) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    let Some(type_inst) = ctx.definitions.get(&type_rid) else {
                        continue;
                    };
                    if type_inst.class.opcode != Op::TypeInt {
                        return Err(ValidationError::SwitchSelectorNotInteger {
                            selector_id: to_id(*selector_id),
                            found_opcode: type_inst.class.opcode,
                        }
                        .into());
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
static BRANCH_CONDITIONAL_RULE: BranchConditionalRule = BranchConditionalRule;
static MAXIMAL_RECONVERGENCE_PREDECESSORS_RULE: MaximalReconvergencePredecessorsRule =
    MaximalReconvergencePredecessorsRule;
static LIFETIME_RULE: LifetimeRule = LifetimeRule;
static SWITCH_CASE_UNIQUENESS_RULE: SwitchCaseUniquenessRule = SwitchCaseUniquenessRule;
static SWITCH_SELECTOR_TYPE_RULE: SwitchSelectorTypeRule = SwitchSelectorTypeRule;

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
        &BRANCH_CONDITIONAL_RULE,
        &MAXIMAL_RECONVERGENCE_PREDECESSORS_RULE,
        &LIFETIME_RULE,
        &SWITCH_CASE_UNIQUENESS_RULE,
        &SWITCH_SELECTOR_TYPE_RULE,
    ]
}
