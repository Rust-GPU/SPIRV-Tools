//! Control Flow Graph analysis utilities.
//!
//! This module provides reusable CFG analysis for SPIR-V validation,
//! including dominator computation, predecessor/successor tracking,
//! and block reachability analysis.

use std::collections::{HashMap, HashSet};

use rspirv::dr::{Block, Function, Instruction, Operand};
use rspirv::spirv::Op;

use super::types::Id;

/// Control Flow Graph for a single SPIR-V function.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Entry block ID.
    pub entry: Id,
    /// All block IDs in the function.
    pub blocks: HashSet<Id>,
    /// Map from block ID to its predecessors.
    pub predecessors: HashMap<Id, HashSet<Id>>,
    /// Map from block ID to its successors.
    pub successors: HashMap<Id, HashSet<Id>>,
    /// Dominator sets: for each block, the set of blocks that dominate it.
    pub dominators: HashMap<Id, HashSet<Id>>,
    /// Blocks that are reachable from the entry block.
    pub reachable: HashSet<Id>,
}

/// Helper to convert a raw u32 ID to our Id wrapper type.
fn to_id(raw: u32) -> Option<Id> {
    Id::try_from(raw).ok()
}

/// Extract block label ID from a block.
pub fn get_block_label(block: &Block) -> Option<Id> {
    block
        .label
        .as_ref()
        .and_then(|inst| inst.result_id)
        .and_then(to_id)
}

/// Extract branch targets from a terminator instruction.
fn get_branch_targets(terminator: &Instruction) -> Vec<Id> {
    let mut targets = Vec::new();

    match terminator.class.opcode {
        Op::Branch => {
            if let Some(Operand::IdRef(target)) = terminator.operands.first() {
                if let Some(id) = to_id(*target) {
                    targets.push(id);
                }
            }
        }
        Op::BranchConditional => {
            // Skip condition (operand 0), get true and false targets
            for op in terminator.operands.iter().skip(1).take(2) {
                if let Operand::IdRef(target) = op {
                    if let Some(id) = to_id(*target) {
                        targets.push(id);
                    }
                }
            }
        }
        Op::Switch => {
            // Operands: selector, default target, then pairs of (literal, target)
            for (index, op) in terminator.operands.iter().enumerate() {
                if index == 0 {
                    continue; // skip selector
                }
                // Default target is at index 1, then case targets are at even indices (2, 4, 6, ...)
                if index == 1 || index % 2 == 0 {
                    if let Operand::IdRef(target) = op {
                        if let Some(id) = to_id(*target) {
                            if !targets.contains(&id) {
                                targets.push(id);
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    targets
}

impl ControlFlowGraph {
    /// Build a control flow graph from a SPIR-V function.
    pub fn build(function: &Function) -> Option<Self> {
        let entry_block = function.blocks.first()?;
        let entry = get_block_label(entry_block)?;

        // Collect all block IDs
        let blocks: HashSet<Id> = function
            .blocks
            .iter()
            .filter_map(get_block_label)
            .collect();

        // Initialize predecessor/successor maps
        let mut predecessors: HashMap<Id, HashSet<Id>> = blocks
            .iter()
            .copied()
            .map(|id| (id, HashSet::new()))
            .collect();
        let mut successors: HashMap<Id, HashSet<Id>> = blocks
            .iter()
            .copied()
            .map(|id| (id, HashSet::new()))
            .collect();

        // Build edges from terminators
        for block in &function.blocks {
            let Some(block_id) = get_block_label(block) else {
                continue;
            };

            // Find terminator (last instruction that's a block terminator)
            let terminator = block
                .instructions
                .iter()
                .find(|inst| rspirv::grammar::reflect::is_block_terminator(inst.class.opcode));

            if let Some(term) = terminator {
                for target in get_branch_targets(term) {
                    if blocks.contains(&target) {
                        if let Some(preds) = predecessors.get_mut(&target) {
                            preds.insert(block_id);
                        }
                        if let Some(succs) = successors.get_mut(&block_id) {
                            succs.insert(target);
                        }
                    }
                }
            }
        }

        // Compute reachable blocks
        let reachable = Self::compute_reachable(entry, &successors);

        // Compute dominators
        let dominators = Self::compute_dominators(entry, &blocks, &predecessors);

        Some(Self {
            entry,
            blocks,
            predecessors,
            successors,
            dominators,
            reachable,
        })
    }

    /// Compute reachable blocks from entry using BFS.
    fn compute_reachable(entry: Id, successors: &HashMap<Id, HashSet<Id>>) -> HashSet<Id> {
        let mut reachable = HashSet::new();
        let mut worklist = vec![entry];

        while let Some(block) = worklist.pop() {
            if reachable.insert(block) {
                if let Some(succs) = successors.get(&block) {
                    for succ in succs {
                        if !reachable.contains(succ) {
                            worklist.push(*succ);
                        }
                    }
                }
            }
        }

        reachable
    }

    /// Compute dominators using iterative dataflow algorithm.
    ///
    /// For each block B, dominators[B] contains all blocks that dominate B.
    /// A block D dominates B if every path from entry to B goes through D.
    fn compute_dominators(
        entry: Id,
        blocks: &HashSet<Id>,
        predecessors: &HashMap<Id, HashSet<Id>>,
    ) -> HashMap<Id, HashSet<Id>> {
        let mut dominators: HashMap<Id, HashSet<Id>> = HashMap::new();

        // Initialize: entry is dominated only by itself,
        // all other blocks are dominated by all blocks (will be refined)
        for id in blocks {
            let mut set: HashSet<Id> = if *id == entry {
                HashSet::new()
            } else {
                blocks.clone()
            };
            set.insert(*id);
            dominators.insert(*id, set);
        }

        // Iterate until fixed point
        let mut changed = true;
        while changed {
            changed = false;

            for block in blocks {
                if *block == entry {
                    continue;
                }

                let preds = predecessors.get(block).cloned().unwrap_or_default();
                if preds.is_empty() {
                    // Unreachable block - skip
                    continue;
                }

                // New dominators = intersection of all predecessors' dominators + self
                let mut new_dom: HashSet<Id> = blocks.clone();
                for pred in preds {
                    if let Some(pred_dom) = dominators.get(&pred) {
                        new_dom = new_dom.intersection(pred_dom).copied().collect();
                    }
                }
                new_dom.insert(*block);

                if new_dom != *dominators.get(block).unwrap_or(&HashSet::new()) {
                    dominators.insert(*block, new_dom);
                    changed = true;
                }
            }
        }

        dominators
    }

    /// Check if block `dominator` dominates block `block`.
    pub fn dominates(&self, dominator: Id, block: Id) -> bool {
        if dominator == block {
            return true;
        }
        self.dominators
            .get(&block)
            .map(|doms| doms.contains(&dominator))
            .unwrap_or(false)
    }

    /// Check if a block is reachable from entry.
    pub fn is_reachable(&self, block: Id) -> bool {
        self.reachable.contains(&block)
    }

    /// Get predecessors of a block.
    pub fn get_predecessors(&self, block: Id) -> Option<&HashSet<Id>> {
        self.predecessors.get(&block)
    }

    /// Get successors of a block.
    pub fn get_successors(&self, block: Id) -> Option<&HashSet<Id>> {
        self.successors.get(&block)
    }

    /// Check if entry block has any predecessors (invalid in SPIR-V).
    pub fn entry_has_predecessors(&self) -> bool {
        self.predecessors
            .get(&self.entry)
            .map(|p| !p.is_empty())
            .unwrap_or(false)
    }
}

/// Merge instruction information extracted from a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeInfo {
    /// OpSelectionMerge with merge block target.
    Selection {
        /// The merge block ID.
        merge_block: Id,
    },
    /// OpLoopMerge with merge block and continue target.
    Loop {
        /// The merge block ID.
        merge_block: Id,
        /// The continue target block ID.
        continue_target: Id,
    },
}

/// Extract merge instruction from a block, if present.
pub fn get_merge_info(block: &Block) -> Option<MergeInfo> {
    for inst in &block.instructions {
        match inst.class.opcode {
            Op::SelectionMerge => {
                if let Some(Operand::IdRef(merge)) = inst.operands.first() {
                    if let Some(merge_block) = to_id(*merge) {
                        return Some(MergeInfo::Selection { merge_block });
                    }
                }
            }
            Op::LoopMerge => {
                let merge = inst.operands.first().and_then(|op| {
                    if let Operand::IdRef(id) = op {
                        to_id(*id)
                    } else {
                        None
                    }
                });
                let continue_target = inst.operands.get(1).and_then(|op| {
                    if let Operand::IdRef(id) = op {
                        to_id(*id)
                    } else {
                        None
                    }
                });
                if let (Some(merge_block), Some(continue_target)) = (merge, continue_target) {
                    return Some(MergeInfo::Loop {
                        merge_block,
                        continue_target,
                    });
                }
            }
            _ => {}
        }
    }
    None
}

/// Get the terminator instruction from a block.
pub fn get_terminator(block: &Block) -> Option<&Instruction> {
    block
        .instructions
        .iter()
        .find(|inst| rspirv::grammar::reflect::is_block_terminator(inst.class.opcode))
}

/// Check if an opcode is a terminator.
pub fn is_terminator(opcode: Op) -> bool {
    rspirv::grammar::reflect::is_block_terminator(opcode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_targets_extraction() {
        // Test Branch
        let mut branch = rspirv::dr::Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(10)],
        );
        let targets = get_branch_targets(&branch);
        assert_eq!(targets.len(), 1);

        // Test BranchConditional
        let branch_cond = rspirv::dr::Instruction::new(
            Op::BranchConditional,
            None,
            None,
            vec![
                Operand::IdRef(5),  // condition
                Operand::IdRef(10), // true target
                Operand::IdRef(20), // false target
            ],
        );
        let targets = get_branch_targets(&branch_cond);
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_dominator_computation_simple() {
        // Simple linear CFG: entry -> B1 -> B2
        let entry = Id::try_from(1u32).unwrap();
        let b1 = Id::try_from(2u32).unwrap();
        let b2 = Id::try_from(3u32).unwrap();

        let blocks: HashSet<Id> = [entry, b1, b2].into_iter().collect();
        let mut predecessors: HashMap<Id, HashSet<Id>> = HashMap::new();
        predecessors.insert(entry, HashSet::new());
        predecessors.insert(b1, [entry].into_iter().collect());
        predecessors.insert(b2, [b1].into_iter().collect());

        let dominators = ControlFlowGraph::compute_dominators(entry, &blocks, &predecessors);

        // entry dominates itself
        assert!(dominators.get(&entry).unwrap().contains(&entry));
        // entry dominates b1
        assert!(dominators.get(&b1).unwrap().contains(&entry));
        // entry dominates b2
        assert!(dominators.get(&b2).unwrap().contains(&entry));
        // b1 dominates b2
        assert!(dominators.get(&b2).unwrap().contains(&b1));
    }

    #[test]
    fn test_dominator_computation_diamond() {
        // Diamond CFG:
        //      entry
        //      /   \
        //     B1   B2
        //      \   /
        //       B3
        let entry = Id::try_from(1u32).unwrap();
        let b1 = Id::try_from(2u32).unwrap();
        let b2 = Id::try_from(3u32).unwrap();
        let b3 = Id::try_from(4u32).unwrap();

        let blocks: HashSet<Id> = [entry, b1, b2, b3].into_iter().collect();
        let mut predecessors: HashMap<Id, HashSet<Id>> = HashMap::new();
        predecessors.insert(entry, HashSet::new());
        predecessors.insert(b1, [entry].into_iter().collect());
        predecessors.insert(b2, [entry].into_iter().collect());
        predecessors.insert(b3, [b1, b2].into_iter().collect());

        let dominators = ControlFlowGraph::compute_dominators(entry, &blocks, &predecessors);

        // entry dominates all
        assert!(dominators.get(&b1).unwrap().contains(&entry));
        assert!(dominators.get(&b2).unwrap().contains(&entry));
        assert!(dominators.get(&b3).unwrap().contains(&entry));

        // b1 does NOT dominate b3 (can reach b3 through b2)
        assert!(!dominators.get(&b3).unwrap().contains(&b1));
        // b2 does NOT dominate b3 (can reach b3 through b1)
        assert!(!dominators.get(&b3).unwrap().contains(&b2));
    }
}
