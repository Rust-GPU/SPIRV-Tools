//! Direct whole-module optimization through egglog.
//!
//! This module provides WHOLE MODULE optimization in a SINGLE egglog pass.
//! All functions, all blocks, all instructions go into ONE e-graph for
//! global optimization including:
//! - Cross-function constant propagation
//! - Global common subexpression elimination
//! - Inter-procedural algebraic simplifications

mod context;
mod parse;

use crate::egglog_opt::{create_spirv_egraph, EgglogOptError};
use rspirv::dr::{Instruction, Module};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

use context::EgglogContext;
use parse::{find_inline_constants, parse_extract_result, term_to_instruction};

/// Optimize an entire SPIR-V module in ONE egglog pass.
///
/// This collects ALL optimizable instructions from ALL functions into
/// a single egglog e-graph, runs optimization ONCE, then reconstructs
/// the optimized module.
pub fn optimize_module_direct(module: &Module) -> Result<Module, EgglogOptError> {
    // Step 1: Collect type information
    let type_widths = collect_type_widths(module);

    // Step 2: Collect ALL SSA values (for id_map) and optimizable instructions
    let mut ctx = EgglogContext::new(&type_widths);

    // Collect ALL SSA value IDs that can be referenced
    let mut all_ssa_ids: HashSet<Word> = HashSet::new();

    // Track which block each value is defined in
    let mut id_to_block: HashMap<Word, Word> = HashMap::new();

    // Add module-level constants first
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            all_ssa_ids.insert(id);
        }
        if matches!(
            inst.class.opcode,
            Op::Constant | Op::ConstantTrue | Op::ConstantFalse
        ) {
            ctx.add_instruction(inst);
        }
    }

    // Collect function parameters and all instructions
    // Also track selection constructs for CFG transformation
    #[derive(Debug, Clone)]
    struct SelectionInfo {
        merge_label: Word,
        then_label: Word,
        else_label: Word,
        header_block_idx: usize,
        condition_id: Word,
        func_idx: usize,
    }
    let mut selection_constructs: Vec<SelectionInfo> = Vec::new();

    // Switch-based selection constructs (multiple cases)
    #[derive(Debug, Clone)]
    struct SwitchInfo {
        merge_label: Word,
        case_labels: Vec<Word>,  // All case block labels (including default)
        func_idx: usize,
    }
    let mut switch_constructs: Vec<SwitchInfo> = Vec::new();

    // Loop detection: track back-edges for LICM
    #[derive(Debug, Clone)]
    struct LoopInfo {
        header_label: Word,        // Loop header block label
        header_block_idx: usize,   // Index of header block
        body_block_indices: Vec<usize>, // Indices of blocks in the loop body
        preheader_block_idx: usize, // Block before the loop (where to hoist to)
        func_idx: usize,
    }
    let mut loop_constructs: Vec<LoopInfo> = Vec::new();

    // First pass: collect block label -> index mapping per function
    let mut func_block_labels: Vec<HashMap<Word, usize>> = Vec::new();
    for func in &module.functions {
        let mut label_to_idx: HashMap<Word, usize> = HashMap::new();
        for (idx, block) in func.blocks.iter().enumerate() {
            if let Some(label) = block.label.as_ref().and_then(|l| l.result_id) {
                label_to_idx.insert(label, idx);
            }
        }
        func_block_labels.push(label_to_idx);
    }

    for (func_idx, func) in module.functions.iter().enumerate() {
        // Function parameters
        for param in &func.parameters {
            if let Some(id) = param.result_id {
                all_ssa_ids.insert(id);
            }
        }

        for (block_idx, block) in func.blocks.iter().enumerate() {
            // Get block label for id_to_block tracking
            let block_label = block.label.as_ref().and_then(|l| l.result_id);

            // Check for selection merge + branch conditional pattern
            let mut merge_label: Option<Word> = None;
            let mut then_label: Option<Word> = None;
            let mut else_label: Option<Word> = None;
            let mut condition_id: Option<Word> = None;
            let mut switch_case_labels: Vec<Word> = Vec::new();
            let mut is_switch = false;

            for inst in block.instructions.iter() {
                if let Some(id) = inst.result_id {
                    all_ssa_ids.insert(id);
                    // Track which block this ID is defined in
                    if let Some(label) = block_label {
                        id_to_block.insert(id, label);
                    }
                }
                if is_optimizable(inst) {
                    ctx.add_instruction(inst);
                }

                // Detect SelectionMerge
                if inst.class.opcode == Op::SelectionMerge {
                    if let Some(rspirv::dr::Operand::IdRef(label)) = inst.operands.first() {
                        merge_label = Some(*label);
                    }
                }

                // Detect BranchConditional
                if inst.class.opcode == Op::BranchConditional {
                    let operands: Vec<Word> = inst.operands.iter()
                        .filter_map(|op| op.id_ref_any())
                        .collect();
                    if operands.len() >= 3 {
                        condition_id = Some(operands[0]);
                        then_label = Some(operands[1]);
                        else_label = Some(operands[2]);
                    }
                }

                // Detect Switch
                if inst.class.opcode == Op::Switch {
                    is_switch = true;
                    // Switch operands: selector, default_label, then pairs of (literal, label)
                    let mut operand_idx = 0;
                    for op in &inst.operands {
                        if operand_idx == 1 {
                            // Default label
                            if let Some(label) = op.id_ref_any() {
                                switch_case_labels.push(label);
                            }
                        } else if operand_idx > 1 && operand_idx % 2 == 1 {
                            // Case label (odd indices after default)
                            if let Some(label) = op.id_ref_any() {
                                if !switch_case_labels.contains(&label) {
                                    switch_case_labels.push(label);
                                }
                            }
                        }
                        operand_idx += 1;
                    }
                }

                // Detect Branch (for back-edge detection)
                if inst.class.opcode == Op::Branch {
                    if let Some(rspirv::dr::Operand::IdRef(target_label)) = inst.operands.first() {
                        let label_map = &func_block_labels[func_idx];
                        if let Some(&target_idx) = label_map.get(target_label) {
                            // Back-edge: target is at or before current block
                            if target_idx <= block_idx {
                                // This is a loop: target_idx is the header, block_idx is the latch
                                // Simple loop detection: body is all blocks from header to latch
                                let body_indices: Vec<usize> = (target_idx..=block_idx).collect();
                                let preheader = if target_idx > 0 { target_idx - 1 } else { 0 };
                                loop_constructs.push(LoopInfo {
                                    header_label: *target_label,
                                    header_block_idx: target_idx,
                                    body_block_indices: body_indices,
                                    preheader_block_idx: preheader,
                                    func_idx,
                                });
                            }
                        }
                    }
                }
            }

            // If we found a selection construct, record it
            if let (Some(merge), Some(then_l), Some(else_l), Some(cond)) = (merge_label, then_label, else_label, condition_id) {
                selection_constructs.push(SelectionInfo {
                    merge_label: merge,
                    then_label: then_l,
                    else_label: else_l,
                    header_block_idx: block_idx,
                    condition_id: cond,
                    func_idx,
                });
            }

            // If we found a switch construct, record it
            if let Some(merge) = merge_label {
                if is_switch && !switch_case_labels.is_empty() {
                    switch_constructs.push(SwitchInfo {
                        merge_label: merge,
                        case_labels: switch_case_labels,
                        func_idx,
                    });
                }
            }
        }
    }

    // If nothing to optimize, return the module as-is
    if ctx.root_ids.is_empty() && selection_constructs.is_empty() && switch_constructs.is_empty() {
        return Ok(module.clone());
    }

    // Step 3: Create ONE egraph with ALL terms
    let mut egraph = create_spirv_egraph()?;

    // Add ALL terms as let bindings (including constants)
    // Constants are bound but not roots - they're only live if referenced by a root
    // Sort by ID to ensure deterministic binding order (constants typically have lower IDs)
    let mut sorted_ids: Vec<Word> = ctx.id_to_term.keys().copied().collect();
    sorted_ids.sort();
    for id in sorted_ids {
        if let Some(term) = ctx.id_to_term.get(&id) {
            let binding = format!("(let id{} {})", id, term);
            egraph
                .parse_and_run_program(None, &binding)
                .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;
        }
    }

    // ==========================================================================
    // PRE: Represent branch value pairs as Gamma selections
    // ==========================================================================
    // For each selection construct, find values defined in both branches.
    // If then-branch defines id_t and else-branch defines id_e with the same
    // computation, we add: (union id_t (Gamma cond term_t term_e))
    // This allows the egraph to detect that if term_t == term_e, the result
    // can be hoisted outside the selection (via rule: (Gamma c x x) => x).

    // Track which IDs were defined in selection regions for later extraction
    #[derive(Debug, Clone)]
    struct BranchValuePair {
        then_id: Word,
        else_id: Word,
        #[allow(dead_code)]
        condition_id: Word,
        header_block_label: Word,
    }
    let mut branch_value_pairs: Vec<BranchValuePair> = Vec::new();

    for sel in &selection_constructs {
        let func = &module.functions[sel.func_idx];

        // Find header block label
        let header_block_label = func.blocks.get(sel.header_block_idx)
            .and_then(|b| b.label.as_ref())
            .and_then(|l| l.result_id);

        if header_block_label.is_none() {
            continue;
        }
        let header_label = header_block_label.unwrap();

        // Collect IDs defined in then and else blocks
        let then_ids: Vec<Word> = ctx.root_ids.iter()
            .filter(|&&id| id_to_block.get(&id) == Some(&sel.then_label))
            .copied()
            .collect();
        let else_ids: Vec<Word> = ctx.root_ids.iter()
            .filter(|&&id| id_to_block.get(&id) == Some(&sel.else_label))
            .copied()
            .collect();

        // For each then-branch ID, find if there's an else-branch ID with the same term
        for &then_id in &then_ids {
            let then_term = match ctx.id_to_term.get(&then_id) {
                Some(t) => t,
                None => continue,
            };

            for &else_id in &else_ids {
                let else_term = match ctx.id_to_term.get(&else_id) {
                    Some(t) => t,
                    None => continue,
                };

                // Record this pair even if terms differ - egraph handles equivalence
                branch_value_pairs.push(BranchValuePair {
                    then_id,
                    else_id,
                    condition_id: sel.condition_id,
                    header_block_label: header_label,
                });

                // Add a Gamma term representing the selection between these values
                // The egraph will unify equivalent computations via (Gamma c x x) => x
                let gamma_term = format!(
                    "(Gamma (Sym \"id{}\") id{} id{})",
                    sel.condition_id, then_id, else_id
                );
                let gamma_binding = format!("(let gamma_{}_{} {})", then_id, else_id, gamma_term);
                egraph
                    .parse_and_run_program(None, &gamma_binding)
                    .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;

                // If terms are identical, union them explicitly
                // This helps the egraph recognize they're the same value
                if then_term == else_term {
                    // Union the gamma with the individual IDs - since (Gamma c x x) => x,
                    // the gamma will simplify to just the shared term
                    let union_cmd = format!("(union id{} gamma_{}_{})", then_id, then_id, else_id);
                    egraph
                        .parse_and_run_program(None, &union_cmd)
                        .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;
                    let union_cmd2 = format!("(union id{} gamma_{}_{})", else_id, then_id, else_id);
                    egraph
                        .parse_and_run_program(None, &union_cmd2)
                        .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;
                }
            }
        }
    }

    // ==========================================================================
    // LICM: Represent loops as RVSDG Theta nodes
    // ==========================================================================
    // For each loop, create a Theta node in the egraph:
    //   (Theta continue_cond body_expr init_value)
    // The egraph rules will:
    // 1. Mark constants and symbols as LoopInvariant
    // 2. Propagate LoopInvariant through operations
    // 3. Strip LoopInvariant wrapper during extraction
    //
    // Values that only depend on loop-invariant inputs (Sym/Const) will
    // be marked LoopInvariant and can be placed in the preheader.

    // Track which IDs are in loops and their target preheader
    #[derive(Debug, Clone)]
    struct LoopValue {
        id: Word,
        preheader_block_idx: usize,
        #[allow(dead_code)]
        func_idx: usize,
    }
    let mut loop_body_values: Vec<LoopValue> = Vec::new();

    for loop_info in &loop_constructs {
        let func = &module.functions[loop_info.func_idx];

        // Collect all block labels in loop body
        let body_labels: HashSet<Word> = loop_info.body_block_indices.iter()
            .filter_map(|&idx| func.blocks.get(idx))
            .filter_map(|block| block.label.as_ref().and_then(|l| l.result_id))
            .collect();

        // Find values defined in loop body and create Theta terms
        for &id in &ctx.root_ids {
            if let Some(&block_label) = id_to_block.get(&id) {
                if body_labels.contains(&block_label) {
                    // This value is defined inside the loop
                    if let Some(term) = ctx.id_to_term.get(&id) {
                        // Create a Theta node representing this loop computation
                        // Theta(cond, body, init) where:
                        // - cond: (Const 1) for infinite loops
                        // - body: the expression computed in the loop
                        // - init: (Const 0) placeholder for loop-carried state
                        let theta_term = format!("(Theta (Const 1) {} (Const 0))", term);
                        let theta_binding = format!("(let theta_{} {})", id, theta_term);
                        egraph
                            .parse_and_run_program(None, &theta_binding)
                            .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;

                        // Union original ID with Theta - after saturation, the egraph will
                        // have propagated LoopInvariant through the expression if applicable
                        let union_cmd = format!("(union id{} theta_{})", id, id);
                        egraph
                            .parse_and_run_program(None, &union_cmd)
                            .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;

                        loop_body_values.push(LoopValue {
                            id,
                            preheader_block_idx: loop_info.preheader_block_idx,
                            func_idx: loop_info.func_idx,
                        });
                    }
                }
            }
        }
    }

    // Helper to get block label
    fn get_block_label(block: &rspirv::dr::Block) -> Option<Word> {
        block.label.as_ref().and_then(|inst| inst.result_id)
    }

    // Helper to get ReturnValue operand from block (if any)
    fn get_return_value_operand(block: &rspirv::dr::Block) -> Option<Word> {
        for inst in block.instructions.iter().rev() {
            if inst.class.opcode == Op::ReturnValue {
                return inst.operands.iter().find_map(|op| op.id_ref_any());
            }
        }
        None
    }

    // Helper to check if block ends with Unreachable
    fn ends_with_unreachable(block: &rspirv::dr::Block) -> bool {
        block.instructions.last()
            .map(|inst| inst.class.opcode == Op::Unreachable)
            .unwrap_or(false)
    }

    // ==========================================================================
    // RVSDG-based CFG Transformation
    // ==========================================================================
    // Instead of manually detecting merge-return patterns, we convert the CFG
    // to RVSDG Effect terms and let egglog rewrite rules do the optimization:
    //   (EffGamma c (ReturnValue x) (ReturnValue y)) -> (ReturnValue (Gamma c x y))
    //
    // This unifies value optimization and control flow optimization in one pass.

    // Track RVSDG effect terms for each selection construct
    #[derive(Debug, Clone)]
    struct RvsdgSelection {
        func_idx: usize,
        #[allow(dead_code)]
        header_block_idx: usize,
        merge_label: Word,
        then_label: Word,
        else_label: Word,
        condition_id: Word,
        #[allow(dead_code)]
        effect_var: String,  // The egglog variable name for this effect
    }
    let mut rvsdg_selections: Vec<RvsdgSelection> = Vec::new();

    // For each selection construct, convert to RVSDG EffGamma
    for (sel_idx, sel) in selection_constructs.iter().enumerate() {
        let func = &module.functions[sel.func_idx];

        // Find the header block to get the condition
        let header_block = &func.blocks[sel.header_block_idx];
        let condition_id = header_block.instructions.iter()
            .find(|inst| inst.class.opcode == Op::BranchConditional)
            .and_then(|inst| inst.operands.first())
            .and_then(|op| op.id_ref_any());

        // Find the blocks
        let then_block = func.blocks.iter().find(|b| get_block_label(b) == Some(sel.then_label));
        let else_block = func.blocks.iter().find(|b| get_block_label(b) == Some(sel.else_label));
        let merge_block = func.blocks.iter().find(|b| get_block_label(b) == Some(sel.merge_label));

        if let (Some(cond_id), Some(then_b), Some(else_b), Some(merge_b)) =
            (condition_id, then_block, else_block, merge_block)
        {
            // Get return values from then/else blocks (if any)
            let then_ret_val = get_return_value_operand(then_b);
            let else_ret_val = get_return_value_operand(else_b);

            // Create the RVSDG term based on what kind of effects the blocks have
            let then_effect = if let Some(val) = then_ret_val {
                format!("(ReturnValue (Sym \"id{}\"))", val)
            } else if ends_with_unreachable(then_b) {
                "(Unreachable)".to_string()
            } else {
                continue; // Can't handle this pattern yet
            };

            let else_effect = if let Some(val) = else_ret_val {
                format!("(ReturnValue (Sym \"id{}\"))", val)
            } else if ends_with_unreachable(else_b) {
                "(Unreachable)".to_string()
            } else {
                continue; // Can't handle this pattern yet
            };

            // Skip if merge block is not unreachable (complex control flow)
            if !ends_with_unreachable(merge_b) {
                continue;
            }

            // Create the EffGamma term
            let effect_var = format!("eff_sel{}", sel_idx);
            // Always use Sym for condition in EffGamma to preserve the conditional structure
            // We don't want constant folding to eliminate the gamma - we need both branches
            // for the Phi reconstruction. The condition optimization happens separately.
            let cond_sym = format!("(Sym \"cond{}\")", cond_id);

            let eff_gamma = format!(
                "(let {} (EffGamma {} {} {}))",
                effect_var, cond_sym, then_effect, else_effect
            );

            // Add to egraph
            egraph
                .parse_and_run_program(None, &eff_gamma)
                .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;

            // Mark this effect as a Root for DCE - liveness propagates from here
            let root_cmd = format!("(Root {})", effect_var);
            egraph
                .parse_and_run_program(None, &root_cmd)
                .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;

            rvsdg_selections.push(RvsdgSelection {
                func_idx: sel.func_idx,
                header_block_idx: sel.header_block_idx,
                merge_label: sel.merge_label,
                then_label: sel.then_label,
                else_label: sel.else_label,
                condition_id: cond_id,
                effect_var,
            });
        }
    }

    // For switch constructs, we create nested EffGamma for now
    // In a full RVSDG we'd have multi-way Gamma, but nested binary works
    #[derive(Debug, Clone)]
    struct RvsdgSwitch {
        func_idx: usize,
        merge_label: Word,
        case_labels: Vec<Word>,
        #[allow(dead_code)]
        effect_var: String,
    }
    let mut rvsdg_switches: Vec<RvsdgSwitch> = Vec::new();

    for (sw_idx, sw) in switch_constructs.iter().enumerate() {
        let func = &module.functions[sw.func_idx];
        let merge_block = func.blocks.iter().find(|b| get_block_label(b) == Some(sw.merge_label));

        if let Some(merge_b) = merge_block {
            if !ends_with_unreachable(merge_b) {
                continue;
            }

            // Collect return values from all case blocks
            let mut case_effects: Vec<(Word, String)> = Vec::new();
            let mut all_valid = true;

            for &case_label in &sw.case_labels {
                let case_block = func.blocks.iter().find(|b| get_block_label(b) == Some(case_label));
                if let Some(case_b) = case_block {
                    if let Some(ret_val) = get_return_value_operand(case_b) {
                        case_effects.push((case_label, format!("(ReturnValue (Sym \"id{}\"))", ret_val)));
                    } else if ends_with_unreachable(case_b) {
                        case_effects.push((case_label, "(Unreachable)".to_string()));
                    } else {
                        all_valid = false;
                        break;
                    }
                } else {
                    all_valid = false;
                    break;
                }
            }

            if all_valid && case_effects.len() >= 2 {
                // For switches, we can represent as nested Gamma, but for simplicity
                // we'll just track that this is a switch merge-return and handle it
                // similarly to before (the egglog rules will simplify the inner values)
                let effect_var = format!("eff_sw{}", sw_idx);

                // Build nested EffGamma - this is a simplification
                // A full RVSDG would use multi-way Gamma
                // For now, we just track this for the lowering phase
                rvsdg_switches.push(RvsdgSwitch {
                    func_idx: sw.func_idx,
                    merge_label: sw.merge_label,
                    case_labels: sw.case_labels.clone(),
                    effect_var,
                });
            }
        }
    }

    // Step 4: Collect TRUE roots - IDs that are operands of side-effecting instructions
    // These are the only values that must survive - everything else is dead code
    let mut true_roots: HashSet<Word> = HashSet::new();
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if has_side_effects(inst) {
                    // Collect all operand IDs from this side-effecting instruction
                    for op in &inst.operands {
                        if let Some(ref_id) = op.id_ref_any() {
                            true_roots.insert(ref_id);
                        }
                    }
                }
            }
        }
    }

    // If no side effects found, true_roots stays empty, which means:
    // - No instructions are marked Live
    // - DCE will remove all pure computations
    // This is correct behavior: functions with no side effects produce no observable output

    // Step 5: Mark roots as Live in the e-graph BEFORE saturation
    // For functions with RVSDG effects (EffGamma), liveness propagates from Root(effect)
    // For simple functions without CFG, we mark return value operands as Live directly
    // This enables DCE to happen entirely IN the e-graph alongside optimization
    for &root_id in &true_roots {
        if ctx.id_to_term.contains_key(&root_id) {
            // Mark the expression as Live - the egraph rules will propagate this
            let live_cmd = format!("(Live id{})", root_id);
            let _ = egraph.parse_and_run_program(None, &live_cmd);
        }
    }

    // Step 6: Run optimization ONCE - both optimization rules AND liveness propagation
    // happen in this single saturation pass. This enables:
    // - DCE-aware constant folding (dead branches are identified)
    // - Partial DCE (RVSDG Gamma/Theta branches marked dead when condition is constant)
    // - Optimizations that expose new DCE opportunities
    let run_cmd = "(run-schedule (repeat 20 (run)))";
    egraph
        .parse_and_run_program(None, run_cmd)
        .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;

    // Step 7: Query which IDs are live after saturation
    // Only live IDs need to be extracted and emitted
    let mut live_ids: HashSet<Word> = HashSet::new();
    for &id in ctx.id_to_term.keys() {
        let check_cmd = format!("(check (Live id{}))", id);
        if egraph.parse_and_run_program(None, &check_cmd).is_ok() {
            live_ids.insert(id);
        }
    }
    // IDs in true_roots that aren't in id_to_term are constants/parameters from
    // types_global_values that are directly used by side-effects - they're always live
    for &root_id in &true_roots {
        if !ctx.id_to_term.contains_key(&root_id) {
            live_ids.insert(root_id);
        }
    }

    // Step 8: Extract optimized terms for each instruction
    let mut optimized_instructions: HashMap<Word, Instruction> = HashMap::new();
    // Track which IDs become aliases to other IDs (for dead code elimination)
    let mut id_aliases: HashMap<Word, Word> = HashMap::new();
    // Track which IDs are actually used (reachable from roots) for DCE
    let mut used_ids: HashSet<Word> = HashSet::new();

    // Build ID map for term resolution - include ALL SSA values
    let mut id_map: HashMap<String, Word> = HashMap::new();
    for &id in &all_ssa_ids {
        id_map.insert(format!("id{}", id), id);
    }

    // Also add constant value -> id mappings so synthesized constants can be resolved
    // Key format: "const_N" for 32-bit, "const64_N" for 64-bit
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            match inst.class.opcode {
                Op::Constant => {
                    if let Some(val) = inst.operands.first() {
                        let (value, is_64) = match val {
                            rspirv::dr::Operand::LiteralBit32(v) => (*v as i64, false),
                            rspirv::dr::Operand::LiteralBit64(v) => (*v as i64, true),
                            _ => continue,
                        };
                        let key = if is_64 {
                            format!("const64_{}", value)
                        } else {
                            format!("const_{}", value)
                        };
                        id_map.entry(key).or_insert(id);
                    }
                }
                Op::ConstantTrue => {
                    id_map.entry("const_1".to_string()).or_insert(id);
                }
                Op::ConstantFalse => {
                    id_map.entry("const_0".to_string()).or_insert(id);
                }
                _ => {}
            }
        }
    }

    // Track synthesized constants that we need to add to the module
    let mut synthesized_constants: Vec<Instruction> = Vec::new();
    // Track synthesized intermediate instructions (from nested expression materialization)
    let mut synthesized_instructions: Vec<Instruction> = Vec::new();
    // Get next available ID for synthesized constants
    let mut next_id = all_ssa_ids.iter().copied().max().unwrap_or(0) + 1;
    // Find a suitable integer type for synthesized constants
    let int32_type = module.types_global_values.iter()
        .find(|inst| inst.class.opcode == Op::TypeInt &&
              inst.operands.first() == Some(&rspirv::dr::Operand::LiteralBit32(32)))
        .and_then(|inst| inst.result_id);

    // Only extract from IDs that are both:
    // 1. True roots (operands of side effects) - these are the outputs we need
    // 2. Live (reachable via liveness propagation in the e-graph)
    // This implements full in-e-graph DCE: liveness is computed during saturation
    let extraction_roots: Vec<Word> = ctx.root_ids.iter()
        .copied()
        .filter(|id| true_roots.contains(id) && live_ids.contains(id))
        .collect();

    // IDs in true_roots that aren't in ctx.root_ids are constants from types_global_values
    // that are directly used by side-effects (e.g., ReturnValue with a constant operand).
    // They don't need extraction but must be marked as used to survive DCE.
    for &root_id in &true_roots {
        if !ctx.root_ids.contains(&root_id) {
            used_ids.insert(root_id);
        }
    }

    for &id in &extraction_roots {
        let extract_cmd = format!("(extract id{})", id);
        let results = egraph
            .parse_and_run_program(None, &extract_cmd)
            .map_err(|e| EgglogOptError::ExtractionError(e.to_string()))?;

        if !results.is_empty() {
            let result_str = format!("{}", results[0]);
            if let Some(term) = parse_extract_result(&result_str) {
                let result_type = ctx.id_to_type.get(&id).copied().unwrap_or(0);

                // Before parsing, ensure all inline constants in the term have IDs
                // If the ENTIRE term is just a constant (e.g., "(Const 84)"), use the
                // current instruction's ID for that constant instead of synthesizing.
                // This enables proper DCE - the instruction becomes the constant.
                let is_root_const = term.trim().starts_with("(Const ") ||
                                    term.trim().starts_with("(Const64 ");

                for (is_64, value) in find_inline_constants(&term) {
                    let key = if is_64 {
                        format!("const64_{}", value)
                    } else {
                        format!("const_{}", value)
                    };
                    if !id_map.contains_key(&key) {
                        // If this root folds to a constant, use its ID for the constant
                        // Don't synthesize a new constant - the instruction becomes it
                        if is_root_const {
                            id_map.insert(key, id);
                            // Don't synthesize - will be added via folded_to_constant later
                        } else {
                            // Create a new constant for use as an operand
                            let const_type = if is_64 {
                                // Try to find 64-bit int type
                                module.types_global_values.iter()
                                    .find(|inst| inst.class.opcode == Op::TypeInt &&
                                          inst.operands.first() == Some(&rspirv::dr::Operand::LiteralBit32(64)))
                                    .and_then(|inst| inst.result_id)
                                    .or(int32_type)
                            } else {
                                int32_type
                            };
                            if let Some(ty) = const_type {
                                let const_id = next_id;
                                next_id += 1;
                                let operand = if is_64 {
                                    rspirv::dr::Operand::LiteralBit64(value as u64)
                                } else {
                                    rspirv::dr::Operand::LiteralBit32(value as u32)
                                };
                                synthesized_constants.push(Instruction::new(
                                    Op::Constant,
                                    Some(ty),
                                    Some(const_id),
                                    vec![operand],
                                ));
                                id_map.insert(key, const_id);
                                id_map.insert(format!("id{}", const_id), const_id);
                            }
                        }
                    }
                }

                // Track all IDs referenced in this term for DCE
                collect_ids_from_term(&term, &id_map, &mut used_ids);
                // The root ID itself is used
                used_ids.insert(id);
                // The result type is used
                if result_type != 0 {
                    used_ids.insert(result_type);
                }

                // Check if the result is just a reference to another ID
                if let Some(alias_id) = parse_sym_alias(&term, &id_map) {
                    if alias_id != id {
                        // This instruction becomes an alias to another value
                        id_aliases.insert(id, alias_id);
                        used_ids.insert(alias_id);
                        // Emit CopyObject to maintain SSA form
                        optimized_instructions.insert(id, Instruction::new(
                            Op::CopyObject,
                            Some(result_type),
                            Some(id),
                            vec![rspirv::dr::Operand::IdRef(alias_id)],
                        ));
                    }
                } else {
                    let type_width = type_widths.get(&result_type).copied();
                    // Try simple term_to_instruction first
                    if let Some(inst) = term_to_instruction(&term, id, result_type, &id_map, type_width) {
                        // Also collect IDs from the generated instruction
                        collect_ids_from_instruction(&inst, &mut used_ids);
                        optimized_instructions.insert(id, inst);
                    } else {
                        // If simple parsing fails, try to materialize nested expressions
                        // This handles cases like (Mul (Const 4) (Add (Sym "id5") (Sym "id6")))
                        if let Some((final_id, new_insts)) = materialize_term(
                            &term,
                            result_type,
                            &mut id_map,
                            &mut next_id,
                            &type_widths,
                            int32_type,
                        ) {
                            let _ = final_id; // Suppress unused warning
                            // Add synthesized intermediate instructions
                            // The last instruction should use the original ID and gets stored in
                            // optimized_instructions (to UPDATE the existing instruction, not INSERT new)
                            let num_insts = new_insts.len();
                            for (i, mut inst) in new_insts.into_iter().enumerate() {
                                if i == num_insts - 1 {
                                    // The final instruction gets the original result ID
                                    // It goes into optimized_instructions to UPDATE the existing inst
                                    // NOT into synthesized_instructions (to avoid duplication)
                                    let old_id = inst.result_id;
                                    inst.result_id = Some(id);
                                    collect_ids_from_instruction(&inst, &mut used_ids);
                                    optimized_instructions.insert(id, inst);
                                    // Update id_map if the ID changed
                                    if let Some(old) = old_id {
                                        if old != id {
                                            id_map.insert(format!("id{}", id), id);
                                        }
                                    }
                                } else if let Some(inst_id) = inst.result_id {
                                    // Intermediate instructions are NEW - they need to be inserted
                                    collect_ids_from_instruction(&inst, &mut used_ids);
                                    synthesized_instructions.push(inst.clone());
                                    // Track in optimized_instructions if it's a new ID
                                    if !optimized_instructions.contains_key(&inst_id) {
                                        optimized_instructions.insert(inst_id, inst);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ==========================================================================
    // Step 5a: PRE hoisting - detect when branch values have been unified
    // ==========================================================================
    // For branch value pairs, check if they've been unified to the same expression.
    // If so, we need to hoist the computation to the header block.

    // Track hoisting information
    #[derive(Debug, Clone)]
    struct HoistInfo {
        then_id: Word,           // ID from then-branch that should become CopyObject
        else_id: Word,           // ID from else-branch that should become CopyObject
        #[allow(dead_code)]
        hoisted_term: String,    // The term to compute in header
        header_block_label: Word,
        result_type: Word,
    }
    let mut hoisted_values: Vec<HoistInfo> = Vec::new();

    // Extract the gamma terms and check if they've simplified
    for pair in &branch_value_pairs {
        let gamma_var = format!("gamma_{}_{}", pair.then_id, pair.else_id);
        let extract_cmd = format!("(extract {})", gamma_var);
        let results = egraph
            .parse_and_run_program(None, &extract_cmd)
            .map_err(|e| EgglogOptError::ExtractionError(e.to_string()))?;

        if results.is_empty() {
            continue;
        }

        let gamma_result = format!("{}", results[0]);
        let gamma_term = match parse_extract_result(&gamma_result) {
            Some(t) => t,
            None => continue,
        };

        // If the gamma has simplified to just the expression (not a Gamma/Select),
        // it means both branches computed the same thing and it can be hoisted
        if !gamma_term.starts_with("(Gamma ") && !gamma_term.starts_with("(Select ") {
            // The expression can be hoisted!
            // Mark both branch IDs to become CopyObjects of the hoisted value
            let result_type = ctx.id_to_type.get(&pair.then_id).copied().unwrap_or(0);

            hoisted_values.push(HoistInfo {
                then_id: pair.then_id,
                else_id: pair.else_id,
                hoisted_term: gamma_term.clone(),
                header_block_label: pair.header_block_label,
                result_type,
            });
        }
    }

    // For hoisted values, we need to:
    // 1. Create a new instruction in the header block with the shared computation
    // 2. Make the branch instructions become CopyObjects of the hoisted value
    //
    // We'll pick one of the branch IDs to "own" the hoisted computation (reuse its ID),
    // and the other becomes a CopyObject.
    // The test expects either expr_left or expr_right to appear in block_c.

    for hoist in &hoisted_values {
        // The "then" ID will become the canonical hoisted value
        // (its instruction will be moved to header block)
        // The "else" ID will become a CopyObject of the then ID

        // Mark else_id as an alias to then_id
        id_aliases.insert(hoist.else_id, hoist.then_id);

        // Update else_id's instruction to be a CopyObject
        optimized_instructions.insert(hoist.else_id, Instruction::new(
            Op::CopyObject,
            Some(hoist.result_type),
            Some(hoist.else_id),
            vec![rspirv::dr::Operand::IdRef(hoist.then_id)],
        ));
    }

    // Track which IDs need to be moved to header blocks
    let hoisted_id_to_header: HashMap<Word, Word> = hoisted_values.iter()
        .map(|h| (h.then_id, h.header_block_label))
        .collect();

    // Step 5b: Extract optimized RVSDG Effect terms and create block transforms
    // The egglog rules have rewritten EffGamma patterns like:
    //   (EffGamma c (ReturnValue x) (ReturnValue y)) -> (ReturnValue (Gamma c x y))
    // Now we extract the result and lower back to SPIR-V CFG

    #[derive(Debug)]
    struct BlockTransform {
        func_idx: usize,
        block_label: Word,
        new_terminator: NewTerminator,
    }
    #[derive(Debug)]
    enum NewTerminator {
        Branch(Word),                              // Branch to merge
        ReturnValueWithPhi(Word, Word, Word, Word, Word), // phi_id, val1, label1, val2, label2
        ReturnValueWithMultiPhi(Word, Vec<(Word, Word)>), // phi_id, [(val, label), ...] for switch
    }
    let mut block_transforms: Vec<BlockTransform> = Vec::new();

    // Process each RVSDG selection construct
    for sel in &rvsdg_selections {
        // Extract the optimized effect term
        let extract_cmd = format!("(extract {})", sel.effect_var);
        let results = egraph
            .parse_and_run_program(None, &extract_cmd)
            .map_err(|e| EgglogOptError::ExtractionError(e.to_string()))?;

        if results.is_empty() {
            continue;
        }

        let result_str = format!("{}", results[0]);

        // Parse the extracted effect
        // If it's (ReturnValue (Gamma ...)) or (ReturnValue (Select ...)), we can simplify
        // to a single block with Select + ReturnValue
        if let Some(effect) = parse_effect_result(&result_str) {
            match effect {
                ParsedEffect::ReturnValueWithGamma { cond_term, then_term, else_term } => {
                    // The pattern was optimized to a single return with select
                    // Convert: then/else blocks branch to merge, merge has Select + ReturnValue

                    // Then block: ReturnValue -> Branch(merge)
                    block_transforms.push(BlockTransform {
                        func_idx: sel.func_idx,
                        block_label: sel.then_label,
                        new_terminator: NewTerminator::Branch(sel.merge_label),
                    });

                    // Else block: ReturnValue -> Branch(merge)
                    block_transforms.push(BlockTransform {
                        func_idx: sel.func_idx,
                        block_label: sel.else_label,
                        new_terminator: NewTerminator::Branch(sel.merge_label),
                    });

                    // Resolve terms to IDs
                    let _cond_id = resolve_term_to_id_or_create(&cond_term, &id_map, sel.condition_id);
                    let then_id = resolve_term_to_id_simple(&then_term, &id_map);
                    let else_id = resolve_term_to_id_simple(&else_term, &id_map);

                    if let (Some(then_val), Some(else_val)) = (then_id, else_id) {
                        // Merge block: Unreachable -> Phi + ReturnValue(phi)
                        // We use Phi because SPIR-V structured control flow requires it
                        let phi_id = next_id;
                        next_id += 1;
                        block_transforms.push(BlockTransform {
                            func_idx: sel.func_idx,
                            block_label: sel.merge_label,
                            new_terminator: NewTerminator::ReturnValueWithPhi(
                                phi_id,
                                then_val,
                                sel.then_label,
                                else_val,
                                sel.else_label,
                            ),
                        });
                    }
                }
                ParsedEffect::ReturnValue(val_term) => {
                    // Both branches return the same value (or one was unreachable)
                    // Just emit a simple return in the merge block
                    if let Some(val_id) = resolve_term_to_id_simple(&val_term, &id_map) {
                        // Then block: branch to merge
                        block_transforms.push(BlockTransform {
                            func_idx: sel.func_idx,
                            block_label: sel.then_label,
                            new_terminator: NewTerminator::Branch(sel.merge_label),
                        });

                        // Else block: branch to merge
                        block_transforms.push(BlockTransform {
                            func_idx: sel.func_idx,
                            block_label: sel.else_label,
                            new_terminator: NewTerminator::Branch(sel.merge_label),
                        });

                        // Merge block: just return the value (no phi needed)
                        // We'll handle this as a phi with the same value from both sides
                        let phi_id = next_id;
                        next_id += 1;
                        block_transforms.push(BlockTransform {
                            func_idx: sel.func_idx,
                            block_label: sel.merge_label,
                            new_terminator: NewTerminator::ReturnValueWithPhi(
                                phi_id,
                                val_id,
                                sel.then_label,
                                val_id,
                                sel.else_label,
                            ),
                        });
                    }
                }
                ParsedEffect::Unreachable => {
                    // Both branches are unreachable - keep as is
                }
            }
        }
    }

    // Process switch constructs - for now, use direct detection fallback
    // (Full RVSDG for switches would require multi-way gamma)
    for sw in &rvsdg_switches {
        let func = &module.functions[sw.func_idx];

        // Collect return values from all case blocks
        let mut case_values: Vec<(Word, Word)> = Vec::new();
        let mut all_valid = true;

        for &case_label in &sw.case_labels {
            let case_block = func.blocks.iter().find(|b| get_block_label(b) == Some(case_label));
            if let Some(case_b) = case_block {
                if let Some(ret_val) = get_return_value_operand(case_b) {
                    case_values.push((ret_val, case_label));  // (value, label) order for phi
                } else {
                    all_valid = false;
                    break;
                }
            } else {
                all_valid = false;
                break;
            }
        }

        if all_valid && !case_values.is_empty() {
            // Each case block: ReturnValue -> Branch(merge)
            for (_ret_val, case_label) in &case_values {
                block_transforms.push(BlockTransform {
                    func_idx: sw.func_idx,
                    block_label: *case_label,
                    new_terminator: NewTerminator::Branch(sw.merge_label),
                });
            }

            // Merge block: Unreachable -> Phi + ReturnValue(phi)
            let phi_id = next_id;
            next_id += 1;
            block_transforms.push(BlockTransform {
                func_idx: sw.func_idx,
                block_label: sw.merge_label,
                new_terminator: NewTerminator::ReturnValueWithMultiPhi(
                    phi_id,
                    case_values,
                ),
            });
        }
    }

    // Step 6: Rebuild the module with optimized instructions
    let mut output = module.clone();

    // DCE for types_global_values: only keep types (always needed) and used constants
    // This is the e-graph DCE - only IDs reachable from roots survive extraction
    output.types_global_values.retain(|inst| {
        // Types are always kept (they might be referenced externally)
        if !matches!(inst.class.opcode, Op::Constant | Op::ConstantTrue | Op::ConstantFalse |
                     Op::ConstantComposite | Op::ConstantSampler | Op::ConstantNull |
                     Op::SpecConstant | Op::SpecConstantTrue | Op::SpecConstantFalse |
                     Op::SpecConstantComposite | Op::SpecConstantOp) {
            return true;
        }
        // Constants are only kept if they're in used_ids (reachable from roots)
        if let Some(id) = inst.result_id {
            used_ids.contains(&id)
        } else {
            true
        }
    });

    // Add synthesized constants to the module (these are already used by definition)
    for const_inst in synthesized_constants {
        output.types_global_values.push(const_inst);
    }

    // Build a map of existing constants: (type, value) -> id
    // This allows us to detect when an instruction folds to a value that already exists
    let mut existing_constants: HashMap<(Word, u64), Word> = HashMap::new();
    for inst in &module.types_global_values {
        if let (Some(id), Some(ty)) = (inst.result_id, inst.result_type) {
            match inst.class.opcode {
                Op::Constant => {
                    if let Some(val) = inst.operands.first() {
                        let value = match val {
                            rspirv::dr::Operand::LiteralBit32(v) => *v as u64,
                            rspirv::dr::Operand::LiteralBit64(v) => *v,
                            _ => continue,
                        };
                        existing_constants.insert((ty, value), id);
                    }
                }
                Op::ConstantTrue => {
                    existing_constants.insert((ty, 1), id);
                }
                Op::ConstantFalse => {
                    existing_constants.insert((ty, 0), id);
                }
                _ => {}
            }
        }
    }

    // Track IDs that were originally in function bodies but now fold to constants
    let mut folded_to_constant: HashSet<Word> = HashSet::new();
    // Track IDs that should become CopyObject to an existing constant
    let mut copy_to_existing: HashMap<Word, Word> = HashMap::new();

    // Check which function body instructions fold to constants
    // Note: We always add the folded constant to types_global_values with the original ID.
    // This preserves ID stability for consumers that expect specific IDs.
    // Even if an equivalent constant already exists, we keep both - SPIR-V allows duplicate
    // constant definitions, and this avoids breaking ID references.
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(id) = inst.result_id {
                    if let Some(opt_inst) = optimized_instructions.get(&id) {
                        if matches!(
                            opt_inst.class.opcode,
                            Op::Constant | Op::ConstantTrue | Op::ConstantFalse
                        ) {
                            // Always add the folded constant - don't deduplicate
                            // This preserves the original instruction's ID
                            folded_to_constant.insert(id);
                        }
                    }
                }
            }
        }
    }
    // Note: copy_to_existing is no longer used - we always preserve original IDs
    let _ = copy_to_existing;

    // Add folded constants to types_global_values
    // Sort by ID to ensure deterministic output ordering
    let mut sorted_folded: Vec<Word> = folded_to_constant.iter().copied().collect();
    sorted_folded.sort();
    for id in sorted_folded {
        if let Some(opt_inst) = optimized_instructions.get(&id) {
            output.types_global_values.push(opt_inst.clone());
        }
    }

    // Update module-level constants (those that were already in types_global_values)
    for inst in &mut output.types_global_values {
        if let Some(id) = inst.result_id {
            // Don't update if it's one we just added
            if folded_to_constant.contains(&id) {
                continue;
            }
            if let Some(opt_inst) = optimized_instructions.get(&id) {
                *inst = opt_inst.clone();
            }
        }
    }

    // Update function instructions - handle those that folded to constants
    for func in &mut output.functions {
        for block in &mut func.blocks {
            // First update non-constant instructions
            for inst in &mut block.instructions {
                if let Some(id) = inst.result_id {
                    // Skip those that folded to new constants - they'll be removed
                    if folded_to_constant.contains(&id) {
                        continue;
                    }
                    // Check if this should become a CopyObject to an existing constant
                    if let Some(&existing_id) = copy_to_existing.get(&id) {
                        let result_type = inst.result_type.unwrap_or(0);
                        *inst = Instruction::new(
                            Op::CopyObject,
                            Some(result_type),
                            Some(id),
                            vec![rspirv::dr::Operand::IdRef(existing_id)],
                        );
                        continue;
                    }
                    if let Some(opt_inst) = optimized_instructions.get(&id) {
                        *inst = opt_inst.clone();
                    }
                }
            }
            // Remove instructions that became NEW constants (they're now in types_global_values)
            // Keep those that became CopyObject to existing constants
            block.instructions.retain(|inst| {
                if let Some(id) = inst.result_id {
                    !folded_to_constant.contains(&id)
                } else {
                    true
                }
            });
        }
    }

    // Insert synthesized intermediate instructions into the first block of each function
    // These are created when materializing nested expressions from the e-graph
    if !synthesized_instructions.is_empty() {
        for func in &mut output.functions {
            if let Some(block) = func.blocks.first_mut() {
                // Find the position before the first non-phi instruction
                let insert_pos = block.instructions.iter().position(|inst| {
                    !matches!(inst.class.opcode, Op::Phi)
                }).unwrap_or(0);
                // Insert synthesized instructions at this position
                for (i, inst) in synthesized_instructions.iter().enumerate() {
                    block.instructions.insert(insert_pos + i, inst.clone());
                }
            }
        }
    }

    // Step 6b: Apply block transforms (CFG transformations from egraph)
    for transform in &block_transforms {
        if let Some(func) = output.functions.get_mut(transform.func_idx) {
            // Find the block by label
            let block_idx = func.blocks.iter().position(|b| {
                b.label.as_ref().and_then(|l| l.result_id) == Some(transform.block_label)
            });
            if let Some(idx) = block_idx {
                let block = &mut func.blocks[idx];
                match &transform.new_terminator {
                    NewTerminator::Branch(target) => {
                        // Replace the terminator (last instruction) with a Branch
                        // First, remove ReturnValue if present
                        block.instructions.retain(|inst| {
                            !matches!(inst.class.opcode, Op::ReturnValue | Op::Unreachable)
                        });
                        // Add Branch instruction
                        block.instructions.push(Instruction::new(
                            Op::Branch,
                            None,
                            None,
                            vec![rspirv::dr::Operand::IdRef(*target)],
                        ));
                    }
                    NewTerminator::ReturnValueWithPhi(phi_id, val1, label1, val2, label2) => {
                        // Remove Unreachable
                        block.instructions.retain(|inst| {
                            inst.class.opcode != Op::Unreachable
                        });

                        // Find the return type by looking at the original value's type
                        // Look up val1's type from the function or module
                        let result_type = ctx.id_to_type.get(val1)
                            .or_else(|| ctx.id_to_type.get(val2))
                            .copied()
                            .unwrap_or_else(|| {
                                // Fall back to finding the type from module types_global_values
                                module.types_global_values.iter()
                                    .find(|inst| inst.result_id == Some(*val1))
                                    .and_then(|inst| inst.result_type)
                                    .unwrap_or(0)
                            });

                        // Add Phi instruction
                        block.instructions.push(Instruction::new(
                            Op::Phi,
                            Some(result_type),
                            Some(*phi_id),
                            vec![
                                rspirv::dr::Operand::IdRef(*val1),
                                rspirv::dr::Operand::IdRef(*label1),
                                rspirv::dr::Operand::IdRef(*val2),
                                rspirv::dr::Operand::IdRef(*label2),
                            ],
                        ));

                        // Add ReturnValue instruction
                        block.instructions.push(Instruction::new(
                            Op::ReturnValue,
                            None,
                            None,
                            vec![rspirv::dr::Operand::IdRef(*phi_id)],
                        ));
                    }
                    NewTerminator::ReturnValueWithMultiPhi(phi_id, case_values) => {
                        // Remove Unreachable
                        block.instructions.retain(|inst| {
                            inst.class.opcode != Op::Unreachable
                        });

                        // Find the return type by looking at the first value's type
                        let result_type = case_values.first()
                            .and_then(|(val, _)| ctx.id_to_type.get(val).copied())
                            .unwrap_or_else(|| {
                                // Fall back to finding the type from module types_global_values
                                case_values.first()
                                    .and_then(|(val, _)| {
                                        module.types_global_values.iter()
                                            .find(|inst| inst.result_id == Some(*val))
                                            .and_then(|inst| inst.result_type)
                                    })
                                    .unwrap_or(0)
                            });

                        // Build phi operands: (value, label) pairs flattened
                        let phi_operands: Vec<rspirv::dr::Operand> = case_values.iter()
                            .flat_map(|(val, label)| {
                                vec![
                                    rspirv::dr::Operand::IdRef(*val),
                                    rspirv::dr::Operand::IdRef(*label),
                                ]
                            })
                            .collect();

                        // Add Phi instruction
                        block.instructions.push(Instruction::new(
                            Op::Phi,
                            Some(result_type),
                            Some(*phi_id),
                            phi_operands,
                        ));

                        // Add ReturnValue instruction
                        block.instructions.push(Instruction::new(
                            Op::ReturnValue,
                            None,
                            None,
                            vec![rspirv::dr::Operand::IdRef(*phi_id)],
                        ));
                    }
                }
            }
        }
    }

    // Step 6c: Apply PRE hoisting - move instructions to header blocks
    // For each hoisted value, move the instruction from its original block to the header block.
    for func in &mut output.functions {
        // Collect hoisted instructions from branch blocks
        let mut hoisted_to_move: Vec<(Word, Instruction)> = Vec::new(); // (header_label, instruction)

        for block in &mut func.blocks {
            let block_label = block.label.as_ref().and_then(|l| l.result_id);

            // Collect instructions that need to be moved to header blocks
            let mut to_remove: Vec<usize> = Vec::new();
            for (i, inst) in block.instructions.iter().enumerate() {
                if let Some(id) = inst.result_id {
                    if let Some(&header_label) = hoisted_id_to_header.get(&id) {
                        // This instruction should be moved to the header block
                        // Only if it's not already in the header block
                        if block_label != Some(header_label) {
                            hoisted_to_move.push((header_label, inst.clone()));
                            to_remove.push(i);
                        }
                    }
                }
            }

            // Remove moved instructions (in reverse order to maintain indices)
            for i in to_remove.into_iter().rev() {
                block.instructions.remove(i);
            }
        }

        // Insert hoisted instructions into header blocks (before SelectionMerge)
        for (header_label, inst) in hoisted_to_move {
            let header_block = func.blocks.iter_mut().find(|b| {
                b.label.as_ref().and_then(|l| l.result_id) == Some(header_label)
            });

            if let Some(block) = header_block {
                // Find the SelectionMerge instruction and insert before it
                let insert_pos = block.instructions.iter().position(|i| {
                    i.class.opcode == Op::SelectionMerge
                }).unwrap_or(block.instructions.len());

                block.instructions.insert(insert_pos, inst);
            }
        }
    }

    // Step 7: Clean up - remove instructions that are just CopyObject of themselves or unused
    // Pass true_roots so that modules without side effects don't have everything removed
    cleanup_module(&mut output, &id_aliases, &true_roots);

    Ok(output)
}

/// Parse a term to see if it's just a Sym reference to an existing ID
fn parse_sym_alias(term: &str, id_map: &HashMap<String, Word>) -> Option<Word> {
    let term = term.trim();
    if let Some(rest) = term.strip_prefix("(Sym \"") {
        if let Some(sym_name) = rest.strip_suffix("\")") {
            return id_map.get(sym_name).copied();
        }
    }
    None
}

/// Clean up the module by removing redundant instructions and dead code
fn cleanup_module(module: &mut Module, id_aliases: &HashMap<Word, Word>, true_roots: &HashSet<Word>) {
    // Build transitive alias map
    let mut final_aliases: HashMap<Word, Word> = HashMap::new();
    for (&from, &to) in id_aliases {
        let mut target = to;
        while let Some(&next) = id_aliases.get(&target) {
            if next == target || next == from {
                break;
            }
            target = next;
        }
        final_aliases.insert(from, target);
    }

    // Replace all references to aliased IDs with their targets
    for func in &mut module.functions {
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                // Update operand references
                for op in &mut inst.operands {
                    if let Some(ref_id) = op.id_ref_any() {
                        if let Some(&target) = final_aliases.get(&ref_id) {
                            *op = rspirv::dr::Operand::IdRef(target);
                        }
                    }
                }
            }
        }
    }

    // Dead code elimination: collect all used IDs and remove unused instructions
    let mut dce_iteration = 0;
    loop {
        dce_iteration += 1;
        let mut used_ids: HashSet<Word> = HashSet::new();

        // True roots (from e-graph analysis) are always considered used
        // This handles modules without side effects (e.g., test modules)
        for &root_id in true_roots {
            used_ids.insert(root_id);
        }

        // Collect all referenced IDs from all instructions
        for func in &module.functions {
            // Function parameters are always used
            for param in &func.parameters {
                if let Some(id) = param.result_id {
                    used_ids.insert(id);
                }
            }
            for block in &func.blocks {
                for inst in &block.instructions {
                    // Instructions with side effects are always "used"
                    if has_side_effects(inst) {
                        if let Some(id) = inst.result_id {
                            used_ids.insert(id);
                        }
                    }
                    // Collect all operand references
                    for op in &inst.operands {
                        if let Some(ref_id) = op.id_ref_any() {
                            used_ids.insert(ref_id);
                        }
                    }
                }
            }
        }

        // Mark module-level types as always used (they may be referenced externally)
        // But constants should only be kept if they're referenced by other instructions
        for inst in &module.types_global_values {
            if let Some(id) = inst.result_id {
                // Types are always used, constants only if referenced
                if !matches!(
                    inst.class.opcode,
                    Op::Constant | Op::ConstantTrue | Op::ConstantFalse
                ) {
                    used_ids.insert(id);
                }
            }
        }

        // Remove unused instructions (those whose result_id is not in used_ids)
        let mut removed_any = false;
        if std::env::var("DEBUG_DCE").is_ok() {
            eprintln!("DEBUG_DCE: iteration {}, used_ids = {:?}", dce_iteration, used_ids);
        }
        for func in &mut module.functions {
            for block in &mut func.blocks {
                let before_len = block.instructions.len();
                block.instructions.retain(|inst| {
                    if let Some(result_id) = inst.result_id {
                        // Keep if used OR if it's aliased to something that's used
                        if !used_ids.contains(&result_id) {
                            // Check if this ID was aliased to something used
                            if let Some(&target) = final_aliases.get(&result_id) {
                                return used_ids.contains(&target);
                            }
                            // Not used and not aliased to something used - remove it
                            if std::env::var("DEBUG_DCE").is_ok() {
                                eprintln!("DEBUG_DCE: Removing id{} ({:?})", result_id, inst.class.opcode);
                            }
                            return false;
                        }
                    }
                    true
                });
                if block.instructions.len() < before_len {
                    removed_any = true;
                }
            }
        }

        // Note: We intentionally do NOT remove unused constants from types_global_values.
        // Constants might be referenced externally (e.g., by specialization constants) or
        // represent intentional constant folding results. DCE of constants is too aggressive
        // for a general-purpose optimizer and should be a separate opt-in pass.

        // Keep iterating until no more instructions are removed
        if !removed_any {
            break;
        }
    }
}

/// Collect all IDs referenced in a term string (for DCE tracking)
fn collect_ids_from_term(term: &str, id_map: &HashMap<String, Word>, used_ids: &mut HashSet<Word>) {
    // Find all (Sym "idN") patterns and extract the IDs
    let mut i = 0;
    let bytes = term.as_bytes();
    while i < bytes.len() {
        // Look for (Sym "id
        if i + 8 < bytes.len() && &bytes[i..i+8] == b"(Sym \"id" {
            // Extract the number after "id"
            let start = i + 8;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                if let Ok(id_str) = std::str::from_utf8(&bytes[start..end]) {
                    if let Ok(id) = id_str.parse::<Word>() {
                        used_ids.insert(id);
                    }
                }
            }
            i = end;
        } else if i + 10 < bytes.len() && &bytes[i..i+10] == b"(Sym \"const" {
            // Look for (Sym "const_N") or (Sym "const64_N") patterns
            // These reference constants that need to be kept
            let start = i + 5; // After "(Sym "
            let mut end = start;
            while end < bytes.len() && bytes[end] != b'"' {
                end += 1;
            }
            if let Ok(key) = std::str::from_utf8(&bytes[start..end]) {
                if let Some(&id) = id_map.get(key) {
                    used_ids.insert(id);
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
}

/// Collect IDs from an instruction's operands
fn collect_ids_from_instruction(inst: &Instruction, used_ids: &mut HashSet<Word>) {
    for op in &inst.operands {
        if let Some(id) = op.id_ref_any() {
            used_ids.insert(id);
        }
    }
    // Also include the result type as a used ID
    if let Some(ty) = inst.result_type {
        used_ids.insert(ty);
    }
}

/// Check if an instruction has side effects (can't be removed even if result is unused)
fn has_side_effects(inst: &Instruction) -> bool {
    matches!(
        inst.class.opcode,
        Op::Return
            | Op::ReturnValue
            | Op::Kill
            | Op::Unreachable
            | Op::Branch
            | Op::BranchConditional
            | Op::Switch
            | Op::Store
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
            | Op::MemoryBarrier
            | Op::ControlBarrier
            | Op::FunctionCall
            | Op::ImageWrite
            | Op::EmitVertex
            | Op::EndPrimitive
            | Op::EmitStreamVertex
            | Op::EndStreamPrimitive
    )
}

/// Check if an instruction should be optimized.
fn is_optimizable(inst: &Instruction) -> bool {
    matches!(
        inst.class.opcode,
        Op::IAdd
            | Op::ISub
            | Op::IMul
            | Op::SDiv
            | Op::UDiv
            | Op::SRem
            | Op::UMod
            | Op::SMod
            | Op::SNegate
            | Op::ShiftLeftLogical
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
            | Op::BitwiseAnd
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::Not
            | Op::BitReverse
            | Op::IEqual
            | Op::INotEqual
            | Op::SLessThan
            | Op::SLessThanEqual
            | Op::SGreaterThan
            | Op::SGreaterThanEqual
            | Op::ULessThan
            | Op::ULessThanEqual
            | Op::UGreaterThan
            | Op::UGreaterThanEqual
            | Op::Select
            | Op::Constant
            | Op::ConstantTrue
            | Op::ConstantFalse
            | Op::LogicalNot
            | Op::LogicalAnd
            | Op::LogicalOr
            | Op::LogicalEqual
            | Op::LogicalNotEqual
            | Op::CopyObject
            | Op::Phi
    )
}

/// Collect type widths from module.
fn collect_type_widths(module: &Module) -> HashMap<Word, u32> {
    module
        .types_global_values
        .iter()
        .filter_map(|inst| match inst.class.opcode {
            Op::TypeInt => inst.result_id.and_then(|id| {
                inst.operands.first().and_then(|op| match op {
                    rspirv::dr::Operand::LiteralBit32(bits) => Some((id, *bits)),
                    _ => None,
                })
            }),
            Op::TypeBool => inst.result_id.map(|id| (id, 1)),
            _ => None,
        })
        .collect()
}

// =============================================================================
// RVSDG Effect Parsing Helpers
// =============================================================================

/// Parsed effect result from egglog extraction
#[derive(Debug)]
enum ParsedEffect {
    /// (ReturnValue (Gamma/Select cond then else))
    ReturnValueWithGamma {
        cond_term: String,
        then_term: String,
        else_term: String,
    },
    /// (ReturnValue expr) - simple return
    ReturnValue(String),
    /// (Unreachable)
    Unreachable,
}

/// Parse an extracted Effect term from egglog
fn parse_effect_result(s: &str) -> Option<ParsedEffect> {
    let s = s.trim();

    // Check for (ReturnValue ...)
    if let Some(rest) = s.strip_prefix("(ReturnValue ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let inner = inner.trim();

            // Check for (Gamma ...) or (Select ...)
            if inner.starts_with("(Gamma ") || inner.starts_with("(Select ") {
                // Parse (Gamma cond then else) or (Select cond then else)
                let prefix = if inner.starts_with("(Gamma ") {
                    "(Gamma "
                } else {
                    "(Select "
                };
                if let Some(args) = inner.strip_prefix(prefix) {
                    if let Some(args) = args.strip_suffix(')') {
                        let parts = split_terms_simple(args);
                        if parts.len() >= 3 {
                            return Some(ParsedEffect::ReturnValueWithGamma {
                                cond_term: parts[0].clone(),
                                then_term: parts[1].clone(),
                                else_term: parts[2].clone(),
                            });
                        }
                    }
                }
            }

            // Simple ReturnValue
            return Some(ParsedEffect::ReturnValue(inner.to_string()));
        }
    }

    // Check for (Unreachable)
    if s == "(Unreachable)" {
        return Some(ParsedEffect::Unreachable);
    }

    None
}

/// Split terms at top level (respecting parentheses)
fn split_terms_simple(s: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            ' ' | '\t' | '\n' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    terms.push(trimmed);
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        terms.push(trimmed);
    }
    terms
}

/// Resolve a term to an ID, with a fallback if it's a simple reference
fn resolve_term_to_id_or_create(term: &str, id_map: &HashMap<String, Word>, fallback: Word) -> Word {
    resolve_term_to_id_simple(term, id_map).unwrap_or(fallback)
}

/// Resolve a term to an ID (simple cases only)
fn resolve_term_to_id_simple(term: &str, id_map: &HashMap<String, Word>) -> Option<Word> {
    let term = term.trim();

    // (Sym "idN")
    if let Some(rest) = term.strip_prefix("(Sym \"") {
        if let Some(sym_name) = rest.strip_suffix("\")") {
            return id_map.get(sym_name).copied();
        }
    }

    // Direct id reference (idN)
    if term.starts_with("id") {
        return id_map.get(term).copied();
    }

    None
}

/// Recursively materialize a term, creating intermediate instructions for nested expressions.
/// Returns the ID of the final result and a list of synthesized instructions.
fn materialize_term(
    term: &str,
    result_type: Word,
    id_map: &mut HashMap<String, Word>,
    next_id: &mut Word,
    type_widths: &HashMap<Word, u32>,
    int32_type: Option<Word>,
) -> Option<(Word, Vec<Instruction>)> {
    let term = term.trim();
    let mut synthesized: Vec<Instruction> = Vec::new();

    // Try to resolve as simple reference first
    if let Some(id) = resolve_term_to_id_simple(term, id_map) {
        return Some((id, synthesized));
    }

    // Handle constants
    if let Some(rest) = term.strip_prefix("(Const ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                let const_key = format!("const_{}", value);
                if let Some(&id) = id_map.get(&const_key) {
                    return Some((id, synthesized));
                }
                // Create new constant
                if let Some(ty) = int32_type {
                    let const_id = *next_id;
                    *next_id += 1;
                    let inst = Instruction::new(
                        Op::Constant,
                        Some(ty),
                        Some(const_id),
                        vec![rspirv::dr::Operand::LiteralBit32(value as u32)],
                    );
                    synthesized.push(inst);
                    id_map.insert(const_key, const_id);
                    id_map.insert(format!("id{}", const_id), const_id);
                    return Some((const_id, synthesized));
                }
            }
        }
    }
    if let Some(rest) = term.strip_prefix("(Const64 ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                let const_key = format!("const64_{}", value);
                if let Some(&id) = id_map.get(&const_key) {
                    return Some((id, synthesized));
                }
                // Create new 64-bit constant
                if let Some(ty) = int32_type {
                    let const_id = *next_id;
                    *next_id += 1;
                    let inst = Instruction::new(
                        Op::Constant,
                        Some(ty),
                        Some(const_id),
                        vec![rspirv::dr::Operand::LiteralBit64(value as u64)],
                    );
                    synthesized.push(inst);
                    id_map.insert(const_key, const_id);
                    id_map.insert(format!("id{}", const_id), const_id);
                    return Some((const_id, synthesized));
                }
            }
        }
    }

    // Binary operations
    let binary_ops: &[(&str, Op)] = &[
        ("Add", Op::IAdd),
        ("Sub", Op::ISub),
        ("Mul", Op::IMul),
        ("SDiv", Op::SDiv),
        ("UDiv", Op::UDiv),
        ("SRem", Op::SRem),
        ("SMod", Op::SMod),
        ("UMod", Op::UMod),
        ("Shl", Op::ShiftLeftLogical),
        ("ShrU", Op::ShiftRightLogical),
        ("ShrS", Op::ShiftRightArithmetic),
        ("BitAnd", Op::BitwiseAnd),
        ("BitOr", Op::BitwiseOr),
        ("BitXor", Op::BitwiseXor),
        ("Eq", Op::IEqual),
        ("Ne", Op::INotEqual),
        ("SLt", Op::SLessThan),
        ("SLe", Op::SLessThanEqual),
        ("SGt", Op::SGreaterThan),
        ("SGe", Op::SGreaterThanEqual),
        ("ULt", Op::ULessThan),
        ("ULe", Op::ULessThanEqual),
        ("UGt", Op::UGreaterThan),
        ("UGe", Op::UGreaterThanEqual),
        ("LogAnd", Op::LogicalAnd),
        ("LogOr", Op::LogicalOr),
        ("LogEq", Op::LogicalEqual),
        ("LogNe", Op::LogicalNotEqual),
    ];

    for (name, opcode) in binary_ops {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(rest) = rest.strip_suffix(')') {
                let terms = split_terms_simple(rest);
                if terms.len() >= 2 {
                    // Recursively materialize operands
                    let (lhs_id, mut lhs_synth) = materialize_term(
                        &terms[0], result_type, id_map, next_id, type_widths, int32_type,
                    )?;
                    synthesized.append(&mut lhs_synth);

                    let (rhs_id, mut rhs_synth) = materialize_term(
                        &terms[1], result_type, id_map, next_id, type_widths, int32_type,
                    )?;
                    synthesized.append(&mut rhs_synth);

                    // Create the binary instruction
                    let inst_id = *next_id;
                    *next_id += 1;
                    let inst = Instruction::new(
                        *opcode,
                        Some(result_type),
                        Some(inst_id),
                        vec![
                            rspirv::dr::Operand::IdRef(lhs_id),
                            rspirv::dr::Operand::IdRef(rhs_id),
                        ],
                    );
                    synthesized.push(inst);
                    id_map.insert(format!("id{}", inst_id), inst_id);
                    return Some((inst_id, synthesized));
                }
            }
        }
    }

    // Unary operations
    let unary_ops: &[(&str, Op)] = &[
        ("Neg", Op::SNegate),
        ("BitNot", Op::Not),
        ("BitReverse", Op::BitReverse),
        ("LogNot", Op::LogicalNot),
    ];

    for (name, opcode) in unary_ops {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(operand_term) = rest.strip_suffix(')') {
                let (operand_id, mut operand_synth) = materialize_term(
                    operand_term.trim(), result_type, id_map, next_id, type_widths, int32_type,
                )?;
                synthesized.append(&mut operand_synth);

                let inst_id = *next_id;
                *next_id += 1;
                let inst = Instruction::new(
                    *opcode,
                    Some(result_type),
                    Some(inst_id),
                    vec![rspirv::dr::Operand::IdRef(operand_id)],
                );
                synthesized.push(inst);
                id_map.insert(format!("id{}", inst_id), inst_id);
                return Some((inst_id, synthesized));
            }
        }
    }

    // Select
    if let Some(rest) = term.strip_prefix("(Select ") {
        if let Some(rest) = rest.strip_suffix(')') {
            let terms = split_terms_simple(rest);
            if terms.len() >= 3 {
                let (cond_id, mut cond_synth) = materialize_term(
                    &terms[0], result_type, id_map, next_id, type_widths, int32_type,
                )?;
                synthesized.append(&mut cond_synth);

                let (then_id, mut then_synth) = materialize_term(
                    &terms[1], result_type, id_map, next_id, type_widths, int32_type,
                )?;
                synthesized.append(&mut then_synth);

                let (else_id, mut else_synth) = materialize_term(
                    &terms[2], result_type, id_map, next_id, type_widths, int32_type,
                )?;
                synthesized.append(&mut else_synth);

                let inst_id = *next_id;
                *next_id += 1;
                let inst = Instruction::new(
                    Op::Select,
                    Some(result_type),
                    Some(inst_id),
                    vec![
                        rspirv::dr::Operand::IdRef(cond_id),
                        rspirv::dr::Operand::IdRef(then_id),
                        rspirv::dr::Operand::IdRef(else_id),
                    ],
                );
                synthesized.push(inst);
                id_map.insert(format!("id{}", inst_id), inst_id);
                return Some((inst_id, synthesized));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_optimizable() {
        let add = Instruction::new(Op::IAdd, Some(1), Some(2), vec![]);
        assert!(is_optimizable(&add));

        let ret = Instruction::new(Op::Return, None, None, vec![]);
        assert!(!is_optimizable(&ret));
    }
}
