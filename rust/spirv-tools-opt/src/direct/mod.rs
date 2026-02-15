//! Direct whole-module optimization through egglog.
//!
//! This module provides WHOLE MODULE optimization in a SINGLE egglog pass.
//! All functions, all blocks, all instructions go into ONE e-graph for
//! global optimization including:
//! - Cross-function constant propagation
//! - Global common subexpression elimination
//! - Inter-procedural algebraic simplifications

mod context;
mod emit;
mod parse;

use crate::egglog_opt::{create_spirv_egraph, EgglogOptError};
use rspirv::dr::{Instruction, Module};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

use context::EgglogContext;
use emit::{parse_sexpr, EmitCtx, Term};
use parse::parse_extract_result;

/// Optimize an entire SPIR-V module in ONE egglog pass.
///
/// This collects ALL optimizable instructions from ALL functions into
/// a single egglog e-graph, runs optimization ONCE, then reconstructs
/// the optimized module.
pub fn optimize_module_direct(module: &Module) -> Result<Module, EgglogOptError> {
    // Step 1: Collect type information
    let type_widths = collect_type_widths(module);
    let type_classes = collect_type_classes(module);

    // Step 2: Collect ALL SSA values (for id_map) and optimizable instructions
    let mut ctx = EgglogContext::new(&type_widths, &type_classes);

    // Detect GLSL.std.450 extended instruction set
    for inst in &module.ext_inst_imports {
        if inst.class.opcode == Op::ExtInstImport {
            if let Some(rspirv::dr::Operand::LiteralString(name)) = inst.operands.first() {
                if name == "GLSL.std.450" {
                    if let Some(id) = inst.result_id {
                        ctx.set_glsl_ext_id(id);
                    }
                }
            }
        }
    }

    // Collect ALL IDs in the module (not just SSA values) so next_id doesn't collide.
    // This includes block labels, function defs, types, etc.
    let mut all_ssa_ids: HashSet<Word> = HashSet::new();

    // Track which block each value is defined in
    let mut id_to_block: HashMap<Word, Word> = HashMap::new();

    // Add module-level constants and types first
    // Pre-registration pass: populate id_to_type and known_instruction_ids
    // BEFORE creating any terms. This ensures cross-block forward references
    // use variable references (id{N}) instead of Sym constructors.
    for inst in &module.types_global_values {
        ctx.pre_register(inst);
    }
    for func in &module.functions {
        for param in &func.parameters {
            ctx.pre_register(param);
        }
        for block in &func.blocks {
            for inst in &block.instructions {
                ctx.pre_register(inst);
            }
        }
    }

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
        case_labels: Vec<Word>, // All case block labels (including default)
        func_idx: usize,
    }
    let mut switch_constructs: Vec<SwitchInfo> = Vec::new();

    // Loop detection: track back-edges for LICM
    #[derive(Debug, Clone)]
    struct LoopInfo {
        body_block_indices: Vec<usize>, // Indices of blocks in the loop body
        continue_block_idx: Option<usize>, // Continue block (may be outside body range)
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
        // Function def/end IDs
        if let Some(id) = func.def.as_ref().and_then(|d| d.result_id) {
            all_ssa_ids.insert(id);
        }

        // Function parameters
        for param in &func.parameters {
            if let Some(id) = param.result_id {
                all_ssa_ids.insert(id);
            }
        }

        for (block_idx, block) in func.blocks.iter().enumerate() {
            // Block label IDs are part of the ID space
            if let Some(label_id) = block.label.as_ref().and_then(|l| l.result_id) {
                all_ssa_ids.insert(label_id);
            }

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
                    let operands: Vec<Word> = inst
                        .operands
                        .iter()
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
                    for (operand_idx, op) in inst.operands.iter().enumerate() {
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
                    }
                }

                // Detect loops via OpLoopMerge (canonical loop marker).
                // This is more reliable than back-edge detection because it
                // catches loops with BranchConditional continue blocks.
                if inst.class.opcode == Op::LoopMerge {
                    if let (
                        Some(rspirv::dr::Operand::IdRef(merge_label)),
                        Some(rspirv::dr::Operand::IdRef(continue_label)),
                    ) = (inst.operands.first(), inst.operands.get(1))
                    {
                        let label_map = &func_block_labels[func_idx];
                        let continue_idx = label_map.get(continue_label).copied();
                        // Collect loop body via CFG traversal from header, stopping
                        // at the merge block. This handles non-contiguous block layouts
                        // that a simple (header..merge) index range would miss.
                        let mut body_indices: Vec<usize> = Vec::new();
                        let mut visited: HashSet<usize> = HashSet::new();
                        let mut worklist: Vec<usize> = vec![block_idx];
                        let merge_idx = label_map.get(merge_label).copied();
                        while let Some(idx) = worklist.pop() {
                            if !visited.insert(idx) {
                                continue;
                            }
                            // Don't include the merge block itself
                            if Some(idx) == merge_idx {
                                continue;
                            }
                            body_indices.push(idx);
                            // Follow control-flow edges (branch targets) from this block.
                            // Only terminators (Branch/BranchConditional/Switch) and
                            // merge instructions (LoopMerge/SelectionMerge) reference
                            // block labels as operands.
                            if let Some(blk) = func.blocks.get(idx) {
                                for bi in &blk.instructions {
                                    if matches!(
                                        bi.class.opcode,
                                        Op::Branch
                                            | Op::BranchConditional
                                            | Op::Switch
                                            | Op::LoopMerge
                                            | Op::SelectionMerge
                                    ) {
                                        for op in &bi.operands {
                                            if let Some(target_label) = op.id_ref_any() {
                                                if let Some(&target_idx) =
                                                    label_map.get(&target_label)
                                                {
                                                    worklist.push(target_idx);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        loop_constructs.push(LoopInfo {
                            body_block_indices: body_indices,
                            continue_block_idx: continue_idx,
                            func_idx,
                        });
                    }
                }
            }

            // If we found a selection construct, record it
            if let (Some(merge), Some(then_l), Some(else_l), Some(cond)) =
                (merge_label, then_label, else_label, condition_id)
            {
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
    // Use topological order: if term A references variable idB, then B must be bound first.
    // This prevents "unbound symbol" errors from forward references.
    let binding_order = topological_sort_bindings(&ctx.id_to_term);
    for id in binding_order {
        if let Some(term) = ctx.id_to_term.get(&id) {
            let binding = format!("(let id{} {})", id, term);
            egraph
                .parse_and_run_program(None, &binding)
                .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;
        }
    }

    // Run additional facts (e.g., ResultWidth seeding for SConvert/UConvert)
    for fact in &ctx.additional_facts {
        let _ = egraph.parse_and_run_program(None, fact);
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
        header_block_label: Word,
    }
    let mut branch_value_pairs: Vec<BranchValuePair> = Vec::new();

    for sel in &selection_constructs {
        let func = &module.functions[sel.func_idx];

        // Find header block label
        let header_block_label = func
            .blocks
            .get(sel.header_block_idx)
            .and_then(|b| b.label.as_ref())
            .and_then(|l| l.result_id);

        if header_block_label.is_none() {
            continue;
        }
        let header_label = header_block_label.unwrap();

        // Collect IDs defined in then and else blocks
        let then_ids: Vec<Word> = ctx
            .root_ids
            .iter()
            .filter(|&&id| id_to_block.get(&id) == Some(&sel.then_label))
            .copied()
            .collect();
        let else_ids: Vec<Word> = ctx
            .root_ids
            .iter()
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

                // Add a Gamma term representing the selection between these values
                // The egraph will unify equivalent computations via (GammaX c x x) => x
                // Use typed Gamma based on branch type class
                let then_type_class = ctx
                    .id_to_type
                    .get(&then_id)
                    .and_then(|ty| type_classes.get(ty))
                    .copied()
                    .unwrap_or(TypeClass::Other);
                let else_type_class = ctx
                    .id_to_type
                    .get(&else_id)
                    .and_then(|ty| type_classes.get(ty))
                    .copied()
                    .unwrap_or(TypeClass::Other);
                // Skip if branch values have different type classes — creating
                // a typed Gamma with mismatched sorts would crash the egraph.
                if then_type_class != else_type_class {
                    continue;
                }

                branch_value_pairs.push(BranchValuePair {
                    then_id,
                    else_id,
                    header_block_label: header_label,
                });

                let cond_term = if ctx.id_to_term.contains_key(&sel.condition_id) {
                    format!("id{}", sel.condition_id)
                } else {
                    format!("(BSym \"id{}\")", sel.condition_id)
                };
                let gamma_term = format!(
                    "({} {} id{} id{})",
                    then_type_class.typed_ctor("Gamma"),
                    cond_term,
                    then_id,
                    else_id
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

    // Track IDs that already have theta bindings to prevent shadowing
    // when the same ID appears in multiple nested loops' body blocks.
    let mut theta_bound_ids: HashSet<Word> = HashSet::new();

    for loop_info in &loop_constructs {
        let func = &module.functions[loop_info.func_idx];

        // Collect all block labels in loop body
        let body_labels: HashSet<Word> = loop_info
            .body_block_indices
            .iter()
            .filter_map(|&idx| func.blocks.get(idx))
            .filter_map(|block| block.label.as_ref().and_then(|l| l.result_id))
            .collect();

        // Find values defined in loop body and create Theta terms
        for &id in &ctx.root_ids {
            if let Some(&block_label) = id_to_block.get(&id) {
                if body_labels.contains(&block_label) {
                    // This value is defined inside the loop
                    if ctx.id_to_term.contains_key(&id) && !theta_bound_ids.contains(&id) {
                        // Use typed Theta matching the value's sort to avoid
                        // sort mismatches (IntExpr/FloatExpr/BoolExpr vs Expr)
                        let value_type_class = ctx
                            .id_to_type
                            .get(&id)
                            .and_then(|ty| type_classes.get(ty))
                            .copied()
                            .unwrap_or(TypeClass::Other);
                        let theta_ctor = value_type_class.typed_ctor("Theta");
                        let init_val = match value_type_class {
                            TypeClass::Int => "(Const 0)".to_string(),
                            TypeClass::Float => "(FConst 0.0)".to_string(),
                            TypeClass::Bool => "(BoolConst 0)".to_string(),
                            TypeClass::Other => format!("(Sym \"theta_init_{}\")", id),
                        };
                        let theta_term =
                            format!("({} (BoolConst 1) id{} {})", theta_ctor, id, init_val);
                        let theta_binding = format!("(let theta_{} {})", id, theta_term);
                        egraph
                            .parse_and_run_program(None, &theta_binding)
                            .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;
                        theta_bound_ids.insert(id);

                        // NOTE: We intentionally do NOT union id{N} with theta_{N}.
                        // Doing so puts FConst(0.0)/Const(0)/BoolConst(0) init values
                        // into the same e-class as the actual computation, which can
                        // cause the extractor to pick the zero constant instead of
                        // the real value. The Theta term exists in the egraph so
                        // LoopInvariant rules can still detect and mark invariants.
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
        block
            .instructions
            .last()
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
        merge_label: Word,
        then_label: Word,
        else_label: Word,
        effect_var: String,
    }
    let mut rvsdg_selections: Vec<RvsdgSelection> = Vec::new();

    // Build a set of (func_idx, block_idx) pairs that are inside loop bodies.
    // Selection constructs inside loops are skipped to avoid breaking loop structure.
    let mut loop_block_set: HashSet<(usize, usize)> = HashSet::new();
    for loop_info in &loop_constructs {
        for &block_idx in &loop_info.body_block_indices {
            loop_block_set.insert((loop_info.func_idx, block_idx));
        }
        // Continue block may lie outside the header..merge range
        if let Some(continue_idx) = loop_info.continue_block_idx {
            loop_block_set.insert((loop_info.func_idx, continue_idx));
        }
    }
    // Expand: if a selection's header is inside a loop, its branch targets
    // and merge block are part of the loop too. Iterate to a fixed point
    // for nested selections.
    loop {
        let prev_len = loop_block_set.len();
        for sel in &selection_constructs {
            if loop_block_set.contains(&(sel.func_idx, sel.header_block_idx)) {
                let label_map = &func_block_labels[sel.func_idx];
                if let Some(&idx) = label_map.get(&sel.then_label) {
                    loop_block_set.insert((sel.func_idx, idx));
                }
                if let Some(&idx) = label_map.get(&sel.else_label) {
                    loop_block_set.insert((sel.func_idx, idx));
                }
                if let Some(&idx) = label_map.get(&sel.merge_label) {
                    loop_block_set.insert((sel.func_idx, idx));
                }
            }
        }
        if loop_block_set.len() == prev_len {
            break;
        }
    }

    // Build a set of (func_idx, block_idx) pairs that are loop continue blocks.
    // Selections whose branch targets overlap these must also be skipped.
    let mut continue_block_set: HashSet<(usize, usize)> = HashSet::new();
    for loop_info in &loop_constructs {
        if let Some(continue_idx) = loop_info.continue_block_idx {
            continue_block_set.insert((loop_info.func_idx, continue_idx));
        }
    }

    // For each selection construct, convert to RVSDG EffGamma
    for (sel_idx, sel) in selection_constructs.iter().enumerate() {
        // Skip selections that overlap with loop bodies or continue blocks.
        // Check ALL blocks involved (header, then, else, merge) — not just the header —
        // because SPIR-V blocks may not be laid out contiguously.
        {
            let label_map = &func_block_labels[sel.func_idx];
            let in_loop = loop_block_set.contains(&(sel.func_idx, sel.header_block_idx))
                || [&sel.then_label, &sel.else_label, &sel.merge_label]
                    .iter()
                    .any(|label| {
                        label_map.get(label).map_or(false, |&idx| {
                            loop_block_set.contains(&(sel.func_idx, idx))
                                || continue_block_set.contains(&(sel.func_idx, idx))
                        })
                    });
            if in_loop {
                continue;
            }
        }
        let func = &module.functions[sel.func_idx];

        // Find the header block to get the condition
        let header_block = &func.blocks[sel.header_block_idx];
        let condition_id = header_block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == Op::BranchConditional)
            .and_then(|inst| inst.operands.first())
            .and_then(|op| op.id_ref_any());

        // Find the blocks
        let then_block = func
            .blocks
            .iter()
            .find(|b| get_block_label(b) == Some(sel.then_label));
        let else_block = func
            .blocks
            .iter()
            .find(|b| get_block_label(b) == Some(sel.else_label));
        let merge_block = func
            .blocks
            .iter()
            .find(|b| get_block_label(b) == Some(sel.merge_label));

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
            // Use the actual condition variable from the egraph so that
            // Gamma simplification rules can properly track arm correspondence.
            // A synthetic "cond{N}" would be disconnected from the actual value,
            // allowing incorrect e-class merges and arm swaps.
            let cond_sym = if ctx.id_to_term.contains_key(&cond_id) {
                format!("id{}", cond_id)
            } else {
                format!("(BSym \"id{}\")", cond_id)
            };

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
                merge_label: sel.merge_label,
                then_label: sel.then_label,
                else_label: sel.else_label,
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
    }
    let mut rvsdg_switches: Vec<RvsdgSwitch> = Vec::new();

    for sw in &switch_constructs {
        let func = &module.functions[sw.func_idx];
        let merge_block = func
            .blocks
            .iter()
            .find(|b| get_block_label(b) == Some(sw.merge_label));

        if let Some(merge_b) = merge_block {
            if !ends_with_unreachable(merge_b) {
                continue;
            }

            // Validate: every case block must have a return value or be unreachable
            let all_valid = sw.case_labels.iter().all(|&case_label| {
                func.blocks
                    .iter()
                    .find(|b| get_block_label(b) == Some(case_label))
                    .is_some_and(|case_b| {
                        get_return_value_operand(case_b).is_some() || ends_with_unreachable(case_b)
                    })
            });

            if all_valid && sw.case_labels.len() >= 2 {
                rvsdg_switches.push(RvsdgSwitch {
                    func_idx: sw.func_idx,
                    merge_label: sw.merge_label,
                    case_labels: sw.case_labels.clone(),
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
            // Mark the expression as Live using the typed variant matching its sort
            let live_class = ctx
                .id_to_type
                .get(&root_id)
                .and_then(|ty| type_classes.get(ty))
                .copied()
                .unwrap_or(TypeClass::Other);
            let live_cmd = format!("({} id{})", live_class.typed_ctor("Live"), root_id);
            let _ = egraph.parse_and_run_program(None, &live_cmd);
        }
    }

    // Step 6: Run optimization ONCE - both optimization rules AND liveness propagation
    // happen in this single saturation pass. This enables:
    // - DCE-aware constant folding (dead branches are identified)
    // - Partial DCE (RVSDG Gamma/Theta branches marked dead when condition is constant)
    // - Optimizations that expose new DCE opportunities
    let run_cmd = "(run-schedule (repeat 10 (run)))";
    egraph
        .parse_and_run_program(None, run_cmd)
        .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;

    // Step 7: Query which IDs are live after saturation
    // Only live IDs need to be extracted and emitted
    let mut live_ids: HashSet<Word> = HashSet::new();
    for &id in ctx.id_to_term.keys() {
        // Check the typed Live variant matching this ID's sort
        let live_class = ctx
            .id_to_type
            .get(&id)
            .and_then(|ty| type_classes.get(ty))
            .copied()
            .unwrap_or(TypeClass::Other);
        let check_cmd = format!("(check ({} id{}))", live_class.typed_ctor("Live"), id);
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
    // Key format: "const_TYPE_N" for ints, "fconst_TYPE_BITS" for floats, "boolconst_N" for bools
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            match inst.class.opcode {
                Op::Constant => {
                    // Check if this constant's type is float
                    let is_float = inst.result_type.and_then(|ty| type_classes.get(&ty))
                        == Some(&TypeClass::Float);

                    if is_float {
                        // Float constant - key by f64 bit pattern
                        if let Some(val) = inst.operands.first() {
                            let bits: u64 = match val {
                                rspirv::dr::Operand::LiteralBit32(v) => {
                                    (f32::from_bits(*v) as f64).to_bits()
                                }
                                rspirv::dr::Operand::LiteralBit64(v) => *v,
                                _ => continue,
                            };
                            let ty = inst.result_type.unwrap_or(0);
                            debug_assert!(ty != 0, "float constant %{} has no result type", id);
                            let key = format!("fconst_{}_{}", ty, bits);
                            id_map.entry(key).or_insert(id);
                        }
                    } else {
                        // Integer constant — sign-extend 32-bit values to match context.rs
                        if let Some(val) = inst.operands.first() {
                            let ty = inst.result_type.unwrap_or(0);
                            debug_assert!(ty != 0, "integer constant %{} has no result type", id);
                            let width = type_widths.get(&ty).copied().unwrap_or(32);
                            let value: i64 = match val {
                                rspirv::dr::Operand::LiteralBit32(v) => {
                                    if width == 32 {
                                        (*v as i32) as i64
                                    } else {
                                        *v as i64
                                    }
                                }
                                rspirv::dr::Operand::LiteralBit64(v) => *v as i64,
                                _ => continue,
                            };
                            let key = format!("const_{}_{}", ty, value);
                            id_map.entry(key).or_insert(id);
                        }
                    }
                }
                Op::ConstantTrue => {
                    id_map.entry("boolconst_1".to_string()).or_insert(id);
                }
                Op::ConstantFalse => {
                    id_map.entry("boolconst_0".to_string()).or_insert(id);
                }
                _ => {}
            }
        }
    }

    // Track synthesized constants that we need to add to the module
    let mut synthesized_constants: Vec<Instruction> = Vec::new();
    // Track synthesized intermediate instructions (from nested expression materialization)
    // Maps root_id -> list of synthesized intermediate instructions that must precede it
    let mut synthesized_for_root: HashMap<Word, Vec<Instruction>> = HashMap::new();
    // Get next available ID for synthesized constants
    let mut next_id = all_ssa_ids.iter().copied().max().unwrap_or(0) + 1;
    // Find suitable types for synthesized constants
    let int32_type = find_spirv_type(module, Op::TypeInt, Some(32));
    let int64_type = find_spirv_type(module, Op::TypeInt, Some(64));
    let bool_type = find_spirv_type(module, Op::TypeBool, None);
    let float32_type = find_spirv_type(module, Op::TypeFloat, Some(32));
    let float64_type = find_spirv_type(module, Op::TypeFloat, Some(64));

    // Build composite → element type mapping for CompositeConstruct emission
    let mut composite_element_types: HashMap<Word, Word> = HashMap::new();
    for inst in &module.types_global_values {
        match inst.class.opcode {
            Op::TypeVector | Op::TypeMatrix | Op::TypeArray | Op::TypeRuntimeArray => {
                if let (Some(composite_id), Some(rspirv::dr::Operand::IdRef(element_id))) =
                    (inst.result_id, inst.operands.first())
                {
                    composite_element_types.insert(composite_id, *element_id);
                }
            }
            _ => {}
        }
    }

    // Only extract from IDs that are both:
    // 1. True roots (operands of side effects) - these are the outputs we need
    // 2. Live (reachable via liveness propagation in the e-graph)
    // This implements full in-e-graph DCE: liveness is computed during saturation
    let extraction_roots: Vec<Word> = ctx
        .root_ids
        .iter()
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
                debug_assert!(result_type != 0, "extraction root %{} has no type", id);

                // Parse the extracted term into a tree (once).
                // The tree walker handles bridge constructors transparently.
                let parsed_term = parse_sexpr(&term);

                // Track all IDs referenced in this term for DCE
                if let Some(ref pt) = parsed_term {
                    collect_ids_from_parsed_term(pt, &id_map, &mut used_ids);
                }
                // The root ID itself is used
                used_ids.insert(id);
                // The result type is used
                if result_type != 0 {
                    used_ids.insert(result_type);
                }

                // Check if the result is just a reference to another ID
                if let Some(alias_id) = parse_sym_alias_from_term(parsed_term.as_ref(), &id_map) {
                    if alias_id != id {
                        // Only alias when both IDs have the same SPIR-V type.
                        // The egraph may unify values across SPIR-V types (e.g. two
                        // constants with the same bit pattern but different type IDs).
                        // Aliasing across types would cause OpStore type mismatches
                        // when resolve_aliases replaces operand references.
                        let type_matches = ctx.id_to_type.get(&id)
                            == ctx.id_to_type.get(&alias_id);
                        if type_matches {
                            id_aliases.insert(id, alias_id);
                        }
                        used_ids.insert(alias_id);
                        // Emit CopyObject to maintain SSA form
                        optimized_instructions.insert(
                            id,
                            Instruction::new(
                                Op::CopyObject,
                                Some(result_type),
                                Some(id),
                                vec![rspirv::dr::Operand::IdRef(alias_id)],
                            ),
                        );
                    }
                } else if let Some(ref term_tree) = parsed_term {
                    // Use the original result_type from the SPIR-V module.
                    // Type correction via IType/FType/BType egraph queries was
                    // removed because bidirectional type propagation rules can
                    // corrupt types across sort boundaries (e.g. shift amount
                    // width, Store pointer-pointee mismatches).
                    let corrected_type = result_type;

                    // Unified emission: handles both flat and nested terms
                    let mut emit_ctx = EmitCtx {
                        id_map: &mut id_map,
                        next_id: &mut next_id,
                        int32_type,
                        int64_type,
                        float32_type,
                        float64_type,
                        bool_type,
                        type_classes: &type_classes,
                        glsl_ext_id: ctx.glsl_ext_id(),
                        type_widths: &type_widths,
                        id_to_type: &ctx.id_to_type,
                        composite_element_types: &composite_element_types,
                    };
                    if let Some((final_id, new_insts)) =
                        emit::emit_term(term_tree, corrected_type, &mut emit_ctx)
                    {
                        let num_insts = new_insts.len();
                        if num_insts == 0 {
                            // emit_term resolved to an existing ID — emit CopyObject
                            // if it's a different ID, or skip if same
                            if final_id != id {
                                // Only alias when SPIR-V types match (see above)
                                let type_matches = ctx.id_to_type.get(&id)
                                    == ctx.id_to_type.get(&final_id);
                                if type_matches {
                                    id_aliases.insert(id, final_id);
                                }
                                used_ids.insert(final_id);
                                optimized_instructions.insert(
                                    id,
                                    Instruction::new(
                                        Op::CopyObject,
                                        Some(corrected_type),
                                        Some(id),
                                        vec![rspirv::dr::Operand::IdRef(final_id)],
                                    ),
                                );
                            }
                        } else {
                            for (i, mut inst) in new_insts.into_iter().enumerate() {
                                if i == num_insts - 1 {
                                    // The final instruction gets the original result ID
                                    let old_id = inst.result_id;
                                    inst.result_id = Some(id);
                                    inst.result_type = Some(corrected_type);
                                    collect_ids_from_instruction(&inst, &mut used_ids);
                                    optimized_instructions.insert(id, inst);
                                    if let Some(old) = old_id {
                                        if old != id {
                                            emit_ctx.id_map.insert(format!("id{}", id), id);
                                        }
                                    }
                                } else if let Some(inst_id) = inst.result_id {
                                    // Intermediate instructions are NEW
                                    collect_ids_from_instruction(&inst, &mut used_ids);
                                    if matches!(
                                        inst.class.opcode,
                                        Op::Constant | Op::ConstantTrue | Op::ConstantFalse
                                    ) {
                                        synthesized_constants.push(inst.clone());
                                    } else {
                                        synthesized_for_root
                                            .entry(id)
                                            .or_default()
                                            .push(inst.clone());
                                    }
                                    optimized_instructions.entry(inst_id).or_insert(inst);
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
        then_id: Word, // ID from then-branch that should become CopyObject
        else_id: Word, // ID from else-branch that should become CopyObject
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

        // If the gamma has simplified to just the expression (not a Gamma/Select variant),
        // it means both branches computed the same thing and it can be hoisted
        let is_gamma_or_select = match parse_sexpr(&gamma_term) {
            Some(Term::App { ref op, .. }) => matches!(
                op.as_str(),
                "Gamma"
                    | "GammaI"
                    | "GammaF"
                    | "GammaB"
                    | "Select"
                    | "SelectI"
                    | "SelectF"
                    | "SelectB"
            ),
            _ => false,
        };
        if !is_gamma_or_select {
            // The expression can be hoisted!
            // Mark both branch IDs to become CopyObjects of the hoisted value
            let result_type = ctx.id_to_type.get(&pair.then_id).copied().unwrap_or(0);
            debug_assert!(
                result_type != 0,
                "hoisted value %{} has no type",
                pair.then_id
            );

            hoisted_values.push(HoistInfo {
                then_id: pair.then_id,
                else_id: pair.else_id,
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
        optimized_instructions.insert(
            hoist.else_id,
            Instruction::new(
                Op::CopyObject,
                Some(hoist.result_type),
                Some(hoist.else_id),
                vec![rspirv::dr::Operand::IdRef(hoist.then_id)],
            ),
        );
    }

    // Track which IDs need to be moved to header blocks
    let hoisted_id_to_header: HashMap<Word, Word> = hoisted_values
        .iter()
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
        Branch(Word),                                     // Branch to merge
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
                ParsedEffect::ReturnValueWithGamma {
                    then_term,
                    else_term,
                } => {
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
                    let then_id = parse_sym_alias_from_term(Some(&then_term), &id_map);
                    let else_id = parse_sym_alias_from_term(Some(&else_term), &id_map);

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
                    if let Some(val_id) = parse_sym_alias_from_term(Some(&val_term), &id_map) {
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
            let case_block = func
                .blocks
                .iter()
                .find(|b| get_block_label(b) == Some(case_label));
            if let Some(case_b) = case_block {
                if let Some(ret_val) = get_return_value_operand(case_b) {
                    case_values.push((ret_val, case_label)); // (value, label) order for phi
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
                new_terminator: NewTerminator::ReturnValueWithMultiPhi(phi_id, case_values),
            });
        }
    }

    // Step 6: Rebuild the module with optimized instructions
    let mut output = module.clone();

    // DCE for types_global_values: remove unused Private/Function variables.
    // Constants are NOT removed here - they will be DCE'd later after cleanup_module
    // has removed dead function body instructions. This ensures we don't remove constants
    // that are still referenced by surviving non-optimized instructions.
    output.types_global_values.retain(|inst| {
        match inst.class.opcode {
            // Variables with Private or Function storage class can be removed if unused.
            // Other storage classes (Input, Output, Uniform, etc.) must be kept as they
            // are part of the shader interface.
            Op::Variable => {
                let storage_class = inst.operands.first().and_then(|op| {
                    if let rspirv::dr::Operand::StorageClass(sc) = op {
                        Some(*sc)
                    } else {
                        None
                    }
                });
                match storage_class {
                    Some(rspirv::spirv::StorageClass::Private)
                    | Some(rspirv::spirv::StorageClass::Function) => {
                        // Private/Function variables can be DCE'd if unused
                        if let Some(id) = inst.result_id {
                            used_ids.contains(&id)
                        } else {
                            true
                        }
                    }
                    // All other storage classes are part of the interface, keep them
                    _ => true,
                }
            }
            // Types, constants, and other instructions are kept for now
            _ => true,
        }
    });

    // Collect existing global IDs once for deduplication
    let mut existing_global_ids: HashSet<Word> = output
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id)
        .collect();

    // Add synthesized constants (already used by definition)
    for const_inst in synthesized_constants {
        if let Some(id) = const_inst.result_id {
            if existing_global_ids.insert(id) {
                output.types_global_values.push(const_inst);
            }
        } else {
            output.types_global_values.push(const_inst);
        }
    }

    // Track IDs that were originally in function bodies but now fold to constants.
    // We preserve the original instruction's ID for stability.
    let mut folded_to_constant: HashSet<Word> = HashSet::new();
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(id) = inst.result_id {
                    if let Some(opt_inst) = optimized_instructions.get(&id) {
                        if matches!(
                            opt_inst.class.opcode,
                            Op::Constant | Op::ConstantTrue | Op::ConstantFalse
                        ) {
                            folded_to_constant.insert(id);
                        }
                    }
                }
            }
        }
    }

    // Add folded constants, sorted by ID for deterministic output
    let mut sorted_folded: Vec<Word> = folded_to_constant.iter().copied().collect();
    sorted_folded.sort();
    for id in sorted_folded {
        if !existing_global_ids.insert(id) {
            continue;
        }
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
                    if let Some(opt_inst) = optimized_instructions.get(&id) {
                        *inst = opt_inst.clone();
                    }
                }
            }
            // Remove instructions that became constants (they're now in types_global_values)
            block.instructions.retain(|inst| {
                if let Some(id) = inst.result_id {
                    !folded_to_constant.contains(&id)
                } else {
                    true
                }
            });
        }
    }

    // Insert synthesized intermediate instructions into the same block as their root instruction.
    // This ensures operand references satisfy SPIR-V dominance requirements.
    if !synthesized_for_root.is_empty() {
        for func in &mut output.functions {
            for block in &mut func.blocks {
                let mut insertions: Vec<(usize, Vec<Instruction>)> = Vec::new();
                for (pos, inst) in block.instructions.iter().enumerate() {
                    if let Some(id) = inst.result_id {
                        if let Some(synth_insts) = synthesized_for_root.get(&id) {
                            insertions.push((pos, synth_insts.clone()));
                        }
                    }
                }
                // Insert in reverse order to preserve positions
                for (pos, insts) in insertions.into_iter().rev() {
                    for (i, inst) in insts.into_iter().enumerate() {
                        block.instructions.insert(pos + i, inst);
                    }
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
                        block
                            .instructions
                            .retain(|inst| inst.class.opcode != Op::Unreachable);

                        // Find the return type by looking at the original value's type
                        // Look up val1's type from the function or module
                        let result_type = ctx
                            .id_to_type
                            .get(val1)
                            .or_else(|| ctx.id_to_type.get(val2))
                            .copied()
                            .unwrap_or_else(|| {
                                // Fall back to finding the type from module types_global_values
                                module
                                    .types_global_values
                                    .iter()
                                    .find(|inst| inst.result_id == Some(*val1))
                                    .and_then(|inst| inst.result_type)
                                    .unwrap_or(0)
                            });
                        debug_assert!(result_type != 0, "selection phi %{} has no type", phi_id);

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
                        block
                            .instructions
                            .retain(|inst| inst.class.opcode != Op::Unreachable);

                        // Find the return type by looking at the first value's type
                        let result_type = case_values
                            .first()
                            .and_then(|(val, _)| ctx.id_to_type.get(val).copied())
                            .unwrap_or_else(|| {
                                // Fall back to finding the type from module types_global_values
                                case_values
                                    .first()
                                    .and_then(|(val, _)| {
                                        module
                                            .types_global_values
                                            .iter()
                                            .find(|inst| inst.result_id == Some(*val))
                                            .and_then(|inst| inst.result_type)
                                    })
                                    .unwrap_or(0)
                            });
                        debug_assert!(result_type != 0, "switch phi has no type");

                        // Build phi operands: (value, label) pairs flattened
                        let phi_operands: Vec<rspirv::dr::Operand> = case_values
                            .iter()
                            .flat_map(|(val, label)| {
                                [
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
            let header_block = func
                .blocks
                .iter_mut()
                .find(|b| b.label.as_ref().and_then(|l| l.result_id) == Some(header_label));

            if let Some(block) = header_block {
                // Find the SelectionMerge instruction and insert before it
                let insert_pos = block
                    .instructions
                    .iter()
                    .position(|i| i.class.opcode == Op::SelectionMerge)
                    .unwrap_or(block.instructions.len());

                block.instructions.insert(insert_pos, inst);
            }
        }
    }

    // Step 7: Clean up - remove instructions that are just CopyObject of themselves or unused
    // Pass true_roots so that modules without side effects don't have everything removed
    cleanup_module(&mut output, &id_aliases, &true_roots);

    // Step 7b: DCE for types_global_values constants.
    // Now that cleanup_module has removed dead function body instructions, we can
    // accurately determine which constants are still referenced. This must happen
    // AFTER cleanup_module so we don't remove constants used by surviving instructions
    // that the e-graph didn't track (e.g., OpAccessChain index constants).
    {
        let mut live_ids: HashSet<Word> = HashSet::new();
        // Collect all IDs referenced by surviving function body instructions
        for func in &output.functions {
            for param in &func.parameters {
                collect_ids_from_instruction(param, &mut live_ids);
            }
            for block in &func.blocks {
                for inst in &block.instructions {
                    collect_ids_from_instruction(inst, &mut live_ids);
                }
            }
        }
        // Collect IDs referenced by annotations (decorations may reference constants)
        for inst in &output.annotations {
            collect_ids_from_instruction(inst, &mut live_ids);
        }
        // Collect IDs referenced by entry points
        for inst in &output.entry_points {
            collect_ids_from_instruction(inst, &mut live_ids);
        }
        // Collect IDs referenced by other types_global_values (e.g., ConstantComposite
        // referencing component constants, or types referencing other types).
        // Iterate to a fixpoint since constants can reference other constants.
        loop {
            let prev_len = live_ids.len();
            for inst in &output.types_global_values {
                if let Some(id) = inst.result_id {
                    if live_ids.contains(&id) {
                        for op in &inst.operands {
                            if let Some(ref_id) = op.id_ref_any() {
                                live_ids.insert(ref_id);
                            }
                        }
                        if let Some(ty) = inst.result_type {
                            live_ids.insert(ty);
                        }
                    }
                }
            }
            if live_ids.len() == prev_len {
                break;
            }
        }
        // Remove constants not referenced by any surviving instruction
        output
            .types_global_values
            .retain(|inst| match inst.class.opcode {
                Op::Constant
                | Op::ConstantTrue
                | Op::ConstantFalse
                | Op::ConstantComposite
                | Op::ConstantSampler
                | Op::ConstantNull
                | Op::SpecConstant
                | Op::SpecConstantTrue
                | Op::SpecConstantFalse
                | Op::SpecConstantComposite
                | Op::SpecConstantOp => {
                    if let Some(id) = inst.result_id {
                        live_ids.contains(&id)
                    } else {
                        true
                    }
                }
                _ => true,
            });
    }

    // Step 8: Update the module's ID bound to account for any new IDs allocated
    // during optimization (synthesized constants, phi nodes, materialized expressions).
    // rspirv's assemble() uses the header bound as-is, so we must update it here.
    if let Some(ref mut header) = output.header {
        if next_id > header.bound {
            header.bound = next_id;
        }
    }

    Ok(output)
}

/// Check if a parsed term is just a Sym reference (possibly wrapped in bridge constructors).
fn parse_sym_alias_from_term(term: Option<&Term>, id_map: &HashMap<String, Word>) -> Option<Word> {
    match term? {
        Term::Atom(s) => id_map.get(s.as_str()).copied(),
        Term::App { op, args } => match op.as_str() {
            "Sym" | "ISym" | "FSym" | "BSym" => {
                if let Some(Term::Atom(name)) = args.first() {
                    id_map.get(name.as_str()).copied()
                } else {
                    None
                }
            }
            "IntToExpr" | "FloatToExpr" | "BoolToExpr" | "ExprToInt" | "ExprToFloat"
            | "ExprToBool" => parse_sym_alias_from_term(args.first(), id_map),
            _ => None,
        },
    }
}

/// Clean up the module by removing redundant instructions and dead code
fn cleanup_module(
    module: &mut Module,
    id_aliases: &HashMap<Word, Word>,
    true_roots: &HashSet<Word>,
) {
    resolve_aliases(module, id_aliases);
    remove_dead_instructions(module, true_roots);
}

/// Phase 1: Build transitive alias map and rewrite all operand references.
/// This is purely mechanical — no decisions, just following CopyObject chains.
fn resolve_aliases(module: &mut Module, id_aliases: &HashMap<Word, Word>) {
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

    // Resolve aliases in types_global_values (constants, composite constants, etc.)
    for inst in &mut module.types_global_values {
        for op in &mut inst.operands {
            if let Some(ref_id) = op.id_ref_any() {
                if let Some(&target) = final_aliases.get(&ref_id) {
                    *op = rspirv::dr::Operand::IdRef(target);
                }
            }
        }
    }

    // Resolve aliases in annotations (decorations referencing value IDs)
    for inst in &mut module.annotations {
        for op in &mut inst.operands {
            if let Some(ref_id) = op.id_ref_any() {
                if let Some(&target) = final_aliases.get(&ref_id) {
                    *op = rspirv::dr::Operand::IdRef(target);
                }
            }
        }
    }

    // Resolve aliases in function body instructions
    for func in &mut module.functions {
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
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
}

/// Phase 2: Iterative DCE — remove instructions whose results aren't referenced.
///
/// The egraph's liveness analysis handles most DCE during saturation
/// (only live IDs are extracted). This post-hoc pass catches residual dead
/// code from non-optimizable instructions that reference now-dead values.
/// Remove instructions whose results are never referenced.
///
/// This catches orphaned intermediates created by the extraction loop.
/// When the egraph constant-folds an expression (e.g., `(4+5)-2` → `7`),
/// the downstream consumer extracts as a constant, but the intermediate
/// instructions (the IAdd for `4+5`) are still emitted because each
/// extraction root is processed independently. This pass removes those
/// dead intermediates.
///
/// Returns the total number of instructions removed.
fn remove_dead_instructions(module: &mut Module, true_roots: &HashSet<Word>) -> usize {
    let mut total_removed: usize = 0;
    loop {
        let mut used_ids: HashSet<Word> = HashSet::new();

        for &root_id in true_roots {
            used_ids.insert(root_id);
        }

        for func in &module.functions {
            for param in &func.parameters {
                if let Some(id) = param.result_id {
                    used_ids.insert(id);
                }
            }
            for block in &func.blocks {
                for inst in &block.instructions {
                    if has_side_effects(inst) {
                        if let Some(id) = inst.result_id {
                            used_ids.insert(id);
                        }
                    }
                    for op in &inst.operands {
                        if let Some(ref_id) = op.id_ref_any() {
                            used_ids.insert(ref_id);
                        }
                    }
                }
            }
        }

        for inst in &module.types_global_values {
            if let Some(id) = inst.result_id {
                if !matches!(
                    inst.class.opcode,
                    Op::Constant | Op::ConstantTrue | Op::ConstantFalse
                ) {
                    used_ids.insert(id);
                }
            }
            // Also mark operand references from types_global_values as used.
            // OpConstantComposite and other globals may reference function-body
            // IDs that were aliased/rewritten by the optimizer.
            for op in &inst.operands {
                if let Some(ref_id) = op.id_ref_any() {
                    used_ids.insert(ref_id);
                }
            }
        }

        let mut removed_any = false;
        for func in &mut module.functions {
            for block in &mut func.blocks {
                let before_len = block.instructions.len();
                block.instructions.retain(|inst| {
                    if let Some(result_id) = inst.result_id {
                        if !used_ids.contains(&result_id) {
                            return false;
                        }
                    }
                    true
                });
                let removed_count = before_len - block.instructions.len();
                if removed_count > 0 {
                    removed_any = true;
                    total_removed += removed_count;
                }
            }
        }

        if !removed_any {
            break;
        }
    }
    total_removed
}

/// Collect all referenced IDs from a parsed Term tree (for DCE tracking).
fn collect_ids_from_parsed_term(
    term: &Term,
    id_map: &HashMap<String, Word>,
    used_ids: &mut HashSet<Word>,
) {
    match term {
        Term::Atom(s) => {
            if let Some(&id) = id_map.get(s.as_str()) {
                used_ids.insert(id);
            }
        }
        Term::App { op, args } => match op.as_str() {
            "Sym" | "ISym" | "FSym" | "BSym" => {
                if let Some(Term::Atom(name)) = args.first() {
                    if let Some(&id) = id_map.get(name.as_str()) {
                        used_ids.insert(id);
                    }
                }
            }
            _ => {
                for arg in args {
                    collect_ids_from_parsed_term(arg, id_map, used_ids);
                }
            }
        },
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

/// Check if an instruction should be optimized via the egraph.
///
/// This includes all pure operations that have both:
/// - an ingestion handler in `instruction_to_term` (context.rs)
/// - an emission handler in the OPS_TABLE (emit.rs)
///
/// Excludes side-effecting ops (Store, AtomicStore, ImageWrite, etc.)
/// and ops handled through RVSDG memory threading (Load, Store).
pub(crate) fn is_optimizable(inst: &Instruction) -> bool {
    matches!(
        inst.class.opcode,
        // Integer arithmetic
        Op::IAdd
            | Op::ISub
            | Op::IMul
            | Op::SDiv
            | Op::UDiv
            | Op::SRem
            | Op::UMod
            | Op::SMod
            | Op::SNegate
            // Shifts
            | Op::ShiftLeftLogical
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
            // Bitwise
            | Op::BitwiseAnd
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::Not
            | Op::BitReverse
            | Op::BitCount
            // Bitfield operations
            | Op::BitFieldSExtract
            | Op::BitFieldUExtract
            | Op::BitFieldInsert
            // Integer comparisons
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
            // Logical
            | Op::LogicalNot
            | Op::LogicalAnd
            | Op::LogicalOr
            | Op::LogicalEqual
            | Op::LogicalNotEqual
            // Select
            | Op::Select
            // Constants
            | Op::Constant
            | Op::ConstantTrue
            | Op::ConstantFalse
            // Copy
            | Op::CopyObject
            | Op::Phi
            // Floating-point arithmetic
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::FRem
            | Op::FMod
            | Op::FNegate
            | Op::Dot
            // Floating-point comparisons (ordered)
            | Op::FOrdEqual
            | Op::FOrdNotEqual
            | Op::FOrdLessThan
            | Op::FOrdLessThanEqual
            | Op::FOrdGreaterThan
            | Op::FOrdGreaterThanEqual
            // Floating-point comparisons (unordered)
            | Op::FUnordEqual
            | Op::FUnordNotEqual
            | Op::FUnordLessThan
            | Op::FUnordLessThanEqual
            | Op::FUnordGreaterThan
            | Op::FUnordGreaterThanEqual
            // Float predicates
            | Op::IsNan
            | Op::IsInf
            | Op::QuantizeToF16
            // Conversions
            | Op::ConvertFToU
            | Op::ConvertFToS
            | Op::ConvertSToF
            | Op::ConvertUToF
            | Op::SConvert
            | Op::UConvert
            | Op::FConvert
            | Op::Bitcast
            // Derivative operations (fragment shader)
            | Op::DPdx
            | Op::DPdy
            | Op::Fwidth
            | Op::DPdxFine
            | Op::DPdyFine
            | Op::FwidthFine
            | Op::DPdxCoarse
            | Op::DPdyCoarse
            | Op::FwidthCoarse
            // Composite operations
            | Op::CompositeExtract
            | Op::CompositeInsert
            | Op::CompositeConstruct
            // Vector operations
            | Op::VectorExtractDynamic
            | Op::VectorInsertDynamic
            | Op::VectorShuffle
            | Op::VectorTimesScalar
            // Matrix operations
            | Op::MatrixTimesScalar
            | Op::MatrixTimesVector
            | Op::VectorTimesMatrix
            | Op::MatrixTimesMatrix
            | Op::Transpose
            | Op::OuterProduct
            // GLSL.std.450 extended instructions
            | Op::ExtInst
            // Access chain (pure pointer arithmetic)
            | Op::AccessChain
            | Op::InBoundsAccessChain
            // Image query operations (pure metadata queries)
            | Op::ImageQuerySize
            | Op::ImageQueryLevels
            | Op::ImageQuerySamples
            | Op::ImageQuerySizeLod
            | Op::ImageQueryLod
            // Image/sampler combining (pure)
            | Op::SampledImage
            | Op::Image
            // Image sampling/fetch (read-only, safe to CSE)
            | Op::ImageSampleImplicitLod
            | Op::ImageSampleExplicitLod
            | Op::ImageFetch
            | Op::ImageRead
    )
}

/// Collect type widths from module (includes int, float, and bool types).
fn collect_type_widths(module: &Module) -> HashMap<Word, u32> {
    module
        .types_global_values
        .iter()
        .filter_map(|inst| match inst.class.opcode {
            Op::TypeInt | Op::TypeFloat => inst.result_id.and_then(|id| {
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

/// Type classification for SPIR-V types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TypeClass {
    Bool,
    Int,
    Float,
    Other,
}

impl TypeClass {
    /// Return the typed egglog constructor name for a given base.
    /// E.g. `TypeClass::Int.typed_ctor("Gamma")` → `"GammaI"`.
    pub(crate) fn typed_ctor(self, base: &str) -> String {
        match self {
            TypeClass::Int => format!("{}I", base),
            TypeClass::Float => format!("{}F", base),
            TypeClass::Bool => format!("{}B", base),
            TypeClass::Other => base.to_string(),
        }
    }
}

/// Find a SPIR-V type declaration's result ID by opcode and optional bit-width.
fn find_spirv_type(module: &Module, opcode: Op, width: Option<u32>) -> Option<Word> {
    module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == opcode
                && match width {
                    Some(w) => {
                        inst.operands.first() == Some(&rspirv::dr::Operand::LiteralBit32(w))
                    }
                    None => true,
                }
        })
        .and_then(|inst| inst.result_id)
}

/// Collect type classes (Bool/Int/Float/Other) for all types in the module.
fn collect_type_classes(module: &Module) -> HashMap<Word, TypeClass> {
    let mut classes = HashMap::new();
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            let class = match inst.class.opcode {
                Op::TypeBool => TypeClass::Bool,
                Op::TypeInt => TypeClass::Int,
                Op::TypeFloat => TypeClass::Float,
                _ => TypeClass::Other,
            };
            if class != TypeClass::Other {
                classes.insert(id, class);
            }
        }
    }
    // NOTE: Vectors are intentionally NOT classified here.
    // Vectors use the general Expr sort (not IntExpr/FloatExpr/BoolExpr),
    // so they must remain TypeClass::Other to avoid mis-typing vector Select/Sym.
    classes
}

/// Query the concrete SPIR-V type ID from the egraph for a given expression.
///
/// Returns the type ID stored in the IType/FType/BType function for the given id,
/// or None if the query fails (e.g., no type was propagated to this expression).
/// Topological sort of binding IDs based on term dependencies.
/// If term for idA contains a bare reference to idB (meaning B is also in id_to_term),
/// then B must be bound before A.
/// Collect bare `idN` atom references from a Term tree (for dependency tracking).
/// Sym/ISym/FSym/BSym wrappers contain string-quoted references that are opaque —
/// only bare atoms like `id5` create ordering dependencies.
fn collect_bare_id_refs(term: &Term, self_id: Word, id_set: &HashSet<Word>, deps: &mut Vec<Word>) {
    match term {
        Term::Atom(s) => {
            if let Some(num_str) = s.strip_prefix("id") {
                if let Ok(ref_id) = num_str.parse::<Word>() {
                    if ref_id != self_id && id_set.contains(&ref_id) {
                        deps.push(ref_id);
                    }
                }
            }
        }
        Term::App { op, args } => match op.as_str() {
            // Sym variants contain quoted string references — skip (no ordering dependency)
            "Sym" | "ISym" | "FSym" | "BSym" => {}
            _ => {
                for arg in args {
                    collect_bare_id_refs(arg, self_id, id_set, deps);
                }
            }
        },
    }
}

fn topological_sort_bindings(id_to_term: &HashMap<Word, String>) -> Vec<Word> {
    use std::collections::VecDeque;

    // Build dependency graph: for each id, which other ids in id_to_term does its term reference?
    let id_set: HashSet<Word> = id_to_term.keys().copied().collect();
    let mut deps: HashMap<Word, Vec<Word>> = HashMap::new();
    let mut reverse_deps: HashMap<Word, Vec<Word>> = HashMap::new();
    let mut in_degree: HashMap<Word, usize> = HashMap::new();

    for (&id, term) in id_to_term {
        let mut my_deps = Vec::new();
        // Parse the term and walk the tree to find bare "idN" atom references
        if let Some(parsed) = parse_sexpr(term) {
            collect_bare_id_refs(&parsed, id, &id_set, &mut my_deps);
        }

        in_degree.insert(id, my_deps.len());
        for &dep in &my_deps {
            reverse_deps.entry(dep).or_default().push(id);
        }
        deps.insert(id, my_deps);
    }

    // Kahn's algorithm
    let mut queue: VecDeque<Word> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&id, _)| id)
        .collect();
    // Sort initial queue for determinism
    let mut initial: Vec<Word> = queue.drain(..).collect();
    initial.sort();
    queue.extend(initial);

    let mut result = Vec::with_capacity(id_to_term.len());
    while let Some(id) = queue.pop_front() {
        result.push(id);
        if let Some(dependents) = reverse_deps.get(&id) {
            let mut ready = Vec::new();
            for &dep_id in dependents {
                if let Some(deg) = in_degree.get_mut(&dep_id) {
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push(dep_id);
                    }
                }
            }
            // Sort for determinism
            ready.sort();
            queue.extend(ready);
        }
    }

    // If there are cycles (shouldn't happen in SSA), append remaining in sorted order
    if result.len() < id_to_term.len() {
        let in_result: HashSet<Word> = result.iter().copied().collect();
        let mut remaining: Vec<Word> = id_to_term
            .keys()
            .filter(|id| !in_result.contains(id))
            .copied()
            .collect();
        remaining.sort();
        result.extend(remaining);
    }

    result
}

// =============================================================================
// RVSDG Effect Parsing Helpers
// =============================================================================

/// Parsed effect result from egglog extraction
#[derive(Debug)]
enum ParsedEffect {
    /// (ReturnValue (Gamma/Select cond then else))
    ReturnValueWithGamma { then_term: Term, else_term: Term },
    /// (ReturnValue expr) - simple return
    ReturnValue(Term),
    /// (Unreachable)
    Unreachable,
}

/// Parse an extracted Effect term from egglog using the Term tree.
fn parse_effect_result(s: &str) -> Option<ParsedEffect> {
    let term = parse_sexpr(s.trim())?;
    match &term {
        Term::App { op, args } if op == "Unreachable" && args.is_empty() => {
            Some(ParsedEffect::Unreachable)
        }
        Term::App { op, args } if op == "ReturnValue" && args.len() == 1 => {
            // Check if the inner term is a Gamma/Select variant
            match &args[0] {
                Term::App {
                    op: inner_op,
                    args: inner_args,
                } if inner_args.len() >= 3
                    && matches!(
                        inner_op.as_str(),
                        "Gamma"
                            | "GammaI"
                            | "GammaF"
                            | "GammaB"
                            | "Select"
                            | "SelectI"
                            | "SelectF"
                            | "SelectB"
                    ) =>
                {
                    Some(ParsedEffect::ReturnValueWithGamma {
                        then_term: inner_args[1].clone(),
                        else_term: inner_args[2].clone(),
                    })
                }
                inner => Some(ParsedEffect::ReturnValue(inner.clone())),
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_optimizable() {
        // Core arithmetic
        let add = Instruction::new(Op::IAdd, Some(1), Some(2), vec![]);
        assert!(is_optimizable(&add));

        // Newly added pure ops
        assert!(is_optimizable(&Instruction::new(
            Op::Dot,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::CompositeExtract,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::CompositeInsert,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::VectorShuffle,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::MatrixTimesMatrix,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::Transpose,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::DPdx,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::Bitcast,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::ExtInst,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::AccessChain,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::ImageQuerySize,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::SampledImage,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::ImageSampleImplicitLod,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::ImageFetch,
            Some(1),
            Some(2),
            vec![]
        )));
        assert!(is_optimizable(&Instruction::new(
            Op::BitFieldSExtract,
            Some(1),
            Some(2),
            vec![]
        )));

        // Side-effecting ops must NOT be optimizable
        let ret = Instruction::new(Op::Return, None, None, vec![]);
        assert!(!is_optimizable(&ret));
        assert!(!is_optimizable(&Instruction::new(
            Op::Store,
            None,
            None,
            vec![]
        )));
        assert!(!is_optimizable(&Instruction::new(
            Op::AtomicStore,
            None,
            None,
            vec![]
        )));
        assert!(!is_optimizable(&Instruction::new(
            Op::ImageWrite,
            None,
            None,
            vec![]
        )));
        assert!(!is_optimizable(&Instruction::new(
            Op::FunctionCall,
            Some(1),
            Some(2),
            vec![]
        )));
        // Load is handled by RVSDG, not per-instruction optimization
        assert!(!is_optimizable(&Instruction::new(
            Op::Load,
            Some(1),
            Some(2),
            vec![]
        )));
    }

    #[test]
    fn optimized_module_id_bound_covers_all_ids() {
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{
            AddressingModel, Capability, ExecutionMode, ExecutionModel, FunctionControl,
            MemoryModel,
        };

        // Build a module with arithmetic that will be constant-folded,
        // potentially creating new IDs for synthesized constants.
        let mut b = Builder::new();
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(int, vec![]);
        let func = b
            .begin_function(int, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let c5 = b.constant_bit32(int, 5);
        let c2 = b.constant_bit32(int, 2);
        let add = b.i_add(int, None, c4, c5).expect("add");
        let sub = b.i_sub(int, None, add, c2).expect("sub");
        b.ret_value(sub).unwrap();
        b.end_function().unwrap();
        b.entry_point(ExecutionModel::GLCompute, func, "main", []);
        b.execution_mode(func, ExecutionMode::LocalSize, [1, 1, 1]);

        let module = b.module();
        let optimized = optimize_module_direct(&module).expect("optimization should succeed");

        // Verify the header bound covers all IDs in the module
        let bound = optimized.header.as_ref().expect("header").bound;
        let max_id = optimized
            .all_inst_iter()
            .filter_map(|inst| inst.result_id)
            .max()
            .unwrap_or(0);
        assert!(
            bound > max_id,
            "ID bound ({bound}) must be greater than max used ID ({max_id})"
        );

        // Also verify the assembled output parses without error
        let words = optimized.assemble();
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&words, &mut loader)
            .expect("optimized module should parse successfully");
    }

    #[test]
    fn constants_referenced_by_non_optimized_instructions_survive_dce() {
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{
            AddressingModel, Capability, ExecutionMode, ExecutionModel, FunctionControl,
            MemoryModel, StorageClass,
        };

        // Build a module where a constant is ONLY referenced by a non-optimized
        // side-effect instruction (OpStore). The e-graph only tracks arithmetic
        // expressions, so this constant wouldn't be in used_ids from extraction.
        // Without the deferred constant DCE fix, this constant would be removed,
        // causing "use of undefined id" errors.
        let mut b = Builder::new();
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let ptr_int = b.type_pointer(None, StorageClass::Function, int);
        let func_ty = b.type_function(void, vec![]);
        let func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        // c0 is only used by OpStore - not part of any arithmetic the e-graph tracks
        let c0 = b.constant_bit32(int, 0);
        // c4 and c5 are used by arithmetic that the e-graph will optimize
        let c4 = b.constant_bit32(int, 4);
        let c5 = b.constant_bit32(int, 5);
        let var = b.variable(ptr_int, None, StorageClass::Function, None);
        let add = b.i_add(int, None, c4, c5).expect("add");
        // OpStore references c0 - a non-optimized instruction
        b.store(var, c0, None, []).unwrap();
        // Use add result so it's not dead
        b.store(var, add, None, []).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
        b.entry_point(ExecutionModel::GLCompute, func, "main", []);
        b.execution_mode(func, ExecutionMode::LocalSize, [1, 1, 1]);

        let module = b.module();
        let optimized = optimize_module_direct(&module).expect("optimization should succeed");

        // Verify the optimized module assembles and parses without error
        // (would fail with "use of undefined id" if c0 was incorrectly DCE'd)
        let words = optimized.assemble();
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&words, &mut loader)
            .expect("optimized module should parse - constants used by OpStore must survive DCE");

        // Verify c0 (constant 0) is still present in the module
        let has_const_0 = optimized.types_global_values.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Constant && inst.result_id == Some(c0)
        });
        assert!(
            has_const_0,
            "constant 0 (used only by OpStore) must survive DCE"
        );
    }

    #[test]
    fn no_duplicate_id_definitions_in_optimized_module() {
        use rspirv::binary::Assemble;
        use rspirv::dr::Builder;
        use rspirv::spirv::{
            AddressingModel, Capability, ExecutionMode, ExecutionModel, FunctionControl,
            MemoryModel,
        };
        use std::collections::HashSet;

        // Build a module with multiple arithmetic operations that may trigger
        // nested expression materialization. This tests that synthesized constants
        // go to types_global_values (not function body) and don't cause duplicates.
        let mut b = Builder::new();
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(int, vec![]);
        let func = b
            .begin_function(int, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        // Create multiple arithmetic operations that the optimizer might fold
        let c1 = b.constant_bit32(int, 1);
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let c4 = b.constant_bit32(int, 4);
        // Nested expressions: ((1 + 2) * (3 + 4))
        let add1 = b.i_add(int, None, c1, c2).expect("add1");
        let add2 = b.i_add(int, None, c3, c4).expect("add2");
        let mul = b.i_mul(int, None, add1, add2).expect("mul");
        b.ret_value(mul).unwrap();
        b.end_function().unwrap();
        b.entry_point(ExecutionModel::GLCompute, func, "main", []);
        b.execution_mode(func, ExecutionMode::LocalSize, [1, 1, 1]);

        let module = b.module();
        let optimized = optimize_module_direct(&module).expect("optimization should succeed");

        // Collect all result IDs from the entire module
        let mut seen_ids: HashSet<u32> = HashSet::new();
        let mut duplicates: Vec<u32> = Vec::new();

        // Check types_global_values
        for inst in &optimized.types_global_values {
            if let Some(id) = inst.result_id {
                if !seen_ids.insert(id) {
                    duplicates.push(id);
                }
            }
        }

        // Check function bodies
        for func in &optimized.functions {
            if let Some(ref def) = func.def {
                if let Some(id) = def.result_id {
                    if !seen_ids.insert(id) {
                        duplicates.push(id);
                    }
                }
            }
            for param in &func.parameters {
                if let Some(id) = param.result_id {
                    if !seen_ids.insert(id) {
                        duplicates.push(id);
                    }
                }
            }
            for block in &func.blocks {
                if let Some(ref label) = block.label {
                    if let Some(id) = label.result_id {
                        if !seen_ids.insert(id) {
                            duplicates.push(id);
                        }
                    }
                }
                for inst in &block.instructions {
                    if let Some(id) = inst.result_id {
                        if !seen_ids.insert(id) {
                            duplicates.push(id);
                        }
                    }
                }
            }
        }

        assert!(
            duplicates.is_empty(),
            "Found duplicate ID definitions: {:?}",
            duplicates
        );

        // Verify the module assembles and parses without error
        let words = optimized.assemble();
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&words, &mut loader)
            .expect("optimized module should have no duplicate IDs");
    }

    #[test]
    fn topological_sort_handles_forward_references() {
        let mut id_to_term: HashMap<Word, String> = HashMap::new();
        // id10 references id20 as a bare variable (both in id_to_term)
        id_to_term.insert(10, "(Add id20 (Const 1))".to_string());
        id_to_term.insert(20, "(Sym \"id5\")".to_string());
        // id5 references nothing in id_to_term
        id_to_term.insert(5, "(Const 42)".to_string());

        let order = topological_sort_bindings(&id_to_term);
        let pos_5 = order.iter().position(|&id| id == 5).unwrap();
        let pos_10 = order.iter().position(|&id| id == 10).unwrap();
        let pos_20 = order.iter().position(|&id| id == 20).unwrap();

        // id20 must come before id10 (id10 references id20)
        assert!(
            pos_20 < pos_10,
            "id20 (pos {pos_20}) must be bound before id10 (pos {pos_10})"
        );
        // id5 has no dependencies from id_to_term references
        let _ = pos_5; // just needs to be present
    }

    // ===== parse_effect_result tests =====

    #[test]
    fn parse_effect_unreachable() {
        let result = parse_effect_result("(Unreachable)");
        assert!(matches!(result, Some(ParsedEffect::Unreachable)));
    }

    #[test]
    fn parse_effect_simple_return() {
        let result = parse_effect_result("(ReturnValue (ISym \"id5\"))");
        match result {
            Some(ParsedEffect::ReturnValue(term)) => {
                assert!(matches!(term, Term::App { ref op, .. } if op == "ISym"));
            }
            other => panic!("Expected ReturnValue, got {:?}", other),
        }
    }

    #[test]
    fn parse_effect_return_with_gamma() {
        let result = parse_effect_result(
            "(ReturnValue (GammaI (BSym \"id1\") (ISym \"id2\") (ISym \"id3\")))",
        );
        match result {
            Some(ParsedEffect::ReturnValueWithGamma {
                then_term,
                else_term,
            }) => {
                assert!(matches!(then_term, Term::App { ref op, .. } if op == "ISym"));
                assert!(matches!(else_term, Term::App { ref op, .. } if op == "ISym"));
            }
            other => panic!("Expected ReturnValueWithGamma, got {:?}", other),
        }
    }

    #[test]
    fn parse_effect_return_with_nested_gamma() {
        let result = parse_effect_result(
            "(ReturnValue (Gamma (BSym \"id1\") (Add (ISym \"id2\") (ISym \"id3\")) (ISym \"id4\")))",
        );
        match result {
            Some(ParsedEffect::ReturnValueWithGamma { then_term, .. }) => {
                // The then_term should be the nested (Add ...) expression
                assert!(matches!(then_term, Term::App { ref op, .. } if op == "Add"));
            }
            other => panic!("Expected ReturnValueWithGamma, got {:?}", other),
        }
    }

    #[test]
    fn parse_effect_return_with_select_variant() {
        let result = parse_effect_result(
            "(ReturnValue (SelectF (BSym \"id1\") (FSym \"id2\") (FSym \"id3\")))",
        );
        assert!(matches!(
            result,
            Some(ParsedEffect::ReturnValueWithGamma { .. })
        ));
    }

    #[test]
    fn parse_effect_non_gamma_return() {
        // ReturnValue with a non-Gamma/Select inner term → simple ReturnValue
        let result = parse_effect_result("(ReturnValue (Add (ISym \"id1\") (ISym \"id2\")))");
        assert!(matches!(result, Some(ParsedEffect::ReturnValue(_))));
    }

    #[test]
    fn parse_effect_empty_returns_none() {
        assert!(parse_effect_result("").is_none());
    }

    #[test]
    fn parse_effect_unknown_returns_none() {
        assert!(parse_effect_result("(Store ptr val)").is_none());
    }

    #[test]
    fn parse_effect_bare_atom_returns_none() {
        assert!(parse_effect_result("id5").is_none());
    }
}
