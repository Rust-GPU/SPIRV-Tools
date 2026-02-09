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
                    if let Some(rspirv::dr::Operand::IdRef(merge_label)) = inst.operands.first() {
                        let label_map = &func_block_labels[func_idx];
                        if let Some(&merge_idx) = label_map.get(merge_label) {
                            // Loop body spans from header to just before merge block
                            let body_indices: Vec<usize> = (block_idx..merge_idx).collect();
                            loop_constructs.push(LoopInfo {
                                body_block_indices: body_indices,
                                func_idx,
                            });
                        }
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
        #[allow(dead_code)]
        condition_id: Word,
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
                    condition_id: sel.condition_id,
                    header_block_label: header_label,
                });

                let cond_term = if ctx.id_to_term.contains_key(&sel.condition_id) {
                    format!("id{}", sel.condition_id)
                } else {
                    format!("(BSym \"id{}\")", sel.condition_id)
                };
                let gamma_ctor = match then_type_class {
                    TypeClass::Int => "GammaI",
                    TypeClass::Float => "GammaF",
                    TypeClass::Bool => "GammaB",
                    TypeClass::Other => "Gamma",
                };
                let gamma_term =
                    format!("({} {} id{} id{})", gamma_ctor, cond_term, then_id, else_id);
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
                        let (theta_ctor, init_val) = match value_type_class {
                            TypeClass::Int => ("ThetaI", "(Const 0)".to_string()),
                            TypeClass::Float => ("ThetaF", "(FConst 0.0)".to_string()),
                            TypeClass::Bool => ("ThetaB", "(BoolConst 0)".to_string()),
                            TypeClass::Other => ("Theta", format!("(Sym \"theta_init_{}\")", id)),
                        };
                        let theta_term =
                            format!("({} (BoolConst 1) id{} {})", theta_ctor, id, init_val);
                        let theta_binding = format!("(let theta_{} {})", id, theta_term);
                        egraph
                            .parse_and_run_program(None, &theta_binding)
                            .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;
                        theta_bound_ids.insert(id);

                        // Union original ID with Theta - after saturation, the egraph will
                        // have propagated LoopInvariant through the expression if applicable
                        let union_cmd = format!("(union id{} theta_{})", id, id);
                        egraph
                            .parse_and_run_program(None, &union_cmd)
                            .map_err(|e| EgglogOptError::ExecutionError(e.to_string()))?;
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
        #[allow(dead_code)]
        header_block_idx: usize,
        merge_label: Word,
        then_label: Word,
        else_label: Word,
        condition_id: Word,
        #[allow(dead_code)]
        effect_var: String, // The egglog variable name for this effect
    }
    let mut rvsdg_selections: Vec<RvsdgSelection> = Vec::new();

    // Build a set of (func_idx, block_idx) pairs that are inside loop bodies.
    // Selection constructs inside loops are skipped to avoid breaking loop structure.
    let mut loop_block_set: HashSet<(usize, usize)> = HashSet::new();
    for loop_info in &loop_constructs {
        for &block_idx in &loop_info.body_block_indices {
            loop_block_set.insert((loop_info.func_idx, block_idx));
        }
    }

    // For each selection construct, convert to RVSDG EffGamma
    for (sel_idx, sel) in selection_constructs.iter().enumerate() {
        // Skip selection constructs inside loop bodies to avoid breaking continue block reachability
        if loop_block_set.contains(&(sel.func_idx, sel.header_block_idx)) {
            continue;
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
            // Use BSym for condition (conditions are always boolean).
            // We use a synthetic name "cond{N}" to preserve the conditional structure
            // so constant folding doesn't eliminate the gamma during saturation.
            let cond_sym = format!("(BSym \"cond{}\")", cond_id);

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
        let merge_block = func
            .blocks
            .iter()
            .find(|b| get_block_label(b) == Some(sw.merge_label));

        if let Some(merge_b) = merge_block {
            if !ends_with_unreachable(merge_b) {
                continue;
            }

            // Collect return values from all case blocks
            let mut case_effects: Vec<(Word, String)> = Vec::new();
            let mut all_valid = true;

            for &case_label in &sw.case_labels {
                let case_block = func
                    .blocks
                    .iter()
                    .find(|b| get_block_label(b) == Some(case_label));
                if let Some(case_b) = case_block {
                    if let Some(ret_val) = get_return_value_operand(case_b) {
                        case_effects
                            .push((case_label, format!("(ReturnValue (Sym \"id{}\"))", ret_val)));
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
            // Mark the expression as Live using the typed variant matching its sort
            let live_ctor = match ctx
                .id_to_type
                .get(&root_id)
                .and_then(|ty| type_classes.get(ty))
            {
                Some(TypeClass::Int) => "LiveI",
                Some(TypeClass::Float) => "LiveF",
                Some(TypeClass::Bool) => "LiveB",
                _ => "Live",
            };
            let live_cmd = format!("({} id{})", live_ctor, root_id);
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
        // Check the typed Live variant matching this ID's sort
        let live_ctor = match ctx.id_to_type.get(&id).and_then(|ty| type_classes.get(ty)) {
            Some(TypeClass::Int) => "LiveI",
            Some(TypeClass::Float) => "LiveF",
            Some(TypeClass::Bool) => "LiveB",
            _ => "Live",
        };
        let check_cmd = format!("(check ({} id{}))", live_ctor, id);
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
    // Key format: "const_N" for 32-bit int, "const64_N" for 64-bit int,
    //             "fconst_BITS" for floats, "boolconst_N" for bools
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
                            let key = format!("fconst_{}", bits);
                            id_map.entry(key).or_insert(id);
                        }
                    } else {
                        // Integer constant
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
    // Find a suitable integer type for synthesized constants
    let int32_type = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands.first() == Some(&rspirv::dr::Operand::LiteralBit32(32))
        })
        .and_then(|inst| inst.result_id);
    let int64_type = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands.first() == Some(&rspirv::dr::Operand::LiteralBit32(64))
        })
        .and_then(|inst| inst.result_id);
    let bool_type = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == Op::TypeBool)
        .and_then(|inst| inst.result_id);
    let float32_type = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeFloat
                && inst.operands.first() == Some(&rspirv::dr::Operand::LiteralBit32(32))
        })
        .and_then(|inst| inst.result_id);

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

                // Before parsing, ensure all inline constants in the term have IDs
                // If the ENTIRE term is just a constant (e.g., "(Const 84)"), use the
                // current instruction's ID for that constant instead of synthesizing.
                // This enables proper DCE - the instruction becomes the constant.
                use parse::InlineConstKind;
                let stripped_root = strip_bridge_constructors(term.trim());
                let is_root_const = stripped_root.starts_with("(Const ")
                    || stripped_root.starts_with("(Const64 ")
                    || stripped_root.starts_with("(BoolConst ")
                    || stripped_root.starts_with("(FConst ");

                for (kind, value) in find_inline_constants(&term) {
                    let key = match kind {
                        InlineConstKind::Int64 => format!("const64_{}", value),
                        InlineConstKind::Int32 => format!("const_{}", value),
                        InlineConstKind::Bool => format!("boolconst_{}", value),
                        InlineConstKind::Float => format!("fconst_{}", value as u64),
                    };
                    if !id_map.contains_key(&key) {
                        // If this root folds to a constant, use its ID for the constant
                        // Don't synthesize a new constant - the instruction becomes it
                        if is_root_const {
                            id_map.insert(key, id);
                            // Don't synthesize - will be added via folded_to_constant later
                        } else if kind == InlineConstKind::Bool {
                            // Synthesize a boolean constant
                            if let Some(ty) = bool_type {
                                let const_id = next_id;
                                next_id += 1;
                                let opcode = if value == 0 {
                                    Op::ConstantFalse
                                } else {
                                    Op::ConstantTrue
                                };
                                synthesized_constants.push(Instruction::new(
                                    opcode,
                                    Some(ty),
                                    Some(const_id),
                                    vec![],
                                ));
                                id_map.insert(key, const_id);
                                id_map.insert(format!("id{}", const_id), const_id);
                            }
                        } else if kind == InlineConstKind::Float {
                            // Synthesize a float constant
                            if let Some(ty) = float32_type {
                                let const_id = next_id;
                                next_id += 1;
                                // value contains f64 bits packed as i64
                                let f64_val = f64::from_bits(value as u64);
                                let operand =
                                    rspirv::dr::Operand::LiteralBit32((f64_val as f32).to_bits());
                                synthesized_constants.push(Instruction::new(
                                    Op::Constant,
                                    Some(ty),
                                    Some(const_id),
                                    vec![operand],
                                ));
                                id_map.insert(key, const_id);
                                id_map.insert(format!("id{}", const_id), const_id);
                            }
                        } else {
                            // Create a new integer constant for use as an operand
                            let const_type = if kind == InlineConstKind::Int64 {
                                // Try to find 64-bit int type
                                module
                                    .types_global_values
                                    .iter()
                                    .find(|inst| {
                                        inst.class.opcode == Op::TypeInt
                                            && inst.operands.first()
                                                == Some(&rspirv::dr::Operand::LiteralBit32(64))
                                    })
                                    .and_then(|inst| inst.result_id)
                                    .or(int32_type)
                            } else {
                                int32_type
                            };
                            if let Some(ty) = const_type {
                                let const_id = next_id;
                                next_id += 1;
                                let operand = if kind == InlineConstKind::Int64 {
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

                // Strip bridge constructors for all downstream parsing.
                // Bridge constructors (IntToExpr, FloatToExpr, etc.) are transparent
                // wrappers from the typed egglog schema that the SPIR-V reconstruction
                // pipeline does not need.
                let stripped_term = strip_bridge_constructors(&term).to_string();

                // Check if the result is just a reference to another ID
                if let Some(alias_id) = parse_sym_alias(&stripped_term, &id_map) {
                    if alias_id != id {
                        // This instruction becomes an alias to another value
                        id_aliases.insert(id, alias_id);
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
                } else {
                    // Query the concrete SPIR-V type ID from the egraph.
                    // The egraph tracks types via IType/FType/BType functions,
                    // seeded from original instructions and propagated through
                    // ONE BIG SATURATION.
                    let term_class = type_class_of_constructor(&stripped_term);
                    let corrected_type =
                        match term_class {
                            TypeClass::Int => query_type_from_egraph(&mut egraph, "IType", id)
                                .unwrap_or(result_type),
                            TypeClass::Float => query_type_from_egraph(&mut egraph, "FType", id)
                                .unwrap_or(result_type),
                            TypeClass::Bool => bool_type.unwrap_or(result_type),
                            TypeClass::Other => result_type,
                        };
                    if corrected_type != result_type {
                        ctx.id_to_type.insert(id, corrected_type);
                    }

                    let type_width = type_widths.get(&corrected_type).copied();
                    // Try simple term_to_instruction first
                    if let Some(inst) =
                        term_to_instruction(&stripped_term, id, corrected_type, &id_map, type_width)
                    {
                        collect_ids_from_instruction(&inst, &mut used_ids);
                        optimized_instructions.insert(id, inst);
                    } else {
                        // If simple parsing fails, try to materialize nested expressions
                        // This handles cases like (Mul (Const 4) (Add (Sym "id5") (Sym "id6")))
                        if let Some((final_id, new_insts)) = materialize_term(
                            &term,
                            corrected_type,
                            &mut id_map,
                            &mut next_id,
                            int32_type,
                            int64_type,
                            float32_type,
                            &ctx.id_to_type,
                            &type_classes,
                            bool_type,
                        ) {
                            let _ = final_id;
                            let num_insts = new_insts.len();
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
                                            id_map.insert(format!("id{}", id), id);
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
        #[allow(dead_code)]
        hoisted_term: String, // The term to compute in header
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
        let is_gamma_or_select = gamma_term.starts_with("(Gamma ")
            || gamma_term.starts_with("(GammaI ")
            || gamma_term.starts_with("(GammaF ")
            || gamma_term.starts_with("(GammaB ")
            || gamma_term.starts_with("(Select ")
            || gamma_term.starts_with("(SelectI ")
            || gamma_term.starts_with("(SelectF ")
            || gamma_term.starts_with("(SelectB ");
        if !is_gamma_or_select {
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
                    cond_term,
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
                    let _cond_id =
                        resolve_term_to_id_or_create(&cond_term, &id_map, sel.condition_id);
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

    // Add synthesized constants to the module (these are already used by definition)
    // Deduplicate: only add if the ID isn't already present in types_global_values
    let existing_global_ids: HashSet<Word> = output
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id)
        .collect();
    for const_inst in synthesized_constants {
        if let Some(id) = const_inst.result_id {
            if !existing_global_ids.contains(&id) {
                output.types_global_values.push(const_inst);
            }
        } else {
            output.types_global_values.push(const_inst);
        }
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
    let copy_to_existing: HashMap<Word, Word> = HashMap::new();

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
    // Deduplicate: skip IDs already present (e.g., from synthesized_constants)
    let existing_folded_ids: HashSet<Word> = output
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id)
        .collect();
    let mut sorted_folded: Vec<Word> = folded_to_constant.iter().copied().collect();
    sorted_folded.sort();
    for id in sorted_folded {
        if existing_folded_ids.contains(&id) {
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

                        // Build phi operands: (value, label) pairs flattened
                        let phi_operands: Vec<rspirv::dr::Operand> = case_values
                            .iter()
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

/// Parse a term to see if it's just a Sym reference to an existing ID
fn parse_sym_alias(term: &str, id_map: &HashMap<String, Word>) -> Option<Word> {
    let term = strip_bridge_constructors(term);
    // Handle typed and untyped Sym variants
    for prefix in &["(Sym \"", "(ISym \"", "(FSym \"", "(BSym \""] {
        if let Some(rest) = term.strip_prefix(prefix) {
            if let Some(sym_name) = rest.strip_suffix("\")") {
                return id_map.get(sym_name).copied();
            }
        }
    }
    None
}

/// Clean up the module by removing redundant instructions and dead code
fn cleanup_module(
    module: &mut Module,
    id_aliases: &HashMap<Word, Word>,
    true_roots: &HashSet<Word>,
) {
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
            eprintln!(
                "DEBUG_DCE: iteration {}, used_ids = {:?}",
                dce_iteration, used_ids
            );
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
                                eprintln!(
                                    "DEBUG_DCE: Removing id{} ({:?})",
                                    result_id, inst.class.opcode
                                );
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
    // Find all Sym variants: (Sym "idN"), (ISym "idN"), (FSym "idN"), (BSym "idN")
    let sym_prefixes: &[&[u8]] = &[b"(Sym \"id", b"(ISym \"id", b"(FSym \"id", b"(BSym \"id"];
    let const_prefixes: &[&[u8]] = &[
        b"(Sym \"const",
        b"(ISym \"const",
        b"(FSym \"const",
        b"(BSym \"const",
    ];
    let mut i = 0;
    let bytes = term.as_bytes();
    while i < bytes.len() {
        let mut matched = false;

        // Check for Sym "id..." patterns (extract numeric IDs)
        for prefix in sym_prefixes {
            if i + prefix.len() < bytes.len() && &bytes[i..i + prefix.len()] == *prefix {
                let start = i + prefix.len();
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
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        // Check for Sym "const..." patterns (constant references)
        for prefix in const_prefixes {
            if i + prefix.len() < bytes.len() && &bytes[i..i + prefix.len()] == *prefix {
                // Find the opening quote position: after "(Sym " or "(ISym " etc.
                // We need to extract the key between the quotes
                // Find the full key between quotes
                let key_start = bytes[i..]
                    .iter()
                    .position(|&b| b == b'"')
                    .map(|p| i + p + 1);
                if let Some(ks) = key_start {
                    let mut end = ks;
                    while end < bytes.len() && bytes[end] != b'"' {
                        end += 1;
                    }
                    if let Ok(key) = std::str::from_utf8(&bytes[ks..end]) {
                        if let Some(&id) = id_map.get(key) {
                            used_ids.insert(id);
                        }
                    }
                    i = end;
                } else {
                    i += prefix.len();
                }
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        i += 1;
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
            // Conversions
            | Op::ConvertFToU
            | Op::ConvertFToS
            | Op::ConvertSToF
            | Op::ConvertUToF
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
fn query_type_from_egraph(egraph: &mut egglog::EGraph, func: &str, id: Word) -> Option<Word> {
    let q = format!("(extract ({} id{}))", func, id);
    egraph
        .parse_and_run_program(None, &q)
        .ok()
        .and_then(|r| r.first().map(|v| format!("{}", v)))
        .and_then(|s| s.trim().parse::<Word>().ok())
}

/// Determine the TypeClass of a term from its egglog constructor name.
///
/// The egglog schema uses typed sorts: Add returns IntExpr, FAdd returns FloatExpr,
/// LogAnd returns BoolExpr, etc. The constructor name deterministically tells us the
/// type class — no operand scanning needed.
fn type_class_of_constructor(term: &str) -> TypeClass {
    // Extract the constructor name from "(ConstructorName ...)" or bare "ConstructorName"
    let name = if let Some(rest) = term.strip_prefix('(') {
        rest.split_whitespace().next().unwrap_or("")
    } else {
        term.split_whitespace().next().unwrap_or("")
    };
    match name {
        // IntExpr constructors (from datatypes.egg)
        "Add" | "Sub" | "Mul" | "Neg" | "SDiv" | "UDiv" | "SRem" | "SMod" | "UMod" | "Shl"
        | "ShrS" | "ShrU" | "BitAnd" | "BitOr" | "BitXor" | "BitNot" | "BitReverse" | "RotL"
        | "RotR" | "SMin" | "SMax" | "UMin" | "UMax" | "SAbs" | "Sign" | "FindILsb"
        | "FindSMsb" | "FindUMsb" | "BitCount" | "BitFieldInsert" | "BitFieldSExtract"
        | "BitFieldUExtract" | "Const" | "Const64" | "ConvertFToS" | "ConvertFToU" | "SConvert"
        | "UConvert" | "SClamp" | "UClamp" | "ISym" | "GammaI" | "SelectI" | "IfI" | "ThetaI"
        | "LoopVarI" | "LoopInvariantI" | "CopyI" | "GroupIAdd" | "GroupIMul" | "GroupSMin"
        | "GroupUMin" | "GroupSMax" | "GroupUMax" | "GroupBitAnd" | "GroupBitOr"
        | "GroupBitXor" | "PackHalf2x16" | "PackSnorm4x8" | "PackSnorm2x16" | "PackUnorm4x8"
        | "PackUnorm2x16" | "PackDouble2x32" | "ExprToInt" | "SubstI" => TypeClass::Int,

        // FloatExpr constructors
        "FAdd" | "FSub" | "FMul" | "FDiv" | "FNeg" | "FRem" | "FMod" | "FAbs" | "FFloor"
        | "FCeil" | "FRound" | "FTrunc" | "QuantizeToF16" | "FMin" | "FMax" | "NMin" | "NMax"
        | "Sqrt" | "InverseSqrt" | "Exp" | "Exp2" | "Log" | "Log2" | "Sin" | "Cos" | "Tan"
        | "Asin" | "Acos" | "Atan" | "Sinh" | "Cosh" | "Tanh" | "Asinh" | "Acosh" | "Atanh"
        | "Fract" | "FSign" | "Radians" | "Degrees" | "Modf" | "Pow" | "Atan2" | "Step"
        | "Ldexp" | "Distance" | "Fma" | "FMix" | "SmoothStep" | "FClamp" | "NClamp" | "Length"
        | "Dot" | "Determinant" | "DPdx" | "DPdy" | "DPdxFine" | "DPdyFine" | "DPdxCoarse"
        | "DPdyCoarse" | "Fwidth" | "FwidthFine" | "FwidthCoarse" | "FConst" | "ConvertSToF"
        | "ConvertUToF" | "FConvert" | "FSym" | "GammaF" | "SelectF" | "IfF" | "ThetaF"
        | "LoopVarF" | "LoopInvariantF" | "CopyF" | "GroupFAdd" | "GroupFMul" | "GroupFMin"
        | "GroupFMax" | "ExprToFloat" | "SubstF" => TypeClass::Float,

        // BoolExpr constructors
        "Eq" | "Ne" | "SLt" | "SLe" | "SGt" | "SGe" | "ULt" | "ULe" | "UGt" | "UGe" | "FOrdEq"
        | "FOrdNe" | "FOrdLt" | "FOrdLe" | "FOrdGt" | "FOrdGe" | "FUnordEq" | "FUnordNe"
        | "FUnordLt" | "FUnordLe" | "FUnordGt" | "FUnordGe" | "FEq" | "FNe" | "FLt" | "FLe"
        | "FGt" | "FGe" | "IsNan" | "IsInf" | "LogNot" | "LogAnd" | "LogOr" | "LogEq" | "LogNe"
        | "BoolConst" | "BSym" | "GammaB" | "SelectB" | "IfB" | "ThetaB" | "LoopVarB"
        | "LoopInvariantB" | "CopyB" | "GroupElect" | "GroupAll" | "GroupAny" | "GroupAllEqual"
        | "GroupLogAnd" | "GroupLogOr" | "GroupLogXor" | "Any" | "All" | "AddOverflows"
        | "SubOverflows" | "MulOverflows" | "FApproxEq" | "ExprToBool" | "SubstB" => {
            TypeClass::Bool
        }

        // Expr sort (vectors, memory, images, etc.) or unknown
        _ => TypeClass::Other,
    }
}

/// Resolve a TypeClass + original SPIR-V result type to the correct concrete type.
///
/// If the original type already matches the required class, keep it (preserves width).
/// Otherwise fall back to a canonical module type (int32, float32, bool).
fn resolve_type_for_class(
    class: TypeClass,
    original_result_type: Word,
    type_classes: &HashMap<Word, TypeClass>,
    int32_type: Option<Word>,
    float32_type: Option<Word>,
    bool_type: Option<Word>,
) -> Word {
    let original_class = type_classes
        .get(&original_result_type)
        .copied()
        .unwrap_or(TypeClass::Other);
    // If original already matches, keep it (preserves width: int16, int64, etc.)
    if original_class == class || class == TypeClass::Other {
        return original_result_type;
    }
    // Otherwise use canonical module type
    match class {
        TypeClass::Int => int32_type.unwrap_or(original_result_type),
        TypeClass::Float => float32_type.unwrap_or(original_result_type),
        TypeClass::Bool => bool_type.unwrap_or(original_result_type),
        TypeClass::Other => original_result_type,
    }
}

/// Topological sort of binding IDs based on term dependencies.
/// If term for idA contains a bare reference to idB (meaning B is also in id_to_term),
/// then B must be bound before A.
fn topological_sort_bindings(id_to_term: &HashMap<Word, String>) -> Vec<Word> {
    use std::collections::VecDeque;

    // Build dependency graph: for each id, which other ids in id_to_term does its term reference?
    let id_set: HashSet<Word> = id_to_term.keys().copied().collect();
    let mut deps: HashMap<Word, Vec<Word>> = HashMap::new();
    let mut reverse_deps: HashMap<Word, Vec<Word>> = HashMap::new();
    let mut in_degree: HashMap<Word, usize> = HashMap::new();

    for (&id, term) in id_to_term {
        let mut my_deps = Vec::new();
        // Scan for bare "idN" references (not inside Sym)
        // These are references to other bound variables
        let bytes = term.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Skip Sym variant patterns - these are safe opaque references
            // Handles: (Sym "..."), (ISym "..."), (FSym "..."), (BSym "...")
            let skip_sym = (i + 5 < bytes.len() && &bytes[i..i + 5] == b"(Sym ")
                || (i + 6 < bytes.len()
                    && (&bytes[i..i + 6] == b"(ISym "
                        || &bytes[i..i + 6] == b"(FSym "
                        || &bytes[i..i + 6] == b"(BSym "));
            if skip_sym {
                // Skip to closing paren
                if let Some(close) = term[i..].find(')') {
                    i += close + 1;
                    continue;
                }
            }
            // Look for bare "id" followed by digits
            if bytes[i] == b'i' && i + 2 < bytes.len() && bytes[i + 1] == b'd' {
                // Check it's not preceded by an alphanumeric (part of a longer token)
                if i > 0 && (bytes[i - 1] as char).is_alphanumeric() {
                    i += 1;
                    continue;
                }
                let start = i + 2;
                let mut end = start;
                while end < bytes.len() && (bytes[end] as char).is_ascii_digit() {
                    end += 1;
                }
                if end > start {
                    if let Ok(ref_id) = term[start..end].parse::<Word>() {
                        if ref_id != id && id_set.contains(&ref_id) {
                            my_deps.push(ref_id);
                        }
                    }
                }
                i = end;
                continue;
            }
            i += 1;
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

            // Check for Gamma/Select variants (typed and untyped)
            let gamma_select_prefixes = [
                "(Gamma ",
                "(GammaI ",
                "(GammaF ",
                "(GammaB ",
                "(Select ",
                "(SelectI ",
                "(SelectF ",
                "(SelectB ",
            ];
            let matched_prefix = gamma_select_prefixes.iter().find(|p| inner.starts_with(*p));
            if let Some(prefix) = matched_prefix {
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
fn resolve_term_to_id_or_create(
    term: &str,
    id_map: &HashMap<String, Word>,
    fallback: Word,
) -> Word {
    resolve_term_to_id_simple(term, id_map).unwrap_or(fallback)
}

/// Strip outermost bridge constructors from a term, recursively.
/// Bridge constructors (IntToExpr, FloatToExpr, BoolToExpr, ExprToInt, ExprToFloat, ExprToBool)
/// are transparent wrappers used by the typed egglog schema. Since each bridge takes exactly
/// one argument, `strip_suffix(')')` safely removes the bridge's closing paren.
fn strip_bridge_constructors(term: &str) -> &str {
    let term = term.trim();
    for prefix in &[
        "(IntToExpr ",
        "(FloatToExpr ",
        "(BoolToExpr ",
        "(ExprToInt ",
        "(ExprToFloat ",
        "(ExprToBool ",
    ] {
        if let Some(rest) = term.strip_prefix(prefix) {
            if let Some(inner) = rest.strip_suffix(')') {
                return strip_bridge_constructors(inner.trim());
            }
        }
    }
    term
}

/// Resolve a term to an ID (simple cases only)
fn resolve_term_to_id_simple(term: &str, id_map: &HashMap<String, Word>) -> Option<Word> {
    let term = strip_bridge_constructors(term);

    // Typed and untyped Sym variants
    for prefix in &["(Sym \"", "(ISym \"", "(FSym \"", "(BSym \""] {
        if let Some(rest) = term.strip_prefix(prefix) {
            if let Some(sym_name) = rest.strip_suffix("\")") {
                return id_map.get(sym_name).copied();
            }
        }
    }

    // Direct id reference (idN)
    if term.starts_with("id") {
        return id_map.get(term).copied();
    }

    // Constant lookups (so materialize_term can reuse existing constants)
    if let Some(rest) = term.strip_prefix("(Const ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                return id_map.get(&format!("const_{}", value)).copied();
            }
        }
    }
    if let Some(rest) = term.strip_prefix("(Const64 ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                return id_map.get(&format!("const64_{}", value)).copied();
            }
        }
    }
    if let Some(rest) = term.strip_prefix("(BoolConst ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                return id_map.get(&format!("boolconst_{}", value)).copied();
            }
        }
    }
    if let Some(rest) = term.strip_prefix("(FConst ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<f64>() {
                return id_map.get(&format!("fconst_{}", value.to_bits())).copied();
            }
        }
    }

    None
}

/// Recursively materialize a term, creating intermediate instructions for nested expressions.
/// Returns the ID of the final result and a list of synthesized instructions.
#[allow(clippy::too_many_arguments)]
fn materialize_term(
    term: &str,
    result_type: Word,
    id_map: &mut HashMap<String, Word>,
    next_id: &mut Word,
    int32_type: Option<Word>,
    int64_type: Option<Word>,
    float32_type: Option<Word>,
    _id_to_type: &HashMap<Word, Word>,
    type_classes: &HashMap<Word, TypeClass>,
    bool_type: Option<Word>,
) -> Option<(Word, Vec<Instruction>)> {
    let term = term.trim();

    // Handle bridge constructors — unwrap and determine concrete type for the sort.
    // Bridge constructors (IntToExpr, ExprToInt, etc.) carry sort info from the egraph.
    // Instead of stripping them and re-deriving types, we use them to set the correct type.
    for (prefix, class) in &[
        ("(IntToExpr ", TypeClass::Int),
        ("(FloatToExpr ", TypeClass::Float),
        ("(BoolToExpr ", TypeClass::Bool),
        ("(ExprToInt ", TypeClass::Int),
        ("(ExprToFloat ", TypeClass::Float),
        ("(ExprToBool ", TypeClass::Bool),
    ] {
        if let Some(rest) = term.strip_prefix(prefix) {
            if let Some(inner) = rest.strip_suffix(')') {
                let bridged_type = resolve_type_for_class(
                    *class,
                    result_type,
                    type_classes,
                    int32_type,
                    float32_type,
                    bool_type,
                );
                return materialize_term(
                    inner.trim(),
                    bridged_type,
                    id_map,
                    next_id,
                    int32_type,
                    int64_type,
                    float32_type,
                    _id_to_type,
                    type_classes,
                    bool_type,
                );
            }
        }
    }

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
                // Create new 64-bit constant — prefer int64 type, fall back to int32
                if let Some(ty) = int64_type.or(int32_type) {
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

    // BoolConst
    if let Some(rest) = term.strip_prefix("(BoolConst ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                let const_key = format!("boolconst_{}", value);
                if let Some(&id) = id_map.get(&const_key) {
                    return Some((id, synthesized));
                }
                if let Some(ty) = bool_type {
                    let const_id = *next_id;
                    *next_id += 1;
                    let op = if value != 0 {
                        Op::ConstantTrue
                    } else {
                        Op::ConstantFalse
                    };
                    let inst = Instruction::new(op, Some(ty), Some(const_id), vec![]);
                    synthesized.push(inst);
                    id_map.insert(const_key, const_id);
                    id_map.insert(format!("id{}", const_id), const_id);
                    return Some((const_id, synthesized));
                }
            }
        }
    }
    // FConst
    if let Some(rest) = term.strip_prefix("(FConst ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<f64>() {
                let const_key = format!("fconst_{}", value.to_bits());
                if let Some(&id) = id_map.get(&const_key) {
                    return Some((id, synthesized));
                }
                if let Some(ty) = float32_type {
                    let const_id = *next_id;
                    *next_id += 1;
                    let bits = (value as f32).to_bits();
                    let inst = Instruction::new(
                        Op::Constant,
                        Some(ty),
                        Some(const_id),
                        vec![rspirv::dr::Operand::LiteralBit32(bits)],
                    );
                    synthesized.push(inst);
                    id_map.insert(const_key, const_id);
                    id_map.insert(format!("id{}", const_id), const_id);
                    return Some((const_id, synthesized));
                }
            }
        }
    }

    // Binary operations — each entry: (name, opcode, result_class, operand_class)
    // The TypeClass fields make type derivation table-driven: no string scanning needed.
    let binary_ops: &[(&str, Op, TypeClass, TypeClass)] = &[
        // Integer arithmetic: result Int, operands Int
        ("Add", Op::IAdd, TypeClass::Int, TypeClass::Int),
        ("Sub", Op::ISub, TypeClass::Int, TypeClass::Int),
        ("Mul", Op::IMul, TypeClass::Int, TypeClass::Int),
        ("SDiv", Op::SDiv, TypeClass::Int, TypeClass::Int),
        ("UDiv", Op::UDiv, TypeClass::Int, TypeClass::Int),
        ("SRem", Op::SRem, TypeClass::Int, TypeClass::Int),
        ("SMod", Op::SMod, TypeClass::Int, TypeClass::Int),
        ("UMod", Op::UMod, TypeClass::Int, TypeClass::Int),
        // Shifts: result Int, operands Int
        ("Shl", Op::ShiftLeftLogical, TypeClass::Int, TypeClass::Int),
        (
            "ShrU",
            Op::ShiftRightLogical,
            TypeClass::Int,
            TypeClass::Int,
        ),
        (
            "ShrS",
            Op::ShiftRightArithmetic,
            TypeClass::Int,
            TypeClass::Int,
        ),
        // Bitwise: result Int, operands Int
        ("BitAnd", Op::BitwiseAnd, TypeClass::Int, TypeClass::Int),
        ("BitOr", Op::BitwiseOr, TypeClass::Int, TypeClass::Int),
        ("BitXor", Op::BitwiseXor, TypeClass::Int, TypeClass::Int),
        // Integer comparisons: result Bool, operands Int
        ("Eq", Op::IEqual, TypeClass::Bool, TypeClass::Int),
        ("Ne", Op::INotEqual, TypeClass::Bool, TypeClass::Int),
        ("SLt", Op::SLessThan, TypeClass::Bool, TypeClass::Int),
        ("SLe", Op::SLessThanEqual, TypeClass::Bool, TypeClass::Int),
        ("SGt", Op::SGreaterThan, TypeClass::Bool, TypeClass::Int),
        (
            "SGe",
            Op::SGreaterThanEqual,
            TypeClass::Bool,
            TypeClass::Int,
        ),
        ("ULt", Op::ULessThan, TypeClass::Bool, TypeClass::Int),
        ("ULe", Op::ULessThanEqual, TypeClass::Bool, TypeClass::Int),
        ("UGt", Op::UGreaterThan, TypeClass::Bool, TypeClass::Int),
        (
            "UGe",
            Op::UGreaterThanEqual,
            TypeClass::Bool,
            TypeClass::Int,
        ),
        // Logical: result Bool, operands Bool
        ("LogAnd", Op::LogicalAnd, TypeClass::Bool, TypeClass::Bool),
        ("LogOr", Op::LogicalOr, TypeClass::Bool, TypeClass::Bool),
        ("LogEq", Op::LogicalEqual, TypeClass::Bool, TypeClass::Bool),
        (
            "LogNe",
            Op::LogicalNotEqual,
            TypeClass::Bool,
            TypeClass::Bool,
        ),
        // Float arithmetic: result Float, operands Float
        ("FAdd", Op::FAdd, TypeClass::Float, TypeClass::Float),
        ("FSub", Op::FSub, TypeClass::Float, TypeClass::Float),
        ("FMul", Op::FMul, TypeClass::Float, TypeClass::Float),
        ("FDiv", Op::FDiv, TypeClass::Float, TypeClass::Float),
        ("FRem", Op::FRem, TypeClass::Float, TypeClass::Float),
        ("FMod", Op::FMod, TypeClass::Float, TypeClass::Float),
        // Float comparisons (ordered): result Bool, operands Float
        ("FOrdEq", Op::FOrdEqual, TypeClass::Bool, TypeClass::Float),
        (
            "FOrdNe",
            Op::FOrdNotEqual,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FOrdLt",
            Op::FOrdLessThan,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FOrdLe",
            Op::FOrdLessThanEqual,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FOrdGt",
            Op::FOrdGreaterThan,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FOrdGe",
            Op::FOrdGreaterThanEqual,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        // Float comparisons (unordered): result Bool, operands Float
        (
            "FUnordEq",
            Op::FUnordEqual,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FUnordNe",
            Op::FUnordNotEqual,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FUnordLt",
            Op::FUnordLessThan,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FUnordLe",
            Op::FUnordLessThanEqual,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FUnordGt",
            Op::FUnordGreaterThan,
            TypeClass::Bool,
            TypeClass::Float,
        ),
        (
            "FUnordGe",
            Op::FUnordGreaterThanEqual,
            TypeClass::Bool,
            TypeClass::Float,
        ),
    ];

    for (name, opcode, result_class, operand_class) in binary_ops {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(rest) = rest.strip_suffix(')') {
                let terms = split_terms_simple(rest);
                if terms.len() >= 2 {
                    // Type derivation is table-driven: result and operand classes
                    // come directly from the ops table entry.
                    let op_result_type = resolve_type_for_class(
                        *result_class,
                        result_type,
                        type_classes,
                        int32_type,
                        float32_type,
                        bool_type,
                    );
                    let operand_type = resolve_type_for_class(
                        *operand_class,
                        result_type,
                        type_classes,
                        int32_type,
                        float32_type,
                        bool_type,
                    );

                    // Recursively materialize operands with the correct type
                    let (lhs_id, mut lhs_synth) = materialize_term(
                        &terms[0],
                        operand_type,
                        id_map,
                        next_id,
                        int32_type,
                        int64_type,
                        float32_type,
                        _id_to_type,
                        type_classes,
                        bool_type,
                    )?;
                    synthesized.append(&mut lhs_synth);

                    let (rhs_id, mut rhs_synth) = materialize_term(
                        &terms[1],
                        operand_type,
                        id_map,
                        next_id,
                        int32_type,
                        int64_type,
                        float32_type,
                        _id_to_type,
                        type_classes,
                        bool_type,
                    )?;
                    synthesized.append(&mut rhs_synth);

                    // Create the binary instruction
                    let inst_id = *next_id;
                    *next_id += 1;
                    let inst = Instruction::new(
                        *opcode,
                        Some(op_result_type),
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

    // Unary operations — each entry: (name, opcode, result_class, operand_class)
    // Conversion ops have DIFFERENT result and operand classes (cross-sort).
    let unary_ops: &[(&str, Op, TypeClass, TypeClass)] = &[
        // Integer
        ("Neg", Op::SNegate, TypeClass::Int, TypeClass::Int),
        ("BitNot", Op::Not, TypeClass::Int, TypeClass::Int),
        ("BitReverse", Op::BitReverse, TypeClass::Int, TypeClass::Int),
        ("LogNot", Op::LogicalNot, TypeClass::Bool, TypeClass::Bool),
        // Floating-point
        ("FNeg", Op::FNegate, TypeClass::Float, TypeClass::Float),
        // Conversions: result and operand are DIFFERENT sorts
        (
            "ConvertFToU",
            Op::ConvertFToU,
            TypeClass::Int,
            TypeClass::Float,
        ),
        (
            "ConvertFToS",
            Op::ConvertFToS,
            TypeClass::Int,
            TypeClass::Float,
        ),
        (
            "ConvertSToF",
            Op::ConvertSToF,
            TypeClass::Float,
            TypeClass::Int,
        ),
        (
            "ConvertUToF",
            Op::ConvertUToF,
            TypeClass::Float,
            TypeClass::Int,
        ),
        // Copy (typed variants)
        ("CopyI", Op::CopyObject, TypeClass::Int, TypeClass::Int),
        ("CopyF", Op::CopyObject, TypeClass::Float, TypeClass::Float),
        ("CopyB", Op::CopyObject, TypeClass::Bool, TypeClass::Bool),
    ];

    for (name, opcode, result_class, operand_class) in unary_ops {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(operand_term) = rest.strip_suffix(')') {
                // Type derivation is table-driven
                let op_result_type = resolve_type_for_class(
                    *result_class,
                    result_type,
                    type_classes,
                    int32_type,
                    float32_type,
                    bool_type,
                );
                let unary_operand_type = resolve_type_for_class(
                    *operand_class,
                    result_type,
                    type_classes,
                    int32_type,
                    float32_type,
                    bool_type,
                );

                let (operand_id, mut operand_synth) = materialize_term(
                    operand_term.trim(),
                    unary_operand_type,
                    id_map,
                    next_id,
                    int32_type,
                    int64_type,
                    float32_type,
                    _id_to_type,
                    type_classes,
                    bool_type,
                )?;
                synthesized.append(&mut operand_synth);

                let inst_id = *next_id;
                *next_id += 1;
                let inst = Instruction::new(
                    *opcode,
                    Some(op_result_type),
                    Some(inst_id),
                    vec![rspirv::dr::Operand::IdRef(operand_id)],
                );
                synthesized.push(inst);
                id_map.insert(format!("id{}", inst_id), inst_id);
                return Some((inst_id, synthesized));
            }
        }
    }

    // Select / Gamma / If (untyped and typed variants) — all map to Op::Select
    // Table-driven: each entry carries the TypeClass of the result.
    let select_ops: &[(&str, TypeClass)] = &[
        ("(Select ", TypeClass::Other),
        ("(Gamma ", TypeClass::Other),
        ("(If ", TypeClass::Other),
        ("(SelectI ", TypeClass::Int),
        ("(GammaI ", TypeClass::Int),
        ("(IfI ", TypeClass::Int),
        ("(SelectF ", TypeClass::Float),
        ("(GammaF ", TypeClass::Float),
        ("(IfF ", TypeClass::Float),
        ("(SelectB ", TypeClass::Bool),
        ("(GammaB ", TypeClass::Bool),
        ("(IfB ", TypeClass::Bool),
    ];
    for (select_prefix, select_class) in select_ops {
        if let Some(rest) = term.strip_prefix(select_prefix) {
            if let Some(rest) = rest.strip_suffix(')') {
                let terms = split_terms_simple(rest);
                if terms.len() >= 3 {
                    let select_type = resolve_type_for_class(
                        *select_class,
                        result_type,
                        type_classes,
                        int32_type,
                        float32_type,
                        bool_type,
                    );

                    // Condition is always bool
                    let cond_type = bool_type.unwrap_or(result_type);
                    let (cond_id, mut cond_synth) = materialize_term(
                        &terms[0],
                        cond_type,
                        id_map,
                        next_id,
                        int32_type,
                        int64_type,
                        float32_type,
                        _id_to_type,
                        type_classes,
                        bool_type,
                    )?;
                    synthesized.append(&mut cond_synth);

                    let (then_id, mut then_synth) = materialize_term(
                        &terms[1],
                        select_type,
                        id_map,
                        next_id,
                        int32_type,
                        int64_type,
                        float32_type,
                        _id_to_type,
                        type_classes,
                        bool_type,
                    )?;
                    synthesized.append(&mut then_synth);

                    let (else_id, mut else_synth) = materialize_term(
                        &terms[2],
                        select_type,
                        id_map,
                        next_id,
                        int32_type,
                        int64_type,
                        float32_type,
                        _id_to_type,
                        type_classes,
                        bool_type,
                    )?;
                    synthesized.append(&mut else_synth);

                    let inst_id = *next_id;
                    *next_id += 1;
                    let inst = Instruction::new(
                        Op::Select,
                        Some(select_type),
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

    #[test]
    fn type_class_of_constructor_categorizes_terms() {
        // IntExpr constructors
        assert_eq!(type_class_of_constructor("(Add x y)"), TypeClass::Int);
        assert_eq!(type_class_of_constructor("(Const 42)"), TypeClass::Int);
        assert_eq!(type_class_of_constructor("(ISym \"id5\")"), TypeClass::Int);
        assert_eq!(type_class_of_constructor("(ConvertFToS x)"), TypeClass::Int);

        // FloatExpr constructors
        assert_eq!(type_class_of_constructor("(FAdd x y)"), TypeClass::Float);
        assert_eq!(type_class_of_constructor("(FConst 3.14)"), TypeClass::Float);
        assert_eq!(
            type_class_of_constructor("(FSym \"id5\")"),
            TypeClass::Float
        );
        assert_eq!(type_class_of_constructor("(Sqrt x)"), TypeClass::Float);

        // BoolExpr constructors
        assert_eq!(type_class_of_constructor("(Eq x y)"), TypeClass::Bool);
        assert_eq!(type_class_of_constructor("(LogAnd x y)"), TypeClass::Bool);
        assert_eq!(type_class_of_constructor("(BSym \"id5\")"), TypeClass::Bool);
        assert_eq!(type_class_of_constructor("(FOrdLt x y)"), TypeClass::Bool);

        // Expr/Other
        assert_eq!(type_class_of_constructor("(VecAdd x y)"), TypeClass::Other);
        assert_eq!(type_class_of_constructor("(Sym \"id5\")"), TypeClass::Other);
        assert_eq!(type_class_of_constructor("id5"), TypeClass::Other);
    }

    #[test]
    fn resolve_type_for_class_preserves_matching_type() {
        let mut type_classes = HashMap::new();
        let int_type: Word = 2;
        let float_type: Word = 3;
        let bool_type: Word = 1;
        type_classes.insert(int_type, TypeClass::Int);
        type_classes.insert(float_type, TypeClass::Float);
        type_classes.insert(bool_type, TypeClass::Bool);

        // Int constructor with int original type → keep it
        assert_eq!(
            resolve_type_for_class(
                TypeClass::Int,
                int_type,
                &type_classes,
                Some(int_type),
                Some(float_type),
                Some(bool_type)
            ),
            int_type
        );
        // Int constructor with bool original type → correct to int
        assert_eq!(
            resolve_type_for_class(
                TypeClass::Int,
                bool_type,
                &type_classes,
                Some(int_type),
                Some(float_type),
                Some(bool_type)
            ),
            int_type
        );
        // Bool constructor with int original type → correct to bool
        assert_eq!(
            resolve_type_for_class(
                TypeClass::Bool,
                int_type,
                &type_classes,
                Some(int_type),
                Some(float_type),
                Some(bool_type)
            ),
            bool_type
        );
        // Other constructor → always keep original
        assert_eq!(
            resolve_type_for_class(
                TypeClass::Other,
                int_type,
                &type_classes,
                Some(int_type),
                Some(float_type),
                Some(bool_type)
            ),
            int_type
        );
    }
}
