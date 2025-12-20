use egg::{define_language, rewrite, AstSize, Extractor, Id, RecExpr, Rewrite, Runner, Symbol};
use rspirv::dr::{Block, Function, Instruction};
use rspirv::spirv::Op;
use std::collections::HashMap;

define_language! {
    enum ControlLang {
        "if" = If([Id; 3]),
        "ret" = Ret,
        "retv" = RetVal(Id),
        "phi" = Phi([Id; 2]),
        Symbol(Symbol),
    }
}

#[derive(Clone)]
struct SelectionCandidate {
    merge_idx: usize,
    merge_label: u32,
    true_idx: usize,
    false_idx: usize,
    true_label: u32,
    false_label: u32,
    cond_id: u32,
    true_val: Option<u32>,
    false_val: Option<u32>,
    return_type: u32,
}

pub fn merge_return_selections_egraph(func: &mut Function, next_id: &mut u32) -> bool {
    let candidates = match selection_candidates(func) {
        Some(list) => list,
        None => return false,
    };
    if candidates.is_empty() {
        return false;
    }

    let rewrites = merge_return_rewrites();
    let mut changed = false;

    for candidate in candidates {
        let (expr, root) =
            selection_expr(candidate.cond_id, candidate.true_val, candidate.false_val);
        let runner = Runner::default().with_expr(&expr).run(&rewrites);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_cost, best) = extractor.find_best(root);

        let Some(root_node) = best.as_ref().last() else {
            continue;
        };
        match root_node {
            ControlLang::Ret => {
                if candidate.true_val.is_some() || candidate.false_val.is_some() {
                    continue;
                }
                if apply_merge_return_void(func, &candidate) {
                    changed = true;
                }
            }
            ControlLang::RetVal(child) => {
                let child_idx = usize::from(*child);
                let Some(ControlLang::Phi([a, b])) = best.as_ref().get(child_idx) else {
                    continue;
                };
                let Some(true_value) = best.as_ref().get(usize::from(*a)).and_then(symbol_id)
                else {
                    continue;
                };
                let Some(false_value) = best.as_ref().get(usize::from(*b)).and_then(symbol_id)
                else {
                    continue;
                };
                if apply_merge_return_value(func, &candidate, true_value, false_value, next_id) {
                    changed = true;
                }
            }
            _ => {}
        }
    }

    changed
}

fn merge_return_rewrites() -> Vec<Rewrite<ControlLang, ()>> {
    vec![
        rewrite!("merge-return-void"; "(if ?c ret ret)" => "ret"),
        rewrite!(
            "merge-return-values";
            "(if ?c (retv ?a) (retv ?b))" => "(retv (phi ?a ?b))"
        ),
    ]
}

fn selection_expr(
    cond_id: u32,
    true_val: Option<u32>,
    false_val: Option<u32>,
) -> (RecExpr<ControlLang>, Id) {
    let mut expr = RecExpr::default();
    let cond = expr.add(ControlLang::Symbol(symbol_for_id(cond_id)));
    let true_node = match true_val {
        Some(id) => {
            let sym = expr.add(ControlLang::Symbol(symbol_for_id(id)));
            expr.add(ControlLang::RetVal(sym))
        }
        None => expr.add(ControlLang::Ret),
    };
    let false_node = match false_val {
        Some(id) => {
            let sym = expr.add(ControlLang::Symbol(symbol_for_id(id)));
            expr.add(ControlLang::RetVal(sym))
        }
        None => expr.add(ControlLang::Ret),
    };
    let root = expr.add(ControlLang::If([cond, true_node, false_node]));
    (expr, root)
}

fn symbol_for_id(id: u32) -> Symbol {
    Symbol::from(format!("id{id}"))
}

fn symbol_id(node: &ControlLang) -> Option<u32> {
    let ControlLang::Symbol(sym) = node else {
        return None;
    };
    sym.as_str().strip_prefix("id")?.parse().ok()
}

fn return_value_operand(inst: &Instruction) -> Option<Option<u32>> {
    match inst.class.opcode {
        Op::Return => Some(None),
        Op::ReturnValue => inst
            .operands
            .get(0)
            .and_then(|op| op.id_ref_any())
            .map(Some),
        _ => None,
    }
}

fn selection_candidates(func: &Function) -> Option<Vec<SelectionCandidate>> {
    let label_to_idx = label_to_index(func)?;
    let preds = compute_predecessors(func, &label_to_idx)?;
    let return_type = func.def.as_ref()?.result_type?;
    let mut candidates = Vec::new();

    for block in &func.blocks {
        let merge_inst = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == Op::SelectionMerge);
        let terminator = block.instructions.last();
        let (Some(merge_inst), Some(terminator)) = (merge_inst, terminator) else {
            continue;
        };
        if terminator.class.opcode != Op::BranchConditional {
            continue;
        }
        let merge_label = merge_inst.operands.get(0).and_then(|op| op.id_ref_any());
        let true_label = terminator.operands.get(1).and_then(|op| op.id_ref_any());
        let false_label = terminator.operands.get(2).and_then(|op| op.id_ref_any());
        let cond_id = terminator.operands.get(0).and_then(|op| op.id_ref_any());
        let (Some(merge_label), Some(true_label), Some(false_label), Some(cond_id)) =
            (merge_label, true_label, false_label, cond_id)
        else {
            continue;
        };
        let Some(&merge_idx) = label_to_idx.get(&merge_label) else {
            continue;
        };
        let Some(&true_idx) = label_to_idx.get(&true_label) else {
            continue;
        };
        let Some(&false_idx) = label_to_idx.get(&false_label) else {
            continue;
        };
        if merge_idx == true_idx || merge_idx == false_idx || true_idx == false_idx {
            continue;
        }
        if !preds
            .get(merge_idx)
            .map(|incoming| incoming.is_empty())
            .unwrap_or(false)
        {
            continue;
        }
        let true_term = func
            .blocks
            .get(true_idx)
            .and_then(|b| b.instructions.last());
        let false_term = func
            .blocks
            .get(false_idx)
            .and_then(|b| b.instructions.last());
        let (Some(true_term), Some(false_term)) = (true_term, false_term) else {
            continue;
        };
        let Some(true_val) = return_value_operand(true_term) else {
            continue;
        };
        let Some(false_val) = return_value_operand(false_term) else {
            continue;
        };
        if true_val.is_some() != false_val.is_some() {
            continue;
        }

        candidates.push(SelectionCandidate {
            merge_idx,
            merge_label,
            true_idx,
            false_idx,
            true_label,
            false_label,
            cond_id,
            true_val,
            false_val,
            return_type,
        });
    }

    Some(candidates)
}

fn apply_merge_return_void(func: &mut Function, candidate: &SelectionCandidate) -> bool {
    let branch_inst = Instruction::new(
        Op::Branch,
        None,
        None,
        vec![rspirv::dr::Operand::IdRef(candidate.merge_label)],
    );
    if !replace_terminator_with_branch(func, candidate.true_idx, branch_inst.clone()) {
        return false;
    }
    if !replace_terminator_with_branch(func, candidate.false_idx, branch_inst.clone()) {
        return false;
    }
    let Some(merge_block) = func.blocks.get_mut(candidate.merge_idx) else {
        return false;
    };
    merge_block.instructions.clear();
    merge_block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    true
}

fn apply_merge_return_value(
    func: &mut Function,
    candidate: &SelectionCandidate,
    true_value: u32,
    false_value: u32,
    next_id: &mut u32,
) -> bool {
    let branch_inst = Instruction::new(
        Op::Branch,
        None,
        None,
        vec![rspirv::dr::Operand::IdRef(candidate.merge_label)],
    );
    if !replace_terminator_with_branch(func, candidate.true_idx, branch_inst.clone()) {
        return false;
    }
    if !replace_terminator_with_branch(func, candidate.false_idx, branch_inst.clone()) {
        return false;
    }
    let Some(merge_block) = func.blocks.get_mut(candidate.merge_idx) else {
        return false;
    };
    merge_block.instructions.clear();
    let phi_id = *next_id;
    *next_id = next_id.saturating_add(1);
    let phi_ops = vec![
        rspirv::dr::Operand::IdRef(true_value),
        rspirv::dr::Operand::IdRef(candidate.true_label),
        rspirv::dr::Operand::IdRef(false_value),
        rspirv::dr::Operand::IdRef(candidate.false_label),
    ];
    merge_block.instructions.push(Instruction::new(
        Op::Phi,
        Some(candidate.return_type),
        Some(phi_id),
        phi_ops,
    ));
    merge_block.instructions.push(Instruction::new(
        Op::ReturnValue,
        None,
        None,
        vec![rspirv::dr::Operand::IdRef(phi_id)],
    ));
    true
}

fn replace_terminator_with_branch(
    func: &mut Function,
    block_idx: usize,
    branch_inst: Instruction,
) -> bool {
    let Some(block) = func.blocks.get_mut(block_idx) else {
        return false;
    };
    let Some(last) = block.instructions.last() else {
        return false;
    };
    if !matches!(last.class.opcode, Op::Return | Op::ReturnValue) {
        return false;
    }
    block.instructions.pop();
    block.instructions.push(branch_inst);
    true
}

fn label_to_index(func: &Function) -> Option<HashMap<u32, usize>> {
    let mut label_to_idx: HashMap<u32, usize> = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        let label_id = block.label.as_ref()?.result_id?;
        label_to_idx.insert(label_id, idx);
    }
    Some(label_to_idx)
}

fn compute_predecessors(
    func: &Function,
    label_to_idx: &HashMap<u32, usize>,
) -> Option<Vec<Vec<usize>>> {
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); func.blocks.len()];
    for (idx, block) in func.blocks.iter().enumerate() {
        for succ in block_successors(block, label_to_idx)? {
            preds[succ].push(idx);
        }
    }
    Some(preds)
}

fn block_successors(block: &Block, label_to_idx: &HashMap<u32, usize>) -> Option<Vec<usize>> {
    let Some(last) = block.instructions.last() else {
        return Some(Vec::new());
    };
    use rspirv::dr::Operand::*;
    let mut targets = Vec::new();
    match last.class.opcode {
        Op::Branch => {
            if let Some(IdRef(label)) = last.operands.get(0) {
                if let Some(idx) = label_to_idx.get(label) {
                    targets.push(*idx);
                }
            }
        }
        Op::BranchConditional => {
            if let Some(IdRef(true_lab)) = last.operands.get(1) {
                if let Some(idx) = label_to_idx.get(true_lab) {
                    targets.push(*idx);
                }
            }
            if let Some(IdRef(false_lab)) = last.operands.get(2) {
                if let Some(idx) = label_to_idx.get(false_lab) {
                    targets.push(*idx);
                }
            }
        }
        Op::Switch => {
            for op in last.operands.iter().skip(1) {
                if let IdRef(label) = op {
                    if let Some(idx) = label_to_idx.get(label) {
                        targets.push(*idx);
                    }
                }
            }
        }
        Op::Return | Op::ReturnValue | Op::Unreachable => {}
        _ => {}
    }
    Some(targets)
}
