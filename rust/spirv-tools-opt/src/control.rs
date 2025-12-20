use crate::SpirvLang;
use egg::{
    rewrite, AstSize, CostFunction, EGraph, Extractor, Id, RecExpr, Rewrite, Runner, Symbol,
};
use rspirv::dr::{Block, Function, Instruction};
use rspirv::spirv::Op;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct ReturnCase {
    pub block_idx: usize,
    pub label: u32,
    pub value: Option<u32>,
}

#[derive(Clone, Debug)]
pub enum ControlKind {
    Selection { cond_id: u32 },
    Switch,
}

#[derive(Clone, Debug)]
pub struct ControlCandidate {
    pub merge_idx: usize,
    pub merge_label: u32,
    pub cases: Vec<ReturnCase>,
    pub return_type: u32,
    pub kind: ControlKind,
}

#[derive(Clone, Debug)]
pub struct ControlRoot {
    pub candidate: ControlCandidate,
    pub root: Id,
}

pub fn merge_return_selections_egraph(func: &mut Function, next_id: &mut u32) -> bool {
    let candidates = match selection_candidates(func).map(|list| {
        list.into_iter()
            .filter(|candidate| matches!(candidate.kind, ControlKind::Selection { .. }))
            .collect::<Vec<_>>()
    }) {
        Some(list) => list,
        None => return false,
    };
    if candidates.is_empty() {
        return false;
    }

    let rewrites = control_rewrites();
    let mut changed = false;

    for candidate in candidates {
        let ControlKind::Selection { cond_id } = candidate.kind else {
            continue;
        };
        let Some((expr, _root)) = selection_expr(cond_id, &candidate.cases) else {
            continue;
        };
        let runner = Runner::default().with_expr(&expr).run(&rewrites);
        let Some(&root) = runner.roots.get(0) else {
            continue;
        };
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_cost, best) = extractor.find_best(root);

        let Some(root_node) = best.as_ref().last() else {
            continue;
        };
        match root_node {
            SpirvLang::Ret => {
                if candidate.cases.iter().any(|case| case.value.is_some()) {
                    continue;
                }
                if apply_merge_return_void_cases(
                    func,
                    candidate.merge_idx,
                    candidate.merge_label,
                    &candidate.cases,
                ) {
                    changed = true;
                }
            }
            SpirvLang::RetVal(child) => {
                let pairs = match extract_pairs(&best, *child) {
                    Some(pairs) => pairs,
                    None => continue,
                };
                if !pairs_match_cases(&pairs, &candidate.cases) {
                    continue;
                }
                if apply_merge_return_pairs(
                    func,
                    candidate.merge_idx,
                    candidate.merge_label,
                    &candidate.cases,
                    &pairs,
                    candidate.return_type,
                    next_id,
                ) {
                    changed = true;
                }
            }
            _ => {}
        }
    }

    changed
}

pub fn merge_return_switches_egraph(func: &mut Function, next_id: &mut u32) -> bool {
    let candidates = match switch_candidates(func).map(|list| {
        list.into_iter()
            .filter(|candidate| matches!(candidate.kind, ControlKind::Switch))
            .collect::<Vec<_>>()
    }) {
        Some(list) => list,
        None => return false,
    };
    if candidates.is_empty() {
        return false;
    }

    let rewrites = control_rewrites();
    let mut changed = false;

    for candidate in candidates {
        let Some((expr, _root)) = merge_expr(&candidate.cases) else {
            continue;
        };
        let runner = Runner::default().with_expr(&expr).run(&rewrites);
        let Some(&root) = runner.roots.get(0) else {
            continue;
        };
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_cost, best) = extractor.find_best(root);

        let Some(root_node) = best.as_ref().last() else {
            continue;
        };
        match root_node {
            SpirvLang::Ret => {
                if candidate.cases.iter().any(|case| case.value.is_some()) {
                    continue;
                }
                if apply_merge_return_void_cases(
                    func,
                    candidate.merge_idx,
                    candidate.merge_label,
                    &candidate.cases,
                ) {
                    changed = true;
                }
            }
            SpirvLang::RetVal(child) => {
                let pairs = match extract_pairs(&best, *child) {
                    Some(pairs) => pairs,
                    None => continue,
                };
                if !pairs_match_cases(&pairs, &candidate.cases) {
                    continue;
                }
                if apply_merge_return_pairs(
                    func,
                    candidate.merge_idx,
                    candidate.merge_label,
                    &candidate.cases,
                    &pairs,
                    candidate.return_type,
                    next_id,
                ) {
                    changed = true;
                }
            }
            SpirvLang::Merge(_) => {
                if candidate.cases.iter().all(|case| case.value.is_none()) {
                    if apply_merge_return_void_cases(
                        func,
                        candidate.merge_idx,
                        candidate.merge_label,
                        &candidate.cases,
                    ) {
                        changed = true;
                    }
                    continue;
                }
                let root_id = Id::from(best.as_ref().len().saturating_sub(1));
                let pairs = match extract_return_pairs(&best, root_id) {
                    Some(pairs) => pairs,
                    None => continue,
                };
                if !pairs_match_cases(&pairs, &candidate.cases) {
                    continue;
                }
                if apply_merge_return_pairs(
                    func,
                    candidate.merge_idx,
                    candidate.merge_label,
                    &candidate.cases,
                    &pairs,
                    candidate.return_type,
                    next_id,
                ) {
                    changed = true;
                }
            }
            _ => {}
        }
    }

    changed
}

pub fn control_rewrites() -> Vec<Rewrite<SpirvLang, ()>> {
    vec![
        rewrite!("merge-return-void"; "(if ?c ret ret)" => "ret"),
        rewrite!(
            "merge-return-values";
            "(if ?c (retv ?a) (retv ?b))" => "(retv (phi ?a ?b))"
        ),
        rewrite!("merge-switch-void"; "(merge ret ret)" => "ret"),
        rewrite!(
            "merge-switch-values";
            "(merge (retv ?a) (retv ?b))" => "(retv (phi ?a ?b))"
        ),
    ]
}

pub fn add_control_roots(func: &Function, egraph: &mut EGraph<SpirvLang, ()>) -> Vec<ControlRoot> {
    let mut roots = Vec::new();
    let Some(mut candidates) = selection_candidates(func) else {
        return roots;
    };
    if let Some(mut switches) = switch_candidates(func) {
        candidates.append(&mut switches);
    }

    for candidate in candidates {
        let expr = match candidate.kind {
            ControlKind::Selection { cond_id } => selection_expr(cond_id, &candidate.cases),
            ControlKind::Switch => merge_expr(&candidate.cases),
        };
        let Some((expr, _root)) = expr else {
            continue;
        };
        let root = egraph.add_expr(&expr);
        roots.push(ControlRoot { candidate, root });
    }
    roots
}

pub fn apply_control_roots<CF: CostFunction<SpirvLang>>(
    func: &mut Function,
    extractor: &Extractor<CF, SpirvLang, ()>,
    roots: &[ControlRoot],
    next_id: &mut u32,
) -> bool {
    let mut changed = false;
    for entry in roots {
        let (_cost, best) = extractor.find_best(entry.root);
        let Some(root_node) = best.as_ref().last() else {
            continue;
        };
        match root_node {
            SpirvLang::Ret => {
                if entry
                    .candidate
                    .cases
                    .iter()
                    .any(|case| case.value.is_some())
                {
                    continue;
                }
                if apply_merge_return_void_cases(
                    func,
                    entry.candidate.merge_idx,
                    entry.candidate.merge_label,
                    &entry.candidate.cases,
                ) {
                    changed = true;
                }
            }
            SpirvLang::RetVal(child) => {
                let pairs = match extract_pairs(&best, *child) {
                    Some(pairs) => pairs,
                    None => continue,
                };
                if !pairs_match_cases(&pairs, &entry.candidate.cases) {
                    continue;
                }
                if apply_merge_return_pairs(
                    func,
                    entry.candidate.merge_idx,
                    entry.candidate.merge_label,
                    &entry.candidate.cases,
                    &pairs,
                    entry.candidate.return_type,
                    next_id,
                ) {
                    changed = true;
                }
            }
            SpirvLang::Merge(_) => {
                if entry
                    .candidate
                    .cases
                    .iter()
                    .all(|case| case.value.is_none())
                {
                    if apply_merge_return_void_cases(
                        func,
                        entry.candidate.merge_idx,
                        entry.candidate.merge_label,
                        &entry.candidate.cases,
                    ) {
                        changed = true;
                    }
                    continue;
                }
                let root_id = Id::from(best.as_ref().len().saturating_sub(1));
                let pairs = match extract_return_pairs(&best, root_id) {
                    Some(pairs) => pairs,
                    None => continue,
                };
                if !pairs_match_cases(&pairs, &entry.candidate.cases) {
                    continue;
                }
                if apply_merge_return_pairs(
                    func,
                    entry.candidate.merge_idx,
                    entry.candidate.merge_label,
                    &entry.candidate.cases,
                    &pairs,
                    entry.candidate.return_type,
                    next_id,
                ) {
                    changed = true;
                }
            }
            _ => {}
        }
    }
    changed
}

fn selection_expr(cond_id: u32, cases: &[ReturnCase]) -> Option<(RecExpr<SpirvLang>, Id)> {
    if cases.len() != 2 {
        return None;
    }
    let true_case = &cases[0];
    let false_case = &cases[1];

    let mut expr = RecExpr::default();
    let cond = expr.add(SpirvLang::Symbol(symbol_for_id(cond_id)));
    let true_node = match true_case.value {
        Some(id) => {
            let pair = pair_node(&mut expr, id, true_case.label);
            expr.add(SpirvLang::RetVal(pair))
        }
        None => expr.add(SpirvLang::Ret),
    };
    let false_node = match false_case.value {
        Some(id) => {
            let pair = pair_node(&mut expr, id, false_case.label);
            expr.add(SpirvLang::RetVal(pair))
        }
        None => expr.add(SpirvLang::Ret),
    };
    let root = expr.add(SpirvLang::If([cond, true_node, false_node]));
    Some((expr, root))
}

fn merge_expr(cases: &[ReturnCase]) -> Option<(RecExpr<SpirvLang>, Id)> {
    if cases.is_empty() {
        return None;
    }
    let mut expr = RecExpr::default();
    let mut roots: Vec<Id> = Vec::new();
    for case in cases {
        let node = match case.value {
            Some(value) => {
                let pair = pair_node(&mut expr, value, case.label);
                expr.add(SpirvLang::RetVal(pair))
            }
            None => expr.add(SpirvLang::Ret),
        };
        roots.push(node);
    }

    let mut iter = roots.into_iter();
    let mut root = iter.next().unwrap_or_else(|| expr.add(SpirvLang::Ret));
    for next in iter {
        root = expr.add(SpirvLang::Merge([root, next]));
    }
    Some((expr, root))
}

fn pair_node(expr: &mut RecExpr<SpirvLang>, value: u32, label: u32) -> Id {
    let val_sym = expr.add(SpirvLang::Symbol(symbol_for_id(value)));
    let label_sym = expr.add(SpirvLang::Symbol(symbol_for_id(label)));
    expr.add(SpirvLang::Pair([val_sym, label_sym]))
}

fn extract_pairs(expr: &RecExpr<SpirvLang>, root: Id) -> Option<Vec<(u32, u32)>> {
    fn collect(expr: &RecExpr<SpirvLang>, node: Id, out: &mut Vec<(u32, u32)>) -> bool {
        match &expr[node] {
            SpirvLang::Pair([value_id, label_id]) => {
                let Some(value) = symbol_id(&expr[*value_id]) else {
                    return false;
                };
                let Some(label) = symbol_id(&expr[*label_id]) else {
                    return false;
                };
                out.push((value, label));
                true
            }
            SpirvLang::Phi([left, right]) => {
                collect(expr, *left, out) && collect(expr, *right, out)
            }
            _ => false,
        }
    }

    let mut out = Vec::new();
    if collect(expr, root, &mut out) {
        Some(out)
    } else {
        None
    }
}

fn extract_return_pairs(expr: &RecExpr<SpirvLang>, root: Id) -> Option<Vec<(u32, u32)>> {
    fn collect_return(expr: &RecExpr<SpirvLang>, node: Id, out: &mut Vec<(u32, u32)>) -> bool {
        match &expr[node] {
            SpirvLang::Merge([left, right]) => {
                collect_return(expr, *left, out) && collect_return(expr, *right, out)
            }
            SpirvLang::RetVal(inner) => extract_pairs(expr, *inner)
                .map(|pairs| {
                    out.extend(pairs);
                    true
                })
                .unwrap_or(false),
            _ => false,
        }
    }

    let mut out = Vec::new();
    if collect_return(expr, root, &mut out) {
        Some(out)
    } else {
        None
    }
}

fn pairs_match_cases(pairs: &[(u32, u32)], cases: &[ReturnCase]) -> bool {
    if pairs.len() != cases.len() {
        return false;
    }
    let mut expected = HashSet::new();
    for case in cases {
        let Some(value) = case.value else {
            return false;
        };
        expected.insert((value, case.label));
    }
    if expected.len() != cases.len() {
        return false;
    }
    let actual: HashSet<_> = pairs.iter().copied().collect();
    actual == expected
}

fn symbol_for_id(id: u32) -> Symbol {
    Symbol::from(format!("id{id}"))
}

fn symbol_id(node: &SpirvLang) -> Option<u32> {
    let SpirvLang::Symbol(sym) = node else {
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

fn selection_candidates(func: &Function) -> Option<Vec<ControlCandidate>> {
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
        let true_val = match func
            .blocks
            .get(true_idx)
            .and_then(|b| b.instructions.last())
            .and_then(return_value_operand)
        {
            Some(val) => val,
            None => continue,
        };
        let false_val = match func
            .blocks
            .get(false_idx)
            .and_then(|b| b.instructions.last())
            .and_then(return_value_operand)
        {
            Some(val) => val,
            None => continue,
        };
        if true_val.is_some() != false_val.is_some() {
            continue;
        }

        candidates.push(ControlCandidate {
            merge_idx,
            merge_label,
            cases: vec![
                ReturnCase {
                    block_idx: true_idx,
                    label: true_label,
                    value: true_val,
                },
                ReturnCase {
                    block_idx: false_idx,
                    label: false_label,
                    value: false_val,
                },
            ],
            return_type,
            kind: ControlKind::Selection { cond_id },
        });
    }

    Some(candidates)
}

fn switch_candidates(func: &Function) -> Option<Vec<ControlCandidate>> {
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
        if terminator.class.opcode != Op::Switch {
            continue;
        }
        let merge_label = merge_inst.operands.get(0).and_then(|op| op.id_ref_any());
        let default_label = terminator.operands.get(1).and_then(|op| op.id_ref_any());
        let Some(merge_label) = merge_label else {
            continue;
        };
        let Some(default_label) = default_label else {
            continue;
        };
        let Some(&merge_idx) = label_to_idx.get(&merge_label) else {
            continue;
        };
        if !preds
            .get(merge_idx)
            .map(|incoming| incoming.is_empty())
            .unwrap_or(false)
        {
            continue;
        }

        let mut labels = Vec::new();
        labels.push(default_label);
        let mut iter = terminator.operands.iter().skip(2);
        while let Some(_literal) = iter.next() {
            let Some(label_op) = iter.next() else {
                labels.clear();
                break;
            };
            let rspirv::dr::Operand::IdRef(label) = label_op else {
                labels.clear();
                break;
            };
            labels.push(*label);
        }
        if labels.is_empty() {
            continue;
        }

        let mut seen = HashSet::new();
        let mut cases = Vec::new();
        let mut value_kind = None;
        for label in labels {
            if label == merge_label {
                cases.clear();
                break;
            }
            if !seen.insert(label) {
                cases.clear();
                break;
            }
            let Some(&block_idx) = label_to_idx.get(&label) else {
                cases.clear();
                break;
            };
            let value = match func
                .blocks
                .get(block_idx)
                .and_then(|b| b.instructions.last())
                .and_then(return_value_operand)
            {
                Some(val) => val,
                None => {
                    cases.clear();
                    break;
                }
            };
            if value_kind.map_or(false, |is_value| is_value != value.is_some()) {
                cases.clear();
                break;
            }
            value_kind = Some(value.is_some());
            cases.push(ReturnCase {
                block_idx,
                label,
                value,
            });
        }

        if cases.len() < 2 {
            continue;
        }

        candidates.push(ControlCandidate {
            merge_idx,
            merge_label,
            cases,
            return_type,
            kind: ControlKind::Switch,
        });
    }

    Some(candidates)
}

fn apply_merge_return_void_cases(
    func: &mut Function,
    merge_idx: usize,
    merge_label: u32,
    cases: &[ReturnCase],
) -> bool {
    let branch_inst = Instruction::new(
        Op::Branch,
        None,
        None,
        vec![rspirv::dr::Operand::IdRef(merge_label)],
    );
    for case in cases {
        if !replace_terminator_with_branch(func, case.block_idx, branch_inst.clone()) {
            return false;
        }
    }
    let Some(merge_block) = func.blocks.get_mut(merge_idx) else {
        return false;
    };
    merge_block.instructions.clear();
    merge_block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    true
}

fn apply_merge_return_pairs(
    func: &mut Function,
    merge_idx: usize,
    merge_label: u32,
    cases: &[ReturnCase],
    pairs: &[(u32, u32)],
    return_type: u32,
    next_id: &mut u32,
) -> bool {
    let branch_inst = Instruction::new(
        Op::Branch,
        None,
        None,
        vec![rspirv::dr::Operand::IdRef(merge_label)],
    );
    for case in cases {
        if !replace_terminator_with_branch(func, case.block_idx, branch_inst.clone()) {
            return false;
        }
    }
    let Some(merge_block) = func.blocks.get_mut(merge_idx) else {
        return false;
    };
    merge_block.instructions.clear();
    let phi_id = *next_id;
    *next_id = next_id.saturating_add(1);
    let mut phi_ops = Vec::with_capacity(pairs.len() * 2);
    for (value, label) in pairs {
        phi_ops.push(rspirv::dr::Operand::IdRef(*value));
        phi_ops.push(rspirv::dr::Operand::IdRef(*label));
    }
    merge_block.instructions.push(Instruction::new(
        Op::Phi,
        Some(return_type),
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
