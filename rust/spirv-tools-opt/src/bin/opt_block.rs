use clap::Parser;
use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use spirv_tools_opt::translate::{
    optimize_arith_block_with_types, rebuild_arith_with_original_ids,
    translate_arith_with_types_dominated, type_widths_from_module,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Optimize a single SPIR-V basic block using the Rust e-graph optimizer."
)]
struct Args {
    /// Input SPIR-V binary; required.
    input: PathBuf,
    /// Optional output path; writes to stdout when omitted.
    output: Option<PathBuf>,
    /// Force the Rust optimizer even if SPIRV_TOOLS_DISABLE_RUST_OPT is set.
    #[arg(long, default_value_t = false)]
    force_rust: bool,
    /// Skip optimization and emit the input unchanged.
    #[arg(long, default_value_t = false)]
    passthrough: bool,
    /// Disable the global (multi-block) optimizer path even when available.
    #[arg(long, default_value_t = false)]
    disable_global: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let input_bytes = fs::read(&args.input)?;
    let words = bytes_to_words(&input_bytes)?;

    let optimized = optimize_module(
        &words,
        args.force_rust,
        args.passthrough,
        args.disable_global,
    )?;
    let output_bytes = words_to_bytes(&optimized);

    if let Some(path) = args.output {
        fs::write(path, output_bytes)?;
    } else {
        std::io::stdout().write_all(&output_bytes)?;
    }
    Ok(())
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    if !bytes.len().is_multiple_of(4) {
        return Err("input length is not a multiple of 4 bytes".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            u32::from_le_bytes(arr)
        })
        .collect())
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn is_arith(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Constant
            | Op::IAdd
            | Op::IMul
            | Op::ISub
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::BitwiseAnd
            | Op::Not
            | Op::SNegate
            | Op::SDiv
            | Op::UDiv
            | Op::SRem
            | Op::UMod
            | Op::ShiftLeftLogical
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
    )
}

fn optimize_module(
    words: &[u32],
    force_rust: bool,
    passthrough: bool,
    disable_global: bool,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    if passthrough {
        return Ok(words.to_vec());
    }
    let force_env = std::env::var_os("SPIRV_TOOLS_FORCE_RUST_OPT").is_some();
    if !force_rust && matches!(env::var("SPIRV_TOOLS_DISABLE_RUST_OPT"), Ok(v) if v == "1") {
        return Ok(words.to_vec());
    }
    let _ = force_env; // reserved for future use; disable env is authoritative unless force flag.

    let mut loader = rspirv::dr::Loader::new();
    parse_words(words, &mut loader)?;
    let mut module = loader.module();
    let type_widths = type_widths_from_module(&module);

    let non_constant_globals: Vec<Instruction> = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode != Op::Constant)
        .cloned()
        .collect();
    let mut constant_map: BTreeMap<u32, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
        .filter(|(_, inst)| inst.class.opcode == Op::Constant)
        .collect();

    let mut preserved_roots: Vec<u32> = Vec::new();

    for func in &mut module.functions {
        for block in &func.blocks {
            if let Some(root_id) = block
                .instructions
                .iter()
                .rev()
                .find(|inst| is_arith(inst.class.opcode))
                .and_then(|inst| inst.result_id)
            {
                preserved_roots.push(root_id);
            }
        }

        let disable_global =
            disable_global || env::var("SPIRV_TOOLS_DISABLE_GLOBAL_OPT").is_ok();
        if func.blocks.len() > 1 && !disable_global {
            if let Some((new_constants, optimized_blocks)) =
                optimize_function_global(func, &type_widths, &constant_map)
            {
                constant_map = new_constants;
                for (block, optimized_block) in func.blocks.iter_mut().zip(optimized_blocks) {
                    replace_block_arith(block, &optimized_block);
                }
                continue;
            }
        }

        for block in &mut func.blocks {
            let original_block = block.instructions.clone();
            let arithmetic: Vec<_> = original_block
                .iter()
                .filter(|inst| is_arith(inst.class.opcode))
                .cloned()
                .collect();
            if arithmetic.is_empty() {
                continue;
            }

            let mut arith_stream = Vec::new();
            arith_stream.extend(constant_map.values().cloned());
            arith_stream.extend(arithmetic.clone());

            let optimized = optimize_arith_block_with_types(&arith_stream, &type_widths)
                .map_err(|e| format!("{e}"))?;

            let mut optimized_block = Vec::new();
            for inst in optimized {
                if inst.class.opcode == Op::Constant {
                    if let Some(id) = inst.result_id {
                        constant_map.insert(id, inst);
                    }
                } else {
                    optimized_block.push(inst);
                }
            }

            let mut new_block = Vec::new();
            let mut inserted = false;
            for inst in original_block {
                if is_arith(inst.class.opcode) {
                    if !inserted {
                        new_block.extend(optimized_block.clone());
                        inserted = true;
                    }
                    continue;
                }
                new_block.push(inst);
            }
            if !inserted {
                new_block.extend(optimized_block);
            }
            block.instructions = new_block;
        }
    }

    module.types_global_values = non_constant_globals;
    module
        .types_global_values
        .extend(constant_map.into_values());

    dead_code_eliminate(&mut module, &preserved_roots);

    Ok(module.assemble())
}

fn dead_code_eliminate(module: &mut rspirv::dr::Module, preserved_roots: &[u32]) {
    let mut candidate_operands: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut worklist: Vec<u32> = Vec::new();
    let mut live: HashSet<u32> = HashSet::new();

    let is_candidate = |op: Op| is_arith(op) || op == Op::Constant;

    let mut record_candidate = |inst: &Instruction| {
        if let Some(id) = inst.result_id {
            if is_candidate(inst.class.opcode) {
                let mut ops = collect_id_operands(inst);
                if let Some(rt) = inst.result_type {
                    ops.push(rt);
                }
                candidate_operands.insert(id, ops);
                return true;
            }
        }
        false
    };

    for func in &module.functions {
        for param in &func.parameters {
            worklist.extend(collect_id_operands(param));
            if let Some(rt) = param.result_type {
                worklist.push(rt);
            }
        }
        for block in &func.blocks {
            let mut last_candidate = None;
            for inst in &block.instructions {
                if record_candidate(inst) {
                    last_candidate = inst.result_id;
                    continue;
                }
                worklist.extend(collect_id_operands(inst));
                if let Some(rt) = inst.result_type {
                    worklist.push(rt);
                }
            }
            if let Some(id) = last_candidate {
                worklist.push(id);
            }
        }
    }

    for inst in &module.types_global_values {
        if record_candidate(inst) {
            continue;
        }
        worklist.extend(collect_id_operands(inst));
        if let Some(rt) = inst.result_type {
            worklist.push(rt);
        }
    }

    worklist.extend(preserved_roots.iter().copied());

    while let Some(id) = worklist.pop() {
        if !live.insert(id) {
            continue;
        }
        if let Some(ops) = candidate_operands.get(&id) {
            worklist.extend(ops.iter().copied());
        }
    }

    for func in &mut module.functions {
        for block in &mut func.blocks {
            block.instructions.retain(|inst| {
                if let Some(id) = inst.result_id {
                    if is_candidate(inst.class.opcode) {
                        return live.contains(&id);
                    }
                }
                true
            });
        }
    }

    module.types_global_values.retain(|inst| {
        if inst.class.opcode == Op::Constant {
            if let Some(id) = inst.result_id {
                return live.contains(&id);
            }
        }
        true
    });
}

fn collect_id_operands(inst: &Instruction) -> Vec<u32> {
    inst.operands
        .iter()
        .filter_map(|op| match op {
            rspirv::dr::Operand::IdRef(id)
            | rspirv::dr::Operand::IdScope(id)
            | rspirv::dr::Operand::IdMemorySemantics(id) => Some(*id),
            _ => None,
        })
        .collect()
}

fn replace_block_arith(block: &mut rspirv::dr::Block, optimized_block: &[Instruction]) {
    let original_block = block.instructions.clone();
    let mut new_block = Vec::new();
    let mut inserted = false;
    for inst in original_block {
        if is_arith(inst.class.opcode) {
            if !inserted {
                new_block.extend_from_slice(optimized_block);
                inserted = true;
            }
            continue;
        }
        new_block.push(inst);
    }
    if !inserted {
        new_block.extend_from_slice(optimized_block);
    }
    block.instructions = new_block;
}

fn optimize_function_global(
    func: &rspirv::dr::Function,
    type_widths: &HashMap<u32, u32>,
    constant_map: &BTreeMap<u32, Instruction>,
) -> Option<(BTreeMap<u32, Instruction>, Vec<Vec<Instruction>>)> {
    let (arithmetic, id_to_block) = collect_arith_topo(func)?;

    if arithmetic.is_empty() {
        return None;
    }

    // Enforce block dominance for intra-function dependencies; if a value's
    // defining block does not dominate the use, fall back to the block-local path.
    let block_dominators = compute_block_dominators(func)?;
    // Quick sanity: block-order dominance check to catch obvious forward uses.
    for inst in &arithmetic {
        let Some(result_id) = inst.result_id else { continue };
        let Some(&block_idx) = id_to_block.get(&result_id) else { continue };
        for op_id in collect_id_operands(inst) {
            if let Some(&op_block) = id_to_block.get(&op_id) {
                if op_block > block_idx {
                    return None;
                }
                if !block_dominators
                    .get(&block_idx)
                    .map(|doms| doms.contains(&op_block))
                    .unwrap_or(false)
                {
                    return None;
                }
            }
        }
    }

    let mut arith_stream = Vec::new();
    arith_stream.extend(constant_map.values().cloned());
    arith_stream.extend(arithmetic.clone());

    let translated =
        translate_arith_with_types_dominated(&arith_stream, type_widths).ok()?;
    let optimized_expr = spirv_tools_opt::optimize_translated(&translated);
    if optimized_expr.as_ref().len() != translated.original_ids.len() {
        return None;
    }
    let optimized = rebuild_arith_with_original_ids(&optimized_expr, &translated).ok()?;

    let original_ids: HashSet<u32> = arith_stream
        .iter()
        .filter_map(|inst| inst.result_id)
        .collect();

    let mut new_constants = constant_map.clone();
    let mut per_block = vec![Vec::new(); func.blocks.len()];

    for inst in optimized {
        let id = inst.result_id?;
        if !original_ids.contains(&id) {
            return None;
        }
        if inst.class.opcode == Op::Constant {
            new_constants.insert(id, inst);
            continue;
        }
        let block_idx = match id_to_block.get(&id) {
            Some(idx) => *idx,
            None => return None,
        };
        per_block[block_idx].push(inst);
    }

    Some((new_constants, per_block))
}

fn collect_arith_topo(
    func: &rspirv::dr::Function,
) -> Option<(Vec<Instruction>, HashMap<u32, usize>)> {
    let mut id_to_inst: HashMap<u32, Instruction> = HashMap::new();
    let mut id_to_block: HashMap<u32, usize> = HashMap::new();
    let mut deps: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut indegree: HashMap<u32, usize> = HashMap::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if !is_arith(inst.class.opcode) {
                continue;
            }
            let id = inst.result_id?;
            indegree.entry(id).or_default();
            id_to_inst.insert(id, inst.clone());
            id_to_block.insert(id, block_idx);
        }
    }

    for inst in id_to_inst.values() {
        let this_id = inst.result_id?;
        for op_id in collect_id_operands(inst) {
            if id_to_inst.contains_key(&op_id) {
                deps.entry(op_id).or_default().push(this_id);
                *indegree.entry(this_id).or_default() += 1;
            }
        }
    }

    let mut queue: Vec<u32> = indegree
        .iter()
        .filter_map(|(id, deg)| (*deg == 0).then_some(*id))
        .collect();
    let mut ordered = Vec::new();
    while let Some(id) = queue.pop() {
        if let Some(inst) = id_to_inst.get(&id) {
            ordered.push(inst.clone());
        }
        if let Some(nexts) = deps.get(&id) {
            for &n in nexts {
                if let Some(entry) = indegree.get_mut(&n) {
                    *entry -= 1;
                    if *entry == 0 {
                        queue.push(n);
                    }
                }
            }
        }
    }

    if ordered.len() != id_to_inst.len() {
        return None;
    }

    Some((ordered, id_to_block))
}

fn compute_block_dominators(
    func: &rspirv::dr::Function,
) -> Option<HashMap<usize, HashSet<usize>>> {
    let mut label_to_idx: HashMap<u32, usize> = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        let label_id = block.label.as_ref()?.result_id?;
        label_to_idx.insert(label_id, idx);
    }
    let block_count = func.blocks.len();
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); block_count];
    for (idx, block) in func.blocks.iter().enumerate() {
        succs[idx] = block_successors(block, &label_to_idx)?;
    }

    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); block_count];
    for (idx, succ_list) in succs.iter().enumerate() {
        for &succ in succ_list {
            preds[succ].push(idx);
        }
    }

    let mut dom: Vec<HashSet<usize>> = vec![HashSet::new(); block_count];
    for i in 0..block_count {
        if i == 0 {
            dom[i].insert(i);
        } else if preds[i].is_empty() {
            dom[i].insert(i);
        } else {
            dom[i] = (0..block_count).collect();
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for b in 1..block_count {
            if preds[b].is_empty() {
                continue;
            }
            let mut new_dom: HashSet<usize> = preds[b]
                .iter()
                .map(|p| dom[*p].clone())
                .reduce(|a, b| a.intersection(&b).copied().collect())
                .unwrap_or_default();
            new_dom.insert(b);
            if new_dom != dom[b] {
                dom[b] = new_dom;
                changed = true;
            }
        }
    }

    let mut result = HashMap::new();
    for (idx, set) in dom.into_iter().enumerate() {
        result.insert(idx, set);
    }
    Some(result)
}

fn block_successors(
    block: &rspirv::dr::Block,
    label_to_idx: &HashMap<u32, usize>,
) -> Option<Vec<usize>> {
    let Some(last) = block.instructions.last() else { return Some(Vec::new()) };
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
            // operands: condition, true label, false label
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
            // operands: selector, default label, literal/label pairs
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

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::dr::{Block, Function};
    use rspirv::spirv::Word;

    fn type_int(id: Word, width: u32) -> Instruction {
        Instruction::new(
            Op::TypeInt,
            None,
            Some(id),
            vec![
                rspirv::dr::Operand::LiteralBit32(width),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        )
    }

    fn make_const(id: Word, ty: Word, value: u32) -> Instruction {
        Instruction::new(
            Op::Constant,
            Some(ty),
            Some(id),
            vec![rspirv::dr::Operand::LiteralBit32(value)],
        )
    }

    #[test]
    fn global_optimizer_bails_on_forward_block_use() {
        // Block0 uses id 3, which is produced in block1. The global optimizer
        // should refuse and let the per-block path handle the function.
        let _ty_int = type_int(1, 32);
        let c_two = make_const(2, 1, 2);
        let block0_add = Instruction::new(
            Op::IAdd,
            Some(1),
            Some(10),
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::IdRef(2),
            ],
        );
        let block1_mul = Instruction::new(
            Op::IMul,
            Some(1),
            Some(3),
            vec![
                rspirv::dr::Operand::IdRef(2),
                rspirv::dr::Operand::IdRef(2),
            ],
        );

        let mut func = Function::new();
        func.blocks.push(Block {
            label: None,
            instructions: vec![block0_add.clone()],
        });
        func.blocks.push(Block {
            label: None,
            instructions: vec![c_two.clone(), block1_mul.clone()],
        });

        let type_widths = HashMap::from_iter([(1u32, 32u32)]);
        let constant_map = BTreeMap::new();

        let res = optimize_function_global(&func, &type_widths, &constant_map);
        assert!(
            res.is_none(),
            "global optimizer should decline when operands are defined in later blocks"
        );
    }

    fn block_with_label(label: Word, instructions: Vec<Instruction>) -> Block {
        Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), Vec::new())),
            instructions,
        }
    }

    fn branch(target: Word) -> Instruction {
        Instruction::new(Op::Branch, None, None, vec![rspirv::dr::Operand::IdRef(target)])
    }

    fn branch_cond(true_t: Word, false_t: Word) -> Instruction {
        Instruction::new(
            Op::BranchConditional,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(999), // condition placeholder
                rspirv::dr::Operand::IdRef(true_t),
                rspirv::dr::Operand::IdRef(false_t),
            ],
        )
    }

    #[test]
    fn dominator_blocks_non_dominating_defs() {
        // def in left branch does not dominate merge, so global optimization
        // should refuse.
        let ty = type_int(1, 32);
        let def_left = make_const(5, 1, 4);
        let use_merge = Instruction::new(
            Op::IAdd,
            Some(1),
            Some(10),
            vec![
                rspirv::dr::Operand::IdRef(5),
                rspirv::dr::Operand::IdRef(5),
            ],
        );

        let mut func = Function::new();
        func.blocks.push(block_with_label(
            100,
            vec![ty.clone(), branch_cond(101, 102)],
        ));
        func.blocks
            .push(block_with_label(101, vec![def_left.clone(), branch(103)]));
        func.blocks
            .push(block_with_label(102, vec![branch(103)]));
        func.blocks.push(block_with_label(103, vec![use_merge.clone()]));

        let type_widths = HashMap::from_iter([(1u32, 32u32)]);
        let constant_map = BTreeMap::new();
        let res = optimize_function_global(&func, &type_widths, &constant_map);
        assert!(
            res.is_none(),
            "non-dominating defs must fall back to block-local optimization"
        );
    }

    #[test]
    fn compute_dominators_handles_switch_and_unreachable() {
        // entry -> switch to two cases, plus an unreachable block with no preds.
        let mut func = Function::new();
        func.blocks.push(block_with_label(
            1,
            vec![branch_cond(2, 3)], // not exactly switch, but multiple succs
        ));
        func.blocks.push(block_with_label(2, vec![branch(4)]));
        func.blocks.push(block_with_label(3, vec![branch(4)]));
        func.blocks.push(block_with_label(4, vec![Instruction::new(
            Op::Return,
            None,
            None,
            Vec::new(),
        )]));
        // unreachable block with no predecessors
        func.blocks
            .push(block_with_label(99, vec![Instruction::new(Op::Return, None, None, Vec::new())]));

        let dom = compute_block_dominators(&func).expect("dominators");
        // Block 0 dominates 1,2,3,4 but not unreachable 5 (idx 4).
        assert!(dom.get(&1).unwrap().contains(&0));
        assert!(dom.get(&2).unwrap().contains(&0));
        assert!(dom.get(&3).unwrap().contains(&0));
        assert!(dom.get(&4).unwrap().contains(&4)); // only itself
        assert!(!dom.get(&1).unwrap().contains(&4));
    }

    #[test]
    fn compute_dominators_handles_loops() {
        // entry -> header -> body -> header (backedge) -> exit.
        let mut func = Function::new();
        func.blocks.push(block_with_label(10, vec![branch(20)]));
        func.blocks.push(block_with_label(
            20,
            vec![branch_cond(30, 40)], // branch to body or exit
        ));
        func.blocks.push(block_with_label(30, vec![branch(20)])); // backedge
        func.blocks.push(block_with_label(
            40,
            vec![Instruction::new(Op::Return, None, None, Vec::new())],
        ));

        let dom = compute_block_dominators(&func).expect("dominators");
        // Header is dominated by entry
        assert!(dom.get(&1).unwrap().contains(&0));
        // Body is dominated by header
        assert!(dom.get(&2).unwrap().contains(&1));
        // Exit is dominated by header as well
        assert!(dom.get(&3).unwrap().contains(&1));
    }
}
