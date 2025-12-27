use clap::Parser;
use egg::{EGraph, Extractor, Id, Language, RecExpr, Runner, Symbol};
use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use spirv_tools_opt::control::{add_control_roots, apply_control_roots, control_rewrites};
use spirv_tools_opt::translate::{
    optimize_arith_block_with_types, translate_arith_with_types, type_widths_from_module,
};
use spirv_tools_opt::{with_symbol_widths, ExprCost, SpirvAnalysis, SpirvLang};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
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
    /// Force the global (multi-block) optimizer path when possible.
    #[arg(long, default_value_t = false)]
    force_global: bool,
}

type PhiReplacement = (u32, u32);
type OptimizedFunction = (
    BTreeMap<u32, Instruction>,
    Vec<Vec<Instruction>>,
    Vec<PhiReplacement>,
);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let input_bytes = fs::read(&args.input)?;
    let words = bytes_to_words(&input_bytes)?;

    let optimized = optimize_module(
        &words,
        args.force_rust,
        args.passthrough,
        args.disable_global,
        args.force_global,
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
            | Op::CopyObject
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::BitwiseAnd
            | Op::Not
            | Op::BitReverse
            | Op::SNegate
            | Op::SDiv
            | Op::UDiv
            | Op::SRem
            | Op::SMod
            | Op::UMod
            | Op::ShiftLeftLogical
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
            | Op::Select
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
            | Op::LogicalNot
            | Op::LogicalAnd
            | Op::LogicalOr
            | Op::LogicalEqual
            | Op::LogicalNotEqual
    )
}

fn optimize_module(
    words: &[u32],
    force_rust: bool,
    passthrough: bool,
    disable_global: bool,
    force_global: bool,
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

    let is_constant = |opcode| {
        matches!(
            opcode,
            Op::Constant | Op::ConstantTrue | Op::ConstantFalse | Op::ConstantNull
        )
    };
    let non_constant_globals: Vec<Instruction> = module
        .types_global_values
        .iter()
        .filter(|inst| !is_constant(inst.class.opcode))
        .cloned()
        .collect();
    let mut constant_map: BTreeMap<u32, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
        .filter(|(_, inst)| is_constant(inst.class.opcode))
        .collect();
    let mut next_id = next_result_id_module(&module);

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

        let disable_global = disable_global || env::var("SPIRV_TOOLS_DISABLE_GLOBAL_OPT").is_ok();
        let force_global_env = env::var("SPIRV_TOOLS_FORCE_GLOBAL_OPT").is_ok();
        let use_global =
            (func.blocks.len() > 1) && (!disable_global || force_global || force_global_env);
        if use_global {
            if let Some((new_constants, optimized_blocks, phi_replacements)) =
                optimize_function_egraph(func, &type_widths, &constant_map, &mut next_id)
            {
                constant_map = new_constants;
                for (block, optimized_block) in func.blocks.iter_mut().zip(optimized_blocks) {
                    replace_block_arith(block, &optimized_block);
                }
                apply_phi_replacements(func, &phi_replacements);
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

        if !disable_global && env::var("SPIRV_TOOLS_DISABLE_GLOBAL_OPT").is_err() {
            if let Some(doms) = compute_block_dominators(func) {
                dedup_common_arith(func, &doms);
                insert_pre_for_shared_arith(func, &doms, &mut next_id);
            }
            // Control merge returns are handled in the global e-graph path.
        }
    }

    module.types_global_values = non_constant_globals;
    module
        .types_global_values
        .extend(constant_map.into_values());

    dedup_constants(&mut module);
    dead_code_eliminate(&mut module, &preserved_roots);
    update_module_bound(&mut module, next_id);

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
        if let Some(last) = new_block.pop() {
            if is_terminator(last.class.opcode) {
                new_block.extend_from_slice(optimized_block);
                new_block.push(last);
                inserted = true;
            } else {
                new_block.push(last);
            }
        }
        if !inserted {
            new_block.extend_from_slice(optimized_block);
        }
    }
    block.instructions = new_block;
}

fn apply_phi_replacements(func: &mut rspirv::dr::Function, replacements: &[PhiReplacement]) {
    if replacements.is_empty() {
        return;
    }
    let mut map = HashMap::new();
    for (result_id, operand_id) in replacements {
        map.insert(*result_id, *operand_id);
    }
    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if inst.class.opcode != Op::Phi {
                continue;
            }
            let Some(result_id) = inst.result_id else {
                continue;
            };
            let Some(&operand_id) = map.get(&result_id) else {
                continue;
            };
            *inst = Instruction::new(
                Op::CopyObject,
                inst.result_type,
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(operand_id)],
            );
        }
    }
}

fn is_terminator(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Branch
            | Op::BranchConditional
            | Op::Switch
            | Op::Return
            | Op::ReturnValue
            | Op::Unreachable
    )
}

fn symbol_for_id(id: u32) -> Symbol {
    Symbol::from(format!("id{id}"))
}

fn symbol_id(sym: &egg::Symbol) -> Option<u32> {
    sym.as_str().strip_prefix("id")?.parse().ok()
}

fn add_expr_to_egraph(
    expr: &egg::RecExpr<SpirvLang>,
    egraph: &mut EGraph<SpirvLang, SpirvAnalysis>,
) -> Vec<Id> {
    let mut ids = Vec::with_capacity(expr.as_ref().len());
    for node in expr.as_ref() {
        let enode = node.clone().map_children(|child| ids[usize::from(child)]);
        let id = egraph.add(enode);
        ids.push(id);
    }
    ids
}

#[derive(Debug)]
struct PhiRoot {
    result_id: u32,
    root: Id,
    operands: Vec<(u32, Id)>,
}

fn add_phi_roots(
    func: &rspirv::dr::Function,
    egraph: &mut EGraph<SpirvLang, SpirvAnalysis>,
    id_to_class: &HashMap<u32, Id>,
) -> Vec<PhiRoot> {
    let mut roots = Vec::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if inst.class.opcode != Op::Phi {
                continue;
            }
            let Some(result_id) = inst.result_id else {
                continue;
            };
            let mut values = Vec::new();
            for (idx, operand) in inst.operands.iter().enumerate() {
                if idx % 2 != 0 {
                    continue;
                }
                let Some(value_id) = operand.id_ref_any() else {
                    continue;
                };
                if value_id == result_id {
                    continue;
                }
                values.push(value_id);
            }
            if values.is_empty() {
                continue;
            }
            let mut expr = RecExpr::default();
            let mut nodes = Vec::with_capacity(values.len());
            for value in &values {
                nodes.push(expr.add(SpirvLang::Symbol(symbol_for_id(*value))));
            }
            let mut root = nodes[0];
            for &node in nodes.iter().skip(1) {
                root = expr.add(SpirvLang::Phi([root, node]));
            }
            let node_classes = add_expr_to_egraph(&expr, egraph);
            let mut operands = Vec::with_capacity(nodes.len());
            for (node_idx, value_id) in nodes.iter().copied().zip(values.iter().copied()) {
                let class = node_classes[usize::from(node_idx)];
                if let Some(&existing) = id_to_class.get(&value_id) {
                    egraph.union(class, existing);
                }
                operands.push((value_id, class));
            }
            roots.push(PhiRoot {
                result_id,
                root: node_classes[usize::from(root)],
                operands,
            });
        }
    }
    roots
}

fn build_best_rec_expr(
    egraph: &EGraph<SpirvLang, SpirvAnalysis>,
    extractor: &Extractor<ExprCost, SpirvLang, SpirvAnalysis>,
    roots: &[Id],
) -> (egg::RecExpr<SpirvLang>, HashMap<Id, Id>, Vec<Id>) {
    fn build_node(
        egraph: &EGraph<SpirvLang, SpirvAnalysis>,
        extractor: &Extractor<ExprCost, SpirvLang, SpirvAnalysis>,
        root: Id,
        memo: &mut HashMap<Id, Id>,
        expr: &mut egg::RecExpr<SpirvLang>,
        classes: &mut Vec<Id>,
    ) -> Id {
        let class = egraph.find(root);
        if let Some(&existing) = memo.get(&class) {
            return existing;
        }
        let best = extractor.find_best_node(class).clone();
        let rebuilt =
            best.map_children(|child| build_node(egraph, extractor, child, memo, expr, classes));
        let id = expr.add(rebuilt);
        memo.insert(class, id);
        classes.push(class);
        id
    }

    let mut expr = egg::RecExpr::default();
    let mut memo: HashMap<Id, Id> = HashMap::new();
    let mut classes: Vec<Id> = Vec::new();
    for &root in roots {
        let _ = build_node(egraph, extractor, root, &mut memo, &mut expr, &mut classes);
    }
    (expr, memo, classes)
}

fn infer_rec_types(
    expr: &egg::RecExpr<SpirvLang>,
    classes: &[Id],
    class_types: &HashMap<Id, u32>,
) -> Option<Vec<Option<u32>>> {
    let mut types: Vec<Option<u32>> = vec![None; expr.as_ref().len()];
    for (idx, class) in classes.iter().enumerate() {
        if let Some(ty) = class_types.get(class) {
            types[idx] = Some(*ty);
        }
    }
    for idx in (0..expr.as_ref().len()).rev() {
        let Some(ty) = types[idx] else {
            continue;
        };
        for child in expr.as_ref()[idx].children() {
            let child_idx = usize::from(*child);
            match types[child_idx] {
                None => types[child_idx] = Some(ty),
                Some(existing) if existing != ty => return None,
                _ => {}
            }
        }
    }
    Some(types)
}

fn optimize_function_egraph(
    func: &mut rspirv::dr::Function,
    type_widths: &HashMap<u32, u32>,
    constant_map: &BTreeMap<u32, Instruction>,
    next_id: &mut u32,
) -> Option<OptimizedFunction> {
    let debug_egraph = std::env::var("SPIRV_TOOLS_DEBUG_EGRAPH").is_ok();
    if debug_egraph {
        eprintln!("egraph start: blocks={}", func.blocks.len());
    }
    let (arithmetic, id_to_block) = collect_arith_topo(func)?;
    if !arithmetic.is_empty() {
        let block_dominators = compute_block_dominators(func)?;
        for inst in &arithmetic {
            let Some(result_id) = inst.result_id else {
                continue;
            };
            let Some(&block_idx) = id_to_block.get(&result_id) else {
                continue;
            };
            for op_id in collect_id_operands(inst) {
                if let Some(&op_block) = id_to_block.get(&op_id) {
                    if !dominates(&block_dominators, op_block, block_idx) {
                        return None;
                    }
                }
            }
        }
    }
    let mut egraph = EGraph::<SpirvLang, SpirvAnalysis>::default();
    let mut id_to_class: HashMap<u32, Id> = HashMap::new();
    let mut class_types: HashMap<Id, u32> = HashMap::new();
    let mut symbol_widths: HashMap<egg::Symbol, u8> = HashMap::new();

    if !arithmetic.is_empty() {
        let mut arith_stream = Vec::new();
        arith_stream.extend(constant_map.values().cloned());
        arith_stream.extend(arithmetic.clone());
        let translated = translate_arith_with_types(&arith_stream, type_widths).ok()?;
        symbol_widths = translated.symbol_widths.clone();
        let node_classes = add_expr_to_egraph(&translated.expr, &mut egraph);
        let arith_ids: HashSet<u32> = arithmetic
            .iter()
            .filter_map(|inst| inst.result_id)
            .collect();
        for (idx, original_id) in translated.original_ids.iter().enumerate() {
            let Some(id) = original_id else {
                continue;
            };
            if arith_ids.contains(id) {
                id_to_class.insert(*id, node_classes[idx]);
            }
        }
        for (idx, ty) in translated.node_types.iter().enumerate() {
            let Some(ty) = ty else {
                continue;
            };
            class_types.entry(node_classes[idx]).or_insert(*ty);
        }
    }

    let control_roots = add_control_roots(func, &mut egraph);
    let phi_roots = add_phi_roots(func, &mut egraph, &id_to_class);
    if arithmetic.is_empty() && control_roots.is_empty() && phi_roots.is_empty() {
        if debug_egraph {
            eprintln!("egraph skip: no arithmetic/control/phi roots");
        }
        return None;
    }

    let mut rewrites = spirv_tools_opt::rewrites();
    rewrites.extend(control_rewrites());
    let runner = with_symbol_widths(&symbol_widths, || {
        Runner::default().with_egraph(egraph).run(&rewrites)
    });
    let egraph = runner.egraph;
    let extractor = Extractor::new(&egraph, ExprCost);
    let _ = apply_control_roots(func, &extractor, &control_roots, next_id);
    let mut phi_replacements = Vec::new();
    for root in &phi_roots {
        let root_class = egraph.find(root.root);
        if let Some((operand_id, _)) = root
            .operands
            .iter()
            .find(|(_, class)| egraph.find(*class) == root_class)
        {
            phi_replacements.push((root.result_id, *operand_id));
        }
    }

    let mut canonical_types: HashMap<Id, u32> = HashMap::new();
    for (class, ty) in class_types {
        canonical_types.entry(egraph.find(class)).or_insert(ty);
    }
    class_types = canonical_types;
    for class in id_to_class.values_mut() {
        *class = egraph.find(*class);
    }

    if arithmetic.is_empty() {
        return Some((
            constant_map.clone(),
            vec![Vec::new(); func.blocks.len()],
            phi_replacements,
        ));
    }

    let mut root_classes = Vec::new();
    let mut seen = HashSet::new();
    for class in id_to_class.values() {
        let class = egraph.find(*class);
        if seen.insert(class) {
            root_classes.push(class);
        }
    }
    let (best_expr, _class_to_rec, rec_classes) =
        build_best_rec_expr(&egraph, &extractor, &root_classes);

    let rec_types = infer_rec_types(&best_expr, &rec_classes, &class_types)?;
    if rec_types.iter().any(|ty| ty.is_none()) {
        if debug_egraph {
            eprintln!("egraph abort: missing inferred types");
            eprintln!("egraph expr: {best_expr}");
            eprintln!("egraph types: {rec_types:?}");
            for (idx, node) in best_expr.as_ref().iter().enumerate() {
                eprintln!("egraph node {idx}: {node:?} type={:?}", rec_types[idx]);
            }
        }
        return None;
    }

    let mut class_ids: HashMap<Id, Vec<u32>> = HashMap::new();
    for (id, class) in &id_to_class {
        class_ids.entry(egraph.find(*class)).or_default().push(*id);
    }
    let mut canonical_by_class: HashMap<Id, u32> = HashMap::new();
    let mut aliases_by_class: HashMap<Id, Vec<u32>> = HashMap::new();
    for (class, mut ids) in class_ids {
        ids.sort_unstable();
        ids.dedup();
        if let Some((&canonical, rest)) = ids.split_first() {
            canonical_by_class.insert(class, canonical);
            if !rest.is_empty() {
                aliases_by_class.insert(class, rest.to_vec());
            }
        }
    }
    if debug_egraph {
        eprintln!("egraph id_to_block: {id_to_block:?}");
        eprintln!("egraph id_to_class: {id_to_class:?}");
        eprintln!("egraph canonical_by_class: {canonical_by_class:?}");
        eprintln!("egraph aliases_by_class: {aliases_by_class:?}");
    }

    let block_labels: Vec<Option<u32>> = if debug_egraph {
        func.blocks
            .iter()
            .map(|block| block.label.as_ref().and_then(|inst| inst.result_id))
            .collect()
    } else {
        Vec::new()
    };

    let mut rec_result_ids = vec![0u32; best_expr.as_ref().len()];
    let mut rec_emit = vec![true; best_expr.as_ref().len()];
    for (idx, node) in best_expr.as_ref().iter().enumerate() {
        match node {
            SpirvLang::Symbol(sym) => {
                let class = rec_classes[idx];
                if let Some(canonical) = canonical_by_class.get(&class) {
                    rec_result_ids[idx] = *canonical;
                    rec_emit[idx] = true;
                } else {
                    let id = symbol_id(sym)?;
                    rec_result_ids[idx] = id;
                    rec_emit[idx] = false;
                }
            }
            _ => {
                let class = rec_classes[idx];
                if let Some(canonical) = canonical_by_class.get(&class) {
                    rec_result_ids[idx] = *canonical;
                } else {
                    let id = *next_id;
                    *next_id = next_id.saturating_add(1);
                    rec_result_ids[idx] = id;
                }
            }
        }
    }

    let block_dominators = compute_block_dominators(func)?;
    let idoms = compute_immediate_dominators(&block_dominators);
    let dom_depths = compute_dominator_depths(&idoms);
    let mut label_to_idx = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        if let Some(label_id) = block.label.as_ref().and_then(|inst| inst.result_id) {
            label_to_idx.insert(label_id, idx);
        }
    }
    let mut loop_headers: HashSet<usize> = HashSet::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        let Some(succs) = block_successors(block, &label_to_idx) else {
            continue;
        };
        for succ in succs {
            if dominates(&block_dominators, succ, idx) {
                loop_headers.insert(succ);
            }
        }
    }
    let mut new_constants = constant_map.clone();
    let mut per_block = vec![Vec::new(); func.blocks.len()];
    let uses_map = collect_use_blocks(func, &id_to_block);
    let mut class_uses: HashMap<Id, HashSet<usize>> = HashMap::new();
    for (id, class) in &id_to_class {
        let class = egraph.find(*class);
        if let Some(uses) = uses_map.get(id) {
            class_uses
                .entry(class)
                .or_default()
                .extend(uses.iter().copied());
        }
        if let Some(block) = id_to_block.get(id) {
            class_uses.entry(class).or_default().insert(*block);
        }
    }

    let mut block_map = id_to_block.clone();

    for (idx, node) in best_expr.as_ref().iter().enumerate() {
        if !rec_emit[idx] {
            continue;
        }
        let result_type = rec_types[idx]?;
        let result_id = rec_result_ids[idx];
        let inst = match node {
            SpirvLang::Const(val) => {
                if val.width_bits() == 1 {
                    let opcode = if val.get_u64() == 0 {
                        Op::ConstantFalse
                    } else {
                        Op::ConstantTrue
                    };
                    Instruction::new(opcode, Some(result_type), Some(result_id), Vec::new())
                } else {
                    let operands = match val.width_bits() {
                        64 => vec![rspirv::dr::Operand::LiteralBit64(val.get_u64())],
                        _ => vec![rspirv::dr::Operand::LiteralBit32(val.get())],
                    };
                    Instruction::new(Op::Constant, Some(result_type), Some(result_id), operands)
                }
            }
            SpirvLang::Add([a, b]) => Instruction::new(
                Op::IAdd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Mul([a, b]) => Instruction::new(
                Op::IMul,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Sub([a, b]) => Instruction::new(
                Op::ISub,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SDiv([a, b]) => Instruction::new(
                Op::SDiv,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UDiv([a, b]) => Instruction::new(
                Op::UDiv,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SRem([a, b]) => Instruction::new(
                Op::SRem,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SMod([a, b]) => Instruction::new(
                Op::SMod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UMod([a, b]) => Instruction::new(
                Op::UMod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Shl([a, b]) => Instruction::new(
                Op::ShiftLeftLogical,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ShrS([a, b]) => Instruction::new(
                Op::ShiftRightArithmetic,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ShrU([a, b]) => Instruction::new(
                Op::ShiftRightLogical,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Eq([a, b]) => Instruction::new(
                Op::IEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Ne([a, b]) => Instruction::new(
                Op::INotEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SLt([a, b]) => Instruction::new(
                Op::SLessThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SLe([a, b]) => Instruction::new(
                Op::SLessThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SGt([a, b]) => Instruction::new(
                Op::SGreaterThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SGe([a, b]) => Instruction::new(
                Op::SGreaterThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ULt([a, b]) => Instruction::new(
                Op::ULessThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ULe([a, b]) => Instruction::new(
                Op::ULessThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UGt([a, b]) => Instruction::new(
                Op::UGreaterThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UGe([a, b]) => Instruction::new(
                Op::UGreaterThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogNot(a) => Instruction::new(
                Op::LogicalNot,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)])],
            ),
            SpirvLang::LogAnd([a, b]) => Instruction::new(
                Op::LogicalAnd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogOr([a, b]) => Instruction::new(
                Op::LogicalOr,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogEq([a, b]) => Instruction::new(
                Op::LogicalEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogNe([a, b]) => Instruction::new(
                Op::LogicalNotEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitAnd([a, b]) => Instruction::new(
                Op::BitwiseAnd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitOr([a, b]) => Instruction::new(
                Op::BitwiseOr,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitXor([a, b]) => Instruction::new(
                Op::BitwiseXor,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitNot(a) => Instruction::new(
                Op::Not,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)])],
            ),
            SpirvLang::BitReverse(a) => Instruction::new(
                Op::BitReverse,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)])],
            ),
            SpirvLang::Neg(a) => Instruction::new(
                Op::SNegate,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*a)])],
            ),
            SpirvLang::Select([c, t, f]) => Instruction::new(
                Op::Select,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*c)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*t)]),
                    rspirv::dr::Operand::IdRef(rec_result_ids[usize::from(*f)]),
                ],
            ),
            SpirvLang::Symbol(sym) => {
                let Some(source) = symbol_id(sym) else {
                    continue;
                };
                Instruction::new(
                    Op::CopyObject,
                    Some(result_type),
                    Some(result_id),
                    vec![rspirv::dr::Operand::IdRef(source)],
                )
            }
            SpirvLang::RotL(_)
            | SpirvLang::RotR(_)
            | SpirvLang::If(_)
            | SpirvLang::Merge(_)
            | SpirvLang::Ret
            | SpirvLang::RetVal(_)
            | SpirvLang::Phi(_)
            | SpirvLang::Pair(_) => continue,
        };

        if inst.class.opcode == Op::Constant {
            new_constants.insert(result_id, inst);
            continue;
        }

        let block_idx = match block_map.get(&result_id) {
            Some(idx) => *idx,
            None => {
                let from_ops = collect_id_operands(&inst)
                    .into_iter()
                    .find_map(|op| block_map.get(&op).copied())
                    .unwrap_or(0);
                block_map.insert(result_id, from_ops);
                from_ops
            }
        };
        let operand_blocks: Vec<_> = collect_id_operands(&inst)
            .into_iter()
            .filter_map(|op| block_map.get(&op).copied())
            .collect();
        let use_blocks = class_uses
            .get(&rec_classes[idx])
            .cloned()
            .unwrap_or_default();

        let placement = if use_blocks.is_empty() {
            block_idx
        } else {
            let mut common: Option<HashSet<usize>> = None;
            for use_block in &use_blocks {
                let Some(doms) = block_dominators.get(use_block) else {
                    continue;
                };
                common = Some(match common {
                    None => doms.clone(),
                    Some(prev) => prev.intersection(doms).copied().collect(),
                });
            }
            let common = common.unwrap_or_else(|| {
                let mut set = HashSet::new();
                set.insert(block_idx);
                set
            });
            let mut candidate = *common
                .iter()
                .max_by_key(|block| dom_depths.get(block).copied().unwrap_or(0))
                .unwrap_or(&block_idx);
            if use_blocks.len() == 1 && !loop_headers.is_empty() {
                if let Some(header) = loop_headers
                    .iter()
                    .filter(|header| dominates(&block_dominators, **header, candidate))
                    .max_by_key(|header| dom_depths.get(header).copied().unwrap_or(0))
                {
                    if let Some(Some(preheader)) = idoms.get(header) {
                        candidate = *preheader;
                    }
                }
            }
            while !operand_blocks
                .iter()
                .all(|op_block| dominates(&block_dominators, *op_block, candidate))
            {
                match idoms.get(&candidate).and_then(|parent| *parent) {
                    Some(parent) => candidate = parent,
                    None => break,
                }
            }
            candidate
        };

        if debug_egraph {
            let ops = collect_id_operands(&inst);
            let label = block_labels.get(placement).and_then(|val| *val);
            eprintln!(
                "egraph place op={:?} id={} block={} label={:?} ops={:?}",
                inst.class.opcode, result_id, placement, label, ops
            );
        }
        per_block[placement].push(inst);
    }

    for (class, aliases) in aliases_by_class {
        let Some(&canonical) = canonical_by_class.get(&class) else {
            continue;
        };
        let result_type = *class_types.get(&class)?;
        for alias in aliases {
            let Some(&block_idx) = id_to_block.get(&alias) else {
                continue;
            };
            let copy = Instruction::new(
                Op::CopyObject,
                Some(result_type),
                Some(alias),
                vec![rspirv::dr::Operand::IdRef(canonical)],
            );
            per_block[block_idx].push(copy);
        }
    }

    Some((new_constants, per_block, phi_replacements))
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

fn compute_block_dominators(func: &rspirv::dr::Function) -> Option<HashMap<usize, HashSet<usize>>> {
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
        if i == 0 || preds[i].is_empty() {
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
    let Some(last) = block.instructions.last() else {
        return Some(Vec::new());
    };
    use rspirv::dr::Operand::*;
    let mut targets = Vec::new();
    match last.class.opcode {
        Op::Branch => {
            if let Some(IdRef(label)) = last.operands.first() {
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

fn compute_immediate_dominators(
    dominators: &HashMap<usize, HashSet<usize>>,
) -> HashMap<usize, Option<usize>> {
    let mut idoms = HashMap::new();
    for (&block, doms) in dominators {
        if doms.len() <= 1 {
            idoms.insert(block, None);
            continue;
        }
        let mut candidates: Vec<_> = doms.iter().copied().filter(|d| *d != block).collect();
        candidates.sort_by_key(|c| dominators.get(c).map(|s| s.len()).unwrap_or(0));
        idoms.insert(block, candidates.last().copied());
    }
    idoms
}

fn compute_dominator_depths(idoms: &HashMap<usize, Option<usize>>) -> HashMap<usize, usize> {
    fn depth_for(
        block: usize,
        idoms: &HashMap<usize, Option<usize>>,
        depths: &mut HashMap<usize, usize>,
    ) -> usize {
        if let Some(&depth) = depths.get(&block) {
            return depth;
        }
        let depth = match idoms.get(&block).and_then(|parent| *parent) {
            Some(parent) => depth_for(parent, idoms, depths).saturating_add(1),
            None => 0,
        };
        depths.insert(block, depth);
        depth
    }

    let mut depths = HashMap::new();
    for &block in idoms.keys() {
        let _ = depth_for(block, idoms, &mut depths);
    }
    depths
}

fn dominates(dominators: &HashMap<usize, HashSet<usize>>, a: usize, b: usize) -> bool {
    dominators
        .get(&b)
        .map(|set| set.contains(&a))
        .unwrap_or(false)
}

fn collect_use_blocks(
    func: &rspirv::dr::Function,
    id_to_block: &HashMap<u32, usize>,
) -> HashMap<u32, HashSet<usize>> {
    let mut uses: HashMap<u32, HashSet<usize>> = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            for op in collect_id_operands(inst) {
                if id_to_block.contains_key(&op) {
                    uses.entry(op).or_default().insert(idx);
                }
            }
        }
    }
    uses
}

#[derive(Clone, Debug, Eq)]
struct ArithKey {
    opcode: Op,
    result_type: Option<u32>,
    operands: Vec<u32>,
}

impl PartialEq for ArithKey {
    fn eq(&self, other: &Self) -> bool {
        self.opcode == other.opcode
            && self.result_type == other.result_type
            && self.operands == other.operands
    }
}

impl Hash for ArithKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.opcode.hash(state);
        self.result_type.hash(state);
        self.operands.hash(state);
    }
}

fn dedup_common_arith(
    func: &mut rspirv::dr::Function,
    dominators: &HashMap<usize, HashSet<usize>>,
) {
    let mut table: HashMap<ArithKey, (u32, usize)> = HashMap::new();
    let mut defs: HashMap<u32, (Op, Vec<u32>, usize)> = HashMap::new();
    for (idx, block) in func.blocks.iter_mut().enumerate() {
        for inst in &mut block.instructions {
            if !is_arith(inst.class.opcode) {
                continue;
            }
            let Some(result_id) = inst.result_id else {
                continue;
            };
            if inst.class.opcode == Op::Constant {
                continue;
            }
            let operands: Vec<u32> = inst
                .operands
                .iter()
                .filter_map(|op| op.id_ref_any())
                .collect();

            // Cross-block affine cancel: (x + y) - y => x (and commuted form).
            if inst.class.opcode == Op::ISub && operands.len() == 2 {
                if let Some((Op::IAdd, add_ops, add_block)) = defs.get(&operands[0]) {
                    if add_ops.len() == 2 && dominates(dominators, *add_block, idx) {
                        let replacement = if add_ops[0] == operands[1] {
                            Some(add_ops[1])
                        } else if add_ops[1] == operands[1] {
                            Some(add_ops[0])
                        } else {
                            None
                        };
                        if let Some(rep) = replacement {
                            *inst = Instruction::new(
                                Op::CopyObject,
                                inst.result_type,
                                Some(result_id),
                                vec![rspirv::dr::Operand::IdRef(rep)],
                            );
                        }
                    }
                }
            }

            let Some(key) = make_arith_key(inst) else {
                defs.insert(result_id, (inst.class.opcode, operands, idx));
                continue;
            };
            if let Some((prev_id, prev_block)) = table.get(&key) {
                if dominates(dominators, *prev_block, idx) {
                    *inst = Instruction::new(
                        Op::CopyObject,
                        inst.result_type,
                        Some(result_id),
                        vec![rspirv::dr::Operand::IdRef(*prev_id)],
                    );
                }
            }
            table.insert(key, (inst.result_id.unwrap_or(result_id), idx));
            defs.insert(result_id, (inst.class.opcode, operands, idx));
        }
    }

    collapse_copy_chains(func);
    prune_dead_copies(func);
}

fn make_arith_key(inst: &Instruction) -> Option<ArithKey> {
    let opcode = inst.class.opcode;
    let commutative = matches!(
        opcode,
        Op::IAdd | Op::IMul | Op::BitwiseAnd | Op::BitwiseOr | Op::BitwiseXor
    );
    let mut operands: Vec<u32> = inst
        .operands
        .iter()
        .filter_map(|op| op.id_ref_any())
        .collect();
    if operands.len() != inst.operands.len() {
        return None;
    }
    if commutative && operands.len() == 2 {
        operands.sort();
    }
    Some(ArithKey {
        opcode,
        result_type: inst.result_type,
        operands,
    })
}

fn collapse_copy_chains(func: &mut rspirv::dr::Function) {
    let mut parent: HashMap<u32, u32> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if inst.class.opcode == Op::CopyObject {
                if let (Some(dst), Some(src)) = (
                    inst.result_id,
                    inst.operands.first().and_then(|op| op.id_ref_any()),
                ) {
                    parent.insert(dst, src);
                }
            }
        }
    }

    fn find_root(id: u32, parent: &mut HashMap<u32, u32>) -> u32 {
        let mut current = id;
        let mut seen = HashSet::new();
        while let Some(&next) = parent.get(&current) {
            if !seen.insert(current) {
                break;
            }
            current = next;
        }
        let root = current;
        for node in seen {
            parent.insert(node, root);
        }
        root
    }

    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            for op in &mut inst.operands {
                if let Some(idref) = op.id_ref_any_mut() {
                    let root = find_root(*idref, &mut parent);
                    *idref = root;
                }
            }
        }
    }
}

fn prune_dead_copies(func: &mut rspirv::dr::Function) {
    loop {
        let mut uses: HashMap<u32, usize> = HashMap::new();
        for inst in func.all_inst_iter() {
            for op in inst.operands.iter().filter_map(|op| op.id_ref_any()) {
                *uses.entry(op).or_default() += 1;
            }
        }

        let mut changed = false;
        for block in &mut func.blocks {
            block.instructions.retain(|inst| {
                if inst.class.opcode == Op::CopyObject {
                    if let Some(id) = inst.result_id {
                        if uses.get(&id).copied().unwrap_or(0) == 0 {
                            changed = true;
                            return false;
                        }
                    }
                }
                true
            });
        }

        if !changed {
            break;
        }
    }
}

fn insert_pre_for_shared_arith(
    func: &mut rspirv::dr::Function,
    dominators: &HashMap<usize, HashSet<usize>>,
    next_id: &mut u32,
) {
    let idoms = compute_immediate_dominators(dominators);
    let dom_depths = compute_dominator_depths(&idoms);
    let id_to_block = collect_id_def_blocks(func);
    let mut expr_to_blocks: HashMap<ArithKey, Vec<usize>> = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if !is_arith(inst.class.opcode) || inst.class.opcode == Op::Constant {
                continue;
            }
            if let Some(key) = make_arith_key(inst) {
                expr_to_blocks.entry(key).or_default().push(idx);
            }
        }
    }

    for (key, blocks) in expr_to_blocks {
        if blocks.len() < 2 {
            continue;
        }
        if let Some(merge) = find_nearest_common_dominator(&blocks, dominators, &dom_depths) {
            // Skip hoisting if the merged expression has no id operands.
            if key.operands.is_empty() {
                continue;
            }
            if key.opcode == Op::CopyObject {
                continue;
            }
            let operand_blocks: Vec<_> = key
                .operands
                .iter()
                .filter_map(|op| id_to_block.get(op).copied())
                .collect();
            if operand_blocks
                .iter()
                .all(|op_block| dominates(dominators, *op_block, merge))
            {
                hoist_expr_to_block(func, &key, merge, &blocks, next_id);
            }
        }
    }
}

fn find_nearest_common_dominator(
    blocks: &[usize],
    dominators: &HashMap<usize, HashSet<usize>>,
    dom_depths: &HashMap<usize, usize>,
) -> Option<usize> {
    let mut dom_sets: Vec<HashSet<usize>> = blocks
        .iter()
        .filter_map(|b| dominators.get(b).cloned())
        .collect();
    if dom_sets.is_empty() {
        return None;
    }
    let mut intersection = dom_sets.pop().unwrap();
    for set in dom_sets {
        intersection = intersection.intersection(&set).copied().collect();
    }
    intersection
        .into_iter()
        .max_by_key(|block| (dom_depths.get(block).copied().unwrap_or(0), *block))
}

fn dedup_constants(module: &mut rspirv::dr::Module) {
    #[derive(Eq, PartialEq, Hash)]
    enum ConstKey {
        Bit32(u32, u32),
        Bit64(u32, u64),
    }

    let mut canonical: HashMap<ConstKey, u32> = HashMap::new();
    let mut id_rewrites: HashMap<u32, u32> = HashMap::new();

    for inst in &module.types_global_values {
        if inst.class.opcode != Op::Constant {
            continue;
        }
        let Some(result_id) = inst.result_id else {
            continue;
        };
        let Some(ty) = inst.result_type else { continue };
        let key = match inst.operands.first() {
            Some(rspirv::dr::Operand::LiteralBit32(v)) => ConstKey::Bit32(ty, *v),
            Some(rspirv::dr::Operand::LiteralBit64(v)) => ConstKey::Bit64(ty, *v),
            _ => continue,
        };
        match canonical.entry(key) {
            Entry::Vacant(v) => {
                v.insert(result_id);
            }
            Entry::Occupied(o) => {
                id_rewrites.insert(result_id, *o.get());
            }
        }
    }

    if id_rewrites.is_empty() {
        return;
    }

    for func in &mut module.functions {
        for inst in func.all_inst_iter_mut() {
            for op in &mut inst.operands {
                if let Some(idref) = op.id_ref_any_mut() {
                    if let Some(&replacement) = id_rewrites.get(idref) {
                        *idref = replacement;
                    }
                }
            }
        }
    }

    module.types_global_values.retain(|inst| {
        if inst.class.opcode == Op::Constant {
            if let Some(id) = inst.result_id {
                return !id_rewrites.contains_key(&id);
            }
        }
        true
    });
}

fn hoist_expr_to_block(
    func: &mut rspirv::dr::Function,
    key: &ArithKey,
    target_block_idx: usize,
    original_blocks: &[usize],
    next_id: &mut u32,
) {
    if target_block_idx >= func.blocks.len() {
        return;
    }
    let operands: Vec<_> = key
        .operands
        .iter()
        .copied()
        .map(rspirv::dr::Operand::IdRef)
        .collect();
    // Heuristic: only hoist when the expression has at least two id operands,
    // otherwise leave it local to avoid bloating dominator blocks.
    let id_operand_count = operands
        .iter()
        .filter(|op| matches!(op, rspirv::dr::Operand::IdRef(_)))
        .count();
    if id_operand_count < 2 {
        return;
    }
    let inst_template = Instruction::new(key.opcode, key.result_type, None, operands);

    let new_id = *next_id;
    *next_id = next_id.saturating_add(1);
    let mut hoisted = inst_template.clone();
    hoisted.result_id = Some(new_id);
    let operand_set: HashSet<u32> = key.operands.iter().copied().collect();
    let mut insert_idx = 0;
    let block_insts = &func.blocks[target_block_idx].instructions;
    for (idx, inst) in block_insts.iter().enumerate() {
        if let Some(id) = inst.result_id {
            if operand_set.contains(&id) {
                insert_idx = insert_idx.max(idx + 1);
            }
        }
    }
    let mut control_idx = block_insts.len();
    for (idx, inst) in block_insts.iter().enumerate() {
        if matches!(inst.class.opcode, Op::SelectionMerge | Op::LoopMerge)
            || is_terminator(inst.class.opcode)
        {
            control_idx = idx;
            break;
        }
    }
    let insert_idx = insert_idx.min(control_idx);
    func.blocks[target_block_idx]
        .instructions
        .insert(insert_idx, hoisted);

    for &blk_idx in original_blocks {
        let block = match func.blocks.get_mut(blk_idx) {
            Some(b) => b,
            None => continue,
        };
        for inst in &mut block.instructions {
            if let Some(existing_key) = make_arith_key(inst) {
                if existing_key == *key {
                    let result_id = inst.result_id;
                    *inst = Instruction::new(
                        Op::CopyObject,
                        inst.result_type,
                        result_id,
                        vec![rspirv::dr::Operand::IdRef(new_id)],
                    );
                }
            }
        }
    }
}

fn next_result_id_module(module: &rspirv::dr::Module) -> u32 {
    let mut max_id = 1;
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            max_id = max_id.max(id + 1);
        }
    }
    for func in &module.functions {
        if let Some(def) = &func.def {
            if let Some(id) = def.result_id {
                max_id = max_id.max(id + 1);
            }
        }
        if let Some(end) = &func.end {
            if let Some(id) = end.result_id {
                max_id = max_id.max(id + 1);
            }
        }
        for param in &func.parameters {
            if let Some(id) = param.result_id {
                max_id = max_id.max(id + 1);
            }
        }
        for block in &func.blocks {
            if let Some(label) = &block.label {
                if let Some(id) = label.result_id {
                    max_id = max_id.max(id + 1);
                }
            }
            for inst in &block.instructions {
                if let Some(id) = inst.result_id {
                    max_id = max_id.max(id + 1);
                }
            }
        }
    }
    max_id
}

fn update_module_bound(module: &mut rspirv::dr::Module, next_id: u32) {
    if let Some(header) = module.header.as_mut() {
        if header.bound < next_id {
            header.bound = next_id;
        }
    }
}

fn collect_id_def_blocks(func: &rspirv::dr::Function) -> HashMap<u32, usize> {
    let mut id_to_block = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        if let Some(label) = &block.label {
            if let Some(id) = label.result_id {
                id_to_block.insert(id, idx);
            }
        }
        for inst in &block.instructions {
            if let Some(id) = inst.result_id {
                id_to_block.insert(id, idx);
            }
        }
    }
    id_to_block
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
            vec![rspirv::dr::Operand::IdRef(3), rspirv::dr::Operand::IdRef(2)],
        );
        let block1_mul = Instruction::new(
            Op::IMul,
            Some(1),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
        );

        let mut func = Function::new();
        func.blocks
            .push(block_with_label(100, vec![block0_add.clone(), branch(101)]));
        func.blocks.push(block_with_label(
            101,
            vec![
                c_two.clone(),
                block1_mul.clone(),
                Instruction::new(Op::Return, None, None, Vec::new()),
            ],
        ));

        let type_widths = HashMap::from_iter([(1u32, 32u32)]);
        let constant_map = BTreeMap::new();

        let mut next_id = 1000;
        let res = optimize_function_egraph(&mut func, &type_widths, &constant_map, &mut next_id);
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
        Instruction::new(
            Op::Branch,
            None,
            None,
            vec![rspirv::dr::Operand::IdRef(target)],
        )
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
            vec![rspirv::dr::Operand::IdRef(5), rspirv::dr::Operand::IdRef(5)],
        );

        let mut func = Function::new();
        func.blocks.push(block_with_label(
            100,
            vec![ty.clone(), branch_cond(101, 102)],
        ));
        func.blocks
            .push(block_with_label(101, vec![def_left.clone(), branch(103)]));
        func.blocks.push(block_with_label(102, vec![branch(103)]));
        func.blocks
            .push(block_with_label(103, vec![use_merge.clone()]));

        let type_widths = HashMap::from_iter([(1u32, 32u32)]);
        let constant_map = BTreeMap::new();
        let mut next_id = 1000;
        let res = optimize_function_egraph(&mut func, &type_widths, &constant_map, &mut next_id);
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
        func.blocks.push(block_with_label(
            4,
            vec![Instruction::new(Op::Return, None, None, Vec::new())],
        ));
        // unreachable block with no predecessors
        func.blocks.push(block_with_label(
            99,
            vec![Instruction::new(Op::Return, None, None, Vec::new())],
        ));

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
