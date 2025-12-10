use clap::Parser;
use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use spirv_tools_opt::translate::{optimize_arith_block_with_types, type_widths_from_module};
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let input_bytes = fs::read(&args.input)?;
    let words = bytes_to_words(&input_bytes)?;

    let optimized = optimize_module(&words, args.force_rust, args.passthrough)?;
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
            if let Some(root_id) = arithmetic.last().and_then(|inst| inst.result_id) {
                preserved_roots.push(root_id);
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
