use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use spirv_tools_opt::translate;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut force_rust = false;
    let mut passthrough = false;
    let mut paths = Vec::<PathBuf>::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--force-rust" => force_rust = true,
            "--passthrough" => passthrough = true,
            _ if arg.starts_with('-') => {
                return Err(
                    "usage: opt_block [--force-rust] [--passthrough] <input.spv> [output.spv]"
                        .into(),
                )
            }
            _ => paths.push(PathBuf::from(arg)),
        }
    }

    let input = paths
        .get(0)
        .cloned()
        .ok_or("usage: opt_block [--force-rust] [--passthrough] <input.spv> [output.spv]")?;
    let output = paths.get(1).cloned();

    let input_bytes = fs::read(&input)?;
    let words = bytes_to_words(&input_bytes)?;

    let optimized = optimize_module(&words, force_rust, passthrough)?;
    let output_bytes = words_to_bytes(&optimized);

    if let Some(path) = output {
        fs::write(path, output_bytes)?;
    } else {
        std::io::stdout().write_all(&output_bytes)?;
    }
    Ok(())
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    if bytes.len() % 4 != 0 {
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
    if !force_rust && matches!(env::var("SPIRV_TOOLS_DISABLE_RUST_OPT"), Ok(v) if v == "1") {
        return Ok(words.to_vec());
    }

    let mut loader = rspirv::dr::Loader::new();
    parse_words(words, &mut loader)?;
    let mut module = loader.module();

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

            let mut arith_stream = Vec::new();
            arith_stream.extend(constant_map.values().cloned());
            arith_stream.extend(arithmetic.clone());

            let optimized =
                translate::optimize_arith_block(&arith_stream).map_err(|e| format!("{e}"))?;

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

    Ok(module.assemble())
}
