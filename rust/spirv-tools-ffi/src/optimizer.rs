use rspirv::binary::Assemble;
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use spirv_tools_opt::translate;
use std::collections::BTreeMap;

/// Optimize a sequence of SPIR-V instructions representing an arithmetic basic block.
///
/// The input is expected to be a contiguous list of instructions in module order
/// (types/globals + block). Non-arithmetic instructions are preserved.
pub fn optimize_basic_block(insts: &[u32]) -> Result<Vec<u32>, String> {
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(insts, &mut loader).map_err(|e| e.to_string())?;
    let mut module = loader.module();
    let is_arith = |opcode: Op| {
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
    };

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
                translate::optimize_arith_block(&arith_stream).map_err(|e| e.to_string())?;

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
