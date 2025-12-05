use rspirv::binary::Assemble;
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use spirv_tools_opt::{
    translate::{optimize_arith_block_with_types, type_widths_from_module},
    ConstValue,
};
use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

fn normalize_constant_operands(inst: &mut Instruction, type_widths: &HashMap<u32, u32>) {
    if inst.class.opcode != Op::Constant {
        return;
    }
    let Some(result_type) = inst.result_type else {
        return;
    };
    let Some(width_bits) = type_widths.get(&result_type) else {
        return;
    };
    let value = inst
        .operands
        .iter()
        .find_map(|op| match op {
            rspirv::dr::Operand::LiteralBit32(v) => Some(*v as u64),
            rspirv::dr::Operand::LiteralBit64(v) => Some(*v),
            _ => None,
        })
        .unwrap_or(0);
    let masked = ConstValue::new_with_width(value, (*width_bits).min(64) as u8);
    let operand = if *width_bits > 32 {
        rspirv::dr::Operand::LiteralBit64(masked.get_u64())
    } else {
        rspirv::dr::Operand::LiteralBit32(masked.get())
    };
    inst.operands.clear();
    inst.operands.push(operand);
}

/// Errors produced by the arithmetic optimizer.
#[derive(Debug, Error)]
pub enum OptimizeError {
    /// The input failed to parse as SPIR-V.
    #[error("failed to parse module: {0}")]
    Parse(String),
    /// The arithmetic optimizer reported a failure.
    #[error("optimization failed: {0}")]
    Rewrite(String),
}

/// Optimize a sequence of SPIR-V instructions representing an arithmetic basic block.
///
/// The input is expected to be a contiguous list of instructions in module order
/// (types/globals + block). Non-arithmetic instructions are preserved.
pub fn optimize_basic_block(insts: &[u32]) -> Result<Vec<u32>, OptimizeError> {
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(insts, &mut loader)
        .map_err(|e| OptimizeError::Parse(e.to_string()))?;
    let mut module = loader.module();
    let type_widths = type_widths_from_module(&module);
    let is_arith = |opcode: Op| {
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
    for inst in constant_map.values_mut() {
        normalize_constant_operands(inst, &type_widths);
    }

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

            let optimized = optimize_arith_block_with_types(&arith_stream, &type_widths)
                .map_err(|e| OptimizeError::Rewrite(e.to_string()))?;

            let mut optimized_block = Vec::new();
            for mut inst in optimized {
                normalize_constant_operands(&mut inst, &type_widths);
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
