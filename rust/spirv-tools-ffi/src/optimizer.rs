use rspirv::binary::{Assemble, Consumer};
use spirv_tools_opt::translate;

/// Optimize a sequence of SPIR-V instructions representing an arithmetic basic block.
///
/// The input is expected to be a contiguous list of instructions in module order
/// (types/globals + block). Non-arithmetic instructions are preserved.
pub fn optimize_basic_block(insts: &[u32]) -> Result<Vec<u32>, String> {
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(insts, &mut loader).map_err(|e| e.to_string())?;
    let module = loader.module();
    let mut output = Vec::new();

    for func in &module.functions {
        for block in &func.blocks {
            let mut collected: Vec<rspirv::dr::Instruction> = module.types_global_values.clone();
            collected.extend(block.instructions.clone());
            let optimized =
                translate::optimize_arith_block(&collected).map_err(|e| e.to_string())?;
            let mut local = rspirv::dr::Loader::new();
            for inst in optimized {
                Consumer::consume_instruction(&mut local, inst);
            }
            let buf = local.module().assemble();
            output.extend(buf);
        }
    }

    Ok(output)
}
