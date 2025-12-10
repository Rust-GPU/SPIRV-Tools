use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use spirv_tools_opt::translate::{
    rebuild_arith_with_original_ids, translate_arith_with_types, type_widths_from_module,
};

fn arith_stream_from_builder<F>(f: F) -> (Vec<rspirv::dr::Instruction>, rspirv::dr::Module)
where
    F: FnOnce(&mut Builder),
{
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    f(&mut b);
    let module = b.module();
    let mut stream = Vec::new();
    stream.extend(
        module
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == Op::Constant)
            .cloned(),
    );
    if let Some(func) = module.functions.first() {
        if let Some(block) = func.blocks.first() {
            stream.extend(
                block
                    .instructions
                    .iter()
                    .filter(|inst| is_arith(inst.class.opcode))
                    .cloned(),
            );
        }
    }
    (stream, module)
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

#[test]
fn rebuild_roundtrips_single_block_arith() {
    let (stream, module) = arith_stream_from_builder(|b| {
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let c5 = b.constant_bit32(int, 5);
        let c2 = b.constant_bit32(int, 2);
        let add = b.i_add(int, None, c4, c5).unwrap();
        let _ = b.i_sub(int, None, add, c2).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
    });

    let type_widths = type_widths_from_module(&module);
    let translated = translate_arith_with_types(&stream, &type_widths).expect("translate");
    let rebuilt =
        rebuild_arith_with_original_ids(&translated.expr, &translated).expect("rebuild succeeds");

    assert_eq!(
        rebuilt.len(),
        stream.len(),
        "rebuilt instructions should match original length"
    );
    assert!(
        rebuilt
            .iter()
            .zip(stream.iter())
            .all(|(a, b)| a.class.opcode == b.class.opcode && a.result_id == b.result_id),
        "rebuilt instructions should preserve opcode and result ids"
    );
}

#[test]
fn rebuild_respects_node_types_and_ids() {
    let (stream, module) = arith_stream_from_builder(|b| {
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c1 = b.constant_bit32(int, 1);
        let c2 = b.constant_bit32(int, 2);
        let mul = b.i_mul(int, None, c1, c2).unwrap();
        let _ = b.i_add(int, None, mul, c2).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
    });

    let type_widths = type_widths_from_module(&module);
    let translated = translate_arith_with_types(&stream, &type_widths).expect("translate");
    let rebuilt =
        rebuild_arith_with_original_ids(&translated.expr, &translated).expect("rebuild succeeds");

    for inst in &rebuilt {
        if let Some(ty) = inst.result_type {
            let width = type_widths.get(&ty).copied().unwrap_or(0);
            if let Some(orig_ty) = stream
                .iter()
                .find(|i| i.result_id == inst.result_id)
                .and_then(|i| i.result_type)
            {
                assert_eq!(
                    width,
                    type_widths.get(&orig_ty).copied().unwrap_or(0),
                    "rebuilt instruction should preserve type width"
                );
            }
        }
    }
}
