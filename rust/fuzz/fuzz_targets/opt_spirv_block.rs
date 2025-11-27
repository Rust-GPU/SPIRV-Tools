#![no_main]

use arbitrary::Unstructured;
use egg::{Id, RecExpr};
use libfuzzer_sys::fuzz_target;
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use rspirv::spirv::Word;
use spirv_tools_opt::{fuzzing::arbitrary_expr, translate, ConstValue, SpirvLang};

fn id_to_word(id: Id) -> Word {
    (usize::from(id) as Word) + 1
}

fn expr_to_insts(expr: &RecExpr<SpirvLang>) -> Vec<Instruction> {
    expr.as_ref()
        .iter()
        .enumerate()
        .map(|(idx, node)| {
            let result_id = (idx as Word) + 1;
            match node {
                SpirvLang::Const(ConstValue(value)) => Instruction::new(
                    Op::Constant,
                    Some(1),
                    Some(result_id),
                    vec![rspirv::dr::Operand::LiteralBit32(value.get())],
                ),
                SpirvLang::Add([a, b]) => Instruction::new(
                    Op::IAdd,
                    Some(1),
                    Some(result_id),
                    vec![
                        rspirv::dr::Operand::IdRef(id_to_word(*a)),
                        rspirv::dr::Operand::IdRef(id_to_word(*b)),
                    ],
                ),
                SpirvLang::Mul([a, b]) => Instruction::new(
                    Op::IMul,
                    Some(1),
                    Some(result_id),
                    vec![
                        rspirv::dr::Operand::IdRef(id_to_word(*a)),
                        rspirv::dr::Operand::IdRef(id_to_word(*b)),
                    ],
                ),
                SpirvLang::Sub([a, b]) => Instruction::new(
                    Op::ISub,
                    Some(1),
                    Some(result_id),
                    vec![
                        rspirv::dr::Operand::IdRef(id_to_word(*a)),
                        rspirv::dr::Operand::IdRef(id_to_word(*b)),
                    ],
                ),
                SpirvLang::Div([a, b]) => Instruction::new(
                    Op::SDiv,
                    Some(1),
                    Some(result_id),
                    vec![
                        rspirv::dr::Operand::IdRef(id_to_word(*a)),
                        rspirv::dr::Operand::IdRef(id_to_word(*b)),
                    ],
                ),
                SpirvLang::Rem([a, b]) => Instruction::new(
                    Op::SRem,
                    Some(1),
                    Some(result_id),
                    vec![
                        rspirv::dr::Operand::IdRef(id_to_word(*a)),
                        rspirv::dr::Operand::IdRef(id_to_word(*b)),
                    ],
                ),
                SpirvLang::Neg(a) => Instruction::new(
                    Op::SNegate,
                    Some(1),
                    Some(result_id),
                    vec![rspirv::dr::Operand::IdRef(id_to_word(*a))],
                ),
                SpirvLang::Symbol(sym) => Instruction::new(
                    Op::Constant,
                    Some(1),
                    Some(result_id),
                    vec![rspirv::dr::Operand::LiteralBit32(
                        sym.as_str().len() as u32,
                    )],
                ),
            }
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok(expr) = arbitrary_expr(&mut u) {
        let insts = expr_to_insts(&expr);
        // Ensure we can translate and optimize without panicking.
        let _ = translate::optimize_arith_block(&insts);
    }
});
