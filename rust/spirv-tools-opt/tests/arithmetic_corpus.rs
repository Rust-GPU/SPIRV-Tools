use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use spirv_tools_opt::translate::optimize_arith_block;

fn inst(
    op: Op,
    result_type: u32,
    result_id: u32,
    operands: Vec<rspirv::dr::Operand>,
) -> Instruction {
    Instruction::new(op, Some(result_type), Some(result_id), operands)
}

#[test]
fn corpus_folds_add_two_constants() {
    let int = 1;
    let c2 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(2)],
    );
    let c3 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(3)],
    );
    let add = inst(
        Op::IAdd,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let block = vec![c2, c3, add];
    let optimized = optimize_arith_block(&block).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(5));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(5)]);
}

#[test]
fn corpus_folds_zero_minus_value_to_negate() {
    let int = 1;
    let c0 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(0)],
    );
    let c9 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(9)],
    );
    let sub = inst(
        Op::ISub,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c0, c9, sub]).expect("optimize");
    let mut saw_negate = false;
    let mut saw_const_operand = false;
    for inst in &optimized {
        if inst.class.opcode == Op::SNegate && inst.result_id == Some(3) {
            saw_negate = true;
            if let Some(rspirv::dr::Operand::IdRef(id)) = inst.operands.first() {
                saw_const_operand |= optimized.iter().any(|cand| {
                    cand.class.opcode == Op::Constant
                        && cand.result_id == Some(*id)
                        && cand.operands == vec![rspirv::dr::Operand::LiteralBit32(9)]
                });
            }
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(3)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(9))]
        {
            // Folded directly to a constant.
            saw_const_operand = true;
            saw_negate = true;
        }
    }
    assert!(saw_negate, "expected negate or folded constant");
    assert!(
        saw_const_operand,
        "negate or folded constant should reference literal 9"
    );
}

#[test]
fn corpus_folds_mul_by_zero() {
    let int = 1;
    let c5 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(5)],
    );
    let c0 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(0)],
    );
    let mul = inst(
        Op::IMul,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c5, c0, mul]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(4));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(0)]);
}

#[test]
fn corpus_folds_div_by_one() {
    let int = 1;
    let c8 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(8)],
    );
    let c1 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let div = inst(
        Op::SDiv,
        int,
        7,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c8, c1, div]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(7));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(8)]);
}

#[test]
fn corpus_folds_rem_by_one() {
    let int = 1;
    let c5 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(5)],
    );
    let c1 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let rem = inst(
        Op::SRem,
        int,
        9,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c5, c1, rem]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(9));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(0)]);
}

#[test]
fn corpus_preserves_rem_by_zero() {
    let int = 1;
    let c5 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(5)],
    );
    let c0 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(0)],
    );
    let rem = inst(
        Op::SRem,
        int,
        9,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let block = vec![c5, c0, rem];
    let optimized = optimize_arith_block(&block).expect("optimize");
    assert_eq!(optimized, block, "div/rem by zero should be preserved");
}

#[test]
fn corpus_folds_mul_by_one() {
    let int = 1;
    let c7 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(7)],
    );
    let c1 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let mul = inst(
        Op::IMul,
        int,
        11,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c7, c1, mul]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(11));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(7)]);
}
