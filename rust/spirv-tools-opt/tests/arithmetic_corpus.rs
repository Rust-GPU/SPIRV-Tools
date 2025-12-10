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
fn corpus_rewrites_band_pow2_mask_into_umod() {
    let int = 1;
    // Build x = 5 + 6, then x & 7.
    let c5 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(5)],
    );
    let c6 = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(6)],
    );
    let add = inst(
        Op::IAdd,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let mask = inst(
        Op::Constant,
        int,
        5,
        vec![rspirv::dr::Operand::LiteralBit32(7)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        6,
        vec![rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(5)],
    );
    let optimized = optimize_arith_block(&[c5, c6, add, mask, band]).expect("optimize");
    // Expect the mask to either become a modulus by 8 or fully fold to a constant.
    let has_umod = optimized.iter().any(|inst| {
        inst.class.opcode == Op::UMod
            && inst
                .operands
                .iter()
                .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(value) if *value == 8))
    });
    let has_constant = optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::Constant);
    assert!(
        has_umod || has_constant,
        "expected band mask to rewrite to modulus by 8 or fold to a constant: {optimized:?}"
    );
    assert!(
        optimized
            .iter()
            .all(|inst| inst.class.opcode != Op::BitwiseAnd),
        "bitwise-and should be eliminated after rewrite"
    );
}

#[test]
fn corpus_strength_reduces_mul_by_pow2() {
    let int = 1;
    let c8 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(8)],
    );
    let c2 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(2)],
    );
    let mul = inst(
        Op::IMul,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c8, c2, mul]).expect("optimize");
    // Allow either shift or fully folded constant if both operands are const.
    let saw_shift = optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::ShiftLeftLogical && inst.result_id == Some(3));
    let saw_const = optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(3)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(16)]
    });
    assert!(
        saw_shift || saw_const,
        "mul by power of two should strength-reduce or fold"
    );
    assert!(
        optimized.iter().all(|inst| inst.class.opcode != Op::IMul),
        "mul should be rewritten or folded"
    );
}

#[test]
fn corpus_folds_const_rotate_left() {
    let int = 1;
    let value = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let shift = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(4)],
    );
    let left = inst(
        Op::ShiftLeftLogical,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let inv_shift = inst(
        Op::Constant,
        int,
        4,
        vec![rspirv::dr::Operand::LiteralBit32(28)],
    );
    let right = inst(
        Op::ShiftRightLogical,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(4)],
    );
    let rot = inst(
        Op::BitwiseOr,
        int,
        6,
        vec![rspirv::dr::Operand::IdRef(3), rspirv::dr::Operand::IdRef(5)],
    );
    let optimized =
        optimize_arith_block(&[value, shift, left, inv_shift, right, rot]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(6));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(16)]);
}

#[test]
fn corpus_folds_const_rotate_left_commuted_or() {
    let int = 1;
    let value = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let shift = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(4)],
    );
    let left = inst(
        Op::ShiftLeftLogical,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let inv_shift = inst(
        Op::Constant,
        int,
        4,
        vec![rspirv::dr::Operand::LiteralBit32(28)],
    );
    let right = inst(
        Op::ShiftRightLogical,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(4)],
    );
    let rot = inst(
        Op::BitwiseOr,
        int,
        6,
        vec![rspirv::dr::Operand::IdRef(5), rspirv::dr::Operand::IdRef(3)],
    );
    let optimized =
        optimize_arith_block(&[value, shift, left, inv_shift, right, rot]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(6));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(16)]);
}

#[test]
fn corpus_does_not_fold_non_complementary_shift_or() {
    let int = 1;
    let value = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let shift_left = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(3)],
    );
    let shift_right = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(3)],
    );
    let left = inst(
        Op::ShiftLeftLogical,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let right = inst(
        Op::ShiftRightLogical,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(3)],
    );
    let or_inst = inst(
        Op::BitwiseOr,
        int,
        6,
        vec![rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(5)],
    );
    let optimized = optimize_arith_block(&[value, shift_left, shift_right, left, right, or_inst])
        .expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(6));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(8)]);
}

#[test]
fn corpus_folds_const_bor_commutes() {
    let int = 1;
    let c1 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let c2 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(2)],
    );
    let bor = inst(
        Op::BitwiseOr,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(1)],
    );
    let optimized = optimize_arith_block(&[c1, c2, bor]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(3)]);
}

#[test]
fn corpus_folds_nested_bor_constants() {
    let int = 1;
    let c1 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let c2 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(2)],
    );
    let c4 = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(4)],
    );
    let inner = inst(
        Op::BitwiseOr,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let outer = inst(
        Op::BitwiseOr,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(3)],
    );
    let optimized = optimize_arith_block(&[c1, c2, c4, inner, outer]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(7)]);
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
fn corpus_simplifies_bor_zero_and_self() {
    let int = 1;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(5)],
    );
    let zero = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(0)],
    );
    let or_zero = inst(
        Op::BitwiseOr,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let or_self = inst(
        Op::BitwiseOr,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[x.clone(), zero, or_zero, or_self]).expect("opt");
    let has_const_five = optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst
                .operands
                .iter()
                .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(v) if *v == 5))
    });
    assert!(has_const_five, "expected constant value 5 after OR folding");
    assert!(
        optimized
            .iter()
            .all(|inst| inst.class.opcode != Op::BitwiseOr),
        "BitwiseOr should fold away when ORing zero or self"
    );
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

#[test]
fn corpus_folds_mul_by_neg_one() {
    let int = 1;
    let c9 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(9)],
    );
    let c_neg_one = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(u32::MAX)],
    );
    let mul = inst(
        Op::IMul,
        int,
        12,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c9, c_neg_one, mul]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(12));
    assert_eq!(
        folded.operands,
        vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(9))]
    );
}

#[test]
fn corpus_folds_add_with_negated_operand() {
    let int = 1;
    let c11 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(11)],
    );
    let c4 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(4)],
    );
    let neg_four = inst(Op::SNegate, int, 3, vec![rspirv::dr::Operand::IdRef(2)]);
    let add = inst(
        Op::IAdd,
        int,
        13,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(3)],
    );
    let optimized = optimize_arith_block(&[c11, c4, neg_four, add]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(13));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(7)]);
}

#[test]
fn corpus_folds_bxor_constants() {
    let int = 1;
    let c1 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(3)],
    );
    let c2 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(5)],
    );
    let xor = inst(
        Op::BitwiseXor,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c1, c2, xor]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(6)]);
}

#[test]
fn corpus_folds_bxor_with_complement() {
    let int = 1;
    let value = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(0xA5A5_5A5A)],
    );
    let complement = inst(Op::Not, int, 2, vec![rspirv::dr::Operand::IdRef(1)]);
    let xor = inst(
        Op::BitwiseXor,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[value, complement, xor]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(3));
    assert_eq!(
        folded.operands,
        vec![rspirv::dr::Operand::LiteralBit32(u32::MAX)]
    );
}

#[test]
fn corpus_folds_bxor_with_complement_commuted() {
    let int = 1;
    let value = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(0xFFFF_0000)],
    );
    let complement = inst(Op::Not, int, 2, vec![rspirv::dr::Operand::IdRef(1)]);
    let xor = inst(
        Op::BitwiseXor,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(1)],
    );
    let optimized = optimize_arith_block(&[value, complement, xor]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(3));
    assert_eq!(
        folded.operands,
        vec![rspirv::dr::Operand::LiteralBit32(u32::MAX)]
    );
}

#[test]
fn corpus_folds_bnot_constants() {
    let int = 1;
    let c1 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(0)],
    );
    let bnot = inst(Op::Not, int, 2, vec![rspirv::dr::Operand::IdRef(1)]);
    let optimized = optimize_arith_block(&[c1, bnot]).expect("optimize");
    let folded = optimized
        .iter()
        .find(|inst| inst.result_id == Some(2))
        .expect("result id 2 present");
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(
        folded.operands,
        vec![rspirv::dr::Operand::LiteralBit32(u32::MAX)]
    );
}

#[test]
fn corpus_cancels_add_sub() {
    let int = 1;
    let ca = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(42)],
    );
    let cb = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(5)],
    );
    let sub = inst(
        Op::ISub,
        int,
        3,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let add = inst(
        Op::IAdd,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(3), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[ca, cb, sub, add]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(4));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(42)]);
}

#[test]
fn corpus_folds_udiv_by_one() {
    let int = 1;
    let c12 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(12)],
    );
    let c1 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let div = inst(
        Op::UDiv,
        int,
        20,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c12, c1, div]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(20));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(12)]);
}

#[test]
fn corpus_folds_umod_by_one() {
    let int = 1;
    let c13 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(13)],
    );
    let c1 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    let rem = inst(
        Op::UMod,
        int,
        21,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[c13, c1, rem]).expect("optimize");
    assert_eq!(optimized.len(), 1);
    let folded = &optimized[0];
    assert_eq!(folded.class.opcode, Op::Constant);
    assert_eq!(folded.result_id, Some(21));
    assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(0)]);
}

#[test]
fn corpus_preserves_umod_by_zero() {
    let int = 1;
    let c7 = inst(
        Op::Constant,
        int,
        1,
        vec![rspirv::dr::Operand::LiteralBit32(7)],
    );
    let c0 = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(0)],
    );
    let rem = inst(
        Op::UMod,
        int,
        22,
        vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
    );
    let block = vec![c7, c0, rem];
    let optimized = optimize_arith_block(&block).expect("optimize");
    assert_eq!(optimized, block, "umod by zero should be preserved");
}

#[test]
fn corpus_absorbs_band_over_bor_constant_case() {
    let int = 1;
    let x_val = 0x1234_u32;
    let y_val = 0xFFFF0000_u32;
    let x_const = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let y_const = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(y_val)],
    );
    let bor = inst(
        Op::BitwiseOr,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
    );
    let optimized = optimize_arith_block(&[x_const, y_const, bor, band]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(x_val)]
    });
    assert!(
        folded.is_some(),
        "band absorption should fold to the left operand constant"
    );
}

#[test]
fn corpus_distributes_bor_over_bxor_and_folds() {
    let int = 1;
    let x_val = 0xF0_u32;
    let y_val = 0x0F_u32;
    let z_val = 0xFF_u32;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let y = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(y_val)],
    );
    let z = inst(
        Op::Constant,
        int,
        4,
        vec![rspirv::dr::Operand::LiteralBit32(z_val)],
    );
    let xor = inst(
        Op::BitwiseXor,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(3), rspirv::dr::Operand::IdRef(4)],
    );
    let bor = inst(
        Op::BitwiseOr,
        int,
        6,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(5)],
    );
    let optimized = optimize_arith_block(&[x, y, z, xor, bor]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(6)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(x_val)]
    });
    assert!(
        folded.is_some(),
        "bor distribution over xor should fold to the expected constant"
    );
}

#[test]
fn corpus_absorbs_add_of_masked_value() {
    let int = 1;
    let x_val = 0xDEAD_BEEFu32;
    let mask_val = 0xFFFF0000u32;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let mask = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(mask_val)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let add = inst(
        Op::IAdd,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[x, mask, band, add]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(x_val)]
    });
    assert!(folded.is_some(), "add of x & mask plus x should fold to x");
}

#[test]
fn corpus_absorbs_bor_with_masked_value() {
    let int = 1;
    let x_val = 0x0F0F_F0F0u32;
    let mask_val = 0xFF00_FF00u32;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let mask = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(mask_val)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let bor = inst(
        Op::BitwiseOr,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
    );
    let optimized = optimize_arith_block(&[x, mask, band, bor]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(x_val)]
    });
    assert!(
        folded.is_some(),
        "bor absorption should fold to the left operand constant"
    );
}

#[test]
fn corpus_absorbs_sub_of_masked_value() {
    let int = 1;
    let x_val = 0xABCD_EF01u32;
    let mask_val = 0xFFFF_FFFFu32;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let mask = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(mask_val)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let sub = inst(
        Op::ISub,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(2)],
    );
    let optimized = optimize_arith_block(&[x, mask, band, sub]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        folded.is_some(),
        "subtraction of x from (x & all_ones) should fold to zero"
    );
}

#[test]
fn corpus_absorbs_xor_with_zero_masked_value() {
    let int = 1;
    let x_val = 0xCAFEBABEu32;
    let mask_val = 0u32;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let mask = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(mask_val)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let bxor = inst(
        Op::BitwiseXor,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
    );
    let optimized = optimize_arith_block(&[x, mask, band, bxor]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(x_val)]
    });
    assert!(folded.is_some(), "xor of x with (x & 0) should fold to x");
}

#[test]
fn corpus_folds_xor_with_all_ones_mask_to_zero() {
    let int = 1;
    let x_val = 0xCAFEBABEu32;
    let mask_val = u32::MAX;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let mask = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(mask_val)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let bxor = inst(
        Op::BitwiseXor,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
    );
    let optimized = optimize_arith_block(&[x, mask, band, bxor]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        folded.is_some(),
        "xor of x with (x & all ones) should fold to zero"
    );
}

#[test]
fn corpus_absorbs_or_with_zero_masked_value() {
    let int = 1;
    let x_val = 0x1234_5678u32;
    let mask_val = 0u32;
    let y_val = 0xFFFF_FFFFu32;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let mask = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(mask_val)],
    );
    let y = inst(
        Op::Constant,
        int,
        4,
        vec![rspirv::dr::Operand::LiteralBit32(y_val)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(3)],
    );
    let bor = inst(
        Op::BitwiseOr,
        int,
        6,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(5)],
    );
    let optimized = optimize_arith_block(&[x, mask, y, band, bor]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(6)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(x_val)]
    });
    assert!(
        folded.is_some(),
        "bor with (y & 0) should fold to the unmasked operand"
    );
}

#[test]
fn corpus_rewrites_xor_with_masked_self() {
    let int = 1;
    let x_val = 0x1234_5678u32;
    let mask_val = 0x00FF_00FFu32;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let mask = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(mask_val)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let bxor = inst(
        Op::BitwiseXor,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
    );
    let optimized = optimize_arith_block(&[x, mask, band, bxor]).expect("optimize");
    assert!(
        !optimized
            .iter()
            .any(|inst| inst.class.opcode == Op::BitwiseXor),
        "xor with masked self should normalize to band + not"
    );
    let mut opcode_by_id = std::collections::HashMap::new();
    for inst in &optimized {
        if let Some(id) = inst.result_id {
            opcode_by_id.insert(id, inst.class.opcode);
        }
    }
    let band_inst = optimized
        .iter()
        .find(|inst| inst.class.opcode == Op::BitwiseAnd)
        .expect("band should remain");
    let operands: Vec<_> = band_inst
        .operands
        .iter()
        .filter_map(|op| op.id_ref_any())
        .collect();
    assert!(operands.contains(&2), "band should keep original x");
    let not_id = operands
        .iter()
        .copied()
        .find(|id| *id != 2)
        .expect("band should include not operand");
    assert_eq!(
        opcode_by_id.get(&not_id),
        Some(&Op::Not),
        "band should use a not of the mask"
    );
    let not_inst = optimized
        .iter()
        .find(|inst| inst.result_id == Some(not_id))
        .expect("not instruction should exist");
    assert_eq!(
        not_inst.operands,
        vec![rspirv::dr::Operand::IdRef(3)],
        "not should target the original mask"
    );
}

#[test]
fn corpus_rewrites_xor_with_or_shared_operand() {
    let int = 1;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(0xAAAA5555)],
    );
    let y = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(0x0F0F0F0F)],
    );
    let bor = inst(
        Op::BitwiseOr,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let bxor = inst(
        Op::BitwiseXor,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
    );
    let optimized = optimize_arith_block(&[x, y, bor, bxor]).expect("optimize");
    assert!(
        !optimized
            .iter()
            .any(|inst| inst.class.opcode == Op::BitwiseXor),
        "xor with (x | y) should normalize to band + not"
    );
    let mut opcode_by_id = std::collections::HashMap::new();
    for inst in &optimized {
        if let Some(id) = inst.result_id {
            opcode_by_id.insert(id, inst.class.opcode);
        }
    }
    let band_inst = optimized
        .iter()
        .find(|inst| inst.class.opcode == Op::BitwiseAnd)
        .expect("band should remain");
    let operands: Vec<_> = band_inst
        .operands
        .iter()
        .filter_map(|op| op.id_ref_any())
        .collect();
    assert!(operands.contains(&3), "band should carry the other operand");
    let not_id = operands
        .iter()
        .copied()
        .find(|id| *id != 3)
        .expect("band should include not of x");
    assert_eq!(
        opcode_by_id.get(&not_id),
        Some(&Op::Not),
        "band should use a not of the shared xor/or operand"
    );
    let not_inst = optimized
        .iter()
        .find(|inst| inst.result_id == Some(not_id))
        .expect("not instruction should exist");
    assert_eq!(
        not_inst.operands,
        vec![rspirv::dr::Operand::IdRef(2)],
        "not should target the shared x"
    );
}

/// The Rust optimizer rewrites a De Morgan xor form into a canonical xor; C++ leaves it expanded.
#[test]
fn corpus_collapses_demorgan_xor() {
    let int = 1;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(0xAAAA_5555)],
    );
    let y = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(0x0F0F_F0F0)],
    );
    let not_x = inst(Op::Not, int, 4, vec![rspirv::dr::Operand::IdRef(2)]);
    let not_y = inst(Op::Not, int, 5, vec![rspirv::dr::Operand::IdRef(3)]);
    let band1 = inst(
        Op::BitwiseAnd,
        int,
        6,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(5)],
    );
    let band2 = inst(
        Op::BitwiseAnd,
        int,
        7,
        vec![rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(3)],
    );
    let bor = inst(
        Op::BitwiseOr,
        int,
        8,
        vec![rspirv::dr::Operand::IdRef(6), rspirv::dr::Operand::IdRef(7)],
    );
    let optimized =
        optimize_arith_block(&[x, y, not_x, not_y, band1, band2, bor]).expect("optimize");
    assert!(
        optimized
            .iter()
            .any(|inst| inst.class.opcode == Op::BitwiseXor && inst.result_id == Some(8)),
        "De Morgan xor form should fold to bxor"
    );
    assert!(
        !optimized
            .iter()
            .any(|inst| inst.class.opcode == Op::BitwiseAnd && inst.result_id == Some(6)),
        "band arms should be removed after xor folding"
    );
    assert!(
        !optimized
            .iter()
            .any(|inst| inst.class.opcode == Op::BitwiseAnd && inst.result_id == Some(7)),
        "band arms should be removed after xor folding"
    );
}

#[test]
fn corpus_absorbs_and_over_or() {
    let int = 1;
    let x_val = 0xABCDu32;
    let y_val = 0x00FFu32;
    let x = inst(
        Op::Constant,
        int,
        2,
        vec![rspirv::dr::Operand::LiteralBit32(x_val)],
    );
    let y = inst(
        Op::Constant,
        int,
        3,
        vec![rspirv::dr::Operand::LiteralBit32(y_val)],
    );
    let bor = inst(
        Op::BitwiseOr,
        int,
        4,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let band = inst(
        Op::BitwiseAnd,
        int,
        5,
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
    );
    let optimized = optimize_arith_block(&[x, y, bor, band]).expect("optimize");
    let folded = optimized.iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(x_val)]
    });
    assert!(folded.is_some(), "band with (x | y) should absorb to x");
}
