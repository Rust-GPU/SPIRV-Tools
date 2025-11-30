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
    // Rotation guard should stop folding this into a full rotate; allow constant folding of the pieces.
    let is_const_rotate = optimized.len() == 1
        && optimized
            .iter()
            .any(|inst| inst.class.opcode == Op::Constant && inst.result_id == Some(6));
    assert!(
        !is_const_rotate,
        "non-complementary shift-or should not fold to a single rotate constant: {optimized:?}"
    );
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
