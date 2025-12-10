#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rspirv::dr::Instruction;
use rspirv::spirv::{Op, StorageClass};
use spirv_tools_opt::translate::optimize_arith_block;

#[derive(Arbitrary, Debug, Clone)]
enum MaskKind {
    And,
    Or,
    Xor,
}

#[derive(Arbitrary, Debug, Clone)]
struct ArithBitwiseCase {
    lhs: u32,
    rhs: u32,
    op: MaskKind,
    include_not: bool,
    split_mode: bool,
}

fn make_const(id: u32, val: u32) -> Instruction {
    Instruction::new(
        Op::Constant,
        Some(1),
        Some(id),
        vec![rspirv::dr::Operand::LiteralBit32(val)],
    )
}

fuzz_target!(|case: ArithBitwiseCase| {
    // Build a tiny module with integer type id 1 and basic operations over ids 2..6.
    let mut insts = Vec::new();
    insts.push(Instruction::new(
        Op::TypeInt,
        None,
        Some(1),
        vec![rspirv::dr::Operand::LiteralBit32(32), rspirv::dr::Operand::LiteralBit32(0)],
    ));
    let lhs = make_const(2, case.lhs);
    let rhs = make_const(3, case.rhs);
    insts.push(lhs);
    insts.push(rhs);

    // Optionally negate one side to explore (~x) patterns.
    let lhs_id = if case.include_not {
        let not_id = 4;
        insts.push(Instruction::new(
            Op::Not,
            Some(1),
            Some(not_id),
            vec![rspirv::dr::Operand::IdRef(2)],
        ));
        not_id
    } else {
        2
    };

    let rhs_id = if case.split_mode {
        // Inject an additional constant and AND it to create split absorption shapes.
        let extra_const = make_const(5, case.rhs.wrapping_add(1));
        insts.push(extra_const);
        insts.push(Instruction::new(
            Op::BitwiseAnd,
            Some(1),
            Some(6),
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::IdRef(5),
            ],
        ));
        6
    } else {
        3
    };

    let bit_inst = match case.op {
        MaskKind::And => Instruction::new(
            Op::BitwiseAnd,
            Some(1),
            Some(7),
            vec![rspirv::dr::Operand::IdRef(lhs_id), rspirv::dr::Operand::IdRef(rhs_id)],
        ),
        MaskKind::Or => Instruction::new(
            Op::BitwiseOr,
            Some(1),
            Some(7),
            vec![rspirv::dr::Operand::IdRef(lhs_id), rspirv::dr::Operand::IdRef(rhs_id)],
        ),
        MaskKind::Xor => Instruction::new(
            Op::BitwiseXor,
            Some(1),
            Some(7),
            vec![rspirv::dr::Operand::IdRef(lhs_id), rspirv::dr::Operand::IdRef(rhs_id)],
        ),
    };
    insts.push(bit_inst);

    let _ = optimize_arith_block(&insts);
});
