#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use rspirv::{dr::Instruction, spirv::Op};
use spirv_tools_opt::translate::optimize_arith_block;

fn generate_block(u: &mut Unstructured<'_>) -> Option<Vec<Instruction>> {
    let len = u.int_in_range::<usize>(1..=48).ok()?;
    let mut insts = Vec::with_capacity(len);
    let mut ids: Vec<u32> = Vec::new();
    let mut next_id: u32 = 1;
    let ty_int = 1;

    let mut next_const = |u: &mut Unstructured<'_>, ids: &mut Vec<u32>, next_id: &mut u32| {
        let literal = u.arbitrary::<u32>().unwrap_or(0);
        let id = *next_id;
        *next_id += 1;
        ids.push(id);
        Instruction::new(
            Op::Constant,
            Some(ty_int),
            Some(id),
            vec![rspirv::dr::Operand::LiteralBit32(literal)],
        )
    };

    insts.push(next_const(u, &mut ids, &mut next_id));

    for _ in 1..len {
        let choice = u
            .choose(&[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .ok()
            .copied()
            .unwrap_or(0);
        let inst = match choice {
            0 => next_const(u, &mut ids, &mut next_id),
            1 => bin_op(&mut ids, &mut next_id, u, Op::IAdd, ty_int)?,
            2 => bin_op(&mut ids, &mut next_id, u, Op::ISub, ty_int)?,
            3 => bin_op(&mut ids, &mut next_id, u, Op::IMul, ty_int)?,
            4 => bin_op(&mut ids, &mut next_id, u, Op::SDiv, ty_int)?,
            5 => bin_op(&mut ids, &mut next_id, u, Op::UDiv, ty_int)?,
            6 => bin_op(&mut ids, &mut next_id, u, Op::BitwiseAnd, ty_int)?,
            7 => bin_op(&mut ids, &mut next_id, u, Op::BitwiseOr, ty_int)?,
            8 => bin_op(&mut ids, &mut next_id, u, Op::BitwiseXor, ty_int)?,
            _ => {
                // Mix in a mask-shift to keep some structured patterns.
                let (a, b) = pick_two(u, &ids)?;
                Instruction::new(
                    Op::ShiftRightLogical,
                    Some(ty_int),
                    Some(take_id(&mut next_id, &mut ids)),
                    vec![id(a), id(b)],
                )
            }
        };
        insts.push(inst);
    }
    Some(insts)
}

fn bin_op(
    ids: &mut Vec<u32>,
    next_id: &mut u32,
    u: &mut Unstructured<'_>,
    op: Op,
    ty: u32,
) -> Option<Instruction> {
    let (a, b) = pick_two(u, ids)?;
    let res = take_id(next_id, ids);
    Some(Instruction::new(op, Some(ty), Some(res), vec![id(a), id(b)]))
}

fn pick_two(u: &mut Unstructured<'_>, ids: &[u32]) -> Option<(u32, u32)> {
    if ids.is_empty() {
        return None;
    }
    let idx_a = u.int_in_range::<usize>(0..ids.len()).ok()?;
    let idx_b = u.int_in_range::<usize>(0..ids.len()).ok()?;
    Some((ids[idx_a], ids[idx_b]))
}

fn take_id(next_id: &mut u32, ids: &mut Vec<u32>) -> u32 {
    let id = *next_id;
    *next_id += 1;
    ids.push(id);
    id
}

fn id(value: u32) -> rspirv::dr::Operand {
    rspirv::dr::Operand::IdRef(value)
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Some(block) = generate_block(&mut u) {
        let _ = optimize_arith_block(&block);
    }
});
