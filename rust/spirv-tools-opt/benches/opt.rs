use criterion::{black_box, criterion_group, criterion_main, Criterion};
use egg::{Id, RecExpr};
use spirv_tools_opt::{optimize_expr, translate::optimize_arith_block, ConstValue, SpirvLang};

fn dense_expr(depth: usize) -> RecExpr<SpirvLang> {
    // Build a left-associative tree: (((c0 + c1) * c2) + c3) * ...
    let mut nodes = Vec::new();
    let mut last = None;
    for i in 0..depth {
        nodes.push(SpirvLang::Const(ConstValue::new(i as u32 + 1)));
        if let Some(prev) = last {
            let add = SpirvLang::Add([prev, Id::from(nodes.len() - 1)]);
            nodes.push(add);
            last = Some(Id::from(nodes.len() - 1));
        } else {
            last = Some(Id::from(nodes.len() - 1));
        }
        if i % 2 == 1 {
            // every other step multiply by the running sum to mix operators
            let mul = SpirvLang::Mul([last.unwrap(), Id::from(nodes.len() - 1)]);
            nodes.push(mul);
            last = Some(Id::from(nodes.len() - 1));
        }
    }
    RecExpr::from(nodes)
}

fn bench_optimize(c: &mut Criterion) {
    let expr_small = dense_expr(8);
    let expr_medium = dense_expr(32);
    let expr_affine = affine_expr();
    let block_fold = spirv_block_add_zero();
    let block_medium = arith_block(32);
    let bitwise_block = bitwise_mixed_block();

    c.bench_function("optimize small expr", |b| {
        b.iter(|| optimize_expr(black_box(&expr_small)))
    });

    c.bench_function("optimize medium expr", |b| {
        b.iter(|| optimize_expr(black_box(&expr_medium)))
    });

    c.bench_function("optimize affine expr", |b| {
        b.iter(|| optimize_expr(black_box(&expr_affine)))
    });

    c.bench_function("optimize arithmetic block", |b| {
        b.iter(|| {
            let optimized = optimize_arith_block(black_box(&block_fold)).unwrap();
            black_box(optimized)
        })
    });

    c.bench_function("optimize arithmetic block medium", |b| {
        b.iter(|| {
            let optimized = optimize_arith_block(black_box(&block_medium)).unwrap();
            black_box(optimized)
        })
    });

    c.bench_function("optimize bitwise block", |b| {
        b.iter(|| {
            let optimized = optimize_arith_block(black_box(&bitwise_block)).unwrap();
            black_box(optimized)
        })
    });
}

fn spirv_block_add_zero() -> Vec<rspirv::dr::Instruction> {
    let int = 1;
    let two_id = 2;
    let zero_id = 3;
    let add_id = 4;

    vec![
        rspirv::dr::Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(two_id),
            vec![rspirv::dr::Operand::LiteralBit32(2)],
        ),
        rspirv::dr::Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(zero_id),
            vec![rspirv::dr::Operand::LiteralBit32(0)],
        ),
        rspirv::dr::Instruction::new(
            rspirv::spirv::Op::IAdd,
            Some(int),
            Some(add_id),
            vec![
                rspirv::dr::Operand::IdRef(two_id),
                rspirv::dr::Operand::IdRef(zero_id),
            ],
        ),
    ]
}

fn arith_block(depth: usize) -> Vec<rspirv::dr::Instruction> {
    use rspirv::dr::Operand;
    use rspirv::spirv::Op;

    let mut insts = Vec::new();
    let ty = 1;
    let mut next_id = 1u32;

    // Seed with one constant.
    insts.push(rspirv::dr::Instruction::new(
        Op::Constant,
        Some(ty),
        Some(next_id),
        vec![Operand::LiteralBit32(1)],
    ));
    next_id += 1;

    for i in 0..depth {
        let const_id = next_id;
        next_id += 1;
        insts.push(rspirv::dr::Instruction::new(
            Op::Constant,
            Some(ty),
            Some(const_id),
            vec![Operand::LiteralBit32(i as u32 + 2)],
        ));

        let prev_id = const_id - 1;
        let result_id = next_id;
        next_id += 1;
        let opcode = if i % 2 == 0 { Op::IAdd } else { Op::IMul };
        insts.push(rspirv::dr::Instruction::new(
            opcode,
            Some(ty),
            Some(result_id),
            vec![Operand::IdRef(prev_id), Operand::IdRef(const_id)],
        ));
    }

    insts
}

fn bitwise_mixed_block() -> Vec<rspirv::dr::Instruction> {
    use rspirv::dr::Operand;
    use rspirv::spirv::Op;

    let mut insts = Vec::new();
    let ty = 1;
    let mut next_id = 1u32;

    let const32 = |val: u32, id: u32| {
        rspirv::dr::Instruction::new(
            Op::Constant,
            Some(ty),
            Some(id),
            vec![Operand::LiteralBit32(val)],
        )
    };

    insts.push(const32(0xFFFF0000, next_id));
    let mask1 = next_id;
    next_id += 1;
    insts.push(const32(0x00FF00FF, next_id));
    let mask2 = next_id;
    next_id += 1;
    insts.push(const32(0x12345678, next_id));
    let base = next_id;
    next_id += 1;

    let band1 = next_id;
    next_id += 1;
    insts.push(rspirv::dr::Instruction::new(
        Op::BitwiseAnd,
        Some(ty),
        Some(band1),
        vec![Operand::IdRef(base), Operand::IdRef(mask1)],
    ));

    let bor = next_id;
    next_id += 1;
    insts.push(rspirv::dr::Instruction::new(
        Op::BitwiseOr,
        Some(ty),
        Some(bor),
        vec![Operand::IdRef(band1), Operand::IdRef(mask2)],
    ));

    let band2 = next_id;
    next_id += 1;
    insts.push(rspirv::dr::Instruction::new(
        Op::BitwiseAnd,
        Some(ty),
        Some(band2),
        vec![Operand::IdRef(bor), Operand::IdRef(mask2)],
    ));

    let add = next_id;
    insts.push(rspirv::dr::Instruction::new(
        Op::IAdd,
        Some(ty),
        Some(add),
        vec![Operand::IdRef(band2), Operand::IdRef(base)],
    ));

    insts
}

criterion_group!(benches, bench_optimize);
criterion_main!(benches);

fn affine_expr() -> RecExpr<SpirvLang> {
    // (2*x) + (x*3) -> affine mixed-constant pattern
    let mut nodes = Vec::new();
    let x = Id::from(0);
    nodes.push(SpirvLang::Symbol("x".into()));
    nodes.push(SpirvLang::Const(ConstValue::new(2)));
    nodes.push(SpirvLang::Const(ConstValue::new(3)));

    let mul1 = SpirvLang::Mul([Id::from(1), x]);
    nodes.push(mul1);
    let mul2 = SpirvLang::Mul([x, Id::from(2)]);
    nodes.push(mul2);
    let add = SpirvLang::Add([Id::from(3), Id::from(4)]);
    nodes.push(add);

    RecExpr::from(nodes)
}
