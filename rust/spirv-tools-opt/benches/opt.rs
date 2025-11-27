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
    let block_fold = spirv_block_add_zero();

    c.bench_function("optimize small expr", |b| {
        b.iter(|| optimize_expr(black_box(&expr_small)))
    });

    c.bench_function("optimize medium expr", |b| {
        b.iter(|| optimize_expr(black_box(&expr_medium)))
    });

    c.bench_function("optimize arithmetic block", |b| {
        b.iter(|| {
            let optimized = optimize_arith_block(black_box(&block_fold)).unwrap();
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

criterion_group!(benches, bench_optimize);
criterion_main!(benches);
