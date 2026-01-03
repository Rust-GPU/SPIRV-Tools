use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spirv_tools_opt::{optimize_expr, ConstValue, Id, RecExpr, SpirvLang, Symbol};

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

    c.bench_function("optimize small expr", |b| {
        b.iter(|| optimize_expr(black_box(&expr_small)))
    });

    c.bench_function("optimize medium expr", |b| {
        b.iter(|| optimize_expr(black_box(&expr_medium)))
    });

    c.bench_function("optimize affine expr", |b| {
        b.iter(|| optimize_expr(black_box(&expr_affine)))
    });
}

criterion_group!(benches, bench_optimize);
criterion_main!(benches);

fn affine_expr() -> RecExpr<SpirvLang> {
    // (2*x) + (x*3) -> affine mixed-constant pattern
    let mut nodes = Vec::new();
    let x = Id::from(0);
    nodes.push(SpirvLang::Symbol(Symbol::from("x")));
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
