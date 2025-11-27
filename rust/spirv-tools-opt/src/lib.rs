//! E-graph driven optimizer scaffolding for SPIR-V tools.
//!
//! This crate builds on `egg` to provide a rewrite-friendly representation of
//! SPIR-V expressions. The initial focus is simple arithmetic canonicalization
//! and constant folding; subsequent work will map these expressions to real
//! SPIR-V modules and expose the optimizer through FFI/CLI surfaces.

use egg::{
    define_language, rewrite, Applier, EGraph, Id, Language, PatternAst, RecExpr, Rewrite, Runner,
    Subst, Symbol, Var,
};
use std::{fmt, str::FromStr};

/// Domain-specific constant value used in the e-graph.
///
/// Keeping this strongly typed lets us define deterministic folding semantics
/// (wrapping 32-bit arithmetic for now) and makes room for SPIR-V-specific
/// numeric domains later (e.g., literals with bit-width).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstValue(u32);

impl ConstValue {
    /// Constructs a new constant value.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the raw 32-bit value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ConstValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for ConstValue {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u32>().map(Self)
    }
}

impl egg::LanguageChildren for ConstValue {
    fn len(&self) -> usize {
        0
    }

    fn can_be_length(n: usize) -> bool {
        n == 0
    }

    fn from_vec(v: Vec<Id>) -> Self {
        assert!(v.is_empty());
        ConstValue(0)
    }

    fn as_slice(&self) -> &[Id] {
        &[]
    }

    fn as_mut_slice(&mut self) -> &mut [Id] {
        &mut []
    }
}

define_language! {
    /// Minimal language for algebraic SPIR-V expressions.
    pub enum SpirvLang {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "const" = Const(ConstValue),
        Symbol(egg::Symbol),
    }
}

/// Lightweight cost function favoring folded constants and shallower trees.
struct ExprCost;

impl egg::CostFunction<SpirvLang> for ExprCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &SpirvLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            SpirvLang::Const(_) => 1,
            _ => enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 1,
        }
    }
}

/// Optimize an expression by applying algebraic rewrites and constant folding.
///
/// The returned expression is the cheapest representative (per `ExprCost`) of
/// the root e-class after saturation.
pub fn optimize_expr(expr: &RecExpr<SpirvLang>) -> RecExpr<SpirvLang> {
    let rewrites = rewrites();
    let runner = Runner::default().with_expr(expr).run(&rewrites);
    let root = runner.roots[0];
    let extractor = egg::Extractor::new(&runner.egraph, ExprCost);
    let (_cost, best) = extractor.find_best(root);
    best
}

fn rewrites() -> Vec<Rewrite<SpirvLang, ()>> {
    vec![
        rewrite!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-assoc"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("mul-assoc"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rewrite!("add-fold"; "(+ ?a ?b)" => { FoldAdd }),
        rewrite!("mul-fold"; "(* ?a ?b)" => { FoldMul }),
    ]
}

struct FoldAdd;
struct FoldMul;

impl Applier<SpirvLang, ()> for FoldAdd {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(a) = const_value(egraph, subst[var("a")]) else {
            return Vec::new();
        };
        let Some(b) = const_value(egraph, subst[var("b")]) else {
            return Vec::new();
        };
        let sum = ConstValue::new(a.get().wrapping_add(b.get()));
        let id = egraph.add(SpirvLang::Const(sum));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for FoldMul {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(a) = const_value(egraph, subst[var("a")]) else {
            return Vec::new();
        };
        let Some(b) = const_value(egraph, subst[var("b")]) else {
            return Vec::new();
        };
        let product = ConstValue::new(a.get().wrapping_mul(b.get()));
        let id = egraph.add(SpirvLang::Const(product));
        egraph.union(eclass, id);
        vec![id]
    }
}

fn const_value(egraph: &EGraph<SpirvLang, ()>, id: Id) -> Option<ConstValue> {
    egraph[id].nodes.iter().find_map(|node| match node {
        SpirvLang::Const(value) => Some(*value),
        _ => None,
    })
}

fn var(name: &str) -> Var {
    let formatted = if name.starts_with('?') {
        name.to_owned()
    } else {
        format!("?{name}")
    };
    Var::from_str(&formatted).expect("valid e-graph variable name")
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn folds_addition() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(2)),
            SpirvLang::Const(ConstValue::new(3)),
            SpirvLang::Add([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(5))])
        );
    }

    #[test]
    fn folds_multiplication_with_commutativity() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(2)),       // 0
            SpirvLang::Const(ConstValue::new(4)),       // 1
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 2 = 2 + 4
            SpirvLang::Const(ConstValue::new(3)),       // 3
            SpirvLang::Mul([Id::from(3), Id::from(2)]), // 4 = 3 * (2+4)
        ]);
        let optimized = optimize_expr(&expr);
        // (2 + 4) -> 6, then 3 * 6 -> 18
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(18))])
        );
    }

    #[test]
    fn preserves_non_constant_expressions() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(ConstValue::new(4)),
            SpirvLang::Add([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        // We cannot fold because x is symbolic; allow commutativity to reorder operands.
        let reordered = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(4)),
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Add([Id::from(0), Id::from(1)]),
        ]);
        let reordered_swapped = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(4)),
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Add([Id::from(1), Id::from(0)]),
        ]);
        assert!(
            optimized == expr || optimized == reordered || optimized == reordered_swapped,
            "unexpected optimization result: {:?}",
            optimized
        );
    }
}
