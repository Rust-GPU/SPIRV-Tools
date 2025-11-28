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

pub mod translate;

/// Helpers for fuzzing and property-based generation.
pub mod fuzzing {
    use super::{ConstValue, SpirvLang};
    use arbitrary::Unstructured;
    use egg::{Id, RecExpr};

    /// Generate a random, well-formed expression from fuzzer input bytes.
    ///
    /// Nodes are created in order so child references always point to earlier
    /// nodes, preserving `RecExpr` invariants even when the input is adversarial.
    pub fn arbitrary_expr(u: &mut Unstructured<'_>) -> arbitrary::Result<RecExpr<SpirvLang>> {
        // Cap size to keep fuzzing bounded.
        let len = u.int_in_range::<usize>(1..=64)?;
        let mut nodes = Vec::with_capacity(len);
        for idx in 0..len {
            // Ensure we always have at least one child to point at.
            let choose_child =
                |u: &mut Unstructured<'_>, max: usize| u.int_in_range::<usize>(0..=max);
            let node = if idx == 0 {
                SpirvLang::Const(ConstValue::new(u.arbitrary()?))
            } else {
                match u.choose(&[0u8, 1, 2, 3, 4, 5, 6])? {
                    0 => SpirvLang::Const(ConstValue::new(u.arbitrary()?)),
                    1 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::Add([Id::from(a), Id::from(b)])
                    }
                    2 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::Mul([Id::from(a), Id::from(b)])
                    }
                    3 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        if *u.choose(&[true, false])? {
                            SpirvLang::Sub([Id::from(a), Id::from(b)])
                        } else {
                            SpirvLang::Neg(Id::from(a))
                        }
                    }
                    4 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        if *u.choose(&[true, false])? {
                            SpirvLang::SDiv([Id::from(a), Id::from(b)])
                        } else {
                            SpirvLang::UDiv([Id::from(a), Id::from(b)])
                        }
                    }
                    _ => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        if *u.choose(&[true, false])? {
                            SpirvLang::SRem([Id::from(a), Id::from(b)])
                        } else {
                            SpirvLang::UMod([Id::from(a), Id::from(b)])
                        }
                    }
                }
            };
            nodes.push(node);
        }
        Ok(RecExpr::from(nodes))
    }
}

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
        "-" = Sub([Id; 2]),
        "sdiv" = SDiv([Id; 2]),
        "udiv" = UDiv([Id; 2]),
        "srem" = SRem([Id; 2]),
        "umod" = UMod([Id; 2]),
        "neg" = Neg(Id),
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

/// Optimize the root of a translated SPIR-V arithmetic expression.
pub fn optimize_translated(expr: &crate::translate::TranslatedExpr) -> RecExpr<SpirvLang> {
    optimize_expr(&expr.expr)
}

fn rewrites() -> Vec<Rewrite<SpirvLang, ()>> {
    vec![
        rewrite!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-assoc"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("mul-assoc"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rewrite!("add-zero"; "(+ ?a ?b)" => { AddZero { a: var("?a"), b: var("?b") } }),
        rewrite!("add-neg-to-sub"; "(+ ?a (neg ?b))" => "(- ?a ?b)"),
        rewrite!("add-neg-to-sub-swap"; "(+ (neg ?a) ?b)" => "(- ?b ?a)"),
        rewrite!("mul-one"; "(* ?a ?b)" => { MulOne { a: var("?a"), b: var("?b") } }),
        rewrite!("mul-zero"; "(* ?a ?b)" => { MulZero { a: var("?a"), b: var("?b") } }),
        rewrite!("mul-neg-one"; "(* ?a ?b)" => { MulNegOne { a: var("?a"), b: var("?b") } }),
        rewrite!("mul-double-neg"; "(* (neg ?a) (neg ?b))" => "(* ?a ?b)"),
        rewrite!("add-fold"; "(+ ?a ?b)" => { FoldAdd }),
        rewrite!("mul-fold"; "(* ?a ?b)" => { FoldMul }),
        rewrite!("sub-fold"; "(- ?a ?b)" => { FoldSub }),
        rewrite!("sub-zero-right"; "(- ?a ?b)" => "?a" if is_const_zero(var("?b"))),
        rewrite!("sub-zero-left"; "(- ?a ?b)" => { SubZeroLeft }),
        rewrite!("sub-self"; "(- ?a ?a)" => { SubSelf }),
        rewrite!("sub-neg-right-to-add"; "(- ?a (neg ?b))" => "(+ ?a ?b)"),
        rewrite!("add-sub-cancel-right"; "(+ (- ?a ?b) ?b)" => "?a"),
        rewrite!("add-sub-cancel-left"; "(+ ?b (- ?a ?b))" => "?a"),
        rewrite!("sub-add-cancel-right"; "(- (+ ?a ?b) ?b)" => "?a"),
        rewrite!("sub-add-cancel-left"; "(- (+ ?a ?b) ?a)" => "?b"),
        rewrite!("sub-sub-cancel-left"; "(- ?a (- ?a ?b))" => "?b"),
        rewrite!("add-factor-consts"; "(+ (* ?x ?c1) (* ?x ?c2))" => {
            AddCommonFactor { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-factor-consts-mixed"; "(+ (* ?c1 ?x) (* ?x ?c2))" => {
            AddCommonFactor { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-factor-consts-right"; "(+ (* ?x ?c1) (* ?c2 ?x))" => {
            AddCommonFactor { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-factor-consts-both"; "(+ (* ?c1 ?x) (* ?c2 ?x))" => {
            AddCommonFactor { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-factor-symbolic"; "(+ (* ?x ?y) (* ?x ?z))" => {
            AddCommonFactorGeneral { x: var("?x"), y: var("?y"), z: var("?z") }
        }),
        rewrite!("add-factor-symbolic-mixed"; "(+ (* ?y ?x) (* ?x ?z))" => {
            AddCommonFactorGeneral { x: var("?x"), y: var("?y"), z: var("?z") }
        }),
        rewrite!("add-factor-symbolic-right"; "(+ (* ?x ?y) (* ?z ?x))" => {
            AddCommonFactorGeneral { x: var("?x"), y: var("?y"), z: var("?z") }
        }),
        rewrite!("add-factor-symbolic-both"; "(+ (* ?y ?x) (* ?z ?x))" => {
            AddCommonFactorGeneral { x: var("?x"), y: var("?y"), z: var("?z") }
        }),
        rewrite!("sub-factor-consts"; "(- (* ?x ?c1) (* ?x ?c2))" => {
            SubCommonFactor { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("sub-factor-consts-mixed"; "(- (* ?c1 ?x) (* ?x ?c2))" => {
            SubCommonFactor { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("sub-factor-consts-right"; "(- (* ?x ?c1) (* ?c2 ?x))" => {
            SubCommonFactor { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("sub-factor-consts-both"; "(- (* ?c1 ?x) (* ?c2 ?x))" => {
            SubCommonFactor { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("sub-factor-symbolic"; "(- (* ?x ?y) (* ?x ?z))" => {
            SubCommonFactorGeneral { x: var("?x"), y: var("?y"), z: var("?z") }
        }),
        rewrite!("sub-factor-symbolic-mixed"; "(- (* ?y ?x) (* ?x ?z))" => {
            SubCommonFactorGeneral { x: var("?x"), y: var("?y"), z: var("?z") }
        }),
        rewrite!("sub-factor-symbolic-right"; "(- (* ?x ?y) (* ?z ?x))" => {
            SubCommonFactorGeneral { x: var("?x"), y: var("?y"), z: var("?z") }
        }),
        rewrite!("sub-factor-symbolic-both"; "(- (* ?y ?x) (* ?z ?x))" => {
            SubCommonFactorGeneral { x: var("?x"), y: var("?y"), z: var("?z") }
        }),
        rewrite!("sdiv-merge-consts"; "(sdiv (sdiv ?x ?c1) ?c2)" => {
            DivMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2"), signed: true }
        }),
        rewrite!("udiv-merge-consts"; "(udiv (udiv ?x ?c1) ?c2)" => {
            DivMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2"), signed: false }
        }),
        rewrite!("sdiv-cancel-common-factor-left"; "(sdiv (* ?c ?x) ?c)" => {
            CancelMulDiv { x: var("?x"), c: var("?c") }
        }),
        rewrite!("sdiv-cancel-common-factor-right"; "(sdiv (* ?x ?c) ?c)" => {
            CancelMulDiv { x: var("?x"), c: var("?c") }
        }),
        rewrite!("udiv-cancel-common-factor-left"; "(udiv (* ?c ?x) ?c)" => {
            CancelMulDiv { x: var("?x"), c: var("?c") }
        }),
        rewrite!("udiv-cancel-common-factor-right"; "(udiv (* ?x ?c) ?c)" => {
            CancelMulDiv { x: var("?x"), c: var("?c") }
        }),
        rewrite!("srem-mul-const-zero-left"; "(srem (* ?c ?x) ?c)" => {
            RemMulConstZero { c: var("?c") }
        }),
        rewrite!("srem-mul-const-zero-right"; "(srem (* ?x ?c) ?c)" => {
            RemMulConstZero { c: var("?c") }
        }),
        rewrite!("umod-mul-const-zero-left"; "(umod (* ?c ?x) ?c)" => {
            RemMulConstZero { c: var("?c") }
        }),
        rewrite!("umod-mul-const-zero-right"; "(umod (* ?x ?c) ?c)" => {
            RemMulConstZero { c: var("?c") }
        }),
        rewrite!("neg-mul-const-left"; "(neg (* ?c ?x))" => {
            NegMulConst { c: var("?c"), x: var("?x") }
        }),
        rewrite!("neg-mul-const-right"; "(neg (* ?x ?c))" => {
            NegMulConst { c: var("?c"), x: var("?x") }
        }),
        rewrite!("mul-merge-consts-right"; "(* (* ?x ?c1) ?c2)" => {
            MulMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("mul-merge-consts-left"; "(* (* ?c1 ?x) ?c2)" => {
            MulMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("mul-merge-consts-root-left"; "(* ?c2 (* ?x ?c1))" => {
            MulMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("mul-merge-consts-root-right"; "(* ?c2 (* ?c1 ?x))" => {
            MulMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-merge-consts-right"; "(+ (+ ?x ?c1) ?c2)" => {
            AddMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-merge-consts-left"; "(+ (+ ?c1 ?x) ?c2)" => {
            AddMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-merge-consts-root-left"; "(+ ?c2 (+ ?x ?c1))" => {
            AddMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-merge-consts-root-right"; "(+ ?c2 (+ ?c1 ?x))" => {
            AddMergeConst { base: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("add-sub-merge-consts"; "(+ (- ?x ?c1) ?c2)" => {
            AddSubMerge { base: var("?x"), sub_const: var("?c1"), add_const: var("?c2") }
        }),
        rewrite!("add-sub-merge-consts-comm"; "(+ ?c2 (- ?x ?c1))" => {
            AddSubMerge { base: var("?x"), sub_const: var("?c1"), add_const: var("?c2") }
        }),
        rewrite!("add-sub-merge-const-lhs"; "(+ (- ?c1 ?x) ?c2)" => {
            AddSubConstLhs { base_const: var("?c1"), rhs: var("?x"), add_const: var("?c2") }
        }),
        rewrite!("add-sub-merge-const-lhs-comm"; "(+ ?c2 (- ?c1 ?x))" => {
            AddSubConstLhs { base_const: var("?c1"), rhs: var("?x"), add_const: var("?c2") }
        }),
        rewrite!("neg-sub-swap"; "(neg (- ?a ?b))" => "(- ?b ?a)"),
        rewrite!("neg-fold"; "(neg ?a)" => { FoldNeg }),
        rewrite!("double-neg"; "(neg (neg ?a))" => "?a"),
        rewrite!("sdiv-fold"; "(sdiv ?a ?b)" => { FoldDiv { signed: true } }),
        rewrite!("udiv-fold"; "(udiv ?a ?b)" => { FoldDiv { signed: false } }),
        rewrite!("sdiv-one"; "(sdiv ?a ?b)" => { DivOne { a: var("?a"), b: var("?b") } }),
        rewrite!("udiv-one"; "(udiv ?a ?b)" => { DivOne { a: var("?a"), b: var("?b") } }),
        rewrite!("srem-fold"; "(srem ?a ?b)" => { FoldRem { signed: true } }),
        rewrite!("umod-fold"; "(umod ?a ?b)" => { FoldRem { signed: false } }),
        rewrite!("srem-one"; "(srem ?a ?b)" => { RemOne { b: var("?b") } }),
        rewrite!("umod-one"; "(umod ?a ?b)" => { RemOne { b: var("?b") } }),
        rewrite!("add-neg-cancel"; "(+ ?a (neg ?a))" => { AddNegZero }),
        rewrite!("add-neg-cancel-swap"; "(+ (neg ?a) ?a)" => { AddNegZero }),
    ]
}

struct FoldAdd;
struct FoldMul;
struct FoldSub;
struct FoldNeg;
struct FoldDiv {
    signed: bool,
}
struct FoldRem {
    signed: bool,
}
struct SubSelf;
struct SubZeroLeft;
struct DivOne {
    a: Var,
    b: Var,
}
struct RemOne {
    b: Var,
}
struct AddNegZero;
struct AddZero {
    a: Var,
    b: Var,
}
struct MulOne {
    a: Var,
    b: Var,
}
struct MulZero {
    a: Var,
    b: Var,
}
struct MulNegOne {
    a: Var,
    b: Var,
}
struct AddMergeConst {
    base: Var,
    c1: Var,
    c2: Var,
}
struct AddSubMerge {
    base: Var,
    sub_const: Var,
    add_const: Var,
}
struct AddSubConstLhs {
    base_const: Var,
    rhs: Var,
    add_const: Var,
}
struct DivMergeConst {
    base: Var,
    c1: Var,
    c2: Var,
    signed: bool,
}
struct AddCommonFactor {
    x: Var,
    c1: Var,
    c2: Var,
}
struct AddCommonFactorGeneral {
    x: Var,
    y: Var,
    z: Var,
}
struct SubCommonFactor {
    x: Var,
    c1: Var,
    c2: Var,
}
struct SubCommonFactorGeneral {
    x: Var,
    y: Var,
    z: Var,
}
struct CancelMulDiv {
    x: Var,
    c: Var,
}
struct RemMulConstZero {
    c: Var,
}
struct MulMergeConst {
    base: Var,
    c1: Var,
    c2: Var,
}
struct NegMulConst {
    c: Var,
    x: Var,
}

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

impl Applier<SpirvLang, ()> for FoldSub {
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
        let diff = ConstValue::new(a.get().wrapping_sub(b.get()));
        let id = egraph.add(SpirvLang::Const(diff));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for FoldNeg {
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
        let negated = ConstValue::new(a.get().wrapping_neg());
        let id = egraph.add(SpirvLang::Const(negated));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for FoldDiv {
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
        if b.get() == 0 {
            return Vec::new();
        }
        let quotient = if self.signed {
            let lhs = a.get() as i32;
            let rhs = b.get() as i32;
            ConstValue::new(lhs.wrapping_div(rhs) as u32)
        } else {
            ConstValue::new(a.get().wrapping_div(b.get()))
        };
        let id = egraph.add(SpirvLang::Const(quotient));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for FoldRem {
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
        if b.get() == 0 {
            return Vec::new();
        }
        let rem = if self.signed {
            let lhs = a.get() as i32;
            let rhs = b.get() as i32;
            ConstValue::new(lhs.wrapping_rem(rhs) as u32)
        } else {
            ConstValue::new(a.get().wrapping_rem(b.get()))
        };
        let id = egraph.add(SpirvLang::Const(rem));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for DivOne {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        if const_value(egraph, subst[self.b]).is_some_and(|c| c.get() == 1) {
            egraph.union(eclass, subst[self.a]);
            return vec![subst[self.a]];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for RemOne {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        if const_value(egraph, subst[self.b]).is_some_and(|c| c.get() == 1) {
            let id = egraph.add(SpirvLang::Const(ConstValue::new(0)));
            egraph.union(eclass, id);
            return vec![id];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for SubSelf {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        _subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let const_zero = egraph.add(SpirvLang::Const(ConstValue::new(0)));
        egraph.union(eclass, const_zero);
        vec![const_zero]
    }
}

impl Applier<SpirvLang, ()> for AddNegZero {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        _subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let const_zero = egraph.add(SpirvLang::Const(ConstValue::new(0)));
        egraph.union(eclass, const_zero);
        vec![const_zero]
    }
}

impl Applier<SpirvLang, ()> for AddZero {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        if const_value(egraph, subst[self.a]).is_some_and(|c| c.get() == 0) {
            egraph.union(eclass, subst[self.b]);
            return vec![subst[self.b]];
        }
        if const_value(egraph, subst[self.b]).is_some_and(|c| c.get() == 0) {
            egraph.union(eclass, subst[self.a]);
            return vec![subst[self.a]];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for MulOne {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        if const_value(egraph, subst[self.a]).is_some_and(|c| c.get() == 1) {
            egraph.union(eclass, subst[self.b]);
            return vec![subst[self.b]];
        }
        if const_value(egraph, subst[self.b]).is_some_and(|c| c.get() == 1) {
            egraph.union(eclass, subst[self.a]);
            return vec![subst[self.a]];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for MulZero {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let zero_left = const_value(egraph, subst[self.a]).is_some_and(|c| c.get() == 0);
        let zero_right = const_value(egraph, subst[self.b]).is_some_and(|c| c.get() == 0);
        if zero_left || zero_right {
            let id = egraph.add(SpirvLang::Const(ConstValue::new(0)));
            egraph.union(eclass, id);
            return vec![id];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for SubZeroLeft {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        if const_value(egraph, subst[var("a")]).is_some_and(|c| c.get() == 0) {
            let neg = egraph.add(SpirvLang::Neg(subst[var("b")]));
            egraph.union(eclass, neg);
            return vec![neg];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for MulNegOne {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let left_is_neg_one =
            const_value(egraph, subst[self.a]).is_some_and(|c| c.get() == u32::MAX);
        let right_is_neg_one =
            const_value(egraph, subst[self.b]).is_some_and(|c| c.get() == u32::MAX);
        if left_is_neg_one {
            let neg = egraph.add(SpirvLang::Neg(subst[self.b]));
            egraph.union(eclass, neg);
            return vec![neg];
        }
        if right_is_neg_one {
            let neg = egraph.add(SpirvLang::Neg(subst[self.a]));
            egraph.union(eclass, neg);
            return vec![neg];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for AddMergeConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(lhs) = const_value(egraph, subst[self.c1]) else {
            return Vec::new();
        };
        let Some(rhs) = const_value(egraph, subst[self.c2]) else {
            return Vec::new();
        };
        let merged = ConstValue::new(lhs.get().wrapping_add(rhs.get()));
        let const_id = egraph.add(SpirvLang::Const(merged));
        let add = egraph.add(SpirvLang::Add([subst[self.base], const_id]));
        egraph.union(eclass, add);
        vec![add]
    }
}

impl Applier<SpirvLang, ()> for AddSubMerge {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(sub_const) = const_value(egraph, subst[self.sub_const]) else {
            return Vec::new();
        };
        let Some(add_const) = const_value(egraph, subst[self.add_const]) else {
            return Vec::new();
        };
        let merged = ConstValue::new(add_const.get().wrapping_sub(sub_const.get()));
        let const_id = egraph.add(SpirvLang::Const(merged));
        let add = egraph.add(SpirvLang::Add([subst[self.base], const_id]));
        egraph.union(eclass, add);
        vec![add]
    }
}

impl Applier<SpirvLang, ()> for AddSubConstLhs {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(lhs_const) = const_value(egraph, subst[self.base_const]) else {
            return Vec::new();
        };
        let Some(rhs_const) = const_value(egraph, subst[self.add_const]) else {
            return Vec::new();
        };
        let merged = ConstValue::new(lhs_const.get().wrapping_add(rhs_const.get()));
        let const_id = egraph.add(SpirvLang::Const(merged));
        let sub = egraph.add(SpirvLang::Sub([const_id, subst[self.rhs]]));
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for DivMergeConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(c1) = const_value(egraph, subst[self.c1]) else {
            return Vec::new();
        };
        let Some(c2) = const_value(egraph, subst[self.c2]) else {
            return Vec::new();
        };
        if c1.get() == 0 || c2.get() == 0 {
            return Vec::new();
        }
        let merged = if self.signed {
            let lhs = c1.get() as i32;
            let rhs = c2.get() as i32;
            ConstValue::new(lhs.wrapping_mul(rhs) as u32)
        } else {
            ConstValue::new(c1.get().wrapping_mul(c2.get()))
        };
        let const_id = egraph.add(SpirvLang::Const(merged));
        let div = if self.signed {
            egraph.add(SpirvLang::SDiv([subst[self.base], const_id]))
        } else {
            egraph.add(SpirvLang::UDiv([subst[self.base], const_id]))
        };
        egraph.union(eclass, div);
        vec![div]
    }
}

impl Applier<SpirvLang, ()> for AddCommonFactor {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(c1) = const_value(egraph, subst[self.c1]) else {
            return Vec::new();
        };
        let Some(c2) = const_value(egraph, subst[self.c2]) else {
            return Vec::new();
        };
        let merged = ConstValue::new(c1.get().wrapping_add(c2.get()));
        let const_id = egraph.add(SpirvLang::Const(merged));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], const_id]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for AddCommonFactorGeneral {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let add = egraph.add(SpirvLang::Add([subst[self.y], subst[self.z]]));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], add]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for SubCommonFactor {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(c1) = const_value(egraph, subst[self.c1]) else {
            return Vec::new();
        };
        let Some(c2) = const_value(egraph, subst[self.c2]) else {
            return Vec::new();
        };
        let merged = ConstValue::new(c1.get().wrapping_sub(c2.get()));
        let const_id = egraph.add(SpirvLang::Const(merged));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], const_id]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for SubCommonFactorGeneral {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let sub = egraph.add(SpirvLang::Sub([subst[self.y], subst[self.z]]));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], sub]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for CancelMulDiv {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(constant) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        if constant.get() == 0 {
            return Vec::new();
        }
        egraph.union(eclass, subst[self.x]);
        vec![subst[self.x]]
    }
}

impl Applier<SpirvLang, ()> for RemMulConstZero {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(constant) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        if constant.get() == 0 {
            return Vec::new();
        }
        let zero = egraph.add(SpirvLang::Const(ConstValue::new(0)));
        egraph.union(eclass, zero);
        vec![zero]
    }
}

impl Applier<SpirvLang, ()> for NegMulConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        // Only fold when we have a constant multiplier to flip the sign.
        let Some(constant) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let negated = ConstValue::new(constant.get().wrapping_neg());
        let const_id = egraph.add(SpirvLang::Const(negated));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], const_id]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for MulMergeConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(c1) = const_value(egraph, subst[self.c1]) else {
            return Vec::new();
        };
        let Some(c2) = const_value(egraph, subst[self.c2]) else {
            return Vec::new();
        };
        let merged = ConstValue::new(c1.get().wrapping_mul(c2.get()));
        let const_id = egraph.add(SpirvLang::Const(merged));
        let mul = egraph.add(SpirvLang::Mul([subst[self.base], const_id]));
        egraph.union(eclass, mul);
        vec![mul]
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

fn is_const_zero(var: Var) -> impl Fn(&mut EGraph<SpirvLang, ()>, Id, &Subst) -> bool + 'static {
    move |egraph, _, subst| const_value(egraph, subst[var]).is_some_and(|c| c.get() == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::optimize_arith_block;
    use arbitrary::Unstructured;
    use pretty_assertions::assert_eq;
    use rspirv::dr::{Builder, Instruction};

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
    fn simplifies_add_zero_and_mul_identity() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(0)),       // 1
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 2 => x + 0
            SpirvLang::Const(ConstValue::new(1)),       // 3
            SpirvLang::Mul([Id::from(2), Id::from(3)]), // 4 => (x + 0) * 1
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
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

    #[test]
    fn random_expr_round_trips_with_reopt() {
        let mut u = Unstructured::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let expr = fuzzing::arbitrary_expr(&mut u).expect("expression generation");
        let optimized = optimize_expr(&expr);
        let second = optimize_expr(&optimized);
        assert_eq!(optimized, second);
    }

    #[test]
    fn folds_division_and_remainder() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(9)),        // 0
            SpirvLang::Const(ConstValue::new(4)),        // 1
            SpirvLang::SDiv([Id::from(0), Id::from(1)]), // 2 => 9 / 4 => 2
            SpirvLang::SRem([Id::from(0), Id::from(1)]), // 3 => 9 % 4 => 1
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized.as_ref().len(), 1);
        match &optimized.as_ref()[0] {
            SpirvLang::Const(_) => {}
            other => panic!("expected folded constant, found {other:?}"),
        }
    }

    #[test]
    fn folds_double_negation() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(7)), // 0
            SpirvLang::Neg(Id::from(0)),          // 1 => -7
            SpirvLang::Neg(Id::from(1)),          // 2 => --7
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(7))])
        );
    }

    #[test]
    fn folds_subtract_self_to_zero() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Sub([Id::from(0), Id::from(0)]), // 1 => x - x
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(0))])
        );
    }

    #[test]
    fn folds_subtract_from_zero_into_negation() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(0)), // 0
            SpirvLang::Symbol(Symbol::from("y")), // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![
                SpirvLang::Symbol(Symbol::from("y")),
                SpirvLang::Neg(Id::from(0))
            ])
        );
    }

    #[test]
    fn folds_add_negation_to_zero() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("y")), // 0
            SpirvLang::Neg(Id::from(0)),          // 1 => -y
            SpirvLang::Add([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(0))])
        );
    }

    #[test]
    fn rewrites_add_negated_rhs_to_sub() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Neg(Id::from(1)),                // 2 = -y
            SpirvLang::Add([Id::from(0), Id::from(2)]), // 3 = x + (-y)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Sub([a, b])) = nodes.last() else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        let lhs = &nodes[usize::from(*a)];
        let rhs = &nodes[usize::from(*b)];
        assert!(
            matches!(lhs, SpirvLang::Symbol(_)) && matches!(rhs, SpirvLang::Symbol(_)),
            "subtraction should reference the two symbols: {lhs:?} {rhs:?}"
        );
    }

    #[test]
    fn rewrites_mul_by_neg_one_into_negate() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(u32::MAX)), // 0 = -1
            SpirvLang::Symbol(Symbol::from("x")),        // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]),  // 2 = -1 * x
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![
                SpirvLang::Symbol(Symbol::from("x")),
                SpirvLang::Neg(Id::from(0))
            ])
        );
    }

    #[test]
    fn rewrites_mul_double_negation() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Neg(Id::from(0)),                // 2 = -x
            SpirvLang::Neg(Id::from(1)),                // 3 = -y
            SpirvLang::Mul([Id::from(2), Id::from(3)]), // 4 = (-x) * (-y)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([a, b])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let lhs = &nodes[usize::from(*a)];
        let rhs = &nodes[usize::from(*b)];
        assert!(
            matches!(lhs, SpirvLang::Symbol(_)) && matches!(rhs, SpirvLang::Symbol(_)),
            "multiplication should reference the two symbols: {lhs:?} {rhs:?}"
        );
    }

    #[test]
    fn rewrites_add_sub_cancel() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")),       // 0
            SpirvLang::Symbol(Symbol::from("b")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = a - b
            SpirvLang::Add([Id::from(2), Id::from(1)]), // 3 = (a - b) + b
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Symbol(sym)) = nodes.last() else {
            panic!("expected symbol root, got {:?}", nodes.last());
        };
        assert_eq!(sym, &Symbol::from("a"));
    }

    #[test]
    fn rewrites_sub_add_cancel() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")),       // 0
            SpirvLang::Symbol(Symbol::from("b")),       // 1
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 2 = a + b
            SpirvLang::Sub([Id::from(2), Id::from(1)]), // 3 = (a + b) - b
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Symbol(sym)) = nodes.last() else {
            panic!("expected symbol root, got {:?}", nodes.last());
        };
        assert_eq!(sym, &Symbol::from("a"));
    }

    #[test]
    fn rewrites_neg_sub_swap() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")),       // 0
            SpirvLang::Symbol(Symbol::from("b")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = a - b
            SpirvLang::Neg(Id::from(2)),                // 3 = -(a - b)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Sub([lhs, rhs])) = nodes.last() else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        let lhs_node = &nodes[usize::from(*lhs)];
        let rhs_node = &nodes[usize::from(*rhs)];
        assert!(
            matches!(lhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("b"))
                && matches!(rhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("a")),
            "expected b - a but got lhs={lhs_node:?} rhs={rhs_node:?}"
        );
    }

    #[test]
    fn merges_nested_add_constants_into_single_offset() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(2)),       // 1
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 2 = x + 2
            SpirvLang::Const(ConstValue::new(3)),       // 3
            SpirvLang::Add([Id::from(2), Id::from(3)]), // 4 = (x + 2) + 3
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Add([lhs, rhs])) = nodes.last() else {
            panic!("expected add root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands for merged add: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 5, "constants should merge to 5");
    }

    #[test]
    fn merges_nested_mul_constants_into_single_factor() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(2)),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = x * 2
            SpirvLang::Const(ConstValue::new(3)),       // 3
            SpirvLang::Mul([Id::from(2), Id::from(3)]), // 4 = (x * 2) * 3
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands for merged mul: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 6, "factors should merge to 6");
    }

    #[test]
    fn merges_mul_constants_with_commutativity() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(5)),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = 5 * y
            SpirvLang::Const(ConstValue::new(4)),       // 3
            SpirvLang::Mul([Id::from(3), Id::from(2)]), // 4 = 4 * (5 * y)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands for merged mul commutative: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("y"));
        assert_eq!(constant.get(), 20, "factors should merge to 20");
    }

    #[test]
    fn factors_common_multiplier_from_addends() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(2)),       // 1
            SpirvLang::Const(ConstValue::new(3)),       // 2
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 3 = x * 2
            SpirvLang::Mul([Id::from(0), Id::from(2)]), // 4 = x * 3
            SpirvLang::Add([Id::from(3), Id::from(4)]), // 5 = x*2 + x*3
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands after factoring: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 5, "factor should sum constants to 5");
    }

    #[test]
    fn factoring_handles_commuted_multiplicands() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(4)),       // 0
            SpirvLang::Const(ConstValue::new(6)),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Mul([Id::from(0), Id::from(2)]), // 3 = 4 * z
            SpirvLang::Mul([Id::from(2), Id::from(1)]), // 4 = z * 6
            SpirvLang::Add([Id::from(3), Id::from(4)]), // 5 = 4*z + z*6
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands after commuted factoring: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("z"));
        assert_eq!(constant.get(), 10, "factor should sum constants to 10");
    }

    #[test]
    fn factors_common_multiplier_from_subtraction() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(9)),       // 1
            SpirvLang::Const(ConstValue::new(4)),       // 2
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 3 = x * 9
            SpirvLang::Mul([Id::from(0), Id::from(2)]), // 4 = x * 4
            SpirvLang::Sub([Id::from(3), Id::from(4)]), // 5 = x*9 - x*4
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root after factoring, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands after factoring sub: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 5, "factor should compute 9 - 4 = 5");
    }

    #[test]
    fn factors_symbolic_multiplier_from_addition() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 3 = x * y
            SpirvLang::Mul([Id::from(0), Id::from(2)]), // 4 = x * z
            SpirvLang::Add([Id::from(3), Id::from(4)]), // 5 = x*y + x*z
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root after factoring, got {:?}", nodes.last());
        };
        let lhs_node = &nodes[usize::from(*lhs)];
        let rhs_node = &nodes[usize::from(*rhs)];
        assert!(
            matches!(lhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")),
            "expected common factor x, got {lhs_node:?}"
        );
        assert!(
            matches!(rhs_node, SpirvLang::Add([a, b])
                if matches!(&nodes[usize::from(*a)], SpirvLang::Symbol(sym) if *sym == Symbol::from("y"))
                && matches!(&nodes[usize::from(*b)], SpirvLang::Symbol(sym) if *sym == Symbol::from("z"))),
            "expected inner add y + z, got {rhs_node:?}"
        );
    }

    #[test]
    fn factors_symbolic_multiplier_from_subtraction() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 3 = x * y
            SpirvLang::Mul([Id::from(0), Id::from(2)]), // 4 = x * z
            SpirvLang::Sub([Id::from(3), Id::from(4)]), // 5 = x*y - x*z
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root after factoring, got {:?}", nodes.last());
        };
        let lhs_node = &nodes[usize::from(*lhs)];
        let rhs_node = &nodes[usize::from(*rhs)];
        assert!(
            matches!(lhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")),
            "expected common factor x, got {lhs_node:?}"
        );
        assert!(
            matches!(rhs_node, SpirvLang::Sub([a, b])
                if matches!(&nodes[usize::from(*a)], SpirvLang::Symbol(sym) if *sym == Symbol::from("y"))
                && matches!(&nodes[usize::from(*b)], SpirvLang::Symbol(sym) if *sym == Symbol::from("z"))),
            "expected inner sub y - z, got {rhs_node:?}"
        );
    }

    #[test]
    fn merges_nested_divisors_for_signed_division() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(2)),        // 1
            SpirvLang::SDiv([Id::from(0), Id::from(1)]), // 2 = x / 2
            SpirvLang::Const(ConstValue::new(3)),        // 3
            SpirvLang::SDiv([Id::from(2), Id::from(3)]), // 4 = (x / 2) / 3
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::SDiv([lhs, rhs])) = nodes.last() else {
            panic!("expected sdiv root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands for merged sdiv: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 6, "divisors should merge to 6");
    }

    #[test]
    fn cancels_common_factor_in_division() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(5)),        // 0
            SpirvLang::Symbol(Symbol::from("x")),        // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]),  // 2 = 5 * x
            SpirvLang::SDiv([Id::from(2), Id::from(0)]), // 3 = (5 * x) / 5
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(root) = nodes.last() else {
            panic!("expected root");
        };
        assert!(
            matches!(root, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")),
            "expected division to cancel to symbol, got {root:?}"
        );
    }

    #[test]
    fn cancels_common_factor_in_unsigned_division() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("y")),        // 0
            SpirvLang::Const(ConstValue::new(3)),        // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]),  // 2 = y * 3
            SpirvLang::UDiv([Id::from(2), Id::from(1)]), // 3 = (y * 3) / 3
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(root) = nodes.last() else {
            panic!("expected root");
        };
        assert!(
            matches!(root, SpirvLang::Symbol(sym) if *sym == Symbol::from("y")),
            "expected unsigned division to cancel to symbol, got {root:?}"
        );
    }

    #[test]
    fn remainder_of_multiple_of_divisor_is_zero() {
        // We cannot guarantee divisibility for arbitrary symbolic values, so
        // we only fold when both the multiplicative factor and divisor are
        // constant, reducing `(c1 * x) % c1` to zero.
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(8)),        // 0
            SpirvLang::Symbol(Symbol::from("z")),        // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]),  // 2 = 8 * z
            SpirvLang::SRem([Id::from(2), Id::from(0)]), // 3 = (8 * z) % 8
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(0))]),
            "expected remainder of multiple to fold to zero"
        );
    }

    #[test]
    fn unsigned_mod_of_multiple_of_divisor_is_zero() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(7)),        // 0
            SpirvLang::Symbol(Symbol::from("w")),        // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]),  // 2 = 7 * w
            SpirvLang::UMod([Id::from(2), Id::from(0)]), // 3 = (7 * w) % 7
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(0))]),
            "expected unsigned mod of multiple to fold to zero"
        );
    }

    #[test]
    fn negating_mul_with_const_flips_constant() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(5)),       // 0
            SpirvLang::Symbol(Symbol::from("v")),       // 1
            SpirvLang::Mul([Id::from(1), Id::from(0)]), // 2 = v * 5
            SpirvLang::Neg(Id::from(2)),                // 3 = -(v * 5)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands after neg mul const: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("v"));
        assert_eq!(constant.get(), 0u32.wrapping_sub(5));
    }

    #[test]
    fn merges_nested_divisors_for_unsigned_division() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(4)),        // 1
            SpirvLang::UDiv([Id::from(0), Id::from(1)]), // 2 = x / 4
            SpirvLang::Const(ConstValue::new(2)),        // 3
            SpirvLang::UDiv([Id::from(2), Id::from(3)]), // 4 = (x / 4) / 2
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::UDiv([lhs, rhs])) = nodes.last() else {
            panic!("expected udiv root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands for merged udiv: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 8, "divisors should merge to 8");
    }

    #[test]
    fn division_merge_skips_zero_divisors() {
        // Ensure we do not create new divide-by-zero cases.
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(0)),        // 1
            SpirvLang::UDiv([Id::from(0), Id::from(1)]), // 2 = x / 0
            SpirvLang::Const(ConstValue::new(2)),        // 3
            SpirvLang::UDiv([Id::from(2), Id::from(3)]), // 4 = (x / 0) / 2
        ]);
        let optimized = optimize_expr(&expr);
        let udiv_count = optimized
            .as_ref()
            .iter()
            .filter(|node| matches!(node, SpirvLang::UDiv(_)))
            .count();
        assert!(
            udiv_count >= 2,
            "should not collapse nested divisions when zeros are present: {optimized:?}"
        );
    }

    #[test]
    fn merges_add_and_sub_constant_offsets() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(2)),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = x - 2
            SpirvLang::Const(ConstValue::new(5)),       // 3
            SpirvLang::Add([Id::from(2), Id::from(3)]), // 4 = (x - 2) + 5
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Add([lhs, rhs])) = nodes.last() else {
            panic!("expected add root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands for merged add/sub: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 3, "offset should reduce to +3");
    }

    #[test]
    fn folds_const_minus_symbol_plus_const_into_single_sub() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(7)),       // 0
            SpirvLang::Symbol(Symbol::from("x")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = 7 - x
            SpirvLang::Const(ConstValue::new(4)),       // 3
            SpirvLang::Add([Id::from(2), Id::from(3)]), // 4 = (7 - x) + 4
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Sub([lhs, rhs])) = nodes.last() else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        let lhs_node = &nodes[usize::from(*lhs)];
        let rhs_node = &nodes[usize::from(*rhs)];
        assert!(
            matches!(lhs_node, SpirvLang::Const(val) if val.get() == 11),
            "expected merged constant 11 on lhs, got {lhs_node:?}"
        );
        assert!(
            matches!(rhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")),
            "expected symbol on rhs, got {rhs_node:?}"
        );
    }

    #[test]
    fn translate_and_optimize_spirv_block() {
        // Build a trivial SPIR-V function body: %c2 = 2, %c0 = 0, %sum = OpIAdd %c2 %c0
        let mut b = Builder::new();
        b.capability(rspirv::spirv::Capability::Shader);
        let int = b.type_int(32, 0);
        let void = b.type_void();
        let func_ty = b.type_function(void, vec![]);
        let _func = b
            .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c0 = b.constant_bit32(int, 0);
        let _sum = b.i_add(int, None, c2, c0);
        b.ret().unwrap();
        b.end_function().unwrap();
        let module = b.module();
        let block = &module.functions[0].blocks[0];
        let insts: Vec<_> = module
            .types_global_values
            .iter()
            .chain(block.instructions.iter())
            .filter(|inst| {
                matches!(
                    inst.class.opcode,
                    rspirv::spirv::Op::Constant | rspirv::spirv::Op::IAdd | rspirv::spirv::Op::IMul
                )
            })
            .cloned()
            .collect();
        let translated = crate::translate::translate_arith(&insts).unwrap();
        // The root should correspond to the add, which simplifies to const 2.
        let optimized = optimize_translated(&translated);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(2))])
        );
        // Ensure the translation kept the root at the last instruction.
        assert_eq!(
            translated.root,
            Id::from(translated.expr.as_ref().len() - 1)
        );
    }

    #[test]
    fn optimize_arith_block_collapses_to_constant() {
        let mut b = Builder::new();
        let int = b.type_int(32, 0);
        let c2 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(2)],
        );
        let c3 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(3)],
        );
        let add = Instruction::new(
            rspirv::spirv::Op::IAdd,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let block = vec![c2.clone(), c3.clone(), add];
        let optimized = optimize_arith_block(&block).expect("folds block");
        assert_eq!(optimized.len(), 1);
        let folded = &optimized[0];
        assert_eq!(folded.class.opcode, rspirv::spirv::Op::Constant);
        assert_eq!(folded.result_id, Some(3));
        assert_eq!(folded.result_type, Some(int));
        assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(5)]);
    }

    #[test]
    fn optimize_arith_block_folds_sub_self_to_zero() {
        let mut b = Builder::new();
        let int = b.type_int(32, 0);
        let c7 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(7)],
        );
        let sub = Instruction::new(
            rspirv::spirv::Op::ISub,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(1)],
        );
        let block = vec![c7, sub];
        let optimized = optimize_arith_block(&block).expect("optimized");
        assert_eq!(optimized.len(), 1);
        let inst = &optimized[0];
        assert_eq!(inst.class.opcode, rspirv::spirv::Op::Constant);
        assert_eq!(inst.result_id, Some(3));
        assert_eq!(inst.operands, vec![rspirv::dr::Operand::LiteralBit32(0)]);
    }

    #[test]
    fn optimize_arith_block_turns_zero_minus_value_into_negate() {
        let int = 1;
        let c0 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(0)],
        );
        let c9 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(9)],
        );
        let sub = Instruction::new(
            rspirv::spirv::Op::ISub,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let block = vec![c0, c9, sub];
        let optimized = optimize_arith_block(&block).expect("optimization should succeed");
        let mut consts = std::collections::HashMap::new();
        let mut saw_negate = false;
        let mut negate_operand = None;
        for inst in &optimized {
            if inst.class.opcode == rspirv::spirv::Op::Constant {
                if let (Some(id), Some(rspirv::dr::Operand::LiteralBit32(val))) =
                    (inst.result_id, inst.operands.first())
                {
                    consts.insert(id, *val);
                }
            }
            if inst.class.opcode == rspirv::spirv::Op::SNegate {
                saw_negate = true;
                assert_eq!(inst.result_id, Some(3));
                if let Some(rspirv::dr::Operand::IdRef(id)) = inst.operands.first() {
                    negate_operand = Some(*id);
                }
            }
        }
        if let Some(val) = consts.get(&3) {
            assert_eq!(
                *val,
                0u32.wrapping_sub(9),
                "expected folded constant -9 with sub result id"
            );
            return;
        }

        let operand_id = negate_operand.expect("negate should have an operand");
        let const_val = consts.get(&operand_id).copied();
        assert_eq!(const_val, Some(9), "negate should target literal 9");
        assert!(saw_negate, "zero minus value should become negate");
    }

    #[test]
    fn optimize_arith_block_folds_mul_by_one() {
        let int = 1;
        let c4 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(4)],
        );
        let c1 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(1)],
        );
        let mul = Instruction::new(
            rspirv::spirv::Op::IMul,
            Some(int),
            Some(5),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let block = vec![c4, c1, mul];
        let optimized = optimize_arith_block(&block).expect("optimization should succeed");
        assert_eq!(optimized.len(), 1);
        let folded = &optimized[0];
        assert_eq!(folded.class.opcode, rspirv::spirv::Op::Constant);
        assert_eq!(folded.result_id, Some(5));
        assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(4)]);
    }

    #[test]
    fn optimize_arith_block_folds_mul_with_zero() {
        let int = 1;
        let c5 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(5)],
        );
        let c0 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(0)],
        );
        let mul = Instruction::new(
            rspirv::spirv::Op::IMul,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let block = vec![c5, c0, mul];
        let optimized = optimize_arith_block(&block).expect("optimization should succeed");
        assert_eq!(optimized.len(), 1);
        let folded = &optimized[0];
        assert_eq!(folded.class.opcode, rspirv::spirv::Op::Constant);
        assert_eq!(folded.result_id, Some(3));
        assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(0)]);
    }

    #[test]
    fn optimize_arith_block_rejects_unsupported_op() {
        let insts = vec![Instruction::new(
            rspirv::spirv::Op::TypeVoid,
            None,
            None,
            vec![],
        )];
        let err = optimize_arith_block(&insts).unwrap_err();
        assert!(
            matches!(
                err,
                crate::translate::TranslateError::UnsupportedOp(_)
                    | crate::translate::TranslateError::MissingResultId(_)
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn div_by_zero_does_not_fold() {
        let int = 1;
        let c2 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(2)],
        );
        let c0 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(0)],
        );
        let div = Instruction::new(
            rspirv::spirv::Op::SDiv,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let block = vec![c2.clone(), c0.clone(), div.clone()];
        let optimized = optimize_arith_block(&block).expect("optimization should succeed");
        assert_eq!(optimized, block);
    }

    #[test]
    fn snegate_translates_and_folds() {
        let int = 1;
        let c5 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(5)],
        );
        let neg = Instruction::new(
            rspirv::spirv::Op::SNegate,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::IdRef(1)],
        );
        let block = vec![c5.clone(), neg];
        let optimized = optimize_arith_block(&block).expect("folds negation");
        assert_eq!(optimized.len(), 1);
        assert_eq!(
            optimized[0],
            Instruction::new(
                rspirv::spirv::Op::Constant,
                Some(int),
                Some(2),
                vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(5))]
            )
        );
    }
}
