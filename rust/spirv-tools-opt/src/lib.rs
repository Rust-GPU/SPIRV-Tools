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
                match u.choose(&[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9])? {
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
                    5 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        if *u.choose(&[true, false])? {
                            SpirvLang::SRem([Id::from(a), Id::from(b)])
                        } else {
                            SpirvLang::UMod([Id::from(a), Id::from(b)])
                        }
                    }
                    6 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::Shl([Id::from(a), Id::from(b)])
                    }
                    7 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        if *u.choose(&[true, false])? {
                            SpirvLang::ShrS([Id::from(a), Id::from(b)])
                        } else {
                            SpirvLang::ShrU([Id::from(a), Id::from(b)])
                        }
                    }
                    8 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::BitAnd([Id::from(a), Id::from(b)])
                    }
                    _ => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::Sub([Id::from(a), Id::from(b)])
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
        "shl" = Shl([Id; 2]),
        "shr_s" = ShrS([Id; 2]),
        "shr_u" = ShrU([Id; 2]),
        "band" = BitAnd([Id; 2]),
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
            SpirvLang::Mul(_) | SpirvLang::SDiv(_) | SpirvLang::UDiv(_) => {
                enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 2
            }
            SpirvLang::Shl(_) | SpirvLang::ShrS(_) | SpirvLang::ShrU(_) => {
                enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 1
            }
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
        rewrite!("neg-neg-cancel"; "(neg (neg ?x))" => "?x"),
        rewrite!("mul-neg-left"; "(* (neg ?a) ?b)" => "(neg (* ?a ?b))"),
        rewrite!("mul-neg-right"; "(* ?a (neg ?b))" => "(neg (* ?a ?b))"),
        rewrite!("sub-neg-both"; "(- (neg ?a) (neg ?b))" => "(- ?b ?a)"),
        rewrite!("mul-one"; "(* ?a ?b)" => { MulOne { a: var("?a"), b: var("?b") } }),
        rewrite!("mul-zero"; "(* ?a ?b)" => { MulZero { a: var("?a"), b: var("?b") } }),
        rewrite!("mul-neg-one"; "(* ?a ?b)" => { MulNegOne { a: var("?a"), b: var("?b") } }),
        rewrite!("mul-double-neg"; "(* (neg ?a) (neg ?b))" => "(* ?a ?b)"),
        rewrite!("add-sub-cancel-right-simple"; "(+ (- ?a ?b) ?b)" => "?a"),
        rewrite!("add-sub-cancel-left-simple"; "(+ ?b (- ?a ?b))" => "?a"),
        rewrite!("sub-add-cancel-right-simple"; "(- (+ ?a ?b) ?b)" => "?a"),
        rewrite!("sub-add-cancel-left-simple"; "(- (+ ?b ?a) ?b)" => "?a"),
        rewrite!("sub-add-cancel-right-symmetric-simple"; "(- (+ ?a ?b) ?a)" => "?b"),
        rewrite!("sub-add-cancel-left-symmetric-simple"; "(- (+ ?b ?a) ?a)" => "?b"),
        rewrite!("add-fold"; "(+ ?a ?b)" => { FoldAdd }),
        rewrite!("mul-fold"; "(* ?a ?b)" => { FoldMul }),
        rewrite!("sub-fold"; "(- ?a ?b)" => { FoldSub }),
        rewrite!("sub-zero-right"; "(- ?a ?b)" => "?a" if is_const_zero(var("?b"))),
        rewrite!("sub-zero-left"; "(- ?a ?b)" => { SubZeroLeft }),
        rewrite!("sub-self"; "(- ?a ?a)" => { SubSelf }),
        rewrite!("sub-neg-right-to-add"; "(- ?a (neg ?b))" => "(+ ?a ?b)"),
        rewrite!("sub-sub-cancel-left"; "(- ?a (- ?a ?b))" => "?b"),
        rewrite!("add-dup-to-mul"; "(+ ?x ?x)" => { AddDuplicateToMul { x: var("?x") } }),
        rewrite!("add-triple-left"; "(+ (+ ?x ?x) ?x)" => {
            AddTripleToMul { x: var("?x") }
        }),
        rewrite!("add-triple-right"; "(+ ?x (+ ?x ?x))" => {
            AddTripleToMul { x: var("?x") }
        }),
        rewrite!("add-quadruple"; "(+ (+ ?x ?x) (+ ?x ?x))" => {
            AddQuadrupleToShift { x: var("?x") }
        }),
        rewrite!("sub-shared-addends"; "(- (+ ?x ?y) (+ ?x ?z))" => "(- ?y ?z)"),
        rewrite!("sub-shared-addends-swap"; "(- (+ ?y ?x) (+ ?x ?z))" => "(- ?y ?z)"),
        rewrite!("sub-shared-const-addends"; "(- (+ ?x ?c1) (+ ?y ?c2))" => {
            SubSharedConstAddend { x: var("?x"), y: var("?y"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("sub-shared-addend-general"; "(- (+ ?x ?s1) (+ ?y ?s2))" => {
            SubSharedAddendEq { x: var("?x"), y: var("?y"), s1: var("?s1"), s2: var("?s2") }
        }),
        rewrite!("sub-shared-addend-general-swap"; "(- (+ ?s1 ?x) (+ ?s2 ?y))" => {
            SubSharedAddendEq { x: var("?x"), y: var("?y"), s1: var("?s1"), s2: var("?s2") }
        }),
        rewrite!("add-sub-shared-term"; "(+ (- ?x ?y) (- ?y ?z))" => "(- ?x ?z)"),
        rewrite!("add-sub-shared-term-comm"; "(+ (- ?y ?x) (- ?x ?z))" => "(- ?y ?z)"),
        rewrite!("add-sub-mirror-cancel"; "(+ ?x (- ?y ?x))" => "?y"),
        rewrite!("add-sub-mirror-cancel-swap"; "(+ (- ?y ?x) ?x)" => "?y"),
        rewrite!("add-cancels-subtrahend"; "(+ (- ?x ?y) ?y)" => "?x"),
        rewrite!("add-cancels-subtrahend-comm"; "(+ ?y (- ?x ?y))" => "?x"),
        rewrite!("merge-shl-const"; "(shl (shl ?x ?a) ?b)" => {
            MergeShift { x: var("?x"), a: var("?a"), b: var("?b"), kind: ShiftKind::Left }
        }),
        rewrite!("merge-shru-const"; "(shr_u (shr_u ?x ?a) ?b)" => {
            MergeShift { x: var("?x"), a: var("?a"), b: var("?b"), kind: ShiftKind::RightUnsigned }
        }),
        rewrite!("merge-shrs-const"; "(shr_s (shr_s ?x ?a) ?b)" => {
            MergeShift { x: var("?x"), a: var("?a"), b: var("?b"), kind: ShiftKind::RightSigned }
        }),
        rewrite!("shl-zero"; "(shl ?x ?c)" => "?x" if is_const_zero(var("?c"))),
        rewrite!("shr-u-zero"; "(shr_u ?x ?c)" => "?x" if is_const_zero(var("?c"))),
        rewrite!("shr-s-zero"; "(shr_s ?x ?c)" => "?x" if is_const_zero(var("?c"))),
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
        rewrite!("sdiv-pull-const-left"; "(sdiv (* ?c1 ?x) ?c2)" => {
            DivPullConst { x: var("?x"), c1: var("?c1"), c2: var("?c2"), signed: true }
        }),
        rewrite!("sdiv-pull-const-right"; "(sdiv (* ?x ?c1) ?c2)" => {
            DivPullConst { x: var("?x"), c1: var("?c1"), c2: var("?c2"), signed: true }
        }),
        rewrite!("udiv-pull-const-left"; "(udiv (* ?c1 ?x) ?c2)" => {
            DivPullConst { x: var("?x"), c1: var("?c1"), c2: var("?c2"), signed: false }
        }),
        rewrite!("udiv-pull-const-right"; "(udiv (* ?x ?c1) ?c2)" => {
            DivPullConst { x: var("?x"), c1: var("?c1"), c2: var("?c2"), signed: false }
        }),
        rewrite!("mul-power-of-two-left"; "(* ?c ?x)" => {
            MulPowerOfTwo { x: var("?x"), c: var("?c") }
        }),
        rewrite!("mul-power-of-two-right"; "(* ?x ?c)" => {
            MulPowerOfTwo { x: var("?x"), c: var("?c") }
        }),
        rewrite!("sdiv-power-of-two"; "(sdiv ?x ?c)" => {
            DivPowerOfTwo { x: var("?x"), c: var("?c"), signed: true }
        }),
        rewrite!("udiv-power-of-two"; "(udiv ?x ?c)" => {
            DivPowerOfTwo { x: var("?x"), c: var("?c"), signed: false }
        }),
        rewrite!("srem-const-decompose"; "(srem ?x ?c)" => {
            SRemConstDecompose { x: var("?x"), c: var("?c") }
        }),
        rewrite!("umod-const-decompose"; "(umod ?x ?c)" => {
            UModConstDecompose { x: var("?x"), c: var("?c") }
        }),
        rewrite!("umod-power-of-two"; "(umod ?x ?c)" => {
            UModPowerOfTwo { x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-const-fold"; "(band ?a ?b)" => { BitAndFold { a: var("?a"), b: var("?b") } }),
        rewrite!("band-one-left"; "(band ?x ?c)" => {
            BitAndConstSimplify { x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-one-right"; "(band ?c ?x)" => {
            BitAndConstSimplify { x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-mask-to-umod"; "(band ?x ?c)" => {
            BitAndToUmod { x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-self"; "(band ?x ?x)" => "?x"),
        rewrite!("umod-power-of-two-mask"; "(umod ?x ?c)" => {
            UModPowerOfTwoMask { x: var("?x"), c: var("?c") }
        }),
        rewrite!("mul-dist-const-over-add"; "(* ?c (+ ?x ?k))" => {
            DistConstMulAdd { c: var("?c"), x: var("?x"), k: var("?k") }
        }),
        rewrite!("mul-dist-const-over-add-comm"; "(* (+ ?x ?k) ?c)" => {
            DistConstMulAdd { c: var("?c"), x: var("?x"), k: var("?k") }
        }),
        rewrite!("mul-dist-const-over-add-swap"; "(* ?c (+ ?k ?x))" => {
            DistConstMulAdd { c: var("?c"), x: var("?x"), k: var("?k") }
        }),
        rewrite!("mul-dist-const-over-sub"; "(* ?c (- ?x ?k))" => {
            DistConstMulSub { c: var("?c"), x: var("?x"), k: var("?k"), flipped: false }
        }),
        rewrite!("mul-dist-const-over-sub-comm"; "(* (- ?x ?k) ?c)" => {
            DistConstMulSub { c: var("?c"), x: var("?x"), k: var("?k"), flipped: false }
        }),
        rewrite!("mul-dist-const-over-sub-flipped"; "(* ?c (- ?k ?x))" => {
            DistConstMulSub { c: var("?c"), x: var("?x"), k: var("?k"), flipped: true }
        }),
        rewrite!("mul-dist-const-over-sub-flipped-comm"; "(* (- ?k ?x) ?c)" => {
            DistConstMulSub { c: var("?c"), x: var("?x"), k: var("?k"), flipped: true }
        }),
        rewrite!("add-gcd-affine"; "(+ (* ?c ?x) ?k)" => {
            FoldAffineGcd { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Add }
        }),
        rewrite!("add-gcd-affine-comm"; "(+ (* ?x ?c) ?k)" => {
            FoldAffineGcd { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Add }
        }),
        rewrite!("add-gcd-affine-left"; "(+ ?k (* ?c ?x))" => {
            FoldAffineGcd { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Add }
        }),
        rewrite!("add-gcd-affine-left-comm"; "(+ ?k (* ?x ?c))" => {
            FoldAffineGcd { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Add }
        }),
        rewrite!("sub-gcd-affine"; "(- (* ?c ?x) ?k)" => {
            FoldAffineGcd { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Sub }
        }),
        rewrite!("sub-gcd-affine-comm"; "(- (* ?x ?c) ?k)" => {
            FoldAffineGcd { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Sub }
        }),
        rewrite!("sub-gcd-affine-flipped"; "(- ?k (* ?c ?x))" => {
            FoldAffineGcd { x: var("?x"), c: var("?c"), k: var("?k"), flipped: true, op: AffineOp::Sub }
        }),
        rewrite!("sub-gcd-affine-flipped-comm"; "(- ?k (* ?x ?c))" => {
            FoldAffineGcd { x: var("?x"), c: var("?c"), k: var("?k"), flipped: true, op: AffineOp::Sub }
        }),
        rewrite!("add-fold-affine-const-right"; "(+ (* ?x ?c) ?k)" => {
            FoldAffineConst { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Add }
        }),
        rewrite!("add-fold-affine-const-left"; "(+ ?k (* ?x ?c))" => {
            FoldAffineConst { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Add }
        }),
        rewrite!("add-fold-affine-const-comm-left"; "(+ (* ?c ?x) ?k)" => {
            FoldAffineConst { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Add }
        }),
        rewrite!("add-fold-affine-const-comm-right"; "(+ ?k (* ?c ?x))" => {
            FoldAffineConst { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Add }
        }),
        rewrite!("sub-fold-affine-const-right"; "(- (* ?x ?c) ?k)" => {
            FoldAffineConst { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Sub }
        }),
        rewrite!("sub-fold-affine-const-left"; "(- ?k (* ?x ?c))" => {
            FoldAffineConst { x: var("?x"), c: var("?c"), k: var("?k"), flipped: true, op: AffineOp::Sub }
        }),
        rewrite!("sub-fold-affine-const-comm"; "(- (* ?c ?x) ?k)" => {
            FoldAffineConst { x: var("?x"), c: var("?c"), k: var("?k"), flipped: false, op: AffineOp::Sub }
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
        rewrite!("sub-of-add-merge-consts"; "(- (+ ?x ?c1) ?c2)" => {
            SubAddConstMerge { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("sub-of-add-merge-consts-comm"; "(- (+ ?c1 ?x) ?c2)" => {
            SubAddConstMerge { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("sub-chain-merge-consts"; "(- (- ?x ?c1) ?c2)" => {
            SubChainConstMerge { x: var("?x"), c1: var("?c1"), c2: var("?c2") }
        }),
        rewrite!("sub-chain-const-lhs-merge"; "(- (- ?c1 ?x) ?c2)" => {
            SubConstLhsChain { c1: var("?c1"), c2: var("?c2"), x: var("?x") }
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
struct SubAddConstMerge {
    x: Var,
    c1: Var,
    c2: Var,
}
struct SubChainConstMerge {
    x: Var,
    c1: Var,
    c2: Var,
}
struct SubSharedConstAddend {
    x: Var,
    y: Var,
    c1: Var,
    c2: Var,
}
struct SubSharedAddendEq {
    x: Var,
    y: Var,
    s1: Var,
    s2: Var,
}
struct SubConstLhsChain {
    c1: Var,
    c2: Var,
    x: Var,
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
struct DivPullConst {
    x: Var,
    c1: Var,
    c2: Var,
    signed: bool,
}
enum AffineOp {
    Add,
    Sub,
}
struct DistConstMulAdd {
    c: Var,
    x: Var,
    k: Var,
}
struct DistConstMulSub {
    c: Var,
    x: Var,
    k: Var,
    flipped: bool,
}
struct FoldAffineConst {
    x: Var,
    c: Var,
    k: Var,
    flipped: bool,
    op: AffineOp,
}
struct FoldAffineGcd {
    x: Var,
    c: Var,
    k: Var,
    flipped: bool,
    op: AffineOp,
}
struct AddDuplicateToMul {
    x: Var,
}
struct AddTripleToMul {
    x: Var,
}
struct AddQuadrupleToShift {
    x: Var,
}
enum ShiftKind {
    Left,
    RightUnsigned,
    RightSigned,
}
struct MulPowerOfTwo {
    x: Var,
    c: Var,
}
struct DivPowerOfTwo {
    x: Var,
    c: Var,
    signed: bool,
}
struct SRemConstDecompose {
    x: Var,
    c: Var,
}
struct UModConstDecompose {
    x: Var,
    c: Var,
}
struct BitAndFold {
    a: Var,
    b: Var,
}
struct BitAndConstSimplify {
    x: Var,
    c: Var,
}
struct BitAndToUmod {
    x: Var,
    c: Var,
}
struct UModPowerOfTwo {
    x: Var,
    c: Var,
}
struct UModPowerOfTwoMask {
    x: Var,
    c: Var,
}
struct MergeShift {
    x: Var,
    a: Var,
    b: Var,
    kind: ShiftKind,
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

impl Applier<SpirvLang, ()> for SubAddConstMerge {
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
        let add = egraph.add(SpirvLang::Add([subst[self.x], const_id]));
        egraph.union(eclass, add);
        vec![add]
    }
}

impl Applier<SpirvLang, ()> for SubChainConstMerge {
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
        let sub = egraph.add(SpirvLang::Sub([subst[self.x], const_id]));
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for SubSharedConstAddend {
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
        if c1.get() != c2.get() {
            return Vec::new();
        }
        let sub = egraph.add(SpirvLang::Sub([subst[self.x], subst[self.y]]));
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for SubSharedAddendEq {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        if egraph.find(subst[self.s1]) != egraph.find(subst[self.s2]) {
            return Vec::new();
        }
        let sub = egraph.add(SpirvLang::Sub([subst[self.x], subst[self.y]]));
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for SubConstLhsChain {
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
        let sub = egraph.add(SpirvLang::Sub([const_id, subst[self.x]]));
        egraph.union(eclass, sub);
        vec![sub]
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

impl Applier<SpirvLang, ()> for DivPullConst {
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
        if c2.get() == 0 {
            return Vec::new();
        }
        let ratio = if self.signed {
            let num = c1.get() as i32;
            let den = c2.get() as i32;
            if den == 0 || num % den != 0 {
                return Vec::new();
            }
            ConstValue::new(num.wrapping_div(den) as u32)
        } else {
            if c1.get() % c2.get() != 0 {
                return Vec::new();
            }
            ConstValue::new(c1.get().wrapping_div(c2.get()))
        };
        let const_id = egraph.add(SpirvLang::Const(ratio));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], const_id]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for DistConstMulAdd {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(multiplier) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let Some(add_const) = const_value(egraph, subst[self.k]) else {
            return Vec::new();
        };
        let scaled_const = ConstValue::new(multiplier.get().wrapping_mul(add_const.get()));
        let const_id = egraph.add(SpirvLang::Const(scaled_const));
        let mul = egraph.add(SpirvLang::Mul([subst[self.c], subst[self.x]]));
        let add = egraph.add(SpirvLang::Add([mul, const_id]));
        egraph.union(eclass, add);
        vec![add]
    }
}

impl Applier<SpirvLang, ()> for DistConstMulSub {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(multiplier) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let Some(sub_const) = const_value(egraph, subst[self.k]) else {
            return Vec::new();
        };
        let scaled_const = ConstValue::new(multiplier.get().wrapping_mul(sub_const.get()));
        let const_id = egraph.add(SpirvLang::Const(scaled_const));
        let mul = egraph.add(SpirvLang::Mul([subst[self.c], subst[self.x]]));
        let sub = if self.flipped {
            egraph.add(SpirvLang::Sub([const_id, mul]))
        } else {
            egraph.add(SpirvLang::Sub([mul, const_id]))
        };
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for FoldAffineConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(const_factor) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let Some(const_term) = const_value(egraph, subst[self.k]) else {
            return Vec::new();
        };
        if const_factor.get() == 0 {
            return Vec::new();
        }
        if const_term.get() % const_factor.get() != 0 {
            return Vec::new();
        }
        let scaled = ConstValue::new(const_term.get().wrapping_div(const_factor.get()));
        let const_scaled_id = egraph.add(SpirvLang::Const(scaled));
        let inner = match self.op {
            AffineOp::Add => egraph.add(SpirvLang::Add([subst[self.x], const_scaled_id])),
            AffineOp::Sub => {
                if self.flipped {
                    egraph.add(SpirvLang::Sub([const_scaled_id, subst[self.x]]))
                } else {
                    egraph.add(SpirvLang::Sub([subst[self.x], const_scaled_id]))
                }
            }
        };
        let mul = egraph.add(SpirvLang::Mul([subst[self.c], inner]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for FoldAffineGcd {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(coeff) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let Some(const_term) = const_value(egraph, subst[self.k]) else {
            return Vec::new();
        };
        // Only factor when coefficients are non-negative to avoid sign surprises with wrapping.
        if coeff.get() >> 31 != 0 || const_term.get() >> 31 != 0 {
            return Vec::new();
        }
        if coeff.get() == 0 || const_term.get() == 0 {
            return Vec::new();
        }
        let gcd = gcd_u32(coeff.get(), const_term.get());
        if gcd <= 1 || gcd == coeff.get() {
            return Vec::new();
        }
        let scaled_coeff = ConstValue::new(coeff.get() / gcd);
        let scaled_const = ConstValue::new(const_term.get() / gcd);
        let gcd_id = egraph.add(SpirvLang::Const(ConstValue::new(gcd)));
        let scaled_const_id = egraph.add(SpirvLang::Const(scaled_const));
        let scaled_coeff_id = egraph.add(SpirvLang::Const(scaled_coeff));
        let mul_x = egraph.add(SpirvLang::Mul([scaled_coeff_id, subst[self.x]]));
        let inner = match self.op {
            AffineOp::Add => egraph.add(SpirvLang::Add([mul_x, scaled_const_id])),
            AffineOp::Sub => {
                if self.flipped {
                    egraph.add(SpirvLang::Sub([scaled_const_id, mul_x]))
                } else {
                    egraph.add(SpirvLang::Sub([mul_x, scaled_const_id]))
                }
            }
        };
        let factored = egraph.add(SpirvLang::Mul([gcd_id, inner]));
        egraph.union(eclass, factored);
        vec![factored]
    }
}

impl Applier<SpirvLang, ()> for AddDuplicateToMul {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let two = egraph.add(SpirvLang::Const(ConstValue::new(2)));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], two]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for AddTripleToMul {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let three = egraph.add(SpirvLang::Const(ConstValue::new(3)));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], three]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for AddQuadrupleToShift {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let two = egraph.add(SpirvLang::Const(ConstValue::new(2)));
        let shl = egraph.add(SpirvLang::Shl([subst[self.x], two]));
        egraph.union(eclass, shl);
        vec![shl]
    }
}

impl Applier<SpirvLang, ()> for MulPowerOfTwo {
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
        let Some(shift) = is_power_of_two(constant.get()) else {
            return Vec::new();
        };
        if shift == 0 {
            return Vec::new();
        }
        let shift_const = egraph.add(SpirvLang::Const(ConstValue::new(shift)));
        let shl = egraph.add(SpirvLang::Shl([subst[self.x], shift_const]));
        egraph.union(eclass, shl);
        vec![shl]
    }
}

impl Applier<SpirvLang, ()> for DivPowerOfTwo {
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
        let Some(shift) = is_power_of_two(constant.get()) else {
            return Vec::new();
        };
        if shift == 0 {
            return Vec::new();
        }
        if !has_symbol(egraph, subst[self.x]) {
            return Vec::new();
        }
        if egraph[egraph.find(subst[self.x])]
            .nodes
            .iter()
            .any(|n| matches!(n, SpirvLang::UDiv(_) | SpirvLang::SDiv(_)))
        {
            return Vec::new();
        }
        let shift_const = egraph.add(SpirvLang::Const(ConstValue::new(shift)));
        if self.signed {
            let sign_shift = egraph.add(SpirvLang::Const(ConstValue::new(31)));
            let sign = egraph.add(SpirvLang::ShrS([subst[self.x], sign_shift]));
            let mask_value = (1u32 << shift).wrapping_sub(1);
            let mask = egraph.add(SpirvLang::Const(ConstValue::new(mask_value)));
            let bias = egraph.add(SpirvLang::BitAnd([sign, mask]));
            let biased = egraph.add(SpirvLang::Add([subst[self.x], bias]));
            let shr = egraph.add(SpirvLang::ShrS([biased, shift_const]));
            egraph.union(eclass, shr);
            vec![shr]
        } else {
            let shr = egraph.add(SpirvLang::ShrU([subst[self.x], shift_const]));
            egraph.union(eclass, shr);
            vec![shr]
        }
    }
}

impl Applier<SpirvLang, ()> for UModPowerOfTwo {
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
        let Some(shift) = is_power_of_two(constant.get()) else {
            return Vec::new();
        };
        if shift == 0 {
            return Vec::new();
        }
        let shift_const = egraph.add(SpirvLang::Const(ConstValue::new(shift)));
        let shr = egraph.add(SpirvLang::ShrU([subst[self.x], shift_const]));
        let shl = egraph.add(SpirvLang::Shl([shr, shift_const]));
        let sub = egraph.add(SpirvLang::Sub([subst[self.x], shl]));
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for SRemConstDecompose {
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
        let c_id = egraph.add(SpirvLang::Const(constant));
        let div = egraph.add(SpirvLang::SDiv([subst[self.x], c_id]));
        let mul = egraph.add(SpirvLang::Mul([div, c_id]));
        let sub = egraph.add(SpirvLang::Sub([subst[self.x], mul]));
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for UModConstDecompose {
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
        let c_id = egraph.add(SpirvLang::Const(constant));
        let div = egraph.add(SpirvLang::UDiv([subst[self.x], c_id]));
        let mul = egraph.add(SpirvLang::Mul([div, c_id]));
        let sub = egraph.add(SpirvLang::Sub([subst[self.x], mul]));
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for BitAndFold {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(lhs) = const_value(egraph, subst[self.a]) else {
            return Vec::new();
        };
        let Some(rhs) = const_value(egraph, subst[self.b]) else {
            return Vec::new();
        };
        let folded = lhs.get() & rhs.get();
        let const_id = egraph.add(SpirvLang::Const(ConstValue::new(folded)));
        egraph.union(eclass, const_id);
        vec![const_id]
    }
}

impl Applier<SpirvLang, ()> for BitAndConstSimplify {
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
        if constant.get() == u32::MAX {
            egraph.union(eclass, subst[self.x]);
            return vec![subst[self.x]];
        }
        if constant.get() == 0 {
            let zero = egraph.add(SpirvLang::Const(ConstValue::new(0)));
            egraph.union(eclass, zero);
            return vec![zero];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for BitAndToUmod {
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
        let mask = constant.get();
        let Some(shift) = is_power_of_two(mask.wrapping_add(1)) else {
            return Vec::new();
        };
        if shift == 0 {
            return Vec::new();
        }
        let const_id = egraph.add(SpirvLang::Const(ConstValue::new(mask.wrapping_add(1))));
        let umod = egraph.add(SpirvLang::UMod([subst[self.x], const_id]));
        egraph.union(eclass, umod);
        vec![umod]
    }
}

impl Applier<SpirvLang, ()> for UModPowerOfTwoMask {
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
        let Some(shift) = is_power_of_two(constant.get()) else {
            return Vec::new();
        };
        if shift == 0 {
            return Vec::new();
        }
        let mask_value = (1u32 << shift).wrapping_sub(1);
        let mask = egraph.add(SpirvLang::Const(ConstValue::new(mask_value)));
        let band = egraph.add(SpirvLang::BitAnd([subst[self.x], mask]));
        egraph.union(eclass, band);
        vec![band]
    }
}

impl Applier<SpirvLang, ()> for MergeShift {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(a) = const_value(egraph, subst[self.a]) else {
            return Vec::new();
        };
        let Some(b) = const_value(egraph, subst[self.b]) else {
            return Vec::new();
        };
        let total = ConstValue::new(a.get().wrapping_add(b.get()));
        let const_id = egraph.add(SpirvLang::Const(total));
        let merged = match self.kind {
            ShiftKind::Left => egraph.add(SpirvLang::Shl([subst[self.x], const_id])),
            ShiftKind::RightUnsigned => egraph.add(SpirvLang::ShrU([subst[self.x], const_id])),
            ShiftKind::RightSigned => egraph.add(SpirvLang::ShrS([subst[self.x], const_id])),
        };
        egraph.union(eclass, merged);
        vec![merged]
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

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let tmp = b;
        b = a % b;
        a = tmp;
    }
    a
}

fn is_power_of_two(value: u32) -> Option<u32> {
    if value == 0 || value.count_ones() != 1 {
        return None;
    }
    Some(value.trailing_zeros())
}

fn has_symbol(egraph: &EGraph<SpirvLang, ()>, id: Id) -> bool {
    let class = egraph.find(id);
    egraph[class]
        .nodes
        .iter()
        .any(|n| matches!(n, SpirvLang::Symbol(_)))
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
    fn folds_bitand_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(0b1010)),
            SpirvLang::Const(ConstValue::new(0b1100)),
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(0b1000))])
        );
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
    fn cancels_sub_then_add_same_rhs() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = x - y
            SpirvLang::Add([Id::from(2), Id::from(1)]), // 3 = (x - y) + y
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn merges_add_then_sub_constants_into_single_offset() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(4)),       // 1
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 2 = x + 4
            SpirvLang::Const(ConstValue::new(2)),       // 3
            SpirvLang::Sub([Id::from(2), Id::from(3)]), // 4 = (x + 4) - 2
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let SpirvLang::Add([lhs, rhs]) = nodes.last().expect("optimized root") else {
            panic!("expected add root, got {:?}", nodes.last());
        };
        let lhs_const = matches!(nodes[usize::from(*lhs)], SpirvLang::Const(v) if v.get() == 2);
        let rhs_const = matches!(nodes[usize::from(*rhs)], SpirvLang::Const(v) if v.get() == 2);
        let lhs_sym = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(_));
        let rhs_sym = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(_));
        assert!(
            (lhs_const && rhs_sym) || (rhs_const && lhs_sym),
            "expected x plus folded const 2, got {nodes:?}"
        );
    }

    #[test]
    fn merges_sub_chain_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(3)),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = x - 3
            SpirvLang::Const(ConstValue::new(5)),       // 3
            SpirvLang::Sub([Id::from(2), Id::from(3)]), // 4 = (x - 3) - 5
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let SpirvLang::Sub([lhs, rhs]) = nodes.last().expect("optimized root") else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        assert!(
            matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(_))
                && matches!(nodes[usize::from(*rhs)], SpirvLang::Const(v) if v.get() == 8),
            "expected x - 8, got {nodes:?}"
        );
    }

    #[test]
    fn merges_commuted_add_then_sub_constants_into_single_offset() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(4)),       // 1
            SpirvLang::Add([Id::from(1), Id::from(0)]), // 2 = 4 + x
            SpirvLang::Const(ConstValue::new(2)),       // 3
            SpirvLang::Sub([Id::from(2), Id::from(3)]), // 4 = (4 + x) - 2
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let SpirvLang::Add([lhs, rhs]) = nodes.last().expect("optimized root") else {
            panic!("expected add root, got {:?}", nodes.last());
        };
        let lhs_const = matches!(nodes[usize::from(*lhs)], SpirvLang::Const(v) if v.get() == 2);
        let rhs_const = matches!(nodes[usize::from(*rhs)], SpirvLang::Const(v) if v.get() == 2);
        let lhs_sym = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(_));
        let rhs_sym = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(_));
        assert!(
            (lhs_const && rhs_sym) || (rhs_const && lhs_sym),
            "expected x plus folded const 2, got {nodes:?}"
        );
    }

    #[test]
    fn merges_const_minus_symbol_chain_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(10)),      // 0
            SpirvLang::Symbol(Symbol::from("x")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = 10 - x
            SpirvLang::Const(ConstValue::new(3)),       // 3
            SpirvLang::Sub([Id::from(2), Id::from(3)]), // 4 = (10 - x) - 3
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let SpirvLang::Sub([lhs, rhs]) = nodes.last().expect("optimized root") else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        assert!(
            matches!(nodes[usize::from(*lhs)], SpirvLang::Const(v) if v.get() == 7)
                && matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(_)),
            "expected const 7 - x, got {nodes:?}"
        );
    }

    #[test]
    fn cancels_shared_addends_in_subtraction() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 3 = x + y
            SpirvLang::Add([Id::from(0), Id::from(2)]), // 4 = x + z
            SpirvLang::Sub([Id::from(3), Id::from(4)]), // 5 = (x + y) - (x + z)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let SpirvLang::Sub([lhs, rhs]) = nodes.last().expect("optimized root") else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        assert!(
            matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("y"))
                && matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("z")),
            "expected y - z, got {nodes:?}"
        );
    }

    #[test]
    fn cancels_shared_const_addends_in_subtraction() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Const(ConstValue::new(5)),       // 2
            SpirvLang::Add([Id::from(0), Id::from(2)]), // 3 = x + 5
            SpirvLang::Add([Id::from(1), Id::from(2)]), // 4 = y + 5
            SpirvLang::Sub([Id::from(3), Id::from(4)]), // 5 = (x + 5) - (y + 5)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let SpirvLang::Sub([lhs, rhs]) = nodes.last().expect("optimized root") else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        assert!(
            matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("x"))
                && matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("y")),
            "expected x - y, got {nodes:?}"
        );
    }

    #[test]
    fn cancels_shared_symbolic_addends_even_when_commuted() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")),       // 0
            SpirvLang::Symbol(Symbol::from("b")),       // 1
            SpirvLang::Symbol(Symbol::from("c")),       // 2
            SpirvLang::Symbol(Symbol::from("d")),       // 3
            SpirvLang::Add([Id::from(1), Id::from(0)]), // 4 = b + a
            SpirvLang::Add([Id::from(3), Id::from(1)]), // 5 = d + b
            SpirvLang::Sub([Id::from(4), Id::from(5)]), // 6 = (b + a) - (d + b)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let SpirvLang::Sub([lhs, rhs]) = nodes.last().expect("optimized root") else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        assert!(
            matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("a"))
                && matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("d")),
            "expected a - d, got {nodes:?}"
        );
    }

    #[test]
    fn collapses_chain_of_subtractions_with_shared_middle_term() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 3 = x - y
            SpirvLang::Sub([Id::from(1), Id::from(2)]), // 4 = y - z
            SpirvLang::Add([Id::from(3), Id::from(4)]), // 5 = (x - y) + (y - z)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let SpirvLang::Sub([lhs, rhs]) = nodes.last().expect("optimized root") else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        assert!(
            matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("x"))
                && matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("z")),
            "expected x - z, got {nodes:?}"
        );
    }

    #[test]
    fn cancels_mirrored_sub_in_addition() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Sub([Id::from(1), Id::from(0)]), // 2 = y - x
            SpirvLang::Add([Id::from(0), Id::from(2)]), // 3 = x + (y - x)
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("y"))])
        );
    }

    #[test]
    fn cancels_subtrahend_in_addition() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = x - y
            SpirvLang::Add([Id::from(2), Id::from(1)]), // 3 = (x - y) + y
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn cancels_add_then_sub_same_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 2 = x + y
            SpirvLang::Sub([Id::from(2), Id::from(0)]), // 3 = (x + y) - x
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("y"))])
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
    fn rewrites_add_dup_into_mul_by_two() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Add([Id::from(0), Id::from(0)]), // 1 = x + x
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_mul = nodes.iter().any(|n| {
            if let SpirvLang::Mul([lhs, rhs]) = n {
                let lhs_const = const_value(&runner.egraph, *lhs);
                let rhs_const = const_value(&runner.egraph, *rhs);
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*rhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                (lhs_sym && rhs_const.map(|c| c.get()) == Some(2))
                    || (rhs_sym && lhs_const.map(|c| c.get()) == Some(2))
            } else {
                false
            }
        });
        assert!(found_mul, "expected x + x to admit x * 2 in the e-graph");
    }

    #[test]
    fn rewrites_add_triple_into_mul_by_three() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Add([Id::from(0), Id::from(0)]), // 1 = x + x
            SpirvLang::Add([Id::from(1), Id::from(0)]), // 2 = (x + x) + x
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_mul = nodes.iter().any(|n| {
            if let SpirvLang::Mul([lhs, rhs]) = n {
                let lhs_const = const_value(&runner.egraph, *lhs);
                let rhs_const = const_value(&runner.egraph, *rhs);
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*rhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                (lhs_sym && rhs_const.map(|c| c.get()) == Some(3))
                    || (rhs_sym && lhs_const.map(|c| c.get()) == Some(3))
            } else {
                false
            }
        });
        assert!(
            found_mul,
            "expected x + x + x to admit x * 3 in the e-graph"
        );
    }

    #[test]
    fn rewrites_add_quadruple_into_shift() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Add([Id::from(0), Id::from(0)]), // 1 = x + x
            SpirvLang::Add([Id::from(1), Id::from(1)]), // 2 = (x + x) + (x + x)
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_shl = nodes.iter().any(|n| {
            if let SpirvLang::Shl([lhs, rhs]) = n {
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_const = const_value(&runner.egraph, *rhs).is_some_and(|c| c.get() == 2);
                lhs_sym && rhs_const
            } else {
                false
            }
        });
        assert!(found_shl, "expected 4*x to admit shift-left-by-2 form");
    }

    #[test]
    fn rewrites_merge_nested_shl_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(1)),       // 1
            SpirvLang::Shl([Id::from(0), Id::from(1)]), // 2 = x << 1
            SpirvLang::Const(ConstValue::new(2)),       // 3
            SpirvLang::Shl([Id::from(2), Id::from(3)]), // 4 = (x << 1) << 2
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_shl = nodes.iter().any(|n| {
            if let SpirvLang::Shl([lhs, rhs]) = n {
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_const = const_value(&runner.egraph, *rhs).is_some_and(|c| c.get() == 3);
                lhs_sym && rhs_const
            } else {
                false
            }
        });
        assert!(
            found_shl,
            "expected nested shifts to merge into a single offset"
        );
    }

    #[test]
    fn rewrites_merge_nested_shru_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(1)),        // 1
            SpirvLang::ShrU([Id::from(0), Id::from(1)]), // 2 = x >> 1 (logical)
            SpirvLang::Const(ConstValue::new(2)),        // 3
            SpirvLang::ShrU([Id::from(2), Id::from(3)]), // 4 = (x >> 1) >> 2
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_shr = nodes.iter().any(|n| {
            if let SpirvLang::ShrU([lhs, rhs]) = n {
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_const = const_value(&runner.egraph, *rhs).is_some_and(|c| c.get() == 3);
                lhs_sym && rhs_const
            } else {
                false
            }
        });
        assert!(
            found_shr,
            "expected nested logical right shifts to merge into a single offset"
        );
    }

    #[test]
    fn rewrites_merge_nested_shrs_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(1)),        // 1
            SpirvLang::ShrS([Id::from(0), Id::from(1)]), // 2 = x >> 1 (arithmetic)
            SpirvLang::Const(ConstValue::new(2)),        // 3
            SpirvLang::ShrS([Id::from(2), Id::from(3)]), // 4 = (x >> 1) >> 2
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_shr = nodes.iter().any(|n| {
            if let SpirvLang::ShrS([lhs, rhs]) = n {
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_const = const_value(&runner.egraph, *rhs).is_some_and(|c| c.get() == 3);
                lhs_sym && rhs_const
            } else {
                false
            }
        });
        assert!(
            found_shr,
            "expected nested arithmetic right shifts to merge into a single offset"
        );
    }

    #[test]
    fn rewrites_shift_left_zero_offset_identity() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(0)),       // 1
            SpirvLang::Shl([Id::from(0), Id::from(1)]), // 2 = x << 0
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let has_symbol = nodes.iter().any(|n| {
            matches!(
                n,
                SpirvLang::Symbol(sym) if *sym == Symbol::from("x")
            )
        });
        assert!(
            has_symbol,
            "expected shift-left by zero to collapse to identity"
        );
    }

    #[test]
    fn rewrites_shift_right_zero_offset_identity() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(0)),        // 1
            SpirvLang::ShrU([Id::from(0), Id::from(1)]), // 2 = x >> 0 (logical)
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let has_symbol = nodes.iter().any(|n| {
            matches!(
                n,
                SpirvLang::Symbol(sym) if *sym == Symbol::from("x")
            )
        });
        assert!(
            has_symbol,
            "expected logical shift-right by zero to collapse to identity"
        );
    }

    #[test]
    fn rewrites_arithmetic_shift_right_zero_offset_identity() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(0)),        // 1
            SpirvLang::ShrS([Id::from(0), Id::from(1)]), // 2 = x >> 0 (arithmetic)
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let has_symbol = nodes.iter().any(|n| {
            matches!(
                n,
                SpirvLang::Symbol(sym) if *sym == Symbol::from("x")
            )
        });
        assert!(
            has_symbol,
            "expected arithmetic shift-right by zero to collapse to identity"
        );
    }

    #[test]
    fn rewrites_mul_power_of_two_into_shift() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(8)),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = x * 8
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_shl = nodes.iter().any(|n| {
            if let SpirvLang::Shl([lhs, rhs]) = n {
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_const = const_value(&runner.egraph, *rhs).is_some_and(|c| c.get() == 3);
                lhs_sym && rhs_const
            } else {
                false
            }
        });
        assert!(found_shl, "expected x * 8 to admit x << 3");
    }

    #[test]
    fn rewrites_udiv_power_of_two_into_shift() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(4)),        // 1
            SpirvLang::UDiv([Id::from(0), Id::from(1)]), // 2 = x / 4
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_shr = nodes.iter().any(|n| {
            if let SpirvLang::ShrU([lhs, rhs]) = n {
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_const = const_value(&runner.egraph, *rhs).is_some_and(|c| c.get() == 2);
                lhs_sym && rhs_const
            } else {
                false
            }
        });
        assert!(found_shr, "expected x / 4 to admit x >> 2");
    }

    #[test]
    fn rewrites_sdiv_power_of_two_into_shift_with_bias() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(4)),        // 1
            SpirvLang::SDiv([Id::from(0), Id::from(1)]), // 2 = x / 4
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_shr = nodes.iter().any(|node| {
            if let SpirvLang::ShrS([lhs, rhs]) = node {
                let rhs_is_two =
                    const_value(&runner.egraph, *rhs).is_some_and(|val| val.get() == 2);
                if !rhs_is_two {
                    return false;
                }
                let add_eclass = runner.egraph.find(*lhs);
                let add_nodes = &runner.egraph[add_eclass].nodes;
                let has_mask = add_nodes.iter().any(|candidate| {
                    if let SpirvLang::Add([_, band]) = candidate {
                        let band_eclass = runner.egraph.find(*band);
                        runner.egraph[band_eclass].nodes.iter().any(|band_node| {
                            if let SpirvLang::BitAnd([_, mask]) = band_node {
                                const_value(&runner.egraph, *mask).is_some_and(|val| val.get() == 3)
                            } else {
                                false
                            }
                        })
                    } else {
                        false
                    }
                });
                rhs_is_two && has_mask
            } else {
                false
            }
        });
        assert!(
            found_shr,
            "expected signed div by power of two to rewrite into biased shift"
        );
    }

    #[test]
    fn rewrites_srem_const_into_div_mul_sub() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(5)),        // 1
            SpirvLang::SRem([Id::from(0), Id::from(1)]), // 2 = x % 5
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_sub = nodes.iter().any(|n| {
            if let SpirvLang::Sub([lhs, rhs]) = n {
                let lhs_is_x = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_eclass = runner.egraph.find(*rhs);
                let rhs_nodes = &runner.egraph[rhs_eclass].nodes;
                let rhs_is_mul_div = rhs_nodes.iter().any(|mul_node| {
                    if let SpirvLang::Mul([div, c]) = mul_node {
                        const_value(&runner.egraph, *c).is_some_and(|cv| cv.get() == 5)
                            && runner.egraph[runner.egraph.find(*div)]
                                .nodes
                                .iter()
                                .any(|div_node| matches!(div_node, SpirvLang::SDiv([_, _])))
                    } else {
                        false
                    }
                });
                lhs_is_x && rhs_is_mul_div
            } else {
                false
            }
        });
        assert!(
            found_sub,
            "expected signed remainder to decompose into x - (x / c) * c"
        );
    }

    #[test]
    fn rewrites_mod_by_one_into_zero() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(1)),        // 1
            SpirvLang::UMod([Id::from(0), Id::from(1)]), // 2 = x % 1
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let has_zero = nodes.iter().any(|n| {
            matches!(
                n,
                SpirvLang::Const(val) if val.get() == 0
            )
        });
        assert!(has_zero, "umod by one should fold to zero");

        let expr_signed = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(1)),        // 1
            SpirvLang::SRem([Id::from(0), Id::from(1)]), // 2 = x % 1
        ]);
        let runner_signed = Runner::default().with_expr(&expr_signed).run(&rewrites());
        let class_signed = runner_signed.egraph.find(runner_signed.roots[0]);
        let nodes_signed = &runner_signed.egraph[class_signed].nodes;
        let has_zero_signed = nodes_signed.iter().any(|n| {
            matches!(
                n,
                SpirvLang::Const(val) if val.get() == 0
            )
        });
        assert!(has_zero_signed, "srem by one should fold to zero");
    }

    #[test]
    fn optimize_arith_block_folds_mod_by_one() {
        let int = 1;
        let c5 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(5)],
        );
        let c1 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(1)],
        );
        let umod = Instruction::new(
            rspirv::spirv::Op::UMod,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let block = vec![c5.clone(), c1.clone(), umod.clone()];
        let optimized = optimize_arith_block(&block).expect("optimization should succeed");
        let folded = optimized.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Constant
                && inst.result_id == Some(3)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        });
        assert!(folded, "umod by one should fold to zero with same id");
    }

    #[test]
    fn rewrites_umod_const_into_div_mul_sub() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(5)),        // 1
            SpirvLang::UMod([Id::from(0), Id::from(1)]), // 2 = x % 5
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_sub = nodes.iter().any(|n| {
            if let SpirvLang::Sub([lhs, rhs]) = n {
                let lhs_is_x = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_eclass = runner.egraph.find(*rhs);
                let rhs_nodes = &runner.egraph[rhs_eclass].nodes;
                let rhs_is_mul_div = rhs_nodes.iter().any(|mul_node| {
                    if let SpirvLang::Mul([div, c]) = mul_node {
                        const_value(&runner.egraph, *c).is_some_and(|cv| cv.get() == 5)
                            && runner.egraph[runner.egraph.find(*div)]
                                .nodes
                                .iter()
                                .any(|div_node| matches!(div_node, SpirvLang::UDiv([_, _])))
                    } else {
                        false
                    }
                });
                lhs_is_x && rhs_is_mul_div
            } else {
                false
            }
        });
        assert!(
            found_sub,
            "expected unsigned remainder to decompose into x - (x / c) * c"
        );
    }

    #[test]
    fn rewrites_band_pow2_mask_into_umod() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),          // 0
            SpirvLang::Const(ConstValue::new(7)),          // 1 = 2^3 - 1
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]), // 2 = x & 7
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_umod = nodes.iter().any(|n| {
            if let SpirvLang::UMod([lhs, rhs]) = n {
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_const = const_value(&runner.egraph, *rhs).is_some_and(|c| c.get() == 8);
                lhs_sym && rhs_const
            } else {
                false
            }
        });
        assert!(
            found_umod,
            "expected bitwise mask to rewrite into modulo by power of two"
        );
    }

    #[test]
    fn rewrites_umod_power_of_two_into_subtract_mask() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(8)),        // 1
            SpirvLang::UMod([Id::from(0), Id::from(1)]), // 2 = x % 8
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let found_sub = runner
            .egraph
            .classes()
            .any(|c| c.nodes.iter().any(|n| matches!(n, SpirvLang::Sub(_))));
        let found_shift = runner
            .egraph
            .classes()
            .any(|c| c.nodes.iter().any(|n| matches!(n, SpirvLang::ShrU(_))));
        assert!(
            found_sub && found_shift,
            "umod pow2 should rewrite into subtract mask via shifts"
        );
    }

    #[test]
    fn rewrites_umod_power_of_two_into_mask() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(8)),        // 1
            SpirvLang::UMod([Id::from(0), Id::from(1)]), // 2 = x % 8
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let found_band = nodes.iter().any(|n| {
            if let SpirvLang::BitAnd([lhs, rhs]) = n {
                let lhs_sym = matches!(
                    runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                    [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                );
                let rhs_const = const_value(&runner.egraph, *rhs).is_some_and(|c| c.get() == 7);
                lhs_sym && rhs_const
            } else {
                false
            }
        });
        assert!(
            found_band,
            "expected umod pow2 to admit bitmask form x & (c-1)"
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
    fn cancels_double_negation() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Neg(Id::from(0)),          // 1 = -x
            SpirvLang::Neg(Id::from(1)),          // 2 = -(-x)
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn normalizes_negated_multiplicand_with_constant() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(5)),       // 0
            SpirvLang::Neg(Id::from(0)),                // 1 = -5
            SpirvLang::Symbol(Symbol::from("x")),       // 2
            SpirvLang::Mul([Id::from(1), Id::from(2)]), // 3 = (-5) * x
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        assert!(
            nodes.iter().all(|node| !matches!(node, SpirvLang::Neg(_))),
            "negation should be folded away: {nodes:?}"
        );
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let const_val = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) if *sym == Symbol::from("x") => val,
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) if *sym == Symbol::from("x") => val,
            other => panic!("unexpected operands for normalized mul: {other:?}"),
        };
        assert_eq!(
            const_val.get(),
            (-5i32) as u32,
            "constant multiplier should carry the negated value"
        );
    }

    #[test]
    fn pulls_constant_factor_out_of_division_signed() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(6)),       // 0
            SpirvLang::Symbol(Symbol::from("x")),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = 6 * x
            SpirvLang::Const(ConstValue::new(3)),       // 3
            SpirvLang::SDiv([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let (symbol, constant) = match nodes.last() {
            Some(SpirvLang::Mul([lhs, rhs])) | Some(SpirvLang::Shl([lhs, rhs])) => {
                match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
                    (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
                    (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
                    other => panic!("unexpected operands for simplified div: {other:?}"),
                }
            }
            other => panic!("expected mul or shl root, got {:?}", other),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        let expected = match nodes.last() {
            Some(SpirvLang::Mul(_)) => 2,
            _ => 1, // shift amount for multiply by two
        };
        assert_eq!(constant.get(), expected);
    }

    #[test]
    fn pulls_constant_factor_out_of_division_unsigned() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(12)),      // 0
            SpirvLang::Symbol(Symbol::from("x")),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = 12 * x
            SpirvLang::Const(ConstValue::new(4)),       // 3
            SpirvLang::UDiv([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands for simplified div: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 3);
    }

    #[test]
    fn distributes_constant_mul_over_add_with_constant() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(3)),       // 0
            SpirvLang::Symbol(Symbol::from("x")),       // 1
            SpirvLang::Const(ConstValue::new(2)),       // 2
            SpirvLang::Add([Id::from(1), Id::from(2)]), // 3 = x + 2
            SpirvLang::Mul([Id::from(0), Id::from(3)]), // 4 = 3 * (x + 2)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Add([lhs, rhs])) = nodes.last() else {
            panic!("expected add root, got {:?}", nodes.last());
        };
        let (mul_node, const_node) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Mul(_), SpirvLang::Const(c)) => (&nodes[usize::from(*lhs)], c),
            (SpirvLang::Const(c), SpirvLang::Mul(_)) => (&nodes[usize::from(*rhs)], c),
            other => panic!("unexpected operands for distributed mul: {other:?}"),
        };
        assert_eq!(const_node.get(), 6, "constant term should be scaled");
        if let SpirvLang::Mul([a, b]) = mul_node {
            let lhs_node = &nodes[usize::from(*a)];
            let rhs_node = &nodes[usize::from(*b)];
            assert!(
                matches!(lhs_node, SpirvLang::Const(c) if c.get() == 3)
                    && matches!(rhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x"))
                    || matches!(rhs_node, SpirvLang::Const(c) if c.get() == 3)
                        && matches!(lhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")),
                "expected 3 * x after distribution, got lhs={lhs_node:?} rhs={rhs_node:?}"
            );
        } else {
            panic!("expected mul term alongside constant after distribution");
        }
    }

    #[test]
    fn distributes_constant_mul_over_sub_with_constant_rhs() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(4)),       // 0
            SpirvLang::Symbol(Symbol::from("x")),       // 1
            SpirvLang::Const(ConstValue::new(1)),       // 2
            SpirvLang::Sub([Id::from(1), Id::from(2)]), // 3 = x - 1
            SpirvLang::Mul([Id::from(0), Id::from(3)]), // 4 = 4 * (x - 1)
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let class = runner.egraph.find(root);
        let nodes = runner.egraph[class].nodes.clone();

        let is_scaled_mul = |id: Id| {
            let class = runner.egraph.find(id);
            runner.egraph[class].nodes.iter().any(|n| match n {
                SpirvLang::Mul([lhs, rhs]) => {
                    let lhs_const = const_value(&runner.egraph, *lhs);
                    let rhs_const = const_value(&runner.egraph, *rhs);
                    let lhs_sym = matches!(
                        runner.egraph[runner.egraph.find(*lhs)].nodes.as_slice(),
                        [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                    );
                    let rhs_sym = matches!(
                        runner.egraph[runner.egraph.find(*rhs)].nodes.as_slice(),
                        [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                    );
                    (lhs_const.map(|c| c.get()) == Some(4) && rhs_sym)
                        || (rhs_const.map(|c| c.get()) == Some(4) && lhs_sym)
                }
                _ => false,
            })
        };

        let distributed = nodes.iter().any(|n| {
            if let SpirvLang::Sub([lhs, rhs]) = n {
                let lhs_const = const_value(&runner.egraph, *lhs);
                let rhs_const = const_value(&runner.egraph, *rhs);
                match (lhs_const, rhs_const) {
                    (Some(c), None) if c.get() == 4 => is_scaled_mul(*rhs),
                    (None, Some(c)) if c.get() == 4 => is_scaled_mul(*lhs),
                    _ => false,
                }
            } else {
                false
            }
        });

        assert!(
            distributed,
            "expected distributed subexpression with scaled constant and mul term"
        );
    }

    #[test]
    fn factors_const_from_affine_add_when_divisible() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(8)),       // 0
            SpirvLang::Symbol(Symbol::from("x")),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = 8x
            SpirvLang::Const(ConstValue::new(16)),      // 3
            SpirvLang::Add([Id::from(2), Id::from(3)]), // 4 = 8x + 16
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let candidates = &runner.egraph[class].nodes;

        let distributed = candidates.iter().any(|n| {
            if let SpirvLang::Mul([lhs, rhs]) = n {
                let (const_id, add_id) = match (
                    const_value(&runner.egraph, *lhs),
                    const_value(&runner.egraph, *rhs),
                ) {
                    (Some(c), None) => (Some(c), Some(*rhs)),
                    (None, Some(c)) => (Some(c), Some(*lhs)),
                    _ => (None, None),
                };
                if let (Some(c), Some(add_id)) = (const_id, add_id) {
                    if c.get() != 8 {
                        return false;
                    }
                    let add_class = runner.egraph.find(add_id);
                    return runner.egraph[add_class].nodes.iter().any(|node| {
                        if let SpirvLang::Add([a, b]) = node {
                            let a_val = const_value(&runner.egraph, *a);
                            let b_val = const_value(&runner.egraph, *b);
                            let a_sym = matches!(
                                runner.egraph[runner.egraph.find(*a)].nodes.as_slice(),
                                [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                            );
                            let b_sym = matches!(
                                runner.egraph[runner.egraph.find(*b)].nodes.as_slice(),
                                [SpirvLang::Symbol(sym)] if *sym == Symbol::from("x")
                            );
                            (a_sym && b_val.map(|v| v.get()) == Some(16 / 8))
                                || (b_sym && a_val.map(|v| v.get()) == Some(16 / 8))
                        } else {
                            false
                        }
                    });
                }
            }
            false
        });

        assert!(
            distributed,
            "expected factored form 8 * (x + 12/8) in e-graph"
        );
    }

    #[test]
    fn factors_const_from_affine_sub_when_divisible() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(6)),       // 0
            SpirvLang::Symbol(Symbol::from("x")),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = 6x
            SpirvLang::Const(ConstValue::new(18)),      // 3
            SpirvLang::Sub([Id::from(2), Id::from(3)]), // 4 = 6x - 18
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (const_node, sub_node) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Const(c), SpirvLang::Sub(_)) => (c, &nodes[usize::from(*rhs)]),
            (SpirvLang::Sub(_), SpirvLang::Const(c)) => (c, &nodes[usize::from(*lhs)]),
            other => panic!("unexpected operands after factoring const: {other:?}"),
        };
        assert_eq!(const_node.get(), 6);
        let SpirvLang::Sub([a, b]) = sub_node else {
            panic!("expected inner sub, got {sub_node:?}");
        };
        let lhs_node = &nodes[usize::from(*a)];
        let rhs_node = &nodes[usize::from(*b)];
        assert!(
            matches!(lhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x"))
                && matches!(rhs_node, SpirvLang::Const(c) if c.get() == 18 / 6)
                || matches!(rhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x"))
                    && matches!(lhs_node, SpirvLang::Const(c) if c.get() == 18 / 6),
            "inner sub should be x - (18/6), got lhs={lhs_node:?} rhs={rhs_node:?}"
        );
    }

    #[test]
    fn skips_affine_factoring_with_negative_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new((-6i32) as u32)), // 0
            SpirvLang::Symbol(Symbol::from("x")),              // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]),        // 2 = -6x
            SpirvLang::Const(ConstValue::new((-9i32) as u32)), // 3
            SpirvLang::Add([Id::from(2), Id::from(3)]),        // 4 = -6x + -9
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let has_positive_factor = runner.egraph.classes().any(|class| {
            class
                .nodes
                .iter()
                .any(|n| matches!(n, SpirvLang::Const(c) if c.get() == 3))
        });
        assert!(
            !has_positive_factor,
            "negative constants should not introduce positive gcd factoring"
        );
    }

    #[test]
    fn swaps_subtraction_of_negated_terms() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Symbol(Symbol::from("y")), // 1
            SpirvLang::Neg(Id::from(0)),          // 2 = -x
            SpirvLang::Neg(Id::from(1)),          // 3 = -y
            SpirvLang::Sub([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Sub([lhs, rhs])) = nodes.last() else {
            panic!("expected sub root, got {:?}", nodes.last());
        };
        let lhs_node = &nodes[usize::from(*lhs)];
        let rhs_node = &nodes[usize::from(*rhs)];
        assert!(
            matches!(lhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("y"))
                && matches!(rhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")),
            "expected y - x after swapping, got lhs={lhs_node:?} rhs={rhs_node:?}"
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
        let (symbol, constant) = match nodes.last() {
            Some(SpirvLang::UDiv([lhs, rhs])) | Some(SpirvLang::ShrU([lhs, rhs])) => {
                match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
                    (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
                    (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
                    other => panic!("unexpected operands for merged udiv: {other:?}"),
                }
            }
            other => panic!("expected udiv or shr root, got {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        let expected = match nodes.last() {
            Some(SpirvLang::UDiv(_)) => 8,
            _ => 3, // shift amount for 8
        };
        assert_eq!(constant.get(), expected, "divisors should merge to 8");
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
    fn optimize_arith_block_preserves_bitwise_and() {
        let int = 1;
        let c3 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(3)],
        );
        let c1 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(1)],
        );
        let band = Instruction::new(
            rspirv::spirv::Op::BitwiseAnd,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let block = vec![c3.clone(), c1.clone(), band.clone()];
        let optimized = optimize_arith_block(&block).expect("bitwise and should be supported");
        let expected_const = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::LiteralBit32(1)],
        );
        assert!(
            optimized == block || optimized == vec![expected_const],
            "bitwise and should either pass through or fold constants"
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
    fn optimize_arith_block_rewrites_mul_pow2_to_shift() {
        let int = 1;
        let c8 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(8)],
        );
        let c_id = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(5)],
        );
        let mul = Instruction::new(
            rspirv::spirv::Op::IMul,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(1)],
        );
        let block = vec![c8, c_id, mul];
        let optimized = optimize_arith_block(&block).expect("optimization should succeed");
        let found_shift = optimized.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::ShiftLeftLogical && inst.result_id == Some(3)
        });
        let found_constant = optimized.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Constant
                && inst.result_id == Some(3)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(40)]
        });
        assert!(
            found_shift || found_constant,
            "expected shift or folded constant for mul by power of two"
        );
    }

    #[test]
    fn optimize_arith_block_rewrites_sdiv_pow2_to_biased_shift() {
        let int = 1;
        let c4 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(4)],
        );
        let c_id = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(5)],
        );
        let div = Instruction::new(
            rspirv::spirv::Op::SDiv,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(1)],
        );
        let block = vec![c4, c_id, div];
        let optimized = optimize_arith_block(&block).expect("optimization should succeed");
        let has_shift = optimized.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::ShiftRightArithmetic
                && inst.result_id == Some(3)
        });
        let folded_const = optimized.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Constant
                && inst.result_id == Some(3)
                && inst
                    .operands
                    .iter()
                    .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(1)))
        });
        assert!(
            has_shift || folded_const,
            "expected biased shift or folded constant for signed power-of-two div"
        );
        if has_shift {
            let has_mask = optimized.iter().any(|inst| {
                inst.class.opcode == rspirv::spirv::Op::BitwiseAnd
                    && inst
                        .operands
                        .iter()
                        .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(3)))
            });
            assert!(has_mask, "expected bias mask in signed div rewrite");
        }
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
