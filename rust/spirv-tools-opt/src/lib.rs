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
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    str::FromStr,
};

pub mod control;
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
                match u.choose(&[
                    0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                ])? {
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
                    9 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::BitOr([Id::from(a), Id::from(b)])
                    }
                    10 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::BitXor([Id::from(a), Id::from(b)])
                    }
                    11 => {
                        let a = choose_child(u, idx - 1)?;
                        SpirvLang::BitNot(Id::from(a))
                    }
                    12 => {
                        let a = choose_child(u, idx - 1)?;
                        SpirvLang::BitReverse(Id::from(a))
                    }
                    13 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::Eq([Id::from(a), Id::from(b)])
                    }
                    14 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::Ne([Id::from(a), Id::from(b)])
                    }
                    15 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        if *u.choose(&[true, false])? {
                            SpirvLang::LogAnd([Id::from(a), Id::from(b)])
                        } else {
                            SpirvLang::LogOr([Id::from(a), Id::from(b)])
                        }
                    }
                    16 => {
                        let a = choose_child(u, idx - 1)?;
                        SpirvLang::LogNot(Id::from(a))
                    }
                    17 => {
                        let a = choose_child(u, idx - 1)?;
                        let b = choose_child(u, idx - 1)?;
                        SpirvLang::SMod([Id::from(a), Id::from(b)])
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
/// We track the literal bits and the bit width (currently 32 or 64) so
/// width-sensitive rewrites (e.g., rotates) can fold correctly while keeping
/// existing 32-bit folds simple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstValue {
    value: u64,
    width: u8,
}

impl ConstValue {
    /// Constructs a new constant value.
    pub const fn new(value: u32) -> Self {
        Self::new_with_width(value as u64, 32)
    }

    /// Constructs a new 64-bit constant value.
    pub const fn new64(value: u64) -> Self {
        Self::new_with_width(value, 64)
    }

    /// Constructs a constant with an explicit bit width (masked to that width).
    pub const fn new_with_width(value: u64, width: u8) -> Self {
        let mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        Self {
            value: value & mask,
            width,
        }
    }

    /// Returns the raw 32-bit value.
    pub const fn get(self) -> u32 {
        (self.value & 0xFFFF_FFFF) as u32
    }

    /// Returns the raw value masked to its bit width.
    pub const fn get_u64(self) -> u64 {
        let mask = if self.width >= 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        };
        self.value & mask
    }

    /// Returns true when the value is zero in its bit width.
    pub const fn is_zero(self) -> bool {
        self.get_u64() == 0
    }

    /// Returns true when the value is one in its bit width.
    pub const fn is_one(self) -> bool {
        self.get_u64() == 1
    }

    /// Returns true when all bits in the width are set.
    pub const fn is_all_ones(self) -> bool {
        self.get_u64() == self.mask()
    }

    /// Returns the bit width.
    pub const fn width_bits(self) -> u8 {
        self.width
    }

    /// Returns a mask for the width.
    pub const fn mask(self) -> u64 {
        if self.width >= 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        }
    }
}

impl fmt::Display for ConstValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

impl std::str::FromStr for ConstValue {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u32>().map(Self::new)
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
        ConstValue::new(0)
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
        "bor" = BitOr([Id; 2]),
        "bxor" = BitXor([Id; 2]),
        "bnot" = BitNot(Id),
        "brev" = BitReverse(Id),
        "rotl" = RotL([Id; 2]),
        "rotr" = RotR([Id; 2]),
        "-" = Sub([Id; 2]),
        "sdiv" = SDiv([Id; 2]),
        "udiv" = UDiv([Id; 2]),
        "srem" = SRem([Id; 2]),
        "smod" = SMod([Id; 2]),
        "umod" = UMod([Id; 2]),
        "neg" = Neg(Id),
        "select" = Select([Id; 3]),
        "eq" = Eq([Id; 2]),
        "ne" = Ne([Id; 2]),
        "slt" = SLt([Id; 2]),
        "sle" = SLe([Id; 2]),
        "sgt" = SGt([Id; 2]),
        "sge" = SGe([Id; 2]),
        "ult" = ULt([Id; 2]),
        "ule" = ULe([Id; 2]),
        "ugt" = UGt([Id; 2]),
        "uge" = UGe([Id; 2]),
        "lnot" = LogNot(Id),
        "land" = LogAnd([Id; 2]),
        "lor" = LogOr([Id; 2]),
        "leq" = LogEq([Id; 2]),
        "lne" = LogNe([Id; 2]),
        "if" = If([Id; 3]),
        "merge" = Merge([Id; 2]),
        "ret" = Ret,
        "retv" = RetVal(Id),
        "phi" = Phi([Id; 2]),
        "pair" = Pair([Id; 2]),
        Const(ConstValue),
        Symbol(egg::Symbol),
    }
}

thread_local! {
    static SYMBOL_WIDTH_HINTS: RefCell<HashMap<Symbol, u8>> = RefCell::new(HashMap::new());
}

pub fn with_symbol_widths<R>(hints: &HashMap<Symbol, u8>, f: impl FnOnce() -> R) -> R {
    SYMBOL_WIDTH_HINTS.with(|cell| {
        let previous = cell.replace(hints.clone());
        let result = f();
        cell.replace(previous);
        result
    })
}

fn symbol_width(sym: &Symbol) -> Option<u8> {
    SYMBOL_WIDTH_HINTS.with(|cell| cell.borrow().get(sym).copied())
}

/// Lightweight cost function favoring folded constants and shallower trees.
pub struct ExprCost;

impl egg::CostFunction<SpirvLang> for ExprCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &SpirvLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            SpirvLang::Const(_) => 1,
            SpirvLang::Mul(_) => enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 2,
            SpirvLang::SDiv(_) | SpirvLang::UDiv(_) => {
                enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 8
            }
            SpirvLang::Shl(_) | SpirvLang::ShrS(_) | SpirvLang::ShrU(_) => {
                enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 1
            }
            SpirvLang::BitAnd(_) | SpirvLang::BitXor(_) => {
                enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 2
            }
            SpirvLang::BitNot(_) | SpirvLang::BitReverse(_) => {
                enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 1
            }
            SpirvLang::If(_)
            | SpirvLang::Merge(_)
            | SpirvLang::Phi(_)
            | SpirvLang::Pair(_)
            | SpirvLang::Select(_) => {
                enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 1
            }
            SpirvLang::Ret => 1,
            SpirvLang::RetVal(child) => costs(*child) + 1,
            _ => enode.children().iter().map(|id| costs(*id)).sum::<usize>() + 1,
        }
    }
}

/// Optimize an expression by applying algebraic rewrites and constant folding.
///
/// The returned expression is the cheapest representative (per `ExprCost`) of
/// the root e-class after saturation.
pub fn optimize_expr(expr: &RecExpr<SpirvLang>) -> RecExpr<SpirvLang> {
    let mut rewrites = rewrites();
    if expr_has_bitwise(expr) {
        // Avoid remainder decomposition in mixed bitwise expressions; it can
        // combine with mask/shift rewrites to drop masked symbols to constants.
        rewrites.retain(|rw| {
            let name = rw.name.as_str();
            !name.contains("affine")
                && !name.contains("gcd")
                && !name.contains("cancel-common-factor")
                && !name.contains("mul-const-zero")
                && !name.contains("rem-const-decompose")
                && !name.contains("mod-const-decompose")
                && !name.contains("umod-power-of-two")
        });
    }
    let runner = Runner::default().with_expr(expr).run(&rewrites);
    let root = runner.roots[0];
    let extractor = egg::Extractor::new(&runner.egraph, ExprCost);
    let (_cost, best) = extractor.find_best(root);
    best
}

fn expr_has_bitwise(expr: &RecExpr<SpirvLang>) -> bool {
    expr.as_ref().iter().any(|node| {
        matches!(
            node,
            SpirvLang::BitAnd(_)
                | SpirvLang::BitOr(_)
                | SpirvLang::BitXor(_)
                | SpirvLang::BitNot(_)
                | SpirvLang::BitReverse(_)
                | SpirvLang::Shl(_)
                | SpirvLang::ShrS(_)
                | SpirvLang::ShrU(_)
        )
    })
}

/// Optimize the root of a translated SPIR-V arithmetic expression.
pub fn optimize_translated(expr: &crate::translate::TranslatedExpr) -> RecExpr<SpirvLang> {
    with_symbol_widths(&expr.symbol_widths, || optimize_expr(&expr.expr))
}

/// Returns the full set of algebraic rewrites used by the optimizer.
pub fn rewrites() -> Vec<Rewrite<SpirvLang, ()>> {
    vec![
        rewrite!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-assoc"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("mul-assoc"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rewrite!("add-zero"; "(+ ?a ?b)" => { AddZero { a: var("?a"), b: var("?b") } }),
        rewrite!("add-neg-to-sub"; "(+ ?a (neg ?b))" => "(- ?a ?b)"),
        rewrite!("add-neg-to-sub-swap"; "(+ (neg ?a) ?b)" => "(- ?b ?a)"),
        rewrite!("add-neg-neg"; "(+ (neg ?a) (neg ?b))" => "(neg (+ ?a ?b))"),
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
        rewrite!("sub-add-cancel-left"; "(- ?x (+ ?x ?y))" => "(neg ?y)"),
        rewrite!("sub-add-cancel-left-comm"; "(- ?x (+ ?y ?x))" => "(neg ?y)"),
        rewrite!("add-fold"; "(+ ?a ?b)" => { FoldAdd }),
        rewrite!("mul-fold"; "(* ?a ?b)" => { FoldMul }),
        rewrite!("sub-fold"; "(- ?a ?b)" => { FoldSub }),
        rewrite!("sub-zero-right"; "(- ?a ?b)" => "?a" if is_const_zero(var("?b"))),
        rewrite!("sub-zero-left"; "(- ?a ?b)" => { SubZeroLeft }),
        rewrite!("sub-self"; "(- ?a ?a)" => { SubSelf }),
        rewrite!("sub-neg-right-to-add"; "(- ?a (neg ?b))" => "(+ ?a ?b)"),
        rewrite!("sub-neg-left-to-neg-add"; "(- (neg ?a) ?b)" => "(neg (+ ?a ?b))"),
        rewrite!("sub-sub-cancel-left"; "(- ?a (- ?a ?b))" => "?b"),
        rewrite!("sub-sub-cancel-right"; "(- (- ?x ?y) ?x)" => "(neg ?y)"),
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
        rewrite!("shl-const-fold"; "(shl ?a ?b)" => { ShlFoldConst { a: var("?a"), b: var("?b") } }),
        rewrite!("shr-u-const-fold"; "(shr_u ?a ?b)" => { ShrUFoldConst { a: var("?a"), b: var("?b") } }),
        rewrite!("shr-s-const-fold"; "(shr_s ?a ?b)" => { ShrSFoldConst { a: var("?a"), b: var("?b") } }),
        rewrite!("shl-zero"; "(shl ?x ?c)" => "?x" if is_const_zero(var("?c"))),
        rewrite!("shr-u-zero"; "(shr_u ?x ?c)" => "?x" if is_const_zero(var("?c"))),
        rewrite!("shr-s-zero"; "(shr_s ?x ?c)" => "?x" if is_const_zero(var("?c"))),
        rewrite!("shl-zero-left"; "(shl ?c ?x)" => "?c" if is_const_zero(var("?c"))),
        rewrite!("shr-u-zero-left"; "(shr_u ?c ?x)" => "?c" if is_const_zero(var("?c"))),
        rewrite!("shr-s-zero-left"; "(shr_s ?c ?x)" => "?c" if is_const_zero(var("?c"))),
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
        rewrite!("add-factor-symbolic-any-order"; "(+ (* ?y ?x) (* ?x ?z))" => {
            FactorSharedAnyOrder { factor: var("?x"), lhs: var("?y"), rhs: var("?z"), subtract: false }
        }),
        rewrite!("add-factor-symbolic-any-order-right"; "(+ (* ?x ?y) (* ?z ?x))" => {
            FactorSharedAnyOrder { factor: var("?x"), lhs: var("?y"), rhs: var("?z"), subtract: false }
        }),
        rewrite!("add-factor-symbolic-any-order-both"; "(+ (* ?y ?x) (* ?z ?x))" => {
            FactorSharedAnyOrder { factor: var("?x"), lhs: var("?y"), rhs: var("?z"), subtract: false }
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
        rewrite!("sub-factor-symbolic-any-order"; "(- (* ?y ?x) (* ?x ?z))" => {
            FactorSharedAnyOrder { factor: var("?x"), lhs: var("?y"), rhs: var("?z"), subtract: true }
        }),
        rewrite!("sub-factor-symbolic-any-order-right"; "(- (* ?x ?y) (* ?z ?x))" => {
            FactorSharedAnyOrder { factor: var("?x"), lhs: var("?y"), rhs: var("?z"), subtract: true }
        }),
        rewrite!("sub-factor-symbolic-any-order-both"; "(- (* ?y ?x) (* ?z ?x))" => {
            FactorSharedAnyOrder { factor: var("?x"), lhs: var("?y"), rhs: var("?z"), subtract: true }
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
        rewrite!("sdiv-zero-left"; "(sdiv ?c ?x)" => "?c" if is_const_zero(var("?c"))),
        rewrite!("udiv-zero-left"; "(udiv ?c ?x)" => "?c" if is_const_zero(var("?c"))),
        rewrite!("srem-const-decompose"; "(srem ?x ?c)" => {
            SRemConstDecompose { x: var("?x"), c: var("?c") }
        }),
        rewrite!("umod-const-decompose"; "(umod ?x ?c)" => {
            UModConstDecompose { x: var("?x"), c: var("?c") }
        }),
        rewrite!("srem-zero-left"; "(srem ?c ?x)" => "?c" if is_const_zero(var("?c"))),
        rewrite!("smod-zero-left"; "(smod ?c ?x)" => "?c" if is_const_zero(var("?c"))),
        rewrite!("umod-zero-left"; "(umod ?c ?x)" => "?c" if is_const_zero(var("?c"))),
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
        rewrite!("band-complement-left"; "(band ?x (bnot ?x))" => { BitAndComplement { _x: var("?x") } }),
        rewrite!("band-complement-right"; "(band (bnot ?x) ?x)" => { BitAndComplement { _x: var("?x") } }),
        rewrite!("band-demorgan"; "(band (bnot ?a) (bnot ?b))" => "(bnot (bor ?a ?b))"),
        rewrite!("band-mask-to-umod"; "(band ?x ?c)" => {
            BitAndToUmod { x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-absorb-right"; "(band ?x (bor ?x ?y))" => "?x"),
        rewrite!("band-absorb-left"; "(band (bor ?x ?y) ?x)" => "?x"),
        rewrite!("band-distribute-over-or-right"; "(band ?x (bor ?y ?z))" => "(bor (band ?x ?y) (band ?x ?z))"),
        rewrite!("band-distribute-over-or-left"; "(band (bor ?y ?z) ?x)" => "(bor (band ?x ?y) (band ?x ?z))"),
        rewrite!("mask-then-shift-to-shift-and-umod";
        "(shr_u (band ?x ?c_mask) ?c_shift)" => {
            MaskThenShift { x: var("?x"), mask: var("?c_mask"), shift: var("?c_shift") }
        }),
        rewrite!("mask-then-arith-shift-to-shift-and-umod";
        "(shr_s (band ?x ?c_mask) ?c_shift)" => {
            MaskThenShiftSigned { x: var("?x"), mask: var("?c_mask"), shift: var("?c_shift") }
        }),
        rewrite!("band-self"; "(band ?x ?x)" => "?x"),
        rewrite!("umod-power-of-two-mask"; "(umod ?x ?c)" => {
            UModPowerOfTwoMask { x: var("?x"), c: var("?c") }
        }),
        rewrite!("bor-comm"; "(bor ?a ?b)" => "(bor ?b ?a)"),
        rewrite!("bor-assoc"; "(bor ?a (bor ?b ?c))" => "(bor (bor ?a ?b) ?c)"),
        rewrite!("bor-reassociate-const-right"; "(bor ?c1 (bor ?x ?c2))" => {
            BitwiseConstReassociate {
                op: BitwiseOp::Or,
                x: var("?x"),
                c1: var("?c1"),
                c2: var("?c2"),
            }
        }),
        rewrite!("bor-reassociate-const-left"; "(bor (bor ?x ?c2) ?c1)" => {
            BitwiseConstReassociate {
                op: BitwiseOp::Or,
                x: var("?x"),
                c1: var("?c1"),
                c2: var("?c2"),
            }
        }),
        rewrite!("bor-const-fold"; "(bor ?a ?b)" => { BitOrFold { a: var("?a"), b: var("?b") } }),
        rewrite!("bor-zero-left"; "(bor ?x ?c)" => {
            BitOrConstSimplify { x: var("?x"), c: var("?c") }
        }),
        rewrite!("bor-zero-right"; "(bor ?c ?x)" => {
            BitOrConstSimplify { x: var("?x"), c: var("?c") }
        }),
        rewrite!("bor-complement-left"; "(bor ?x (bnot ?x))" => { BitOrComplement { _x: var("?x") } }),
        rewrite!("bor-complement-right"; "(bor (bnot ?x) ?x)" => { BitOrComplement { _x: var("?x") } }),
        rewrite!("bor-demorgan"; "(bor (bnot ?a) (bnot ?b))" => "(bnot (band ?a ?b))"),
        rewrite!("bor-self"; "(bor ?x ?x)" => "?x"),
        rewrite!("bor-absorb-right"; "(bor ?x (band ?x ?y))" => "?x"),
        rewrite!("bor-absorb-left"; "(bor (band ?x ?y) ?x)" => "?x"),
        rewrite!("band-absorbs-or"; "(band ?x (bor ?x ?y))" => "?x"),
        rewrite!("band-absorbs-or-commuted"; "(band (bor ?x ?y) ?x)" => "?x"),
        rewrite!("bor-distribute-over-and-right"; "(bor ?x (band ?y ?z))" => "(band (bor ?x ?y) (bor ?x ?z))"),
        rewrite!("bor-distribute-over-and-left"; "(bor (band ?y ?z) ?x)" => "(band (bor ?x ?y) (bor ?x ?z))"),
        rewrite!("bor-absorbs-and-left-zero"; "(bor (band (bnot ?x) ?y) ?y)" => "?y"),
        rewrite!("bor-absorbs-and-right-zero"; "(bor ?y (band (bnot ?x) ?y))" => "?y"),
        rewrite!("bor-absorbs-and-left-one"; "(bor (band ?x ?y) ?x)" => "?x"),
        rewrite!("bor-absorbs-and-right-one"; "(bor ?x (band ?x ?y))" => "?x"),
        rewrite!("bor-absorbs-split-y"; "(bor (band ?x ?y) (band (bnot ?x) ?y))" => "?y"),
        rewrite!("bor-absorbs-split-y-comm"; "(bor (band (bnot ?x) ?y) (band ?x ?y))" => "?y"),
        rewrite!("bor-absorbs-split-x"; "(bor (band ?x ?y) (band ?x (bnot ?y)))" => "?x"),
        rewrite!("bor-absorbs-split-x-comm"; "(bor (band ?x (bnot ?y)) (band ?x ?y))" => "?x"),
        rewrite!("bxor-absorbs-split-y"; "(bxor (band ?x ?y) (band (bnot ?x) ?y))" => "?y"),
        rewrite!("bxor-absorbs-split-y-comm"; "(bxor (band (bnot ?x) ?y) (band ?x ?y))" => "?y"),
        rewrite!("bxor-absorbs-split-x"; "(bxor (band ?x ?y) (band ?x (bnot ?y)))" => "?x"),
        rewrite!("bxor-absorbs-split-x-comm"; "(bxor (band ?x (bnot ?y)) (band ?x ?y))" => "?x"),
        // Rust-only improvement: absorb complement masks in OR/AND/XOR.
        rewrite!("bor-absorb-complement-mask-right"; "(bor ?x (band (bnot ?x) ?y))" => "(bor ?x ?y)"),
        rewrite!("bor-absorb-complement-mask-right-comm"; "(bor ?x (band ?y (bnot ?x)))" => "(bor ?x ?y)"),
        rewrite!("band-absorb-complement-or-right"; "(band ?x (bor (bnot ?x) ?y))" => "(band ?x ?y)"),
        rewrite!("band-absorb-complement-or-left"; "(band (bor (bnot ?x) ?y) ?x)" => "(band ?x ?y)"),
        rewrite!("bxor-absorb-complement-mask-right"; "(bxor ?x (band (bnot ?x) ?y))" => "(bor ?x ?y)"),
        rewrite!("bxor-absorb-complement-mask-right-comm"; "(bxor ?x (band ?y (bnot ?x)))" => "(bor ?x ?y)"),
        rewrite!("bor-absorb-xor"; "(bor ?x (bxor ?x ?y))" => "(bor ?x ?y)"),
        // Rust-only improvement: x & (x ^ y) == x & ~y, removing the xor.
        rewrite!("band-bxor-self-right"; "(band ?x (bxor ?x ?y))" => "(band ?x (bnot ?y))"),
        rewrite!("band-bxor-self-left"; "(band (bxor ?x ?y) ?x)" => "(band ?x (bnot ?y))"),
        // Rust-only improvement: factor shared masks out of OR/XOR to shrink DAG size.
        rewrite!("bor-factor-shared-mask"; "(bor (band ?x ?m) (band ?y ?m))" => "(band (bor ?x ?y) ?m)"),
        rewrite!("bor-factor-shared-mask-comm"; "(bor (band ?m ?x) (band ?m ?y))" => "(band (bor ?x ?y) ?m)"),
        rewrite!("bxor-factor-shared-mask"; "(bxor (band ?x ?m) (band ?y ?m))" => "(band (bxor ?x ?y) ?m)"),
        rewrite!("bxor-factor-shared-mask-comm"; "(bxor (band ?m ?x) (band ?m ?y))" => "(band (bxor ?x ?y) ?m)"),
        // Rust-only improvement: merge distinct masks on the same value.
        rewrite!("bor-merge-masked"; "(bor (band ?x ?m) (band ?x ?n))" => "(band ?x (bor ?m ?n))"),
        rewrite!("bor-merge-masked-comm"; "(bor (band ?m ?x) (band ?n ?x))" => "(band ?x (bor ?m ?n))"),
        rewrite!("bxor-merge-masked"; "(bxor (band ?x ?m) (band ?x ?n))" => "(band ?x (bxor ?m ?n))"),
        rewrite!("bxor-merge-masked-comm"; "(bxor (band ?m ?x) (band ?n ?x))" => "(band ?x (bxor ?m ?n))"),
        // Rust-only improvement: collapse nested masks to a single AND.
        rewrite!("band-merge-nested"; "(band (band ?x ?m) ?n)" => "(band ?x (band ?m ?n))"),
        rewrite!("band-merge-nested-comm-left"; "(band ?n (band ?x ?m))" => "(band ?x (band ?m ?n))"),
        rewrite!("band-merge-nested-comm-right"; "(band (band ?m ?x) ?n)" => "(band ?x (band ?m ?n))"),
        // Rust-only improvement: consensus theorem for shared OR terms.
        rewrite!("band-consensus-or"; "(band (bor ?x ?y) (bor ?x ?z))" => "(bor ?x (band ?y ?z))"),
        rewrite!("band-consensus-or-comm"; "(band (bor ?y ?x) (bor ?z ?x))" => "(bor ?x (band ?y ?z))"),
        rewrite!("band-redundant-or-const"; "(band ?m (bor ?x ?c))" => {
            BandRedundantOr { mask: var("?m"), x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-redundant-or-const-comm"; "(band (bor ?x ?c) ?m)" => {
            BandRedundantOr { mask: var("?m"), x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-redundant-xor-const"; "(band ?m (bxor ?x ?c))" => {
            BandRedundantXor { mask: var("?m"), x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-redundant-xor-const-comm"; "(band (bxor ?x ?c) ?m)" => {
            BandRedundantXor { mask: var("?m"), x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-redundant-add-const"; "(band ?m (+ ?x ?c))" => {
            BandRedundantAdd { mask: var("?m"), x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-redundant-add-const-comm"; "(band (+ ?x ?c) ?m)" => {
            BandRedundantAdd { mask: var("?m"), x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-redundant-sub-const"; "(band ?m (- ?x ?c))" => {
            BandRedundantSub { mask: var("?m"), x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-redundant-sub-const-comm"; "(band (- ?x ?c) ?m)" => {
            BandRedundantSub { mask: var("?m"), x: var("?x"), c: var("?c") }
        }),
        rewrite!("band-redundant-shl"; "(band ?m (shl ?x ?c))" => {
            BandRedundantShift { mask: var("?m"), shift: var("?c"), kind: ShiftKind::Left }
        }),
        rewrite!("band-redundant-shl-comm"; "(band (shl ?x ?c) ?m)" => {
            BandRedundantShift { mask: var("?m"), shift: var("?c"), kind: ShiftKind::Left }
        }),
        rewrite!("band-redundant-shr"; "(band ?m (shr_u ?x ?c))" => {
            BandRedundantShift { mask: var("?m"), shift: var("?c"), kind: ShiftKind::RightUnsigned }
        }),
        rewrite!("band-redundant-shr-comm"; "(band (shr_u ?x ?c) ?m)" => {
            BandRedundantShift { mask: var("?m"), shift: var("?c"), kind: ShiftKind::RightUnsigned }
        }),
        rewrite!("bxor-diff-masked-right"; "(bxor ?x (band ?x ?y))" => "(band ?x (bnot ?y))"),
        rewrite!("bxor-diff-masked-left"; "(bxor (band ?x ?y) ?x)" => "(band ?x (bnot ?y))"),
        rewrite!("bxor-diff-masked-or-right"; "(bxor ?x (bor ?x ?y))" => "(band ?y (bnot ?x))"),
        rewrite!("bxor-diff-masked-or-left"; "(bxor (bor ?x ?y) ?x)" => "(band ?y (bnot ?x))"),
        rewrite!("bxor-bnot-bnot"; "(bxor (bnot ?x) (bnot ?y))" => "(bxor ?x ?y)"),
        rewrite!("bxor-comm"; "(bxor ?a ?b)" => "(bxor ?b ?a)"),
        rewrite!("bxor-assoc"; "(bxor ?a (bxor ?b ?c))" => "(bxor (bxor ?a ?b) ?c)"),
        rewrite!("bxor-reassociate-const-right"; "(bxor ?c1 (bxor ?x ?c2))" => {
            BitwiseConstReassociate {
                op: BitwiseOp::Xor,
                x: var("?x"),
                c1: var("?c1"),
                c2: var("?c2"),
            }
        }),
        rewrite!("bxor-reassociate-const-left"; "(bxor (bxor ?x ?c2) ?c1)" => {
            BitwiseConstReassociate {
                op: BitwiseOp::Xor,
                x: var("?x"),
                c1: var("?c1"),
                c2: var("?c2"),
            }
        }),
        rewrite!("bxor-const-fold"; "(bxor ?a ?b)" => { BitXorFold { a: var("?a"), b: var("?b") } }),
        rewrite!("bxor-zero-left"; "(bxor ?x ?c)" => {
            BitXorConstSimplify { x: var("?x"), c: var("?c") }
        }),
        rewrite!("bxor-zero-right"; "(bxor ?c ?x)" => {
            BitXorConstSimplify { x: var("?x"), c: var("?c") }
        }),
        rewrite!("bxor-complement-left"; "(bxor ?x (bnot ?x))" => {
            BitXorComplement { _x: var("?x") }
        }),
        rewrite!("bxor-complement-right"; "(bxor (bnot ?x) ?x)" => {
            BitXorComplement { _x: var("?x") }
        }),
        rewrite!("bxor-self"; "(bxor ?x ?x)" => { BitXorSelf { _x: var("?x") } }),
        rewrite!("bor-absorbs-band"; "(bor ?x (band ?x ?y))" => "?x"),
        rewrite!("bor-absorbs-band-commuted"; "(bor (band ?x ?y) ?x)" => "?x"),
        rewrite!("sub-mask-absorb"; "(- (band ?x ?mask) ?x)" => { SubMaskAbsorb { x: var("?x"), mask: var("?mask") } }),
        rewrite!("bxor-absorbs-and"; "(bxor ?x (band ?x ?mask))" => {
            BitXorAbsorb { x: var("?x"), mask: var("?mask") }
        }),
        rewrite!("bxor-zero-when-masked-all-ones"; "(bxor ?x (band ?x ?mask))" => {
            BitXorAllOnes { x: var("?x"), mask: var("?mask") }
        }),
        rewrite!("or-with-zero-band"; "(bor ?x (band ?y 0))" => "?x"),
        rewrite!("bnot-const-fold"; "(bnot ?x)" => { BitNotFold { x: var("?x") } }),
        rewrite!("bnot-double"; "(bnot (bnot ?x))" => { BitNotDouble { x: var("?x") } }),
        rewrite!("brev-const-fold"; "(brev ?x)" => { BitReverseFold { x: var("?x") } }),
        rewrite!("rotate-const-pattern"; "(bor (shl ?x ?s) (shr_u ?x ?t))" => {
            RotatePatternFold { x: var("?x"), s: var("?s"), t: var("?t") }
        }),
        rewrite!("rotate-const-pattern-comm"; "(bor (shr_u ?x ?t) (shl ?x ?s))" => {
            RotatePatternFold { x: var("?x"), s: var("?s"), t: var("?t") }
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
        rewrite!("smod-mul-const-zero-left"; "(smod (* ?c ?x) ?c)" => {
            RemMulConstZero { c: var("?c") }
        }),
        rewrite!("smod-mul-const-zero-right"; "(smod (* ?x ?c) ?c)" => {
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
        rewrite!("add-factor-const-with-one-left"; "(+ (* ?x ?c) ?x)" => {
            AddFactorWithOne { x: var("?x"), c: var("?c") }
        }),
        rewrite!("add-factor-const-with-one-right"; "(+ ?x (* ?x ?c))" => {
            AddFactorWithOne { x: var("?x"), c: var("?c") }
        }),
        rewrite!("add-factor-const-with-one-mixed-left"; "(+ (* ?c ?x) ?x)" => {
            AddFactorWithOne { x: var("?x"), c: var("?c") }
        }),
        rewrite!("add-factor-const-with-one-mixed-right"; "(+ ?x (* ?c ?x))" => {
            AddFactorWithOne { x: var("?x"), c: var("?c") }
        }),
        rewrite!("sub-factor-const-with-one-left"; "(- (* ?x ?c) ?x)" => {
            SubFactorWithOne { x: var("?x"), c: var("?c"), lhs_mul: true }
        }),
        rewrite!("sub-factor-const-with-one-right"; "(- ?x (* ?x ?c))" => {
            SubFactorWithOne { x: var("?x"), c: var("?c"), lhs_mul: false }
        }),
        rewrite!("sub-factor-const-with-one-mixed-left"; "(- (* ?c ?x) ?x)" => {
            SubFactorWithOne { x: var("?x"), c: var("?c"), lhs_mul: true }
        }),
        rewrite!("sub-factor-const-with-one-mixed-right"; "(- ?x (* ?c ?x))" => {
            SubFactorWithOne { x: var("?x"), c: var("?c"), lhs_mul: false }
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
        // Rust-only improvement: adding complementary mask splits reconstructs the original value.
        rewrite!("add-mask-split-x"; "(+ (band ?x ?m) (band ?x (bnot ?m)))" => "?x"),
        rewrite!("add-mask-split-x-mixed-left"; "(+ (band ?x ?m) (band (bnot ?m) ?x))" => "?x"),
        rewrite!("add-mask-split-x-mixed-right"; "(+ (band ?m ?x) (band ?x (bnot ?m)))" => "?x"),
        rewrite!("add-mask-split-x-comm"; "(+ (band ?m ?x) (band (bnot ?m) ?x))" => "?x"),
        // Rust-only improvement: adding split x under a shared mask yields the mask.
        rewrite!("add-mask-split-mask"; "(+ (band ?x ?m) (band (bnot ?x) ?m))" => "?m"),
        rewrite!("add-mask-split-mask-mixed-left"; "(+ (band ?x ?m) (band ?m (bnot ?x)))" => "?m"),
        rewrite!("add-mask-split-mask-mixed-right"; "(+ (band ?m ?x) (band (bnot ?x) ?m))" => "?m"),
        rewrite!("add-mask-split-mask-comm"; "(+ (band ?m ?x) (band ?m (bnot ?x)))" => "?m"),
        rewrite!("neg-sub-swap"; "(neg (- ?a ?b))" => "(- ?b ?a)"),
        rewrite!("neg-add-const"; "(neg (+ ?a ?b))" => { NegAddConst { a: var("?a"), b: var("?b") } }),
        rewrite!("neg-sdiv-const"; "(neg (sdiv ?a ?b))" => { NegDivConst { a: var("?a"), b: var("?b") } }),
        rewrite!("neg-fold"; "(neg ?a)" => { FoldNeg }),
        rewrite!("double-neg"; "(neg (neg ?a))" => "?a"),
        rewrite!("sdiv-fold"; "(sdiv ?a ?b)" => { FoldDiv { signed: true } }),
        rewrite!("udiv-fold"; "(udiv ?a ?b)" => { FoldDiv { signed: false } }),
        rewrite!("sdiv-one"; "(sdiv ?a ?b)" => { DivOne { a: var("?a"), b: var("?b") } }),
        rewrite!("udiv-one"; "(udiv ?a ?b)" => { DivOne { a: var("?a"), b: var("?b") } }),
        rewrite!("srem-fold"; "(srem ?a ?b)" => { FoldRem { signed: true } }),
        rewrite!("smod-fold"; "(smod ?a ?b)" => { FoldSMod { a: var("?a"), b: var("?b") } }),
        rewrite!("umod-fold"; "(umod ?a ?b)" => { FoldRem { signed: false } }),
        rewrite!("srem-one"; "(srem ?a ?b)" => { RemOne { b: var("?b") } }),
        rewrite!("smod-one"; "(smod ?a ?b)" => { RemOne { b: var("?b") } }),
        rewrite!("umod-one"; "(umod ?a ?b)" => { RemOne { b: var("?b") } }),
        rewrite!("add-neg-cancel"; "(+ ?a (neg ?a))" => { AddNegZero }),
        rewrite!("add-neg-cancel-swap"; "(+ (neg ?a) ?a)" => { AddNegZero }),
        rewrite!("eq-fold"; "(eq ?a ?b)" => {
            FoldCmp { kind: CmpKind::Eq, a: var("?a"), b: var("?b") }
        }),
        rewrite!("ne-fold"; "(ne ?a ?b)" => {
            FoldCmp { kind: CmpKind::Ne, a: var("?a"), b: var("?b") }
        }),
        rewrite!("slt-fold"; "(slt ?a ?b)" => {
            FoldCmp { kind: CmpKind::SLt, a: var("?a"), b: var("?b") }
        }),
        rewrite!("sle-fold"; "(sle ?a ?b)" => {
            FoldCmp { kind: CmpKind::SLe, a: var("?a"), b: var("?b") }
        }),
        rewrite!("sgt-fold"; "(sgt ?a ?b)" => {
            FoldCmp { kind: CmpKind::SGt, a: var("?a"), b: var("?b") }
        }),
        rewrite!("sge-fold"; "(sge ?a ?b)" => {
            FoldCmp { kind: CmpKind::SGe, a: var("?a"), b: var("?b") }
        }),
        rewrite!("ult-fold"; "(ult ?a ?b)" => {
            FoldCmp { kind: CmpKind::ULt, a: var("?a"), b: var("?b") }
        }),
        rewrite!("ule-fold"; "(ule ?a ?b)" => {
            FoldCmp { kind: CmpKind::ULe, a: var("?a"), b: var("?b") }
        }),
        rewrite!("ugt-fold"; "(ugt ?a ?b)" => {
            FoldCmp { kind: CmpKind::UGt, a: var("?a"), b: var("?b") }
        }),
        rewrite!("uge-fold"; "(uge ?a ?b)" => {
            FoldCmp { kind: CmpKind::UGe, a: var("?a"), b: var("?b") }
        }),
        rewrite!("lognot-fold"; "(lnot ?a)" => { FoldLogicalNot { a: var("?a") } }),
        rewrite!(
            "logand-fold";
            "(land ?a ?b)" => { FoldLogical { kind: LogicalKind::And, a: var("?a"), b: var("?b") } }
        ),
        rewrite!(
            "logor-fold";
            "(lor ?a ?b)" => { FoldLogical { kind: LogicalKind::Or, a: var("?a"), b: var("?b") } }
        ),
        rewrite!("logeq-fold"; "(leq ?a ?b)" => {
            FoldCmp { kind: CmpKind::LogEq, a: var("?a"), b: var("?b") }
        }),
        rewrite!("logne-fold"; "(lne ?a ?b)" => {
            FoldCmp { kind: CmpKind::LogNe, a: var("?a"), b: var("?b") }
        }),
        rewrite!("eq-comm"; "(eq ?a ?b)" => "(eq ?b ?a)"),
        rewrite!("ne-comm"; "(ne ?a ?b)" => "(ne ?b ?a)"),
        rewrite!("logand-comm"; "(land ?a ?b)" => "(land ?b ?a)"),
        rewrite!("logor-comm"; "(lor ?a ?b)" => "(lor ?b ?a)"),
        rewrite!("logeq-comm"; "(leq ?a ?b)" => "(leq ?b ?a)"),
        rewrite!("logne-comm"; "(lne ?a ?b)" => "(lne ?b ?a)"),
        rewrite!("eq-self"; "(eq ?a ?a)" => { BoolConst { value: true } }),
        rewrite!("ne-self"; "(ne ?a ?a)" => { BoolConst { value: false } }),
        rewrite!("logeq-not-not"; "(leq (lnot ?a) (lnot ?b))" => "(leq ?a ?b)"),
        rewrite!("logne-not-not"; "(lne (lnot ?a) (lnot ?b))" => "(lne ?a ?b)"),
        rewrite!("eq-neg-neg"; "(eq (neg ?a) (neg ?b))" => "(eq ?a ?b)"),
        rewrite!("ne-neg-neg"; "(ne (neg ?a) (neg ?b))" => "(ne ?a ?b)"),
        rewrite!("eq-bnot-bnot"; "(eq (bnot ?a) (bnot ?b))" => "(eq ?a ?b)"),
        rewrite!("ne-bnot-bnot"; "(ne (bnot ?a) (bnot ?b))" => "(ne ?a ?b)"),
        rewrite!("eq-add-cancel-left"; "(eq (+ ?x ?y) (+ ?x ?z))" => "(eq ?y ?z)"),
        rewrite!("ne-add-cancel-left"; "(ne (+ ?x ?y) (+ ?x ?z))" => "(ne ?y ?z)"),
        rewrite!("eq-sub-cancel-left"; "(eq (- ?x ?y) (- ?x ?z))" => "(eq ?y ?z)"),
        rewrite!("ne-sub-cancel-left"; "(ne (- ?x ?y) (- ?x ?z))" => "(ne ?y ?z)"),
        rewrite!("eq-sub-cancel-right"; "(eq (- ?y ?x) (- ?z ?x))" => "(eq ?y ?z)"),
        rewrite!("ne-sub-cancel-right"; "(ne (- ?y ?x) (- ?z ?x))" => "(ne ?y ?z)"),
        rewrite!("eq-bxor-cancel-left"; "(eq (bxor ?x ?y) (bxor ?x ?z))" => "(eq ?y ?z)"),
        rewrite!("ne-bxor-cancel-left"; "(ne (bxor ?x ?y) (bxor ?x ?z))" => "(ne ?y ?z)"),
        rewrite!("eq-bxor-zero"; "(eq (bxor ?x ?y) ?c)" => "(eq ?x ?y)" if is_const_zero(var("?c"))),
        rewrite!("ne-bxor-zero"; "(ne (bxor ?x ?y) ?c)" => "(ne ?x ?y)" if is_const_zero(var("?c"))),
        rewrite!("eq-sub-zero"; "(eq (- ?x ?y) ?c)" => "(eq ?x ?y)" if is_const_zero(var("?c"))),
        rewrite!("ne-sub-zero"; "(ne (- ?x ?y) ?c)" => "(ne ?x ?y)" if is_const_zero(var("?c"))),
        rewrite!("eq-neg-zero"; "(eq (neg ?x) ?c)" => "(eq ?x ?c)" if is_const_zero(var("?c"))),
        rewrite!("ne-neg-zero"; "(ne (neg ?x) ?c)" => "(ne ?x ?c)" if is_const_zero(var("?c"))),
        rewrite!("eq-bxor-all-ones"; "(eq (bxor ?x ?y) ?c)" => "(eq ?x (bnot ?y))" if is_const_all_ones(var("?c"))),
        rewrite!("ne-bxor-all-ones"; "(ne (bxor ?x ?y) ?c)" => "(ne ?x (bnot ?y))" if is_const_all_ones(var("?c"))),
        rewrite!("eq-neg-const"; "(eq (neg ?x) ?c)" => { CmpNegConst { x: var("?x"), c: var("?c"), eq: true } }),
        rewrite!("ne-neg-const"; "(ne (neg ?x) ?c)" => { CmpNegConst { x: var("?x"), c: var("?c"), eq: false } }),
        rewrite!("eq-bnot-const"; "(eq (bnot ?x) ?c)" => { CmpBNotConst { x: var("?x"), c: var("?c"), eq: true } }),
        rewrite!("ne-bnot-const"; "(ne (bnot ?x) ?c)" => { CmpBNotConst { x: var("?x"), c: var("?c"), eq: false } }),
        rewrite!("eq-bxor-const"; "(eq (bxor ?x ?c1) ?c2)" => {
            CmpXorConst { x: var("?x"), c1: var("?c1"), c2: var("?c2"), eq: true }
        }),
        rewrite!("ne-bxor-const"; "(ne (bxor ?x ?c1) ?c2)" => {
            CmpXorConst { x: var("?x"), c1: var("?c1"), c2: var("?c2"), eq: false }
        }),
        rewrite!("slt-self"; "(slt ?a ?a)" => { BoolConst { value: false } }),
        rewrite!("sle-self"; "(sle ?a ?a)" => { BoolConst { value: true } }),
        rewrite!("sgt-self"; "(sgt ?a ?a)" => { BoolConst { value: false } }),
        rewrite!("sge-self"; "(sge ?a ?a)" => { BoolConst { value: true } }),
        rewrite!("ult-self"; "(ult ?a ?a)" => { BoolConst { value: false } }),
        rewrite!("ule-self"; "(ule ?a ?a)" => { BoolConst { value: true } }),
        rewrite!("ugt-self"; "(ugt ?a ?a)" => { BoolConst { value: false } }),
        rewrite!("uge-self"; "(uge ?a ?a)" => { BoolConst { value: true } }),
        rewrite!("logeq-self"; "(leq ?a ?a)" => { BoolConst { value: true } }),
        rewrite!("logne-self"; "(lne ?a ?a)" => { BoolConst { value: false } }),
        rewrite!("logand-neg"; "(land ?a (lnot ?a))" => { BoolConst { value: false } }),
        rewrite!("logand-neg-comm"; "(land (lnot ?a) ?a)" => { BoolConst { value: false } }),
        rewrite!("logor-neg"; "(lor ?a (lnot ?a))" => { BoolConst { value: true } }),
        rewrite!("logor-neg-comm"; "(lor (lnot ?a) ?a)" => { BoolConst { value: true } }),
        rewrite!("logand-demorgan"; "(land (lnot ?a) (lnot ?b))" => "(lnot (lor ?a ?b))"),
        rewrite!("logor-demorgan"; "(lor (lnot ?a) (lnot ?b))" => "(lnot (land ?a ?b))"),
        rewrite!("lognot-eq"; "(lnot (eq ?a ?b))" => "(ne ?a ?b)"),
        rewrite!("lognot-ne"; "(lnot (ne ?a ?b))" => "(eq ?a ?b)"),
        rewrite!("lognot-slt"; "(lnot (slt ?a ?b))" => "(sge ?a ?b)"),
        rewrite!("lognot-sle"; "(lnot (sle ?a ?b))" => "(sgt ?a ?b)"),
        rewrite!("lognot-sgt"; "(lnot (sgt ?a ?b))" => "(sle ?a ?b)"),
        rewrite!("lognot-sge"; "(lnot (sge ?a ?b))" => "(slt ?a ?b)"),
        rewrite!("lognot-ult"; "(lnot (ult ?a ?b))" => "(uge ?a ?b)"),
        rewrite!("lognot-ule"; "(lnot (ule ?a ?b))" => "(ugt ?a ?b)"),
        rewrite!("lognot-ugt"; "(lnot (ugt ?a ?b))" => "(ule ?a ?b)"),
        rewrite!("lognot-uge"; "(lnot (uge ?a ?b))" => "(ult ?a ?b)"),
        rewrite!("lognot-logeq"; "(lnot (leq ?a ?b))" => "(lne ?a ?b)"),
        rewrite!("lognot-logne"; "(lnot (lne ?a ?b))" => "(leq ?a ?b)"),
        rewrite!("lognot-double"; "(lnot (lnot ?a))" => "?a"),
        rewrite!("logand-idem"; "(land ?a ?a)" => "?a"),
        rewrite!("logor-idem"; "(lor ?a ?a)" => "?a"),
        rewrite!("select-same"; "(select ?c ?a ?a)" => "?a"),
        rewrite!("select-neg-cond"; "(select (lnot ?c) ?t ?f)" => "(select ?c ?f ?t)"),
        rewrite!(
            "select-const";
            "(select ?c ?t ?f)" => { SelectConstCond { cond: var("?c"), t: var("?t"), f: var("?f") } }
        ),
        rewrite!(
            "select-bool-arms";
            "(select ?c ?t ?f)" => { SelectBoolArms { cond: var("?c"), t: var("?t"), f: var("?f") } }
        ),
        rewrite!("phi-same"; "(phi ?a ?a)" => "?a"),
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
struct FoldSMod {
    a: Var,
    b: Var,
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
struct AddFactorWithOne {
    x: Var,
    c: Var,
}
struct SubFactorWithOne {
    x: Var,
    c: Var,
    lhs_mul: bool,
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
struct FactorSharedAnyOrder {
    factor: Var,
    lhs: Var,
    rhs: Var,
    subtract: bool,
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
enum BitwiseOp {
    Or,
    Xor,
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
struct BitAndComplement {
    _x: Var,
}
struct BitXorComplement {
    _x: Var,
}
struct UModPowerOfTwo {
    x: Var,
    c: Var,
}
struct UModPowerOfTwoMask {
    x: Var,
    c: Var,
}
struct BitOrFold {
    a: Var,
    b: Var,
}
struct BitOrConstSimplify {
    x: Var,
    c: Var,
}
struct BitwiseConstReassociate {
    op: BitwiseOp,
    x: Var,
    c1: Var,
    c2: Var,
}
struct BitOrComplement {
    _x: Var,
}
struct BitXorFold {
    a: Var,
    b: Var,
}
struct BitXorConstSimplify {
    x: Var,
    c: Var,
}
struct BitXorSelf {
    _x: Var,
}
struct SubMaskAbsorb {
    x: Var,
    mask: Var,
}
struct BitXorAbsorb {
    x: Var,
    mask: Var,
}
struct BitXorAllOnes {
    x: Var,
    mask: Var,
}
struct BandRedundantOr {
    mask: Var,
    x: Var,
    c: Var,
}
struct BandRedundantXor {
    mask: Var,
    x: Var,
    c: Var,
}
struct BandRedundantAdd {
    mask: Var,
    x: Var,
    c: Var,
}
struct BandRedundantSub {
    mask: Var,
    x: Var,
    c: Var,
}
struct BandRedundantShift {
    mask: Var,
    shift: Var,
    kind: ShiftKind,
}
struct BitNotFold {
    x: Var,
}
struct BitNotDouble {
    x: Var,
}
struct BitReverseFold {
    x: Var,
}
struct RotatePatternFold {
    x: Var,
    s: Var,
    t: Var,
}
struct MaskThenShift {
    x: Var,
    mask: Var,
    shift: Var,
}
struct MaskThenShiftSigned {
    x: Var,
    mask: Var,
    shift: Var,
}
struct MergeShift {
    x: Var,
    a: Var,
    b: Var,
    kind: ShiftKind,
}
struct ShlFoldConst {
    a: Var,
    b: Var,
}
struct ShrUFoldConst {
    a: Var,
    b: Var,
}
struct ShrSFoldConst {
    a: Var,
    b: Var,
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
struct NegAddConst {
    a: Var,
    b: Var,
}
struct NegDivConst {
    a: Var,
    b: Var,
}
struct CmpNegConst {
    x: Var,
    c: Var,
    eq: bool,
}
struct CmpBNotConst {
    x: Var,
    c: Var,
    eq: bool,
}
struct CmpXorConst {
    x: Var,
    c1: Var,
    c2: Var,
    eq: bool,
}
struct SelectConstCond {
    cond: Var,
    t: Var,
    f: Var,
}
struct SelectBoolArms {
    cond: Var,
    t: Var,
    f: Var,
}
#[derive(Clone, Copy)]
enum CmpKind {
    Eq,
    Ne,
    SLt,
    SLe,
    SGt,
    SGe,
    ULt,
    ULe,
    UGt,
    UGe,
    LogEq,
    LogNe,
}
#[derive(Clone, Copy)]
enum LogicalKind {
    And,
    Or,
}
struct FoldCmp {
    kind: CmpKind,
    a: Var,
    b: Var,
}
struct FoldLogical {
    kind: LogicalKind,
    a: Var,
    b: Var,
}
struct FoldLogicalNot {
    a: Var,
}
struct BoolConst {
    value: bool,
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
        let sum = combine_consts(a, b, u64::wrapping_add);
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
        let product = combine_consts(a, b, u64::wrapping_mul);
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
        let diff = combine_consts(a, b, u64::wrapping_sub);
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
        let negated = map_const(a, u64::wrapping_neg);
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
        if b.get_u64() == 0 {
            return Vec::new();
        }
        let quotient = if self.signed {
            combine_signed_consts(a, b, i128::wrapping_div)
        } else {
            combine_consts(a, b, u64::wrapping_div)
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
        if b.get_u64() == 0 {
            return Vec::new();
        }
        let rem = if self.signed {
            combine_signed_consts(a, b, i128::wrapping_rem)
        } else {
            combine_consts(a, b, u64::wrapping_rem)
        };
        let id = egraph.add(SpirvLang::Const(rem));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for FoldSMod {
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
        if b.get_u64() == 0 {
            return Vec::new();
        }
        let rem = combine_signed_mod(a, b);
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
        if const_value(egraph, subst[self.b]).is_some_and(|c| c.is_one()) {
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
        if let Some(constant) = const_value(egraph, subst[self.b]) {
            if constant.is_one() {
                let id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
                    0,
                    constant.width_bits(),
                )));
                egraph.union(eclass, id);
                return vec![id];
            }
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for SubSelf {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(width) = width_hint(egraph, eclass, [subst[var("a")]]) else {
            return Vec::new();
        };
        let const_zero = egraph.add(SpirvLang::Const(ConstValue::new_with_width(0, width)));
        egraph.union(eclass, const_zero);
        vec![const_zero]
    }
}

impl Applier<SpirvLang, ()> for AddNegZero {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(width) = width_hint(egraph, eclass, [subst[var("a")]]) else {
            return Vec::new();
        };
        let const_zero = egraph.add(SpirvLang::Const(ConstValue::new_with_width(0, width)));
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
        if const_value(egraph, subst[self.a]).is_some_and(|c| c.is_zero()) {
            egraph.union(eclass, subst[self.b]);
            return vec![subst[self.b]];
        }
        if const_value(egraph, subst[self.b]).is_some_and(|c| c.is_zero()) {
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
        if const_value(egraph, subst[self.a]).is_some_and(|c| c.is_one()) {
            egraph.union(eclass, subst[self.b]);
            return vec![subst[self.b]];
        }
        if const_value(egraph, subst[self.b]).is_some_and(|c| c.is_one()) {
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
        let left_const = const_value(egraph, subst[self.a]);
        let right_const = const_value(egraph, subst[self.b]);
        let zero_left = left_const.is_some_and(|c| c.is_zero());
        let zero_right = right_const.is_some_and(|c| c.is_zero());
        if zero_left || zero_right {
            let width = left_const
                .or(right_const)
                .map(|c| c.width_bits())
                .or_else(|| width_hint(egraph, eclass, [subst[self.a], subst[self.b]]))
                .unwrap_or(32);
            let id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(0, width)));
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
        if const_value(egraph, subst[var("a")]).is_some_and(|c| c.is_zero()) {
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
        let left_is_neg_one = const_value(egraph, subst[self.a]).is_some_and(|c| c.is_all_ones());
        let right_is_neg_one = const_value(egraph, subst[self.b]).is_some_and(|c| c.is_all_ones());
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
        let merged = combine_consts(lhs, rhs, u64::wrapping_add);
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
        let merged = combine_consts(add_const, sub_const, u64::wrapping_sub);
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
        let merged = combine_consts(c1, c2, u64::wrapping_sub);
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
        let merged = combine_consts(c1, c2, u64::wrapping_add);
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
        if c1 != c2 {
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

impl Applier<SpirvLang, ()> for FactorSharedAnyOrder {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let inner = if self.subtract {
            egraph.add(SpirvLang::Sub([subst[self.lhs], subst[self.rhs]]))
        } else {
            egraph.add(SpirvLang::Add([subst[self.lhs], subst[self.rhs]]))
        };
        let mul = egraph.add(SpirvLang::Mul([subst[self.factor], inner]));
        egraph.union(eclass, mul);
        vec![mul]
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
        let merged = combine_consts(c1, c2, u64::wrapping_sub);
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
        let merged = combine_consts(lhs_const, rhs_const, u64::wrapping_add);
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
        if c1.get_u64() == 0 || c2.get_u64() == 0 {
            return Vec::new();
        }
        let merged = if self.signed {
            combine_signed_consts(c1, c2, i128::wrapping_mul)
        } else {
            combine_consts(c1, c2, u64::wrapping_mul)
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
        let merged = combine_consts(c1, c2, u64::wrapping_add);
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

impl Applier<SpirvLang, ()> for AddFactorWithOne {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(c) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let merged = map_const(c, |v| v.wrapping_add(1));
        let const_id = egraph.add(SpirvLang::Const(merged));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], const_id]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for SubFactorWithOne {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(c) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let merged = if self.lhs_mul {
            map_const(c, |v| v.wrapping_sub(1))
        } else {
            ConstValue::new_with_width(1u64.wrapping_sub(c.get_u64()), c.width_bits().max(1))
        };
        let const_id = egraph.add(SpirvLang::Const(merged));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], const_id]));
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
        let merged = combine_consts(c1, c2, u64::wrapping_sub);
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
        if constant.get_u64() == 0 {
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
        if constant.get_u64() == 0 {
            return Vec::new();
        }
        let zero = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            0,
            constant.width_bits(),
        )));
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
        let negated = map_const(constant, u64::wrapping_neg);
        let const_id = egraph.add(SpirvLang::Const(negated));
        let mul = egraph.add(SpirvLang::Mul([subst[self.x], const_id]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

impl Applier<SpirvLang, ()> for NegAddConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let left_const = const_value(egraph, subst[self.a]);
        let right_const = const_value(egraph, subst[self.b]);
        let (constant, other) = match (left_const, right_const) {
            (Some(c), None) => (c, subst[self.b]),
            (None, Some(c)) => (c, subst[self.a]),
            _ => return Vec::new(),
        };
        let negated = map_const(constant, u64::wrapping_neg);
        let const_id = egraph.add(SpirvLang::Const(negated));
        let sub = egraph.add(SpirvLang::Sub([const_id, other]));
        egraph.union(eclass, sub);
        vec![sub]
    }
}

impl Applier<SpirvLang, ()> for NegDivConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let left_const = const_value(egraph, subst[self.a]);
        let right_const = const_value(egraph, subst[self.b]);
        let (constant, left, right) = match (left_const, right_const) {
            (Some(c), None) => (c, None, Some(subst[self.b])),
            (None, Some(c)) => (c, Some(subst[self.a]), None),
            _ => return Vec::new(),
        };
        let negated = map_const(constant, u64::wrapping_neg);
        let const_id = egraph.add(SpirvLang::Const(negated));
        let div = match (left, right) {
            (Some(lhs), None) => egraph.add(SpirvLang::SDiv([lhs, const_id])),
            (None, Some(rhs)) => egraph.add(SpirvLang::SDiv([const_id, rhs])),
            _ => return Vec::new(),
        };
        egraph.union(eclass, div);
        vec![div]
    }
}

impl Applier<SpirvLang, ()> for CmpNegConst {
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
        let negated = map_const(constant, u64::wrapping_neg);
        let const_id = egraph.add(SpirvLang::Const(negated));
        let cmp = if self.eq {
            egraph.add(SpirvLang::Eq([subst[self.x], const_id]))
        } else {
            egraph.add(SpirvLang::Ne([subst[self.x], const_id]))
        };
        egraph.union(eclass, cmp);
        vec![cmp]
    }
}

impl Applier<SpirvLang, ()> for CmpBNotConst {
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
        let inverted = map_const(constant, |value| !value);
        let const_id = egraph.add(SpirvLang::Const(inverted));
        let cmp = if self.eq {
            egraph.add(SpirvLang::Eq([subst[self.x], const_id]))
        } else {
            egraph.add(SpirvLang::Ne([subst[self.x], const_id]))
        };
        egraph.union(eclass, cmp);
        vec![cmp]
    }
}

impl Applier<SpirvLang, ()> for CmpXorConst {
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
        let merged = combine_consts(c1, c2, |a, b| a ^ b);
        let const_id = egraph.add(SpirvLang::Const(merged));
        let cmp = if self.eq {
            egraph.add(SpirvLang::Eq([subst[self.x], const_id]))
        } else {
            egraph.add(SpirvLang::Ne([subst[self.x], const_id]))
        };
        egraph.union(eclass, cmp);
        vec![cmp]
    }
}

impl Applier<SpirvLang, ()> for SelectConstCond {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(condition) = const_value(egraph, subst[self.cond]) else {
            return Vec::new();
        };
        let Some(cond) = bool_const(condition) else {
            return Vec::new();
        };
        let chosen = if cond { subst[self.t] } else { subst[self.f] };
        egraph.union(eclass, chosen);
        vec![chosen]
    }
}

impl Applier<SpirvLang, ()> for SelectBoolArms {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(true_arm) = const_value(egraph, subst[self.t]).and_then(bool_const) else {
            return Vec::new();
        };
        let Some(false_arm) = const_value(egraph, subst[self.f]).and_then(bool_const) else {
            return Vec::new();
        };
        let cond = subst[self.cond];
        let rewritten = match (true_arm, false_arm) {
            (true, false) => cond,
            (false, true) => egraph.add(SpirvLang::LogNot(cond)),
            _ => return Vec::new(),
        };
        egraph.union(eclass, rewritten);
        vec![rewritten]
    }
}

impl Applier<SpirvLang, ()> for BoolConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        _subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let id = egraph.add(SpirvLang::Const(const_bool(self.value)));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for FoldCmp {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let lhs = const_value(egraph, subst[self.a]);
        let rhs = const_value(egraph, subst[self.b]);
        match self.kind {
            CmpKind::LogEq | CmpKind::LogNe => {
                let lhs_bool = lhs.and_then(bool_const);
                let rhs_bool = rhs.and_then(bool_const);
                match (lhs_bool, rhs_bool) {
                    (Some(a), Some(b)) => {
                        let result = match self.kind {
                            CmpKind::LogEq => a == b,
                            CmpKind::LogNe => a != b,
                            _ => unreachable!(),
                        };
                        let id = egraph.add(SpirvLang::Const(const_bool(result)));
                        egraph.union(eclass, id);
                        vec![id]
                    }
                    (Some(value), None) => {
                        let target = match (self.kind, value) {
                            (CmpKind::LogEq, true) | (CmpKind::LogNe, false) => subst[self.b],
                            (CmpKind::LogEq, false) | (CmpKind::LogNe, true) => {
                                egraph.add(SpirvLang::LogNot(subst[self.b]))
                            }
                            _ => subst[self.b],
                        };
                        egraph.union(eclass, target);
                        vec![target]
                    }
                    (None, Some(value)) => {
                        let target = match (self.kind, value) {
                            (CmpKind::LogEq, true) | (CmpKind::LogNe, false) => subst[self.a],
                            (CmpKind::LogEq, false) | (CmpKind::LogNe, true) => {
                                egraph.add(SpirvLang::LogNot(subst[self.a]))
                            }
                            _ => subst[self.a],
                        };
                        egraph.union(eclass, target);
                        vec![target]
                    }
                    _ => Vec::new(),
                }
            }
            _ => {
                let Some(lhs) = lhs else {
                    return Vec::new();
                };
                let Some(rhs) = rhs else {
                    return Vec::new();
                };
                let result = match self.kind {
                    CmpKind::Eq => lhs.get_u64() == rhs.get_u64(),
                    CmpKind::Ne => lhs.get_u64() != rhs.get_u64(),
                    CmpKind::SLt => {
                        let width = max_width(lhs, rhs);
                        sign_extend_bits(lhs, width) < sign_extend_bits(rhs, width)
                    }
                    CmpKind::SLe => {
                        let width = max_width(lhs, rhs);
                        sign_extend_bits(lhs, width) <= sign_extend_bits(rhs, width)
                    }
                    CmpKind::SGt => {
                        let width = max_width(lhs, rhs);
                        sign_extend_bits(lhs, width) > sign_extend_bits(rhs, width)
                    }
                    CmpKind::SGe => {
                        let width = max_width(lhs, rhs);
                        sign_extend_bits(lhs, width) >= sign_extend_bits(rhs, width)
                    }
                    CmpKind::ULt => lhs.get_u64() < rhs.get_u64(),
                    CmpKind::ULe => lhs.get_u64() <= rhs.get_u64(),
                    CmpKind::UGt => lhs.get_u64() > rhs.get_u64(),
                    CmpKind::UGe => lhs.get_u64() >= rhs.get_u64(),
                    CmpKind::LogEq | CmpKind::LogNe => return Vec::new(),
                };
                let id = egraph.add(SpirvLang::Const(const_bool(result)));
                egraph.union(eclass, id);
                vec![id]
            }
        }
    }
}

impl Applier<SpirvLang, ()> for FoldLogical {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let lhs = const_value(egraph, subst[self.a]).and_then(bool_const);
        let rhs = const_value(egraph, subst[self.b]).and_then(bool_const);
        if let (Some(a), Some(b)) = (lhs, rhs) {
            let result = match self.kind {
                LogicalKind::And => a && b,
                LogicalKind::Or => a || b,
            };
            let id = egraph.add(SpirvLang::Const(const_bool(result)));
            egraph.union(eclass, id);
            vec![id]
        } else if let Some(val) = lhs {
            match (self.kind, val) {
                (LogicalKind::And, false) => {
                    let id = egraph.add(SpirvLang::Const(const_bool(false)));
                    egraph.union(eclass, id);
                    vec![id]
                }
                (LogicalKind::And, true) => {
                    egraph.union(eclass, subst[self.b]);
                    vec![subst[self.b]]
                }
                (LogicalKind::Or, true) => {
                    let id = egraph.add(SpirvLang::Const(const_bool(true)));
                    egraph.union(eclass, id);
                    vec![id]
                }
                (LogicalKind::Or, false) => {
                    egraph.union(eclass, subst[self.b]);
                    vec![subst[self.b]]
                }
            }
        } else if let Some(val) = rhs {
            match (self.kind, val) {
                (LogicalKind::And, false) => {
                    let id = egraph.add(SpirvLang::Const(const_bool(false)));
                    egraph.union(eclass, id);
                    vec![id]
                }
                (LogicalKind::And, true) => {
                    egraph.union(eclass, subst[self.a]);
                    vec![subst[self.a]]
                }
                (LogicalKind::Or, true) => {
                    let id = egraph.add(SpirvLang::Const(const_bool(true)));
                    egraph.union(eclass, id);
                    vec![id]
                }
                (LogicalKind::Or, false) => {
                    egraph.union(eclass, subst[self.a]);
                    vec![subst[self.a]]
                }
            }
        } else {
            Vec::new()
        }
    }
}

impl Applier<SpirvLang, ()> for FoldLogicalNot {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(value) = const_value(egraph, subst[self.a]).and_then(bool_const) else {
            return Vec::new();
        };
        let id = egraph.add(SpirvLang::Const(const_bool(!value)));
        egraph.union(eclass, id);
        vec![id]
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
        if c2.get_u64() == 0 {
            return Vec::new();
        }
        let ratio = if self.signed {
            let width = max_width(c1, c2);
            let num = sign_extend_bits(c1, width);
            let den = sign_extend_bits(c2, width);
            if den == 0 || num % den != 0 {
                return Vec::new();
            }
            ConstValue::new_with_width(num.wrapping_div(den) as u64, width)
        } else {
            if c1.get_u64() % c2.get_u64() != 0 {
                return Vec::new();
            }
            combine_consts(c1, c2, u64::wrapping_div)
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
        let scaled_const = combine_consts(multiplier, add_const, u64::wrapping_mul);
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
        let scaled_const = combine_consts(multiplier, sub_const, u64::wrapping_mul);
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
        if const_value(egraph, subst[self.x]).is_some() {
            return Vec::new();
        }
        let Some(const_factor) = pure_const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let Some(const_term) = pure_const_value(egraph, subst[self.k]) else {
            return Vec::new();
        };
        if const_factor.get_u64() == 0 {
            return Vec::new();
        }
        if const_term.get_u64() % const_factor.get_u64() != 0 {
            return Vec::new();
        }
        let scaled = combine_consts(const_term, const_factor, u64::wrapping_div);
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
        if const_value(egraph, subst[self.x]).is_some() {
            return Vec::new();
        }
        let Some(coeff) = pure_const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let Some(const_term) = pure_const_value(egraph, subst[self.k]) else {
            return Vec::new();
        };
        let width = max_width(coeff, const_term);
        if width > 32 {
            return Vec::new();
        }
        // Only factor when coefficients are non-negative to avoid sign surprises with wrapping.
        let sign_mask = if width == 0 {
            return Vec::new();
        } else {
            1u64 << (width - 1)
        };
        if coeff.get_u64() & sign_mask != 0 || const_term.get_u64() & sign_mask != 0 {
            return Vec::new();
        }
        if coeff.get_u64() == 0 || const_term.get_u64() == 0 {
            return Vec::new();
        }
        let gcd = gcd_u32(coeff.get_u64() as u32, const_term.get_u64() as u32);
        if gcd <= 1 || gcd as u64 == coeff.get_u64() {
            return Vec::new();
        }
        let scaled_coeff = ConstValue::new_with_width(coeff.get_u64() / gcd as u64, width);
        let scaled_const = ConstValue::new_with_width(const_term.get_u64() / gcd as u64, width);
        let gcd_id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            gcd as u64, width,
        )));
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
        let Some(width) = width_hint(egraph, eclass, [subst[self.x]]) else {
            return Vec::new();
        };
        let two = egraph.add(SpirvLang::Const(ConstValue::new_with_width(2, width)));
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
        let Some(width) = width_hint(egraph, eclass, [subst[self.x]]) else {
            return Vec::new();
        };
        let three = egraph.add(SpirvLang::Const(ConstValue::new_with_width(3, width)));
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
        let Some(width) = width_hint(egraph, eclass, [subst[self.x]]) else {
            return Vec::new();
        };
        let two = egraph.add(SpirvLang::Const(ConstValue::new_with_width(2, width)));
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
        let Some(shift) = is_power_of_two(constant.get_u64()) else {
            return Vec::new();
        };
        let width = u32::from(constant.width_bits());
        if shift == 0 || shift >= width {
            return Vec::new();
        }
        let shift_const = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            shift as u64,
            constant.width_bits(),
        )));
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
        let Some(shift) = is_power_of_two(constant.get_u64()) else {
            return Vec::new();
        };
        let width = u32::from(constant.width_bits());
        if shift == 0 || shift >= width {
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
        let shift_const = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            shift as u64,
            constant.width_bits(),
        )));
        if self.signed {
            let sign_shift = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
                constant.width_bits().saturating_sub(1).max(1) as u64,
                constant.width_bits(),
            )));
            let sign = egraph.add(SpirvLang::ShrS([subst[self.x], sign_shift]));
            let mask_value = ((1u128 << shift) - 1) as u64;
            let mask = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
                mask_value,
                constant.width_bits(),
            )));
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
        let Some(shift) = is_power_of_two(constant.get_u64()) else {
            return Vec::new();
        };
        let width = u32::from(constant.width_bits());
        if shift == 0 || shift >= width {
            return Vec::new();
        }
        let shift_const = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            shift as u64,
            constant.width_bits(),
        )));
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
        if constant.is_zero() {
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
        if constant.is_zero() {
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
        let folded = combine_consts(lhs, rhs, |x, y| x & y);
        let const_id = egraph.add(SpirvLang::Const(folded));
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
        if constant.get_u64() == constant.mask() {
            egraph.union(eclass, subst[self.x]);
            return vec![subst[self.x]];
        }
        if constant.get_u64() == 0 {
            let zero = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
                0,
                constant.width_bits(),
            )));
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
        let mask = constant.get_u64();
        let Some(shift) = is_power_of_two(mask.wrapping_add(1)) else {
            return Vec::new();
        };
        let width = u32::from(constant.width_bits());
        if shift == 0 || shift >= width {
            return Vec::new();
        }
        let const_id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            mask.wrapping_add(1),
            constant.width_bits(),
        )));
        let umod = egraph.add(SpirvLang::UMod([subst[self.x], const_id]));
        egraph.union(eclass, umod);
        vec![umod]
    }
}

impl Applier<SpirvLang, ()> for BitAndComplement {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(width) = width_hint(egraph, eclass, [subst[self._x]]) else {
            return Vec::new();
        };
        let zero = egraph.add(SpirvLang::Const(ConstValue::new_with_width(0, width)));
        egraph.union(eclass, zero);
        vec![zero]
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
        let Some(shift) = is_power_of_two(constant.get_u64()) else {
            return Vec::new();
        };
        let width = u32::from(constant.width_bits());
        if shift == 0 || shift >= width {
            return Vec::new();
        }
        let mask_value = (1u128 << shift) - 1;
        let mask = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            mask_value as u64,
            constant.width_bits(),
        )));
        let band = egraph.add(SpirvLang::BitAnd([subst[self.x], mask]));
        egraph.union(eclass, band);
        vec![band]
    }
}

impl Applier<SpirvLang, ()> for BitOrFold {
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
        let folded = combine_consts(a, b, |x, y| x | y);
        let id = egraph.add(SpirvLang::Const(folded));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for BitOrConstSimplify {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(c) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        match c.get_u64() {
            0 => {
                egraph.union(eclass, subst[self.x]);
                vec![subst[self.x]]
            }
            v if v == c.mask() => {
                let ones = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
                    c.mask(),
                    c.width_bits(),
                )));
                egraph.union(eclass, ones);
                vec![ones]
            }
            _ => Vec::new(),
        }
    }
}

impl Applier<SpirvLang, ()> for BitwiseConstReassociate {
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
        let merged = match self.op {
            BitwiseOp::Or => combine_consts(c1, c2, |x, y| x | y),
            BitwiseOp::Xor => combine_consts(c1, c2, |x, y| x ^ y),
        };
        let const_id = egraph.add(SpirvLang::Const(merged));
        let node = match self.op {
            BitwiseOp::Or => egraph.add(SpirvLang::BitOr([subst[self.x], const_id])),
            BitwiseOp::Xor => egraph.add(SpirvLang::BitXor([subst[self.x], const_id])),
        };
        egraph.union(eclass, node);
        vec![node]
    }
}

impl Applier<SpirvLang, ()> for BitOrComplement {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(width) = width_hint(egraph, eclass, [subst[self._x]]) else {
            return Vec::new();
        };
        let ones = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            u64::MAX,
            width,
        )));
        egraph.union(eclass, ones);
        vec![ones]
    }
}

impl Applier<SpirvLang, ()> for BitXorComplement {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(width) = width_hint(egraph, eclass, [subst[self._x]]) else {
            return Vec::new();
        };
        let ones = egraph.add(SpirvLang::Const(ConstValue::new_with_width(
            u64::MAX,
            width,
        )));
        egraph.union(eclass, ones);
        vec![ones]
    }
}

impl Applier<SpirvLang, ()> for BitXorFold {
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
        let folded = combine_consts(a, b, |x, y| x ^ y);
        let id = egraph.add(SpirvLang::Const(folded));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for BitXorConstSimplify {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(c) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        if c.is_zero() {
            egraph.union(eclass, subst[self.x]);
            return vec![subst[self.x]];
        }
        if c.is_all_ones() {
            let bnot = egraph.add(SpirvLang::BitNot(subst[self.x]));
            egraph.union(eclass, bnot);
            return vec![bnot];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for BitXorSelf {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(width) = width_hint(egraph, eclass, [subst[self._x]]) else {
            return Vec::new();
        };
        let zero = egraph.add(SpirvLang::Const(ConstValue::new_with_width(0, width)));
        egraph.union(eclass, zero);
        vec![zero]
    }
}

impl Applier<SpirvLang, ()> for SubMaskAbsorb {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _ast: Option<&PatternAst<SpirvLang>>,
        _runner: Symbol,
    ) -> Vec<Id> {
        if let Some(mask_val) = const_value(egraph, subst[self.mask]) {
            if mask_val.is_all_ones() {
                egraph.union(eclass, subst[self.x]);
            }
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for BitXorAbsorb {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _ast: Option<&PatternAst<SpirvLang>>,
        _runner: Symbol,
    ) -> Vec<Id> {
        if let Some(mask_val) = const_value(egraph, subst[self.mask]) {
            if mask_val.is_zero() {
                egraph.union(eclass, subst[self.x]);
            }
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for BitXorAllOnes {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _ast: Option<&PatternAst<SpirvLang>>,
        _runner: Symbol,
    ) -> Vec<Id> {
        if let Some(mask_val) = const_value(egraph, subst[self.mask]) {
            if mask_val.is_all_ones() {
                if let Some(width) = width_hint(egraph, eclass, [subst[self.x]]) {
                    let zero = egraph.add(SpirvLang::Const(ConstValue::new_with_width(0, width)));
                    egraph.union(eclass, zero);
                }
            }
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for BandRedundantOr {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _ast: Option<&PatternAst<SpirvLang>>,
        _runner: Symbol,
    ) -> Vec<Id> {
        let Some(mask_val) = const_value(egraph, subst[self.mask]) else {
            return Vec::new();
        };
        let Some(or_const) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let width = max_width(mask_val, or_const);
        if width > 64 {
            return Vec::new();
        }
        let mask = ConstValue::new_with_width(mask_val.get_u64(), width).get_u64();
        let or_const = ConstValue::new_with_width(or_const.get_u64(), width).get_u64();
        let overlap = mask & or_const;
        if overlap == mask {
            let mask_id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(mask, width)));
            egraph.union(eclass, mask_id);
            return vec![mask_id];
        }
        if overlap == 0 {
            let mask_id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(mask, width)));
            let band = egraph.add(SpirvLang::BitAnd([subst[self.x], mask_id]));
            egraph.union(eclass, band);
            return vec![band];
        }
        Vec::new()
    }
}

impl Applier<SpirvLang, ()> for BandRedundantXor {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _ast: Option<&PatternAst<SpirvLang>>,
        _runner: Symbol,
    ) -> Vec<Id> {
        let Some(mask_val) = const_value(egraph, subst[self.mask]) else {
            return Vec::new();
        };
        let Some(xor_const) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let width = max_width(mask_val, xor_const);
        if width > 64 {
            return Vec::new();
        }
        let mask = ConstValue::new_with_width(mask_val.get_u64(), width).get_u64();
        let xor_const = ConstValue::new_with_width(xor_const.get_u64(), width).get_u64();
        if (mask & xor_const) != 0 {
            return Vec::new();
        }
        let mask_id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(mask, width)));
        let band = egraph.add(SpirvLang::BitAnd([subst[self.x], mask_id]));
        egraph.union(eclass, band);
        vec![band]
    }
}

impl Applier<SpirvLang, ()> for BandRedundantAdd {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _ast: Option<&PatternAst<SpirvLang>>,
        _runner: Symbol,
    ) -> Vec<Id> {
        let Some(mask_val) = const_value(egraph, subst[self.mask]) else {
            return Vec::new();
        };
        let Some(add_const) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let width = max_width(mask_val, add_const);
        if width > 64 {
            return Vec::new();
        }
        let mask = ConstValue::new_with_width(mask_val.get_u64(), width).get_u64();
        let add = ConstValue::new_with_width(add_const.get_u64(), width).get_u64();
        let lsb = add & add.wrapping_neg();
        if lsb == 0 || lsb <= mask {
            return Vec::new();
        }
        let mask_id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(mask, width)));
        let band = egraph.add(SpirvLang::BitAnd([subst[self.x], mask_id]));
        egraph.union(eclass, band);
        vec![band]
    }
}

impl Applier<SpirvLang, ()> for BandRedundantSub {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _ast: Option<&PatternAst<SpirvLang>>,
        _runner: Symbol,
    ) -> Vec<Id> {
        let Some(mask_val) = const_value(egraph, subst[self.mask]) else {
            return Vec::new();
        };
        let Some(sub_const) = const_value(egraph, subst[self.c]) else {
            return Vec::new();
        };
        let width = max_width(mask_val, sub_const);
        if width > 64 {
            return Vec::new();
        }
        let mask = ConstValue::new_with_width(mask_val.get_u64(), width).get_u64();
        let sub = ConstValue::new_with_width(sub_const.get_u64(), width).get_u64();
        let lsb = sub & sub.wrapping_neg();
        if lsb == 0 || lsb <= mask {
            return Vec::new();
        }
        let mask_id = egraph.add(SpirvLang::Const(ConstValue::new_with_width(mask, width)));
        let band = egraph.add(SpirvLang::BitAnd([subst[self.x], mask_id]));
        egraph.union(eclass, band);
        vec![band]
    }
}

impl Applier<SpirvLang, ()> for BandRedundantShift {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _ast: Option<&PatternAst<SpirvLang>>,
        _runner: Symbol,
    ) -> Vec<Id> {
        let Some(mask_val) = const_value(egraph, subst[self.mask]) else {
            return Vec::new();
        };
        let Some(shift_val) = const_value(egraph, subst[self.shift]) else {
            return Vec::new();
        };
        let width = max_width(mask_val, shift_val);
        if width > 64 {
            return Vec::new();
        }
        let shift = shift_val.get_u64();
        if shift >= u64::from(width) {
            return Vec::new();
        }
        let mask = ConstValue::new_with_width(mask_val.get_u64(), width).get_u64();
        let width_mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let shift = shift as u32;
        let clears_mask = match self.kind {
            ShiftKind::Left => (mask >> shift) == 0,
            ShiftKind::RightUnsigned => (mask.wrapping_shl(shift) & width_mask) == 0,
            ShiftKind::RightSigned => false,
        };
        if !clears_mask {
            return Vec::new();
        }
        let zero = egraph.add(SpirvLang::Const(ConstValue::new_with_width(0, width)));
        egraph.union(eclass, zero);
        vec![zero]
    }
}

impl Applier<SpirvLang, ()> for BitNotFold {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(x) = const_value(egraph, subst[self.x]) else {
            return Vec::new();
        };
        let folded = ConstValue::new_with_width(!x.get_u64(), x.width_bits());
        let id = egraph.add(SpirvLang::Const(folded));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for BitNotDouble {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        egraph.union(eclass, subst[self.x]);
        vec![subst[self.x]]
    }
}

impl Applier<SpirvLang, ()> for BitReverseFold {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(value) = unique_const_value(egraph, subst[self.x]) else {
            return Vec::new();
        };
        let width = value.width_bits();
        if width == 0 || width > 64 {
            return Vec::new();
        }
        let reversed = if width == 64 {
            value.get_u64().reverse_bits()
        } else {
            value.get_u64().reverse_bits() >> (64 - width as u32)
        };
        let folded = ConstValue::new_with_width(reversed, width);
        let id = egraph.add(SpirvLang::Const(folded));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for RotatePatternFold {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(x) = const_value(egraph, subst[self.x]) else {
            return Vec::new();
        };
        let Some(s) = const_value(egraph, subst[self.s]) else {
            return Vec::new();
        };
        let Some(t) = const_value(egraph, subst[self.t]) else {
            return Vec::new();
        };
        let width = x.width_bits();
        if s.width_bits() != width || t.width_bits() != width {
            return Vec::new();
        }
        let width_u64 = width as u64;
        // Only fold when the shifts are complementary (s + t == word size).
        if (s.get_u64().wrapping_add(t.get_u64())) % width_u64 != 0 {
            return Vec::new();
        }
        let shl_amt = (s.get_u64() % width_u64) as u32;
        let shr_amt = (t.get_u64() % width_u64) as u32;
        let mask = x.mask();
        let left = (x.get_u64().wrapping_shl(shl_amt)) & mask;
        let right = (x.get_u64().wrapping_shr(shr_amt)) & mask;
        let folded = ConstValue::new_with_width(left | right, width);
        let id = egraph.add(SpirvLang::Const(folded));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for MaskThenShift {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(mask_val) = const_value(egraph, subst[self.mask]) else {
            return Vec::new();
        };
        let Some(shift_val) = const_value(egraph, subst[self.shift]) else {
            return Vec::new();
        };
        let width = max_width(mask_val, shift_val);
        if width > 64 {
            return Vec::new();
        }
        let shift = shift_val.get_u64();
        if shift >= u64::from(width) {
            return Vec::new();
        }
        let Some(mask_pow) = is_power_of_two(mask_val.get_u64()) else {
            return Vec::new();
        };
        if mask_pow == 0 {
            return Vec::new();
        }
        // (x & (2^n - 1)) >> k  =>  (x >> k) & (2^n - 1 >> k)
        let shr_const = map_const(shift_val, |v| v);
        let shr_const_id = egraph.add(SpirvLang::Const(shr_const));
        let shr = egraph.add(SpirvLang::ShrU([subst[self.x], shr_const_id]));
        let new_mask =
            ConstValue::new_with_width(mask_val.get_u64().wrapping_shr(shift as u32), width);
        let mask_id = egraph.add(SpirvLang::Const(new_mask));
        let band = egraph.add(SpirvLang::BitAnd([shr, mask_id]));
        egraph.union(eclass, band);
        vec![band]
    }
}

impl Applier<SpirvLang, ()> for MaskThenShiftSigned {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(mask_val) = const_value(egraph, subst[self.mask]) else {
            return Vec::new();
        };
        let Some(shift_val) = const_value(egraph, subst[self.shift]) else {
            return Vec::new();
        };
        let width = max_width(mask_val, shift_val);
        if width > 64 {
            return Vec::new();
        }
        let shift = shift_val.get_u64();
        if shift >= u64::from(width) {
            return Vec::new();
        }
        let Some(mask_pow) = is_power_of_two(mask_val.get_u64()) else {
            return Vec::new();
        };
        if mask_pow == 0 {
            return Vec::new();
        }
        let shr_const = map_const(shift_val, |v| v);
        let shr_const_id = egraph.add(SpirvLang::Const(shr_const));
        let shr = egraph.add(SpirvLang::ShrS([subst[self.x], shr_const_id]));
        let new_mask =
            ConstValue::new_with_width(mask_val.get_u64().wrapping_shr(shift as u32), width);
        let mask_id = egraph.add(SpirvLang::Const(new_mask));
        let band = egraph.add(SpirvLang::BitAnd([shr, mask_id]));
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
        let total = combine_consts(a, b, u64::wrapping_add);
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

impl Applier<SpirvLang, ()> for ShlFoldConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(lhs) = unique_const_value(egraph, subst[self.a]) else {
            return Vec::new();
        };
        let Some(rhs) = unique_const_value(egraph, subst[self.b]) else {
            return Vec::new();
        };
        let width = lhs.width_bits().max(rhs.width_bits()).max(1);
        let amount = (rhs.get_u64() % width as u64) as u32;
        let folded = ConstValue::new_with_width(lhs.get_u64().wrapping_shl(amount), width);
        let id = egraph.add(SpirvLang::Const(folded));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for ShrUFoldConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(lhs) = unique_const_value(egraph, subst[self.a]) else {
            return Vec::new();
        };
        let Some(rhs) = unique_const_value(egraph, subst[self.b]) else {
            return Vec::new();
        };
        let width = lhs.width_bits().max(rhs.width_bits()).max(1);
        let amount = (rhs.get_u64() % width as u64) as u32;
        let folded = ConstValue::new_with_width(lhs.get_u64().wrapping_shr(amount), width);
        let id = egraph.add(SpirvLang::Const(folded));
        egraph.union(eclass, id);
        vec![id]
    }
}

impl Applier<SpirvLang, ()> for ShrSFoldConst {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, ()>,
        eclass: Id,
        subst: &Subst,
        _pat: Option<&PatternAst<SpirvLang>>,
        _symbol: Symbol,
    ) -> Vec<Id> {
        let Some(lhs) = unique_const_value(egraph, subst[self.a]) else {
            return Vec::new();
        };
        let Some(rhs) = unique_const_value(egraph, subst[self.b]) else {
            return Vec::new();
        };
        let width = lhs.width_bits().max(rhs.width_bits()).max(1);
        let amount = (rhs.get_u64() % width as u64) as u32;
        let shift_for_sign = 64 - width.min(64) as i32;
        let signed = ((lhs.get_u64() << shift_for_sign) as i64).wrapping_shr(shift_for_sign as u32);
        let folded = ConstValue::new_with_width((signed >> amount) as u64, width);
        let id = egraph.add(SpirvLang::Const(folded));
        egraph.union(eclass, id);
        vec![id]
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
        let merged = combine_consts(c1, c2, u64::wrapping_mul);
        let const_id = egraph.add(SpirvLang::Const(merged));
        let mul = egraph.add(SpirvLang::Mul([subst[self.base], const_id]));
        egraph.union(eclass, mul);
        vec![mul]
    }
}

fn const_value(egraph: &EGraph<SpirvLang, ()>, id: Id) -> Option<ConstValue> {
    unique_const_value(egraph, id)
}

fn pure_const_value(egraph: &EGraph<SpirvLang, ()>, id: Id) -> Option<ConstValue> {
    let class = egraph.find(id);
    let mut value: Option<ConstValue> = None;
    for node in &egraph[class].nodes {
        match node {
            SpirvLang::Const(constant) => {
                if let Some(existing) = value {
                    if existing != *constant {
                        return None;
                    }
                } else {
                    value = Some(*constant);
                }
            }
            _ => return None,
        }
    }
    value
}

fn bool_const(value: ConstValue) -> Option<bool> {
    if value.width_bits() == 1 {
        Some(value.get_u64() != 0)
    } else {
        None
    }
}

fn const_bool(value: bool) -> ConstValue {
    ConstValue::new_with_width(if value { 1 } else { 0 }, 1)
}

fn width_from_eclass(
    egraph: &EGraph<SpirvLang, ()>,
    id: Id,
    visited: &mut HashSet<Id>,
) -> Option<u8> {
    let class = egraph.find(id);
    if !visited.insert(class) {
        return None;
    }
    let mut width: Option<u8> = None;
    for node in &egraph[class].nodes {
        let candidate = match node {
            SpirvLang::Const(c) => Some(c.width_bits()),
            SpirvLang::Symbol(sym) => symbol_width(sym),
            _ => node
                .children()
                .iter()
                .filter_map(|child| width_from_eclass(egraph, *child, visited))
                .max(),
        };
        if let Some(bits) = candidate {
            width = Some(match width {
                Some(current) => current.max(bits),
                None => bits,
            });
        }
    }
    width
}

fn width_hint(
    egraph: &EGraph<SpirvLang, ()>,
    eclass: Id,
    ids: impl IntoIterator<Item = Id>,
) -> Option<u8> {
    let mut visited = HashSet::new();
    let mut width: Option<u8> = None;
    for id in ids {
        if let Some(bits) = width_from_eclass(egraph, id, &mut visited) {
            width = Some(match width {
                Some(current) => current.max(bits),
                None => bits,
            });
        }
    }
    if let Some(bits) = width_from_eclass(egraph, eclass, &mut visited) {
        width = Some(match width {
            Some(current) => current.max(bits),
            None => bits,
        });
    }
    width.or(Some(32))
}

fn unique_const_value(egraph: &EGraph<SpirvLang, ()>, id: Id) -> Option<ConstValue> {
    let mut constants = egraph[egraph.find(id)]
        .nodes
        .iter()
        .filter_map(|node| match node {
            SpirvLang::Const(value) => Some(*value),
            _ => None,
        });
    let value = constants.next()?;
    if constants.any(|other| other != value) {
        return None;
    }
    Some(value)
}

fn max_width(lhs: ConstValue, rhs: ConstValue) -> u8 {
    lhs.width_bits().max(rhs.width_bits()).max(1)
}

fn combine_consts(lhs: ConstValue, rhs: ConstValue, f: impl Fn(u64, u64) -> u64) -> ConstValue {
    let width = max_width(lhs, rhs);
    ConstValue::new_with_width(f(lhs.get_u64(), rhs.get_u64()), width)
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let tmp = b;
        b = a % b;
        a = tmp;
    }
    a
}

fn map_const(value: ConstValue, f: impl Fn(u64) -> u64) -> ConstValue {
    let width = value.width_bits().max(1);
    ConstValue::new_with_width(f(value.get_u64()), width)
}

fn sign_extend_bits(value: ConstValue, width: u8) -> i128 {
    let shift = 128u32.saturating_sub(width as u32);
    ((value.get_u64() as i128) << shift) >> shift
}

fn combine_signed_consts(
    lhs: ConstValue,
    rhs: ConstValue,
    f: impl Fn(i128, i128) -> i128,
) -> ConstValue {
    let width = max_width(lhs, rhs);
    let l = sign_extend_bits(lhs, width);
    let r = sign_extend_bits(rhs, width);
    ConstValue::new_with_width(f(l, r) as u64, width)
}

fn div_floor(lhs: i128, rhs: i128) -> i128 {
    let (d, r) = (lhs / rhs, lhs % rhs);
    if r != 0 && ((r > 0) != (rhs > 0)) {
        d - 1
    } else {
        d
    }
}

fn combine_signed_mod(lhs: ConstValue, rhs: ConstValue) -> ConstValue {
    let width = max_width(lhs, rhs);
    let l = sign_extend_bits(lhs, width);
    let r = sign_extend_bits(rhs, width);
    let div = div_floor(l, r);
    let rem = l - r * div;
    ConstValue::new_with_width(rem as u64, width)
}

fn is_power_of_two(value: u64) -> Option<u32> {
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
    move |egraph, _, subst| const_value(egraph, subst[var]).is_some_and(|c| c.get_u64() == 0)
}

fn is_const_all_ones(var: Var) -> impl Fn(&mut EGraph<SpirvLang, ()>, Id, &Subst) -> bool + 'static {
    move |egraph, _, subst| const_value(egraph, subst[var]).is_some_and(ConstValue::is_all_ones)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::{optimize_arith_block, optimize_arith_block_with_types};
    use arbitrary::Unstructured;
    use pretty_assertions::assert_eq;
    use rspirv::dr::{Builder, Instruction};
    use std::collections::HashMap;

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
    fn folds_addition_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new64(2)),
            SpirvLang::Const(ConstValue::new64(3)),
            SpirvLang::Add([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(5, 64))])
        );
    }

    #[test]
    fn folds_division_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new64(8)),
            SpirvLang::Const(ConstValue::new64(4)),
            SpirvLang::UDiv([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(2, 64))])
        );
    }

    #[test]
    fn folds_remainder_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new64(10)),
            SpirvLang::Const(ConstValue::new64(4)),
            SpirvLang::UMod([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(2, 64))])
        );
    }

    #[test]
    fn folds_signed_division_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new64((-8i64) as u64)),
            SpirvLang::Const(ConstValue::new64(2)),
            SpirvLang::SDiv([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(
                (-4i64) as u64,
                64
            ))])
        );
    }

    #[test]
    fn folds_signed_remainder_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new64((-9i64) as u64)),
            SpirvLang::Const(ConstValue::new64(4)),
            SpirvLang::SRem([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(
                (-1i64) as u64,
                64
            ))])
        );
    }

    #[test]
    fn folds_signed_mod_u64() {
        let expr_neg_dividend = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new64((-9i64) as u64)),
            SpirvLang::Const(ConstValue::new64(4)),
            SpirvLang::SMod([Id::from(0), Id::from(1)]),
        ]);
        let optimized_neg_dividend = optimize_expr(&expr_neg_dividend);
        assert_eq!(
            optimized_neg_dividend,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(3, 64))])
        );

        let expr_neg_divisor = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new64(9)),
            SpirvLang::Const(ConstValue::new64((-4i64) as u64)),
            SpirvLang::SMod([Id::from(0), Id::from(1)]),
        ]);
        let optimized_neg_divisor = optimize_expr(&expr_neg_divisor);
        assert_eq!(
            optimized_neg_divisor,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(
                (-3i64) as u64,
                64
            ))])
        );
    }

    #[test]
    fn mul_zero_preserves_width_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new64(0)),
            SpirvLang::Symbol("x".into()),
            SpirvLang::Mul([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(0, 64))])
        );
    }

    #[test]
    fn bitand_zero_preserves_width_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol("x".into()),
            SpirvLang::Const(ConstValue::new64(0)),
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(0, 64))])
        );
    }

    #[test]
    fn add_zero_checks_full_width_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol("x".into()),
            SpirvLang::Const(ConstValue::new_with_width(1u64 << 32, 64)),
            SpirvLang::Add([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert!(
            matches!(optimized.as_ref().last(), Some(SpirvLang::Add(_))),
            "expected add to remain, got {optimized:?}"
        );
    }

    #[test]
    fn mul_one_checks_full_width_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol("x".into()),
            SpirvLang::Const(ConstValue::new_with_width(0x1_0000_0001, 64)),
            SpirvLang::Mul([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert!(
            matches!(optimized.as_ref().last(), Some(SpirvLang::Mul(_))),
            "expected mul to remain, got {optimized:?}"
        );
    }

    #[test]
    fn mul_neg_one_requires_full_width_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol("x".into()),
            SpirvLang::Const(ConstValue::new_with_width(0xFFFF_FFFF, 64)),
            SpirvLang::Mul([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert!(
            matches!(optimized.as_ref().last(), Some(SpirvLang::Mul(_))),
            "expected mul to remain, got {optimized:?}"
        );
    }

    #[test]
    fn bxor_all_ones_requires_full_width_u64() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol("x".into()),
            SpirvLang::Const(ConstValue::new_with_width(0xFFFF_FFFF, 64)),
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert!(
            matches!(optimized.as_ref().last(), Some(SpirvLang::BitXor(_))),
            "expected bxor to remain, got {optimized:?}"
        );
    }

    #[test]
    fn merges_nested_bor_constants_with_symbol() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(0x0F)),      // 0
            SpirvLang::Symbol(Symbol::from("x")),         // 1
            SpirvLang::Const(ConstValue::new(0xF0)),      // 2
            SpirvLang::BitOr([Id::from(1), Id::from(2)]), // 3 = x | 0xF0
            SpirvLang::BitOr([Id::from(0), Id::from(3)]), // 4 = 0x0F | (x | 0xF0)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let const_id = nodes
            .iter()
            .position(|node| matches!(node, SpirvLang::Const(val) if val.get_u64() == 0xFF));
        let sym_id = nodes
            .iter()
            .position(|node| matches!(node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")));
        let is_merged = nodes.last().is_some_and(|node| match node {
            SpirvLang::BitOr([lhs, rhs]) => {
                let lhs = usize::from(*lhs);
                let rhs = usize::from(*rhs);
                (Some(lhs) == const_id && Some(rhs) == sym_id)
                    || (Some(rhs) == const_id && Some(lhs) == sym_id)
            }
            _ => false,
        });
        assert!(is_merged, "expected merged constant, got {optimized:?}");
    }

    #[test]
    fn merges_nested_bxor_constants_with_symbol() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(0x0F)),       // 0
            SpirvLang::Symbol(Symbol::from("x")),          // 1
            SpirvLang::Const(ConstValue::new(0xF0)),       // 2
            SpirvLang::BitXor([Id::from(1), Id::from(2)]), // 3 = x ^ 0xF0
            SpirvLang::BitXor([Id::from(0), Id::from(3)]), // 4 = 0x0F ^ (x ^ 0xF0)
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let const_id = nodes
            .iter()
            .position(|node| matches!(node, SpirvLang::Const(val) if val.get_u64() == 0xFF));
        let sym_id = nodes
            .iter()
            .position(|node| matches!(node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")));
        let is_merged = nodes.last().is_some_and(|node| match node {
            SpirvLang::BitXor([lhs, rhs]) => {
                let lhs = usize::from(*lhs);
                let rhs = usize::from(*rhs);
                (Some(lhs) == const_id && Some(rhs) == sym_id)
                    || (Some(rhs) == const_id && Some(lhs) == sym_id)
            }
            _ => false,
        });
        assert!(is_merged, "expected merged constant, got {optimized:?}");
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
    fn bitand_with_zero_folds_to_zero() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(ConstValue::new(0)),
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(0))])
        );
    }

    #[test]
    fn bitand_with_all_ones_preserves_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(ConstValue::new(u32::MAX)),
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn bitand_self_simplifies() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::BitAnd([Id::from(0), Id::from(0)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn bitand_complement_simplifies_to_zero() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::BitNot(Id::from(0)),
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(0))])
        );
    }

    #[test]
    fn bitand_complement_defaults_to_32bit_zero() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::BitNot(Id::from(0)),
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        assert_eq!(nodes.len(), 1, "expected fold to a single constant");
        if let SpirvLang::Const(c) = nodes[0] {
            assert_eq!(c.get(), 0);
            assert_eq!(c.width_bits(), 32);
        } else {
            panic!("expected constant; got {optimized:?}");
        }
    }

    #[test]
    fn bitand_complement_respects_type_width_hint() {
        use rspirv::dr::Operand;

        let type_int64 = Instruction::new(
            rspirv::spirv::Op::TypeInt,
            None,
            Some(1),
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        );
        let imul = Instruction::new(
            rspirv::spirv::Op::IMul,
            Some(1),
            Some(2),
            vec![Operand::IdRef(3), Operand::IdRef(4)],
        );
        let not_inst = Instruction::new(
            rspirv::spirv::Op::Not,
            Some(1),
            Some(5),
            vec![Operand::IdRef(2)],
        );
        let band = Instruction::new(
            rspirv::spirv::Op::BitwiseAnd,
            Some(1),
            Some(6),
            vec![Operand::IdRef(2), Operand::IdRef(5)],
        );
        let type_widths = HashMap::from([(1u32, 64u32)]);
        let optimized =
            optimize_arith_block_with_types(&[type_int64, imul, not_inst, band], &type_widths)
                .expect("optimize");
        assert_eq!(optimized.len(), 1);
        let inst = &optimized[0];
        assert_eq!(inst.class.opcode, rspirv::spirv::Op::Constant);
        assert_eq!(inst.result_type, Some(1));
        assert_eq!(inst.result_id, Some(6));
        assert_eq!(inst.operands, vec![Operand::LiteralBit64(0)]);
    }

    #[test]
    fn band_mask_prefers_umod_pow2() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(ConstValue::new(7)),
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let has_umod = nodes.iter().any(|node| {
            if let SpirvLang::UMod([lhs, rhs]) = node {
                matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == Symbol::from("x"))
                    && matches!(nodes[usize::from(*rhs)], SpirvLang::Const(c) if c.get() == 8)
            } else {
                false
            }
        });
        assert!(
            has_umod,
            "expected x & (2^n-1) to become x % 2^n; got {optimized:?}"
        );
    }

    #[test]
    fn bitor_complement_simplifies_to_all_ones() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::BitNot(Id::from(0)),
            SpirvLang::BitOr([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(u32::MAX))])
        );
    }

    #[test]
    fn bitor_complement_defaults_to_32bit_width() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::BitNot(Id::from(0)),
            SpirvLang::BitOr([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        assert_eq!(nodes.len(), 1, "expected fold to a single constant");
        if let SpirvLang::Const(c) = nodes[0] {
            assert_eq!(c.get(), u32::MAX);
            assert_eq!(c.width_bits(), 32);
        } else {
            panic!("expected constant; got {optimized:?}");
        }
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
    fn rewrites_add_double_negation_to_negated_sum() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Symbol(Symbol::from("y")), // 1
            SpirvLang::Neg(Id::from(0)),          // 2 = -x
            SpirvLang::Neg(Id::from(1)),          // 3 = -y
            SpirvLang::Add([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let neg_sum: RecExpr<SpirvLang> = "(neg (+ x y))".parse().unwrap();
        let neg_sum_comm: RecExpr<SpirvLang> = "(neg (+ y x))".parse().unwrap();
        let neg_sum_id = runner
            .egraph
            .lookup_expr(&neg_sum)
            .or_else(|| runner.egraph.lookup_expr(&neg_sum_comm))
            .expect("expected negated sum to be introduced by rewrites");
        assert!(
            runner.egraph.find(root) == runner.egraph.find(neg_sum_id),
            "expected negated sum to be equivalent to (-x) + (-y)"
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
    fn cancels_sub_subtracting_original_lhs() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = x - y
            SpirvLang::Sub([Id::from(2), Id::from(0)]), // 3 = (x - y) - x
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![
                SpirvLang::Symbol(Symbol::from("y")),
                SpirvLang::Neg(Id::from(0)),
            ])
        );
    }

    #[test]
    fn rewrites_subtract_additive_self_to_negated_other() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 2 = x + y
            SpirvLang::Sub([Id::from(0), Id::from(2)]), // 3 = x - (x + y)
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![
                SpirvLang::Symbol(Symbol::from("y")),
                SpirvLang::Neg(Id::from(0)),
            ])
        );
    }

    #[test]
    fn rewrites_subtract_commuted_additive_self_to_negated_other() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Add([Id::from(1), Id::from(0)]), // 2 = y + x
            SpirvLang::Sub([Id::from(0), Id::from(2)]), // 3 = x - (y + x)
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![
                SpirvLang::Symbol(Symbol::from("y")),
                SpirvLang::Neg(Id::from(0)),
            ])
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
    fn folds_const_sub_then_add_back() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(42)), // 0
            SpirvLang::Const(ConstValue::new(5)),  // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]),
            SpirvLang::Add([Id::from(2), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(42))])
        );
    }

    #[test]
    fn folds_const_rotate_left_width32() {
        // (x << 8) | (x >> 24) with x=0x12345678 => rotate-left by 8 => 0x34567812
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(0x1234_5678)), // 0
            SpirvLang::Const(ConstValue::new(8)),           // 1
            SpirvLang::Shl([Id::from(0), Id::from(1)]),     // 2
            SpirvLang::Const(ConstValue::new(24)),          // 3
            SpirvLang::ShrU([Id::from(0), Id::from(3)]),    // 4
            SpirvLang::BitOr([Id::from(2), Id::from(4)]),   // 5
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        assert_eq!(nodes.len(), 1, "expected constant fold");
        if let SpirvLang::Const(c) = nodes[0] {
            assert_eq!(c.get(), 0x3456_7812);
            assert_eq!(c.width_bits(), 32);
        } else {
            panic!("expected constant; got {optimized:?}");
        }
    }

    #[test]
    fn folds_const_rotate_left_width64() {
        // (x << 16) | (x >> 48) with x=0x0123456789ABCDEF => rotate-left by 16 => 0x456789ABCDEF0123
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new_with_width(0x0123_4567_89AB_CDEF, 64)), // 0
            SpirvLang::Const(ConstValue::new_with_width(16, 64)),                    // 1
            SpirvLang::Shl([Id::from(0), Id::from(1)]),                              // 2
            SpirvLang::Const(ConstValue::new_with_width(48, 64)),                    // 3
            SpirvLang::ShrU([Id::from(0), Id::from(3)]),                             // 4
            SpirvLang::BitOr([Id::from(2), Id::from(4)]),                            // 5
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        assert_eq!(nodes.len(), 1, "expected constant fold");
        if let SpirvLang::Const(c) = nodes[0] {
            assert_eq!(c.get_u64(), 0x4567_89AB_CDEF_0123);
            assert_eq!(c.width_bits(), 64);
        } else {
            panic!("expected constant; got {optimized:?}");
        }
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
    fn folds_constant_shifts() {
        let cases = [
            (
                RecExpr::from(vec![
                    SpirvLang::Const(ConstValue::new(3)),       // 0
                    SpirvLang::Const(ConstValue::new(1)),       // 1
                    SpirvLang::Shl([Id::from(0), Id::from(1)]), // 2 = 3 << 1
                ]),
                ConstValue::new(6),
            ),
            (
                RecExpr::from(vec![
                    SpirvLang::Const(ConstValue::new(4)),        // 0
                    SpirvLang::Const(ConstValue::new(1)),        // 1
                    SpirvLang::ShrU([Id::from(0), Id::from(1)]), // 2 = 4 >> 1
                ]),
                ConstValue::new(2),
            ),
            (
                RecExpr::from(vec![
                    SpirvLang::Const(ConstValue::new(0x8000_0000)), // 0
                    SpirvLang::Const(ConstValue::new(1)),           // 1
                    SpirvLang::ShrS([Id::from(0), Id::from(1)]),    // 2 = arithmetic right shift
                ]),
                ConstValue::new(0xC000_0000),
            ),
            (
                RecExpr::from(vec![
                    SpirvLang::Const(ConstValue::new64(3)),     // 0
                    SpirvLang::Const(ConstValue::new64(1)),     // 1
                    SpirvLang::Shl([Id::from(0), Id::from(1)]), // 2 = 3 << 1 (u64)
                ]),
                ConstValue::new_with_width(6, 64),
            ),
            (
                RecExpr::from(vec![
                    SpirvLang::Const(ConstValue::new64(4)),      // 0
                    SpirvLang::Const(ConstValue::new64(1)),      // 1
                    SpirvLang::ShrU([Id::from(0), Id::from(1)]), // 2 = 4 >> 1 (u64)
                ]),
                ConstValue::new_with_width(2, 64),
            ),
            (
                RecExpr::from(vec![
                    SpirvLang::Const(ConstValue::new64(0x8000_0000_0000_0000)), // 0
                    SpirvLang::Const(ConstValue::new64(1)),                     // 1
                    SpirvLang::ShrS([Id::from(0), Id::from(1)]), // arithmetic right shift
                ]),
                ConstValue::new_with_width(0xC000_0000_0000_0000, 64),
            ),
        ];

        for (expr, expected) in cases {
            let optimized = optimize_expr(&expr);
            assert_eq!(
                optimized,
                RecExpr::from(vec![SpirvLang::Const(expected)]),
                "failed to fold shift in {expr:?}"
            );
        }
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
    fn folds_bitwise_xor_identities() {
        // x ^ 0 => x
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),          // 0
            SpirvLang::Const(ConstValue::new(0)),          // 1
            SpirvLang::BitXor([Id::from(0), Id::from(1)]), // 2
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let has_symbol = nodes
            .iter()
            .any(|n| matches!(n, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")));
        assert!(has_symbol, "expected x ^ 0 to reduce to x");

        // c ^ c => 0, 3 ^ 5 => 6
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new(3)),          // 0
            SpirvLang::Const(ConstValue::new(5)),          // 1
            SpirvLang::BitXor([Id::from(0), Id::from(0)]), // 2
            SpirvLang::BitXor([Id::from(0), Id::from(1)]), // 3
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new(6))])
        );
    }

    #[test]
    fn rewrites_bxor_with_double_bitnot_operands() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Symbol(Symbol::from("y")), // 1
            SpirvLang::BitNot(Id::from(0)),        // 2 = ~x
            SpirvLang::BitNot(Id::from(1)),        // 3 = ~y
            SpirvLang::BitXor([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::BitXor([lhs, rhs])) = nodes.last() else {
            panic!("expected bxor root, got {:?}", nodes.last());
        };
        let x = Symbol::from("x");
        let y = Symbol::from("y");
        let lhs_is_x = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == x);
        let rhs_is_y = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == y);
        let lhs_is_y = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == y);
        let rhs_is_x = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == x);
        assert!(
            (lhs_is_x && rhs_is_y) || (lhs_is_y && rhs_is_x),
            "expected bxor between x and y, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_bxor_split_y_term_to_mask() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),          // 0
            SpirvLang::Symbol(Symbol::from("y")),          // 1
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]), // 2 = x & y
            SpirvLang::BitNot(Id::from(0)),                // 3 = ~x
            SpirvLang::BitAnd([Id::from(3), Id::from(1)]), // 4 = ~x & y
            SpirvLang::BitXor([Id::from(2), Id::from(4)]), // 5 = (x & y) ^ (~x & y)
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("y"))])
        );
    }

    #[test]
    fn rewrites_bxor_split_x_term_to_mask() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),          // 0
            SpirvLang::Symbol(Symbol::from("y")),          // 1
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]), // 2 = x & y
            SpirvLang::BitNot(Id::from(1)),                // 3 = ~y
            SpirvLang::BitAnd([Id::from(0), Id::from(3)]), // 4 = x & ~y
            SpirvLang::BitXor([Id::from(2), Id::from(4)]), // 5 = (x & y) ^ (x & ~y)
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn rewrites_band_of_bitnot_operands_to_negated_or() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")), // 0
            SpirvLang::Symbol(Symbol::from("b")), // 1
            SpirvLang::BitNot(Id::from(0)),        // 2 = ~a
            SpirvLang::BitNot(Id::from(1)),        // 3 = ~b
            SpirvLang::BitAnd([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::BitNot(inner)) = nodes.last() else {
            panic!("expected bnot root, got {:?}", nodes.last());
        };
        let SpirvLang::BitOr([lhs, rhs]) = nodes[usize::from(*inner)] else {
            panic!("expected bor under bnot, got {:?}", nodes[usize::from(*inner)]);
        };
        let a = Symbol::from("a");
        let b = Symbol::from("b");
        let lhs_is_a = matches!(nodes[usize::from(lhs)], SpirvLang::Symbol(sym) if sym == a);
        let rhs_is_b = matches!(nodes[usize::from(rhs)], SpirvLang::Symbol(sym) if sym == b);
        let lhs_is_b = matches!(nodes[usize::from(lhs)], SpirvLang::Symbol(sym) if sym == b);
        let rhs_is_a = matches!(nodes[usize::from(rhs)], SpirvLang::Symbol(sym) if sym == a);
        assert!(
            (lhs_is_a && rhs_is_b) || (lhs_is_b && rhs_is_a),
            "expected bor between a and b, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_bor_of_bitnot_operands_to_negated_and() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")), // 0
            SpirvLang::Symbol(Symbol::from("b")), // 1
            SpirvLang::BitNot(Id::from(0)),        // 2 = ~a
            SpirvLang::BitNot(Id::from(1)),        // 3 = ~b
            SpirvLang::BitOr([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::BitNot(inner)) = nodes.last() else {
            panic!("expected bnot root, got {:?}", nodes.last());
        };
        let SpirvLang::BitAnd([lhs, rhs]) = nodes[usize::from(*inner)] else {
            panic!("expected band under bnot, got {:?}", nodes[usize::from(*inner)]);
        };
        let a = Symbol::from("a");
        let b = Symbol::from("b");
        let lhs_is_a = matches!(nodes[usize::from(lhs)], SpirvLang::Symbol(sym) if sym == a);
        let rhs_is_b = matches!(nodes[usize::from(rhs)], SpirvLang::Symbol(sym) if sym == b);
        let lhs_is_b = matches!(nodes[usize::from(lhs)], SpirvLang::Symbol(sym) if sym == b);
        let rhs_is_a = matches!(nodes[usize::from(rhs)], SpirvLang::Symbol(sym) if sym == a);
        assert!(
            (lhs_is_a && rhs_is_b) || (lhs_is_b && rhs_is_a),
            "expected band between a and b, got {nodes:?}"
        );
    }

    #[test]
    fn bitwise_const_simplify_respects_width() {
        // x & all_ones (64-bit) => x
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),                       // 0
            SpirvLang::Const(ConstValue::new_with_width(u64::MAX, 64)), // 1
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),              // 2
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let has_symbol = nodes
            .iter()
            .any(|n| matches!(n, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")));
        assert!(has_symbol, "expected x & all_ones to reduce to x");

        // x | all_ones (64-bit) => all_ones (width preserved)
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),                       // 0
            SpirvLang::Const(ConstValue::new_with_width(u64::MAX, 64)), // 1
            SpirvLang::BitOr([Id::from(0), Id::from(1)]),               // 2
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(ConstValue::new_with_width(
                u64::MAX,
                64
            ))])
        );
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
    fn rewrites_large_sdiv_power_of_two_with_wide_mask() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Const(ConstValue::new_with_width(
                1u64 << 33, // 2^33
                64,
            )), // 1
            SpirvLang::SDiv([Id::from(0), Id::from(1)]), // 2 = x / 2^33
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        let mask_value = (1u128 << 33) - 1;
        let found_shr = nodes.iter().any(|node| {
            if let SpirvLang::ShrS([lhs, rhs]) = node {
                let rhs_is_shift = const_value(&runner.egraph, *rhs)
                    .is_some_and(|val| val.get() == 33 && val.width_bits() == 64);
                if !rhs_is_shift {
                    return false;
                }
                let add_eclass = runner.egraph.find(*lhs);
                let add_nodes = &runner.egraph[add_eclass].nodes;
                let has_mask = add_nodes.iter().any(|candidate| {
                    if let SpirvLang::Add([_, band]) = candidate {
                        let band_eclass = runner.egraph.find(*band);
                        runner.egraph[band_eclass].nodes.iter().any(|band_node| {
                            if let SpirvLang::BitAnd([_, mask]) = band_node {
                                const_value(&runner.egraph, *mask).is_some_and(|val| {
                                    val.get_u64() == mask_value as u64 && val.width_bits() == 64
                                })
                            } else {
                                false
                            }
                        })
                    } else {
                        false
                    }
                });
                rhs_is_shift && has_mask
            } else {
                false
            }
        });
        assert!(
            found_shr,
            "expected signed div by large power of two to rewrite into biased shift with wide mask"
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

        let expr_mod = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),        // 0
            SpirvLang::Const(ConstValue::new(1)),        // 1
            SpirvLang::SMod([Id::from(0), Id::from(1)]), // 2 = x mod 1
        ]);
        let runner_mod = Runner::default().with_expr(&expr_mod).run(&rewrites());
        let class_mod = runner_mod.egraph.find(runner_mod.roots[0]);
        let nodes_mod = &runner_mod.egraph[class_mod].nodes;
        let has_zero_mod = nodes_mod.iter().any(|n| {
            matches!(
                n,
                SpirvLang::Const(val) if val.get() == 0
            )
        });
        assert!(has_zero_mod, "smod by one should fold to zero");
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
    fn optimize_arith_block_folds_smod_by_one() {
        let int = 1;
        let c9 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(9)],
        );
        let c1 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(1)],
        );
        let smod = Instruction::new(
            rspirv::spirv::Op::SMod,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let block = vec![c9, c1, smod];
        let optimized = optimize_arith_block(&block).expect("optimization should succeed");
        let folded = optimized.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Constant
                && inst.result_id == Some(3)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        });
        assert!(folded, "smod by one should fold to zero with same id");
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
    fn skips_mask_then_shift_when_shift_meets_width() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),          // 0
            SpirvLang::Const(ConstValue::new(7)),          // 1 = 2^3 - 1
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]), // 2 = x & 7
            SpirvLang::Const(ConstValue::new(32)),         // 3 = shift amount
            SpirvLang::ShrU([Id::from(2), Id::from(3)]),   // 4 = (x & 7) >> 32
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let mut found_zero_mask = false;
        for class in runner.egraph.classes() {
            for node in &class.nodes {
                if let SpirvLang::BitAnd([_, rhs]) = node {
                    if const_value(&runner.egraph, *rhs)
                        .is_some_and(|c| c.get_u64() == 0 && c.width_bits() == 32)
                    {
                        found_zero_mask = true;
                    }
                }
            }
        }
        assert!(
            !found_zero_mask,
            "width-sized shifts should not rewrite mask into zero"
        );
    }

    #[test]
    fn preserves_full_mask_band_without_umod_rewrite() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),                     // 0
            SpirvLang::Const(ConstValue::new_with_width(0xFFFF, 16)), // 1 = all ones in 16 bits
            SpirvLang::BitAnd([Id::from(0), Id::from(1)]),            // 2 = x & 0xFFFF
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let class = runner.egraph.find(runner.roots[0]);
        let nodes = &runner.egraph[class].nodes;
        assert!(
            nodes.iter().any(|n| matches!(n, SpirvLang::BitAnd(_))),
            "bitand with full mask should remain available"
        );
        assert!(
            nodes.iter().all(|n| !matches!(n, SpirvLang::UMod(_))),
            "full-mask bitand must not rewrite into modulo with width-sized shift"
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
    fn rewrites_neg_add_const_to_sub() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Const(ConstValue::new(5)), // 1
            SpirvLang::Add([Id::from(0), Id::from(1)]),
            SpirvLang::Neg(Id::from(2)),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let neg_const = ConstValue::new_with_width(0u32.wrapping_sub(5) as u64, 32);
        let sub_expr = RecExpr::from(vec![
            SpirvLang::Const(neg_const),
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Sub([Id::from(0), Id::from(1)]),
        ]);
        let Some(sub_id) = runner.egraph.lookup_expr(&sub_expr) else {
            panic!("expected negated add to introduce const-sub form");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(sub_id));
    }

    #[test]
    fn rewrites_neg_sdiv_const() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Const(ConstValue::new(7)), // 1
            SpirvLang::SDiv([Id::from(0), Id::from(1)]),
            SpirvLang::Neg(Id::from(2)),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let neg_const = ConstValue::new_with_width(0u32.wrapping_sub(7) as u64, 32);
        let div_expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(neg_const),
            SpirvLang::SDiv([Id::from(0), Id::from(1)]),
        ]);
        let Some(div_id) = runner.egraph.lookup_expr(&div_expr) else {
            panic!("expected negated sdiv to introduce const divisor");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(div_id));
    }

    #[test]
    fn rewrites_sub_neg_left_to_neg_add() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")),       // 0
            SpirvLang::Symbol(Symbol::from("b")),       // 1
            SpirvLang::Neg(Id::from(0)),                // 2 = -a
            SpirvLang::Sub([Id::from(2), Id::from(1)]), // 3 = (-a) - b
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let neg_add: RecExpr<SpirvLang> = "(neg (+ a b))".parse().unwrap();
        let Some(neg_add_id) = runner.egraph.lookup_expr(&neg_add) else {
            panic!("expected -(a + b) to be introduced by rewrites");
        };
        assert_eq!(
            runner.egraph.find(root),
            runner.egraph.find(neg_add_id),
            "expected -(a + b) to be equivalent to (-a) - b"
        );
    }

    #[test]
    fn rewrites_select_true_branch() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(ConstValue::new_with_width(1, 1)), // 0
            SpirvLang::Symbol(Symbol::from("x")),               // 1
            SpirvLang::Symbol(Symbol::from("y")),               // 2
            SpirvLang::Select([Id::from(0), Id::from(1), Id::from(2)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn rewrites_select_negated_condition() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("c")), // 0
            SpirvLang::LogNot(Id::from(0)),       // 1 = !c
            SpirvLang::Symbol(Symbol::from("x")), // 2
            SpirvLang::Symbol(Symbol::from("y")), // 3
            SpirvLang::Select([Id::from(1), Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Select([cond, t, f])) = nodes.last() else {
            panic!("expected select root, got {:?}", nodes.last());
        };
        let cond_sym = matches!(nodes[usize::from(*cond)], SpirvLang::Symbol(sym) if sym == Symbol::from("c"));
        let t_sym = matches!(nodes[usize::from(*t)], SpirvLang::Symbol(sym) if sym == Symbol::from("y"));
        let f_sym = matches!(nodes[usize::from(*f)], SpirvLang::Symbol(sym) if sym == Symbol::from("x"));
        assert!(
            cond_sym && t_sym && f_sym,
            "expected select c y x after negated cond rewrite, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_select_bool_arms_to_condition() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("c")), // 0
            SpirvLang::Const(const_bool(true)),   // 1
            SpirvLang::Const(const_bool(false)),  // 2
            SpirvLang::Select([Id::from(0), Id::from(1), Id::from(2)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("c"))])
        );
    }

    #[test]
    fn rewrites_select_bool_arms_to_negated_condition() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("c")), // 0
            SpirvLang::Const(const_bool(false)),  // 1
            SpirvLang::Const(const_bool(true)),   // 2
            SpirvLang::Select([Id::from(0), Id::from(1), Id::from(2)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![
                SpirvLang::Symbol(Symbol::from("c")),
                SpirvLang::LogNot(Id::from(0))
            ])
        );
    }

    #[test]
    fn rewrites_phi_same_value() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Phi([Id::from(0), Id::from(0)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn rewrites_eq_self_to_true() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Eq([Id::from(0), Id::from(0)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(const_bool(true))])
        );
    }

    #[test]
    fn rewrites_ne_self_to_false() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Ne([Id::from(0), Id::from(0)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(const_bool(false))])
        );
    }

    #[test]
    fn rewrites_eq_with_negated_operands() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Symbol(Symbol::from("y")), // 1
            SpirvLang::Neg(Id::from(0)),          // 2 = -x
            SpirvLang::Neg(Id::from(1)),          // 3 = -y
            SpirvLang::Eq([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Eq([lhs, rhs])) = nodes.last() else {
            panic!("expected eq root, got {:?}", nodes.last());
        };
        let x = Symbol::from("x");
        let y = Symbol::from("y");
        let lhs_is_x = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == x);
        let rhs_is_y = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == y);
        let lhs_is_y = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == y);
        let rhs_is_x = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == x);
        assert!(
            (lhs_is_x && rhs_is_y) || (lhs_is_y && rhs_is_x),
            "expected eq between x and y, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_ne_with_negated_operands() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Symbol(Symbol::from("y")), // 1
            SpirvLang::Neg(Id::from(0)),          // 2 = -x
            SpirvLang::Neg(Id::from(1)),          // 3 = -y
            SpirvLang::Ne([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Ne([lhs, rhs])) = nodes.last() else {
            panic!("expected ne root, got {:?}", nodes.last());
        };
        let x = Symbol::from("x");
        let y = Symbol::from("y");
        let lhs_is_x = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == x);
        let rhs_is_y = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == y);
        let lhs_is_y = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == y);
        let rhs_is_x = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == x);
        assert!(
            (lhs_is_x && rhs_is_y) || (lhs_is_y && rhs_is_x),
            "expected ne between x and y, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_eq_with_bitnot_operands() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Symbol(Symbol::from("y")), // 1
            SpirvLang::BitNot(Id::from(0)),        // 2 = ~x
            SpirvLang::BitNot(Id::from(1)),        // 3 = ~y
            SpirvLang::Eq([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Eq([lhs, rhs])) = nodes.last() else {
            panic!("expected eq root, got {:?}", nodes.last());
        };
        let x = Symbol::from("x");
        let y = Symbol::from("y");
        let lhs_is_x = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == x);
        let rhs_is_y = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == y);
        let lhs_is_y = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == y);
        let rhs_is_x = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == x);
        assert!(
            (lhs_is_x && rhs_is_y) || (lhs_is_y && rhs_is_x),
            "expected eq between x and y, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_ne_with_bitnot_operands() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Symbol(Symbol::from("y")), // 1
            SpirvLang::BitNot(Id::from(0)),        // 2 = ~x
            SpirvLang::BitNot(Id::from(1)),        // 3 = ~y
            SpirvLang::Ne([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Ne([lhs, rhs])) = nodes.last() else {
            panic!("expected ne root, got {:?}", nodes.last());
        };
        let x = Symbol::from("x");
        let y = Symbol::from("y");
        let lhs_is_x = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == x);
        let rhs_is_y = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == y);
        let lhs_is_y = matches!(nodes[usize::from(*lhs)], SpirvLang::Symbol(sym) if sym == y);
        let rhs_is_x = matches!(nodes[usize::from(*rhs)], SpirvLang::Symbol(sym) if sym == x);
        assert!(
            (lhs_is_x && rhs_is_y) || (lhs_is_y && rhs_is_x),
            "expected ne between x and y, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_eq_with_shared_add_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 3 = x + y
            SpirvLang::Add([Id::from(0), Id::from(2)]), // 4 = x + z
            SpirvLang::Eq([Id::from(3), Id::from(4)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(eq y z)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(eq z y)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected eq between y and z to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_ne_with_shared_add_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Add([Id::from(0), Id::from(1)]), // 3 = x + y
            SpirvLang::Add([Id::from(0), Id::from(2)]), // 4 = x + z
            SpirvLang::Ne([Id::from(3), Id::from(4)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(ne y z)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(ne z y)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected ne between y and z to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_eq_with_shared_sub_left_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 3 = x - y
            SpirvLang::Sub([Id::from(0), Id::from(2)]), // 4 = x - z
            SpirvLang::Eq([Id::from(3), Id::from(4)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(eq y z)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(eq z y)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected eq between y and z to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_ne_with_shared_sub_left_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 3 = x - y
            SpirvLang::Sub([Id::from(0), Id::from(2)]), // 4 = x - z
            SpirvLang::Ne([Id::from(3), Id::from(4)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(ne y z)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(ne z y)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected ne between y and z to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_eq_with_shared_sub_right_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Sub([Id::from(1), Id::from(0)]), // 3 = y - x
            SpirvLang::Sub([Id::from(2), Id::from(0)]), // 4 = z - x
            SpirvLang::Eq([Id::from(3), Id::from(4)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(eq y z)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(eq z y)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected eq between y and z to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_ne_with_shared_sub_right_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Sub([Id::from(1), Id::from(0)]), // 3 = y - x
            SpirvLang::Sub([Id::from(2), Id::from(0)]), // 4 = z - x
            SpirvLang::Ne([Id::from(3), Id::from(4)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(ne y z)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(ne z y)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected ne between y and z to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_eq_with_shared_bxor_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),            // 0
            SpirvLang::Symbol(Symbol::from("y")),            // 1
            SpirvLang::Symbol(Symbol::from("z")),            // 2
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),   // 3 = x ^ y
            SpirvLang::BitXor([Id::from(0), Id::from(2)]),   // 4 = x ^ z
            SpirvLang::Eq([Id::from(3), Id::from(4)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(eq y z)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(eq z y)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected eq between y and z to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_ne_with_shared_bxor_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),            // 0
            SpirvLang::Symbol(Symbol::from("y")),            // 1
            SpirvLang::Symbol(Symbol::from("z")),            // 2
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),   // 3 = x ^ y
            SpirvLang::BitXor([Id::from(0), Id::from(2)]),   // 4 = x ^ z
            SpirvLang::Ne([Id::from(3), Id::from(4)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(ne y z)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(ne z y)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected ne between y and z to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_eq_bxor_zero_to_eq() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),            // 0
            SpirvLang::Symbol(Symbol::from("y")),            // 1
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),   // 2 = x ^ y
            SpirvLang::Const(ConstValue::new(0)),            // 3
            SpirvLang::Eq([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(eq x y)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(eq y x)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected eq between x and y to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_ne_bxor_zero_to_ne() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),            // 0
            SpirvLang::Symbol(Symbol::from("y")),            // 1
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),   // 2 = x ^ y
            SpirvLang::Const(ConstValue::new(0)),            // 3
            SpirvLang::Ne([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(ne x y)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(ne y x)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected ne between x and y to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_eq_sub_zero_to_eq() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = x - y
            SpirvLang::Const(ConstValue::new(0)),       // 3
            SpirvLang::Eq([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(eq x y)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(eq y x)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected eq between x and y to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_ne_sub_zero_to_ne() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Sub([Id::from(0), Id::from(1)]), // 2 = x - y
            SpirvLang::Const(ConstValue::new(0)),       // 3
            SpirvLang::Ne([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(ne x y)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(ne y x)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected ne between x and y to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_eq_neg_zero_to_eq() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Neg(Id::from(0)),          // 1 = -x
            SpirvLang::Const(ConstValue::new(0)), // 2
            SpirvLang::Eq([Id::from(1), Id::from(2)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(eq x 0)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(eq 0 x)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected eq between x and 0 to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_ne_neg_zero_to_ne() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")), // 0
            SpirvLang::Neg(Id::from(0)),          // 1 = -x
            SpirvLang::Const(ConstValue::new(0)), // 2
            SpirvLang::Ne([Id::from(1), Id::from(2)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(ne x 0)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(ne 0 x)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected ne between x and 0 to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_eq_bxor_all_ones_to_bnot() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),                      // 0
            SpirvLang::Symbol(Symbol::from("y")),                      // 1
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),             // 2 = x ^ y
            SpirvLang::Const(ConstValue::new_with_width(u32::MAX as u64, 32)), // 3
            SpirvLang::Eq([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let mut expected_id = None;
        for candidate in [
            "(eq x (bnot y))",
            "(eq (bnot y) x)",
            "(eq y (bnot x))",
            "(eq (bnot x) y)",
        ] {
            let expr: RecExpr<SpirvLang> = candidate.parse().unwrap();
            if let Some(id) = runner.egraph.lookup_expr(&expr) {
                expected_id = Some(id);
                break;
            }
        }
        let Some(expected_id) = expected_id else {
            panic!("expected eq with bnot to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_ne_bxor_all_ones_to_bnot() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),                      // 0
            SpirvLang::Symbol(Symbol::from("y")),                      // 1
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),             // 2 = x ^ y
            SpirvLang::Const(ConstValue::new_with_width(u32::MAX as u64, 32)), // 3
            SpirvLang::Ne([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let mut expected_id = None;
        for candidate in [
            "(ne x (bnot y))",
            "(ne (bnot y) x)",
            "(ne y (bnot x))",
            "(ne (bnot x) y)",
        ] {
            let expr: RecExpr<SpirvLang> = candidate.parse().unwrap();
            if let Some(id) = runner.egraph.lookup_expr(&expr) {
                expected_id = Some(id);
                break;
            }
        }
        let Some(expected_id) = expected_id else {
            panic!("expected ne with bnot to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_eq_negated_const_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),     // 0
            SpirvLang::Const(ConstValue::new(5)),     // 1
            SpirvLang::Neg(Id::from(0)),              // 2 = -x
            SpirvLang::Eq([Id::from(2), Id::from(1)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let class = runner.egraph.find(root);
        let expected_const = map_const(ConstValue::new(5), u64::wrapping_neg);
        let mut found = false;
        for node in &runner.egraph[class].nodes {
            let SpirvLang::Eq([lhs, rhs]) = node else {
                continue;
            };
            let lhs_is_x = has_symbol(&runner.egraph, *lhs);
            let rhs_is_x = has_symbol(&runner.egraph, *rhs);
            let lhs_const = const_value(&runner.egraph, *lhs);
            let rhs_const = const_value(&runner.egraph, *rhs);
            if (lhs_is_x && rhs_const == Some(expected_const))
                || (rhs_is_x && lhs_const == Some(expected_const))
            {
                found = true;
                break;
            }
        }
        assert!(found, "expected eq to compare x against negated const");
    }

    #[test]
    fn rewrites_ne_negated_const_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),     // 0
            SpirvLang::Const(ConstValue::new(9)),     // 1
            SpirvLang::Neg(Id::from(0)),              // 2 = -x
            SpirvLang::Ne([Id::from(2), Id::from(1)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let class = runner.egraph.find(root);
        let expected_const = map_const(ConstValue::new(9), u64::wrapping_neg);
        let mut found = false;
        for node in &runner.egraph[class].nodes {
            let SpirvLang::Ne([lhs, rhs]) = node else {
                continue;
            };
            let lhs_is_x = has_symbol(&runner.egraph, *lhs);
            let rhs_is_x = has_symbol(&runner.egraph, *rhs);
            let lhs_const = const_value(&runner.egraph, *lhs);
            let rhs_const = const_value(&runner.egraph, *rhs);
            if (lhs_is_x && rhs_const == Some(expected_const))
                || (rhs_is_x && lhs_const == Some(expected_const))
            {
                found = true;
                break;
            }
        }
        assert!(found, "expected ne to compare x against negated const");
    }

    #[test]
    fn rewrites_eq_bitnot_const_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),     // 0
            SpirvLang::Const(ConstValue::new(6)),     // 1
            SpirvLang::BitNot(Id::from(0)),           // 2 = ~x
            SpirvLang::Eq([Id::from(2), Id::from(1)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let class = runner.egraph.find(root);
        let expected_const = map_const(ConstValue::new(6), |value| !value);
        let mut found = false;
        for node in &runner.egraph[class].nodes {
            let SpirvLang::Eq([lhs, rhs]) = node else {
                continue;
            };
            let lhs_is_x = has_symbol(&runner.egraph, *lhs);
            let rhs_is_x = has_symbol(&runner.egraph, *rhs);
            let lhs_const = const_value(&runner.egraph, *lhs);
            let rhs_const = const_value(&runner.egraph, *rhs);
            if (lhs_is_x && rhs_const == Some(expected_const))
                || (rhs_is_x && lhs_const == Some(expected_const))
            {
                found = true;
                break;
            }
        }
        assert!(found, "expected eq to compare x against bitnot const");
    }

    #[test]
    fn rewrites_ne_bitnot_const_operand() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),     // 0
            SpirvLang::Const(ConstValue::new(12)),    // 1
            SpirvLang::BitNot(Id::from(0)),           // 2 = ~x
            SpirvLang::Ne([Id::from(2), Id::from(1)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let class = runner.egraph.find(root);
        let expected_const = map_const(ConstValue::new(12), |value| !value);
        let mut found = false;
        for node in &runner.egraph[class].nodes {
            let SpirvLang::Ne([lhs, rhs]) = node else {
                continue;
            };
            let lhs_is_x = has_symbol(&runner.egraph, *lhs);
            let rhs_is_x = has_symbol(&runner.egraph, *rhs);
            let lhs_const = const_value(&runner.egraph, *lhs);
            let rhs_const = const_value(&runner.egraph, *rhs);
            if (lhs_is_x && rhs_const == Some(expected_const))
                || (rhs_is_x && lhs_const == Some(expected_const))
            {
                found = true;
                break;
            }
        }
        assert!(found, "expected ne to compare x against bitnot const");
    }

    #[test]
    fn rewrites_eq_bxor_with_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),           // 0
            SpirvLang::Const(ConstValue::new(5)),           // 1
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),  // 2 = x ^ 5
            SpirvLang::Const(ConstValue::new(9)),           // 3
            SpirvLang::Eq([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let class = runner.egraph.find(root);
        let expected_const = combine_consts(ConstValue::new(5), ConstValue::new(9), |a, b| a ^ b);
        let mut found = false;
        for node in &runner.egraph[class].nodes {
            let SpirvLang::Eq([lhs, rhs]) = node else {
                continue;
            };
            let lhs_is_x = has_symbol(&runner.egraph, *lhs);
            let rhs_is_x = has_symbol(&runner.egraph, *rhs);
            let lhs_const = const_value(&runner.egraph, *lhs);
            let rhs_const = const_value(&runner.egraph, *rhs);
            if (lhs_is_x && rhs_const == Some(expected_const))
                || (rhs_is_x && lhs_const == Some(expected_const))
            {
                found = true;
                break;
            }
        }
        assert!(found, "expected eq to compare x against xor-folded const");
    }

    #[test]
    fn rewrites_ne_bxor_with_constants() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),           // 0
            SpirvLang::Const(ConstValue::new(3)),           // 1
            SpirvLang::BitXor([Id::from(0), Id::from(1)]),  // 2 = x ^ 3
            SpirvLang::Const(ConstValue::new(12)),          // 3
            SpirvLang::Ne([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let class = runner.egraph.find(root);
        let expected_const = combine_consts(ConstValue::new(3), ConstValue::new(12), |a, b| a ^ b);
        let mut found = false;
        for node in &runner.egraph[class].nodes {
            let SpirvLang::Ne([lhs, rhs]) = node else {
                continue;
            };
            let lhs_is_x = has_symbol(&runner.egraph, *lhs);
            let rhs_is_x = has_symbol(&runner.egraph, *rhs);
            let lhs_const = const_value(&runner.egraph, *lhs);
            let rhs_const = const_value(&runner.egraph, *rhs);
            if (lhs_is_x && rhs_const == Some(expected_const))
                || (rhs_is_x && lhs_const == Some(expected_const))
            {
                found = true;
                break;
            }
        }
        assert!(found, "expected ne to compare x against xor-folded const");
    }

    #[test]
    fn rewrites_logical_and_with_true() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(const_bool(true)),
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::LogAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn rewrites_logical_not_true() {
        let expr = RecExpr::from(vec![
            SpirvLang::Const(const_bool(true)),
            SpirvLang::LogNot(Id::from(0)),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(const_bool(false))])
        );
    }

    #[test]
    fn rewrites_logical_eq_with_true() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(const_bool(true)),
            SpirvLang::LogEq([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn rewrites_logical_eq_with_false() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(const_bool(false)),
            SpirvLang::LogEq([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![
                SpirvLang::Symbol(Symbol::from("x")),
                SpirvLang::LogNot(Id::from(0))
            ])
        );
    }

    #[test]
    fn rewrites_logical_ne_with_true() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(const_bool(true)),
            SpirvLang::LogNe([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![
                SpirvLang::Symbol(Symbol::from("x")),
                SpirvLang::LogNot(Id::from(0))
            ])
        );
    }

    #[test]
    fn rewrites_logical_ne_with_false() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Const(const_bool(false)),
            SpirvLang::LogNe([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Symbol(Symbol::from("x"))])
        );
    }

    #[test]
    fn rewrites_logeq_of_negated_operands() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")), // 0
            SpirvLang::Symbol(Symbol::from("b")), // 1
            SpirvLang::LogNot(Id::from(0)),       // 2 = !a
            SpirvLang::LogNot(Id::from(1)),       // 3 = !b
            SpirvLang::LogEq([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(leq a b)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(leq b a)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected leq to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_logne_of_negated_operands() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")), // 0
            SpirvLang::Symbol(Symbol::from("b")), // 1
            SpirvLang::LogNot(Id::from(0)),       // 2 = !a
            SpirvLang::LogNot(Id::from(1)),       // 3 = !b
            SpirvLang::LogNe([Id::from(2), Id::from(3)]),
        ]);
        let runner = Runner::default().with_expr(&expr).run(&rewrites());
        let root = runner.roots[0];
        let expected: RecExpr<SpirvLang> = "(lne a b)".parse().unwrap();
        let expected_comm: RecExpr<SpirvLang> = "(lne b a)".parse().unwrap();
        let Some(expected_id) = runner
            .egraph
            .lookup_expr(&expected)
            .or_else(|| runner.egraph.lookup_expr(&expected_comm))
        else {
            panic!("expected lne to be introduced by rewrites");
        };
        assert_eq!(runner.egraph.find(root), runner.egraph.find(expected_id));
    }

    #[test]
    fn rewrites_logical_and_with_negation() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::LogNot(Id::from(0)),
            SpirvLang::LogAnd([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(const_bool(false))])
        );
    }

    #[test]
    fn rewrites_logical_or_with_negation() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::LogNot(Id::from(0)),
            SpirvLang::LogOr([Id::from(0), Id::from(1)]),
        ]);
        let optimized = optimize_expr(&expr);
        assert_eq!(
            optimized,
            RecExpr::from(vec![SpirvLang::Const(const_bool(true))])
        );
    }

    #[test]
    fn rewrites_logand_of_negations_to_negated_or() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")), // 0
            SpirvLang::Symbol(Symbol::from("b")), // 1
            SpirvLang::LogNot(Id::from(0)),       // 2
            SpirvLang::LogNot(Id::from(1)),       // 3
            SpirvLang::LogAnd([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::LogNot(inner)) = nodes.last() else {
            panic!("expected lognot root, got {:?}", nodes.last());
        };
        let SpirvLang::LogOr([lhs, rhs]) = nodes[usize::from(*inner)] else {
            panic!("expected logor under lognot, got {:?}", nodes[usize::from(*inner)]);
        };
        let a = Symbol::from("a");
        let b = Symbol::from("b");
        let lhs_is_a = matches!(nodes[usize::from(lhs)], SpirvLang::Symbol(sym) if sym == a);
        let rhs_is_b = matches!(nodes[usize::from(rhs)], SpirvLang::Symbol(sym) if sym == b);
        let lhs_is_b = matches!(nodes[usize::from(lhs)], SpirvLang::Symbol(sym) if sym == b);
        let rhs_is_a = matches!(nodes[usize::from(rhs)], SpirvLang::Symbol(sym) if sym == a);
        assert!(
            (lhs_is_a && rhs_is_b) || (lhs_is_b && rhs_is_a),
            "expected logor between a and b, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_logor_of_negations_to_negated_and() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("a")), // 0
            SpirvLang::Symbol(Symbol::from("b")), // 1
            SpirvLang::LogNot(Id::from(0)),       // 2
            SpirvLang::LogNot(Id::from(1)),       // 3
            SpirvLang::LogOr([Id::from(2), Id::from(3)]),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::LogNot(inner)) = nodes.last() else {
            panic!("expected lognot root, got {:?}", nodes.last());
        };
        let SpirvLang::LogAnd([lhs, rhs]) = nodes[usize::from(*inner)] else {
            panic!("expected logand under lognot, got {:?}", nodes[usize::from(*inner)]);
        };
        let a = Symbol::from("a");
        let b = Symbol::from("b");
        let lhs_is_a = matches!(nodes[usize::from(lhs)], SpirvLang::Symbol(sym) if sym == a);
        let rhs_is_b = matches!(nodes[usize::from(rhs)], SpirvLang::Symbol(sym) if sym == b);
        let lhs_is_b = matches!(nodes[usize::from(lhs)], SpirvLang::Symbol(sym) if sym == b);
        let rhs_is_a = matches!(nodes[usize::from(rhs)], SpirvLang::Symbol(sym) if sym == a);
        assert!(
            (lhs_is_a && rhs_is_b) || (lhs_is_b && rhs_is_a),
            "expected logand between a and b, got {nodes:?}"
        );
    }

    #[test]
    fn rewrites_lognot_eq_to_ne() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Symbol(Symbol::from("y")),
            SpirvLang::Eq([Id::from(0), Id::from(1)]),
            SpirvLang::LogNot(Id::from(2)),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Ne([lhs, rhs])) = nodes.last() else {
            panic!("expected optimized expr to end in ne, got {nodes:?}");
        };
        let lhs_node = &nodes[usize::from(*lhs)];
        let rhs_node = &nodes[usize::from(*rhs)];
        let symbols = [lhs_node, rhs_node];
        assert!(
            symbols
                .iter()
                .any(|n| matches!(n, SpirvLang::Symbol(sym) if *sym == Symbol::from("x")))
                && symbols
                    .iter()
                    .any(|n| matches!(n, SpirvLang::Symbol(sym) if *sym == Symbol::from("y"))),
            "expected ne to compare x and y, got lhs={lhs_node:?} rhs={rhs_node:?}"
        );
    }

    #[test]
    fn rewrites_lognot_slt_to_sge() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),
            SpirvLang::Symbol(Symbol::from("y")),
            SpirvLang::SLt([Id::from(0), Id::from(1)]),
            SpirvLang::LogNot(Id::from(2)),
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::SGe([lhs, rhs])) = nodes.last() else {
            panic!("expected optimized expr to end in sge, got {nodes:?}");
        };
        let lhs_node = &nodes[usize::from(*lhs)];
        let rhs_node = &nodes[usize::from(*rhs)];
        assert!(
            matches!(lhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("x"))
                && matches!(rhs_node, SpirvLang::Symbol(sym) if *sym == Symbol::from("y")),
            "expected sge to compare x >= y, got lhs={lhs_node:?} rhs={rhs_node:?}"
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
    fn factors_add_with_implicit_unit_coefficient() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(2)),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = 2x
            SpirvLang::Add([Id::from(2), Id::from(0)]), // 3 = 2x + x
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (sym_node, const_node) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(c)) => (sym, c),
            (SpirvLang::Const(c), SpirvLang::Symbol(sym)) => (sym, c),
            other => panic!("unexpected operands after factoring implicit unit: {other:?}"),
        };
        assert_eq!(sym_node, &Symbol::from("x"));
        assert_eq!(const_node.get(), 3);
    }

    #[test]
    fn factors_sub_with_implicit_unit_coefficient() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(7)),       // 1
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 2 = 7x
            SpirvLang::Sub([Id::from(2), Id::from(0)]), // 3 = 7x - x
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root, got {:?}", nodes.last());
        };
        let (sym_node, const_node) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(c)) => (sym, c),
            (SpirvLang::Const(c), SpirvLang::Symbol(sym)) => (sym, c),
            other => panic!("unexpected operands after factoring implicit unit in sub: {other:?}"),
        };
        assert_eq!(sym_node, &Symbol::from("x"));
        assert_eq!(const_node.get(), 6);
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
    fn factors_symbolic_multiplier_when_multiplicands_commuted() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Mul([Id::from(1), Id::from(0)]), // 3 = y * x
            SpirvLang::Mul([Id::from(2), Id::from(0)]), // 4 = z * x
            SpirvLang::Add([Id::from(3), Id::from(4)]), // 5 = y*x + z*x
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
    fn factors_symbolic_multiplier_from_subtraction_when_commuted() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Mul([Id::from(1), Id::from(0)]), // 3 = y * x
            SpirvLang::Mul([Id::from(2), Id::from(0)]), // 4 = z * x
            SpirvLang::Sub([Id::from(3), Id::from(4)]), // 5 = y*x - z*x
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
    fn factors_symbolic_multiplier_from_addition_when_commuted() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Symbol(Symbol::from("y")),       // 1
            SpirvLang::Symbol(Symbol::from("z")),       // 2
            SpirvLang::Mul([Id::from(1), Id::from(0)]), // 3 = y * x
            SpirvLang::Mul([Id::from(2), Id::from(0)]), // 4 = z * x
            SpirvLang::Add([Id::from(3), Id::from(4)]), // 5 = y*x + z*x
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
    fn factors_common_multiplier_from_three_addends() {
        let expr = RecExpr::from(vec![
            SpirvLang::Symbol(Symbol::from("x")),       // 0
            SpirvLang::Const(ConstValue::new(2)),       // 1
            SpirvLang::Const(ConstValue::new(3)),       // 2
            SpirvLang::Const(ConstValue::new(5)),       // 3
            SpirvLang::Mul([Id::from(0), Id::from(1)]), // 4 = x * 2
            SpirvLang::Mul([Id::from(0), Id::from(2)]), // 5 = x * 3
            SpirvLang::Add([Id::from(4), Id::from(5)]), // 6 = x*2 + x*3
            SpirvLang::Mul([Id::from(0), Id::from(3)]), // 7 = x * 5
            SpirvLang::Add([Id::from(6), Id::from(7)]), // 8 = (x*2 + x*3) + x*5
        ]);
        let optimized = optimize_expr(&expr);
        let nodes = optimized.as_ref();
        let Some(SpirvLang::Mul([lhs, rhs])) = nodes.last() else {
            panic!("expected mul root after factoring, got {:?}", nodes.last());
        };
        let (symbol, constant) = match (&nodes[usize::from(*lhs)], &nodes[usize::from(*rhs)]) {
            (SpirvLang::Symbol(sym), SpirvLang::Const(val)) => (sym, val),
            (SpirvLang::Const(val), SpirvLang::Symbol(sym)) => (sym, val),
            other => panic!("unexpected operands after factoring three addends: {other:?}"),
        };
        assert_eq!(symbol, &Symbol::from("x"));
        assert_eq!(constant.get(), 10, "factor should sum constants to 10");
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
    fn optimize_arith_block_folds_bitwise_xor_constants() {
        let int = 1;
        let c3 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(3)],
        );
        let c5 = Instruction::new(
            rspirv::spirv::Op::Constant,
            Some(int),
            Some(2),
            vec![rspirv::dr::Operand::LiteralBit32(5)],
        );
        let bxor = Instruction::new(
            rspirv::spirv::Op::BitwiseXor,
            Some(int),
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        );
        let optimized = optimize_arith_block(&[c3, c5, bxor]).expect("bitwise xor supported");
        assert_eq!(optimized.len(), 1);
        let folded = &optimized[0];
        assert_eq!(folded.class.opcode, rspirv::spirv::Op::Constant);
        assert_eq!(folded.operands, vec![rspirv::dr::Operand::LiteralBit32(6)]);
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
