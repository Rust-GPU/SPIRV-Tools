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
                match u.choose(&[0u8, 1, 2, 3])? {
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
        "-" = Sub([Id; 2]),
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
        rewrite!("mul-one"; "(* ?a ?b)" => { MulOne { a: var("?a"), b: var("?b") } }),
        rewrite!("mul-zero"; "(* ?a ?b)" => { MulZero { a: var("?a"), b: var("?b") } }),
        rewrite!("add-fold"; "(+ ?a ?b)" => { FoldAdd }),
        rewrite!("mul-fold"; "(* ?a ?b)" => { FoldMul }),
        rewrite!("sub-fold"; "(- ?a ?b)" => { FoldSub }),
        rewrite!("sub-zero-right"; "(- ?a ?b)" => "?a" if is_const_zero(var("?b"))),
    ]
}

struct FoldAdd;
struct FoldMul;
struct FoldSub;
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
}
