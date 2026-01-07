//! Tests for logical operation optimization rules (LogicalAnd, LogicalOr, LogicalNot, Select).
//! Based on C++ fold_test.cpp logical patterns.

use super::common::{OptimizedModule, OptimizerEnvGuard, TestModuleBuilder};
use rspirv::spirv::Op;

// =============================================================================
// LogicalNot Tests
// =============================================================================

#[test]
fn logical_not_not_cancels() {
    // !(!x) should fold to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let not_x = b.builder.logical_not(b.bool_ty, None, x).expect("not1");
    let _ = b.builder.logical_not(b.bool_ty, None, not_x).expect("not2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalNot),
        "!(!x) should cancel to x"
    );
}

#[test]
fn logical_not_true_folds() {
    // !true should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let t = b.const_true();
    let _ = b.builder.logical_not(b.bool_ty, None, t).expect("not");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalNot),
        "!true should fold to false"
    );
}

#[test]
fn logical_not_false_folds() {
    // !false should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let f = b.const_false();
    let _ = b.builder.logical_not(b.bool_ty, None, f).expect("not");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalNot),
        "!false should fold to true"
    );
}

// =============================================================================
// LogicalAnd Tests
// =============================================================================

#[test]
fn logical_and_with_true_simplifies() {
    // x && true should simplify to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let t = b.const_true();
    let _ = b.builder.logical_and(b.bool_ty, None, x, t).expect("and");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalAnd),
        "x && true should simplify to x"
    );
}

#[test]
fn logical_and_with_false_simplifies() {
    // x && false should simplify to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let f = b.const_false();
    let _ = b.builder.logical_and(b.bool_ty, None, x, f).expect("and");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalAnd),
        "x && false should simplify to false"
    );
}

#[test]
fn logical_and_same_simplifies() {
    // x && x should simplify to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let _ = b.builder.logical_and(b.bool_ty, None, x, x).expect("and");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalAnd),
        "x && x should simplify to x"
    );
}

#[test]
fn logical_and_complement_to_false() {
    // x && !x should simplify to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let not_x = b.builder.logical_not(b.bool_ty, None, x).expect("not");
    let _ = b.builder.logical_and(b.bool_ty, None, x, not_x).expect("and");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should fold away both operations
    assert!(
        !result.has_opcode(Op::LogicalAnd),
        "x && !x should simplify to false"
    );
}

// =============================================================================
// LogicalOr Tests
// =============================================================================

#[test]
fn logical_or_with_true_simplifies() {
    // x || true should simplify to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let t = b.const_true();
    let _ = b.builder.logical_or(b.bool_ty, None, x, t).expect("or");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalOr),
        "x || true should simplify to true"
    );
}

#[test]
fn logical_or_with_false_simplifies() {
    // x || false should simplify to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let f = b.const_false();
    let _ = b.builder.logical_or(b.bool_ty, None, x, f).expect("or");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalOr),
        "x || false should simplify to x"
    );
}

#[test]
fn logical_or_same_simplifies() {
    // x || x should simplify to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let _ = b.builder.logical_or(b.bool_ty, None, x, x).expect("or");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalOr),
        "x || x should simplify to x"
    );
}

#[test]
fn logical_or_complement_to_true() {
    // x || !x should simplify to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let not_x = b.builder.logical_not(b.bool_ty, None, x).expect("not");
    let _ = b.builder.logical_or(b.bool_ty, None, x, not_x).expect("or");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should fold away both operations
    assert!(
        !result.has_opcode(Op::LogicalOr),
        "x || !x should simplify to true"
    );
}

// =============================================================================
// LogicalEqual / LogicalNotEqual Tests
// =============================================================================

#[test]
fn logical_equal_same_folds_to_true() {
    // x == x (logical) should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let _ = b.builder.logical_equal(b.bool_ty, None, x, x).expect("eq");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalEqual),
        "x == x (logical) should fold to true"
    );
}

#[test]
fn logical_not_equal_same_folds_to_false() {
    // x != x (logical) should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty]);
    let x = params[0];
    let _ = b.builder.logical_not_equal(b.bool_ty, None, x, x).expect("ne");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalNotEqual),
        "x != x (logical) should fold to false"
    );
}

// =============================================================================
// Select Tests
// =============================================================================

#[test]
fn select_with_true_condition_folds() {
    // select(true, a, b) should fold to a
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty]);
    let (a, bval) = (params[0], params[1]);
    let t = b.const_true();
    let _ = b.builder.select(b.int_ty, None, t, a, bval).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::Select),
        "select(true, a, b) should fold to a"
    );
}

#[test]
fn select_with_false_condition_folds() {
    // select(false, a, b) should fold to b
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty]);
    let (a, bval) = (params[0], params[1]);
    let f = b.const_false();
    let _ = b.builder.select(b.int_ty, None, f, a, bval).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::Select),
        "select(false, a, b) should fold to b"
    );
}

#[test]
fn select_with_same_branches_folds() {
    // select(cond, a, a) should fold to a
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty]);
    let (cond, a) = (params[0], params[1]);
    let _ = b.builder.select(b.int_ty, None, cond, a, a).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::Select),
        "select(cond, a, a) should fold to a"
    );
}

#[test]
fn select_with_negated_condition_normalizes() {
    // select(!cond, a, b) can be transformed to select(cond, b, a)
    // This test verifies that select with negated condition gets optimized
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty, b.int_ty]);
    let (cond, a, bval) = (params[0], params[1], params[2]);
    let not_cond = b.builder.logical_not(b.bool_ty, None, cond).expect("not");
    let _ = b.builder.select(b.int_ty, None, not_cond, a, bval).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // The negation should be eliminated
    assert!(
        !result.has_opcode(Op::LogicalNot),
        "select(!cond, a, b) should eliminate the negation"
    );
}

// =============================================================================
// Absorption and De Morgan Tests
// =============================================================================

#[test]
fn logical_and_absorption_left() {
    // x && (x || y) should simplify to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.bool_ty]);
    let (x, y) = (params[0], params[1]);
    let or = b.builder.logical_or(b.bool_ty, None, x, y).expect("or");
    let _ = b.builder.logical_and(b.bool_ty, None, x, or).expect("and");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should simplify away both operations
    assert!(
        !result.has_opcode(Op::LogicalOr) && !result.has_opcode(Op::LogicalAnd),
        "x && (x || y) should simplify to x"
    );
}

#[test]
fn logical_or_absorption_left() {
    // x || (x && y) should simplify to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.bool_ty]);
    let (x, y) = (params[0], params[1]);
    let and = b.builder.logical_and(b.bool_ty, None, x, y).expect("and");
    let _ = b.builder.logical_or(b.bool_ty, None, x, and).expect("or");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should simplify away both operations
    assert!(
        !result.has_opcode(Op::LogicalAnd) && !result.has_opcode(Op::LogicalOr),
        "x || (x && y) should simplify to x"
    );
}

// =============================================================================
// Constant Folding Tests
// =============================================================================

#[test]
fn constant_logical_and_folds() {
    // true && false should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let t = b.const_true();
    let f = b.const_false();
    let _ = b.builder.logical_and(b.bool_ty, None, t, f).expect("and");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalAnd),
        "constant logical and should fold"
    );
}

#[test]
fn constant_logical_or_folds() {
    // false || true should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let t = b.const_true();
    let f = b.const_false();
    let _ = b.builder.logical_or(b.bool_ty, None, f, t).expect("or");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalOr),
        "constant logical or should fold"
    );
}

#[test]
fn constant_logical_equal_folds() {
    // true == true should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let t1 = b.const_true();
    let t2 = b.const_true();
    let _ = b.builder.logical_equal(b.bool_ty, None, t1, t2).expect("eq");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::LogicalEqual),
        "constant logical equal should fold"
    );
}
