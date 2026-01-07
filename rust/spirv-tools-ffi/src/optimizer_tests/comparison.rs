//! Tests for comparison optimization rules (IEqual, INotEqual, SLessThan, etc.).
//! Based on C++ fold_test.cpp comparison patterns.

use super::common::{OptimizedModule, OptimizerEnvGuard, TestModuleBuilder};
use rspirv::spirv::Op;

// =============================================================================
// Reflexive Comparison Tests
// =============================================================================

#[test]
fn eq_self_folds_to_true() {
    // x == x should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let _ = b.builder.i_equal(b.bool_ty, None, x, x).expect("eq");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::IEqual),
        "x == x should fold to true constant"
    );
}

#[test]
fn ne_self_folds_to_false() {
    // x != x should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let _ = b.builder.i_not_equal(b.bool_ty, None, x, x).expect("ne");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::INotEqual),
        "x != x should fold to false constant"
    );
}

#[test]
fn slt_self_folds_to_false() {
    // x < x should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let _ = b.builder.s_less_than(b.bool_ty, None, x, x).expect("slt");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::SLessThan),
        "x < x should fold to false constant"
    );
}

#[test]
fn sle_self_folds_to_true() {
    // x <= x should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let _ = b.builder.s_less_than_equal(b.bool_ty, None, x, x).expect("sle");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::SLessThanEqual),
        "x <= x should fold to true constant"
    );
}

#[test]
fn sgt_self_folds_to_false() {
    // x > x should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let _ = b.builder.s_greater_than(b.bool_ty, None, x, x).expect("sgt");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::SGreaterThan),
        "x > x should fold to false constant"
    );
}

#[test]
fn sge_self_folds_to_true() {
    // x >= x should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let _ = b.builder.s_greater_than_equal(b.bool_ty, None, x, x).expect("sge");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::SGreaterThanEqual),
        "x >= x should fold to true constant"
    );
}

// =============================================================================
// Unsigned Reflexive Comparison Tests
// =============================================================================

#[test]
fn ult_self_folds_to_false() {
    // x < x (unsigned) should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let _ = b.builder.u_less_than(b.bool_ty, None, x, x).expect("ult");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::ULessThan),
        "x < x (unsigned) should fold to false constant"
    );
}

#[test]
fn ule_self_folds_to_true() {
    // x <= x (unsigned) should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let _ = b.builder.u_less_than_equal(b.bool_ty, None, x, x).expect("ule");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::ULessThanEqual),
        "x <= x (unsigned) should fold to true constant"
    );
}

#[test]
fn ugt_self_folds_to_false() {
    // x > x (unsigned) should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let _ = b.builder.u_greater_than(b.bool_ty, None, x, x).expect("ugt");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::UGreaterThan),
        "x > x (unsigned) should fold to false constant"
    );
}

#[test]
fn uge_self_folds_to_true() {
    // x >= x (unsigned) should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let _ = b.builder.u_greater_than_equal(b.bool_ty, None, x, x).expect("uge");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::UGreaterThanEqual),
        "x >= x (unsigned) should fold to true constant"
    );
}

// =============================================================================
// Comparison with Subtraction Tests
// =============================================================================

#[test]
fn eq_sub_zero_simplifies() {
    // (x - y) == 0 should simplify to x == y
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty]);
    let (x, y) = (params[0], params[1]);
    let c0 = b.const_i32(0);
    let sub = b.builder.i_sub(b.int_ty, None, x, y).expect("sub");
    let _ = b.builder.i_equal(b.bool_ty, None, sub, c0).expect("eq");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should simplify to x == y (no subtraction needed)
    assert!(
        !result.has_opcode(Op::ISub),
        "(x - y) == 0 should simplify to x == y"
    );
}

#[test]
fn ne_sub_zero_simplifies() {
    // (x - y) != 0 should simplify to x != y
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty]);
    let (x, y) = (params[0], params[1]);
    let c0 = b.const_i32(0);
    let sub = b.builder.i_sub(b.int_ty, None, x, y).expect("sub");
    let _ = b.builder.i_not_equal(b.bool_ty, None, sub, c0).expect("ne");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should simplify to x != y (no subtraction needed)
    assert!(
        !result.has_opcode(Op::ISub),
        "(x - y) != 0 should simplify to x != y"
    );
}

// =============================================================================
// Unsigned Comparison with Zero/One Tests
// =============================================================================

#[test]
fn ult_one_to_eq_zero() {
    // x < 1 (unsigned) should become x == 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let c1 = b.const_u32(1);
    let _ = b.builder.u_less_than(b.bool_ty, None, x, c1).expect("ult");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should become x == 0
    assert!(
        !result.has_opcode(Op::ULessThan),
        "x < 1 (unsigned) should become x == 0"
    );
}

#[test]
fn ule_zero_to_eq_zero() {
    // x <= 0 (unsigned) should become x == 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let c0 = b.const_u32(0);
    let _ = b.builder.u_less_than_equal(b.bool_ty, None, x, c0).expect("ule");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should become x == 0
    assert!(
        !result.has_opcode(Op::ULessThanEqual),
        "x <= 0 (unsigned) should become x == 0"
    );
}

#[test]
fn ugt_zero_to_ne_zero() {
    // x > 0 (unsigned) should become x != 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let c0 = b.const_u32(0);
    let _ = b.builder.u_greater_than(b.bool_ty, None, x, c0).expect("ugt");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should become x != 0
    assert!(
        !result.has_opcode(Op::UGreaterThan),
        "x > 0 (unsigned) should become x != 0"
    );
}

#[test]
fn uge_one_to_ne_zero() {
    // x >= 1 (unsigned) should become x != 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let c1 = b.const_u32(1);
    let _ = b.builder.u_greater_than_equal(b.bool_ty, None, x, c1).expect("uge");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should become x != 0
    assert!(
        !result.has_opcode(Op::UGreaterThanEqual),
        "x >= 1 (unsigned) should become x != 0"
    );
}

// =============================================================================
// Comparison with Common Addend Tests
// =============================================================================

#[test]
fn eq_with_common_addend_simplifies() {
    // (a + c) == (b + c) should simplify to a == b
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty, b.int_ty]);
    let (a, bval, c) = (params[0], params[1], params[2]);
    let add_ac = b.builder.i_add(b.int_ty, None, a, c).expect("add1");
    let add_bc = b.builder.i_add(b.int_ty, None, bval, c).expect("add2");
    let _ = b.builder.i_equal(b.bool_ty, None, add_ac, add_bc).expect("eq");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should simplify away the additions
    assert!(
        !result.has_opcode(Op::IAdd),
        "(a + c) == (b + c) should simplify to a == b"
    );
}

#[test]
fn ne_with_common_addend_simplifies() {
    // (a + c) != (b + c) should simplify to a != b
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty, b.int_ty]);
    let (a, bval, c) = (params[0], params[1], params[2]);
    let add_ac = b.builder.i_add(b.int_ty, None, a, c).expect("add1");
    let add_bc = b.builder.i_add(b.int_ty, None, bval, c).expect("add2");
    let _ = b.builder.i_not_equal(b.bool_ty, None, add_ac, add_bc).expect("ne");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should simplify away the additions
    assert!(
        !result.has_opcode(Op::IAdd),
        "(a + c) != (b + c) should simplify to a != b"
    );
}

// =============================================================================
// Floating-Point Reflexive Comparison Tests
// =============================================================================

#[test]
fn ford_eq_self_folds() {
    // x == x (ordered) should fold to true (assuming not NaN)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let _ = b.builder.f_ord_equal(b.bool_ty, None, x, x).expect("feq");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FOrdEqual),
        "x == x (ordered) should fold"
    );
}

#[test]
fn ford_ne_self_folds() {
    // x != x (ordered) should fold to false (assuming not NaN)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let _ = b.builder.f_ord_not_equal(b.bool_ty, None, x, x).expect("fne");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FOrdNotEqual),
        "x != x (ordered) should fold"
    );
}

#[test]
fn ford_lt_self_folds() {
    // x < x (ordered) should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let _ = b.builder.f_ord_less_than(b.bool_ty, None, x, x).expect("flt");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FOrdLessThan),
        "x < x (ordered) should fold"
    );
}

#[test]
fn ford_le_self_folds() {
    // x <= x (ordered) should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let _ = b.builder.f_ord_less_than_equal(b.bool_ty, None, x, x).expect("fle");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FOrdLessThanEqual),
        "x <= x (ordered) should fold"
    );
}

#[test]
fn ford_gt_self_folds() {
    // x > x (ordered) should fold to false
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let _ = b.builder.f_ord_greater_than(b.bool_ty, None, x, x).expect("fgt");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FOrdGreaterThan),
        "x > x (ordered) should fold"
    );
}

#[test]
fn ford_ge_self_folds() {
    // x >= x (ordered) should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let _ = b.builder.f_ord_greater_than_equal(b.bool_ty, None, x, x).expect("fge");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FOrdGreaterThanEqual),
        "x >= x (ordered) should fold"
    );
}

// =============================================================================
// Constant Folding Comparison Tests
// =============================================================================

#[test]
fn constant_eq_folds() {
    // 5 == 5 should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c5a = b.const_i32(5);
    let c5b = b.const_i32(5);
    let _ = b.builder.i_equal(b.bool_ty, None, c5a, c5b).expect("eq");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::IEqual),
        "constant comparison should fold"
    );
}

#[test]
fn constant_ne_folds() {
    // 5 != 3 should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c5 = b.const_i32(5);
    let c3 = b.const_i32(3);
    let _ = b.builder.i_not_equal(b.bool_ty, None, c5, c3).expect("ne");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::INotEqual),
        "constant comparison should fold"
    );
}

#[test]
fn constant_slt_folds() {
    // 3 < 5 should fold to true
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c3 = b.const_i32(3);
    let c5 = b.const_i32(5);
    let _ = b.builder.s_less_than(b.bool_ty, None, c3, c5).expect("slt");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::SLessThan),
        "constant comparison should fold"
    );
}
