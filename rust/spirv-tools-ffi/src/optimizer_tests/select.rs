//! Tests for select (gamma/conditional) optimization rules.

use super::common::{OptimizedModule, OptimizerEnvGuard, TestModuleBuilder};
use rspirv::spirv::Op;

// =============================================================================
// Select Constant Condition Tests
// =============================================================================

#[test]
fn select_true_condition() {
    // select(true, a, b) should become a
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty]);
    let (a, b_param) = (params[0], params[1]);
    let true_const = b.const_true();
    let _ = b.builder.select(b.int_ty, None, true_const, a, b_param).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::Select),
        "select(true, a, b) should simplify to a"
    );
}

#[test]
fn select_false_condition() {
    // select(false, a, b) should become b
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty]);
    let (a, b_param) = (params[0], params[1]);
    let false_const = b.const_false();
    let _ = b.builder.select(b.int_ty, None, false_const, a, b_param).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::Select),
        "select(false, a, b) should simplify to b"
    );
}

#[test]
fn select_same_both_arms() {
    // select(c, x, x) should become x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty]);
    let (cond, x) = (params[0], params[1]);
    let _ = b.builder.select(b.int_ty, None, cond, x, x).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::Select),
        "select(c, x, x) should simplify to x"
    );
}

// =============================================================================
// Select Absorption Tests
// =============================================================================

#[test]
fn select_with_add_same_base() {
    // select(c, x + y, x) should optimize via absorption rule
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty, b.int_ty]);
    let (cond, x, y) = (params[0], params[1], params[2]);
    let x_plus_y = b.builder.i_add(b.int_ty, None, x, y).expect("add");
    let _ = b.builder.select(b.int_ty, None, cond, x_plus_y, x).expect("select");
    let words = b.finish();

    // Just verify it runs without error
    let _ = OptimizedModule::from_words(&words).expect("optimizer runs");
}

#[test]
fn select_with_mul_same_base() {
    // select(c, x * y, x) should optimize
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty, b.int_ty]);
    let (cond, x, y) = (params[0], params[1], params[2]);
    let x_mul_y = b.builder.i_mul(b.int_ty, None, x, y).expect("mul");
    let _ = b.builder.select(b.int_ty, None, cond, x_mul_y, x).expect("select");
    let words = b.finish();

    let _ = OptimizedModule::from_words(&words).expect("optimizer runs");
}

// =============================================================================
// Arithmetic with Conditional Zero Tests
// =============================================================================

#[test]
fn sub_with_conditional_zero() {
    // x - select(c, y, 0) should become select(c, x - y, x)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty, b.int_ty]);
    let (cond, x, y) = (params[0], params[1], params[2]);
    let zero = b.const_i32(0);
    let select_y_or_zero = b.builder.select(b.int_ty, None, cond, y, zero).expect("select");
    let _ = b.builder.i_sub(b.int_ty, None, x, select_y_or_zero).expect("sub");
    let words = b.finish();

    let _ = OptimizedModule::from_words(&words).expect("optimizer runs");
}

#[test]
fn add_with_conditional_zero() {
    // x + select(c, y, 0) should become select(c, x + y, x)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty, b.int_ty]);
    let (cond, x, y) = (params[0], params[1], params[2]);
    let zero = b.const_i32(0);
    let select_y_or_zero = b.builder.select(b.int_ty, None, cond, y, zero).expect("select");
    let _ = b.builder.i_add(b.int_ty, None, x, select_y_or_zero).expect("add");
    let words = b.finish();

    let _ = OptimizedModule::from_words(&words).expect("optimizer runs");
}

// =============================================================================
// Logical Select Tests
// =============================================================================

#[test]
fn select_with_logical_and_condition() {
    // select(c, c && x, false) should become c && x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.bool_ty]);
    let (c, x) = (params[0], params[1]);
    let c_and_x = b.builder.logical_and(b.bool_ty, None, c, x).expect("and");
    let false_const = b.const_false();
    let _ = b.builder.select(b.bool_ty, None, c, c_and_x, false_const).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::Select),
        "select(c, c && x, false) should simplify to c && x"
    );
}

// =============================================================================
// Nested Select Tests
// =============================================================================

#[test]
fn nested_select_same_condition() {
    // select(c, select(c, a, b), d) should become select(c, a, d)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty, b.int_ty, b.int_ty]);
    let (cond, a, b_param, d) = (params[0], params[1], params[2], params[3]);
    let inner = b.builder.select(b.int_ty, None, cond, a, b_param).expect("inner select");
    let _ = b.builder.select(b.int_ty, None, cond, inner, d).expect("outer select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        result.count_opcode(Op::Select) <= 1,
        "nested select(c, select(c, a, b), d) should simplify to one select"
    );
}

// =============================================================================
// Select Distribution Tests
// =============================================================================

#[test]
fn select_distribution_over_add() {
    // select(c, a, b) + select(c, x, y) => select(c, a+x, b+y)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.bool_ty, b.int_ty, b.int_ty, b.int_ty, b.int_ty]);
    let (cond, a, b_param, x, y) = (params[0], params[1], params[2], params[3], params[4]);
    let sel1 = b.builder.select(b.int_ty, None, cond, a, b_param).expect("select1");
    let sel2 = b.builder.select(b.int_ty, None, cond, x, y).expect("select2");
    let _ = b.builder.i_add(b.int_ty, None, sel1, sel2).expect("add");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        result.count_opcode(Op::Select) <= 1,
        "select(c,a,b) + select(c,x,y) should combine to single select"
    );
}

// =============================================================================
// Pattern Recognition Tests
// =============================================================================

#[test]
#[ignore = "absolute value pattern requires GLSL extended instructions not yet supported in lowering"]
fn abs_pattern_with_select() {
    // select(x >= 0, x, -x) should become abs(x)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let zero = b.const_i32(0);
    let cond = b.builder.s_greater_than_equal(b.bool_ty, None, x, zero).expect("cmp");
    let neg_x = b.builder.s_negate(b.int_ty, None, x).expect("neg");
    let _ = b.builder.select(b.int_ty, None, cond, x, neg_x).expect("select");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    let has_select = result.has_opcode(Op::Select);
    let has_sge = result.has_opcode(Op::SGreaterThanEqual);
    let has_snegate = result.has_opcode(Op::SNegate);
    assert!(
        !has_select && !has_sge && !has_snegate,
        "select(x>=0, x, -x) pattern should be replaced with abs"
    );
}
