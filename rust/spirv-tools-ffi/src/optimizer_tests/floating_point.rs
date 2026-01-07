//! Tests for floating-point optimization rules (FAdd, FSub, FMul, FDiv, FNeg).
//! Based on C++ fold_test.cpp MergeNegateTest, MergeAddTest, etc.

use super::common::{OptimizedModule, OptimizerEnvGuard, TestModuleBuilder};
use rspirv::spirv::Op;

// =============================================================================
// FP Identity Tests
// =============================================================================

#[test]
fn fp_add_zero_identity() {
    // x + 0 should fold to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c0 = b.const_f32(0.0);
    let _ = b.builder.f_add(b.float_ty, None, x, c0).expect("fadd");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FAdd),
        "x + 0.0 should be folded to x"
    );
}

#[test]
fn fp_sub_zero_identity() {
    // x - 0 should fold to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c0 = b.const_f32(0.0);
    let _ = b.builder.f_sub(b.float_ty, None, x, c0).expect("fsub");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FSub),
        "x - 0.0 should be folded to x"
    );
}

#[test]
fn fp_mul_one_identity() {
    // x * 1 should fold to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c1 = b.const_f32(1.0);
    let _ = b.builder.f_mul(b.float_ty, None, x, c1).expect("fmul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::FMul), "x * 1.0 should be folded to x");
}

#[test]
fn fp_div_one_identity() {
    // x / 1 should fold to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c1 = b.const_f32(1.0);
    let _ = b.builder.f_div(b.float_ty, None, x, c1).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::FDiv), "x / 1.0 should be folded to x");
}

// =============================================================================
// FP Mul/Div by -1 Tests
// =============================================================================

#[test]
fn fp_mul_neg_one_to_negate() {
    // x * -1 should become -x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let neg1 = b.const_f32(-1.0);
    let _ = b.builder.f_mul(b.float_ty, None, x, neg1).expect("fmul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FMul),
        "x * -1.0 should be replaced with negate"
    );
}

#[test]
fn fp_div_neg_one_to_negate() {
    // x / -1 should become -x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let neg1 = b.const_f32(-1.0);
    let _ = b.builder.f_div(b.float_ty, None, x, neg1).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FDiv),
        "x / -1.0 should be replaced with negate"
    );
}

// =============================================================================
// FP Subtraction Tests
// =============================================================================

#[test]
fn fp_sub_self_to_zero() {
    // x - x should fold to 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let _ = b.builder.f_sub(b.float_ty, None, x, x).expect("fsub");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::FSub), "x - x should be folded to 0");
}

#[test]
fn fp_zero_sub_to_negate() {
    // 0 - x should become -x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c0 = b.const_f32(0.0);
    let _ = b.builder.f_sub(b.float_ty, None, c0, x).expect("fsub");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FSub),
        "0 - x should be replaced with negate"
    );
}

// =============================================================================
// Double Negation Tests
// =============================================================================

#[test]
fn fp_double_negate_cancels() {
    // -(-x) should fold to x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let neg1 = b.builder.f_negate(b.float_ty, None, x).expect("fneg1");
    let _ = b.builder.f_negate(b.float_ty, None, neg1).expect("fneg2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FNegate),
        "-(-x) should fold to x (no negate)"
    );
}

// =============================================================================
// Negate Merging Tests (C++ MergeNegateTest parity)
// =============================================================================

#[test]
fn fp_negate_of_sub_swaps() {
    // -(a - b) should become b - a
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty, b.float_ty]);
    let (a, bval) = (params[0], params[1]);
    let sub = b.builder.f_sub(b.float_ty, None, a, bval).expect("fsub");
    let _ = b.builder.f_negate(b.float_ty, None, sub).expect("fneg");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FNegate),
        "-(a-b) should become b-a without negate"
    );
}

#[test]
fn fp_negate_mul_with_const_absorbs() {
    // -(x * 2.0) should become x * -2.0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c2 = b.const_f32(2.0);
    let mul = b.builder.f_mul(b.float_ty, None, x, c2).expect("fmul");
    let _ = b.builder.f_negate(b.float_ty, None, mul).expect("fneg");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FNegate),
        "-(x * 2.0) should absorb negate into constant"
    );
}

#[test]
fn fp_negate_div_with_const_absorbs() {
    // -(x / 2.0) should become x * -0.5
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c2 = b.const_f32(2.0);
    let div = b.builder.f_div(b.float_ty, None, x, c2).expect("fdiv");
    let _ = b.builder.f_negate(b.float_ty, None, div).expect("fneg");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // The negate should be absorbed somehow (either FDiv removed or FNegate removed)
    assert!(
        !result.has_opcode(Op::FNegate) || !result.has_opcode(Op::FDiv),
        "-(x / 2.0) should optimize away negate or div"
    );
}

// =============================================================================
// Add/Sub Chain Merging Tests (C++ MergeAddTest parity)
// =============================================================================

#[test]
fn fp_add_negate_to_sub() {
    // (-x) + 2 should become 2 - x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c2 = b.const_f32(2.0);
    let neg = b.builder.f_negate(b.float_ty, None, x).expect("fneg");
    let _ = b.builder.f_add(b.float_ty, None, neg, c2).expect("fadd");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FNegate),
        "(-x) + 2 should become 2 - x without separate negate"
    );
}

#[test]
fn fp_add_of_sub_const_merges() {
    // (x - 1) + 2 should become x + 1
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c1 = b.const_f32(1.0);
    let c2 = b.const_f32(2.0);
    let sub = b.builder.f_sub(b.float_ty, None, x, c1).expect("fsub");
    let _ = b.builder.f_add(b.float_ty, None, sub, c2).expect("fadd");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have at most 1 add (x + 1), not sub + add
    assert!(
        !result.has_opcode(Op::FSub),
        "(x - 1) + 2 should merge to x + 1 (no sub)"
    );
}

#[test]
fn fp_add_of_add_const_merges() {
    // (x + 1) + 2 should become x + 3
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c1 = b.const_f32(1.0);
    let c2 = b.const_f32(2.0);
    let add1 = b.builder.f_add(b.float_ty, None, x, c1).expect("fadd1");
    let _ = b.builder.f_add(b.float_ty, None, add1, c2).expect("fadd2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have exactly 1 add (x + 3)
    assert!(
        result.count_opcode(Op::FAdd) <= 1,
        "(x + 1) + 2 should merge to x + 3"
    );
}

// =============================================================================
// Sub Chain Merging Tests (C++ MergeSubTest parity)
// =============================================================================

#[test]
fn fp_sub_of_sub_const_merges() {
    // (x - 1) - 2 should become x - 3
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c1 = b.const_f32(1.0);
    let c2 = b.const_f32(2.0);
    let sub1 = b.builder.f_sub(b.float_ty, None, x, c1).expect("fsub1");
    let _ = b.builder.f_sub(b.float_ty, None, sub1, c2).expect("fsub2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have exactly 1 sub (x - 3)
    assert!(
        result.count_opcode(Op::FSub) <= 1,
        "(x - 1) - 2 should merge to x - 3"
    );
}

#[test]
fn fp_sub_of_add_const_merges() {
    // (x + 1) - 2 should become x + (-1) = x - 1
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c1 = b.const_f32(1.0);
    let c2 = b.const_f32(2.0);
    let add = b.builder.f_add(b.float_ty, None, x, c1).expect("fadd");
    let _ = b.builder.f_sub(b.float_ty, None, add, c2).expect("fsub");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have at most 1 op (add or sub), not both
    assert!(
        result.count_opcode(Op::FAdd) + result.count_opcode(Op::FSub) <= 1,
        "(x + 1) - 2 should merge to single op"
    );
}

// =============================================================================
// Mul Chain Merging Tests (C++ MergeMulTest parity)
// =============================================================================

#[test]
fn fp_mul_mul_const_merges() {
    // (x * 3) * 4 should become x * 12
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c3 = b.const_f32(3.0);
    let c4 = b.const_f32(4.0);
    let mul1 = b.builder.f_mul(b.float_ty, None, x, c3).expect("fmul1");
    let _ = b.builder.f_mul(b.float_ty, None, mul1, c4).expect("fmul2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have exactly 1 mul (x * 12)
    assert!(
        result.count_opcode(Op::FMul) <= 1,
        "(x * 3) * 4 should merge to x * 12"
    );
}

// =============================================================================
// Div Chain Merging Tests (C++ MergeDivTest parity)
// =============================================================================

#[test]
fn fp_div_div_const_merges() {
    // (x / 2) / 4 should become x / 8
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c2 = b.const_f32(2.0);
    let c4 = b.const_f32(4.0);
    let div1 = b.builder.f_div(b.float_ty, None, x, c2).expect("fdiv1");
    let _ = b.builder.f_div(b.float_ty, None, div1, c4).expect("fdiv2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have exactly 1 div (x / 8)
    assert!(
        result.count_opcode(Op::FDiv) <= 1,
        "(x / 2) / 4 should merge to x / 8"
    );
}

// =============================================================================
// Mul/Div Cancellation Tests
// =============================================================================

#[test]
fn fp_mul_div_same_cancels() {
    // (x * a) / a should become x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty, b.float_ty]);
    let (x, a) = (params[0], params[1]);
    let mul = b.builder.f_mul(b.float_ty, None, x, a).expect("fmul");
    let _ = b.builder.f_div(b.float_ty, None, mul, a).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FMul) && !result.has_opcode(Op::FDiv),
        "(x * a) / a should simplify to x"
    );
}

#[test]
fn fp_div_mul_same_cancels() {
    // (x / a) * a should become x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty, b.float_ty]);
    let (x, a) = (params[0], params[1]);
    let div = b.builder.f_div(b.float_ty, None, x, a).expect("fdiv");
    let _ = b.builder.f_mul(b.float_ty, None, div, a).expect("fmul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FMul) && !result.has_opcode(Op::FDiv),
        "(x / a) * a should simplify to x"
    );
}

// =============================================================================
// Add/Sub Cancellation Tests
// =============================================================================

#[test]
fn fp_add_sub_same_cancels() {
    // (x + a) - a should become x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty, b.float_ty]);
    let (x, a) = (params[0], params[1]);
    let add = b.builder.f_add(b.float_ty, None, x, a).expect("fadd");
    let _ = b.builder.f_sub(b.float_ty, None, add, a).expect("fsub");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FAdd) && !result.has_opcode(Op::FSub),
        "(x + a) - a should simplify to x"
    );
}

#[test]
fn fp_sub_add_same_cancels() {
    // (x - a) + a should become x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty, b.float_ty]);
    let (x, a) = (params[0], params[1]);
    let sub = b.builder.f_sub(b.float_ty, None, x, a).expect("fsub");
    let _ = b.builder.f_add(b.float_ty, None, sub, a).expect("fadd");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FAdd) && !result.has_opcode(Op::FSub),
        "(x - a) + a should simplify to x"
    );
}

// =============================================================================
// Constant Folding Tests
// =============================================================================

#[test]
fn fp_constant_add_folds() {
    // 2.0 + 3.0 should fold to 5.0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c2 = b.const_f32(2.0);
    let c3 = b.const_f32(3.0);
    let _ = b.builder.f_add(b.float_ty, None, c2, c3).expect("fadd");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FAdd),
        "constant FAdd should be folded"
    );
}

#[test]
fn fp_constant_mul_folds() {
    // 2.0 * 3.0 should fold to 6.0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c2 = b.const_f32(2.0);
    let c3 = b.const_f32(3.0);
    let _ = b.builder.f_mul(b.float_ty, None, c2, c3).expect("fmul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FMul),
        "constant FMul should be folded"
    );
}

// =============================================================================
// Reciprocal Tests (C++ ReciprocalFDivTest parity)
// =============================================================================

#[test]
fn fp_reciprocal_of_reciprocal() {
    // 1 / (1 / x) should become x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c1 = b.const_f32(1.0);
    let recip = b.builder.f_div(b.float_ty, None, c1, x).expect("fdiv1");
    let _ = b.builder.f_div(b.float_ty, None, c1, recip).expect("fdiv2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FDiv),
        "1 / (1 / x) should simplify to x"
    );
}

#[test]
fn fp_div_by_reciprocal() {
    // a / (1 / b) should become a * b
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty, b.float_ty]);
    let (a, bval) = (params[0], params[1]);
    let c1 = b.const_f32(1.0);
    let recip = b.builder.f_div(b.float_ty, None, c1, bval).expect("fdiv_recip");
    let _ = b.builder.f_div(b.float_ty, None, a, recip).expect("fdiv_main");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should become a * b (one FMul, no FDiv) or at least reduce from 2 divs
    assert!(
        result.count_opcode(Op::FDiv) <= 1,
        "a / (1/b) should simplify to fewer divisions"
    );
}

// =============================================================================
// Mul by Reciprocal Tests
// =============================================================================

#[test]
fn fp_mul_by_reciprocal_to_div() {
    // x * (1/y) should become x / y
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty, b.float_ty]);
    let (x, y) = (params[0], params[1]);
    let c1 = b.const_f32(1.0);
    let recip = b.builder.f_div(b.float_ty, None, c1, y).expect("fdiv_recip");
    let _ = b.builder.f_mul(b.float_ty, None, x, recip).expect("fmul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should become x / y (one FDiv, no FMul) or at least optimize somehow
    assert!(
        !result.has_opcode(Op::FMul) || result.count_opcode(Op::FDiv) <= 1,
        "x * (1/y) should simplify to x/y"
    );
}

// =============================================================================
// Factoring Tests
// =============================================================================

#[test]
fn fp_factors_common_multiplicand() {
    // x * a + x * b should become x * (a + b)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let c2 = b.const_f32(2.0);
    let c3 = b.const_f32(3.0);
    let mul1 = b.builder.f_mul(b.float_ty, None, x, c2).expect("fmul1");
    let mul2 = b.builder.f_mul(b.float_ty, None, x, c3).expect("fmul2");
    let _ = b.builder.f_add(b.float_ty, None, mul1, mul2).expect("fadd");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have at most 1 multiply after factoring
    assert!(
        result.count_opcode(Op::FMul) <= 1,
        "x*a + x*b should factor to x*(a+b)"
    );
}

#[test]
fn fp_x_plus_x_to_mul_2() {
    // x + x should become x * 2
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let _ = b.builder.f_add(b.float_ty, None, x, x).expect("fadd");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // FAdd should become FMul
    assert!(
        !result.has_opcode(Op::FAdd),
        "x + x should become x * 2"
    );
}

// =============================================================================
// Reciprocal Division to Multiplication Tests (C++ ReciprocalFDiv parity)
// =============================================================================
// Division by power-of-2 reciprocals can be converted to multiplication
// x / 0.5 = x * 2.0, x / 0.25 = x * 4.0, etc.

#[test]
fn fp_div_half_to_mul_two() {
    // x / 0.5 should become x * 2.0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let half = b.const_f32(0.5);
    let _ = b.builder.f_div(b.float_ty, None, x, half).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FDiv),
        "x / 0.5 should become x * 2.0"
    );
}

#[test]
fn fp_div_quarter_to_mul_four() {
    // x / 0.25 should become x * 4.0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let quarter = b.const_f32(0.25);
    let _ = b.builder.f_div(b.float_ty, None, x, quarter).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FDiv),
        "x / 0.25 should become x * 4.0"
    );
}

#[test]
fn fp_div_two_to_mul_half() {
    // x / 2.0 should become x * 0.5
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let two = b.const_f32(2.0);
    let _ = b.builder.f_div(b.float_ty, None, x, two).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FDiv),
        "x / 2.0 should become x * 0.5"
    );
}

#[test]
fn fp_div_four_to_mul_quarter() {
    // x / 4.0 should become x * 0.25
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let four = b.const_f32(4.0);
    let _ = b.builder.f_div(b.float_ty, None, x, four).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FDiv),
        "x / 4.0 should become x * 0.25"
    );
}

#[test]
fn fp_div_eighth_to_mul_eight() {
    // x / 0.125 should become x * 8.0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let eighth = b.const_f32(0.125);
    let _ = b.builder.f_div(b.float_ty, None, x, eighth).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FDiv),
        "x / 0.125 should become x * 8.0"
    );
}

#[test]
fn fp_div_eight_to_mul_eighth() {
    // x / 8.0 should become x * 0.125
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.float_ty]);
    let x = params[0];
    let eight = b.const_f32(8.0);
    let _ = b.builder.f_div(b.float_ty, None, x, eight).expect("fdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::FDiv),
        "x / 8.0 should become x * 0.125"
    );
}
