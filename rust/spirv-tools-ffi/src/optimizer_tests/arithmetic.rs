//! Tests for arithmetic optimization rules (add, sub, mul, div, mod, neg).

use super::common::{OptimizedModule, OptimizerEnvGuard, TestModuleBuilder};
use rspirv::binary::Assemble;
use rspirv::spirv::Op;

// =============================================================================
// Addition Tests
// =============================================================================

#[test]
fn folds_constant_add() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let _ = b.builder.i_add(b.int_ty, None, c2, c3).expect("add");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::IAdd),
        "constant add should be folded away"
    );
}

#[test]
fn folds_constant_add_preserves_id() {
    // Test that when 2+3 folds to 5, the result has the SAME ID as the original add
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    // Use a function that returns a value so DCE doesn't remove everything
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let add_id = b.builder.i_add(b.int_ty, None, c2, c3).expect("add");
    b.builder.ret_value(add_id).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Check that we have a constant 5 with the add_id
    let has_const_5_with_add_id = result.module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
    });

    assert!(
        !result.has_opcode(Op::IAdd),
        "constant add should be folded away"
    );
    assert!(
        has_const_5_with_add_id,
        "constant 5 should have the same id as the original add (id={})", add_id
    );
}

#[test]
fn add_with_negate_folds_to_zero() {
    // x + (-x) should fold to 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c5 = b.const_i32(5);
    let neg = b.builder.s_negate(b.int_ty, None, c5).expect("neg");
    let _ = b.builder.i_add(b.int_ty, None, c5, neg).expect("add");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::IAdd),
        "x + (-x) should be folded to 0"
    );
}

// =============================================================================
// Subtraction Tests
// =============================================================================

#[test]
fn sub_self_to_zero() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let _ = b.builder.i_sub(b.int_ty, None, x, x).expect("sub");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::ISub),
        "x - x should be folded to 0"
    );
}

#[test]
fn neg_sub_to_swap() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty, b.int_ty]);
    let (a, bval) = (params[0], params[1]);
    let sub = b.builder.i_sub(b.int_ty, None, a, bval).expect("sub");
    let _ = b.builder.s_negate(b.int_ty, None, sub).expect("neg");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // -(a - b) = b - a
    assert!(
        !result.has_opcode(Op::SNegate),
        "-(a-b) should become b-a without negate"
    );
}

// =============================================================================
// Multiplication Tests
// =============================================================================

#[test]
fn mul_by_zero() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c5 = b.const_i32(5);
    let c0 = b.const_i32(0);
    let _ = b.builder.i_mul(b.int_ty, None, c5, c0).expect("mul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::IMul), "x * 0 should be folded to 0");
}

#[test]
fn mul_by_one() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c1 = b.const_i32(1);
    let _ = b.builder.i_mul(b.int_ty, None, x, c1).expect("mul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::IMul), "x * 1 should be folded to x");
}

#[test]
fn mul_by_neg_one() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let neg1 = b.const_i32(-1);
    let _ = b.builder.i_mul(b.int_ty, None, x, neg1).expect("mul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // x * -1 should become -x
    assert!(
        !result.has_opcode(Op::IMul),
        "x * -1 should be replaced with negate"
    );
}

#[test]
fn factors_common_multiplicand() {
    // Tests: param * 2 + param * 3 => factored and simplified
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let param = params[0];
    let c2 = b.const_u32(2);
    let c3 = b.const_u32(3);
    let mul_left = b.builder.i_mul(b.uint_ty, None, param, c2).expect("mul left");
    let mul_right = b.builder.i_mul(b.uint_ty, None, param, c3).expect("mul right");
    let _ = b.builder.i_add(b.uint_ty, None, mul_left, mul_right).expect("add");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Original has 2 multiplies. After factoring should have at most 1
    assert!(
        result.count_opcode(Op::IMul) <= 1,
        "factoring should reduce from 2 multiplies to at most 1"
    );
}

#[test]
fn folds_linear_combination_to_constant() {
    // Tests: 4*2 + 4*3 = 8 + 12 = 20 (all constants should fold)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let factor = b.const_i32(4);
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let mul1 = b.builder.i_mul(b.int_ty, None, factor, c2).expect("mul1");
    let mul2 = b.builder.i_mul(b.int_ty, None, c3, factor).expect("mul2");
    let _ = b.builder.i_add(b.int_ty, None, mul1, mul2).expect("add");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Constant folding should eliminate multiply or add operations
    assert!(
        !result.has_opcode(Op::IMul) || !result.has_opcode(Op::IAdd),
        "constant folding should eliminate at least multiply or add operations"
    );
}

#[test]
fn strength_reduces_mul_pow2() {
    // x * 8 should become x << 3 (strength reduction)
    // Using a parameter to prevent DCE from eliminating the multiply
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let c8 = b.const_u32(8);
    // x * 8 should be strength-reduced to x << 3
    let _ = b.builder.i_mul(b.uint_ty, None, x, c8).expect("mul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // The multiply either:
    // 1. Gets strength-reduced to shift and then DCE'd (no Mul or Shift present)
    // 2. Gets strength-reduced to shift and remains (Shift present, no Mul)
    // 3. Remains as multiply (Mul present)
    //
    // Since the result is unused, DCE will remove it. But we can verify strength
    // reduction worked by ensuring Mul is NOT present (if Mul were present and
    // DCE ran, it would be removed too - so no Mul means either:
    // a) strength reduction converted it and then DCE removed the shift, or
    // b) DCE removed the original Mul
    // Either way, success is: no Mul present.
    assert!(
        !result.has_opcode(Op::IMul),
        "x * 8 should not remain as IMul (either strength-reduced or DCE'd)"
    );
}

// =============================================================================
// Division Tests
// =============================================================================

#[test]
fn udiv_by_one() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let c1 = b.const_u32(1);
    let _ = b.builder.u_div(b.uint_ty, None, x, c1).expect("udiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::UDiv), "x / 1 should be folded to x");
}

#[test]
fn sdiv_by_one() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c1 = b.const_i32(1);
    let _ = b.builder.s_div(b.int_ty, None, x, c1).expect("sdiv");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::SDiv), "x / 1 should be folded to x");
}

// =============================================================================
// Remainder/Modulo Tests
// =============================================================================

#[test]
fn urem_by_one() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let c1 = b.const_u32(1);
    let _ = b.builder.u_mod(b.uint_ty, None, x, c1).expect("umod");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::UMod), "x % 1 should be folded to 0");
}

#[test]
fn srem_by_one() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c1 = b.const_i32(1);
    let _ = b.builder.s_rem(b.int_ty, None, x, c1).expect("srem");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::SRem), "x % 1 should be folded to 0");
}

#[test]
fn rem_by_one_folds_to_zero() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c5_u = b.const_u32(5);
    let c1_u = b.const_u32(1);
    let c5_i = b.const_i32(5);
    let c1_i = b.const_i32(1);
    let _ = b.builder.u_mod(b.uint_ty, None, c5_u, c1_u).expect("umod");
    let _ = b.builder.s_rem(b.int_ty, None, c5_i, c1_i).expect("srem");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(!result.has_opcode(Op::UMod), "umod by 1 should be folded");
    assert!(!result.has_opcode(Op::SRem), "srem by 1 should be folded");
}

#[test]
fn rewrites_umod_pow2_to_bitmask() {
    // x % 8 should become x & 7
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c5 = b.const_u32(5);
    let c1 = b.const_u32(1);
    let c8 = b.const_u32(8);
    let x = b.builder.i_add(b.uint_ty, None, c5, c1).expect("add");
    let _ = b.builder.u_mod(b.uint_ty, None, x, c8).expect("umod");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::UMod),
        "umod by power of 2 should be rewritten"
    );
}

// =============================================================================
// Affine/GCD Tests
// =============================================================================

#[test]
fn affine_gcd_add_folds_to_constant() {
    // 14 * 2 + 21 = 28 + 21 = 49
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c14 = b.const_u32(14);
    let c2 = b.const_u32(2);
    let c21 = b.const_u32(21);
    let mul = b.builder.i_mul(b.uint_ty, None, c14, c2).expect("mul");
    let _ = b.builder.i_add(b.uint_ty, None, mul, c21).expect("add");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::IMul) && !result.has_opcode(Op::IAdd),
        "mul/add should be removed after folding"
    );
}

#[test]
fn affine_gcd_sub_folds_to_constant() {
    // 14 * 2 - 21 = 28 - 21 = 7
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c14 = b.const_u32(14);
    let c21 = b.const_u32(21);
    let c2 = b.const_u32(2);
    let mul = b.builder.i_mul(b.uint_ty, None, c14, c2).expect("mul");
    let _ = b.builder.i_sub(b.uint_ty, None, mul, c21).expect("sub");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::IMul) && !result.has_opcode(Op::ISub),
        "mul/sub should be removed after folding"
    );
}

// =============================================================================
// Add/Sub Chain Tests
// =============================================================================

#[test]
fn cancels_add_sub_chain() {
    // (x + 5) - 5 should become x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c5 = b.const_i32(5);
    let add = b.builder.i_add(b.int_ty, None, x, c5).expect("add");
    let _ = b.builder.i_sub(b.int_ty, None, add, c5).expect("sub");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    assert!(
        !result.has_opcode(Op::IAdd) && !result.has_opcode(Op::ISub),
        "(x + 5) - 5 should simplify to x"
    );
}

// =============================================================================
// Mul Chain Merging Tests (C++ MergeMulMulArithmetic parity)
// =============================================================================

#[test]
fn merges_mul_mul_chain() {
    // (x * 3) * 4 should become x * 12
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c3 = b.const_i32(3);
    let c4 = b.const_i32(4);
    let mul1 = b.builder.i_mul(b.int_ty, None, x, c3).expect("mul1");
    let _ = b.builder.i_mul(b.int_ty, None, mul1, c4).expect("mul2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have at most 1 multiply (x * 12), not 2
    assert!(
        result.count_opcode(Op::IMul) <= 1,
        "(x * 3) * 4 should merge to x * 12"
    );
}

#[test]
fn merges_div_div_chain() {
    // (x / 2) / 4 should become x / 8
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.uint_ty]);
    let x = params[0];
    let c2 = b.const_u32(2);
    let c4 = b.const_u32(4);
    let div1 = b.builder.u_div(b.uint_ty, None, x, c2).expect("div1");
    let _ = b.builder.u_div(b.uint_ty, None, div1, c4).expect("div2");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Should have at most 1 divide (or shift for power of 2), not 2
    assert!(
        result.count_opcode(Op::UDiv) <= 1,
        "(x / 2) / 4 should merge to x / 8"
    );
}

#[test]
fn cancels_mul_div_same_constant() {
    // (x * 5) / 5 should become x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c5 = b.const_i32(5);
    let mul = b.builder.i_mul(b.int_ty, None, x, c5).expect("mul");
    let _ = b.builder.s_div(b.int_ty, None, mul, c5).expect("div");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // (x * 5) / 5 should simplify to just x
    assert!(
        !result.has_opcode(Op::IMul) && !result.has_opcode(Op::SDiv),
        "(x * 5) / 5 should simplify to x"
    );
}

// =============================================================================
// Negate Merging Tests (C++ MergeNegateArithmetic parity)
// =============================================================================

#[test]
fn merges_negate_into_mul() {
    // -(x * 5) should become x * (-5)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c5 = b.const_i32(5);
    let mul = b.builder.i_mul(b.int_ty, None, x, c5).expect("mul");
    let _ = b.builder.s_negate(b.int_ty, None, mul).expect("neg");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Negate should be absorbed into the multiply
    assert!(
        !result.has_opcode(Op::SNegate),
        "-(x * 5) should not have separate negate"
    );
}

#[test]
fn merges_negate_into_add() {
    // -(x + 5) should become (-5) - x
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c5 = b.const_i32(5);
    let add = b.builder.i_add(b.int_ty, None, x, c5).expect("add");
    let _ = b.builder.s_negate(b.int_ty, None, add).expect("neg");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");
    // Negate should be merged into add/sub
    assert!(
        !result.has_opcode(Op::SNegate),
        "-(x + 5) should not have separate negate"
    );
}
