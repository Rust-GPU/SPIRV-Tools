//! Tests for Dead Code Elimination (DCE).
//!
//! These tests verify that the optimizer correctly:
//! - Removes dead instructions that don't feed side effects
//! - Removes dead constants only used by dead instructions
//! - Preserves live instructions that feed return values or stores
//! - Handles chains of dead instructions
//! - Properly tracks liveness through the e-graph

use super::common::{OptimizedModule, OptimizerEnvGuard, TestModuleBuilder};
use rspirv::binary::Assemble;
use rspirv::spirv::Op;

// =============================================================================
// Basic DCE Tests
// =============================================================================

#[test]
fn removes_dead_add_not_feeding_return() {
    // dead = 10 + 20 (not used)
    // live = 3 + 4
    // return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    // Dead computation
    let c10 = b.const_i32(10);
    let c20 = b.const_i32(20);
    let dead_add = b.builder.i_add(b.int_ty, None, c10, c20).expect("dead add");

    // Live computation
    let c3 = b.const_i32(3);
    let c4 = b.const_i32(4);
    let live_add = b.builder.i_add(b.int_ty, None, c3, c4).expect("live add");

    b.builder.ret_value(live_add).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Dead add should be removed
    let dead_exists = result
        .module
        .all_inst_iter()
        .any(|inst| inst.result_id == Some(dead_add));
    assert!(!dead_exists, "dead add instruction should be removed");

    // Live result should be folded to 7
    assert!(
        result.has_constant_u32(7),
        "live computation 3+4 should fold to 7"
    );
}

#[test]
fn removes_dead_instruction_chain() {
    // dead1 = 10 + 20
    // dead2 = dead1 * 2
    // dead3 = dead2 - 5
    // live = 3 + 4
    // return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    // Dead chain
    let c10 = b.const_i32(10);
    let c20 = b.const_i32(20);
    let dead1 = b.builder.i_add(b.int_ty, None, c10, c20).expect("dead1");
    let c2 = b.const_i32(2);
    let dead2 = b.builder.i_mul(b.int_ty, None, dead1, c2).expect("dead2");
    let c5 = b.const_i32(5);
    let dead3 = b.builder.i_sub(b.int_ty, None, dead2, c5).expect("dead3");

    // Live computation
    let c3 = b.const_i32(3);
    let c4 = b.const_i32(4);
    let live = b.builder.i_add(b.int_ty, None, c3, c4).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All dead chain instructions should be removed
    for dead_id in [dead1, dead2, dead3] {
        let dead_exists = result
            .module
            .all_inst_iter()
            .any(|inst| inst.result_id == Some(dead_id));
        assert!(!dead_exists, "dead instruction {} should be removed", dead_id);
    }

    // Dead constants (10, 20, 2, 5) should be removed
    assert!(
        !result.has_constant_u32(10),
        "dead constant 10 should be removed"
    );
    assert!(
        !result.has_constant_u32(20),
        "dead constant 20 should be removed"
    );

    // Live result should be folded to 7
    assert!(
        result.has_constant_u32(7),
        "live computation 3+4 should fold to 7"
    );
}

#[test]
fn removes_dead_but_preserves_shared_operand() {
    // shared = 5
    // dead = shared + 100  (dead - not used)
    // live = shared * 3    (live - returned)
    // return live
    // Use 100 for dead constant so it doesn't collide with result (5*3=15)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let shared = b.const_i32(5);
    let c100 = b.const_i32(100);
    let dead = b.builder.i_add(b.int_ty, None, shared, c100).expect("dead");
    let c3 = b.const_i32(3);
    let live = b.builder.i_mul(b.int_ty, None, shared, c3).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Dead instruction should be removed
    let dead_exists = result
        .module
        .all_inst_iter()
        .any(|inst| inst.result_id == Some(dead));
    assert!(!dead_exists, "dead instruction should be removed");

    // Dead constant 100 should be removed (only used by dead instruction)
    assert!(
        !result.has_constant_u32(100),
        "dead constant 100 should be removed"
    );

    // Live result should be folded to 15 (5 * 3)
    assert!(
        result.has_constant_u32(15),
        "live computation 5*3 should fold to 15"
    );
}

#[test]
fn removes_dead_constant_only_used_by_dead_code() {
    // c100 = 100 (only used by dead add)
    // c200 = 200 (only used by dead add)
    // dead = c100 + c200 (not used)
    // live = 1 + 1
    // return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c100 = b.const_i32(100);
    let c200 = b.const_i32(200);
    let _dead = b.builder.i_add(b.int_ty, None, c100, c200).expect("dead");

    let c1a = b.const_i32(1);
    let c1b = b.const_i32(1);
    let live = b.builder.i_add(b.int_ty, None, c1a, c1b).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Dead constants should be removed
    assert!(
        !result.has_constant_u32(100),
        "dead constant 100 should be removed"
    );
    assert!(
        !result.has_constant_u32(200),
        "dead constant 200 should be removed"
    );

    // Live result should be folded to 2
    assert!(result.has_constant_u32(2), "1+1 should fold to 2");
}

// =============================================================================
// Tests with Function Parameters (External Values)
// =============================================================================

#[test]
fn removes_dead_using_param_preserves_live_using_param() {
    // fn f(x: i32) -> i32 {
    //   dead = x + 10  (not used)
    //   live = x * 2   (returned)
    //   return live
    // }
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param = b.builder.function_parameter(b.int_ty).expect("param");
    b.builder.begin_block(None).expect("begin block");

    let c10 = b.const_i32(10);
    let dead = b.builder.i_add(b.int_ty, None, param, c10).expect("dead");
    let c2 = b.const_i32(2);
    let live = b.builder.i_mul(b.int_ty, None, param, c2).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Dead instruction should be removed
    let dead_exists = result
        .module
        .all_inst_iter()
        .any(|inst| inst.result_id == Some(dead));
    assert!(!dead_exists, "dead add should be removed");

    // Live multiplication should remain (can't fold with unknown param)
    assert!(
        result.has_opcode(Op::IMul),
        "live mul with param should remain"
    );

    // Dead constant 10 should be removed
    assert!(
        !result.has_constant_u32(10),
        "dead constant 10 should be removed"
    );

    // Live constant 2 should remain (used by live mul)
    assert!(result.has_constant_u32(2), "live constant 2 should remain");
}

// =============================================================================
// Tests for Preserving Live Code
// =============================================================================

#[test]
fn preserves_all_live_instructions() {
    // a = 1 + 2
    // b = a * 3
    // c = b - 1
    // return c (all are live)
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let a = b.builder.i_add(b.int_ty, None, c1, c2).expect("a");
    let c3 = b.const_i32(3);
    let bb = b.builder.i_mul(b.int_ty, None, a, c3).expect("b");
    let c1b = b.const_i32(1);
    let c = b.builder.i_sub(b.int_ty, None, bb, c1b).expect("c");

    b.builder.ret_value(c).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Everything should be constant folded to 8: ((1+2)*3)-1 = 8
    assert!(result.has_constant_u32(8), "result should fold to 8");

    // No arithmetic ops should remain (all folded)
    assert!(!result.has_opcode(Op::IAdd), "add should be folded");
    assert!(!result.has_opcode(Op::IMul), "mul should be folded");
    assert!(!result.has_opcode(Op::ISub), "sub should be folded");
}

// =============================================================================
// Void Function Tests (no return value)
// =============================================================================

#[test]
fn handles_void_function_with_no_side_effects() {
    // void function with dead code but no side effects
    // For test modules, we preserve leaf instructions
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();

    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let _ = b.builder.i_add(b.int_ty, None, c1, c2).expect("add");

    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // In void functions with no side effects, we treat leaf instructions as roots
    // for test compatibility. The add should be folded to constant 3.
    assert!(
        result.has_constant_u32(3) || !result.has_opcode(Op::IAdd),
        "add should either fold to 3 or be removed"
    );
}

// =============================================================================
// Multiple Dead Branches Tests
// =============================================================================

#[test]
fn removes_multiple_independent_dead_branches() {
    // dead_a = 10 + 20
    // dead_b = 30 * 40
    // dead_c = 50 - 60
    // live = 1 + 2
    // return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    // Dead branch A
    let c10 = b.const_i32(10);
    let c20 = b.const_i32(20);
    let dead_a = b.builder.i_add(b.int_ty, None, c10, c20).expect("dead_a");

    // Dead branch B
    let c30 = b.const_i32(30);
    let c40 = b.const_i32(40);
    let dead_b = b.builder.i_mul(b.int_ty, None, c30, c40).expect("dead_b");

    // Dead branch C
    let c50 = b.const_i32(50);
    let c60 = b.const_i32(60);
    let dead_c = b.builder.i_sub(b.int_ty, None, c50, c60).expect("dead_c");

    // Live computation
    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let live = b.builder.i_add(b.int_ty, None, c1, c2).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All dead instructions should be removed
    for dead_id in [dead_a, dead_b, dead_c] {
        let dead_exists = result
            .module
            .all_inst_iter()
            .any(|inst| inst.result_id == Some(dead_id));
        assert!(!dead_exists, "dead instruction {} should be removed", dead_id);
    }

    // Dead constants should be removed
    for val in [10, 20, 30, 40, 50, 60] {
        assert!(
            !result.has_constant_u32(val as u32),
            "dead constant {} should be removed",
            val
        );
    }

    // Live result should be folded to 3
    assert!(result.has_constant_u32(3), "1+2 should fold to 3");
}

#[test]
fn removes_dead_tree_structure() {
    // Tree structure where multiple dead branches converge:
    //     dead1 = 10 + 20
    //     dead2 = 30 + 40
    //     dead3 = dead1 + dead2  (convergence point, still dead)
    //     live = 1 + 2
    //     return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c10 = b.const_i32(10);
    let c20 = b.const_i32(20);
    let dead1 = b.builder.i_add(b.int_ty, None, c10, c20).expect("dead1");

    let c30 = b.const_i32(30);
    let c40 = b.const_i32(40);
    let dead2 = b.builder.i_add(b.int_ty, None, c30, c40).expect("dead2");

    // Convergence - dead because it only feeds dead3
    let dead3 = b.builder.i_add(b.int_ty, None, dead1, dead2).expect("dead3");

    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let live = b.builder.i_add(b.int_ty, None, c1, c2).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All dead instructions should be removed
    for dead_id in [dead1, dead2, dead3] {
        let dead_exists = result
            .module
            .all_inst_iter()
            .any(|inst| inst.result_id == Some(dead_id));
        assert!(!dead_exists, "dead instruction {} should be removed", dead_id);
    }

    // Live result should be folded to 3
    assert!(result.has_constant_u32(3), "1+2 should fold to 3");
}

// =============================================================================
// Diamond Pattern Tests (shared subexpressions)
// =============================================================================

#[test]
fn preserves_diamond_pattern_all_live() {
    // Diamond pattern where shared value feeds multiple live paths:
    //     shared = 5
    //     left = shared + 1
    //     right = shared + 2
    //     result = left + right
    //     return result
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let shared = b.const_i32(5);
    let c1 = b.const_i32(1);
    let left = b.builder.i_add(b.int_ty, None, shared, c1).expect("left");
    let c2 = b.const_i32(2);
    let right = b.builder.i_add(b.int_ty, None, shared, c2).expect("right");
    let result = b.builder.i_add(b.int_ty, None, left, right).expect("result");

    b.builder.ret_value(result).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result_mod = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Result should be folded: (5+1) + (5+2) = 6 + 7 = 13
    assert!(result_mod.has_constant_u32(13), "result should fold to 13");
}

#[test]
fn removes_dead_branch_in_diamond() {
    // Diamond pattern but one branch is dead:
    //     shared = 5
    //     dead_branch = shared + 100  (not used)
    //     live_branch = shared + 1
    //     return live_branch
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let shared = b.const_i32(5);
    let c100 = b.const_i32(100);
    let dead_branch = b.builder.i_add(b.int_ty, None, shared, c100).expect("dead");
    let c1 = b.const_i32(1);
    let live_branch = b.builder.i_add(b.int_ty, None, shared, c1).expect("live");

    b.builder.ret_value(live_branch).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Dead branch should be removed
    let dead_exists = result
        .module
        .all_inst_iter()
        .any(|inst| inst.result_id == Some(dead_branch));
    assert!(!dead_exists, "dead branch should be removed");

    // Dead constant 100 should be removed
    assert!(
        !result.has_constant_u32(100),
        "dead constant 100 should be removed"
    );

    // Live result should be 6 (5+1)
    assert!(result.has_constant_u32(6), "5+1 should fold to 6");
}

// =============================================================================
// Parameter Usage Tests
// =============================================================================

#[test]
fn preserves_live_chain_with_param() {
    // fn f(x: i32) -> i32 {
    //     a = x + 1
    //     b = a * 2
    //     c = b - 3
    //     return c  (all live)
    // }
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param = b.builder.function_parameter(b.int_ty).expect("param");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let a = b.builder.i_add(b.int_ty, None, param, c1).expect("a");
    let c2 = b.const_i32(2);
    let bb = b.builder.i_mul(b.int_ty, None, a, c2).expect("b");
    let c3 = b.const_i32(3);
    let c = b.builder.i_sub(b.int_ty, None, bb, c3).expect("c");

    b.builder.ret_value(c).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // With an unknown param, we can't fold everything, but we should preserve the chain
    // The optimizer might simplify: (x+1)*2 - 3 = 2x + 2 - 3 = 2x - 1
    // or keep the chain. Either way, some computation should remain.
    let has_arith = result.has_opcode(Op::IAdd)
        || result.has_opcode(Op::IMul)
        || result.has_opcode(Op::ISub);
    assert!(
        has_arith,
        "some arithmetic should remain with unknown param"
    );
}

#[test]
fn removes_dead_param_use_keeps_live_param_use() {
    // fn f(x: i32, y: i32) -> i32 {
    //     dead = y + 100   (y is used but result is dead)
    //     live = x * 2     (returned)
    //     return live
    // }
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param_x = b.builder.function_parameter(b.int_ty).expect("param x");
    let param_y = b.builder.function_parameter(b.int_ty).expect("param y");
    b.builder.begin_block(None).expect("begin block");

    let c100 = b.const_i32(100);
    let dead = b.builder.i_add(b.int_ty, None, param_y, c100).expect("dead");
    let c2 = b.const_i32(2);
    let live = b.builder.i_mul(b.int_ty, None, param_x, c2).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Dead instruction using param_y should be removed
    let dead_exists = result
        .module
        .all_inst_iter()
        .any(|inst| inst.result_id == Some(dead));
    assert!(!dead_exists, "dead add should be removed");

    // Dead constant 100 should be removed
    assert!(
        !result.has_constant_u32(100),
        "dead constant 100 should be removed"
    );

    // Live multiplication should remain
    assert!(result.has_opcode(Op::IMul), "live mul should remain");

    // Live constant 2 should remain
    assert!(result.has_constant_u32(2), "live constant 2 should remain");
}

// =============================================================================
// Mixed Operation Tests
// =============================================================================

#[test]
fn removes_dead_across_different_op_types() {
    // dead_add = 10 + 20
    // dead_mul = dead_add * 3
    // dead_sub = dead_mul - 5
    // dead_and = dead_sub & 0xFF (bitwise)
    // live = 1 + 2
    // return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c10 = b.const_i32(10);
    let c20 = b.const_i32(20);
    let dead_add = b.builder.i_add(b.int_ty, None, c10, c20).expect("dead_add");
    let c3 = b.const_i32(3);
    let dead_mul = b
        .builder
        .i_mul(b.int_ty, None, dead_add, c3)
        .expect("dead_mul");
    let c5 = b.const_i32(5);
    let dead_sub = b
        .builder
        .i_sub(b.int_ty, None, dead_mul, c5)
        .expect("dead_sub");
    let c255 = b.const_i32(255);
    let dead_and = b
        .builder
        .bitwise_and(b.int_ty, None, dead_sub, c255)
        .expect("dead_and");

    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let live = b.builder.i_add(b.int_ty, None, c1, c2).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All dead instructions should be removed
    for dead_id in [dead_add, dead_mul, dead_sub, dead_and] {
        let dead_exists = result
            .module
            .all_inst_iter()
            .any(|inst| inst.result_id == Some(dead_id));
        assert!(!dead_exists, "dead instruction {} should be removed", dead_id);
    }

    // Live result should be 3
    assert!(result.has_constant_u32(3), "1+2 should fold to 3");
}

#[test]
fn removes_dead_bitwise_chain() {
    // dead_or = 0xF0 | 0x0F
    // dead_xor = dead_or ^ 0xFF
    // dead_not = ~dead_xor
    // live = 1 + 1
    // return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c_f0 = b.const_i32(0xF0_i32);
    let c_0f = b.const_i32(0x0F);
    let dead_or = b
        .builder
        .bitwise_or(b.int_ty, None, c_f0, c_0f)
        .expect("dead_or");
    let c_ff = b.const_i32(0xFF_i32);
    let dead_xor = b
        .builder
        .bitwise_xor(b.int_ty, None, dead_or, c_ff)
        .expect("dead_xor");
    let dead_not = b.builder.not(b.int_ty, None, dead_xor).expect("dead_not");

    let c1a = b.const_i32(1);
    let c1b = b.const_i32(1);
    let live = b.builder.i_add(b.int_ty, None, c1a, c1b).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All dead bitwise ops should be removed
    for dead_id in [dead_or, dead_xor, dead_not] {
        let dead_exists = result
            .module
            .all_inst_iter()
            .any(|inst| inst.result_id == Some(dead_id));
        assert!(!dead_exists, "dead instruction {} should be removed", dead_id);
    }

    // Live result should be 2
    assert!(result.has_constant_u32(2), "1+1 should fold to 2");
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn handles_single_live_instruction() {
    // Minimal case: just return a constant
    // return 42
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c42 = b.const_i32(42);
    b.builder.ret_value(c42).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Constant 42 should remain
    assert!(result.has_constant_u32(42), "constant 42 should remain");
}

#[test]
fn handles_all_dead_except_return() {
    // Everything is dead except the returned constant
    // dead1 = 101 + 102
    // dead2 = 103 + 104
    // dead3 = 105 + 106
    // return 99
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c101 = b.const_i32(101);
    let c102 = b.const_i32(102);
    let dead1 = b.builder.i_add(b.int_ty, None, c101, c102).expect("dead1");
    let c103 = b.const_i32(103);
    let c104 = b.const_i32(104);
    let dead2 = b.builder.i_add(b.int_ty, None, c103, c104).expect("dead2");
    let c105 = b.const_i32(105);
    let c106 = b.const_i32(106);
    let dead3 = b.builder.i_add(b.int_ty, None, c105, c106).expect("dead3");

    let c99 = b.const_i32(99);
    b.builder.ret_value(c99).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All dead instructions should be removed
    for dead_id in [dead1, dead2, dead3] {
        let dead_exists = result
            .module
            .all_inst_iter()
            .any(|inst| inst.result_id == Some(dead_id));
        assert!(!dead_exists, "dead instruction {} should be removed", dead_id);
    }

    // Dead constants should be removed
    for val in [101u32, 102, 103, 104, 105, 106] {
        assert!(
            !result.has_constant_u32(val),
            "dead constant {} should be removed",
            val
        );
    }

    // Only the return constant should remain
    assert!(result.has_constant_u32(99), "return constant 99 should remain");
}

#[test]
fn handles_deeply_nested_dead_expression() {
    // Deep nesting that's all dead:
    // a = 1 + 2
    // b = a + 3
    // c = b + 4
    // d = c + 5
    // e = d + 6
    // f = e + 7 (deep dead chain)
    // live = 100
    // return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let a = b.builder.i_add(b.int_ty, None, c1, c2).expect("a");
    let c3 = b.const_i32(3);
    let bb = b.builder.i_add(b.int_ty, None, a, c3).expect("b");
    let c4 = b.const_i32(4);
    let c = b.builder.i_add(b.int_ty, None, bb, c4).expect("c");
    let c5 = b.const_i32(5);
    let d = b.builder.i_add(b.int_ty, None, c, c5).expect("d");
    let c6 = b.const_i32(6);
    let e = b.builder.i_add(b.int_ty, None, d, c6).expect("e");
    let c7 = b.const_i32(7);
    let f = b.builder.i_add(b.int_ty, None, e, c7).expect("f");

    let c100 = b.const_i32(100);
    b.builder.ret_value(c100).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All dead chain should be removed
    for dead_id in [a, bb, c, d, e, f] {
        let dead_exists = result
            .module
            .all_inst_iter()
            .any(|inst| inst.result_id == Some(dead_id));
        assert!(!dead_exists, "dead instruction {} should be removed", dead_id);
    }

    // Return constant should remain
    assert!(result.has_constant_u32(100), "constant 100 should remain");
}

#[test]
fn handles_identity_operations_dead() {
    // Dead identity operations that might not be optimized away normally:
    // dead_add0 = x + 0  (identity, but dead)
    // dead_mul1 = y * 1  (identity, but dead)
    // live = 5 + 5
    // return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param_x = b.builder.function_parameter(b.int_ty).expect("param x");
    let param_y = b.builder.function_parameter(b.int_ty).expect("param y");
    b.builder.begin_block(None).expect("begin block");

    let c0 = b.const_i32(0);
    let dead_add0 = b
        .builder
        .i_add(b.int_ty, None, param_x, c0)
        .expect("dead x+0");
    let c1 = b.const_i32(1);
    let dead_mul1 = b
        .builder
        .i_mul(b.int_ty, None, param_y, c1)
        .expect("dead y*1");

    let c5a = b.const_i32(5);
    let c5b = b.const_i32(5);
    let live = b.builder.i_add(b.int_ty, None, c5a, c5b).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Dead identity operations should be removed
    for dead_id in [dead_add0, dead_mul1] {
        let dead_exists = result
            .module
            .all_inst_iter()
            .any(|inst| inst.result_id == Some(dead_id));
        assert!(!dead_exists, "dead instruction {} should be removed", dead_id);
    }

    // Live result should be 10
    assert!(result.has_constant_u32(10), "5+5 should fold to 10");
}

#[test]
fn preserves_both_uses_of_shared_value() {
    // Shared value used by TWO live paths:
    //     shared = x + 1
    //     a = shared * 2
    //     b = shared * 3
    //     result = a + b
    //     return result
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param = b.builder.function_parameter(b.int_ty).expect("param");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let shared = b.builder.i_add(b.int_ty, None, param, c1).expect("shared");
    let c2 = b.const_i32(2);
    let a = b.builder.i_mul(b.int_ty, None, shared, c2).expect("a");
    let c3 = b.const_i32(3);
    let bb = b.builder.i_mul(b.int_ty, None, shared, c3).expect("b");
    let result = b.builder.i_add(b.int_ty, None, a, bb).expect("result");

    b.builder.ret_value(result).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result_mod = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Some arithmetic should remain (can't fold with unknown param)
    // The expression is: (x+1)*2 + (x+1)*3 = 2x+2 + 3x+3 = 5x+5 = 5*(x+1)
    let has_arith = result_mod.has_opcode(Op::IAdd)
        || result_mod.has_opcode(Op::IMul)
        || result_mod.has_opcode(Op::ISub);
    assert!(
        has_arith,
        "some arithmetic should remain with unknown param"
    );
}

// =============================================================================
// Negative Tests (Things That Should NOT Be Removed)
// =============================================================================

#[test]
fn does_not_remove_returned_constant() {
    // Simple: just return a constant
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c42 = b.const_i32(42);
    b.builder.ret_value(c42).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        result.has_constant_u32(42),
        "returned constant must not be removed"
    );
}

#[test]
fn does_not_remove_computation_feeding_return() {
    // a = 10 + 20
    // return a
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c10 = b.const_i32(10);
    let c20 = b.const_i32(20);
    let a = b.builder.i_add(b.int_ty, None, c10, c20).expect("a");

    b.builder.ret_value(a).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Should be folded to 30, but the VALUE must be present
    assert!(
        result.has_constant_u32(30),
        "computation feeding return should fold to 30"
    );
}

#[test]
fn does_not_remove_param_based_computation() {
    // fn f(x: i32) -> i32 {
    //     return x + 5
    // }
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param = b.builder.function_parameter(b.int_ty).expect("param");
    b.builder.begin_block(None).expect("begin block");

    let c5 = b.const_i32(5);
    let result = b.builder.i_add(b.int_ty, None, param, c5).expect("result");

    b.builder.ret_value(result).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result_mod = OptimizedModule::from_words(&words).expect("optimizer runs");

    // IAdd should remain (can't fold with unknown param)
    assert!(
        result_mod.has_opcode(Op::IAdd),
        "add with param must not be removed"
    );

    // Constant 5 must remain
    assert!(
        result_mod.has_constant_u32(5),
        "constant 5 must not be removed"
    );
}

#[test]
fn does_not_remove_intermediate_in_live_chain() {
    // a = x + 1
    // b = a * 2   (intermediate, must not be removed)
    // return b
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param = b.builder.function_parameter(b.int_ty).expect("param");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let a = b.builder.i_add(b.int_ty, None, param, c1).expect("a");
    let c2 = b.const_i32(2);
    let bb = b.builder.i_mul(b.int_ty, None, a, c2).expect("b");

    b.builder.ret_value(bb).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Both operations should remain (can't fold with unknown param)
    // Or it might be optimized to 2*x + 2, either way there should be arithmetic
    let has_arith = result.has_opcode(Op::IAdd) || result.has_opcode(Op::IMul);
    assert!(has_arith, "arithmetic operations must not be removed");
}

// =============================================================================
// Edge Case Tests - Void Functions
// =============================================================================

#[test]
fn void_function_all_dead_code_removed() {
    // Void function with no side effects - all computations are dead
    // fn f() { let x = 1 + 2; let y = x * 3; }
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let dead_add = b.builder.i_add(b.int_ty, None, c1, c2).expect("dead_add");
    let c3 = b.const_i32(3);
    let _dead_mul = b.builder.i_mul(b.int_ty, None, dead_add, c3).expect("dead_mul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All arithmetic should be removed - no side effects in void function
    assert!(
        !result.has_opcode(Op::IAdd),
        "dead add in void function should be removed"
    );
    assert!(
        !result.has_opcode(Op::IMul),
        "dead mul in void function should be removed"
    );
}

#[test]
fn void_function_with_param_all_dead() {
    // Void function that uses a parameter but has no side effects
    // fn f(x: i32) { let y = x + 1; let z = y * 2; }
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let params = b.begin_function_with_params(vec![b.int_ty]);
    let x = params[0];
    let c1 = b.const_i32(1);
    let dead_add = b.builder.i_add(b.int_ty, None, x, c1).expect("dead_add");
    let c2 = b.const_i32(2);
    let _dead_mul = b.builder.i_mul(b.int_ty, None, dead_add, c2).expect("dead_mul");
    let words = b.finish();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All arithmetic should be removed - no side effects
    assert!(
        !result.has_opcode(Op::IAdd),
        "dead add with param in void function should be removed"
    );
    assert!(
        !result.has_opcode(Op::IMul),
        "dead mul with param in void function should be removed"
    );
}

// =============================================================================
// Edge Case Tests - Multiple Functions
// =============================================================================

#[test]
fn dce_across_multiple_functions() {
    // Two functions in same module - each should be DCE'd independently
    // fn f() -> i32 { return 5 + 3; }
    // fn g() { let dead = 10 + 20; }
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();

    // First function: returns a value (live)
    let func1_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func1_ty)
        .expect("begin f");
    b.builder.begin_block(None).expect("begin block");
    let c5 = b.const_i32(5);
    let c3 = b.const_i32(3);
    let live_add = b.builder.i_add(b.int_ty, None, c5, c3).expect("live_add");
    b.builder.ret_value(live_add).expect("ret");
    b.builder.end_function().expect("end f");

    // Second function: void with dead code
    let func2_ty = b.builder.type_function(b.void_ty, vec![]);
    b.builder
        .begin_function(b.void_ty, None, rspirv::spirv::FunctionControl::NONE, func2_ty)
        .expect("begin g");
    b.builder.begin_block(None).expect("begin block");
    let c10 = b.const_i32(10);
    let c20 = b.const_i32(20);
    let _dead_add = b.builder.i_add(b.int_ty, None, c10, c20).expect("dead_add");
    b.builder.ret().expect("ret");
    b.builder.end_function().expect("end g");

    let words = b.builder.module().assemble();
    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Function 1's result should be folded to 8
    assert!(
        result.has_constant_u32(8),
        "live computation 5+3 should fold to 8"
    );

    // Function 2's dead add (10+20=30) should NOT produce a constant 30
    // (since the add is dead and should be removed)
    // Note: constant 10 or 20 might exist if they're used elsewhere, but 30 should not
    let has_30 = result.module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.operands.iter().any(|op| match op {
                rspirv::dr::Operand::LiteralBit32(v) => *v == 30,
                _ => false,
            })
    });
    assert!(
        !has_30,
        "dead computation 10+20 should not produce constant 30"
    );
}

// =============================================================================
// Edge Case Tests - Complex Liveness Patterns
// =============================================================================

#[test]
fn dce_with_diamond_use_pattern() {
    // Diamond pattern where shared value feeds two paths, only one is live:
    //     base = param + 1
    //     live_path = base * 2
    //     dead_path = base * 3  (not returned)
    //     return live_path
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param = b.builder.function_parameter(b.int_ty).expect("param");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let base = b.builder.i_add(b.int_ty, None, param, c1).expect("base");
    let c2 = b.const_i32(2);
    let live_path = b.builder.i_mul(b.int_ty, None, base, c2).expect("live_path");
    let c3 = b.const_i32(3);
    let dead_path = b.builder.i_mul(b.int_ty, None, base, c3).expect("dead_path");

    b.builder.ret_value(live_path).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // dead_path instruction should be removed
    let dead_exists = result
        .module
        .all_inst_iter()
        .any(|inst| inst.result_id == Some(dead_path));
    assert!(
        !dead_exists,
        "dead_path (base * 3) should be removed since it's not returned"
    );

    // base and live_path should remain (or be optimized but not removed)
    // The expression (param + 1) * 2 should be preserved in some form
    let has_arith = result.has_opcode(Op::IAdd) || result.has_opcode(Op::IMul);
    assert!(has_arith, "live path arithmetic should be preserved");
}

#[test]
fn dce_constant_used_by_both_dead_and_live() {
    // Constant used by both dead and live code - constant must be kept
    //     shared_const = 5
    //     dead_use = shared_const + 1
    //     live_use = param * shared_const
    //     return live_use
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param = b.builder.function_parameter(b.int_ty).expect("param");
    b.builder.begin_block(None).expect("begin block");

    let shared_const = b.const_i32(5);
    let c1 = b.const_i32(1);
    let _dead_use = b.builder.i_add(b.int_ty, None, shared_const, c1).expect("dead_use");
    let live_use = b.builder.i_mul(b.int_ty, None, param, shared_const).expect("live_use");

    b.builder.ret_value(live_use).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Constant 5 must be preserved (used by live_use)
    assert!(
        result.has_constant_u32(5),
        "constant 5 must be preserved (used by live code)"
    );

    // Constant 6 (from dead 5+1) should NOT exist
    let has_6 = result.module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.operands.iter().any(|op| match op {
                rspirv::dr::Operand::LiteralBit32(v) => *v == 6,
                _ => false,
            })
    });
    assert!(
        !has_6,
        "dead computation 5+1=6 should not produce constant"
    );
}

#[test]
fn dce_long_dead_chain_completely_removed() {
    // Long chain of dead computations
    //     d1 = 1 + 2
    //     d2 = d1 * 3
    //     d3 = d2 - 4
    //     d4 = d3 + 5
    //     d5 = d4 * 6
    //     return 42
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let d1 = b.builder.i_add(b.int_ty, None, c1, c2).expect("d1");
    let c3 = b.const_i32(3);
    let d2 = b.builder.i_mul(b.int_ty, None, d1, c3).expect("d2");
    let c4 = b.const_i32(4);
    let d3 = b.builder.i_sub(b.int_ty, None, d2, c4).expect("d3");
    let c5 = b.const_i32(5);
    let d4 = b.builder.i_add(b.int_ty, None, d3, c5).expect("d4");
    let c6 = b.const_i32(6);
    let _d5 = b.builder.i_mul(b.int_ty, None, d4, c6).expect("d5");

    let c42 = b.const_i32(42);
    b.builder.ret_value(c42).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All arithmetic operations should be removed (the entire chain is dead)
    assert!(
        !result.has_opcode(Op::IAdd),
        "dead chain IAdd should be removed"
    );
    assert!(
        !result.has_opcode(Op::IMul),
        "dead chain IMul should be removed"
    );
    assert!(
        !result.has_opcode(Op::ISub),
        "dead chain ISub should be removed"
    );

    // Return value 42 must be preserved
    assert!(result.has_constant_u32(42), "return value 42 must be preserved");
}

#[test]
fn dce_preserves_all_ops_in_live_chain() {
    // Long live chain - nothing should be removed
    //     a = param + 1
    //     b = a * 2
    //     c = b - 3
    //     d = c + 4
    //     return d
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let param = b.builder.function_parameter(b.int_ty).expect("param");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let a = b.builder.i_add(b.int_ty, None, param, c1).expect("a");
    let c2 = b.const_i32(2);
    let bb = b.builder.i_mul(b.int_ty, None, a, c2).expect("b");
    let c3 = b.const_i32(3);
    let c = b.builder.i_sub(b.int_ty, None, bb, c3).expect("c");
    let c4 = b.const_i32(4);
    let d = b.builder.i_add(b.int_ty, None, c, c4).expect("d");

    b.builder.ret_value(d).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Some arithmetic must remain (can't fully fold with unknown param)
    // The expression is: ((param + 1) * 2 - 3) + 4 = 2*param + 2 - 3 + 4 = 2*param + 3
    // Either way, some ops should remain
    let has_arith = result.has_opcode(Op::IAdd)
        || result.has_opcode(Op::IMul)
        || result.has_opcode(Op::ISub);
    assert!(has_arith, "live chain arithmetic should be preserved");
}

// =============================================================================
// Edge Case Tests - Bitwise Operations
// =============================================================================

#[test]
fn dce_dead_bitwise_operations() {
    // Dead bitwise operations should be removed
    //     dead_and = x & y
    //     dead_or = dead_and | z
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    let z = b.builder.function_parameter(b.int_ty).expect("z");
    b.builder.begin_block(None).expect("begin block");

    let dead_and = b.builder.bitwise_and(b.int_ty, None, x, y).expect("dead_and");
    let _dead_or = b.builder.bitwise_or(b.int_ty, None, dead_and, z).expect("dead_or");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All bitwise ops should be removed
    assert!(
        !result.has_opcode(Op::BitwiseAnd),
        "dead BitwiseAnd should be removed"
    );
    assert!(
        !result.has_opcode(Op::BitwiseOr),
        "dead BitwiseOr should be removed"
    );
}

#[test]
fn dce_preserves_live_bitwise_operations() {
    // Live bitwise operations should NOT be removed
    //     result = (x & y) | z
    //     return result
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    let z = b.builder.function_parameter(b.int_ty).expect("z");
    b.builder.begin_block(None).expect("begin block");

    let and_result = b.builder.bitwise_and(b.int_ty, None, x, y).expect("and");
    let result = b.builder.bitwise_or(b.int_ty, None, and_result, z).expect("or");

    b.builder.ret_value(result).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result_mod = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Bitwise ops should remain (can't fold with unknown params)
    assert!(
        result_mod.has_opcode(Op::BitwiseAnd),
        "live BitwiseAnd should be preserved"
    );
    assert!(
        result_mod.has_opcode(Op::BitwiseOr),
        "live BitwiseOr should be preserved"
    );
}

// =============================================================================
// Edge Case Tests - Shift Operations
// =============================================================================

#[test]
fn dce_dead_shift_operations() {
    // Dead shift operations should be removed
    //     dead_shl = x << 2
    //     dead_shr = dead_shl >> 1
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let c2 = b.const_i32(2);
    let dead_shl = b.builder.shift_left_logical(b.int_ty, None, x, c2).expect("dead_shl");
    let c1 = b.const_i32(1);
    let _dead_shr = b.builder.shift_right_logical(b.int_ty, None, dead_shl, c1).expect("dead_shr");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        !result.has_opcode(Op::ShiftLeftLogical),
        "dead ShiftLeftLogical should be removed"
    );
    assert!(
        !result.has_opcode(Op::ShiftRightLogical),
        "dead ShiftRightLogical should be removed"
    );
}

#[test]
fn dce_preserves_live_shift_operations() {
    // Live shift operations should NOT be removed
    //     result = (x << 2) >> 1
    //     return result
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let c2 = b.const_i32(2);
    let shl = b.builder.shift_left_logical(b.int_ty, None, x, c2).expect("shl");
    let c1 = b.const_i32(1);
    let result = b.builder.shift_right_logical(b.int_ty, None, shl, c1).expect("shr");

    b.builder.ret_value(result).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result_mod = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Some shift ops should remain (may be optimized to single shift by 1)
    let has_shift = result_mod.has_opcode(Op::ShiftLeftLogical)
        || result_mod.has_opcode(Op::ShiftRightLogical);
    assert!(has_shift, "live shift operations should be preserved");
}

// =============================================================================
// Edge Case Tests - Comparison Operations
// =============================================================================

#[test]
fn dce_dead_comparison_operations() {
    // Dead comparison operations should be removed
    //     dead_cmp = x < y
    //     return 42
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    b.builder.begin_block(None).expect("begin block");

    let _dead_cmp = b.builder.s_less_than(b.bool_ty, None, x, y).expect("dead_cmp");

    let c42 = b.const_i32(42);
    b.builder.ret_value(c42).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        !result.has_opcode(Op::SLessThan),
        "dead SLessThan should be removed"
    );
}

#[test]
fn dce_dead_equality_comparison() {
    // Dead equality comparison should be removed
    //     dead_eq = x == y
    //     dead_ne = x != y
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    b.builder.begin_block(None).expect("begin block");

    let _dead_eq = b.builder.i_equal(b.bool_ty, None, x, y).expect("dead_eq");
    let _dead_ne = b.builder.i_not_equal(b.bool_ty, None, x, y).expect("dead_ne");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        !result.has_opcode(Op::IEqual),
        "dead IEqual should be removed"
    );
    assert!(
        !result.has_opcode(Op::INotEqual),
        "dead INotEqual should be removed"
    );
}

// =============================================================================
// Edge Case Tests - Negation Operations
// =============================================================================

#[test]
fn dce_dead_negation() {
    // Dead negation should be removed
    //     dead_neg = -x
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let _dead_neg = b.builder.s_negate(b.int_ty, None, x).expect("dead_neg");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        !result.has_opcode(Op::SNegate),
        "dead SNegate should be removed"
    );
}

#[test]
fn dce_preserves_live_negation() {
    // Live negation should NOT be removed
    //     result = -x
    //     return result
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let result = b.builder.s_negate(b.int_ty, None, x).expect("neg");

    b.builder.ret_value(result).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result_mod = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Negation might be represented as Mul by -1, but some arithmetic should remain
    let has_neg = result_mod.has_opcode(Op::SNegate) || result_mod.has_opcode(Op::IMul);
    assert!(has_neg, "live negation should be preserved");
}

#[test]
fn dce_dead_bitwise_not() {
    // Dead bitwise NOT should be removed
    //     dead_not = ~x
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let _dead_not = b.builder.not(b.int_ty, None, x).expect("dead_not");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        !result.has_opcode(Op::Not),
        "dead Not should be removed"
    );
}

// =============================================================================
// Edge Case Tests - Deeply Nested Dead Expressions
// =============================================================================

#[test]
fn dce_deeply_nested_dead_expression_10_levels() {
    // 10-level deep dead expression tree
    //     d1 = x + 1
    //     d2 = d1 * 2
    //     d3 = d2 + 3
    //     d4 = d3 * 4
    //     d5 = d4 + 5
    //     d6 = d5 * 6
    //     d7 = d6 + 7
    //     d8 = d7 * 8
    //     d9 = d8 + 9
    //     d10 = d9 * 10
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let d1 = b.builder.i_add(b.int_ty, None, x, c1).expect("d1");
    let c2 = b.const_i32(2);
    let d2 = b.builder.i_mul(b.int_ty, None, d1, c2).expect("d2");
    let c3 = b.const_i32(3);
    let d3 = b.builder.i_add(b.int_ty, None, d2, c3).expect("d3");
    let c4 = b.const_i32(4);
    let d4 = b.builder.i_mul(b.int_ty, None, d3, c4).expect("d4");
    let c5 = b.const_i32(5);
    let d5 = b.builder.i_add(b.int_ty, None, d4, c5).expect("d5");
    let c6 = b.const_i32(6);
    let d6 = b.builder.i_mul(b.int_ty, None, d5, c6).expect("d6");
    let c7 = b.const_i32(7);
    let d7 = b.builder.i_add(b.int_ty, None, d6, c7).expect("d7");
    let c8 = b.const_i32(8);
    let d8 = b.builder.i_mul(b.int_ty, None, d7, c8).expect("d8");
    let c9 = b.const_i32(9);
    let d9 = b.builder.i_add(b.int_ty, None, d8, c9).expect("d9");
    let c10 = b.const_i32(10);
    let _d10 = b.builder.i_mul(b.int_ty, None, d9, c10).expect("d10");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All arithmetic should be removed
    assert!(
        !result.has_opcode(Op::IAdd),
        "dead nested IAdd should be removed"
    );
    assert!(
        !result.has_opcode(Op::IMul),
        "dead nested IMul should be removed"
    );
}

// =============================================================================
// Edge Case Tests - Multiple Independent Dead Chains
// =============================================================================

#[test]
fn dce_multiple_independent_dead_chains() {
    // Three independent dead chains
    //     chain1: a1 = 1+2, a2 = a1*3
    //     chain2: b1 = 4+5, b2 = b1*6
    //     chain3: c1 = 7+8, c2 = c1*9
    //     return 42
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    // Chain 1
    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let a1 = b.builder.i_add(b.int_ty, None, c1, c2).expect("a1");
    let c3 = b.const_i32(3);
    let _a2 = b.builder.i_mul(b.int_ty, None, a1, c3).expect("a2");

    // Chain 2
    let c4 = b.const_i32(4);
    let c5 = b.const_i32(5);
    let b1 = b.builder.i_add(b.int_ty, None, c4, c5).expect("b1");
    let c6 = b.const_i32(6);
    let _b2 = b.builder.i_mul(b.int_ty, None, b1, c6).expect("b2");

    // Chain 3
    let c7 = b.const_i32(7);
    let c8 = b.const_i32(8);
    let cc1 = b.builder.i_add(b.int_ty, None, c7, c8).expect("c1");
    let c9 = b.const_i32(9);
    let _cc2 = b.builder.i_mul(b.int_ty, None, cc1, c9).expect("c2");

    let c42 = b.const_i32(42);
    b.builder.ret_value(c42).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All arithmetic should be removed
    assert!(
        !result.has_opcode(Op::IAdd),
        "dead chains' IAdd should be removed"
    );
    assert!(
        !result.has_opcode(Op::IMul),
        "dead chains' IMul should be removed"
    );
    // Return value preserved
    assert!(result.has_constant_u32(42), "return value 42 must be preserved");
}

// =============================================================================
// Edge Case Tests - Same Operand Used Multiple Times
// =============================================================================

#[test]
fn dce_same_operand_dead_and_live_uses() {
    // Same param used by both dead and live computations
    //     dead = x * 2
    //     live = x + 1
    //     return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let c2 = b.const_i32(2);
    let dead_mul = b.builder.i_mul(b.int_ty, None, x, c2).expect("dead");
    let c1 = b.const_i32(1);
    let live_add = b.builder.i_add(b.int_ty, None, x, c1).expect("live");

    b.builder.ret_value(live_add).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Dead mul should be removed
    let dead_exists = result
        .module
        .all_inst_iter()
        .any(|inst| inst.result_id == Some(dead_mul));
    assert!(!dead_exists, "dead mul should be removed");

    // Live add should remain
    assert!(result.has_opcode(Op::IAdd), "live add should be preserved");
}

#[test]
fn dce_same_constant_multiple_dead_uses() {
    // Same constant used by multiple dead computations
    //     dead1 = 5 + x
    //     dead2 = 5 * y
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    b.builder.begin_block(None).expect("begin block");

    let c5 = b.const_i32(5);
    let _dead1 = b.builder.i_add(b.int_ty, None, c5, x).expect("dead1");
    let _dead2 = b.builder.i_mul(b.int_ty, None, c5, y).expect("dead2");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All arithmetic should be removed
    assert!(
        !result.has_opcode(Op::IAdd),
        "dead IAdd should be removed"
    );
    assert!(
        !result.has_opcode(Op::IMul),
        "dead IMul should be removed"
    );
}

// =============================================================================
// Edge Case Tests - Constant-Only Computations
// =============================================================================

#[test]
fn dce_dead_constant_computation_not_folded_to_output() {
    // Dead constant computation should not produce output constant
    //     dead = 100 + 200  (= 300, but dead)
    //     return 1
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c100 = b.const_i32(100);
    let c200 = b.const_i32(200);
    let _dead = b.builder.i_add(b.int_ty, None, c100, c200).expect("dead");

    let c1 = b.const_i32(1);
    b.builder.ret_value(c1).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Constant 300 should NOT exist (the dead computation shouldn't be folded to output)
    let has_300 = result.module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.operands.iter().any(|op| match op {
                rspirv::dr::Operand::LiteralBit32(v) => *v == 300,
                _ => false,
            })
    });
    assert!(
        !has_300,
        "dead computation 100+200 should not produce constant 300"
    );

    // Return value 1 must be preserved
    assert!(result.has_constant_u32(1), "return value 1 must be preserved");
}

#[test]
fn dce_live_constant_computation_is_folded() {
    // Live constant computation should be folded
    //     live = 100 + 200  (= 300, live)
    //     return live
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    b.builder.begin_block(None).expect("begin block");

    let c100 = b.const_i32(100);
    let c200 = b.const_i32(200);
    let live = b.builder.i_add(b.int_ty, None, c100, c200).expect("live");

    b.builder.ret_value(live).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // Should be folded to 300
    assert!(
        result.has_constant_u32(300),
        "live computation 100+200 should fold to 300"
    );
    // IAdd should be eliminated (folded)
    assert!(
        !result.has_opcode(Op::IAdd),
        "constant add should be folded away"
    );
}

// =============================================================================
// Edge Case Tests - Mixed Operation Types
// =============================================================================

#[test]
fn dce_mixed_dead_ops_arithmetic_bitwise_shift() {
    // Mix of arithmetic, bitwise, and shift ops - all dead
    //     d1 = x + 1
    //     d2 = d1 & 0xFF
    //     d3 = d2 << 2
    //     d4 = d3 | 0x10
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let d1 = b.builder.i_add(b.int_ty, None, x, c1).expect("d1");
    let c_ff = b.const_i32(0xFF);
    let d2 = b.builder.bitwise_and(b.int_ty, None, d1, c_ff).expect("d2");
    let c2 = b.const_i32(2);
    let d3 = b.builder.shift_left_logical(b.int_ty, None, d2, c2).expect("d3");
    let c_10 = b.const_i32(0x10);
    let _d4 = b.builder.bitwise_or(b.int_ty, None, d3, c_10).expect("d4");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All ops should be removed
    assert!(!result.has_opcode(Op::IAdd), "dead IAdd should be removed");
    assert!(!result.has_opcode(Op::BitwiseAnd), "dead BitwiseAnd should be removed");
    assert!(!result.has_opcode(Op::ShiftLeftLogical), "dead ShiftLeftLogical should be removed");
    assert!(!result.has_opcode(Op::BitwiseOr), "dead BitwiseOr should be removed");
}

#[test]
fn dce_mixed_live_ops_arithmetic_bitwise_shift() {
    // Mix of arithmetic, bitwise, and shift ops - all live
    //     l1 = x + 1
    //     l2 = l1 & 0xFF
    //     l3 = l2 << 2
    //     l4 = l3 | 0x10
    //     return l4
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    b.builder.begin_block(None).expect("begin block");

    let c1 = b.const_i32(1);
    let l1 = b.builder.i_add(b.int_ty, None, x, c1).expect("l1");
    let c_ff = b.const_i32(0xFF);
    let l2 = b.builder.bitwise_and(b.int_ty, None, l1, c_ff).expect("l2");
    let c2 = b.const_i32(2);
    let l3 = b.builder.shift_left_logical(b.int_ty, None, l2, c2).expect("l3");
    let c_10 = b.const_i32(0x10);
    let l4 = b.builder.bitwise_or(b.int_ty, None, l3, c_10).expect("l4");

    b.builder.ret_value(l4).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    // All ops should remain (can't fold with unknown param)
    assert!(result.has_opcode(Op::IAdd), "live IAdd should be preserved");
    assert!(result.has_opcode(Op::BitwiseAnd), "live BitwiseAnd should be preserved");
    assert!(result.has_opcode(Op::ShiftLeftLogical), "live ShiftLeftLogical should be preserved");
    assert!(result.has_opcode(Op::BitwiseOr), "live BitwiseOr should be preserved");
}

// =============================================================================
// Edge Case Tests - XOR Operations
// =============================================================================

#[test]
fn dce_dead_xor_operations() {
    // Dead XOR should be removed
    //     dead_xor = x ^ y
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    b.builder.begin_block(None).expect("begin block");

    let _dead_xor = b.builder.bitwise_xor(b.int_ty, None, x, y).expect("dead_xor");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        !result.has_opcode(Op::BitwiseXor),
        "dead BitwiseXor should be removed"
    );
}

#[test]
fn dce_preserves_live_xor_operations() {
    // Live XOR should NOT be removed
    //     result = x ^ y
    //     return result
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    b.builder.begin_block(None).expect("begin block");

    let result = b.builder.bitwise_xor(b.int_ty, None, x, y).expect("xor");

    b.builder.ret_value(result).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result_mod = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        result_mod.has_opcode(Op::BitwiseXor),
        "live BitwiseXor should be preserved"
    );
}

// =============================================================================
// Edge Case Tests - Subtraction Operations
// =============================================================================

#[test]
fn dce_dead_subtraction() {
    // Dead subtraction should be removed
    //     dead_sub = x - y
    //     return 0
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    b.builder.begin_block(None).expect("begin block");

    let _dead_sub = b.builder.i_sub(b.int_ty, None, x, y).expect("dead_sub");

    let c0 = b.const_i32(0);
    b.builder.ret_value(c0).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        !result.has_opcode(Op::ISub),
        "dead ISub should be removed"
    );
}

#[test]
fn dce_preserves_live_subtraction() {
    // Live subtraction should NOT be removed
    //     result = x - y
    //     return result
    let _guard = OptimizerEnvGuard::new();

    let mut b = TestModuleBuilder::new();
    let func_ty = b.builder.type_function(b.int_ty, vec![b.int_ty, b.int_ty]);
    b.builder
        .begin_function(b.int_ty, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("begin function");
    let x = b.builder.function_parameter(b.int_ty).expect("x");
    let y = b.builder.function_parameter(b.int_ty).expect("y");
    b.builder.begin_block(None).expect("begin block");

    let result = b.builder.i_sub(b.int_ty, None, x, y).expect("sub");

    b.builder.ret_value(result).expect("ret");
    b.builder.end_function().expect("end function");
    let words = b.builder.module().assemble();

    let result_mod = OptimizedModule::from_words(&words).expect("optimizer runs");

    assert!(
        result_mod.has_opcode(Op::ISub),
        "live ISub should be preserved"
    );
}
