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
