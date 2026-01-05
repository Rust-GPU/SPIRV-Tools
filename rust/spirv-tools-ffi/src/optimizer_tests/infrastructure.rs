//! Infrastructure tests for optimizer error handling, environment variables, and overrides.

use super::common::{OptimizerEnvGuard, TestModuleBuilder};
use crate::optimize_basic_block as optimize_wrapped_block;
use crate::optimizer::optimize_basic_block;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Loader};
use rspirv::spirv::{AddressingModel, FunctionControl, MemoryModel};

#[test]
fn reports_parse_error() {
    let _guard = OptimizerEnvGuard::new();
    let invalid_words = vec![0u32]; // not a valid module header
    let result = optimize_wrapped_block(&invalid_words);
    assert!(!result.success);
    assert!(matches!(
        result.error,
        crate::OptimizeError::Parse | crate::OptimizeError::Optimize
    ));
}

#[test]
fn reports_disabled_kind() {
    let _guard = OptimizerEnvGuard::new();
    std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let _ = b.builder.i_add(b.int_ty, None, c2, c3);
    let words = b.finish();

    let result = optimize_wrapped_block(&words);
    assert!(result.success, "disable should passthrough successfully");
    assert_eq!(result.error, crate::OptimizeError::Disabled);
    assert_eq!(result.words, words, "disable should leave module unchanged");
}

#[test]
fn respects_disable_env() {
    let _guard = OptimizerEnvGuard::new();
    std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let _ = b.builder.i_add(b.int_ty, None, c2, c3);
    let words = b.finish();

    let result = optimize_wrapped_block(&words);
    assert!(result.success);
    assert_eq!(result.words, words, "module should be unchanged when disabled");
}

#[test]
fn force_env_is_cleared_with_override_reset() {
    let _guard = OptimizerEnvGuard::new();
    std::env::set_var("SPIRV_TOOLS_FORCE_RUST_OPT", "1");
    crate::clear_rust_optimizer_override();
    // After clear, env var should be checked again
    std::env::remove_var("SPIRV_TOOLS_FORCE_RUST_OPT");
}

#[test]
fn disable_env_wins_over_force_env() {
    let _guard = OptimizerEnvGuard::new();
    std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
    std::env::set_var("SPIRV_TOOLS_FORCE_RUST_OPT", "1");

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let _ = b.builder.i_add(b.int_ty, None, c2, c3);
    let words = b.finish();

    let result = optimize_wrapped_block(&words);
    assert!(result.success);
    assert_eq!(
        result.words, words,
        "disable should win over force and leave module unchanged"
    );
}

#[test]
fn override_can_disable_even_without_env() {
    let _guard = OptimizerEnvGuard::new();
    crate::set_rust_optimizer_override(false);

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let _ = b.builder.i_add(b.int_ty, None, c2, c3);
    let words = b.finish();

    let result = optimize_wrapped_block(&words);
    assert!(result.success);
    assert_eq!(result.error, crate::OptimizeError::Disabled);
}

#[test]
fn override_can_enable_even_with_env_disable() {
    let _guard = OptimizerEnvGuard::new();
    std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
    crate::set_rust_optimizer_override(true);

    let mut b = TestModuleBuilder::new();
    b.begin_void_function();
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let _ = b.builder.i_add(b.int_ty, None, c2, c3);
    let words = b.finish();

    let result = optimize_wrapped_block(&words);
    assert!(result.success);
    // Should optimize because override enables it
    assert_ne!(result.error, crate::OptimizeError::Disabled);
}

#[test]
fn basic_block_pass_through_non_arith() {
    let _guard = OptimizerEnvGuard::new();

    let mut b = Builder::new();
    let void = b.type_void();
    let func_ty = b.type_function(void, vec![]);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let words = b.module().assemble();
    let optimized = optimize_basic_block(&words).expect("optimizer runs");
    let mut loader = Loader::new();
    rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
}
