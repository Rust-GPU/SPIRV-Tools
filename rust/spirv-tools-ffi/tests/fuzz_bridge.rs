use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv::{Capability, ExecutionModel, FunctionControl, MemoryModel};
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{
    default_fuzz_options, fuzz_module_with_cpp, fuzz_module_with_options, validate_binary,
};

fn build_minimal_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, vec![]);
    let main = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .expect("begin function");
    b.begin_block(None).expect("block");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.entry_point(ExecutionModel::Vertex, main, "main", []);
    b.module().assemble()
}

#[test]
fn rust_fuzzer_runs_with_seed() {
    let binary = build_minimal_module();

    let mut options = default_fuzz_options();
    options.random_seed = 1;
    options.replay_range = 1;
    options.enable_fuzzer_pass_validation = true;

    let result = fuzz_module_with_options(&binary, &options);
    assert!(result.success, "fuzzing should succeed: {}", result.message);
    assert!(
        !result.words.is_empty(),
        "fuzzing should yield a non-empty module"
    );
    assert!(validate_binary(TargetEnv::Universal1_6, &result.words).success);
}

#[test]
fn fuzz_seed_changes_target_block() {
    let binary = build_minimal_module();

    let mut opts1 = default_fuzz_options();
    opts1.random_seed = 1;
    let mut opts2 = default_fuzz_options();
    opts2.random_seed = 2;

    let a = fuzz_module_with_options(&binary, &opts1);
    let b = fuzz_module_with_options(&binary, &opts2);
    assert!(a.success && b.success);
    assert_ne!(
        a.words, b.words,
        "different seeds should produce different layouts"
    );
}

#[test]
fn cpp_fuzz_bridge_reports_disabled() {
    let binary = build_minimal_module();
    let opts = default_fuzz_options();
    let result = fuzz_module_with_cpp(&binary, &opts);
    assert!(
        !result.success,
        "C++ fuzz bridge is expected to be disabled in Rust-first builds"
    );
    assert!(
        result.message.contains("unavailable") || result.message.contains("disabled"),
        "disabled bridge should surface a user-facing reason"
    );
}
