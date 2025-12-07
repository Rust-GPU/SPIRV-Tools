use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{default_fuzz_options, fuzz_module_with_options, validate_binary};

#[test]
fn cpp_fuzz_bridge_runs_with_seed() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble module");

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

    // The module should remain valid after the fuzz run.
    assert!(validate_binary(TargetEnv::Universal1_6, &result.words).success);
}
