use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{default_fuzz_options, fuzz_module_with_options, validate_binary};

#[test]
fn rust_fuzzer_can_emit_intentionally_invalid() {
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

    let mut opts = default_fuzz_options();
    opts.enable_fuzzer_pass_validation = false; // allow invalid mutations

    let result = fuzz_module_with_options(&binary, &opts);
    assert!(result.success);
    if validate_binary(TargetEnv::Universal1_6, &result.words).success {
        // Valid path is fine too.
        return;
    }
    assert!(
        result.message.contains("intentionally invalid"),
        "expected intentionally invalid marker"
    );
}
