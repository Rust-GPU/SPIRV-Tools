use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{default_fuzz_options, fuzz_module_with_options, validate_binary};

fn minimal_words() -> Vec<u32> {
    assemble_text(
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
OpEntryPoint Vertex %main \"main\"
",
    )
    .expect("assemble")
}

#[test]
fn fuzz_rejects_empty_input() {
    let opts = default_fuzz_options();
    let result = fuzz_module_with_options(&[], &opts);
    assert!(!result.success, "empty input should be rejected");
    assert!(
        result.message.contains("empty module"),
        "expected empty-module message, got: {}",
        result.message
    );
    assert!(result.words.is_empty(), "empty input should not yield output");
}

#[test]
fn fuzz_rejects_invalid_input() {
    // Missing memory model.
    let invalid = vec![0x07230203, 0, 0, 2, 0]; // header only
    let opts = default_fuzz_options();
    let result = fuzz_module_with_options(&invalid, &opts);
    assert!(!result.success, "invalid input should be rejected");
    assert!(
        !result.message.is_empty(),
        "expected validation error message, got empty string"
    );
}

#[test]
fn fuzz_validation_passthrough_keeps_input() {
    let words = minimal_words();
    let mut opts = default_fuzz_options();
    opts.enable_fuzzer_pass_validation = true;
    let result = fuzz_module_with_options(&words, &opts);
    assert!(result.success, "validation-only path should succeed");
    assert_eq!(result.words, words, "validation passthrough should not mutate");
    assert!(
        validate_binary(TargetEnv::Universal1_6, &result.words).success,
        "returned module should validate"
    );
}
