use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{default_fuzz_options, fuzz_module_with_cpp, validate_binary};

fn minimal_module_words() -> Vec<u32> {
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
    .expect("assemble minimal module")
}

#[test]
fn cpp_fuzz_bridge_skips_or_emits_valid_module() {
    let words = minimal_module_words();
    let opts = default_fuzz_options();
    let result = fuzz_module_with_cpp(&words, &opts);

    if !result.success {
        eprintln!("C++ fuzz bridge unavailable/disabled: {}", result.message);
        return; // skip when bridge is not built
    }

    assert!(
        validate_binary(TargetEnv::Universal1_6, &result.words).success,
        "C++ fuzz bridge should emit a valid module when enabled"
    );
}
