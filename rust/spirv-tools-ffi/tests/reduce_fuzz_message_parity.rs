use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{set_rust_text_assembler_override, validate_binary};

fn invalid_module() -> Vec<u32> {
    assemble_text("OpCapability Shader").expect("assemble invalid module")
}

#[test]
fn ffi_validation_returns_message_on_error() {
    let binary = invalid_module();
    let result = validate_binary(TargetEnv::Universal1_6, &binary);
    assert!(
        !result.success,
        "invalid module should fail validation via FFI"
    );
    assert!(
        !result.message.is_empty(),
        "validation failure should include message text"
    );
}

#[test]
fn ffi_validation_respects_rust_assembler_override_for_reducer_inputs() {
    // Force the Rust assembler so downstream reducer/fuzzer callers see Rust-assembled binaries.
    set_rust_text_assembler_override(true);
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
    let binary = assemble_text(&text).expect("assemble");
    let result = validate_binary(TargetEnv::Universal1_6, &binary);
    assert!(result.success, "expected valid module");
    assert!(result.message.is_empty());
    set_rust_text_assembler_override(false);
}
