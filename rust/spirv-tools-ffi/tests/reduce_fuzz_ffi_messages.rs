use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{set_rust_text_assembler_override, validate_binary};

fn invalid_module() -> Vec<u32> {
    assemble_text("OpCapability Shader").expect("assemble invalid module")
}

fn valid_module() -> Vec<u32> {
    assemble_text(
        "OpCapability Shader\n\
         OpMemoryModel Logical GLSL450\n\
         %int = OpTypeInt 32 1\n\
         %void = OpTypeVoid\n\
         %fn = OpTypeFunction %void\n\
         %main = OpFunction %void None %fn\n\
         %entry = OpLabel\n\
         %c = OpConstant %int 1\n\
         OpReturn\n\
         OpFunctionEnd\n",
    )
    .expect("assemble valid module")
}

#[test]
fn reducer_validator_surfaces_messages_on_error() {
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
fn reducer_validator_succeeds_on_valid_input() {
    set_rust_text_assembler_override(true);
    let binary = valid_module();
    let result = validate_binary(TargetEnv::Universal1_6, &binary);
    assert!(result.success, "expected valid module");
    assert!(result.message.is_empty());
    set_rust_text_assembler_override(false);
}
