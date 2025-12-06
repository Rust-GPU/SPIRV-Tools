use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::disassembly::disassemble_binary;
use spirv_tools_ffi::{
    set_rust_text_assembler_override, try_assemble_text, try_disassemble_binary,
};
use std::ptr;

fn simple_module_text() -> String {
    [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n")
}

#[test]
fn ffi_assembler_matches_core_output() {
    set_rust_text_assembler_override(true);
    let text = simple_module_text();
    let binary = assemble_text(&text).expect("assemble via core");

    // Pass a dummy non-null context pointer; we don't use the consumer in the Rust path.
    let handle = spirv_tools_ffi::create_context(
        spirv_tools_core::TargetEnv::Universal1_6.to_raw(),
        ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr() as usize,
    );
    assert_ne!(handle, 0);
    let result = try_assemble_text(
        handle,
        text.as_bytes(),
        spirv_tools_core::assembly::TextToBinaryOptions::NONE.bits(),
    );
    assert!(result.success);
    assert_eq!(result.binary, binary);
    unsafe { spirv_tools_ffi::destroy_context(handle) };
    set_rust_text_assembler_override(false);
}

#[test]
fn ffi_disassembler_matches_core_output() {
    let text = simple_module_text();
    let binary = assemble_text(&text).expect("assemble via core");
    let handle = spirv_tools_ffi::create_context(
        spirv_tools_core::TargetEnv::Universal1_6.to_raw(),
        std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr() as usize,
    );
    assert_ne!(handle, 0);

    let result = try_disassemble_binary(
        handle,
        &binary,
        spirv_tools_core::assembly::BinaryToTextOptions::NONE.bits(),
    );
    assert!(result.success, "ffi disassemble should succeed");
    let direct = disassemble_binary(
        &binary,
        spirv_tools_core::assembly::BinaryToTextOptions::NONE,
    )
    .expect("core disassemble");
    assert_eq!(result.text.trim(), direct.trim());
    assert!(result.diagnostics.is_empty());

    unsafe { spirv_tools_ffi::destroy_context(handle) };
}

#[test]
fn ffi_disassembler_reports_parse_errors_like_core() {
    let handle = spirv_tools_ffi::create_context(
        spirv_tools_core::TargetEnv::Universal1_0.to_raw(),
        std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr() as usize,
    );
    assert_ne!(handle, 0);
    let invalid = vec![0x0723_0203u32];

    let result = try_disassemble_binary(
        handle,
        &invalid,
        spirv_tools_core::assembly::BinaryToTextOptions::NONE.bits(),
    );
    assert!(!result.success);
    assert!(
        !result.diagnostics.is_empty(),
        "expected diagnostics from FFI disassembler"
    );

    let core = disassemble_binary(
        &invalid,
        spirv_tools_core::assembly::BinaryToTextOptions::NONE,
    );
    assert!(core.is_err(), "core disassembler should also fail");

    unsafe { spirv_tools_ffi::destroy_context(handle) };
}
