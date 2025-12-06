use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::disassembly::disassemble_binary;
use spirv_tools_core::validation::validate_module;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{set_rust_text_assembler_override, try_assemble_text, validate_binary};

fn corpus_modules() -> Vec<String> {
    vec![
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
        .join("\n"),
        [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Fragment %main \"main\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n"),
    ]
}

#[test]
fn ffi_assemble_matches_core_for_corpus() {
    let handle = spirv_tools_ffi::create_context(
        TargetEnv::Universal1_6.to_raw(),
        std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr() as usize,
    );
    assert_ne!(handle, 0);
    set_rust_text_assembler_override(true);
    for (idx, text) in corpus_modules().iter().enumerate() {
        let core_binary = assemble_text(text).expect("assemble via core");
        // C++ assembler preserves layout skips that the Rust assembler rejects; ensure we match the
        // core text->binary path by sticking to valid modules.
        let core_dis = disassemble_binary(
            &core_binary,
            spirv_tools_core::assembly::BinaryToTextOptions::NONE,
        )
        .expect("disassemble core binary");
        assert!(
            core_dis.contains("OpCapability"),
            "core disassembly sanity check failed"
        );
        let ffi = try_assemble_text(
            handle,
            text.as_bytes(),
            spirv_tools_core::assembly::TextToBinaryOptions::NONE.bits(),
        );
        assert!(
            ffi.success,
            "ffi assemble should succeed for corpus {idx}: {ffi:?}"
        );
        assert_eq!(
            core_binary, ffi.binary,
            "ffi assemble binary mismatch on corpus {idx}"
        );
    }
    unsafe { spirv_tools_ffi::destroy_context(handle) };
    set_rust_text_assembler_override(false);
}

#[test]
fn ffi_validate_matches_core_for_corpus() {
    for (idx, text) in corpus_modules().iter().enumerate() {
        let binary = assemble_text(text).expect("assemble");
        let core = validate_module(&binary, TargetEnv::Universal1_6);
        let ffi = validate_binary(TargetEnv::Universal1_6, &binary);
        assert_eq!(
            core.is_ok(),
            ffi.success,
            "ffi validation success mismatch on corpus {idx}"
        );
    }
}
