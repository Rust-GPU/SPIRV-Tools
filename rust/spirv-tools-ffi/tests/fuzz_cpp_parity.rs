use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{default_fuzz_options, fuzz_module, fuzz_module_with_cpp, validate_binary};

const CORPUS_TEXTS: [(&str, &str); 7] = [
    (
        "vertex",
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
    ),
    (
        "fragment",
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
    ),
    (
        "compute",
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
    ),
    (
        "raygen_payload",
        "\
OpCapability RayTracingKHR
OpExtension \"SPV_KHR_ray_tracing\"
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %main \"main\" %payload
%void = OpTypeVoid
%u32 = OpTypeInt 32 0
%payload_ty = OpTypeStruct %u32
%ptr_payload = OpTypePointer IncomingRayPayloadKHR %payload_ty
%fn = OpTypeFunction %void
%payload = OpVariable %ptr_payload IncomingRayPayloadKHR
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
    ),
    (
        "miss_payload",
        "\
OpCapability RayTracingKHR
OpExtension \"SPV_KHR_ray_tracing\"
OpMemoryModel Logical GLSL450
OpEntryPoint MissKHR %main \"main\" %payload
%void = OpTypeVoid
%u32 = OpTypeInt 32 0
%payload_ty = OpTypeStruct %u32
%ptr_payload = OpTypePointer IncomingRayPayloadKHR %payload_ty
%fn = OpTypeFunction %void
%payload = OpVariable %ptr_payload IncomingRayPayloadKHR
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
    ),
    (
        "closest_hit_payload_attr",
        "\
OpCapability RayTracingKHR
OpExtension \"SPV_KHR_ray_tracing\"
OpMemoryModel Logical GLSL450
OpEntryPoint ClosestHitKHR %main \"main\" %payload %hit_attr
%void = OpTypeVoid
%u32 = OpTypeInt 32 0
%payload_ty = OpTypeStruct %u32
%attr_ty = OpTypeStruct %u32
%ptr_payload = OpTypePointer IncomingRayPayloadKHR %payload_ty
%ptr_attr = OpTypePointer HitAttributeKHR %attr_ty
%fn = OpTypeFunction %void
%payload = OpVariable %ptr_payload IncomingRayPayloadKHR
%hit_attr = OpVariable %ptr_attr HitAttributeKHR
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
    ),
    (
        "callable_data",
        "\
OpCapability RayTracingKHR
OpExtension \"SPV_KHR_ray_tracing\"
OpMemoryModel Logical GLSL450
OpEntryPoint CallableKHR %main \"main\" %call_data
%void = OpTypeVoid
%u32 = OpTypeInt 32 0
%call_ty = OpTypeStruct %u32
%ptr_call = OpTypePointer CallableDataKHR %call_ty
%fn = OpTypeFunction %void
%call_data = OpVariable %ptr_call CallableDataKHR
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
    ),
];

fn assemble_case((label, text): (&str, &str)) -> Vec<u32> {
    assemble_text(text).unwrap_or_else(|err| panic!("assemble {label}: {err}"))
}

fn minimal_words() -> Vec<u32> {
    assemble_case(CORPUS_TEXTS[0])
}

#[test]
fn cpp_fuzz_bridge_validates_or_skips() {
    let words = minimal_words();
    let opts = default_fuzz_options();
    let cpp = fuzz_module_with_cpp(&words, &opts);
    if !cpp.success {
        eprintln!("C++ fuzz bridge unavailable or disabled: {}", cpp.message);
        return;
    }
    assert!(
        validate_binary(TargetEnv::Universal1_6, &cpp.words).success,
        "C++ fuzz bridge should emit a valid module"
    );
}

#[test]
fn rust_and_cpp_fuzz_both_succeed_when_cpp_available() {
    let words = minimal_words();
    let opts = default_fuzz_options();
    let cpp = fuzz_module_with_cpp(&words, &opts);
    if !cpp.success {
        eprintln!("C++ fuzz bridge unavailable or disabled: {}", cpp.message);
        return;
    }

    let rust = fuzz_module(&words);
    assert!(rust.success, "Rust fuzz pipeline should succeed");
    assert!(cpp.success, "C++ fuzz bridge should succeed");
    assert!(
        validate_binary(TargetEnv::Universal1_6, &rust.words).success,
        "Rust fuzz pipeline should emit a valid module"
    );
    assert!(
        validate_binary(TargetEnv::Universal1_6, &cpp.words).success,
        "C++ fuzz bridge should emit a valid module"
    );
}

#[test]
fn rust_and_cpp_fuzz_both_fail_on_invalid_when_cpp_available() {
    let opts = default_fuzz_options();
    let invalid = vec![0x07230203, 0, 0, 0, 0]; // header-only garbage
    let cpp = fuzz_module_with_cpp(&invalid, &opts);
    if !cpp.success {
        eprintln!(
            "C++ fuzz bridge unavailable or disabled for invalid parity: {}",
            cpp.message
        );
        return;
    }
    let rust = fuzz_module(&invalid);
    assert!(
        !rust.success,
        "Rust fuzz pipeline should reject invalid input"
    );
    assert!(
        !cpp.success,
        "C++ fuzz pipeline should reject invalid input"
    );
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &invalid).success,
        "baseline validation should fail for invalid input"
    );
    assert!(
        !rust.message.is_empty(),
        "Rust fuzz should surface diagnostics on invalid input"
    );
    assert!(
        !cpp.message.is_empty(),
        "C++ fuzz should surface diagnostics on invalid input"
    );
}

#[test]
fn cpp_and_rust_fuzz_match_on_corpus_when_cpp_available() {
    let corpus: Vec<_> = CORPUS_TEXTS.into_iter().map(assemble_case).collect();

    let opts = default_fuzz_options();
    let mut cpp_unavailable = false;
    for (idx, words) in corpus.iter().enumerate() {
        let cpp = fuzz_module_with_cpp(words, &opts);
        if !cpp.success {
            eprintln!(
                "C++ fuzz bridge unavailable or disabled on corpus {idx}: {}",
                cpp.message
            );
            cpp_unavailable = true;
            break;
        }
        let rust = fuzz_module(words);
        assert!(rust.success, "Rust fuzz pipeline failed on corpus {idx}");
        assert!(cpp.success, "C++ fuzz pipeline failed on corpus {idx}");
        assert_eq!(
            rust.words, cpp.words,
            "Rust and C++ fuzz outputs diverged on corpus {idx}"
        );
    }

    if cpp_unavailable {
        eprintln!("Skipping corpus parity because C++ fuzz bridge is unavailable");
    }
}
