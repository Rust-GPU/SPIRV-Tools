use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{
    default_reduce_options, reduce_module, reduce_module_with_cpp, validate_binary,
};

fn corpus() -> Vec<(&'static str, Vec<u32>)> {
    vec![
        (
            "vertex",
            assemble_text(
                "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
            )
            .expect("assemble vertex"),
        ),
        (
            "fragment",
            assemble_text(
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
            )
            .expect("assemble fragment"),
        ),
        (
            "compute",
            assemble_text(
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
            )
            .expect("assemble compute"),
        ),
        (
            "raygen_payload",
            assemble_text(
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
            )
            .expect("assemble raygen"),
        ),
        (
            "miss_payload",
            assemble_text(
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
            )
            .expect("assemble miss"),
        ),
        (
            "closest_hit_payload_attr",
            assemble_text(
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
            )
            .expect("assemble closest-hit"),
        ),
        (
            "callable_data",
            assemble_text(
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
            )
            .expect("assemble callable"),
        ),
    ]
}

#[test]
fn cpp_reduce_bridge_validates_or_skips() {
    let opts = default_reduce_options();
    let (_, words) = &corpus()[0];
    let cpp = reduce_module_with_cpp(words, &opts);
    if !cpp.success {
        eprintln!("C++ reduce bridge unavailable or disabled: {}", cpp.message);
        return;
    }
    assert!(
        validate_binary(TargetEnv::Universal1_6, &cpp.words).success,
        "C++ reduce bridge should emit a valid module"
    );
}

#[test]
fn rust_and_cpp_reduce_both_succeed_when_cpp_available() {
    let opts = default_reduce_options();
    let (_, words) = &corpus()[0];
    let cpp = reduce_module_with_cpp(words, &opts);
    if !cpp.success {
        eprintln!("C++ reduce bridge unavailable or disabled: {}", cpp.message);
        return;
    }

    let rust = reduce_module(words);
    assert!(rust.success, "Rust reduce path should succeed");
    assert!(cpp.success, "C++ reduce bridge should succeed");
    assert_eq!(rust.words, cpp.words, "Rust vs C++ reduce mismatch");
}

#[test]
fn rust_and_cpp_reduce_fail_on_invalid_when_cpp_available() {
    let opts = default_reduce_options();
    let invalid = vec![0x07230203, 0, 0, 0, 0];
    let cpp = reduce_module_with_cpp(&invalid, &opts);
    if !cpp.success {
        eprintln!(
            "C++ reduce bridge unavailable or disabled for invalid parity: {}",
            cpp.message
        );
        return;
    }
    let rust = reduce_module(&invalid);
    assert!(
        !rust.success,
        "Rust reduce should reject invalid input: {}",
        rust.message
    );
    assert!(
        !cpp.success,
        "C++ reduce should reject invalid input: {}",
        cpp.message
    );
}

#[test]
fn cpp_and_rust_reduce_match_on_corpus_when_cpp_available() {
    let opts = default_reduce_options();
    for (idx, (label, words)) in corpus().iter().enumerate() {
        let cpp = reduce_module_with_cpp(words, &opts);
        if !cpp.success {
            eprintln!(
                "C++ reduce bridge unavailable or disabled on corpus {idx} ({label}): {}",
                cpp.message
            );
            return;
        }
        let rust = reduce_module(words);
        assert!(rust.success, "Rust reduce failed on corpus {idx} ({label})");
        assert!(cpp.success, "C++ reduce failed on corpus {idx} ({label})");
        assert_eq!(
            rust.words, cpp.words,
            "Rust and C++ reduce outputs diverged on corpus {idx} ({label})"
        );
    }
}
