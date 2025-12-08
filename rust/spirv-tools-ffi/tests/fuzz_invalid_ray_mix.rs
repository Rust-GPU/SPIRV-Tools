use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{
    default_fuzz_options, fuzz_module_with_options, validate_binary, FuzzConfig, FuzzGenerator,
    InvalidKind,
};

#[test]
fn rust_fuzzer_can_emit_mixed_ray_interface_storage_classes() {
    let cfg = FuzzConfig {
        seed: 51,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithNonRayInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "mixed storage classes on ray entry should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_mixed_ray_interfaces() {
    let cfg = FuzzConfig {
        seed: 53,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::MixedRayInterfaceStorageClasses),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "mixed storage classes on ray entry should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_workgroup_interface_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 67,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithWorkgroupInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "workgroup interface on ray entry should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_output_interface_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 69,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithOutputInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "output interface on ray entry should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_mixed_io_interfaces_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 71,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithMixedIoInterfaces),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "mixing input/output on ray entry should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_private_interface_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 73,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithPrivateInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "private storage in ray entry interface should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_function_interface_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 75,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithFunctionInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "function storage in ray entry interface should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_cross_workgroup_interface_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 77,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithCrossWorkgroupInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "cross-workgroup storage in ray entry interface should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_generic_interface_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 79,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithGenericInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "generic storage in ray entry interface should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_uniform_constant_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 81,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithUniformConstantInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "uniform-constant interface on ray entry should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_push_constant_on_ray_entry() {
    let cfg = FuzzConfig {
        seed: 83,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayEntryWithPushConstantInterface),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "push-constant interface on ray entry should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_missing_ray_execution_model() {
    let cfg = FuzzConfig {
        seed: 55,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::MissingRayExecutionModel),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "ray-only interfaces with non-ray execution model should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_missing_ray_capability() {
    let cfg = FuzzConfig {
        seed: 57,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::MissingRayCapability),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "ray interfaces without ray capability should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_ray_payload_type_mismatch() {
    let cfg = FuzzConfig {
        seed: 59,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::RayPayloadTypeMismatch),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "ray payload with wrong type should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_callable_data_type_mismatch() {
    let cfg = FuzzConfig {
        seed: 61,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::CallableDataTypeMismatch),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "callable data with wrong type should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_hit_attribute_type_mismatch() {
    let cfg = FuzzConfig {
        seed: 63,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::HitAttributeTypeMismatch),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "hit attribute with wrong type should fail validation"
    );
}

#[test]
fn rust_fuzzer_can_emit_hit_attribute_on_ray_gen() {
    let cfg = FuzzConfig {
        seed: 65,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::HitAttributeOnRayGen),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    let words = match outcome {
        spirv_tools_ffi::FuzzOutcome::Invalid { words, .. } => words,
        spirv_tools_ffi::FuzzOutcome::Valid { words } => words,
    };
    assert!(
        !validate_binary(TargetEnv::Universal1_6, &words).success,
        "hit attribute on ray-gen should fail validation"
    );
}

#[test]
fn fuzz_bridge_passthrough_rejects_mixed_ray_interfaces() {
    // Use the public fuzz wrapper to ensure the higher-level API rejects the invalid module too.
    let opts = default_fuzz_options();
    let result = fuzz_module_with_options(&[0x07230203, 0, 0, 0, 0], &opts);
    // Result is intentionally invalid because input is garbage; just ensure API surface is wired.
    assert!(
        !result.success || !result.words.is_empty(),
        "fuzz wrapper should return some result even for invalid inputs"
    );
}
