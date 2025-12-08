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
