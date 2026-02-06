use super::*;

// ============================================================================
// Int64ImageEXT capability tests
// ============================================================================

#[test]
fn image_type_64bit_int_sampled_type_requires_int64image_capability() {
    // The grammar-level capability validation catches this before our check:
    // R64ui format requires Int64ImageEXT capability at the operand level.
    let text = [
        "OpCapability Shader",
        "OpCapability Int64",
        "OpCapability StorageImageExtendedFormats",
        "OpMemoryModel Logical GLSL450",
        "%u64 = OpTypeInt 64 0",
        "%img = OpTypeImage %u64 2D 0 0 0 2 R64ui",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_3)
        .expect_err("64-bit int image without Int64ImageEXT should fail");
    // May fail with grammar-level MissingOperandCapability or our ImageTypeRequiresInt64ImageCapability
    let is_capability_error = matches!(
        err,
        ValidationError::ImageTypeRequiresInt64ImageCapability
            | ValidationError::MissingOperandCapability { .. }
    );
    assert!(
        is_capability_error,
        "Expected capability error, got: {err:?}"
    );
}

#[test]
fn image_type_64bit_int_sampled_type_passes_with_int64image_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int64",
        "OpCapability Int64ImageEXT",
        "OpCapability StorageImageExtendedFormats",
        "OpExtension \"SPV_EXT_shader_image_int64\"",
        "OpMemoryModel Logical GLSL450",
        "%u64 = OpTypeInt 64 0",
        "%img = OpTypeImage %u64 2D 0 0 0 2 R64ui",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_3)
        .expect("64-bit int image with Int64ImageEXT should pass");
}

#[test]
fn image_type_32bit_int_sampled_type_does_not_require_int64image() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%u32 = OpTypeInt 32 0",
        "%img = OpTypeImage %u32 2D 0 0 0 2 R32ui",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_3)
        .expect("32-bit int image should not require Int64ImageEXT");
}

// ============================================================================
// StorageImageMultisample capability tests
// ============================================================================

#[test]
fn image_type_multisampled_storage_requires_storage_image_multisample() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%f32 = OpTypeFloat 32",
        // MS=1, Sampled=2 (storage image) -> requires StorageImageMultisample
        "%img = OpTypeImage %f32 2D 0 0 1 2 Rgba32f",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_3)
        .expect_err("multisampled storage image without StorageImageMultisample should fail");
    assert!(
        matches!(
            err,
            ValidationError::ImageTypeRequiresStorageImageMultisampleCapability
        ),
        "Expected ImageTypeRequiresStorageImageMultisampleCapability, got: {err:?}"
    );
}

#[test]
fn image_type_multisampled_storage_passes_with_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability StorageImageMultisample",
        "OpMemoryModel Logical GLSL450",
        "%f32 = OpTypeFloat 32",
        "%img = OpTypeImage %f32 2D 0 0 1 2 Rgba32f",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_3)
        .expect("multisampled storage image with StorageImageMultisample should pass");
}

#[test]
fn image_type_multisampled_sampling_image_does_not_require_storage_multisample() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%f32 = OpTypeFloat 32",
        // MS=1, Sampled=1 (sampling image, not storage) -> no capability needed
        "%img = OpTypeImage %f32 2D 0 0 1 1 Unknown",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_3)
        .expect("multisampled sampling image should not require StorageImageMultisample");
}

#[test]
fn image_type_non_multisampled_storage_does_not_require_storage_multisample() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%f32 = OpTypeFloat 32",
        // MS=0, Sampled=2 (non-multisampled storage image) -> no capability needed
        "%img = OpTypeImage %f32 2D 0 0 0 2 Rgba32f",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_3)
        .expect("non-multisampled storage image should not require StorageImageMultisample");
}

// ============================================================================
// ImageQueryFormat / ImageQueryOrder tests
// ============================================================================

#[test]
fn image_query_format_valid_kernel() {
    // OpImageQueryFormat requires Kernel capability
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",

        "OpMemoryModel Physical64 OpenCL",
        "OpEntryPoint Kernel %main \"main\"",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 0",
        "%float = OpTypeFloat 32",
        "%fn = OpTypeFunction %void",
        "%img_ty = OpTypeImage %float 2D 0 0 0 0 Unknown",
        "%ptr_img = OpTypePointer CrossWorkgroup %img_ty",
        "%var = OpVariable %ptr_img CrossWorkgroup",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%img = OpLoad %img_ty %var",
        "%fmt = OpImageQueryFormat %int %img",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_3)
        .expect("valid OpImageQueryFormat in Kernel should pass");
}

#[test]
fn image_query_order_valid_kernel() {
    // OpImageQueryOrder requires Kernel capability
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",

        "OpMemoryModel Physical64 OpenCL",
        "OpEntryPoint Kernel %main \"main\"",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 0",
        "%float = OpTypeFloat 32",
        "%fn = OpTypeFunction %void",
        "%img_ty = OpTypeImage %float 2D 0 0 0 0 Unknown",
        "%ptr_img = OpTypePointer CrossWorkgroup %img_ty",
        "%var = OpVariable %ptr_img CrossWorkgroup",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%img = OpLoad %img_ty %var",
        "%ord = OpImageQueryOrder %int %img",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_3)
        .expect("valid OpImageQueryOrder in Kernel should pass");
}

#[test]
fn image_query_format_requires_int_scalar_result_kernel() {
    // OpImageQueryFormat result type must be int scalar
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",

        "OpMemoryModel Physical64 OpenCL",
        "OpEntryPoint Kernel %main \"main\"",
        "%void = OpTypeVoid",
        "%float = OpTypeFloat 32",
        "%fn = OpTypeFunction %void",
        "%img_ty = OpTypeImage %float 2D 0 0 0 0 Unknown",
        "%ptr_img = OpTypePointer CrossWorkgroup %img_ty",
        "%var = OpVariable %ptr_img CrossWorkgroup",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%img = OpLoad %img_ty %var",
        // Using float result type instead of int - should fail
        "%fmt = OpImageQueryFormat %float %img",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_3)
        .expect_err("OpImageQueryFormat with float result should fail");
    assert!(
        matches!(
            err,
            ValidationError::ImageQueryResultTypeInvalid { .. }
        ),
        "Expected ImageQueryResultTypeInvalid, got: {err:?}"
    );
}
