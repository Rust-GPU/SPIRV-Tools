use super::super::*;

#[test]
fn cooperative_matrix_capability_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability CooperativeMatrixNV",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("CooperativeMatrixNV requires its enabling extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::CooperativeMatrixNV,
            required_extension: "SPV_NV_cooperative_matrix".to_string()
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability CooperativeMatrixNV",
        "OpExtension \"SPV_NV_cooperative_matrix\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let validated = with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("capability should be accepted with required extension");
    assert_eq!(validated.header().schema(), Schema::ZERO);
}

#[test]
fn cooperative_matrix_khr_capability_requires_extension() {
    // Test without the extension - should fail (also needs VulkanMemoryModel)
    let text = [
        "OpCapability Shader",
        "OpCapability CooperativeMatrixKHR",
        "OpCapability VulkanMemoryModel",
        "OpExtension \"SPV_KHR_vulkan_memory_model\"",
        "OpMemoryModel Logical VulkanKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("CooperativeMatrixKHR requires its enabling extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::CooperativeMatrixKHR,
            required_extension: "SPV_KHR_cooperative_matrix".to_string()
        }
    );
    // With extension - should pass
    let with_extension = [
        "OpCapability Shader",
        "OpCapability CooperativeMatrixKHR",
        "OpCapability VulkanMemoryModel",
        "OpExtension \"SPV_KHR_cooperative_matrix\"",
        "OpExtension \"SPV_KHR_vulkan_memory_model\"",
        "OpMemoryModel Logical VulkanKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let validated = with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("capability should be accepted with required extension");
    assert_eq!(validated.header().schema(), Schema::ZERO);
}

#[test]
fn ray_tracing_motion_blur_capability_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingNV",
        "OpCapability RayTracingMotionBlurNV",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("RayTracingMotionBlurNV requires its enabling extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::RayTracingMotionBlurNV,
            required_extension: "SPV_NV_ray_tracing_motion_blur".to_string()
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability RayTracingNV",
        "OpCapability RayTracingMotionBlurNV",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpExtension \"SPV_NV_ray_tracing_motion_blur\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("capability should be accepted with required extensions");
}

#[test]
fn ray_tracing_displacement_micromap_capability_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpCapability RayTracingNV",
        "OpCapability RayTracingDisplacementMicromapNV",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("RayTracingDisplacementMicromapNV requires its enabling extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::RayTracingDisplacementMicromapNV,
            required_extension: "SPV_NV_displacement_micromap".to_string()
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpCapability RayTracingNV",
        "OpCapability RayTracingDisplacementMicromapNV",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpExtension \"SPV_NV_displacement_micromap\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("capability should be accepted with required extensions");
}

#[test]
fn ray_tracing_linear_swept_spheres_capability_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingLinearSweptSpheresGeometryNV",
        "OpCapability RayTracingKHR",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("RayTracingLinearSweptSpheresGeometryNV requires its enabling extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::RayTracingLinearSweptSpheresGeometryNV,
            required_extension: "SPV_NV_linear_swept_spheres".to_string()
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability RayTracingLinearSweptSpheresGeometryNV",
        "OpCapability RayTracingKHR",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpExtension \"SPV_NV_linear_swept_spheres\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("capability should be accepted with required extensions");
}

#[test]
fn ray_tracing_opacity_micromap_capability_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpCapability RayTracingOpacityMicromapEXT",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("RayTracingOpacityMicromapEXT requires its enabling extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::RayTracingOpacityMicromapEXT,
            required_extension: "SPV_EXT_opacity_micromap".to_string()
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpCapability RayTracingOpacityMicromapEXT",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpExtension \"SPV_EXT_opacity_micromap\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("capability should be accepted with required extensions");
}

#[test]
fn tile_shading_capability_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability TileShadingQCOM",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_3)
        .expect_err("TileShadingQCOM requires its enabling extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::TileShadingQCOM,
            required_extension: "SPV_QCOM_tile_shading".to_string()
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability TileShadingQCOM",
        "OpExtension \"SPV_QCOM_tile_shading\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let validated = with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_3)
        .expect("capability should be accepted with required extension");
    assert_eq!(validated.header().schema(), Schema::ZERO);
}

#[test]
fn arm_core_builtins_capability_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability CoreBuiltinsARM",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("CoreBuiltinsARM requires declaring the extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::CoreBuiltinsARM,
            required_extension: "SPV_ARM_core_builtins".to_string(),
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability CoreBuiltinsARM",
        "OpExtension \"SPV_ARM_core_builtins\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("extension declared should satisfy CoreBuiltinsARM capability");
}
