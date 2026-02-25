use super::super::*;

#[test]
fn cooperative_matrix_nv_capability_rejected_outside_vulkan() {
    let text = [
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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("CooperativeMatrixNV should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_cooperative_matrix"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("CooperativeMatrixNV should be accepted for Vulkan targets");
}

#[test]
fn tile_shading_capability_rejected_outside_vulkan() {
    let text = [
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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("TileShadingQCOM should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_QCOM_tile_shading"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_4)
        .expect("TileShadingQCOM should be accepted for Vulkan targets");
}

#[test]
fn ray_tracing_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("RayTracingKHR should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_ray_tracing"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingKHR should be accepted for Vulkan targets");
}

#[test]
fn mesh_shading_nv_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability MeshShadingNV",
        "OpExtension \"SPV_NV_mesh_shader\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("MeshShadingNV should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_mesh_shader"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("MeshShadingNV should be accepted for Vulkan targets");
}

#[test]
fn mesh_shading_ext_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability MeshShadingEXT",
        "OpExtension \"SPV_EXT_mesh_shader\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("MeshShadingEXT should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_EXT_mesh_shader"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("MeshShadingEXT should be accepted for Vulkan targets");
}

#[test]
fn cooperative_matrix_khr_capability_rejected_outside_vulkan_even_with_extension() {
    // CooperativeMatrixKHR + Shader requires VulkanMemoryModel capability
    let text = [
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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("CooperativeMatrixKHR should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_cooperative_matrix"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("CooperativeMatrixKHR should be accepted for Vulkan targets");
}

#[test]
fn ray_tracing_motion_blur_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text.as_str().validate(env).expect_err(
            "RayTracingMotionBlurNV should be rejected when its extension is disallowed",
        );
        match error {
            ValidationError::DisallowedExtension {
                extension,
                env: actual_env,
            } => {
                assert_eq!(actual_env, env);
                assert!(
                    extension == ExtensionName::from("SPV_NV_ray_tracing_motion_blur")
                        || extension == ExtensionName::from("SPV_NV_ray_tracing"),
                    "unexpected extension blocked: {extension:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingMotionBlurNV should be accepted for Vulkan targets");
}

#[test]
fn ray_tracing_displacement_micromap_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text.as_str().validate(env).expect_err(
            "RayTracingDisplacementMicromapNV should be rejected when its extension is disallowed",
        );
        match error {
            ValidationError::DisallowedExtension {
                extension,
                env: actual_env,
            } => {
                assert_eq!(actual_env, env);
                assert!(
                    extension == ExtensionName::from("SPV_NV_displacement_micromap")
                        || extension == ExtensionName::from("SPV_NV_ray_tracing")
                        || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                    "unexpected extension blocked: {extension:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingDisplacementMicromapNV should be accepted for Vulkan targets");
}

#[test]
fn ray_tracing_linear_swept_spheres_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpCapability RayTracingLinearSweptSpheresGeometryNV",
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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text.as_str().validate(env).expect_err(
                "RayTracingLinearSweptSpheresGeometryNV should be rejected when its extension is disallowed",
            );
        match error {
            ValidationError::DisallowedExtension {
                extension,
                env: actual_env,
            } => {
                assert_eq!(actual_env, env);
                assert!(
                    extension == ExtensionName::from("SPV_NV_linear_swept_spheres")
                        || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                    "unexpected extension blocked: {extension:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingLinearSweptSpheresGeometryNV should be accepted for Vulkan targets");
}

#[test]
fn ray_tracing_opacity_micromap_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text.as_str().validate(env).expect_err(
            "RayTracingOpacityMicromapEXT should be rejected when its extension is disallowed",
        );
        match error {
            ValidationError::DisallowedExtension {
                extension,
                env: actual_env,
            } => {
                assert_eq!(actual_env, env);
                assert!(
                    extension == ExtensionName::from("SPV_EXT_opacity_micromap")
                        || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                    "unexpected extension blocked: {extension:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingOpacityMicromapEXT should be accepted for Vulkan targets");
}

#[test]
fn shader_invocation_reorder_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpCapability ShaderInvocationReorderNV",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpExtension \"SPV_NV_shader_invocation_reorder\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text.as_str().validate(env).expect_err(
            "ShaderInvocationReorderNV should be rejected when its extension is disallowed",
        );
        match error {
            ValidationError::DisallowedExtension {
                extension,
                env: actual_env,
            } => {
                assert_eq!(actual_env, env);
                assert!(
                    extension == ExtensionName::from("SPV_NV_shader_invocation_reorder")
                        || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                    "unexpected extension blocked: {extension:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderInvocationReorderNV should be accepted for Vulkan targets");
}

#[test]
fn cluster_acceleration_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpCapability RayTracingClusterAccelerationStructureNV",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpExtension \"SPV_NV_cluster_acceleration_structure\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text.as_str().validate(env).expect_err(
                "RayTracingClusterAccelerationStructureNV should be rejected when its extension is disallowed",
            );
        match error {
            ValidationError::DisallowedExtension {
                extension,
                env: actual_env,
            } => {
                assert_eq!(actual_env, env);
                assert!(
                    extension == ExtensionName::from("SPV_NV_cluster_acceleration_structure")
                        || extension == ExtensionName::from("SPV_KHR_ray_tracing"),
                    "unexpected extension blocked: {extension:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingClusterAccelerationStructureNV should be accepted for Vulkan targets");
}

#[test]
fn shader_sm_builtins_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability ShaderSMBuiltinsNV",
        "OpExtension \"SPV_NV_shader_sm_builtins\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("ShaderSMBuiltinsNV should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_shader_sm_builtins"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderSMBuiltinsNV should be accepted for Vulkan targets");
}

#[test]
fn fragment_shader_interlock_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentShaderPixelInterlockEXT",
        "OpExtension \"SPV_EXT_fragment_shader_interlock\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text.as_str().validate(env).expect_err(
            "FragmentShaderPixelInterlockEXT should be rejected when its extension is disallowed",
        );
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_EXT_fragment_shader_interlock"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentShaderPixelInterlockEXT should be accepted for Vulkan targets");
}

#[test]
fn image_footprint_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability ImageFootprintNV",
        "OpExtension \"SPV_NV_shader_image_footprint\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("ImageFootprintNV should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_shader_image_footprint"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ImageFootprintNV should be accepted for Vulkan targets");
}

#[test]
fn shader_atomic_float_add_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability AtomicFloat32AddEXT",
        "OpExtension \"SPV_EXT_shader_atomic_float_add\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("AtomicFloat32AddEXT should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_EXT_shader_atomic_float_add"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("AtomicFloat32AddEXT should be accepted for Vulkan targets");
}

#[test]
fn fragment_shading_rate_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentShadingRateKHR",
        "OpExtension \"SPV_KHR_fragment_shading_rate\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text.as_str().validate(env).expect_err(
            "FragmentShadingRateKHR should be rejected when its extension is disallowed",
        );
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_fragment_shading_rate"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentShadingRateKHR should be accepted for Vulkan targets");
}

#[test]
fn fragment_invocation_density_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentDensityEXT",
        "OpExtension \"SPV_EXT_fragment_invocation_density\"",
        "OpExtension \"SPV_NV_shading_rate\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("FragmentDensityEXT should be rejected when its extension is disallowed");
        match error {
            ValidationError::DisallowedExtension {
                extension,
                env: actual_env,
            } => {
                assert_eq!(actual_env, env);
                assert!(
                    extension == ExtensionName::from("SPV_EXT_fragment_invocation_density")
                        || extension == ExtensionName::from("SPV_NV_shading_rate"),
                    "unexpected extension in disallowance: {extension:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentDensityEXT should be accepted for Vulkan targets");
}

#[test]
fn maximal_reconvergence_extension_rejected_outside_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_maximal_reconvergence\"",
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
        .validate(TargetEnv::OpenCl2_2)
        .expect_err("maximal reconvergence is Vulkan-only");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_maximal_reconvergence"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn ray_cull_mask_extension_rejected_outside_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_ray_cull_mask\"",
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
        .validate(TargetEnv::OpenCl2_2)
        .expect_err("ray cull mask is Vulkan-only");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_ray_cull_mask"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn shader_clock_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability ShaderClockKHR",
        "OpExtension \"SPV_KHR_shader_clock\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("ShaderClockKHR should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_shader_clock"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderClockKHR should be accepted for Vulkan targets");
}

#[test]
fn device_group_capability_rejected_outside_vulkan_even_with_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability DeviceGroup",
        "OpExtension \"SPV_KHR_device_group\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("DeviceGroup should be rejected when its extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_KHR_device_group"),
                env
            }
        );
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("DeviceGroup should be accepted for Vulkan targets");
}
