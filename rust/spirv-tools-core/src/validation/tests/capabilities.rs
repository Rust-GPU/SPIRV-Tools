use super::*;

#[test]
fn vulkan_rejects_kernel_capability() {
    let text = [
        "OpCapability Kernel",
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
        .expect_err("Vulkan should reject Kernel capability");
    assert_eq!(
        error,
        ValidationError::DisallowedCapability {
            capability: rspirv::spirv::Capability::Kernel,
            env: TargetEnv::Vulkan1_2
        }
    );
}

#[test]
fn vulkan_rejects_opencl_only_capabilities() {
    let text = [
        "OpCapability DeviceEnqueue",
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
        .expect_err("Vulkan should reject DeviceEnqueue capability");
    assert_eq!(
        error,
        ValidationError::DisallowedCapability {
            capability: rspirv::spirv::Capability::DeviceEnqueue,
            env: TargetEnv::Vulkan1_2
        }
    );
}

#[test]
fn vulkan_1_0_rejects_group_non_uniform() {
    let text = [
        "OpCapability Shader",
        "OpCapability GroupNonUniform",
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
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("GroupNonUniform is optional from Vulkan 1.1+");
    assert_eq!(
        error,
        ValidationError::DisallowedCapability {
            capability: rspirv::spirv::Capability::GroupNonUniform,
            env: TargetEnv::Vulkan1_0
        }
    );
}

#[test]
fn vulkan_1_1_allows_group_non_uniform() {
    let text = [
        "OpCapability Shader",
        "OpCapability GroupNonUniform",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let module = text
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("Vulkan 1.1 allows GroupNonUniform");
    assert_eq!(module.env(), TargetEnv::Vulkan1_1);
}

#[test]
fn vulkan_1_0_rejects_vulkan_memory_model() {
    let text = [
        "OpCapability Shader",
        "OpCapability VulkanMemoryModel",
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
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("Vulkan memory model requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::CapabilityRequiresSpirvVersion {
            capability: rspirv::spirv::Capability::VulkanMemoryModel,
            required_version: SpirvVersion::new(1, 5),
            target_version: TargetEnv::Vulkan1_0.spirv_version(),
        }
    );
}

#[test]
fn vulkan_1_1_allows_vulkan_memory_model_with_extension() {
    // VulkanMemoryModel is available in Vulkan 1.1 via SPV_KHR_vulkan_memory_model extension.
    // The capability requires SPIR-V 1.5, but the extension enables it on earlier versions.
    let text = [
        "OpCapability Shader",
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
    let module = text
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("VulkanMemoryModel is optional in Vulkan 1.1 (via SPV_KHR_vulkan_memory_model)");
    assert_eq!(module.env(), TargetEnv::Vulkan1_1);
}

#[test]
fn vulkan_1_2_allows_physical_storage_buffer_addresses() {
    let text = [
        "OpCapability Shader",
        "OpCapability PhysicalStorageBufferAddresses",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let module = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("PhysicalStorageBufferAddresses is optional in Vulkan 1.2");
    assert_eq!(module.env(), TargetEnv::Vulkan1_2);
}

#[test]
fn vulkan_1_2_allows_vulkankhr_memory_model_operand() {
    // VulkanKHR memory model operand requires SPV_KHR_vulkan_memory_model extension
    // but the extension is promoted to core in SPIR-V 1.5, and Vulkan 1.2 is SPIR-V 1.5.
    let text = [
        "OpCapability Shader",
        "OpCapability VulkanMemoryModel",
        "OpMemoryModel Logical VulkanKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let module = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("VulkanKHR memory model is allowed in Vulkan 1.2 (SPIR-V 1.5)");
    assert_eq!(module.env(), TargetEnv::Vulkan1_2);
}

#[test]
fn opencl_allows_optional_float64() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",
        "OpCapability Float64",
        "OpMemoryModel Physical32 OpenCL",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::OpenCl1_2)
        .expect("Float64 is optional in OpenCL 1.2");
}

#[test]
fn opencl_embedded_allows_optional_float64() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",
        "OpCapability Float64",
        "OpMemoryModel Physical32 OpenCL",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::OpenClEmbedded1_2)
        .expect("Float64 is optional in OpenCL 1.2 embedded");
}

#[test]
fn webgpu_rejects_non_shader_capabilities() {
    let text = [
        "OpCapability Kernel",
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
        .validate(TargetEnv::WebGpu0)
        .expect_err("WebGPU should reject Kernel capability");
    assert_eq!(
        error,
        ValidationError::DisallowedCapability {
            capability: rspirv::spirv::Capability::Kernel,
            env: TargetEnv::WebGpu0
        }
    );
}

#[test]
fn opencl_rejects_shader_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical OpenCL",
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
        .expect_err("OpenCL should reject Shader capability");
    assert_eq!(
        error,
        ValidationError::DisallowedCapability {
            capability: rspirv::spirv::Capability::Shader,
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn opencl_rejects_vulkan_specific_extension() {
    let text = [
        "OpCapability Kernel",
        "OpExtension \"SPV_KHR_vulkan_memory_model\"",
        "OpMemoryModel Logical OpenCL",
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
        .expect_err("OpenCL should reject Vulkan-specific extensions");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn opencl_rejects_nv_vendor_extension() {
    // OpenCL should reject NV vendor extensions.
    let ext_words = [
        1599492179, 1834964558, 1600680805, 1684105331, 29285, // "SPV_NV_mesh_shader\0"
    ];
    let binary = [
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        2,           // bound
        0,           // schema
        0x0006_000a, // OpExtension, word count 6
        ext_words[0],
        ext_words[1],
        ext_words[2],
        ext_words[3],
        ext_words[4],
        0x0003_000e, // OpMemoryModel Logical OpenCL
        0,
        2,
    ];
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::OpenCl2_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_NV_mesh_shader"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn vulkan_accepts_nv_vendor_extension() {
    let ext_words = [
        1599492179, 1834964558, 1600680805, 1684105331, 29285, // "SPV_NV_mesh_shader\0"
    ];
    let binary = [
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        2,           // bound
        0,           // schema
        0x0002_0011, // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        0x0006_000a, // OpExtension, word count 6
        ext_words[0],
        ext_words[1],
        ext_words[2],
        ext_words[3],
        ext_words[4],
        0x0003_000e, // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let validated = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Vulkan1_2)
        .expect("NV vendor extension should be allowed for Vulkan");
    assert_eq!(validated.env(), TargetEnv::Vulkan1_2);
}

#[test]
fn vulkan_accepts_google_vendor_extension() {
    let ext_words = [
        1599492179, 1196379975, 1683965260, 1919902565, 1600484449, 1769108595, 26478,
    ]; // "SPV_GOOGLE_decorate_string\0"
    let binary = [
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        2,           // bound
        0,           // schema
        0x0002_0011, // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        0x0008_000a, // OpExtension, word count 8
        ext_words[0],
        ext_words[1],
        ext_words[2],
        ext_words[3],
        ext_words[4],
        ext_words[5],
        ext_words[6],
        0x0003_000e, // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let validated = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Vulkan1_2)
        .expect("Google vendor extension should be allowed for Vulkan");
    assert_eq!(validated.env(), TargetEnv::Vulkan1_2);
}

#[test]
fn vulkan_rejects_intel_vendor_extension() {
    let intel_function_variants_ext = [
        1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
    ];
    let binary = [
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        2,           // bound
        0,           // schema
        0x0002_0011, // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        0x0008_000a, // OpExtension, word count 8
        intel_function_variants_ext[0],
        intel_function_variants_ext[1],
        intel_function_variants_ext[2],
        intel_function_variants_ext[3],
        intel_function_variants_ext[4],
        intel_function_variants_ext[5],
        intel_function_variants_ext[6],
        0x0003_000e, // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_INTEL_function_variants"),
            env: TargetEnv::Vulkan1_2
        }
    );
}

#[test]
fn qcom_extension_requires_vulkan_environment() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_QCOM_image_processing\"",
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
        .expect_err("QCOM extension should be disallowed outside Vulkan");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_QCOM_image_processing"),
            env: TargetEnv::OpenCl2_2
        }
    );
    let validated = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("QCOM extension should be accepted for Vulkan");
    assert_eq!(validated.env(), TargetEnv::Vulkan1_2);
}

#[test]
fn qcom_image_processing_requires_spirv_1_4() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_QCOM_image_processing\"",
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
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("SPIR-V 1.4 is required for QCOM image processing");
    assert_eq!(
        error,
        ValidationError::ExtensionRequiresSpirvVersion {
            extension: ExtensionName::from("SPV_QCOM_image_processing"),
            required_version: SpirvVersion::new(1, 4),
            target_version: TargetEnv::Vulkan1_0.spirv_version(),
        }
    );
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("extension should be accepted with SPIR-V 1.4+");
}

#[test]
fn universal_accepts_vulkan_specific_extension() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_vulkan_memory_model\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("Universal env should accept Vulkan-specific extensions");
}

#[test]
fn universal_accepts_descriptor_indexing_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability RuntimeDescriptorArray",
        "OpExtension \"SPV_EXT_descriptor_indexing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Universal1_3)
        .expect("Universal env should accept SPV_EXT_descriptor_indexing");
}

#[test]
fn opencl_rejects_descriptor_indexing_extension() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_EXT_descriptor_indexing\"",
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
        .expect_err("OpenCL should reject SPV_EXT_descriptor_indexing");
    assert!(
        matches!(error, ValidationError::DisallowedExtension { .. }),
        "Expected DisallowedExtension, got: {error:?}"
    );
}

#[test]
fn vulkan_memory_model_extension_requires_spirv_1_3() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_vulkan_memory_model\"",
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
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("SPIR-V 1.3 is required for Vulkan memory model extension");
    assert_eq!(
        error,
        ValidationError::ExtensionRequiresSpirvVersion {
            extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
            required_version: SpirvVersion::new(1, 3),
            target_version: TargetEnv::Vulkan1_0.spirv_version(),
        }
    );
    // A newer environment should accept the extension.
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("extension should be accepted with SPIR-V 1.4+");
}

#[test]
fn qcom_cooperative_matrix_conversion_requires_spirv_1_3() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_QCOM_cooperative_matrix_conversion\"",
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
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("SPIR-V 1.3 is required for QCOM cooperative matrix conversion");
    assert_eq!(
        error,
        ValidationError::ExtensionRequiresSpirvVersion {
            extension: ExtensionName::from("SPV_QCOM_cooperative_matrix_conversion"),
            required_version: SpirvVersion::new(1, 3),
            target_version: TargetEnv::Vulkan1_0.spirv_version(),
        }
    );
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("extension should be accepted with SPIR-V 1.3+");
}

#[test]
fn capability_requires_declared_vendor_extension() {
    let text = [
        "OpCapability RayTracingNV",
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
        .expect_err("Vendor capability without required extension should be rejected");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::RayTracingNV,
            required_extension: "SPV_NV_ray_tracing".to_string()
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability RayTracingNV",
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
    with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("Vendor capability should be allowed with its extension declared");
}

#[test]
fn vendor_capability_requiring_disallowed_extension_reports_env_error() {
    let text = [
        "OpCapability RayTracingNV",
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("Capability should be rejected when its required extension is disallowed");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_NV_ray_tracing"),
                env
            }
        );
    }
}

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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
fn nv_shader_invocation_reorder_requires_spirv_1_4() {
    let text = [
        "OpCapability Shader",
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
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect_err("SPIR-V 1.4 is required for NV shader invocation reorder");
    assert_eq!(
        error,
        ValidationError::ExtensionRequiresSpirvVersion {
            extension: ExtensionName::from("SPV_NV_shader_invocation_reorder"),
            required_version: SpirvVersion::new(1, 4),
            target_version: TargetEnv::Vulkan1_1.spirv_version(),
        }
    );
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("extension should be accepted with SPIR-V 1.3+");
}

#[test]
fn ext_shader_invocation_reorder_requires_spirv_1_4() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_EXT_shader_invocation_reorder\"",
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
        .validate(TargetEnv::Vulkan1_1)
        .expect_err("SPIR-V 1.4 is required for EXT shader invocation reorder");
    assert_eq!(
        error,
        ValidationError::ExtensionRequiresSpirvVersion {
            extension: ExtensionName::from("SPV_EXT_shader_invocation_reorder"),
            required_version: SpirvVersion::new(1, 4),
            target_version: TargetEnv::Vulkan1_1.spirv_version(),
        }
    );
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("extension should be accepted with SPIR-V 1.4+");
}

#[test]
fn extension_version_check_respects_module_version() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 0);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let fn_type = builder.type_function(void, std::iter::empty::<u32>());
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("module version 1.0 cannot use Vulkan memory model");
    assert_eq!(
        error,
        ValidationError::ExtensionRequiresSpirvVersion {
            extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
            required_version: SpirvVersion::new(1, 3),
            target_version: SpirvVersion::new(1, 0),
        }
    );
}

#[test]
fn extension_version_clamps_to_env_when_module_is_newer() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let fn_type = builder.type_function(void, std::iter::empty::<u32>());
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("env version should clamp module version when gating extension");
    assert_eq!(
        error,
        ValidationError::ExtensionRequiresSpirvVersion {
            extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
            required_version: SpirvVersion::new(1, 3),
            target_version: TargetEnv::Vulkan1_0.spirv_version(),
        }
    );
}

#[test]
fn physical_storage_buffer_extension_requires_spirv_1_3() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_physical_storage_buffer\"",
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
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("physical storage buffer requires SPIR-V 1.3");
    assert_eq!(
        error,
        ValidationError::ExtensionRequiresSpirvVersion {
            extension: ExtensionName::from("SPV_KHR_physical_storage_buffer"),
            required_version: SpirvVersion::new(1, 3),
            target_version: TargetEnv::Vulkan1_0.spirv_version(),
        }
    );
    text.as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("extension should be accepted with SPIR-V 1.3+");
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
fn non_opencl_env_rejects_opencl_extension() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_opencl_enqueue\"",
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
        .expect_err("Vulkan should reject OpenCL-specific extension");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_opencl_enqueue"),
            env: TargetEnv::Vulkan1_2
        }
    );
}

#[test]
fn vulkan_allows_optional_geometry_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability Geometry",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let module = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("optional Vulkan capability should be permitted");
    assert_eq!(module.env(), TargetEnv::Vulkan1_0);
}

#[test]
fn opencl_requires_image_basic_for_image_capabilities() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",
        "OpCapability Sampled1D",
        "OpCapability Image1D",
        "OpMemoryModel Physical32 OpenCL",
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
        .validate(TargetEnv::OpenCl2_0)
        .expect_err("Image1D requires ImageBasic");
    match error {
        ValidationError::MissingRequiredCapability {
            required_capability,
            capability,
        } => {
            assert_eq!(required_capability, rspirv::spirv::Capability::ImageBasic);
            assert!(
                capability == rspirv::spirv::Capability::Image1D
                    || capability == rspirv::spirv::Capability::Sampled1D
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
    let text_with_basic = [
        "OpCapability Kernel",
        "OpCapability Addresses",
        "OpCapability ImageBasic",
        "OpCapability Sampled1D",
        "OpCapability Image1D",
        "OpMemoryModel Physical32 OpenCL",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text_with_basic
        .as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("ImageBasic enables other image capabilities");
}

#[test]
fn opencl_embedded_rejects_int64_capability() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Int64",
        "OpMemoryModel Physical32 OpenCL",
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
        .validate(TargetEnv::OpenClEmbedded1_2)
        .expect_err("embedded OpenCL should reject Int64");
    assert_eq!(
        error,
        ValidationError::DisallowedCapability {
            capability: rspirv::spirv::Capability::Int64,
            env: TargetEnv::OpenClEmbedded1_2
        }
    );
}

#[test]
fn valid_module_cache_reuses_entries() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let mut cache = ValidModuleCache::default();
    let first = cache
        .validate_words(&binary, TargetEnv::Universal1_6)
        .expect("first validation");
    let second = cache
        .validate_words(&binary, TargetEnv::Universal1_6)
        .expect("cached validation");
    assert_eq!(
        Arc::as_ptr(&first),
        Arc::as_ptr(&second),
        "cached entries should reuse the same allocation"
    );
}

#[test]
fn spec_conditional_capability_requires_extension() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Linkage",
        "OpCapability SpecConditionalINTEL",
        "OpMemoryModel Logical OpenCL",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect_err("SpecConditionalINTEL requires SPV_INTEL_function_variants");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::SpecConditionalINTEL,
            required_extension: "SPV_INTEL_function_variants".to_string()
        }
    );
}

#[test]
fn function_variants_capability_requires_spec_conditional() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Linkage",
        "OpCapability FunctionVariantsINTEL",
        "OpExtension \"SPV_INTEL_function_variants\"",
        "OpMemoryModel Logical OpenCL",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect_err("FunctionVariantsINTEL requires SpecConditionalINTEL capability");
    assert_eq!(
        error,
        ValidationError::MissingRequiredCapability {
            required_capability: rspirv::spirv::Capability::SpecConditionalINTEL,
            capability: rspirv::spirv::Capability::FunctionVariantsINTEL
        }
    );
}

#[test]
fn function_variants_capability_accepts_extension_and_dependency() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Linkage",
        "OpCapability SpecConditionalINTEL",
        "OpCapability FunctionVariantsINTEL",
        "OpExtension \"SPV_INTEL_function_variants\"",
        "OpMemoryModel Logical OpenCL",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("FunctionVariantsINTEL should be accepted with required extension and capability");
}

#[test]
fn function_variants_extension_rejected_for_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpCapability SpecConditionalINTEL",
        "OpCapability FunctionVariantsINTEL",
        "OpExtension \"SPV_INTEL_function_variants\"",
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
        .expect_err("Intel function variants extension should be rejected for Vulkan");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_INTEL_function_variants"),
            env: TargetEnv::Vulkan1_2
        }
    );
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("Universal environment should accept vendor extensions");
}

#[test]
fn conditional_entry_point_accepts_execution_modes() {
    let intel_function_variants_ext = [
        1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
    ];
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        9,          // bound (ids up to 8)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(2, 17), // OpCapability SpecConditionalINTEL
        rspirv::spirv::Capability::SpecConditionalINTEL as u32,
        0x0008_000a, // OpExtension "SPV_INTEL_function_variants"
        intel_function_variants_ext[0],
        intel_function_variants_ext[1],
        intel_function_variants_ext[2],
        intel_function_variants_ext[3],
        intel_function_variants_ext[4],
        intel_function_variants_ext[5],
        intel_function_variants_ext[6],
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(6, 6249), // OpConditionalEntryPointINTEL %5 Fragment %7 "main"
        5,
        rspirv::spirv::ExecutionModel::Fragment as u32,
        7,
        0x6e69_616d,
        0,
        op(3, 16), // OpExecutionMode %7 OriginUpperLeft
        7,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(2, 20), // OpTypeBool %3
        3,
        op(3, 41), // OpConstantTrue %3 %5
        3,
        5,
        op(5, 54), // OpFunction %1 %7 None %2
        1,
        7,
        0,
        2,
        op(2, 248), // OpLabel %8
        8,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("conditional entry points should participate in execution-mode validation");
}

#[test]
fn capability_requiring_extension_must_declare_it() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
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
        .expect_err("capability should require extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::RayTracingKHR,
            required_extension: "SPV_KHR_ray_tracing".to_string()
        }
    );
    let text_with_extension = [
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
    let validated = text_with_extension
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("extension present should allow capability");
    assert_eq!(validated.header().schema(), Schema::ZERO);
}

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
fn shader_clock_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_KHR_shader_clock");
}

#[test]
fn fragment_shader_barycentric_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_KHR_fragment_shader_barycentric");
}

#[test]
fn qcom_cooperative_matrix_conversion_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_QCOM_cooperative_matrix_conversion");
}

#[test]
fn untyped_pointers_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_KHR_untyped_pointers");
}

#[test]
fn subgroup_uniform_control_flow_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_KHR_subgroup_uniform_control_flow");
}

#[test]
fn nv_fragment_shader_barycentric_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_NV_fragment_shader_barycentric");
}

#[test]
fn workgroup_memory_explicit_layout_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_KHR_workgroup_memory_explicit_layout");
}

#[test]
fn physical_storage_buffer_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_KHR_physical_storage_buffer");
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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
fn universal_rejects_tile_shading_extension() {
    let text = [
        "OpCapability Shader",
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
    let error = text
        .as_str()
        .validate(TargetEnv::OpenCl2_2)
        .expect_err("Tile shading extension should be Vulkan-only");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_QCOM_tile_shading"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn nvx_extensions_are_vulkan_only() {
    let text = module_with_extension("SPV_NVX_multiview_per_view_attributes");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("NVX extensions should be accepted for Vulkan targets");
    let error = text
        .as_str()
        .validate(TargetEnv::OpenCl2_2)
        .expect_err("NVX extensions are Vulkan-only");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_NVX_multiview_per_view_attributes"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn amdx_extensions_are_vulkan_only() {
    let text = module_with_extension("SPV_AMDX_shader_enqueue");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("AMDX extensions should be accepted for Vulkan targets");
    let error = text
        .as_str()
        .validate(TargetEnv::OpenCl2_2)
        .expect_err("AMDX extensions are Vulkan-only");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_AMDX_shader_enqueue"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn arm_extensions_are_vulkan_only() {
    let text = module_with_extension("SPV_ARM_graph");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ARM extensions should be accepted for Vulkan targets");
    let error = text
        .as_str()
        .validate(TargetEnv::OpenCl2_2)
        .expect_err("ARM extensions are Vulkan-only");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_ARM_graph"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn altera_extensions_reject_vulkan() {
    let text = opencl_module_with_extension("SPV_ALTERA_fpga_memory_attributes");
    text.as_str()
        .validate(TargetEnv::OpenCl2_2)
        .expect("ALTERA extensions are permitted for OpenCL targets");
    text.as_str()
        .validate(TargetEnv::Universal1_5)
        .expect("ALTERA extensions are permitted for universal targets");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("ALTERA extensions should be rejected for Vulkan");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_ALTERA_fpga_memory_attributes"),
            env: TargetEnv::Vulkan1_2
        }
    );
}

#[test]
fn universal_allows_google_and_amd_extensions() {
    let google = module_with_extension("SPV_GOOGLE_decorate_string");
    google
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("GOOGLE extensions should be allowed for universal environments");
    let amd = module_with_extension("SPV_AMD_shader_ballot");
    amd.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("AMD extensions should be allowed for universal environments");
}

#[test]
fn opencl_rejects_google_and_amd_vendor_extensions() {
    let google = opencl_module_with_extension("SPV_GOOGLE_decorate_string");
    let error = google
        .as_str()
        .validate(TargetEnv::OpenCl2_1)
        .expect_err("GOOGLE vendor extensions are not permitted for OpenCL targets");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_GOOGLE_decorate_string"),
            env: TargetEnv::OpenCl2_1
        }
    );
    let amd = opencl_module_with_extension("SPV_AMD_shader_ballot");
    let error = amd
        .as_str()
        .validate(TargetEnv::OpenCl2_1)
        .expect_err("AMD vendor extensions are not permitted for OpenCL targets");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_AMD_shader_ballot"),
            env: TargetEnv::OpenCl2_1
        }
    );
}

#[test]
fn opengl_rejects_google_vendor_extension() {
    let text = module_with_extension("SPV_GOOGLE_decorate_string");
    let error = text
        .as_str()
        .validate(TargetEnv::OpenGl4_5)
        .expect_err("OpenGL should reject GOOGLE vendor extensions");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_GOOGLE_decorate_string"),
            env: TargetEnv::OpenGl4_5
        }
    );
}

#[test]
fn opengl_rejects_amd_vendor_extension() {
    let text = module_with_extension("SPV_AMD_shader_trinary_minmax");
    let error = text
        .as_str()
        .validate(TargetEnv::OpenGl4_5)
        .expect_err("OpenGL should reject AMD vendor extensions");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_AMD_shader_trinary_minmax"),
            env: TargetEnv::OpenGl4_5
        }
    );
}

#[test]
fn vulkan_memory_model_extension_is_vulkan_only() {
    let text = module_with_extension("SPV_KHR_vulkan_memory_model");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("Vulkan memory model should be accepted for Vulkan targets");
    let error = text
        .as_str()
        .validate(TargetEnv::OpenCl2_2)
        .expect_err("Vulkan memory model should be rejected for non-Vulkan targets");
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn intel_function_variants_allowed_for_opencl_and_universal_only() {
    let opencl_text = opencl_module_with_extension("SPV_INTEL_function_variants");
    opencl_text
        .as_str()
        .validate(TargetEnv::OpenCl2_2)
        .expect("INTEL function variants should be accepted for OpenCL targets");
    let universal_text = module_with_extension("SPV_INTEL_function_variants");
    universal_text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("INTEL function variants should be accepted for universal targets");
    for env in [TargetEnv::Vulkan1_2, TargetEnv::OpenGl4_5] {
        let error = universal_text
            .as_str()
            .validate(env)
            .expect_err("INTEL function variants should be rejected outside OpenCL/Universal");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_INTEL_function_variants"),
                env
            }
        );
    }
}

#[test]
fn mesh_shader_extension_is_vulkan_only() {
    let text = module_with_extension("SPV_EXT_mesh_shader");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("Mesh shader extension should be accepted for Vulkan targets");
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("Mesh shader extension should be rejected outside Vulkan");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from("SPV_EXT_mesh_shader"),
                env
            }
        );
    }
}

#[test]
fn descriptor_indexing_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_EXT_descriptor_indexing");
}

#[test]
fn fragment_shader_interlock_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_EXT_fragment_shader_interlock");
}

#[test]
fn fragment_invocation_density_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_EXT_fragment_invocation_density");
}

#[test]
fn shader_atomic_float_min_max_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_EXT_shader_atomic_float_min_max");
}

#[test]
fn shader_invocation_reorder_ext_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_EXT_shader_invocation_reorder");
}

#[test]
fn shader_atomic_float_add_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_EXT_shader_atomic_float_add");
}

#[test]
fn qcom_image_processing_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_QCOM_image_processing");
}

#[test]
fn opencl_environment_accepts_opencl_extension() {
    let text = opencl_module_with_extension("SPV_KHR_opencl_enqueue");
    text.validate(TargetEnv::OpenCl2_2)
        .expect("OpenCL targets should accept OpenCL-specific extensions");
}

#[test]
fn validate_module_rejects_duplicate_extension() {
    // Hand-assemble a module with duplicate OpExtension instructions.
    let extension_word = 0x0008_000a; // word count 8, opcode OpExtension (10)
    let extension_words = [
        0x5f56_5053, // "SPV_"
        0x474f_4f47, // "GOOG"
        0x645f_454c, // "LE_d"
        0x726f_6365, // "ecor"
        0x5f65_7461, // "ate_"
        0x6972_7473, // "stri"
        0x0000_676e, // "ng\0"
    ];
    let binary = [
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        6,           // bound (ids up to 5)
        0,           // schema
        0x0002_0011, // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        extension_word,
        extension_words[0],
        extension_words[1],
        extension_words[2],
        extension_words[3],
        extension_words[4],
        extension_words[5],
        extension_words[6],
        extension_word, // duplicate extension
        extension_words[0],
        extension_words[1],
        extension_words[2],
        extension_words[3],
        extension_words[4],
        extension_words[5],
        extension_words[6],
        0x0003_000e, // OpMemoryModel Logical GLSL450
        0,
        1,
        0x0002_0013, // OpTypeVoid %1
        1,
    ];
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateExtension {
            extension: ExtensionName::from("SPV_GOOGLE_decorate_string")
        }
    );
}

#[test]
fn validate_module_rejects_duplicate_conditional_extension() {
    // Duplicate OpConditionalExtensionINTEL instructions should be rejected.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        3,          // bound (ids up to 2)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string"
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
        op(8, 6248), // duplicate
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateExtension {
            extension: ExtensionName::from("GOOGLE_decorate_string")
        }
    );
}

#[test]
fn conditional_extension_rejected_in_non_vulkan_env() {
    // Vulkan-only conditional extensions must be rejected for non-Vulkan targets.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        0x0009_1868, // OpConditionalExtensionINTEL %1 "SPV_KHR_vulkan_memory_model"
        1,           // condition id (non-zero to satisfy parsing)
        0x5f56_5053, // "SPV_"
        0x5f52_484b, // "KHR_"
        0x6b6c_7576, // "vulk"
        0x6d5f_6e61, // "an_m"
        0x726f_6d65, // "emor"
        0x6f6d_5f79, // "y_mo"
        0x006c_6564, // "del\0"
        op(3, 14),   // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %2
        2,
        op(3, 33), // OpTypeFunction %3 %2
        3,
        2,
        op(5, 54), // OpFunction %2 %4 None %3
        2,
        4,
        0,
        3,
        op(2, 248), // OpLabel %5
        5,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::OpenCl2_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_vulkan_memory_model"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn entry_point_cannot_precede_memory_model() {
    // Entry points are mode-setting instructions that must follow the memory model.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        4,          // bound (ids up to 3)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(5, 15), // OpEntryPoint Vertex %1 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        1,
        0x6e69_616d, // "main"
        0,           // string padding
        op(3, 14),   // OpMemoryModel Logical GLSL450 (misordered after entry point)
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_5).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemoryModel
        }
    );
}

#[test]
fn execution_mode_cannot_precede_memory_model() {
    // Execution modes must follow the memory model stage.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        2,          // bound (ids up to 1)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(6, 16), // OpExecutionMode %1 LocalSize 1 1 1 (misordered before memory model)
        1,
        rspirv::spirv::ExecutionMode::LocalSize as u32,
        1,
        1,
        1,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_5).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemoryModel
        }
    );
}

#[test]
fn conditional_extension_rejected_in_webgpu() {
    // WebGPU forbids all extensions, including conditional ones.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        0x0008_1868, // OpConditionalExtensionINTEL %1 "SPV_KHR_shader_clock"
        1,           // condition id
        0x5f56_5053, // "SPV_"
        0x5f52_484b, // "KHR_"
        0x6461_6873, // "shad"
        0x635f_7265, // "er_c"
        0x6b63_6f6c, // "lock"
        0x0000_0000, // null terminator padding
        op(3, 14),   // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %2
        2,
        op(3, 33), // OpTypeFunction %3 %2
        3,
        2,
        op(5, 54), // OpFunction %2 %4 None %3
        2,
        4,
        0,
        3,
        op(2, 248), // OpLabel %5
        5,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::WebGpu0).unwrap_err();
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_shader_clock"),
            env: TargetEnv::WebGpu0
        }
    );
}

#[test]
fn webgpu_disallows_extensions_for_text_and_binary() {
    let module_text = [
        "OpCapability Shader",
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
    let expected_error = ValidationError::DisallowedExtension {
        extension: ExtensionName::from("SPV_KHR_ray_tracing"),
        env: TargetEnv::WebGpu0,
    };
    let text_error = module_text
        .as_str()
        .validate(TargetEnv::WebGpu0)
        .unwrap_err();
    assert_eq!(text_error, expected_error);
    let binary = assemble_text(&module_text).expect("assemble");
    let binary_error = binary.as_slice().validate(TargetEnv::WebGpu0).unwrap_err();
    assert_eq!(binary_error, expected_error);
    let validated = binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect("extension should be accepted for Vulkan environments");
    assert_eq!(validated.header().schema(), Schema::ZERO);
}

#[test]
fn extensions_cannot_appear_inside_functions_even_when_layout_skipped() {
    // When layout checks are disabled, function-local extensions remain invalid.
    const EXT_SPV_KHR_RAY_TRACING: [u32; 5] = [
        0x5f56_5053, // "SPV_"
        0x5f52_484b, // "KHR_"
        0x5f79_6172, // "ray_"
        0x6361_7274, // "trac"
        0x0067_6e69, // "ing\0"
    ];
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %1 %3 None %2
        1,
        3,
        rspirv::spirv::FunctionControl::NONE.bits(),
        2,
        op(2, 248), // OpLabel %4
        4,
        op(6, rspirv::spirv::Op::Extension as u16), // OpExtension "SPV_KHR_ray_tracing" (inside function)
        EXT_SPV_KHR_RAY_TRACING[0],
        EXT_SPV_KHR_RAY_TRACING[1],
        EXT_SPV_KHR_RAY_TRACING[2],
        EXT_SPV_KHR_RAY_TRACING[3],
        EXT_SPV_KHR_RAY_TRACING[4],
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let options = ValidationOptions {
        skip_block_layout: true,
        ..ValidationOptions::default()
    };
    let error = validate_module_with_options(&binary, TargetEnv::Vulkan1_3, options).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn conditional_capability_requires_extension() {
    // Extension dependencies apply to conditional capabilities.
    let binary = vec![
        0x07230203,  // magic
        0x00010000,  // version
        0,           // generator
        6,           // bound
        0,           // schema
        op(3, 6250), // OpConditionalCapabilityINTEL %1 SpecConditionalINTEL
        1,
        rspirv::spirv::Capability::SpecConditionalINTEL as u32,
        op(3, 14), // OpMemoryModel Logical OpenCL
        2,         // OpenCL
        0,
        op(2, 19), // OpTypeVoid %2
        2,
        op(3, 33), // OpTypeFunction %3 %2
        3,
        2,
        op(5, 54), // OpFunction %2 %4 None %3
        2,
        4,
        0,
        3,
        op(2, 248), // OpLabel %5
        5,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::SpecConditionalINTEL,
            required_extension: "SPV_INTEL_function_variants".to_string()
        }
    );
}

#[test]
fn effective_spirv_version_clamps_to_env() {
    use super::effective_spirv_version;
    assert_eq!(
        effective_spirv_version(TargetEnv::Vulkan1_0, SpirvVersion::new(1, 3)),
        TargetEnv::Vulkan1_0.spirv_version()
    );
    assert_eq!(
        effective_spirv_version(TargetEnv::Vulkan1_3, SpirvVersion::new(1, 1)),
        SpirvVersion::new(1, 1)
    );
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

#[test]
fn device_group_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_KHR_device_group");
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
    for env in [
        TargetEnv::OpenCl2_2,
        TargetEnv::OpenGl4_5,
    ] {
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

#[test]
fn variable_pointers_requires_storage_buffer_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability VariablePointers",
        "OpExtension \"SPV_KHR_variable_pointers\"",
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
        .validate(TargetEnv::Universal1_6)
        .expect_err("VariablePointers requires VariablePointersStorageBuffer");
    assert_eq!(
        error,
        ValidationError::MissingRequiredCapability {
            required_capability: rspirv::spirv::Capability::VariablePointersStorageBuffer,
            capability: rspirv::spirv::Capability::VariablePointers,
        }
    );
    let with_dependency = [
        "OpCapability Shader",
        "OpCapability VariablePointersStorageBuffer",
        "OpCapability VariablePointers",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_dependency
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("dependency declared should satisfy requirement");
}

#[test]
fn shader_capability_does_not_require_matrix_soft_dependency() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("Shader should not require Matrix due to soft dependency");
}

#[test]
fn image_buffer_requires_sampled_buffer_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability ImageBuffer",
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
        .expect_err("ImageBuffer requires SampledBuffer capability");
    assert_eq!(
        error,
        ValidationError::MissingRequiredCapability {
            required_capability: rspirv::spirv::Capability::SampledBuffer,
            capability: rspirv::spirv::Capability::ImageBuffer
        }
    );
    let with_dependency = [
        "OpCapability Shader",
        "OpCapability SampledBuffer",
        "OpCapability ImageBuffer",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_dependency
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("dependency declared should satisfy requirement");
}

#[test]
fn sampled_cube_array_requires_shader_capability() {
    let text = [
        "OpCapability SampledCubeArray",
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
        .validate(TargetEnv::Universal1_2)
        .expect_err("SampledCubeArray requires Shader capability");
    assert_eq!(
        error,
        ValidationError::MissingRequiredCapability {
            required_capability: rspirv::spirv::Capability::Shader,
            capability: rspirv::spirv::Capability::SampledCubeArray
        }
    );
    let with_shader = [
        "OpCapability Shader",
        "OpCapability SampledCubeArray",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_shader
        .as_str()
        .validate(TargetEnv::Universal1_2)
        .expect("Shader capability declared should satisfy dependency");
}

#[test]
fn image_ms_array_requires_shader_capability() {
    let text = [
        "OpCapability ImageMSArray",
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
        .validate(TargetEnv::Universal1_2)
        .expect_err("ImageMSArray requires Shader capability");
    assert_eq!(
        error,
        ValidationError::MissingRequiredCapability {
            required_capability: rspirv::spirv::Capability::Shader,
            capability: rspirv::spirv::Capability::ImageMSArray
        }
    );
    let with_shader = [
        "OpCapability Shader",
        "OpCapability ImageMSArray",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_shader
        .as_str()
        .validate(TargetEnv::Universal1_2)
        .expect("Shader capability declared should satisfy dependency");
}

#[test]
fn ray_tracing_requires_shader_capability() {
    let text = [
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
        .expect_err("RayTracingKHR requires Shader capability");
    assert_eq!(
        error,
        ValidationError::MissingRequiredCapability {
            required_capability: rspirv::spirv::Capability::Shader,
            capability: rspirv::spirv::Capability::RayTracingKHR
        }
    );
    let with_shader = [
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
    with_shader
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("Shader capability declared should satisfy dependency");
}

#[test]
fn group_non_uniform_arithmetic_requires_group_non_uniform() {
    let text = [
        "OpCapability Shader",
        "OpCapability GroupNonUniformArithmetic",
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
        .expect_err("GroupNonUniformArithmetic requires GroupNonUniform");
    assert_eq!(
        error,
        ValidationError::MissingRequiredCapability {
            required_capability: rspirv::spirv::Capability::GroupNonUniform,
            capability: rspirv::spirv::Capability::GroupNonUniformArithmetic
        }
    );
    let with_base = [
        "OpCapability Shader",
        "OpCapability GroupNonUniform",
        "OpCapability GroupNonUniformArithmetic",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_base
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("base capability declared should satisfy dependency");
}

#[test]
fn device_enqueue_requires_kernel() {
    let text = [
        "OpCapability DeviceEnqueue",
        "OpCapability Addresses",
        "OpMemoryModel Physical32 OpenCL",
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
        .validate(TargetEnv::OpenCl2_0)
        .expect_err("DeviceEnqueue requires Kernel capability");
    assert_eq!(
        error,
        ValidationError::MissingRequiredCapability {
            required_capability: rspirv::spirv::Capability::Kernel,
            capability: rspirv::spirv::Capability::DeviceEnqueue
        }
    );
    let with_kernel = [
        "OpCapability Kernel",
        "OpCapability Addresses",
        "OpCapability DeviceEnqueue",
        "OpMemoryModel Physical32 OpenCL",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    with_kernel
        .as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("kernel capability enables device enqueue");
}

#[test]
fn physical_storage_buffer_capability_requires_spirv_1_4() {
    let text = [
        "OpCapability Shader",
        "OpCapability PhysicalStorageBufferAddresses",
        "OpExtension \"SPV_KHR_physical_storage_buffer\"",
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
        .expect_err("requires SPIR-V 1.4");
    match error {
        ValidationError::CapabilityRequiresSpirvVersion {
            capability,
            required_version,
            target_version,
        } => {
            assert_eq!(
                capability,
                rspirv::spirv::Capability::PhysicalStorageBufferAddresses
            );
            assert_eq!(required_version, SpirvVersion::new(1, 4));
            assert_eq!(target_version, SpirvVersion::new(1, 2));
        }
        ValidationError::ExtensionRequiresSpirvVersion {
            extension,
            required_version,
            target_version,
        } => {
            assert_eq!(
                extension,
                ExtensionName::from("SPV_KHR_physical_storage_buffer")
            );
            assert_eq!(required_version, SpirvVersion::new(1, 4));
            assert_eq!(target_version, SpirvVersion::new(1, 2));
        }
        ValidationError::DisallowedExtension { extension, env } => {
            assert_eq!(
                extension,
                ExtensionName::from("SPV_KHR_physical_storage_buffer")
            );
            assert_eq!(env, TargetEnv::OpenCl2_2);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("succeeds on newer SPIR-V");
}

#[test]
fn opencl_1_2_rejects_image_read_write() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",
        "OpCapability ImageBasic",
        "OpCapability ImageReadWrite",
        "OpMemoryModel Physical32 OpenCL",
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
        .validate(TargetEnv::OpenCl1_2)
        .expect_err("ImageReadWrite disallowed in OpenCL 1.2");
    match error {
        ValidationError::DisallowedCapability { capability, .. } => {
            assert_eq!(capability, rspirv::spirv::Capability::ImageReadWrite);
        }
        other => panic!("expected DisallowedCapability, got {other:?}"),
    }
}

#[test]
fn opencl_2_0_allows_image_read_write_with_image_basic() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",
        "OpCapability ImageBasic",
        "OpCapability ImageReadWrite",
        "OpMemoryModel Physical32 OpenCL",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("ImageReadWrite allowed in OpenCL 2.0 with ImageBasic");
}

#[test]
fn opencl_image_read_write_requires_image_basic() {
    let text = [
        "OpCapability Kernel",
        "OpCapability Addresses",
        "OpCapability ImageReadWrite",
        "OpMemoryModel Physical32 OpenCL",
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
        .validate(TargetEnv::OpenCl2_0)
        .expect_err("ImageReadWrite requires ImageBasic");
    match error {
        ValidationError::MissingRequiredCapability {
            required_capability,
            capability,
        } => {
            assert_eq!(required_capability, rspirv::spirv::Capability::ImageBasic);
            assert_eq!(capability, rspirv::spirv::Capability::ImageReadWrite);
        }
        other => panic!("expected MissingRequiredCapability, got {other:?}"),
    }
}
