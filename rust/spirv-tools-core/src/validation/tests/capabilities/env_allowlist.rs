use super::super::*;

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
