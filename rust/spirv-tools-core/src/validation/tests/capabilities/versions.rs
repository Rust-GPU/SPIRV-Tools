use super::super::*;

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
fn effective_spirv_version_clamps_to_env() {
    use super::super::effective_spirv_version;
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
