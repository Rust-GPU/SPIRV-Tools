use super::*;

#[test]
fn uniform_without_block_is_rejected() {
    // Tests that Uniform storage class requires Block decoration.
    // Use 32-bit types so no 16-bit capability issues.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%buf = OpTypeStruct %u32",
        "%ptr = OpTypePointer Uniform %buf",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate(text).expect_err("expected validation error");
    assert!(
        matches!(error, ValidationError::MissingBlockDecoration { .. }),
        "expected MissingBlockDecoration, got: {error:?}"
    );
}

#[test]
fn uniform_16bit_with_buffer_block_allows_storage_buffer_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability StorageBuffer16BitAccess",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "OpDecorate %buf BufferBlock",
        "OpMemberDecorate %buf 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%buf = OpTypeStruct %u16",
        "%ptr = OpTypePointer Uniform %buf",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate(text)
        .expect("BufferBlock + StorageBuffer16BitAccess should satisfy uniform 16-bit");
}

#[test]
fn uniform_constant_16bit_is_disallowed() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%ptr = OpTypePointer UniformConstant %u16",
        "%var = OpVariable %ptr UniformConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate(text).expect_err("expected storage class rejection");
    assert_eq!(
        error,
        ValidationError::SmallTypeDisallowedInStorageClass {
            bit_width: 16,
            storage_class: StorageClass::UniformConstant
        }
    );
}

#[test]
fn storage_buffer_8bit_without_int8_requires_capability() {
    // When Int8 capability is NOT declared, storage class restrictions apply
    let text = [
        "OpCapability Shader",
        // Note: No Int8 capability - this triggers storage class checks
        "OpCapability VariablePointers",
        "OpCapability VariablePointersStorageBuffer",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpExtension \"SPV_KHR_8bit_storage\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "OpDecorate %buf Block",
        "OpMemberDecorate %buf 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u8 = OpTypeInt 8 0",
        "%buf = OpTypeStruct %u8",
        "%ptr = OpTypePointer StorageBuffer %buf",
        "%var = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate(text).expect_err("expected missing capability");
    assert_eq!(
        error,
        ValidationError::SmallTypeMissingCapability {
            bit_width: 8,
            storage_class: StorageClass::StorageBuffer,
            required_capability: Capability::StorageBuffer8BitAccess
        }
    );
}

#[test]
fn storage_buffer_8bit_with_capability_is_allowed() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int8",
        "OpCapability VariablePointers",
        "OpCapability VariablePointersStorageBuffer",
        "OpCapability StorageBuffer8BitAccess",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpExtension \"SPV_KHR_8bit_storage\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "OpDecorate %buf Block",
        "OpMemberDecorate %buf 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u8 = OpTypeInt 8 0",
        "%buf = OpTypeStruct %u8",
        "%ptr = OpTypePointer StorageBuffer %buf",
        "%var = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect("StorageBuffer8BitAccess should allow 8-bit storage buffer");
}

#[test]
fn buffer_block_disallowed_after_spirv_1_3() {
    let text = [
        "OpCapability Shader",
        "OpCapability Linkage",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %buf BufferBlock",
        "OpMemberDecorate %buf 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%buf = OpTypeStruct %u32",
        "%ptr = OpTypePointer Uniform %buf",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_4)
        .expect_err("BufferBlock should be disallowed after SPIR-V 1.3");
    assert_eq!(
        err,
        ValidationError::DecorationRequiresSpirvVersion {
            decoration: rspirv::spirv::Decoration::BufferBlock,
            required_version: SpirvVersion::new(1, 3),
            target_version: SpirvVersion::new(1, 4)
        }
    );
}

#[test]
fn buffer_block_cannot_be_used_for_push_constant() {
    let text = [
        "OpCapability Shader",
        "OpCapability Linkage",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %buf BufferBlock",
        "OpMemberDecorate %buf 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%buf = OpTypeStruct %u32",
        "%ptr = OpTypePointer PushConstant %buf",
        "%var = OpVariable %ptr PushConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_3)
        .expect_err("BufferBlock should not apply to push constants");
    assert_eq!(
        err,
        ValidationError::InvalidBlockDecorationStorageClass {
            decoration: rspirv::spirv::Decoration::BufferBlock,
            storage_class: rspirv::spirv::StorageClass::PushConstant
        }
    );
}

#[test]
fn block_cannot_be_used_for_input_storage() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %buf Block",
        "OpMemberDecorate %buf 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%buf = OpTypeStruct %u32",
        "%ptr = OpTypePointer Input %buf",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("Block should not apply to input variables");
    assert_eq!(
        err,
        ValidationError::InvalidBlockDecorationStorageClass {
            decoration: rspirv::spirv::Decoration::Block,
            storage_class: rspirv::spirv::StorageClass::Input
        }
    );
}

#[test]
fn descriptor_on_input_storage_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var Binding 0",
        "OpDecorate %var DescriptorSet 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Input %u32",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("descriptor decorations on Input should be rejected");
    assert_eq!(
        err,
        ValidationError::InvalidDescriptorStorageClass {
            storage_class: rspirv::spirv::StorageClass::Input
        }
    );
}

#[test]
fn descriptor_on_push_constant_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var Binding 0",
        "OpDecorate %var DescriptorSet 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer PushConstant %u32",
        "%var = OpVariable %ptr PushConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("descriptor decorations on PushConstant should be rejected");
    assert_eq!(
        err,
        ValidationError::InvalidDescriptorStorageClass {
            storage_class: rspirv::spirv::StorageClass::PushConstant
        }
    );
}

#[test]
fn resource_variable_requires_descriptor_set_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var Binding 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Uniform %u32",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("Uniform variables require DescriptorSet in Vulkan");
    assert!(matches!(
        err,
        ValidationError::MissingDescriptorSetDecoration { .. }
    ));
}

#[test]
fn resource_variable_requires_binding_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var DescriptorSet 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Uniform %u32",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("Uniform variables require Binding in Vulkan");
    assert!(matches!(
        err,
        ValidationError::MissingBindingDecoration { .. }
    ));
}

#[test]
fn location_on_uniform_storage_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var Location 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Uniform %u32",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("Location should be rejected on Uniform storage");
    assert_eq!(
        err,
        ValidationError::InvalidLocationStorageClass {
            storage_class: rspirv::spirv::StorageClass::Uniform
        }
    );
}

#[test]
fn location_on_push_constant_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var Location 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer PushConstant %u32",
        "%var = OpVariable %ptr PushConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("Location should be rejected on PushConstant storage");
    assert_eq!(
        err,
        ValidationError::InvalidLocationStorageClass {
            storage_class: rspirv::spirv::StorageClass::PushConstant
        }
    );
}

#[test]
fn interpolation_decorations_require_input_or_output_storage() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var Flat",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%ptr = OpTypePointer Uniform %f32",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("Flat on Uniform should be rejected");
    assert_eq!(
        err,
        ValidationError::InterpolationDecorationInvalidStorageClass {
            decoration: rspirv::spirv::Decoration::Flat,
            storage_class: rspirv::spirv::StorageClass::Uniform
        }
    );
}

#[test]
fn uniform_constant_8bit_is_disallowed() {
    // When Int8 capability is NOT present, 8-bit types in UniformConstant
    // storage class should be rejected because there's no enabling capability.
    // Note: Without Int8, the 8-bit type definition itself requires a storage
    // capability like StorageBuffer8BitAccess, StoragePushConstant8, etc.
    let text = [
        "OpCapability Shader",
        // No Int8 capability - so storage class restrictions apply
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u8 = OpTypeInt 8 0",
        "%ptr = OpTypePointer UniformConstant %u8",
        "%var = OpVariable %ptr UniformConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect_err("expected uniform constant rejection");
    assert_eq!(
        error,
        ValidationError::SmallTypeDisallowedInStorageClass {
            bit_width: 8,
            storage_class: StorageClass::UniformConstant
        }
    );
}

#[test]
fn input_8bit_is_disallowed() {
    // When Int8 capability is NOT present, 8-bit types in Input storage class
    // should be rejected because there's no enabling capability for Input/Output.
    let text = [
        "OpCapability Shader",
        // No Int8 capability - so storage class restrictions apply
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u8 = OpTypeInt 8 0",
        "%ptr = OpTypePointer Input %u8",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect_err("expected disallowed input");
    assert_eq!(
        error,
        ValidationError::SmallTypeDisallowedInStorageClass {
            bit_width: 8,
            storage_class: StorageClass::Input
        }
    );
}

#[test]
fn input_16bit_with_storage_input_output_capability_is_allowed() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int16",
        "OpCapability StorageInputOutput16",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%ptr = OpTypePointer Input %u16",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_3)
        .expect("StorageInputOutput16 should allow 16-bit input");
}

#[test]
fn output_16bit_with_storage_input_output_capability_is_allowed() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int16",
        "OpCapability StorageInputOutput16",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%ptr = OpTypePointer Output %u16",
        "%var = OpVariable %ptr Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_3)
        .expect("StorageInputOutput16 should allow 16-bit output");
}

#[test]
fn storage_buffer_16bit_with_capability_is_allowed() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int16",
        "OpCapability VariablePointers",
        "OpCapability VariablePointersStorageBuffer",
        "OpCapability StorageBuffer16BitAccess",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "OpDecorate %buf Block",
        "OpMemberDecorate %buf 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%buf = OpTypeStruct %u16",
        "%ptr = OpTypePointer StorageBuffer %buf",
        "%var = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_3)
        .expect("StorageBuffer16BitAccess should allow 16-bit storage buffer");
}

#[test]
fn push_constant_16bit_with_capability_is_allowed() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int16",
        "OpCapability StoragePushConstant16",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%ptr = OpTypePointer PushConstant %u16",
        "%var = OpVariable %ptr PushConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_3)
        .expect("StoragePushConstant16 should allow 16-bit push constants");
}

#[test]
fn workgroup_16bit_requires_capability() {
    // When Int16 capability is NOT present, 16-bit types in Workgroup storage class
    // require WorkgroupMemoryExplicitLayout16BitAccessKHR capability.
    let text = [
        "OpCapability Shader",
        // No Int16 capability - so storage class restrictions apply
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%ptr = OpTypePointer Workgroup %u16",
        "%var = OpVariable %ptr Workgroup",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate_with_env(text, TargetEnv::Universal1_4)
        .expect_err("expected workgroup capability");
    assert_eq!(
        error,
        ValidationError::SmallTypeMissingCapability {
            bit_width: 16,
            storage_class: StorageClass::Workgroup,
            required_capability: Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR
        }
    );
}

#[test]
fn workgroup_16bit_with_capability_is_allowed() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int16",
        "OpCapability WorkgroupMemoryExplicitLayoutKHR",
        "OpCapability WorkgroupMemoryExplicitLayout16BitAccessKHR",
        "OpExtension \"SPV_KHR_workgroup_memory_explicit_layout\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%ptr = OpTypePointer Workgroup %u16",
        "%var = OpVariable %ptr Workgroup",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(&text, TargetEnv::OpenCl2_2)
        .expect_err("extension gate should reject workgroup layout without allowed env");

    assemble_and_validate_with_env(&text, TargetEnv::Vulkan1_1Spirv1_4)
        .expect("WorkgroupMemoryExplicitLayout16BitAccessKHR should allow 16-bit workgroup");
}

#[test]
fn workgroup_8bit_with_capability_is_allowed() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int8",
        "OpCapability WorkgroupMemoryExplicitLayoutKHR",
        "OpCapability WorkgroupMemoryExplicitLayout8BitAccessKHR",
        "OpExtension \"SPV_KHR_workgroup_memory_explicit_layout\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u8 = OpTypeInt 8 0",
        "%ptr = OpTypePointer Workgroup %u8",
        "%var = OpVariable %ptr Workgroup",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_1Spirv1_4)
        .expect("WorkgroupMemoryExplicitLayout8BitAccessKHR should allow 8-bit workgroup");
}

#[test]
fn push_constant_8bit_requires_capability() {
    // When Int8 capability is NOT present, 8-bit types in PushConstant storage class
    // require StoragePushConstant8 capability.
    let text = [
        "OpCapability Shader",
        // No Int8 capability - so storage class restrictions apply
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u8 = OpTypeInt 8 0",
        "%ptr = OpTypePointer PushConstant %u8",
        "%var = OpVariable %ptr PushConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect_err("expected push constant capability");
    assert_eq!(
        error,
        ValidationError::SmallTypeMissingCapability {
            bit_width: 8,
            storage_class: StorageClass::PushConstant,
            required_capability: Capability::StoragePushConstant8
        }
    );
}

#[test]
fn push_constant_8bit_with_capability_is_allowed() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int8",
        "OpCapability StoragePushConstant8",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u8 = OpTypeInt 8 0",
        "%ptr = OpTypePointer PushConstant %u8",
        "%var = OpVariable %ptr PushConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect("StoragePushConstant8 should allow 8-bit push constants");
}

#[test]
fn composite_extract_result_type_must_match_component() {
    use rspirv::{binary::Assemble, dr::Builder};

    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );

    let void = b.type_void();
    let int = b.type_int(32, 1);
    let uint = b.type_int(32, 0);
    let vec_ty = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let composite = b.undef(vec_ty, None);
    b.composite_extract(uint, None, composite, [0]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("composite extract result type must match component type");
    assert_eq!(
        err,
        ValidationError::CompositeOperandTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::CompositeExtract,
            result_type: TypeId::try_from(uint).unwrap(),
        }
    );
}

#[test]
fn vulkan_push_constant_must_have_block_decoration() {
    // PushConstant WITHOUT Block decoration
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer PushConstant %struct",
        "%var = OpVariable %ptr PushConstant",
        "OpEntryPoint Vertex %main \"main\" %var",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("PushConstant without Block should fail in Vulkan");
    assert!(matches!(
        err,
        ValidationError::VulkanBufferMissingBlockDecoration { .. }
    ));
}

#[test]
fn vulkan_push_constant_with_block_is_valid() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer PushConstant %struct",
        "%var = OpVariable %ptr PushConstant",
        "OpEntryPoint Vertex %main \"main\" %var",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("PushConstant with Block decoration should pass");
}

#[test]
fn vulkan_storage_buffer_with_buffer_block_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct BufferBlock",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect_err("StorageBuffer with BufferBlock should fail in Vulkan");
    assert!(matches!(
        err,
        ValidationError::VulkanStorageBufferHasBufferBlock { .. }
    ));
}

#[test]
fn vulkan_storage_buffer_must_have_block_decoration() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect_err("StorageBuffer without Block should fail in Vulkan");
    assert!(matches!(
        err,
        ValidationError::VulkanBufferMissingBlockDecoration { .. }
    ));
}

#[test]
fn vulkan_storage_buffer_with_block_is_valid() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("StorageBuffer with Block decoration should pass");
}

#[test]
fn vulkan_uniform_must_have_block_or_buffer_block() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("Uniform without Block or BufferBlock should fail in Vulkan");
    assert!(matches!(
        err,
        ValidationError::VulkanUniformMissingBlockDecoration { .. }
    ));
}

#[test]
fn vulkan_uniform_with_block_is_valid() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("Uniform with Block decoration should pass");
}

#[test]
fn vulkan_uniform_with_buffer_block_is_valid() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct BufferBlock",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("Uniform with BufferBlock decoration should pass");
}

#[test]
fn universal_env_uses_base_spec_error_for_missing_block() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    // Under Universal, the base spec catches the missing Block decoration
    let err = text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect_err("Missing Block should fail in any env");
    assert!(matches!(
        err,
        ValidationError::MissingBlockDecoration { .. }
    ));
}

#[test]
fn opengl_uniform_block_missing_binding_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%voidfn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %voidfn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::OpenGl4_5)
        .expect_err("OpenGL Uniform Block without Binding should fail");
    assert!(matches!(
        err,
        ValidationError::OpenGlBufferMissingBindingDecoration { .. }
    ));
}

#[test]
fn opengl_uniform_block_with_binding_is_valid() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%voidfn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %voidfn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::OpenGl4_5)
        .expect("OpenGL Uniform Block with Binding should pass");
}

#[test]
fn opengl_uniform_buffer_block_missing_binding_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %struct BufferBlock",
        "OpDecorate %var DescriptorSet 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%voidfn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %voidfn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::OpenGl4_5)
        .expect_err("OpenGL Uniform BufferBlock without Binding should fail");
    assert!(matches!(
        err,
        ValidationError::OpenGlBufferMissingBindingDecoration { .. }
    ));
}

#[test]
fn opengl_storage_buffer_block_missing_binding_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%voidfn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %voidfn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::OpenGl4_5)
        .expect_err("OpenGL StorageBuffer Block without Binding should fail");
    assert!(matches!(
        err,
        ValidationError::OpenGlBufferMissingBindingDecoration { .. }
    ));
}

#[test]
fn opengl_storage_buffer_block_with_binding_is_valid() {
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%voidfn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %voidfn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::OpenGl4_5)
        .expect("OpenGL StorageBuffer Block with Binding should pass");
}

#[test]
fn opengl_uniform_without_block_does_not_need_binding() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var DescriptorSet 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%voidfn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %voidfn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    // Without Block or BufferBlock, the OpenGL binding check should not trigger.
    // (Other rules may still catch this, but not the OpenGL binding rule.)
    let result = text.as_str().validate(TargetEnv::OpenGl4_5);
    if let Err(ref err) = result {
        assert!(
            !matches!(err, ValidationError::OpenGlBufferMissingBindingDecoration { .. }),
            "OpenGL binding check should not trigger without Block/BufferBlock"
        );
    }
}

#[test]
fn vulkan_env_does_not_trigger_opengl_binding_check() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%voidfn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %voidfn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    // This should pass Vulkan validation - missing Binding is caught by the
    // Vulkan descriptor binding rule, not by the OpenGL binding rule.
    // We just verify the error is NOT the OpenGL-specific one.
    let result = text.as_str().validate(TargetEnv::Vulkan1_0);
    if let Err(ref err) = result {
        assert!(
            !matches!(err, ValidationError::OpenGlBufferMissingBindingDecoration { .. }),
            "Vulkan env should not trigger OpenGL binding check"
        );
    }
}

#[test]
fn opengl_unreferenced_variable_does_not_need_binding() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "%void = OpTypeVoid",
        "%voidfn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %voidfn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    // Variable not in entry point interface - should not trigger OpenGL binding check
    let result = text.as_str().validate(TargetEnv::OpenGl4_5);
    if let Err(ref err) = result {
        assert!(
            !matches!(err, ValidationError::OpenGlBufferMissingBindingDecoration { .. }),
            "Unreferenced variable should not trigger OpenGL binding check"
        );
    }
}
