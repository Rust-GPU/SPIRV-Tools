use super::*;

#[test]
fn builtin_requires_variable_or_constant_targets() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "OpDecorate %main BuiltIn Position",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble BuiltIn on function");
    let expected = ValidationError::InvalidDecorationTargetKind {
        decoration: rspirv::spirv::Decoration::BuiltIn,
        target: Id::try_from(3).unwrap(),
        found: rspirv::spirv::Op::Function,
        expected: DecorationTargetKind::Variable,
    };
    for module in [
        MaybeValidModule::Text(text.as_str()),
        MaybeValidModule::Binary(binary.as_slice()),
    ] {
        let error = module
            .validate(TargetEnv::Universal1_6)
            .expect_err("BuiltIn must target variables/constants");
        assert_eq!(error, expected);
    }
}

#[test]
fn entry_point_function_must_reference_function_op() {
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %1 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        1,
        0x6e69616d, // "main" (null terminator implicit via padding)
        0,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        2,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::InvalidEntryPointTarget {
            target: Id::try_from(1).unwrap(),
            opcode: rspirv::spirv::Op::TypeVoid
        }
    );
}

#[test]
fn entry_point_interfaces_must_reference_variables() {
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
        op(6, 15), // OpEntryPoint Vertex %3 "main" %1 (interface %1 is not a variable)
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69616d,
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        2,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::InvalidEntryPointTarget {
            target: Id::try_from(1).unwrap(),
            opcode: rspirv::spirv::Op::TypeVoid
        }
    );
}

#[test]
fn entry_point_interfaces_cannot_reference_function_variables() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 0",
        "%ptr_fn = OpTypePointer Function %int",
        "%fn = OpTypeFunction %void",
        "OpEntryPoint Vertex %main \"main\" %func_var",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%func_var = OpVariable %ptr_fn Function",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassInvalid {
            entry_point: Id::try_from(5).unwrap(),
            interface: Id::try_from(6).unwrap(),
            storage_class: rspirv::spirv::StorageClass::Function,
        }
    );
}

#[test]
fn entry_point_interfaces_must_be_unique() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 0",
        "%ptr_in = OpTypePointer Input %int",
        "%fn = OpTypeFunction %void",
        "%var = OpVariable %ptr_in Input",
        "OpEntryPoint Vertex %main \"main\" %var %var",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateEntryPointInterface {
            entry_point: Id::try_from(6).unwrap(),
            interface: Id::try_from(5).unwrap(),
        }
    );
}

#[test]
fn duplicate_push_constant_interface_is_rejected_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%pc = OpTypeStruct %int",
        "OpDecorate %pc Block",
        "OpMemberDecorate %pc 0 Offset 0",
        "%ptr = OpTypePointer PushConstant %pc",
        "%pc0 = OpVariable %ptr PushConstant",
        "%pc1 = OpVariable %ptr PushConstant",
        "OpEntryPoint Vertex %main \"main\" %pc0 %pc1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassDuplicate {
            entry_point: Id::try_from(8).unwrap(),
            storage_class: rspirv::spirv::StorageClass::PushConstant,
        }
    );
}

#[test]
fn duplicate_push_constant_interface_is_allowed_outside_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%pc = OpTypeStruct %int",
        "OpDecorate %pc Block",
        "OpMemberDecorate %pc 0 Offset 0",
        "%ptr = OpTypePointer PushConstant %pc",
        "%pc0 = OpVariable %ptr PushConstant",
        "%pc1 = OpVariable %ptr PushConstant",
        "OpEntryPoint Vertex %main \"main\" %pc0 %pc1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap();
}

#[test]
fn duplicate_incoming_callable_data_interface_is_rejected_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpCapability RayTracingNV",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%data = OpTypeStruct %int",
        "%ptr = OpTypePointer IncomingCallableDataKHR %data",
        "%c0 = OpVariable %ptr IncomingCallableDataKHR",
        "%c1 = OpVariable %ptr IncomingCallableDataKHR",
        "OpEntryPoint CallableKHR %main \"main\" %c0 %c1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassDuplicate {
            entry_point: Id::try_from(8).unwrap(),
            storage_class: rspirv::spirv::StorageClass::IncomingCallableDataKHR,
        }
    );
}

#[test]
fn duplicate_incoming_ray_payload_interface_is_rejected_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpCapability RayTracingNV",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%payload = OpTypeStruct %int",
        "%ptr = OpTypePointer IncomingRayPayloadKHR %payload",
        "%p0 = OpVariable %ptr IncomingRayPayloadKHR",
        "%p1 = OpVariable %ptr IncomingRayPayloadKHR",
        "OpEntryPoint RayGenerationKHR %main \"main\" %p0 %p1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassDuplicate {
            entry_point: Id::try_from(8).unwrap(),
            storage_class: rspirv::spirv::StorageClass::IncomingRayPayloadKHR,
        }
    );
}

#[test]
fn non_ray_entry_point_rejects_ray_payload_interface() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpCapability RayTracingNV",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%payload = OpTypeStruct %int",
        "%ptr = OpTypePointer IncomingRayPayloadKHR %payload",
        "%p0 = OpVariable %ptr IncomingRayPayloadKHR",
        "OpEntryPoint Vertex %main \"main\" %p0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassInvalid {
            entry_point: Id::try_from(7).unwrap(),
            interface: Id::try_from(6).unwrap(),
            storage_class: rspirv::spirv::StorageClass::IncomingRayPayloadKHR,
        }
    );
}

#[test]
fn non_ray_entry_point_rejects_shader_record_buffer_interface() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpCapability RayTracingNV",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%ptr = OpTypePointer ShaderRecordBufferKHR %int",
        "%buf = OpVariable %ptr ShaderRecordBufferKHR",
        "OpEntryPoint Vertex %main \"main\" %buf",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassInvalid {
            entry_point: Id::try_from(6).unwrap(),
            interface: Id::try_from(5).unwrap(),
            storage_class: rspirv::spirv::StorageClass::ShaderRecordBufferKHR,
        }
    );
}

#[test]
fn compute_entry_point_rejects_output_interface() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%ptr = OpTypePointer Output %int",
        "%out = OpVariable %ptr Output",
        "OpEntryPoint GLCompute %main \"main\" %out",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassInvalid {
            entry_point: Id::try_from(6).unwrap(),
            interface: Id::try_from(5).unwrap(),
            storage_class: rspirv::spirv::StorageClass::Output,
        }
    );
}

#[test]
fn duplicate_interface_ids_are_rejected_per_entry_point() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%ptr = OpTypePointer Input %int",
        "%in = OpVariable %ptr Input",
        "OpEntryPoint Vertex %main \"main\" %in %in",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateEntryPointInterface {
            entry_point: Id::try_from(6).unwrap(),
            interface: Id::try_from(5).unwrap(),
        }
    );
}

#[test]
fn function_scope_interface_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%ptr = OpTypePointer Function %int",
        "OpEntryPoint Vertex %main \"main\" %local",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%local = OpVariable %ptr Function",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassInvalid {
            entry_point: Id::try_from(5).unwrap(),
            interface: Id::try_from(6).unwrap(),
            storage_class: rspirv::spirv::StorageClass::Function,
        }
    );
}

#[test]
fn patch_interface_requires_tessellation_execution_model() {
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%ptr = OpTypePointer Input %int",
        "%patch = OpVariable %ptr Input",
        "OpDecorate %patch Patch",
        "OpEntryPoint Vertex %main \"main\" %patch",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::PatchDecorationRequiresTessellation {
            execution_model: rspirv::spirv::ExecutionModel::Vertex
        }
    );
}

#[test]
fn patch_interface_requires_tessellation_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%ptr = OpTypePointer Input %int",
        "%patch = OpVariable %ptr Input",
        "OpDecorate %patch Patch",
        "OpEntryPoint TessellationControl %main \"main\" %patch",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert!(matches!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::EntryPoint,
            required_capability: rspirv::spirv::Capability::Tessellation,
            ..
        }
    ));
}

#[test]
fn duplicate_patch_locations_conflict() {
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint TessellationControl %main \"main\" %p0 %p1",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%ptr = OpTypePointer Input %float",
        "%p0 = OpVariable %ptr Input",
        "%p1 = OpVariable %ptr Input",
        "OpDecorate %p0 Patch",
        "OpDecorate %p0 Location 0",
        "OpDecorate %p1 Patch",
        "OpDecorate %p1 Location 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert!(matches!(
        error,
        ValidationError::EntryPointInterfaceLocationConflict {
            storage_class: rspirv::spirv::StorageClass::Input,
            location: 0,
            component: 0,
            ..
        }
    ));
}

#[test]
fn duplicate_hit_attribute_interface_is_rejected_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpCapability RayTracingNV",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%attr = OpTypeStruct %int",
        "%ptr = OpTypePointer HitAttributeKHR %attr",
        "%h0 = OpVariable %ptr HitAttributeKHR",
        "%h1 = OpVariable %ptr HitAttributeKHR",
        "OpEntryPoint ClosestHitKHR %main \"main\" %h0 %h1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceStorageClassDuplicate {
            entry_point: Id::try_from(8).unwrap(),
            storage_class: rspirv::spirv::StorageClass::HitAttributeKHR,
        }
    );
}

#[test]
fn bfloat16_interface_is_rejected_for_vulkan_input_output() {
    // Include StorageInputOutput16 to pass the small type capability check,
    // so we specifically test the BFloat16 encoding rejection for interface variables
    let text = [
        "OpCapability Shader",
        "OpCapability BFloat16TypeKHR",
        "OpCapability StorageInputOutput16",
        "OpExtension \"SPV_KHR_bfloat16\"",
        "OpExtension \"SPV_KHR_bfloat16_conversion\"",
        "OpExtension \"SPV_KHR_16bit_storage\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bf16 = OpTypeFloat 16 BFloat16KHR",
        "%ptr = OpTypePointer Input %bf16",
        "%var = OpVariable %ptr Input",
        "OpEntryPoint Vertex %main \"main\" %var",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    // ID layout: void=1, fn=2, bf16=3, ptr=4, var=5, main=6, entry=7
    assert_eq!(
        error,
        ValidationError::EntryPointInterfaceFloatEncodingInvalid {
            interface: Id::try_from(5).unwrap(),
            storage_class: rspirv::spirv::StorageClass::Input,
            encoding: rspirv::spirv::FPEncoding::BFloat16KHR,
        }
    );
}

#[test]
fn duplicate_entry_point_declarations_are_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "OpEntryPoint Vertex %main \"main\"",
        "OpEntryPoint Vertex %main \"main\"",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateEntryPoint {
            function: Id::try_from(3).unwrap(),
            execution_model: rspirv::spirv::ExecutionModel::Vertex,
        }
    );
}

#[test]
fn duplicate_input_locations_are_rejected_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%ptr = OpTypePointer Input %float",
        "%in0 = OpVariable %ptr Input",
        "%in1 = OpVariable %ptr Input",
        "OpDecorate %in0 Location 0",
        "OpDecorate %in1 Location 0",
        "OpEntryPoint Vertex %main \"main\" %in0 %in1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert!(matches!(
        error,
        ValidationError::EntryPointInterfaceLocationConflict {
            storage_class: rspirv::spirv::StorageClass::Input,
            location: 0,
            component: 0,
            ..
        }
    ));
}

#[test]
fn duplicate_input_locations_are_allowed_outside_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%ptr = OpTypePointer Input %float",
        "%in0 = OpVariable %ptr Input",
        "%in1 = OpVariable %ptr Input",
        "OpDecorate %in0 Location 0",
        "OpDecorate %in1 Location 0",
        "OpEntryPoint Vertex %main \"main\" %in0 %in1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .expect("Location overlap checks are Vulkan-specific");
}

#[test]
fn duplicate_patch_locations_are_rejected_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint TessellationControl %main \"main\" %in0 %in1",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%ptr = OpTypePointer Input %float",
        "%in0 = OpVariable %ptr Input",
        "%in1 = OpVariable %ptr Input",
        "OpDecorate %in0 Patch",
        "OpDecorate %in1 Patch",
        "OpDecorate %in0 Location 0",
        "OpDecorate %in1 Location 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert!(matches!(
        error,
        ValidationError::EntryPointInterfaceLocationConflict {
            storage_class: rspirv::spirv::StorageClass::Input,
            location: 0,
            component: 0,
            ..
        }
    ));
}

#[test]
fn component_values_must_be_within_range() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %in",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%ptr = OpTypePointer Input %float",
        "%in = OpVariable %ptr Input",
        "OpDecorate %in Component 5",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(err, ValidationError::ComponentOutOfRange { component: 5 });
}

#[test]
fn component_requires_location() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %in",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%ptr = OpTypePointer Input %float",
        "%in = OpVariable %ptr Input",
        "OpDecorate %in Component 1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert_eq!(err, ValidationError::ComponentMissingLocation);
}

#[test]
fn patch_struct_spill_conflicts_with_patch_neighbor() {
    // Patch variable consumes location 0 component 2..3 and spills into location 1 component 0..1.
    // Another patch variable starts at location 1 component 1, so they overlap in the patch domain.
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint TessellationControl %main \"main\" %p0 %p1",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec4 = OpTypeVector %float 4",
        "%ptr = OpTypePointer Input %vec4",
        "%p0 = OpVariable %ptr Input",
        "%p1 = OpVariable %ptr Input",
        "OpDecorate %p0 Patch",
        "OpDecorate %p0 Location 0",
        "OpDecorate %p0 Component 2", // spans loc0 comp2,3 and loc1 comp0,1
        "OpDecorate %p1 Patch",
        "OpDecorate %p1 Location 1",
        "OpDecorate %p1 Component 1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert!(matches!(
        err,
        ValidationError::EntryPointInterfaceLocationConflict {
            storage_class: rspirv::spirv::StorageClass::Input,
            location: 1,
            component: 1,
            ..
        }
    ));
}

#[test]
fn capabilities_after_functions_are_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let words = assemble_text(text.as_ref()).expect("assemble text");
    let reordered = reorder_opcode_to_end(words, rspirv::spirv::Op::Capability);
    let err = validate_module(&reordered, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn patch_and_non_patch_locations_use_separate_domains() {
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint TessellationControl %main \"main\" %patch %nonpatch",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%ptr = OpTypePointer Input %float",
        "%ptr_vec = OpTypePointer Input %vec2",
        "%patch = OpVariable %ptr Input",
        "%nonpatch = OpVariable %ptr Input",
        "OpDecorate %patch Patch",
        "OpDecorate %patch Location 0",
        "OpDecorate %nonpatch Location 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .expect("Patch and non-Patch locations use separate domains");
}

#[test]
fn patch_and_non_patch_locations_separate_even_when_components_spill() {
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint TessellationControl %main \"main\" %patch %nonpatch",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%ptr = OpTypePointer Input %float",
        "%ptr_vec = OpTypePointer Input %vec2",
        "%patch = OpVariable %ptr_vec Input",
        "%nonpatch = OpVariable %ptr Input",
        "OpDecorate %patch Patch",
        "OpDecorate %patch Location 0",
        "OpDecorate %patch Component 3", // occupies loc0 comp3 and spills into loc1 comp0
        "OpDecorate %nonpatch Location 1",
        "OpDecorate %nonpatch Component 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .expect("Patch and non-Patch domains remain separate even when components spill");
}

#[test]
fn non_patch_spill_does_not_conflict_with_patch() {
    // Non-Patch variable spills from location 0 component 2 into location 1 component 0.
    // Patch variable at location 1 component 0 is in a separate domain and should be allowed.
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint TessellationControl %main \"main\" %patch %nonpatch",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%ptr = OpTypePointer Input %float",
        "%ptr_vec = OpTypePointer Input %vec2",
        "%patch = OpVariable %ptr Input",
        "%nonpatch = OpVariable %ptr_vec Input",
        "OpDecorate %patch Patch",
        "OpDecorate %patch Location 1",
        "OpDecorate %patch Component 0",
        "OpDecorate %nonpatch Location 0",
        "OpDecorate %nonpatch Component 2", // spills into location 1 component 0
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    MaybeValidModule::Text(&text)
            .validate(TargetEnv::Vulkan1_2)
            .expect("Patch and non-Patch domains are distinct even when non-Patch spills into the next location");
}

#[test]
fn geometry_output_vertices_execution_mode_is_accepted() {
    let text = [
        "OpCapability Shader",
        "OpCapability Geometry",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Geometry %main \"main\"",
        "OpExecutionMode %main Triangles",
        "OpExecutionMode %main OutputTriangleStrip",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .expect("Geometry OutputVertices should not trigger operand capability errors");
}

#[test]
fn validate_module_reports_missing_memory_model_without_other_globals() {
    // A module that declares only capabilities should still fail for a missing memory model.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        1,          // bound
        0,          // schema
        op(2, 17),  // OpCapability Shader
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(error, ValidationError::MissingMemoryModel);
}
