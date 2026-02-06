use super::*;

#[test]
fn sample_interpolation_requires_sample_rate_shading_capability() {
    let missing_cap = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var Sample
OpDecorate %var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%ptr = OpTypePointer Input %f32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(missing_cap, TargetEnv::Vulkan1_2)
        .expect_err("Sample interpolation requires SampleRateShading capability");
    assert!(
        matches!(
            err,
            ValidationError::DecorationRequiresCapability {
                decoration: rspirv::spirv::Decoration::Sample,
                capability: rspirv::spirv::Capability::SampleRateShading
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected SampleRateShading capability error, got {err:?}"
    );

    let with_cap = r#"
OpCapability Shader
OpCapability SampleRateShading
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var Sample
OpDecorate %var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%ptr = OpTypePointer Input %f32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(with_cap, TargetEnv::Vulkan1_2)
        .expect("Sample interpolation allowed when SampleRateShading is declared");
}

#[test]
fn interpolation_decorations_allowed_on_input_and_output() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %in %out",
        "OpExecutionMode %main OriginUpperLeft",
        "OpDecorate %in NoPerspective",
        "OpDecorate %out Centroid",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%ptr_in = OpTypePointer Input %f32",
        "%ptr_out = OpTypePointer Output %f32",
        "%in = OpVariable %ptr_in Input",
        "%out = OpVariable %ptr_out Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect("interpolation decorations should be accepted on Input/Output");
}

#[test]
fn interpolation_decorations_are_exclusive_within_each_class() {
    let base_conflict = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var Flat
OpDecorate %var NoPerspective
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(base_conflict, TargetEnv::Vulkan1_2)
        .expect_err("only one base interpolation decoration is permitted");
    assert_eq!(
        err,
        ValidationError::InterpolationDecorationConflict {
            decoration: rspirv::spirv::Decoration::NoPerspective,
            existing: rspirv::spirv::Decoration::Flat
        }
    );

    let centroid_sample_conflict = r#"
OpCapability Shader
OpCapability SampleRateShading
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var Centroid
OpDecorate %var Sample
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(centroid_sample_conflict, TargetEnv::Vulkan1_2)
        .expect_err("Centroid/Sample/Patch are exclusive");
    assert_eq!(
        err,
        ValidationError::InterpolationDecorationConflict {
            decoration: rspirv::spirv::Decoration::Sample,
            existing: rspirv::spirv::Decoration::Centroid
        }
    );

    let flat_sample_conflict = r#"
OpCapability Shader
OpCapability SampleRateShading
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var Flat
OpDecorate %var Sample
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(flat_sample_conflict, TargetEnv::Vulkan1_2)
        .expect_err("Flat may not combine with Sample/Centroid");
    assert_eq!(
        err,
        ValidationError::InterpolationDecorationConflict {
            decoration: rspirv::spirv::Decoration::Sample,
            existing: rspirv::spirv::Decoration::Flat
        }
    );

    let sample_then_flat_conflict = r#"
OpCapability Shader
OpCapability SampleRateShading
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var Sample
OpDecorate %var Flat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(sample_then_flat_conflict, TargetEnv::Vulkan1_2)
        .expect_err("Flat may not combine with Sample/Centroid");
    assert_eq!(
        err,
        ValidationError::InterpolationDecorationConflict {
            decoration: rspirv::spirv::Decoration::Flat,
            existing: rspirv::spirv::Decoration::Sample
        }
    );
}

#[test]
fn interpolation_decorations_require_fragment_execution_model() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %in",
        "OpDecorate %in Flat",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%ptr = OpTypePointer Input %f32",
        "%in = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("interpolation decorations require a Fragment entry point");
    assert_eq!(
        err,
        ValidationError::InterpolationDecorationRequiresFragment {
            decoration: rspirv::spirv::Decoration::Flat
        }
    );
}

#[test]
fn location_conflicts_with_builtin() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn Position",
        "OpDecorate %var Location 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%ptr = OpTypePointer Output %vec4",
        "%var = OpVariable %ptr Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect_err("Location must not be used with BuiltIn");
    assert_eq!(err, ValidationError::LocationConflictsWithBuiltIn);
}

#[test]
fn builtin_requires_appropriate_storage_class() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn Position",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%ptr = OpTypePointer Uniform %vec4",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect_err("BuiltIn Position should not be allowed on Uniform");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInStorageClass {
            builtin: rspirv::spirv::BuiltIn::Position,
            storage_class: rspirv::spirv::StorageClass::Uniform
        }
    );

    // WorkgroupSize is only allowed on Workgroup storage.
    let workgroup_text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %var BuiltIn WorkgroupSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec3 = OpTypeVector %u32 3
%ptr = OpTypePointer Input %vec3
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(workgroup_text, TargetEnv::Universal1_5)
        .expect_err("WorkgroupSize must follow target-kind rules");
    assert!(matches!(
        err,
        ValidationError::InvalidDecorationTargetKind { decoration, .. }
            if decoration == rspirv::spirv::Decoration::BuiltIn
    ));
}

#[test]
fn fragment_only_builtin_requires_fragment_entry_model() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn FragCoord",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%ptr = OpTypePointer Input %vec4",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect_err("FragCoord requires a Fragment entry point");
    assert_eq!(
        err,
        ValidationError::BuiltInRequiresFragment {
            builtin: rspirv::spirv::BuiltIn::FragCoord
        }
    );
}

#[test]
fn fragment_only_builtin_allows_fragment_entry_model() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %var",
        "OpExecutionMode %main OriginUpperLeft",
        "OpDecorate %var BuiltIn FragDepth",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%ptr = OpTypePointer Output %f32",
        "%var = OpVariable %ptr Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect("fragment-only BuiltIns should be accepted for fragment entry points");
}

#[test]
fn barycentric_builtin_requires_fragment_entry_model() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentBarycentricKHR",
        "OpExtension \"SPV_KHR_fragment_shader_barycentric\"",
        "OpExtension \"SPV_NV_fragment_shader_barycentric\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn BaryCoordKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec2 = OpTypeVector %f32 2",
        "%ptr = OpTypePointer Input %vec2",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("BaryCoord requires a Fragment entry point");
    assert_eq!(
        err,
        ValidationError::BuiltInRequiresFragment {
            builtin: rspirv::spirv::BuiltIn::BaryCoordKHR
        }
    );
}

#[test]
fn barycentric_builtin_allows_fragment_entry_model() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentBarycentricKHR",
        "OpExtension \"SPV_KHR_fragment_shader_barycentric\"",
        "OpExtension \"SPV_NV_fragment_shader_barycentric\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %var",
        "OpExecutionMode %main OriginUpperLeft",
        "OpDecorate %var BuiltIn BaryCoordNoPerspKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec3 = OpTypeVector %f32 3",
        "%ptr = OpTypePointer Input %vec3",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("barycentric BuiltIns should be accepted for fragment entry points");
}

#[test]
fn ray_builtins_require_ray_execution_models() {
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpCapability RayTracingNV",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpExtension \"SPV_NV_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn LaunchIdKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%vec3 = OpTypeVector %u32 3",
        "%ptr = OpTypePointer Input %vec3",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("ray built-ins require ray tracing execution models");
    assert!(matches!(
        err,
        ValidationError::BuiltInRequiresExecutionModel {
            builtin: rspirv::spirv::BuiltIn::LaunchIdKHR,
            ..
        }
    ));

    let ok = r#"
OpCapability Shader
OpCapability RayTracingKHR
OpCapability RayTracingNV
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_NV_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %main "main" %var
OpDecorate %var BuiltIn LaunchSizeKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec3 = OpTypeVector %u32 3
%ptr = OpTypePointer Input %vec3
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(ok, TargetEnv::Vulkan1_2)
        .expect("ray built-ins should be accepted for ray tracing entry points");
}

#[test]
fn vertex_id_is_disallowed_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn VertexId",
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
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("VertexId is not allowed in Vulkan");
    assert_eq!(
        err,
        ValidationError::BuiltInDisallowedForEnv {
            builtin: rspirv::spirv::BuiltIn::VertexId,
            env: TargetEnv::Vulkan1_2
        }
    );
}

#[test]
fn shading_rate_builtins_are_vulkan_only() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentShadingRateKHR",
        "OpExtension \"SPV_KHR_fragment_shading_rate\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %var",
        "OpExecutionMode %main OriginUpperLeft",
        "OpDecorate %var BuiltIn ShadingRateKHR",
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
    let err = assemble_and_validate_with_env(text, TargetEnv::OpenCl2_2)
        .expect_err("fragment shading rate built-ins are Vulkan-only");
    assert_eq!(
        err,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_fragment_shading_rate"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn primitive_shading_rate_builtin_is_vulkan_only() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentShadingRateKHR",
        "OpExtension \"SPV_KHR_fragment_shading_rate\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn PrimitiveShadingRateKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Output %u32",
        "%var = OpVariable %ptr Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::OpenCl2_2)
        .expect_err("primitive shading rate built-ins are Vulkan-only");
    assert_eq!(
        err,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_KHR_fragment_shading_rate"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn mesh_builtins_are_vulkan_only() {
    let text = [
        "OpCapability Shader",
        "OpCapability MeshShadingEXT",
        "OpExtension \"SPV_EXT_mesh_shader\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint MeshEXT %main \"main\" %var",
        "OpDecorate %var BuiltIn PrimitivePointIndicesEXT",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%arr3 = OpTypeArray %u32 %c3",
        "%c3 = OpConstant %u32 3",
        "%ptr = OpTypePointer Output %arr3",
        "%var = OpVariable %ptr Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::OpenCl2_2)
        .expect_err("mesh built-ins are Vulkan-only");
    assert_eq!(
        err,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("SPV_EXT_mesh_shader"),
            env: TargetEnv::OpenCl2_2
        }
    );
}

#[test]
fn instance_id_is_disallowed_in_vulkan() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn InstanceId",
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
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("InstanceId is not allowed in Vulkan");
    assert_eq!(
        err,
        ValidationError::BuiltInDisallowedForEnv {
            builtin: rspirv::spirv::BuiltIn::InstanceId,
            env: TargetEnv::Vulkan1_2
        }
    );
}

#[test]
fn compute_workgroup_builtins_require_compute_entry_point() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn GlobalInvocationId",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%vec3 = OpTypeVector %u32 3",
        "%ptr = OpTypePointer Input %vec3",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("workgroup built-ins require compute entry points");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresExecutionModel {
                builtin: rspirv::spirv::BuiltIn::GlobalInvocationId,
                ..
            }
        ),
        "expected BuiltInRequiresExecutionModel for GlobalInvocationId, got {err:?}"
    );

    let ok = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %var
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %var BuiltIn WorkgroupId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec3 = OpTypeVector %u32 3
%ptr = OpTypePointer Input %vec3
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(ok, TargetEnv::Vulkan1_2)
        .expect("workgroup built-ins should be accepted for compute entry points");

    let subgroup_vertex = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %var
OpDecorate %var BuiltIn SubgroupId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(subgroup_vertex, TargetEnv::Vulkan1_2)
        .expect_err("subgroup built-ins require compute/kernel entry models");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresExecutionModel {
                builtin: rspirv::spirv::BuiltIn::SubgroupId,
                allowed: ref models
            } if models.contains(&rspirv::spirv::ExecutionModel::GLCompute)
                && models.contains(&rspirv::spirv::ExecutionModel::Kernel)
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected subgroup built-ins to require compute or kernel, got {err:?}"
    );

    let missing_group_capability = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn NumSubgroups
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(missing_group_capability, TargetEnv::Universal1_6)
        .expect_err("subgroup built-ins require GroupNonUniform capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::NumSubgroups,
                capability: rspirv::spirv::Capability::GroupNonUniform
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected GroupNonUniform capability error, got {err:?}"
    );

    let missing_ballot_capability = r#"
OpCapability Shader
OpCapability Kernel
OpCapability GroupNonUniform
OpCapability SubgroupBallotKHR
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupEqMask
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec4 = OpTypeVector %u32 4
%ptr = OpTypePointer Input %vec4
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(missing_ballot_capability, TargetEnv::Universal1_6)
        .expect_err("subgroup mask built-ins require GroupNonUniformBallot capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::SubgroupEqMask,
                capability: rspirv::spirv::Capability::GroupNonUniformBallot
            } | ValidationError::DisallowedCapabilityMissingExtension { .. }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected GroupNonUniformBallot capability error, got {err:?}"
    );

    let missing_device_enqueue = r#"
OpCapability Shader
OpCapability Kernel
OpCapability GroupNonUniform
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn NumEnqueuedSubgroups
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(missing_device_enqueue, TargetEnv::Universal1_6)
        .expect_err("NumEnqueuedSubgroups requires DeviceEnqueue capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::NumEnqueuedSubgroups,
                capability: rspirv::spirv::Capability::DeviceEnqueue
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected DeviceEnqueue capability error, got {err:?}"
    );

    let device_enqueue_ok = r#"
OpCapability Shader
OpCapability Kernel
OpCapability GroupNonUniform
OpCapability DeviceEnqueue
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn NumEnqueuedSubgroups
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(device_enqueue_ok, TargetEnv::Universal1_6)
        .expect("NumEnqueuedSubgroups allowed with DeviceEnqueue capability");

    let subgroup_size_missing_capability = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err =
        assemble_and_validate_with_env(subgroup_size_missing_capability, TargetEnv::Universal1_6)
            .expect_err("SubgroupSize requires GroupNonUniform capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::SubgroupSize,
                capability: rspirv::spirv::Capability::GroupNonUniform
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected GroupNonUniform capability error, got {err:?}"
    );

    let subgroup_max_size_requires_kernel_model = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint GLCompute %main "main" %var
OpDecorate %var BuiltIn SubgroupMaxSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(
        subgroup_max_size_requires_kernel_model,
        TargetEnv::Universal1_6,
    )
    .expect_err("SubgroupMaxSize is kernel-only");
    assert!(matches!(
        err,
        ValidationError::BuiltInRequiresExecutionModel {
            builtin: rspirv::spirv::BuiltIn::SubgroupMaxSize,
            ..
        }
    ));

    let subgroup_max_size_requires_kernel_capability = r#"
OpCapability Shader
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupMaxSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(
        subgroup_max_size_requires_kernel_capability,
        TargetEnv::Universal1_6,
    )
    .expect_err("SubgroupMaxSize requires Kernel capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::SubgroupMaxSize,
                capability: rspirv::spirv::Capability::Kernel
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. })
    );

    let subgroup_max_size_ok = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupMaxSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(subgroup_max_size_ok, TargetEnv::Universal1_6)
        .expect("SubgroupMaxSize allowed for Kernel execution model with Kernel capability");

    let subgroup_local_invocation_missing_capability = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupLocalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(
        subgroup_local_invocation_missing_capability,
        TargetEnv::Universal1_6,
    )
    .expect_err("SubgroupLocalInvocationId requires GroupNonUniform capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::SubgroupLocalInvocationId,
                capability: rspirv::spirv::Capability::GroupNonUniform
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected GroupNonUniform capability error, got {err:?}"
    );

    let subgroup_id_missing_capability = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err =
        assemble_and_validate_with_env(subgroup_id_missing_capability, TargetEnv::Universal1_6)
            .expect_err("SubgroupId requires GroupNonUniform capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::SubgroupId,
                capability: rspirv::spirv::Capability::GroupNonUniform
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected GroupNonUniform capability error, got {err:?}"
    );

    let subgroup_ok = r#"
OpCapability Shader
OpCapability Kernel
OpCapability GroupNonUniform
OpCapability GroupNonUniformBallot
OpCapability SubgroupBallotKHR
OpExtension "SPV_KHR_shader_ballot"
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var %mask
OpDecorate %var BuiltIn NumSubgroups
OpDecorate %mask BuiltIn SubgroupGeMask
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec4 = OpTypeVector %u32 4
%ptr_u32 = OpTypePointer Input %u32
%ptr_vec = OpTypePointer Input %vec4
%var = OpVariable %ptr_u32 Input
%mask = OpVariable %ptr_vec Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(subgroup_ok, TargetEnv::Universal1_6)
        .expect("subgroup built-ins should be accepted when capabilities are declared");

    // Test that SPV_KHR_shader_ballot extension is not required in SPIR-V 1.3+ when
    // GroupNonUniformBallot capability is used (the extension was promoted to core in 1.3)
    let subgroup_mask_without_extension_spirv13 = r#"
OpCapability Shader
OpCapability Kernel
OpCapability GroupNonUniform
OpCapability GroupNonUniformBallot
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %mask
OpDecorate %mask BuiltIn SubgroupEqMask
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec4 = OpTypeVector %u32 4
%ptr_vec = OpTypePointer Input %vec4
%mask = OpVariable %ptr_vec Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(
        subgroup_mask_without_extension_spirv13,
        TargetEnv::Universal1_3,
    )
    .expect(
        "SubgroupEqMask should be accepted without SPV_KHR_shader_ballot in SPIR-V 1.3+ \
             when GroupNonUniformBallot capability is present",
    );

    let subgroup_size_missing_capability = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err =
        assemble_and_validate_with_env(subgroup_size_missing_capability, TargetEnv::Universal1_6)
            .expect_err("SubgroupSize requires GroupNonUniform capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::SubgroupSize,
                capability: rspirv::spirv::Capability::GroupNonUniform
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected GroupNonUniform capability error, got {err:?}"
    );

    let subgroup_local_invocation_missing_capability = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupLocalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(
        subgroup_local_invocation_missing_capability,
        TargetEnv::Universal1_6,
    )
    .expect_err("SubgroupLocalInvocationId requires GroupNonUniform capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::SubgroupLocalInvocationId,
                capability: rspirv::spirv::Capability::GroupNonUniform
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected GroupNonUniform capability error, got {err:?}"
    );

    let subgroup_max_size_kernel_only = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint GLCompute %main "main" %var
OpDecorate %var BuiltIn SubgroupMaxSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err =
        assemble_and_validate_with_env(subgroup_max_size_kernel_only, TargetEnv::Universal1_6)
            .expect_err("SubgroupMaxSize is kernel-only and requires Kernel capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresExecutionModel {
                builtin: rspirv::spirv::BuiltIn::SubgroupMaxSize,
                ..
            }
        ) || matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::SubgroupMaxSize,
                capability: rspirv::spirv::Capability::Kernel
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected kernel-only SubgroupMaxSize error, got {err:?}"
    );

    let subgroup_max_size_kernel_ok = r#"
OpCapability Shader
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn SubgroupMaxSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(subgroup_max_size_kernel_ok, TargetEnv::Universal1_6)
        .expect("SubgroupMaxSize allowed for Kernel execution model with Kernel capability");

    let num_enqueued_requires_kernel = r#"
OpCapability Shader
OpCapability Kernel
OpCapability DeviceEnqueue
OpMemoryModel Logical OpenCL
OpEntryPoint GLCompute %main "main" %var
OpDecorate %var BuiltIn NumEnqueuedSubgroups
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(num_enqueued_requires_kernel, TargetEnv::Universal1_6)
        .expect_err("NumEnqueuedSubgroups requires Kernel execution model");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresExecutionModel {
                builtin: rspirv::spirv::BuiltIn::NumEnqueuedSubgroups,
                ref allowed
            } if allowed == &vec![rspirv::spirv::ExecutionModel::Kernel]
        ) || matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::NumEnqueuedSubgroups,
                capability: rspirv::spirv::Capability::DeviceEnqueue
            }
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected kernel-only or capability error, got {err:?}"
    );

    let num_enqueued_kernel_ok = r#"
OpCapability Shader
OpCapability Kernel
OpCapability DeviceEnqueue
OpMemoryModel Logical OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn NumEnqueuedSubgroups
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(num_enqueued_kernel_ok, TargetEnv::Universal1_6)
        .expect("NumEnqueuedSubgroups allowed for Kernel execution model with DeviceEnqueue");
}

#[test]
fn vertex_index_and_instance_index_require_vertex_model() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %vid %iid",
        "OpExecutionMode %main OriginUpperLeft",
        "OpDecorate %vid BuiltIn VertexIndex",
        "OpDecorate %iid BuiltIn InstanceIndex",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Input %u32",
        "%vid = OpVariable %ptr Input",
        "%iid = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("VertexIndex/InstanceIndex require vertex execution model");
    assert!(matches!(
        err,
        ValidationError::BuiltInRequiresExecutionModel {
            builtin: rspirv::spirv::BuiltIn::VertexIndex,
            ..
        }
    ));

    let ok = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %vid %iid
OpDecorate %vid BuiltIn VertexIndex
OpDecorate %iid BuiltIn InstanceIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%vid = OpVariable %ptr Input
%iid = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(ok, TargetEnv::Vulkan1_2)
        .expect("VertexIndex/InstanceIndex allowed for vertex entry points");
}

#[test]
fn view_index_disallowed_for_compute() {
    let bad = r#"
OpCapability Shader
OpCapability Geometry
OpCapability MultiView
OpExtension "SPV_KHR_multiview"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %var
OpDecorate %var BuiltIn ViewIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(bad, TargetEnv::Vulkan1_2)
        .expect_err("ViewIndex is not allowed with GLCompute");
    assert!(matches!(
        err,
        ValidationError::BuiltInRequiresExecutionModel {
            builtin: rspirv::spirv::BuiltIn::ViewIndex,
            allowed
        } if allowed.contains(&rspirv::spirv::ExecutionModel::Vertex)
    ));

    let ok = r#"
OpCapability Shader
OpCapability Geometry
OpCapability MultiView
OpExtension "SPV_KHR_multiview"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %var
OpExecutionMode %main Triangles
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpDecorate %var BuiltIn ViewIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(ok, TargetEnv::Vulkan1_2)
        .expect("ViewIndex allowed in geometry");
}

#[test]
fn view_index_requires_multiview_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn ViewIndex",
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
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("ViewIndex requires MultiView capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::ViewIndex,
                capability: rspirv::spirv::Capability::MultiView
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected MultiView capability error, got {err:?}"
    );
}

#[test]
fn device_index_requires_device_group_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability MultiView",
        "OpExtension \"SPV_KHR_multiview\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn DeviceIndex",
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
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("DeviceIndex requires DeviceGroup capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::DeviceIndex,
                capability: rspirv::spirv::Capability::DeviceGroup
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected DeviceGroup capability error, got {err:?}"
    );

    let ok = r#"
OpCapability Shader
OpCapability MultiView
OpCapability DeviceGroup
OpExtension "SPV_KHR_multiview"
OpExtension "SPV_KHR_device_group"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %var
OpDecorate %var BuiltIn DeviceIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(ok, TargetEnv::Vulkan1_2)
        .expect("DeviceIndex allowed when DeviceGroup declared");
}

#[test]
fn shader_sm_builtins_require_capability() {
    let nv_missing = r#"
OpCapability Shader
OpExtension "SPV_NV_shader_sm_builtins"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %var
OpDecorate %var BuiltIn WarpsPerSMNV
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(nv_missing, TargetEnv::Vulkan1_2)
        .expect_err("SM built-ins require ShaderSMBuiltins or ShaderCoreBuiltinsARM");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::WarpsPerSMNV,
                capability: rspirv::spirv::Capability::ShaderSMBuiltinsNV
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected SM built-in capability error, got {err:?}"
    );

    let nv_ok = r#"
OpCapability Shader
OpCapability ShaderSMBuiltinsNV
OpExtension "SPV_NV_shader_sm_builtins"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %var
OpDecorate %var BuiltIn SMCountNV
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(nv_ok, TargetEnv::Vulkan1_2)
        .expect("SM built-ins allowed with ShaderSMBuiltins");

    let arm_missing = r#"
OpCapability Shader
OpExtension "SPV_ARM_core_builtins"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %var
OpDecorate %var BuiltIn CoreIDARM
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(arm_missing, TargetEnv::Vulkan1_2)
        .expect_err("ARM core built-ins require ShaderCoreBuiltinsARM capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::CoreIDARM,
                capability: rspirv::spirv::Capability::CoreBuiltinsARM
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected ARM core capability error, got {err:?}"
    );

    let arm_ok = r#"
OpCapability Shader
OpCapability CoreBuiltinsARM
OpExtension "SPV_ARM_core_builtins"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %var
OpDecorate %var BuiltIn CoreIDARM
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(arm_ok, TargetEnv::Vulkan1_2)
        .expect("ARM core built-ins allowed when CoreBuiltinsARM capability declared");
}

#[test]
fn builtin_types_are_checked_for_common_inputs() {
    let front_facing_wrong_type = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn FrontFacing
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(front_facing_wrong_type, TargetEnv::Vulkan1_2)
        .expect_err("FrontFacing must be a bool");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInType {
            builtin: rspirv::spirv::BuiltIn::FrontFacing,
            expected: "bool"
        }
    );

    let global_invocation_wrong_len = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %var
OpDecorate %var BuiltIn GlobalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec2 = OpTypeVector %u32 2
%ptr = OpTypePointer Input %vec2
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(global_invocation_wrong_len, TargetEnv::Vulkan1_2)
        .expect_err("GlobalInvocationId must be a vec3<i32>");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInType {
            builtin: rspirv::spirv::BuiltIn::GlobalInvocationId,
            expected: "vec3<i32>"
        }
    );

    let barycoord_wrong_width = r#"
OpCapability Shader
OpCapability FragmentBarycentricKHR
OpExtension "SPV_KHR_fragment_shader_barycentric"
OpExtension "SPV_NV_fragment_shader_barycentric"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn BaryCoordKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%vec2 = OpTypeVector %f32 2
%ptr = OpTypePointer Input %vec2
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(barycoord_wrong_width, TargetEnv::Vulkan1_2)
        .expect_err("BaryCoordKHR must be vec3<f32>");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInType {
            builtin: rspirv::spirv::BuiltIn::BaryCoordKHR,
            expected: "vec3<f32>"
        }
    );
}

#[test]
fn shading_rate_builtins_require_fragment_execution_model() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentShadingRateKHR",
        "OpExtension \"SPV_KHR_fragment_shading_rate\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn ShadingRateKHR",
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
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("shading rate built-ins require fragment entry points");
    assert!(matches!(
        err,
        ValidationError::BuiltInRequiresExecutionModel {
            builtin: rspirv::spirv::BuiltIn::ShadingRateKHR,
            ..
        }
    ));

    let ok = r#"
OpCapability Shader
OpCapability FragmentShadingRateKHR
OpCapability Geometry
OpExtension "SPV_KHR_fragment_shading_rate"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %var
OpExecutionMode %main Triangles
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpDecorate %var BuiltIn PrimitiveShadingRateKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Output %u32
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(ok, TargetEnv::Vulkan1_2)
        .expect("shading rate built-ins should be accepted for fragment entry points");

    let bad_type = r#"
OpCapability Shader
OpCapability FragmentShadingRateKHR
OpExtension "SPV_KHR_fragment_shading_rate"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn ShadingRateKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec2 = OpTypeVector %u32 2
%ptr = OpTypePointer Input %vec2
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(bad_type, TargetEnv::Vulkan1_2)
        .expect_err("shading rate built-ins require 32-bit int scalars");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInType {
            builtin: rspirv::spirv::BuiltIn::ShadingRateKHR,
            expected: "i32"
        }
    );

    let wrong_storage = r#"
OpCapability Shader
OpCapability FragmentShadingRateKHR
OpExtension "SPV_KHR_fragment_shading_rate"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn ShadingRateKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Output %u32
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(wrong_storage, TargetEnv::Vulkan1_2)
        .expect_err("ShadingRateKHR must be Input storage");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInStorageClass {
            builtin: rspirv::spirv::BuiltIn::ShadingRateKHR,
            storage_class: rspirv::spirv::StorageClass::Output
        }
    );

    let wrong_storage_primitive = r#"
OpCapability Shader
OpCapability FragmentShadingRateKHR
OpExtension "SPV_KHR_fragment_shading_rate"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn PrimitiveShadingRateKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(wrong_storage_primitive, TargetEnv::Vulkan1_2)
        .expect_err("PrimitiveShadingRateKHR must be Output storage");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInStorageClass {
            builtin: rspirv::spirv::BuiltIn::PrimitiveShadingRateKHR,
            storage_class: rspirv::spirv::StorageClass::Input
        }
    );

    let wrong_model = r#"
OpCapability Shader
OpCapability FragmentShadingRateKHR
OpExtension "SPV_KHR_fragment_shading_rate"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn PrimitiveShadingRateKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Output %u32
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(wrong_model, TargetEnv::Vulkan1_2)
        .expect_err("PrimitiveShadingRateKHR limited to vertex/geometry/mesh");
    assert!(matches!(
        err,
        ValidationError::BuiltInRequiresExecutionModel {
            builtin: rspirv::spirv::BuiltIn::PrimitiveShadingRateKHR,
            ..
        }
    ));

    let allowed_model = r#"
OpCapability Shader
OpCapability Geometry
OpCapability FragmentShadingRateKHR
OpExtension "SPV_KHR_fragment_shading_rate"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %var
OpExecutionMode %main Triangles
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpDecorate %var BuiltIn PrimitiveShadingRateKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Output %u32
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(allowed_model, TargetEnv::Vulkan1_2)
        .expect("PrimitiveShadingRateKHR should allow geometry entry points");

    let disallowed_model = r#"
OpCapability Shader
OpCapability FragmentShadingRateKHR
OpExtension "SPV_KHR_fragment_shading_rate"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn PrimitiveShadingRateKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Output %u32
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(disallowed_model, TargetEnv::Vulkan1_2)
        .expect_err("PrimitiveShadingRateKHR should reject fragment-only modules");
    assert!(matches!(
        err,
        ValidationError::BuiltInRequiresExecutionModel {
            builtin: rspirv::spirv::BuiltIn::PrimitiveShadingRateKHR,
            ..
        }
    ));

    let missing_capability = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn ShadingRateKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(missing_capability, TargetEnv::Vulkan1_2)
        .expect_err("ShadingRateKHR requires FragmentShadingRateKHR capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::ShadingRateKHR,
                ..
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected capability error, got {err:?}"
    );

    let sample_without_capability = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var Sample
OpDecorate %var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%vec2 = OpTypeVector %f32 2
%ptr = OpTypePointer Input %vec2
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(sample_without_capability, TargetEnv::Vulkan1_2)
        .expect_err("Sample decoration requires SampleRateShading capability");
    assert!(
        matches!(
            err,
            ValidationError::DecorationRequiresCapability {
                decoration: rspirv::spirv::Decoration::Sample,
                capability: rspirv::spirv::Capability::SampleRateShading
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected SampleRateShading capability error, got {err:?}"
    );

    let sample_with_capability = r#"
OpCapability Shader
OpCapability SampleRateShading
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var Sample
OpDecorate %var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%vec2 = OpTypeVector %f32 2
%ptr = OpTypePointer Input %vec2
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(sample_with_capability, TargetEnv::Vulkan1_2)
        .expect("Sample decoration allowed when SampleRateShading capability declared");

    let missing_mesh_capability = r#"
OpCapability Shader
OpExtension "SPV_EXT_mesh_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint MeshEXT %main "main" %var
OpDecorate %var BuiltIn PrimitivePointIndicesEXT
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%const3 = OpConstant %u32 3
%arr = OpTypeArray %u32 %const3
%ptr = OpTypePointer Output %arr
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(missing_mesh_capability, TargetEnv::Vulkan1_2)
        .expect_err("mesh built-ins require MeshShadingEXT capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::PrimitivePointIndicesEXT,
                ..
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected mesh capability error, got {err:?}"
    );

    let missing_ray_capability = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %main "main" %var
OpDecorate %var BuiltIn LaunchIdKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec3 = OpTypeVector %u32 3
%ptr = OpTypePointer Input %vec3
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(missing_ray_capability, TargetEnv::Vulkan1_2)
        .expect_err("ray tracing built-ins require ray tracing capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::LaunchIdKHR,
                ..
            } | ValidationError::MissingOperandCapability { .. }
        ),
        "expected ray capability error, got {err:?}"
    );
}

#[test]
fn tess_level_builtins_require_patch_decoration() {
    let missing_patch = r#"
OpCapability Shader
OpCapability Tessellation
OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationEvaluation %main "main" %var
OpExecutionMode %main Triangles
OpDecorate %var BuiltIn TessLevelOuter
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%uint_4 = OpConstant %u32 4
%arr = OpTypeArray %float %uint_4
%ptr = OpTypePointer Input %arr
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(missing_patch, TargetEnv::Vulkan1_2)
        .expect_err("tessellation level built-ins require Patch decoration");
    assert_eq!(
        err,
        ValidationError::BuiltInRequiresPatchDecoration {
            builtin: rspirv::spirv::BuiltIn::TessLevelOuter
        }
    );

    let with_patch = r#"
OpCapability Shader
OpCapability Tessellation
OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationEvaluation %main "main" %var
OpExecutionMode %main Triangles
OpDecorate %var BuiltIn TessLevelOuter
OpDecorate %var Patch
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%uint_4 = OpConstant %u32 4
%arr = OpTypeArray %float %uint_4
%ptr = OpTypePointer Input %arr
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(with_patch, TargetEnv::Vulkan1_2)
        .expect("tessellation level built-ins should require Patch decoration only");
}

#[test]
fn mesh_builtins_require_mesh_execution_models() {
    let text = [
        "OpCapability Shader",
        "OpCapability Geometry",
        "OpCapability MeshShadingEXT",
        "OpExtension \"SPV_EXT_mesh_shader\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpDecorate %var BuiltIn PrimitivePointIndicesEXT",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%arr = OpTypeArray %u32 %uint_3",
        "%uint_3 = OpConstant %u32 3",
        "%ptr = OpTypePointer Output %arr",
        "%var = OpVariable %ptr Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("mesh built-ins require mesh execution models");
    assert!(matches!(
        err,
        ValidationError::BuiltInRequiresExecutionModel {
            builtin: rspirv::spirv::BuiltIn::PrimitivePointIndicesEXT,
            ..
        }
    ));

    let ok = r#"
OpCapability Shader
OpCapability MeshShadingEXT
OpExtension "SPV_EXT_mesh_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint MeshEXT %main "main" %var
OpExecutionMode %main OutputTrianglesEXT
OpExecutionMode %main OutputVertices 3
OpExecutionMode %main OutputPrimitivesEXT 1
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %var BuiltIn CullPrimitiveEXT
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%ptr = OpTypePointer Output %bool
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(ok, TargetEnv::Vulkan1_2)
        .expect("mesh built-ins should be accepted for mesh entry points");

    let bad_storage = r#"
OpCapability Shader
OpCapability MeshShadingEXT
OpExtension "SPV_EXT_mesh_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint MeshEXT %main "main" %var
OpExecutionMode %main OutputTrianglesEXT
OpExecutionMode %main OutputVertices 3
OpExecutionMode %main OutputPrimitivesEXT 1
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %var BuiltIn PrimitiveTriangleIndicesEXT
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(bad_storage, TargetEnv::Vulkan1_2)
        .expect_err("mesh built-ins must use Output storage");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInStorageClass {
            builtin: rspirv::spirv::BuiltIn::PrimitiveTriangleIndicesEXT,
            storage_class: rspirv::spirv::StorageClass::Input
        }
    );
}

#[test]
fn kernel_only_builtins_require_kernel_execution_model() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\" %var",
        "OpDecorate %var BuiltIn WorkDim",
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
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("WorkDim is a Kernel-only built-in");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresExecutionModel {
                builtin: rspirv::spirv::BuiltIn::WorkDim,
                allowed: ref models
            } if *models == vec![rspirv::spirv::ExecutionModel::Kernel]
        ) || matches!(err, ValidationError::MissingOperandCapability { .. }),
        "expected WorkDim to require Kernel model, got {err:?}"
    );

    let kernel_ok = r#"
OpCapability Addresses
OpCapability Kernel
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn GlobalSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec3 = OpTypeVector %u32 3
%ptr = OpTypePointer Input %vec3
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(kernel_ok, TargetEnv::Universal1_6)
        .expect("Kernel execution model should allow kernel-only built-ins");
}

#[test]
fn kernel_execution_model_allows_compute_builtins() {
    let text = [
        "OpCapability Addresses",
        "OpCapability Kernel",
        "OpMemoryModel Physical32 OpenCL",
        "OpEntryPoint Kernel %main \"main\" %var",
        "OpDecorate %var BuiltIn GlobalInvocationId",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%vec3 = OpTypeVector %u32 3",
        "%ptr = OpTypePointer Input %vec3",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect("Kernel execution model should allow compute built-ins");
}

#[test]
fn kernel_only_builtins_require_kernel_capability() {
    let text = [
        "OpCapability Addresses",
        "OpMemoryModel Physical32 OpenCL",
        "OpEntryPoint Kernel %main \"main\" %var",
        "OpDecorate %var BuiltIn GlobalOffset",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%vec3 = OpTypeVector %u32 3",
        "%ptr = OpTypePointer Input %vec3",
        "%var = OpVariable %ptr Input",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("Kernel-only built-ins require Kernel capability");
    assert!(
        matches!(
            err,
            ValidationError::BuiltInRequiresCapability {
                builtin: rspirv::spirv::BuiltIn::GlobalOffset,
                capability: rspirv::spirv::Capability::Kernel
            } | ValidationError::MissingOperandCapability { .. }
                | ValidationError::MissingRequiredCapability { .. }
        ),
        "expected Kernel capability error, got {err:?}"
    );

    let ok = r#"
OpCapability Addresses
OpCapability Kernel
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %main "main" %var
OpDecorate %var BuiltIn GlobalOffset
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%vec3 = OpTypeVector %u32 3
%ptr = OpTypePointer Input %vec3
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(ok, TargetEnv::Universal1_6)
        .expect("Kernel capability should allow kernel-only built-ins");
}

#[test]
fn barycentric_builtin_requires_input_storage() {
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentBarycentricKHR",
        "OpExtension \"SPV_KHR_fragment_shader_barycentric\"",
        "OpExtension \"SPV_NV_fragment_shader_barycentric\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %var",
        "OpExecutionMode %main OriginUpperLeft",
        "OpDecorate %var BuiltIn BaryCoordKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec2 = OpTypeVector %f32 2",
        "%ptr_out = OpTypePointer Output %vec2",
        "%var = OpVariable %ptr_out Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("barycentric built-ins must target Input storage");
    assert_eq!(
        err,
        ValidationError::InvalidBuiltInStorageClass {
            builtin: rspirv::spirv::BuiltIn::BaryCoordKHR,
            storage_class: rspirv::spirv::StorageClass::Output
        }
    );

    let input_text = r#"
OpCapability Shader
OpCapability FragmentBarycentricKHR
OpExtension "SPV_KHR_fragment_shader_barycentric"
OpExtension "SPV_NV_fragment_shader_barycentric"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn BaryCoordKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%vec3 = OpTypeVector %f32 3
%ptr_in = OpTypePointer Input %vec3
%var = OpVariable %ptr_in Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(input_text, TargetEnv::Vulkan1_2)
        .expect("barycentric built-ins should be accepted on Input storage");
}

#[test]
fn uniform_8bit_with_block_allows_uniform_and_storage_buffer_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int8",
        "OpCapability VariablePointers",
        "OpCapability VariablePointersStorageBuffer",
        "OpCapability UniformAndStorageBuffer8BitAccess",
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
        "%ptr = OpTypePointer Uniform %buf",
        "%var = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect("UniformAndStorageBuffer8BitAccess with Block should allow uniform 8-bit");
}

#[test]
fn stencil_ref_replacing_ext_on_vertex_rejected() {
    // StencilRefReplacingEXT is Fragment-only; using it on a Vertex entry point should fail
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op};

    let mut module = rspirv::dr::Module::new();
    module.header = Some(rspirv::dr::ModuleHeader::new(10));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Shader)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::StencilExportEXT)],
    ));
    module.extensions.push(Instruction::new(
        Op::Extension,
        None,
        None,
        vec![Operand::LiteralString(
            "SPV_EXT_shader_stencil_export".to_string(),
        )],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(AddressingModel::Logical),
            Operand::MemoryModel(MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(ExecutionModel::Vertex),
            Operand::IdRef(3),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(3),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::StencilRefReplacingEXT),
        ],
    ));
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(2),
        None,
        vec![Operand::IdRef(1)],
    ));
    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(3),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(2),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(4), None, vec![]));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert!(
        matches!(
            error,
            ValidationError::ExecutionModeRequiresExecutionModel { .. }
        ),
        "Expected ExecutionModeRequiresExecutionModel, got: {error:?}"
    );
}

#[test]
fn stencil_ref_replacing_ext_on_fragment_accepted() {
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op};

    let mut module = rspirv::dr::Module::new();
    module.header = Some(rspirv::dr::ModuleHeader::new(10));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Shader)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::StencilExportEXT)],
    ));
    module.extensions.push(Instruction::new(
        Op::Extension,
        None,
        None,
        vec![Operand::LiteralString(
            "SPV_EXT_shader_stencil_export".to_string(),
        )],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(AddressingModel::Logical),
            Operand::MemoryModel(MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(ExecutionModel::Fragment),
            Operand::IdRef(3),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(3),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::OriginUpperLeft),
        ],
    ));
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(3),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::StencilRefReplacingEXT),
        ],
    ));
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(2),
        None,
        vec![Operand::IdRef(1)],
    ));
    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(3),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(2),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(4), None, vec![]));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("StencilRefReplacingEXT on Fragment should be accepted");
}

#[test]
fn post_depth_coverage_on_vertex_rejected() {
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op};

    let mut module = rspirv::dr::Module::new();
    module.header = Some(rspirv::dr::ModuleHeader::new(10));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Shader)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::SampleMaskPostDepthCoverage)],
    ));
    module.extensions.push(Instruction::new(
        Op::Extension,
        None,
        None,
        vec![Operand::LiteralString(
            "SPV_KHR_post_depth_coverage".to_string(),
        )],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(AddressingModel::Logical),
            Operand::MemoryModel(MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(ExecutionModel::Vertex),
            Operand::IdRef(3),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(3),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::PostDepthCoverage),
        ],
    ));
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(2),
        None,
        vec![Operand::IdRef(1)],
    ));
    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(3),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(2),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(4), None, vec![]));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert!(
        matches!(
            error,
            ValidationError::ExecutionModeRequiresExecutionModel { .. }
        ),
        "Expected ExecutionModeRequiresExecutionModel, got: {error:?}"
    );
}

#[test]
fn lifetime_start_valid_function_pointer() {
    // OpLifetimeStart with a Function storage class pointer and size 0 should pass
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op};

    let mut module = rspirv::dr::Module::new();
    module.header = Some(rspirv::dr::ModuleHeader::new(20));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Kernel)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Addresses)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(AddressingModel::Physical32),
            Operand::MemoryModel(MemoryModel::OpenCL),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(ExecutionModel::Kernel),
            Operand::IdRef(5),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    // %1 = OpTypeVoid
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    // %2 = OpTypeInt 32 0
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(2),
        None,
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    // %3 = OpTypePointer Function %2
    module.types_global_values.push(Instruction::new(
        Op::TypePointer,
        Some(3),
        None,
        vec![
            Operand::StorageClass(rspirv::spirv::StorageClass::Function),
            Operand::IdRef(2),
        ],
    ));
    // %4 = OpTypeFunction %1
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(4),
        None,
        vec![Operand::IdRef(1)],
    ));
    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(5),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(4),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(6), None, vec![]));
    // %7 = OpVariable %3 Function
    block.instructions.push(Instruction::new(
        Op::Variable,
        Some(3),
        Some(7),
        vec![Operand::StorageClass(rspirv::spirv::StorageClass::Function)],
    ));
    // OpLifetimeStart %7 0
    block.instructions.push(Instruction::new(
        Op::LifetimeStart,
        None,
        None,
        vec![Operand::IdRef(7), Operand::LiteralBit32(0)],
    ));
    // OpLifetimeStop %7 0
    block.instructions.push(Instruction::new(
        Op::LifetimeStop,
        None,
        None,
        vec![Operand::IdRef(7), Operand::LiteralBit32(0)],
    ));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("Valid OpLifetimeStart/Stop with Function pointer should pass");
}

// ============================================================================
// Commit 1: Built-in execution model bug fixes
// ============================================================================

#[test]
fn subgroup_size_valid_in_fragment() {
    // SubgroupSize and SubgroupLocalInvocationId are valid in all stages, not just compute
    let text = r#"
OpCapability Shader
OpCapability GroupNonUniform
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn SubgroupSize
OpDecorate %var Flat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("SubgroupSize should be valid in Fragment shader");
}

#[test]
fn subgroup_local_invocation_id_valid_in_fragment() {
    let text = r#"
OpCapability Shader
OpCapability GroupNonUniform
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn SubgroupLocalInvocationId
OpDecorate %var Flat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("SubgroupLocalInvocationId should be valid in Fragment shader");
}

#[test]
fn local_invocation_index_valid_in_mesh_ext() {
    // Compute-like builtins should also be valid in Mesh/Task shaders
    let text = r#"
OpCapability Shader
OpCapability MeshShadingEXT
OpExtension "SPV_EXT_mesh_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint MeshEXT %main "main" %var
OpExecutionMode %main LocalSize 1 1 1
OpExecutionMode %main OutputVertices 1
OpExecutionMode %main OutputPrimitivesEXT 1
OpExecutionMode %main OutputTrianglesEXT
OpDecorate %var BuiltIn LocalInvocationIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("LocalInvocationIndex should be valid in MeshEXT shader");
}

#[test]
fn num_subgroups_valid_in_task_ext() {
    let text = r#"
OpCapability Shader
OpCapability MeshShadingEXT
OpCapability GroupNonUniform
OpExtension "SPV_EXT_mesh_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint TaskEXT %main "main" %var
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %var BuiltIn NumSubgroups
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("NumSubgroups should be valid in TaskEXT shader");
}

#[test]
fn compute_only_builtin_invalid_in_vertex() {
    // Compute-only builtins should still be rejected in Vertex
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %var
OpDecorate %var BuiltIn LocalInvocationIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("LocalInvocationIndex should not be valid in Vertex shader");
    assert!(
        matches!(err, ValidationError::BuiltInRequiresExecutionModel { .. }),
        "expected BuiltInRequiresExecutionModel, got {err:?}"
    );
}

#[test]
fn patch_vertices_valid_in_tess_evaluation() {
    // PatchVertices should be valid in both TessControl and TessEvaluation
    let text = r#"
OpCapability Shader
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationEvaluation %main "main" %var
OpExecutionMode %main Triangles
OpDecorate %var BuiltIn PatchVertices
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("PatchVertices should be valid in TessellationEvaluation");
}

#[test]
fn patch_vertices_invalid_in_fragment() {
    let text = r#"
OpCapability Shader
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn PatchVertices
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("PatchVertices should not be valid in Fragment shader");
    assert!(
        matches!(err, ValidationError::BuiltInRequiresExecutionModel { .. }),
        "expected BuiltInRequiresExecutionModel, got {err:?}"
    );
}

#[test]
fn tess_level_outer_valid_in_tess_control() {
    // TessLevelOuter should be valid in both TessControl (output) and TessEvaluation (input)
    let text = r#"
OpCapability Shader
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationControl %main "main" %var
OpExecutionMode %main OutputVertices 3
OpDecorate %var BuiltIn TessLevelOuter
OpDecorate %var Patch
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%u32_4 = OpConstant %u32 4
%arr = OpTypeArray %f32 %u32_4
%ptr = OpTypePointer Output %arr
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("TessLevelOuter should be valid in TessellationControl");
}

#[test]
fn tess_level_inner_valid_in_tess_control() {
    let text = r#"
OpCapability Shader
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationControl %main "main" %var
OpExecutionMode %main OutputVertices 3
OpDecorate %var BuiltIn TessLevelInner
OpDecorate %var Patch
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%u32_2 = OpConstant %u32 2
%arr = OpTypeArray %f32 %u32_2
%ptr = OpTypePointer Output %arr
%var = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("TessLevelInner should be valid in TessellationControl");
}

#[test]
fn view_index_valid_in_fragment() {
    let text = r#"
OpCapability Shader
OpCapability MultiView
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft
OpDecorate %var BuiltIn ViewIndex
OpDecorate %var Flat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("ViewIndex should be valid in Fragment shader");
}

#[test]
fn view_index_invalid_in_compute() {
    let text = r#"
OpCapability Shader
OpCapability MultiView
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %var
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %var BuiltIn ViewIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%var = OpVariable %ptr Input
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("ViewIndex should not be valid in GLCompute shader");
    assert!(
        matches!(err, ValidationError::BuiltInRequiresExecutionModel { .. }),
        "expected BuiltInRequiresExecutionModel, got {err:?}"
    );
}
