use super::super::*;

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
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
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
fn mesh_shader_extension_is_vulkan_only() {
    let text = module_with_extension("SPV_EXT_mesh_shader");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("Mesh shader extension should be accepted for Vulkan targets");
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
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
fn device_group_extension_is_vulkan_only() {
    assert_vulkan_only_extension("SPV_KHR_device_group");
}
