use super::super::*;

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
