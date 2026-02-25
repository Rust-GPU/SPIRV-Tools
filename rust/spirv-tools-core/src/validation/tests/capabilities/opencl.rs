use super::super::*;

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
fn opencl_environment_accepts_opencl_extension() {
    let text = opencl_module_with_extension("SPV_KHR_opencl_enqueue");
    text.validate(TargetEnv::OpenCl2_2)
        .expect("OpenCL targets should accept OpenCL-specific extensions");
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
