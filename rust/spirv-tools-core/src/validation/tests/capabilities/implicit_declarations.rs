use super::super::*;

#[test]
fn function_variants_implicitly_declares_spec_conditional() {
    // FunctionVariantsINTEL implicitly declares SpecConditionalINTEL per the spec.
    let text = [
        "OpCapability Kernel",
        "OpCapability Linkage",
        "OpCapability FunctionVariantsINTEL",
        "OpExtension \"SPV_INTEL_function_variants\"",
        "OpMemoryModel Logical OpenCL",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("FunctionVariantsINTEL implicitly declares SpecConditionalINTEL");
}

#[test]
fn variable_pointers_implicitly_declares_storage_buffer_capability() {
    // VariablePointers implicitly declares VariablePointersStorageBuffer per the spec.
    let without_dep = [
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
    without_dep
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("VariablePointers implicitly declares VariablePointersStorageBuffer");

    let with_dep = [
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
    with_dep
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("explicit + implicit declaration both allowed");
}

#[test]
fn shader_implicitly_declares_matrix() {
    // Shader implicitly declares Matrix per the spec's capability table.
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
        .expect("Shader implicitly declares Matrix");
}

#[test]
fn image_buffer_implicitly_declares_sampled_buffer() {
    // ImageBuffer implicitly declares SampledBuffer per the spec.
    let without_dep = [
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
    without_dep
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ImageBuffer implicitly declares SampledBuffer");

    let with_dep = [
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
    with_dep
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("explicit + implicit declaration both allowed");
}

#[test]
fn sampled_cube_array_implicitly_declares_shader() {
    // SampledCubeArray implicitly declares Shader per the spec.
    let without_shader = [
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
    without_shader
        .as_str()
        .validate(TargetEnv::Universal1_2)
        .expect("SampledCubeArray implicitly declares Shader");

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
        .expect("explicit + implicit declaration both allowed");
}

#[test]
fn image_ms_array_implicitly_declares_shader() {
    // ImageMSArray implicitly declares Shader per the spec.
    let without_shader = [
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
    without_shader
        .as_str()
        .validate(TargetEnv::Universal1_2)
        .expect("ImageMSArray implicitly declares Shader");

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
        .expect("explicit + implicit declaration both allowed");
}

#[test]
fn ray_tracing_implicitly_declares_shader() {
    // RayTracingKHR implicitly declares Shader per the spec.
    let without_shader = [
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
    without_shader
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingKHR implicitly declares Shader");

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
        .expect("explicit + implicit declaration both allowed");
}

#[test]
fn group_non_uniform_arithmetic_implicitly_declares_group_non_uniform() {
    // GroupNonUniformArithmetic implicitly declares GroupNonUniform per the spec.
    // Both with and without explicit GroupNonUniform should succeed.
    let without_base = [
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
    without_base
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("GroupNonUniformArithmetic implicitly declares GroupNonUniform");

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
        .expect("explicit + implicit declaration both allowed");
}

#[test]
fn device_enqueue_implicitly_declares_kernel() {
    // DeviceEnqueue implicitly declares Kernel per the spec.
    let without_kernel = [
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
    without_kernel
        .as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("DeviceEnqueue implicitly declares Kernel");

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
        .expect("explicit + implicit declaration both allowed");
}

#[test]
fn opencl_image_read_write_implicitly_declares_image_basic() {
    // ImageReadWrite implicitly declares ImageBasic per the spec.
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
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("ImageReadWrite implicitly declares ImageBasic");
}
