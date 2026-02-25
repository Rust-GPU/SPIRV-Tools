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

// ============================================================================
// Comprehensive implicit capability declaration tests
// Generated from spirv.core.grammar.json — one test per capability with
// implicit declarations, verifying that declaring only the capability
// (without explicitly declaring its dependencies) passes validation.
// ============================================================================

#[test]
fn atomic_storage_implicitly_declares_deps() {
    // AtomicStorage implicitly declares: Shader
    let text = [
        "OpCapability AtomicStorage",
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
        .expect("AtomicStorage implicitly declares Shader");
}

#[test]
fn atomic_storage_ops_implicitly_declares_deps() {
    // AtomicStorageOps implicitly declares: AtomicStorage
    let text = [
        "OpCapability AtomicStorageOps",
        "OpExtension \"SPV_KHR_shader_atomic_counter_ops\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("AtomicStorageOps implicitly declares AtomicStorage");
}

#[test]
fn b_float16_cooperative_matrix_khr_implicitly_declares_deps() {
    // BFloat16CooperativeMatrixKHR implicitly declares: BFloat16TypeKHR, CooperativeMatrixKHR
    // CooperativeMatrixKHR + Shader requires VulkanMemoryModel capability and VulkanKHR memory model.
    let text = [
        "OpCapability BFloat16CooperativeMatrixKHR",
        "OpCapability Shader",
        "OpCapability VulkanMemoryModel",
        "OpExtension \"SPV_KHR_bfloat16\"",
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
    text.as_str().validate(TargetEnv::Universal1_6).expect(
        "BFloat16CooperativeMatrixKHR implicitly declares BFloat16TypeKHR, CooperativeMatrixKHR",
    );
}

#[test]
fn b_float16_dot_product_khr_implicitly_declares_deps() {
    // BFloat16DotProductKHR implicitly declares: BFloat16TypeKHR
    let text = [
        "OpCapability BFloat16DotProductKHR",
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_bfloat16\"",
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
        .expect("BFloat16DotProductKHR implicitly declares BFloat16TypeKHR");
}

#[test]
fn clip_distance_implicitly_declares_deps() {
    // ClipDistance implicitly declares: Shader
    let text = [
        "OpCapability ClipDistance",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ClipDistance implicitly declares Shader");
}

#[test]
fn compute_derivative_group_linear_khr_implicitly_declares_deps() {
    // ComputeDerivativeGroupLinearKHR implicitly declares: Shader
    let text = [
        "OpCapability ComputeDerivativeGroupLinearKHR",
        "OpExtension \"SPV_NV_compute_shader_derivatives\"",
        "OpExtension \"SPV_KHR_compute_shader_derivatives\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ComputeDerivativeGroupLinearKHR implicitly declares Shader");
}

#[test]
fn compute_derivative_group_quads_khr_implicitly_declares_deps() {
    // ComputeDerivativeGroupQuadsKHR implicitly declares: Shader
    let text = [
        "OpCapability ComputeDerivativeGroupQuadsKHR",
        "OpExtension \"SPV_NV_compute_shader_derivatives\"",
        "OpExtension \"SPV_KHR_compute_shader_derivatives\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ComputeDerivativeGroupQuadsKHR implicitly declares Shader");
}

#[test]
fn cooperative_matrix_conversion_qcom_implicitly_declares_deps() {
    // CooperativeMatrixConversionQCOM implicitly declares: CooperativeMatrixKHR
    // CooperativeMatrixKHR + Shader requires VulkanMemoryModel capability and VulkanKHR memory model.
    let text = [
        "OpCapability CooperativeMatrixConversionQCOM",
        "OpCapability Shader",
        "OpCapability VulkanMemoryModel",
        "OpExtension \"SPV_QCOM_cooperative_matrix_conversion\"",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("CooperativeMatrixConversionQCOM implicitly declares CooperativeMatrixKHR");
}

#[test]
fn cooperative_matrix_nv_implicitly_declares_deps() {
    // CooperativeMatrixNV implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("CooperativeMatrixNV implicitly declares Shader");
}

#[test]
fn cull_distance_implicitly_declares_deps() {
    // CullDistance implicitly declares: Shader
    let text = [
        "OpCapability CullDistance",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("CullDistance implicitly declares Shader");
}

#[test]
fn demote_to_helper_invocation_implicitly_declares_deps() {
    // DemoteToHelperInvocation implicitly declares: Shader
    let text = [
        "OpCapability DemoteToHelperInvocation",
        "OpExtension \"SPV_EXT_demote_to_helper_invocation\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("DemoteToHelperInvocation implicitly declares Shader");
}

#[test]
fn derivative_control_implicitly_declares_deps() {
    // DerivativeControl implicitly declares: Shader
    let text = [
        "OpCapability DerivativeControl",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("DerivativeControl implicitly declares Shader");
}

#[test]
#[ignore = "rspirv does not yet recognize DescriptorHeapEXT as a capability name"]
fn descriptor_heap_ext_implicitly_declares_deps() {
    // DescriptorHeapEXT implicitly declares: UntypedPointersKHR
    let text = [
        "OpCapability DescriptorHeapEXT",
        "OpCapability Shader",
        "OpExtension \"SPV_EXT_descriptor_heap\"",
        "OpExtension \"SPV_KHR_untyped_pointers\"",
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
        .expect("DescriptorHeapEXT implicitly declares UntypedPointersKHR");
}

#[test]
fn device_enqueue_implicitly_declares_deps() {
    // DeviceEnqueue implicitly declares: Kernel
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
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("DeviceEnqueue implicitly declares Kernel");
}

#[test]
fn displacement_micromap_nv_implicitly_declares_deps() {
    // DisplacementMicromapNV implicitly declares: Shader
    let text = [
        "OpCapability DisplacementMicromapNV",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("DisplacementMicromapNV implicitly declares Shader");
}

#[test]
fn dot_product_input4x8_bit_implicitly_declares_deps() {
    // DotProductInput4x8Bit implicitly declares: Int8
    let text = [
        "OpCapability DotProductInput4x8Bit",
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_integer_dot_product\"",
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
        .expect("DotProductInput4x8Bit implicitly declares Int8");
}

#[test]
fn draw_parameters_implicitly_declares_deps() {
    // DrawParameters implicitly declares: Shader
    let text = [
        "OpCapability DrawParameters",
        "OpExtension \"SPV_KHR_shader_draw_parameters\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("DrawParameters implicitly declares Shader");
}

#[test]
fn fp_fast_math_mode_intel_implicitly_declares_deps() {
    // FPFastMathModeINTEL implicitly declares: Kernel
    let text = [
        "OpCapability FPFastMathModeINTEL",
        "OpCapability Addresses",
        "OpExtension \"SPV_INTEL_fp_fast_math_mode\"",
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
        .expect("FPFastMathModeINTEL implicitly declares Kernel");
}

#[test]
fn fpga_cluster_attributes_v2_altera_implicitly_declares_deps() {
    // FPGAClusterAttributesV2ALTERA implicitly declares: FPGAClusterAttributesALTERA
    let text = [
        "OpCapability FPGAClusterAttributesV2ALTERA",
        "OpCapability Shader",
        "OpExtension \"SPV_ALTERA_fpga_cluster_attributes\"",
        "OpExtension \"SPV_INTEL_fpga_cluster_attributes\"",
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
        .expect("FPGAClusterAttributesV2ALTERA implicitly declares FPGAClusterAttributesALTERA");
}

#[test]
fn fpga_kernel_attributesv2_intel_implicitly_declares_deps() {
    // FPGAKernelAttributesv2INTEL implicitly declares: FPGAKernelAttributesINTEL
    let text = [
        "OpCapability FPGAKernelAttributesv2INTEL",
        "OpCapability Shader",
        "OpExtension \"SPV_INTEL_kernel_attributes\"",
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
        .expect("FPGAKernelAttributesv2INTEL implicitly declares FPGAKernelAttributesINTEL");
}

#[test]
fn float16_buffer_implicitly_declares_deps() {
    // Float16Buffer implicitly declares: Kernel
    let text = [
        "OpCapability Float16Buffer",
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
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("Float16Buffer implicitly declares Kernel");
}

#[test]
fn float16_image_amd_implicitly_declares_deps() {
    // Float16ImageAMD implicitly declares: Shader
    let text = [
        "OpCapability Float16ImageAMD",
        "OpExtension \"SPV_AMD_gpu_shader_half_float_fetch\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("Float16ImageAMD implicitly declares Shader");
}

#[test]
fn float8_cooperative_matrix_ext_implicitly_declares_deps() {
    // Float8CooperativeMatrixEXT implicitly declares: Float8EXT, CooperativeMatrixKHR
    // CooperativeMatrixKHR + Shader requires VulkanMemoryModel capability and VulkanKHR memory model.
    let text = [
        "OpCapability Float8CooperativeMatrixEXT",
        "OpCapability Shader",
        "OpCapability VulkanMemoryModel",
        "OpExtension \"SPV_EXT_float8\"",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("Float8CooperativeMatrixEXT implicitly declares Float8EXT, CooperativeMatrixKHR");
}

#[test]
fn fragment_density_ext_implicitly_declares_deps() {
    // FragmentDensityEXT implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentDensityEXT implicitly declares Shader");
}

#[test]
fn fragment_fully_covered_ext_implicitly_declares_deps() {
    // FragmentFullyCoveredEXT implicitly declares: Shader
    let text = [
        "OpCapability FragmentFullyCoveredEXT",
        "OpExtension \"SPV_EXT_fragment_fully_covered\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentFullyCoveredEXT implicitly declares Shader");
}

#[test]
fn fragment_mask_amd_implicitly_declares_deps() {
    // FragmentMaskAMD implicitly declares: Shader
    let text = [
        "OpCapability FragmentMaskAMD",
        "OpExtension \"SPV_AMD_shader_fragment_mask\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentMaskAMD implicitly declares Shader");
}

#[test]
fn fragment_shader_pixel_interlock_ext_implicitly_declares_deps() {
    // FragmentShaderPixelInterlockEXT implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentShaderPixelInterlockEXT implicitly declares Shader");
}

#[test]
fn fragment_shader_sample_interlock_ext_implicitly_declares_deps() {
    // FragmentShaderSampleInterlockEXT implicitly declares: Shader
    let text = [
        "OpCapability FragmentShaderSampleInterlockEXT",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentShaderSampleInterlockEXT implicitly declares Shader");
}

#[test]
fn fragment_shader_shading_rate_interlock_ext_implicitly_declares_deps() {
    // FragmentShaderShadingRateInterlockEXT implicitly declares: Shader
    let text = [
        "OpCapability FragmentShaderShadingRateInterlockEXT",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentShaderShadingRateInterlockEXT implicitly declares Shader");
}

#[test]
fn fragment_shading_rate_khr_implicitly_declares_deps() {
    // FragmentShadingRateKHR implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("FragmentShadingRateKHR implicitly declares Shader");
}

#[test]
fn function_variants_intel_implicitly_declares_deps() {
    // FunctionVariantsINTEL implicitly declares: SpecConditionalINTEL
    let text = [
        "OpCapability FunctionVariantsINTEL",
        "OpCapability Shader",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("FunctionVariantsINTEL implicitly declares SpecConditionalINTEL");
}

#[test]
fn generic_pointer_implicitly_declares_deps() {
    // GenericPointer implicitly declares: Addresses
    let text = [
        "OpCapability GenericPointer",
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
        .expect("GenericPointer implicitly declares Addresses");
}

#[test]
fn geometry_implicitly_declares_deps() {
    // Geometry implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("Geometry implicitly declares Shader");
}

#[test]
fn geometry_point_size_implicitly_declares_deps() {
    // GeometryPointSize implicitly declares: Geometry
    let text = [
        "OpCapability GeometryPointSize",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("GeometryPointSize implicitly declares Geometry");
}

#[test]
fn geometry_shader_passthrough_nv_implicitly_declares_deps() {
    // GeometryShaderPassthroughNV implicitly declares: Geometry
    let text = [
        "OpCapability GeometryShaderPassthroughNV",
        "OpExtension \"SPV_NV_geometry_shader_passthrough\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("GeometryShaderPassthroughNV implicitly declares Geometry");
}

#[test]
fn geometry_streams_implicitly_declares_deps() {
    // GeometryStreams implicitly declares: Geometry
    let text = [
        "OpCapability GeometryStreams",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("GeometryStreams implicitly declares Geometry");
}

#[test]
fn group_non_uniform_arithmetic_implicitly_declares_deps() {
    // GroupNonUniformArithmetic implicitly declares: GroupNonUniform
    let text = [
        "OpCapability GroupNonUniformArithmetic",
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
        .expect("GroupNonUniformArithmetic implicitly declares GroupNonUniform");
}

#[test]
fn group_non_uniform_ballot_implicitly_declares_deps() {
    // GroupNonUniformBallot implicitly declares: GroupNonUniform
    let text = [
        "OpCapability GroupNonUniformBallot",
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
        .expect("GroupNonUniformBallot implicitly declares GroupNonUniform");
}

#[test]
fn group_non_uniform_clustered_implicitly_declares_deps() {
    // GroupNonUniformClustered implicitly declares: GroupNonUniform
    let text = [
        "OpCapability GroupNonUniformClustered",
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
        .expect("GroupNonUniformClustered implicitly declares GroupNonUniform");
}

#[test]
fn group_non_uniform_quad_implicitly_declares_deps() {
    // GroupNonUniformQuad implicitly declares: GroupNonUniform
    let text = [
        "OpCapability GroupNonUniformQuad",
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
        .expect("GroupNonUniformQuad implicitly declares GroupNonUniform");
}

#[test]
fn group_non_uniform_rotate_khr_implicitly_declares_deps() {
    // GroupNonUniformRotateKHR implicitly declares: GroupNonUniform
    let text = [
        "OpCapability GroupNonUniformRotateKHR",
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_subgroup_rotate\"",
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
        .expect("GroupNonUniformRotateKHR implicitly declares GroupNonUniform");
}

#[test]
fn group_non_uniform_shuffle_implicitly_declares_deps() {
    // GroupNonUniformShuffle implicitly declares: GroupNonUniform
    let text = [
        "OpCapability GroupNonUniformShuffle",
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
        .expect("GroupNonUniformShuffle implicitly declares GroupNonUniform");
}

#[test]
fn group_non_uniform_shuffle_relative_implicitly_declares_deps() {
    // GroupNonUniformShuffleRelative implicitly declares: GroupNonUniform
    let text = [
        "OpCapability GroupNonUniformShuffleRelative",
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
        .expect("GroupNonUniformShuffleRelative implicitly declares GroupNonUniform");
}

#[test]
fn group_non_uniform_vote_implicitly_declares_deps() {
    // GroupNonUniformVote implicitly declares: GroupNonUniform
    let text = [
        "OpCapability GroupNonUniformVote",
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
        .expect("GroupNonUniformVote implicitly declares GroupNonUniform");
}

#[test]
fn image1_d_implicitly_declares_deps() {
    // Image1D implicitly declares: Sampled1D
    let text = [
        "OpCapability Image1D",
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
        .expect("Image1D implicitly declares Sampled1D");
}

#[test]
fn image_basic_implicitly_declares_deps() {
    // ImageBasic implicitly declares: Kernel
    let text = [
        "OpCapability ImageBasic",
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
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("ImageBasic implicitly declares Kernel");
}

#[test]
fn image_buffer_implicitly_declares_deps() {
    // ImageBuffer implicitly declares: SampledBuffer
    let text = [
        "OpCapability ImageBuffer",
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
        .expect("ImageBuffer implicitly declares SampledBuffer");
}

#[test]
fn image_cube_array_implicitly_declares_deps() {
    // ImageCubeArray implicitly declares: SampledCubeArray
    let text = [
        "OpCapability ImageCubeArray",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ImageCubeArray implicitly declares SampledCubeArray");
}

#[test]
fn image_gather_bias_lod_amd_implicitly_declares_deps() {
    // ImageGatherBiasLodAMD implicitly declares: Shader
    let text = [
        "OpCapability ImageGatherBiasLodAMD",
        "OpExtension \"SPV_AMD_texture_gather_bias_lod\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ImageGatherBiasLodAMD implicitly declares Shader");
}

#[test]
fn image_gather_extended_implicitly_declares_deps() {
    // ImageGatherExtended implicitly declares: Shader
    let text = [
        "OpCapability ImageGatherExtended",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ImageGatherExtended implicitly declares Shader");
}

#[test]
fn image_ms_array_implicitly_declares_deps() {
    // ImageMSArray implicitly declares: Shader
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ImageMSArray implicitly declares Shader");
}

#[test]
fn image_mipmap_implicitly_declares_deps() {
    // ImageMipmap implicitly declares: ImageBasic
    // Not in OpenCL 2.0 allowlist, use Universal env.
    let text = [
        "OpCapability ImageMipmap",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("ImageMipmap implicitly declares ImageBasic");
}

#[test]
fn image_query_implicitly_declares_deps() {
    // ImageQuery implicitly declares: Shader
    let text = [
        "OpCapability ImageQuery",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ImageQuery implicitly declares Shader");
}

#[test]
fn image_read_write_implicitly_declares_deps() {
    // ImageReadWrite implicitly declares: ImageBasic
    let text = [
        "OpCapability ImageReadWrite",
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
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("ImageReadWrite implicitly declares ImageBasic");
}

#[test]
fn image_read_write_lod_amd_implicitly_declares_deps() {
    // ImageReadWriteLodAMD implicitly declares: Shader
    let text = [
        "OpCapability ImageReadWriteLodAMD",
        "OpExtension \"SPV_AMD_shader_image_load_store_lod\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ImageReadWriteLodAMD implicitly declares Shader");
}

#[test]
fn image_rect_implicitly_declares_deps() {
    // ImageRect implicitly declares: SampledRect
    // Not in Vulkan allowlist, use Universal env.
    let text = [
        "OpCapability ImageRect",
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
        .expect("ImageRect implicitly declares SampledRect");
}

#[test]
fn input_attachment_implicitly_declares_deps() {
    // InputAttachment implicitly declares: Shader
    let text = [
        "OpCapability InputAttachment",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("InputAttachment implicitly declares Shader");
}

#[test]
fn input_attachment_array_dynamic_indexing_implicitly_declares_deps() {
    // InputAttachmentArrayDynamicIndexing implicitly declares: InputAttachment
    let text = [
        "OpCapability InputAttachmentArrayDynamicIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("InputAttachmentArrayDynamicIndexing implicitly declares InputAttachment");
}

#[test]
fn input_attachment_array_non_uniform_indexing_implicitly_declares_deps() {
    // InputAttachmentArrayNonUniformIndexing implicitly declares: InputAttachment, ShaderNonUniform
    let text = [
        "OpCapability InputAttachmentArrayNonUniformIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("InputAttachmentArrayNonUniformIndexing implicitly declares InputAttachment, ShaderNonUniform");
}

#[test]
fn int4_cooperative_matrix_intel_implicitly_declares_deps() {
    // Int4CooperativeMatrixINTEL implicitly declares: Int4TypeINTEL, CooperativeMatrixKHR
    // CooperativeMatrixKHR + Shader requires VulkanMemoryModel capability and VulkanKHR memory model.
    let text = [
        "OpCapability Int4CooperativeMatrixINTEL",
        "OpCapability Shader",
        "OpCapability VulkanMemoryModel",
        "OpExtension \"SPV_INTEL_int4\"",
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
    text.as_str().validate(TargetEnv::Universal1_6).expect(
        "Int4CooperativeMatrixINTEL implicitly declares Int4TypeINTEL, CooperativeMatrixKHR",
    );
}

#[test]
fn int64_atomics_implicitly_declares_deps() {
    // Int64Atomics implicitly declares: Int64
    let text = [
        "OpCapability Int64Atomics",
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
        .expect("Int64Atomics implicitly declares Int64");
}

#[test]
fn int64_image_ext_implicitly_declares_deps() {
    // Int64ImageEXT implicitly declares: Shader
    let text = [
        "OpCapability Int64ImageEXT",
        "OpExtension \"SPV_EXT_shader_image_int64\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("Int64ImageEXT implicitly declares Shader");
}

#[test]
fn interpolation_function_implicitly_declares_deps() {
    // InterpolationFunction implicitly declares: Shader
    let text = [
        "OpCapability InterpolationFunction",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("InterpolationFunction implicitly declares Shader");
}

#[test]
fn literal_sampler_implicitly_declares_deps() {
    // LiteralSampler implicitly declares: Kernel
    // OpenCL special rule: LiteralSampler requires ImageBasic (not an implicit declaration).
    let text = [
        "OpCapability LiteralSampler",
        "OpCapability ImageBasic",
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
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("LiteralSampler implicitly declares Kernel");
}

#[test]
fn mesh_shading_ext_implicitly_declares_deps() {
    // MeshShadingEXT implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("MeshShadingEXT implicitly declares Shader");
}

#[test]
fn mesh_shading_nv_implicitly_declares_deps() {
    // MeshShadingNV implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("MeshShadingNV implicitly declares Shader");
}

#[test]
fn min_lod_implicitly_declares_deps() {
    // MinLod implicitly declares: Shader
    let text = [
        "OpCapability MinLod",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("MinLod implicitly declares Shader");
}

#[test]
fn multi_view_implicitly_declares_deps() {
    // MultiView implicitly declares: Shader
    let text = [
        "OpCapability MultiView",
        "OpExtension \"SPV_KHR_multiview\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("MultiView implicitly declares Shader");
}

#[test]
fn multi_viewport_implicitly_declares_deps() {
    // MultiViewport implicitly declares: Geometry
    let text = [
        "OpCapability MultiViewport",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("MultiViewport implicitly declares Geometry");
}

#[test]
fn named_barrier_implicitly_declares_deps() {
    // NamedBarrier implicitly declares: Kernel
    // Not in OpenCL 2.0 allowlist, use Universal env.
    let text = [
        "OpCapability NamedBarrier",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("NamedBarrier implicitly declares Kernel");
}

#[test]
fn per_view_attributes_nv_implicitly_declares_deps() {
    // PerViewAttributesNV implicitly declares: MultiView
    let text = [
        "OpCapability PerViewAttributesNV",
        "OpExtension \"SPV_NVX_multiview_per_view_attributes\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("PerViewAttributesNV implicitly declares MultiView");
}

#[test]
fn physical_storage_buffer_addresses_implicitly_declares_deps() {
    // PhysicalStorageBufferAddresses implicitly declares: Shader
    let text = [
        "OpCapability PhysicalStorageBufferAddresses",
        "OpExtension \"SPV_EXT_physical_storage_buffer\"",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("PhysicalStorageBufferAddresses implicitly declares Shader");
}

#[test]
fn pipe_storage_implicitly_declares_deps() {
    // PipeStorage implicitly declares: Pipes
    // Not in OpenCL 2.0 allowlist, use Universal env.
    let text = [
        "OpCapability PipeStorage",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("PipeStorage implicitly declares Pipes");
}

#[test]
fn pipes_implicitly_declares_deps() {
    // Pipes implicitly declares: Kernel
    let text = [
        "OpCapability Pipes",
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
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("Pipes implicitly declares Kernel");
}

#[test]
#[ignore = "rspirv does not yet recognize PushConstantBanksNV as a capability name"]
fn push_constant_banks_nv_implicitly_declares_deps() {
    // PushConstantBanksNV implicitly declares: Shader
    let text = [
        "OpCapability PushConstantBanksNV",
        "OpExtension \"SPV_NV_push_constant_bank\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("PushConstantBanksNV implicitly declares Shader");
}

#[test]
fn ray_query_khr_implicitly_declares_deps() {
    // RayQueryKHR implicitly declares: Shader
    let text = [
        "OpCapability RayQueryKHR",
        "OpExtension \"SPV_KHR_ray_query\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayQueryKHR implicitly declares Shader");
}

#[test]
fn ray_query_position_fetch_khr_implicitly_declares_deps() {
    // RayQueryPositionFetchKHR implicitly declares: Shader
    let text = [
        "OpCapability RayQueryPositionFetchKHR",
        "OpExtension \"SPV_KHR_ray_tracing_position_fetch\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayQueryPositionFetchKHR implicitly declares Shader");
}

#[test]
fn ray_query_provisional_khr_implicitly_declares_deps() {
    // RayQueryProvisionalKHR implicitly declares: Shader
    let text = [
        "OpCapability RayQueryProvisionalKHR",
        "OpExtension \"SPV_KHR_ray_query\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayQueryProvisionalKHR implicitly declares Shader");
}

#[test]
fn ray_tracing_cluster_acceleration_structure_nv_implicitly_declares_deps() {
    // RayTracingClusterAccelerationStructureNV implicitly declares: RayTracingKHR
    let text = [
        "OpCapability RayTracingClusterAccelerationStructureNV",
        "OpExtension \"SPV_NV_cluster_acceleration_structure\"",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingClusterAccelerationStructureNV implicitly declares RayTracingKHR");
}

#[test]
fn ray_tracing_displacement_micromap_nv_implicitly_declares_deps() {
    // RayTracingDisplacementMicromapNV implicitly declares: RayTracingKHR
    let text = [
        "OpCapability RayTracingDisplacementMicromapNV",
        "OpExtension \"SPV_NV_displacement_micromap\"",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingDisplacementMicromapNV implicitly declares RayTracingKHR");
}

#[test]
fn ray_tracing_khr_implicitly_declares_deps() {
    // RayTracingKHR implicitly declares: Shader
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingKHR implicitly declares Shader");
}

#[test]
fn ray_tracing_motion_blur_nv_implicitly_declares_deps() {
    // RayTracingMotionBlurNV implicitly declares: Shader
    let text = [
        "OpCapability RayTracingMotionBlurNV",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingMotionBlurNV implicitly declares Shader");
}

#[test]
fn ray_tracing_nv_implicitly_declares_deps() {
    // RayTracingNV implicitly declares: Shader
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingNV implicitly declares Shader");
}

#[test]
fn ray_tracing_opacity_micromap_ext_implicitly_declares_deps() {
    // RayTracingOpacityMicromapEXT implicitly declares: Shader
    let text = [
        "OpCapability RayTracingOpacityMicromapEXT",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingOpacityMicromapEXT implicitly declares Shader");
}

#[test]
fn ray_tracing_position_fetch_khr_implicitly_declares_deps() {
    // RayTracingPositionFetchKHR implicitly declares: Shader
    let text = [
        "OpCapability RayTracingPositionFetchKHR",
        "OpExtension \"SPV_KHR_ray_tracing_position_fetch\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingPositionFetchKHR implicitly declares Shader");
}

#[test]
fn ray_tracing_provisional_khr_implicitly_declares_deps() {
    // RayTracingProvisionalKHR implicitly declares: Shader
    let text = [
        "OpCapability RayTracingProvisionalKHR",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTracingProvisionalKHR implicitly declares Shader");
}

#[test]
fn ray_traversal_primitive_culling_khr_implicitly_declares_deps() {
    // RayTraversalPrimitiveCullingKHR implicitly declares: RayQueryKHR, RayTracingKHR
    let text = [
        "OpCapability RayTraversalPrimitiveCullingKHR",
        "OpExtension \"SPV_KHR_ray_query\"",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("RayTraversalPrimitiveCullingKHR implicitly declares RayQueryKHR, RayTracingKHR");
}

#[test]
fn runtime_descriptor_array_implicitly_declares_deps() {
    // RuntimeDescriptorArray implicitly declares: Shader
    let text = [
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("RuntimeDescriptorArray implicitly declares Shader");
}

#[test]
fn sample_mask_override_coverage_nv_implicitly_declares_deps() {
    // SampleMaskOverrideCoverageNV implicitly declares: SampleRateShading
    let text = [
        "OpCapability SampleMaskOverrideCoverageNV",
        "OpExtension \"SPV_NV_sample_mask_override_coverage\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("SampleMaskOverrideCoverageNV implicitly declares SampleRateShading");
}

#[test]
fn sample_rate_shading_implicitly_declares_deps() {
    // SampleRateShading implicitly declares: Shader
    let text = [
        "OpCapability SampleRateShading",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("SampleRateShading implicitly declares Shader");
}

#[test]
fn sampled_cube_array_implicitly_declares_deps() {
    // SampledCubeArray implicitly declares: Shader
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("SampledCubeArray implicitly declares Shader");
}

#[test]
fn sampled_image_array_dynamic_indexing_implicitly_declares_deps() {
    // SampledImageArrayDynamicIndexing implicitly declares: Shader
    let text = [
        "OpCapability SampledImageArrayDynamicIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("SampledImageArrayDynamicIndexing implicitly declares Shader");
}

#[test]
fn sampled_image_array_non_uniform_indexing_implicitly_declares_deps() {
    // SampledImageArrayNonUniformIndexing implicitly declares: ShaderNonUniform
    let text = [
        "OpCapability SampledImageArrayNonUniformIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("SampledImageArrayNonUniformIndexing implicitly declares ShaderNonUniform");
}

#[test]
fn sampled_rect_implicitly_declares_deps() {
    // SampledRect implicitly declares: Shader
    // Not in Vulkan allowlist, use Universal env.
    let text = [
        "OpCapability SampledRect",
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
        .expect("SampledRect implicitly declares Shader");
}

#[test]
fn shader_implicitly_declares_deps() {
    // Shader implicitly declares: Matrix
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("Shader implicitly declares Matrix");
}

#[test]
fn shader_enqueue_amdx_implicitly_declares_deps() {
    // ShaderEnqueueAMDX implicitly declares: Shader
    let text = [
        "OpCapability ShaderEnqueueAMDX",
        "OpExtension \"SPV_AMDX_shader_enqueue\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderEnqueueAMDX implicitly declares Shader");
}

#[test]
fn shader_invocation_reorder_ext_implicitly_declares_deps() {
    // ShaderInvocationReorderEXT implicitly declares: RayTracingKHR
    let text = [
        "OpCapability ShaderInvocationReorderEXT",
        "OpExtension \"SPV_EXT_shader_invocation_reorder\"",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderInvocationReorderEXT implicitly declares RayTracingKHR");
}

#[test]
fn shader_invocation_reorder_nv_implicitly_declares_deps() {
    // ShaderInvocationReorderNV implicitly declares: RayTracingKHR
    let text = [
        "OpCapability ShaderInvocationReorderNV",
        "OpExtension \"SPV_NV_shader_invocation_reorder\"",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderInvocationReorderNV implicitly declares RayTracingKHR");
}

#[test]
fn shader_non_uniform_implicitly_declares_deps() {
    // ShaderNonUniform implicitly declares: Shader
    let text = [
        "OpCapability ShaderNonUniform",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderNonUniform implicitly declares Shader");
}

#[test]
fn shader_sm_builtins_nv_implicitly_declares_deps() {
    // ShaderSMBuiltinsNV implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderSMBuiltinsNV implicitly declares Shader");
}

#[test]
fn shader_stereo_view_nv_implicitly_declares_deps() {
    // ShaderStereoViewNV implicitly declares: ShaderViewportMaskNV
    let text = [
        "OpCapability ShaderStereoViewNV",
        "OpExtension \"SPV_NV_stereo_view_rendering\"",
        "OpExtension \"SPV_EXT_shader_viewport_index_layer\"",
        "OpExtension \"SPV_NV_viewport_array2\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderStereoViewNV implicitly declares ShaderViewportMaskNV");
}

#[test]
fn shader_viewport_index_layer_ext_implicitly_declares_deps() {
    // ShaderViewportIndexLayerEXT implicitly declares: MultiViewport
    let text = [
        "OpCapability ShaderViewportIndexLayerEXT",
        "OpExtension \"SPV_EXT_shader_viewport_index_layer\"",
        "OpExtension \"SPV_NV_viewport_array2\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderViewportIndexLayerEXT implicitly declares MultiViewport");
}

#[test]
fn shader_viewport_mask_nv_implicitly_declares_deps() {
    // ShaderViewportMaskNV implicitly declares: ShaderViewportIndexLayerEXT
    let text = [
        "OpCapability ShaderViewportMaskNV",
        "OpExtension \"SPV_NV_viewport_array2\"",
        "OpExtension \"SPV_EXT_shader_viewport_index_layer\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderViewportMaskNV implicitly declares ShaderViewportIndexLayerEXT");
}

#[test]
fn sparse_residency_implicitly_declares_deps() {
    // SparseResidency implicitly declares: Shader
    let text = [
        "OpCapability SparseResidency",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("SparseResidency implicitly declares Shader");
}

#[test]
fn stencil_export_ext_implicitly_declares_deps() {
    // StencilExportEXT implicitly declares: Shader
    let text = [
        "OpCapability StencilExportEXT",
        "OpExtension \"SPV_EXT_shader_stencil_export\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StencilExportEXT implicitly declares Shader");
}

#[test]
fn storage_buffer_array_dynamic_indexing_implicitly_declares_deps() {
    // StorageBufferArrayDynamicIndexing implicitly declares: Shader
    let text = [
        "OpCapability StorageBufferArrayDynamicIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageBufferArrayDynamicIndexing implicitly declares Shader");
}

#[test]
fn storage_buffer_array_non_uniform_indexing_implicitly_declares_deps() {
    // StorageBufferArrayNonUniformIndexing implicitly declares: ShaderNonUniform
    let text = [
        "OpCapability StorageBufferArrayNonUniformIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageBufferArrayNonUniformIndexing implicitly declares ShaderNonUniform");
}

#[test]
fn storage_image_array_dynamic_indexing_implicitly_declares_deps() {
    // StorageImageArrayDynamicIndexing implicitly declares: Shader
    let text = [
        "OpCapability StorageImageArrayDynamicIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageImageArrayDynamicIndexing implicitly declares Shader");
}

#[test]
fn storage_image_array_non_uniform_indexing_implicitly_declares_deps() {
    // StorageImageArrayNonUniformIndexing implicitly declares: ShaderNonUniform
    let text = [
        "OpCapability StorageImageArrayNonUniformIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageImageArrayNonUniformIndexing implicitly declares ShaderNonUniform");
}

#[test]
fn storage_image_extended_formats_implicitly_declares_deps() {
    // StorageImageExtendedFormats implicitly declares: Shader
    let text = [
        "OpCapability StorageImageExtendedFormats",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageImageExtendedFormats implicitly declares Shader");
}

#[test]
fn storage_image_multisample_implicitly_declares_deps() {
    // StorageImageMultisample implicitly declares: Shader
    let text = [
        "OpCapability StorageImageMultisample",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageImageMultisample implicitly declares Shader");
}

#[test]
fn storage_image_read_without_format_implicitly_declares_deps() {
    // StorageImageReadWithoutFormat implicitly declares: Shader
    let text = [
        "OpCapability StorageImageReadWithoutFormat",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageImageReadWithoutFormat implicitly declares Shader");
}

#[test]
fn storage_image_write_without_format_implicitly_declares_deps() {
    // StorageImageWriteWithoutFormat implicitly declares: Shader
    let text = [
        "OpCapability StorageImageWriteWithoutFormat",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageImageWriteWithoutFormat implicitly declares Shader");
}

#[test]
fn storage_texel_buffer_array_dynamic_indexing_implicitly_declares_deps() {
    // StorageTexelBufferArrayDynamicIndexing implicitly declares: ImageBuffer
    let text = [
        "OpCapability StorageTexelBufferArrayDynamicIndexing",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("StorageTexelBufferArrayDynamicIndexing implicitly declares ImageBuffer");
}

#[test]
fn storage_texel_buffer_array_non_uniform_indexing_implicitly_declares_deps() {
    // StorageTexelBufferArrayNonUniformIndexing implicitly declares: ImageBuffer, ShaderNonUniform
    let text = [
        "OpCapability StorageTexelBufferArrayNonUniformIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("StorageTexelBufferArrayNonUniformIndexing implicitly declares ImageBuffer, ShaderNonUniform");
}

#[test]
fn subgroup2_d_block_transform_intel_implicitly_declares_deps() {
    // Subgroup2DBlockTransformINTEL implicitly declares: Subgroup2DBlockIOINTEL
    let text = [
        "OpCapability Subgroup2DBlockTransformINTEL",
        "OpCapability Shader",
        "OpExtension \"SPV_INTEL_2d_block_io\"",
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
        .expect("Subgroup2DBlockTransformINTEL implicitly declares Subgroup2DBlockIOINTEL");
}

#[test]
fn subgroup2_d_block_transpose_intel_implicitly_declares_deps() {
    // Subgroup2DBlockTransposeINTEL implicitly declares: Subgroup2DBlockIOINTEL
    let text = [
        "OpCapability Subgroup2DBlockTransposeINTEL",
        "OpCapability Shader",
        "OpExtension \"SPV_INTEL_2d_block_io\"",
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
        .expect("Subgroup2DBlockTransposeINTEL implicitly declares Subgroup2DBlockIOINTEL");
}

#[test]
fn subgroup_dispatch_implicitly_declares_deps() {
    // SubgroupDispatch implicitly declares: DeviceEnqueue
    // Not in OpenCL 2.0 allowlist, use Universal env.
    let text = [
        "OpCapability SubgroupDispatch",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("SubgroupDispatch implicitly declares DeviceEnqueue");
}

#[test]
fn tessellation_implicitly_declares_deps() {
    // Tessellation implicitly declares: Shader
    let text = [
        "OpCapability Tessellation",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("Tessellation implicitly declares Shader");
}

#[test]
fn tessellation_point_size_implicitly_declares_deps() {
    // TessellationPointSize implicitly declares: Tessellation
    let text = [
        "OpCapability TessellationPointSize",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("TessellationPointSize implicitly declares Tessellation");
}

#[test]
fn tile_shading_qcom_implicitly_declares_deps() {
    // TileShadingQCOM implicitly declares: Shader
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("TileShadingQCOM implicitly declares Shader");
}

#[test]
fn transform_feedback_implicitly_declares_deps() {
    // TransformFeedback implicitly declares: Shader
    let text = [
        "OpCapability TransformFeedback",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("TransformFeedback implicitly declares Shader");
}

#[test]
fn uniform_and_storage_buffer16_bit_access_implicitly_declares_deps() {
    // UniformAndStorageBuffer16BitAccess implicitly declares: StorageBuffer16BitAccess
    let text = [
        "OpCapability UniformAndStorageBuffer16BitAccess",
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_16bit_storage\"",
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
        .expect("UniformAndStorageBuffer16BitAccess implicitly declares StorageBuffer16BitAccess");
}

#[test]
fn uniform_and_storage_buffer8_bit_access_implicitly_declares_deps() {
    // UniformAndStorageBuffer8BitAccess implicitly declares: StorageBuffer8BitAccess
    let text = [
        "OpCapability UniformAndStorageBuffer8BitAccess",
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_8bit_storage\"",
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
        .expect("UniformAndStorageBuffer8BitAccess implicitly declares StorageBuffer8BitAccess");
}

#[test]
fn uniform_buffer_array_dynamic_indexing_implicitly_declares_deps() {
    // UniformBufferArrayDynamicIndexing implicitly declares: Shader
    let text = [
        "OpCapability UniformBufferArrayDynamicIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("UniformBufferArrayDynamicIndexing implicitly declares Shader");
}

#[test]
fn uniform_buffer_array_non_uniform_indexing_implicitly_declares_deps() {
    // UniformBufferArrayNonUniformIndexing implicitly declares: ShaderNonUniform
    let text = [
        "OpCapability UniformBufferArrayNonUniformIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("UniformBufferArrayNonUniformIndexing implicitly declares ShaderNonUniform");
}

#[test]
fn uniform_texel_buffer_array_dynamic_indexing_implicitly_declares_deps() {
    // UniformTexelBufferArrayDynamicIndexing implicitly declares: SampledBuffer
    let text = [
        "OpCapability UniformTexelBufferArrayDynamicIndexing",
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
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("UniformTexelBufferArrayDynamicIndexing implicitly declares SampledBuffer");
}

#[test]
fn uniform_texel_buffer_array_non_uniform_indexing_implicitly_declares_deps() {
    // UniformTexelBufferArrayNonUniformIndexing implicitly declares: SampledBuffer, ShaderNonUniform
    let text = [
        "OpCapability UniformTexelBufferArrayNonUniformIndexing",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("UniformTexelBufferArrayNonUniformIndexing implicitly declares SampledBuffer, ShaderNonUniform");
}

#[test]
fn untyped_variable_length_array_intel_implicitly_declares_deps() {
    // UntypedVariableLengthArrayINTEL implicitly declares: VariableLengthArrayINTEL, UntypedPointersKHR
    let text = [
        "OpCapability UntypedVariableLengthArrayINTEL",
        "OpCapability Shader",
        "OpExtension \"SPV_INTEL_variable_length_array\"",
        "OpExtension \"SPV_KHR_untyped_pointers\"",
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
        .expect("UntypedVariableLengthArrayINTEL implicitly declares VariableLengthArrayINTEL, UntypedPointersKHR");
}

#[test]
fn variable_pointers_implicitly_declares_deps() {
    // VariablePointers implicitly declares: VariablePointersStorageBuffer
    let text = [
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("VariablePointers implicitly declares VariablePointersStorageBuffer");
}

#[test]
fn variable_pointers_storage_buffer_implicitly_declares_deps() {
    // VariablePointersStorageBuffer implicitly declares: Shader
    let text = [
        "OpCapability VariablePointersStorageBuffer",
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
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("VariablePointersStorageBuffer implicitly declares Shader");
}

#[test]
fn vector16_implicitly_declares_deps() {
    // Vector16 implicitly declares: Kernel
    let text = [
        "OpCapability Vector16",
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
    text.as_str()
        .validate(TargetEnv::OpenCl2_0)
        .expect("Vector16 implicitly declares Kernel");
}

#[test]
fn vector_compute_intel_implicitly_declares_deps() {
    // VectorComputeINTEL implicitly declares: VectorAnyINTEL
    let text = [
        "OpCapability VectorComputeINTEL",
        "OpCapability Shader",
        "OpExtension \"SPV_INTEL_vector_compute\"",
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
        .expect("VectorComputeINTEL implicitly declares VectorAnyINTEL");
}

#[test]
fn workgroup_memory_explicit_layout16_bit_access_khr_implicitly_declares_deps() {
    // WorkgroupMemoryExplicitLayout16BitAccessKHR implicitly declares: WorkgroupMemoryExplicitLayoutKHR
    let text = [
        "OpCapability WorkgroupMemoryExplicitLayout16BitAccessKHR",
        "OpExtension \"SPV_KHR_workgroup_memory_explicit_layout\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("WorkgroupMemoryExplicitLayout16BitAccessKHR implicitly declares WorkgroupMemoryExplicitLayoutKHR");
}

#[test]
fn workgroup_memory_explicit_layout8_bit_access_khr_implicitly_declares_deps() {
    // WorkgroupMemoryExplicitLayout8BitAccessKHR implicitly declares: WorkgroupMemoryExplicitLayoutKHR
    let text = [
        "OpCapability WorkgroupMemoryExplicitLayout8BitAccessKHR",
        "OpExtension \"SPV_KHR_workgroup_memory_explicit_layout\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("WorkgroupMemoryExplicitLayout8BitAccessKHR implicitly declares WorkgroupMemoryExplicitLayoutKHR");
}

#[test]
fn workgroup_memory_explicit_layout_khr_implicitly_declares_deps() {
    // WorkgroupMemoryExplicitLayoutKHR implicitly declares: Shader
    let text = [
        "OpCapability WorkgroupMemoryExplicitLayoutKHR",
        "OpExtension \"SPV_KHR_workgroup_memory_explicit_layout\"",
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
        .validate(TargetEnv::Vulkan1_2)
        .expect("WorkgroupMemoryExplicitLayoutKHR implicitly declares Shader");
}
