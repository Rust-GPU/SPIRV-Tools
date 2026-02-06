use super::*;

#[test]
fn capability_version_clamps_for_binary_modules() {
    // Binary declares SPIR-V 1.6 and RayTracingKHR capability; Vulkan 1.0 should clamp.
    let binary = vec![
        0x07230203, // magic
        SpirvVersion::new(1, 6).to_word(),
        0,         // generator
        1,         // bound
        0,         // schema
        op(2, 17), // OpCapability
        rspirv::spirv::Capability::RayTracingKHR as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_0)
        .expect_err("RayTracingKHR requires SPIR-V 1.4+ and should clamp to env");
    match error {
        ValidationError::CapabilityRequiresSpirvVersion {
            capability,
            required_version,
            target_version,
        } => {
            assert_eq!(capability, rspirv::spirv::Capability::RayTracingKHR);
            assert_eq!(required_version, SpirvVersion::new(1, 4));
            assert_eq!(target_version, SpirvVersion::new(1, 0));
        }
        ValidationError::DisallowedCapability { capability, env } => {
            assert_eq!(capability, rspirv::spirv::Capability::RayTracingKHR);
            assert_eq!(env, TargetEnv::Vulkan1_0);
        }
        ValidationError::ExtensionRequiresSpirvVersion {
            extension,
            required_version,
            target_version,
        } => {
            assert_eq!(extension, ExtensionName::from("SPV_KHR_ray_tracing"));
            assert_eq!(required_version, SpirvVersion::new(1, 4));
            assert_eq!(target_version, SpirvVersion::new(1, 0));
        }
        other => panic!("unexpected error {other:?}"),
    }
}

#[test]
fn instruction_requires_spirv_version_from_grammar() {
    // Test that OpTerminateInvocation with the extension declared is allowed in 1.5
    // (the extension enables the instruction before it became core in 1.6)
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_terminate_invocation");
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
    builder.terminate_invocation().unwrap();
    builder.end_function().unwrap();
    let module = builder.module();
    assert!(
        module
            .extensions
            .iter()
            .any(|inst| super::extension_operand(inst)
                == Some(ExtensionName::from("SPV_KHR_terminate_invocation"))),
        "extension must be declared for opcode that requires it"
    );
    let words = module.assemble();
    // With the extension declared, the instruction is allowed even in SPIR-V 1.5
    words
        .as_slice()
        .validate(TargetEnv::Universal1_5)
        .expect("OpTerminateInvocation with extension should be allowed in SPIR-V 1.5");
}

#[test]
fn memory_model_vulkan_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Vulkan,
    );
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_1Spirv1_4)
        .expect_err("VulkanKHR memory model operand requires VulkanMemoryModel capability");
    // The extension satisfies the SPIR-V version requirement, so the real
    // error surfaces: the VulkanMemoryModel capability is not declared.
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::MemoryModel,
            operand_index: 1,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel,
        }
    );
}

#[test]
fn physical_storage_addressing_model_requires_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble physical storage memory model");
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::MemoryModel,
            operand_index: 0,
            required_capability: rspirv::spirv::Capability::PhysicalStorageBufferAddresses,
        }
    );
}

#[test]
fn memory_access_non_private_pointer_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    // Provide the extension but deliberately omit the VulkanMemoryModel capability.
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 1);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Function, int);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    let var_id = builder.variable(ptr, None, rspirv::spirv::StorageClass::Function, None);
    builder
        .load(
            int,
            None,
            var_id,
            Some(rspirv::spirv::MemoryAccess::NON_PRIVATE_POINTER),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let binary = builder.module().assemble();
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::Load,
            operand_index: 1,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_access_non_private_pointer_store_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 1);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Function, int);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let zero = builder.constant_bit32(int, 0);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    let var_id = builder.variable(ptr, None, rspirv::spirv::StorageClass::Function, None);
    builder
        .store(
            var_id,
            zero,
            Some(rspirv::spirv::MemoryAccess::NON_PRIVATE_POINTER),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let binary = builder.module().assemble();
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::Store,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_semantics_make_visible_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    let semantics =
        builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::MAKE_VISIBLE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .control_barrier(workgroup_scope, workgroup_scope, semantics)
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakeVisible semantics requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ControlBarrier,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_semantics_make_visible_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    let semantics =
        builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::MAKE_VISIBLE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .control_barrier(workgroup_scope, workgroup_scope, semantics)
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("MakeVisible semantics requires VulkanMemoryModel capability");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::ControlBarrier,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_semantics_make_available_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    let semantics =
        builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::MAKE_AVAILABLE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .control_barrier(workgroup_scope, workgroup_scope, semantics)
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakeAvailable semantics requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ControlBarrier,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_semantics_make_available_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    let semantics =
        builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::MAKE_AVAILABLE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .control_barrier(workgroup_scope, workgroup_scope, semantics)
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("MakeAvailable semantics requires VulkanMemoryModel capability");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::ControlBarrier,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn queue_family_scope_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let queue_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::QueueFamilyKHR as u32);
    let semantics = builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::ACQUIRE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder.memory_barrier(queue_scope, semantics).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("QueueFamilyKHR scope requires VulkanMemoryModel capability");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::MemoryBarrier,
            operand_index: 0,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn queue_family_scope_allows_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let queue_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::QueueFamilyKHR as u32);
    let semantics = builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::ACQUIRE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder.memory_barrier(queue_scope, semantics).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect("QueueFamilyKHR scope is allowed with VulkanMemoryModel capability");
}

#[test]
fn shader_call_scope_requires_ray_tracing_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_ray_tracing");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let shader_call_scope =
        builder.constant_bit32(uint, rspirv::spirv::Scope::ShaderCallKHR as u32);
    let semantics = builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::ACQUIRE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .control_barrier(shader_call_scope, shader_call_scope, semantics)
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("ShaderCallKHR scope requires RayTracingKHR capability when present");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::ControlBarrier,
            operand_index: 0,
            required_capability: rspirv::spirv::Capability::RayTracingKHR
        }
    );
}

#[test]
fn shader_call_scope_allows_ray_tracing_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.capability(rspirv::spirv::Capability::RayTracingKHR);
    builder.extension("SPV_KHR_ray_tracing");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let shader_call_scope =
        builder.constant_bit32(uint, rspirv::spirv::Scope::ShaderCallKHR as u32);
    let semantics = builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::ACQUIRE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .control_barrier(shader_call_scope, shader_call_scope, semantics)
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect("ShaderCallKHR scope allowed with RayTracingKHR capability");
}

#[test]
fn memory_semantics_volatile_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    let semantics = builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::VOLATILE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .control_barrier(workgroup_scope, workgroup_scope, semantics)
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("Volatile semantics requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ControlBarrier,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn decoration_non_uniform_requires_spirv_1_5() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%ptr = OpTypePointer Uniform %bool",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %var NonUniform",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Universal1_4)
        .expect_err("NonUniform decoration requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::Decorate,
            operand_index: 1,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn decoration_non_uniform_requires_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%ptr = OpTypePointer Uniform %bool",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %var NonUniform",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Universal1_5)
        .expect_err("NonUniform decoration requires ShaderNonUniform capability");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::Decorate,
            operand_index: 1,
            required_capability: rspirv::spirv::Capability::ShaderNonUniform,
        }
    );
}

#[test]
fn image_operands_make_texel_visible_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let semantics =
        builder.constant_bit32(int, rspirv::spirv::MemorySemantics::MAKE_VISIBLE.bits());
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::MAKE_TEXEL_VISIBLE),
            [rspirv::dr::Operand::IdRef(semantics)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakeTexelVisible image operand requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn image_operands_make_texel_visible_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let _texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let semantics = builder.constant_bit32(int, rspirv::spirv::MemorySemantics::ACQUIRE.bits());
    // MakeTexelVisible is for read operations, not write
    builder
        .image_read(
            v4float,
            None,
            img,
            coord,
            Some(
                rspirv::spirv::ImageOperands::MAKE_TEXEL_VISIBLE
                    | rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL,
            ),
            [rspirv::dr::Operand::IdRef(semantics)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("MakeTexelVisible image operand requires VulkanMemoryModel capability");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::ImageRead,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn image_operands_make_texel_visible_allows_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let _texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder.decorate(
        img,
        rspirv::spirv::Decoration::DescriptorSet,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    builder.decorate(
        img,
        rspirv::spirv::Decoration::Binding,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let semantics = builder.constant_bit32(int, rspirv::spirv::MemorySemantics::ACQUIRE.bits());
    // MakeTexelVisible is for read operations, not write
    builder
        .image_read(
            v4float,
            None,
            img,
            coord,
            Some(
                rspirv::spirv::ImageOperands::MAKE_TEXEL_VISIBLE
                    | rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL,
            ),
            [rspirv::dr::Operand::IdRef(semantics)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect("MakeTexelVisible image operand allowed with VulkanMemoryModel capability");
}

#[test]
fn image_operands_make_texel_available_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let semantics = builder.constant_bit32(int, rspirv::spirv::MemorySemantics::ACQUIRE.bits());
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(
                rspirv::spirv::ImageOperands::MAKE_TEXEL_AVAILABLE
                    | rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL,
            ),
            [rspirv::dr::Operand::IdRef(semantics)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("MakeTexelAvailable image operand requires VulkanMemoryModel capability");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn image_operands_make_texel_available_allows_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder.decorate(
        img,
        rspirv::spirv::Decoration::DescriptorSet,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    builder.decorate(
        img,
        rspirv::spirv::Decoration::Binding,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let semantics = builder.constant_bit32(int, rspirv::spirv::MemorySemantics::ACQUIRE.bits());
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(
                rspirv::spirv::ImageOperands::MAKE_TEXEL_AVAILABLE
                    | rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL,
            ),
            [rspirv::dr::Operand::IdRef(semantics)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect("MakeTexelAvailable image operand allowed with VulkanMemoryModel capability");
}

#[test]
fn image_operands_non_private_texel_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("NonPrivateTexel image operand requires VulkanMemoryModel capability");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn image_operands_non_private_texel_allows_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder.decorate(
        img,
        rspirv::spirv::Decoration::DescriptorSet,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    builder.decorate(
        img,
        rspirv::spirv::Decoration::Binding,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect("NonPrivateTexel image operand allowed with VulkanMemoryModel capability");
}

#[test]
fn image_operands_nontemporal_requires_spirv_1_6() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::NONTEMPORAL),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_5)
        .expect_err("Nontemporal image operand requires SPIR-V 1.6");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_version: SpirvVersion::new(1, 6),
            target_version: SpirvVersion::new(1, 5),
        }
    );
}

#[test]
fn image_operands_make_texel_available_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    let semantics =
        builder.constant_bit32(int, rspirv::spirv::MemorySemantics::MAKE_AVAILABLE.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::MAKE_TEXEL_AVAILABLE),
            [rspirv::dr::Operand::IdRef(semantics)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakeTexelAvailable image operand requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn image_operands_sign_extend_requires_spirv_1_4() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 3);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::SIGN_EXTEND),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_3)
        .expect_err("SignExtend image operand requires SPIR-V 1.4");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_version: SpirvVersion::new(1, 4),
            target_version: SpirvVersion::new(1, 3),
        }
    );
}

#[test]
fn image_operands_zero_extend_requires_spirv_1_4() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 3);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::ZERO_EXTEND),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_3)
        .expect_err("ZeroExtend image operand requires SPIR-V 1.4");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_version: SpirvVersion::new(1, 4),
            target_version: SpirvVersion::new(1, 3),
        }
    );
}

#[test]
fn image_operands_non_private_texel_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::NON_PRIVATE_TEXEL),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("NonPrivateTexel image operand requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn image_operands_volatile_texel_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let v2int = builder.type_vector(int, 2);
    let v4float = builder.type_vector(float, 4);
    let image = builder.type_image(
        float,
        rspirv::spirv::Dim::Dim2D,
        0,
        0,
        0,
        2,
        rspirv::spirv::ImageFormat::Rgba32f,
        None,
    );
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::UniformConstant, image);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let int_0 = builder.constant_bit32(int, 0);
    let float_0 = builder.constant_bit32(float, 0.0f32.to_bits());
    let coord = builder.constant_composite(v2int, [int_0, int_0]);
    let texel = builder.constant_composite(v4float, [float_0, float_0, float_0, float_0]);
    let img = builder.variable(
        ptr,
        None,
        rspirv::spirv::StorageClass::UniformConstant,
        None,
    );
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_write(
            img,
            coord,
            texel,
            Some(rspirv::spirv::ImageOperands::VOLATILE_TEXEL),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("VolatileTexel image operand requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ImageWrite,
            operand_index: 3,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_semantics_output_memory_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let workgroup_scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    let semantics =
        builder.constant_bit32(uint, rspirv::spirv::MemorySemantics::OUTPUT_MEMORY.bits());
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .control_barrier(workgroup_scope, workgroup_scope, semantics)
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("OutputMemory semantics requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ControlBarrier,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_make_pointer_visible_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let value = builder.constant_bit32(uint, 0);
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .store(
            var,
            value,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_VISIBLE),
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakePointerVisible memory access requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::Store,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_make_pointer_visible_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let value = builder.constant_bit32(uint, 0);
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .store(
            var,
            value,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_VISIBLE),
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words.as_slice().validate(TargetEnv::Vulkan1_2).expect_err(
        "MakePointerVisible requires VulkanMemoryModel capability when version is satisfied",
    );
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::Store,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_access_make_pointer_available_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let value = builder.constant_bit32(uint, 0);
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .store(
            var,
            value,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_AVAILABLE),
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakePointerAvailable memory access requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::Store,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_make_pointer_available_load_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .load(
            uint,
            None,
            var,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_AVAILABLE),
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakePointerAvailable (load) memory access requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::Load,
            operand_index: 1,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_make_pointer_available_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let value = builder.constant_bit32(uint, 0);
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .store(
            var,
            value,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_AVAILABLE),
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let binary = builder.module().assemble();
    let error = binary.as_slice().validate(TargetEnv::Vulkan1_2).expect_err(
        "MakePointerAvailable requires VulkanMemoryModel capability when version is satisfied",
    );
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::Store,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_access_make_pointer_visible_load_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .load(
            uint,
            None,
            var,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_VISIBLE),
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakePointerVisible (load) memory access requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::Load,
            operand_index: 1,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_make_pointer_visible_load_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .load(
            uint,
            None,
            var,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_VISIBLE),
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words.as_slice().validate(TargetEnv::Vulkan1_2).expect_err(
        "MakePointerVisible (load) requires VulkanMemoryModel capability when version is satisfied",
    );
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::Load,
            operand_index: 1,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_access_make_pointer_available_load_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .load(
            uint,
            None,
            var,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_AVAILABLE),
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
            .as_slice()
            .validate(TargetEnv::Vulkan1_2)
            .expect_err("MakePointerAvailable (load) requires VulkanMemoryModel capability when version is satisfied");
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::Load,
            operand_index: 1,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_access_non_private_pointer_copy_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let src = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let dst = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .copy_memory(
            dst,
            src,
            Some(rspirv::spirv::MemoryAccess::NON_PRIVATE_POINTER),
            None,
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("NonPrivatePointer copy requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::CopyMemory,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_non_private_pointer_copy_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let src = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let dst = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .copy_memory(
            dst,
            src,
            Some(rspirv::spirv::MemoryAccess::NON_PRIVATE_POINTER),
            None,
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words.as_slice().validate(TargetEnv::Vulkan1_2).expect_err(
        "NonPrivatePointer copy requires VulkanMemoryModel capability when version is satisfied",
    );
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::CopyMemory,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_access_make_pointer_visible_copy_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let src = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let dst = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .copy_memory(
            dst,
            src,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_VISIBLE),
            None,
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakePointerVisible copy requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::CopyMemory,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_make_pointer_visible_copy_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let src = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let dst = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .copy_memory(
            dst,
            src,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_VISIBLE),
            None,
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words.as_slice().validate(TargetEnv::Vulkan1_2).expect_err(
        "MakePointerVisible copy requires VulkanMemoryModel capability when version is satisfied",
    );
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::CopyMemory,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_access_make_pointer_available_copy_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let src = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let dst = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .copy_memory(
            dst,
            src,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_AVAILABLE),
            None,
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("MakePointerAvailable copy requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::CopyMemory,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_make_pointer_available_copy_requires_vulkan_memory_model_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.extension("SPV_KHR_vulkan_memory_model");
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let src = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let dst = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    let scope = builder.constant_bit32(uint, rspirv::spirv::Scope::Workgroup as u32);
    builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .copy_memory(
            dst,
            src,
            Some(rspirv::spirv::MemoryAccess::MAKE_POINTER_AVAILABLE),
            None,
            [rspirv::dr::Operand::IdScope(scope)],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words.as_slice().validate(TargetEnv::Vulkan1_2).expect_err(
        "MakePointerAvailable copy requires VulkanMemoryModel capability when version is satisfied",
    );
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::CopyMemory,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::VulkanMemoryModel
        }
    );
}

#[test]
fn memory_access_non_private_pointer_requires_spirv_1_5() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 4);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let value = builder.constant_bit32(uint, 0);
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .store(
            var,
            value,
            Some(rspirv::spirv::MemoryAccess::NON_PRIVATE_POINTER),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_4)
        .expect_err("NonPrivatePointer memory access requires SPIR-V 1.5");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::Store,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 5),
            target_version: SpirvVersion::new(1, 4),
        }
    );
}

#[test]
fn memory_access_nontemporal_requires_spirv_1_6() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let ptr = builder.type_pointer(None, rspirv::spirv::StorageClass::Workgroup, uint);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let value = builder.constant_bit32(uint, 0);
    let var = builder.variable(ptr, None, rspirv::spirv::StorageClass::Workgroup, None);
    builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .store(
            var,
            value,
            Some(rspirv::spirv::MemoryAccess::NONTEMPORAL),
            std::iter::empty::<rspirv::dr::Operand>(),
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_5)
        .expect_err("NonTemporal memory access requires SPIR-V 1.6");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::Store,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 6),
            target_version: SpirvVersion::new(1, 5),
        }
    );
}

#[test]
fn storage_buffer_requires_spirv_1_3() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 0",
        "%ptr = OpTypePointer StorageBuffer %int",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr StorageBuffer",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Universal1_2)
        .expect_err("StorageBuffer storage class requires SPIR-V 1.3");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::TypePointer,
            operand_index: 0,
            required_version: SpirvVersion::new(1, 3),
            target_version: SpirvVersion::new(1, 2),
        }
    );
}

#[test]
fn loop_control_dependency_length_requires_spirv_1_1() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%fn = OpTypeFunction %void",
        "%true = OpConstantTrue %bool",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpBranch %loop",
        "%loop = OpLabel",
        "OpLoopMerge %merge %continue DependencyLength 1",
        "OpBranch %continue",
        "%continue = OpLabel",
        "OpBranchConditional %true %loop %merge",
        "%merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Universal1_0)
        .expect_err("DependencyLength loop control requires SPIR-V 1.1");
    assert_eq!(
        error,
        ValidationError::OperandRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::LoopMerge,
            operand_index: 2,
            required_version: SpirvVersion::new(1, 1),
            target_version: SpirvVersion::new(1, 0),
        }
    );
}

#[test]
fn execution_mode_id_requires_spirv_1_2() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 2);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let uint = builder.type_int(32, 0);
    let function_type = builder.type_function(void, std::iter::empty::<u32>());
    let local_size_x = builder.constant_bit32(uint, 1);
    let local_size_y = builder.constant_bit32(uint, 1);
    let local_size_z = builder.constant_bit32(uint, 1);
    let entry_point = builder
        .begin_function(
            void,
            None,
            rspirv::spirv::FunctionControl::NONE,
            function_type,
        )
        .unwrap();
    builder.begin_block(None).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    builder.entry_point(
        rspirv::spirv::ExecutionModel::Vertex,
        entry_point,
        "main",
        [],
    );
    builder.execution_mode_id(
        entry_point,
        rspirv::spirv::ExecutionMode::LocalSizeId,
        [local_size_x, local_size_y, local_size_z],
    );
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_1)
        .expect_err("ExecutionModeId::LocalSizeId requires SPIR-V 1.2");
    assert_eq!(
        error,
        ValidationError::InstructionRequiresSpirvVersion {
            opcode: rspirv::spirv::Op::ExecutionModeId,
            required_version: SpirvVersion::new(1, 2),
            target_version: SpirvVersion::new(1, 1),
        },
    );
}

#[test]
fn shader_clock_capability_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpCapability ShaderClockKHR",
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
        .validate(TargetEnv::Vulkan1_3)
        .expect_err("ShaderClockKHR requires declaring the extension");
    assert_eq!(
        error,
        ValidationError::DisallowedCapabilityMissingExtension {
            capability: rspirv::spirv::Capability::ShaderClockKHR,
            required_extension: "SPV_KHR_shader_clock".to_string(),
        }
    );
    let with_extension = [
        "OpCapability Shader",
        "OpCapability ShaderClockKHR",
        "OpExtension \"SPV_KHR_shader_clock\"",
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
        .validate(TargetEnv::Vulkan1_3)
        .expect("extension declared should satisfy capability");
}
