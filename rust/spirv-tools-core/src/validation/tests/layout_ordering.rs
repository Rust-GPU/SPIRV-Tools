use super::*;

#[test]
fn mode_stage_orders_mode_settings() {
    use super::instruction_layout::{mode_stage, ModeStage};

    assert_eq!(mode_stage(Op::Capability), Some(ModeStage::Capabilities));
    assert_eq!(
        mode_stage(Op::ConditionalCapabilityINTEL),
        Some(ModeStage::Capabilities)
    );
    assert_eq!(mode_stage(Op::Extension), Some(ModeStage::Extensions));
    assert_eq!(
        mode_stage(Op::ConditionalExtensionINTEL),
        Some(ModeStage::Extensions)
    );
    assert_eq!(
        mode_stage(Op::ConditionalEntryPointINTEL),
        Some(ModeStage::EntryPoint)
    );
    assert_eq!(
        mode_stage(Op::ExtInstImport),
        Some(ModeStage::ExtInstImport)
    );
    assert_eq!(mode_stage(Op::MemoryModel), Some(ModeStage::MemoryModel));
    assert_eq!(mode_stage(Op::EntryPoint), Some(ModeStage::EntryPoint));
    assert_eq!(
        mode_stage(Op::ExecutionMode),
        Some(ModeStage::ExecutionMode)
    );
    assert_eq!(
        mode_stage(Op::ExecutionModeId),
        Some(ModeStage::ExecutionMode)
    );
    assert_eq!(mode_stage(Op::TypeVoid), None);
}

#[test]
fn validate_module_rejects_missing_header() {
    let binary = vec![0x07230203, 0, 0, 0, 0];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(error, ValidationError::MissingMemoryModel);
}

#[test]
fn validate_module_rejects_ids_beyond_bound() {
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
    ]
    .join("\n");
    let mut binary = assemble_text(&text).expect("assemble");
    // Clamp the declared id bound to 1, which is lower than any type id emitted.
    binary[3] = 1;
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::IdExceedsBound {
            id: Id::new(NonZeroU32::new(1).unwrap()),
            bound: CheckedBound::new(DeclaredBound(1)).unwrap()
        }
    );
}

#[test]
fn validate_module_requires_memory_model() {
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::InstructionBeforeMemoryModel {
            opcode: rspirv::spirv::Op::TypeVoid,
        }
    );
}

#[test]
fn operand_requires_capability_from_grammar() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "%void = OpTypeVoid",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Input %u32",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr Input",
        "OpDecorate %var BuiltIn SubgroupSize",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::MissingOperandCapability {
            opcode: rspirv::spirv::Op::Decorate,
            operand_index: 2,
            required_capability: rspirv::spirv::Capability::Kernel
        }
    );
}

#[test]
fn operand_requires_extension_from_grammar() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "OpExecutionMode %main SubgroupUniformControlFlowKHR",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::MissingOperandExtension {
            opcode: rspirv::spirv::Op::ExecutionMode,
            operand_index: 1,
            required_extension: ExtensionName::from("SPV_KHR_subgroup_uniform_control_flow"),
        }
    );
}

#[test]
fn conditional_extension_rejected_when_disallowed() {
    use rspirv::binary::Assemble;
    use rspirv::dr::Instruction;
    use rspirv::spirv::{AddressingModel, Capability, MemoryModel, Op};
    let mut module = rspirv::dr::Module::new();
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![rspirv::dr::Operand::Capability(Capability::Shader)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            rspirv::dr::Operand::AddressingModel(AddressingModel::Logical),
            rspirv::dr::Operand::MemoryModel(MemoryModel::GLSL450),
        ],
    ));
    module.extensions.push(Instruction::new(
        Op::ConditionalExtensionINTEL,
        None,
        None,
        vec![rspirv::dr::Operand::LiteralString(
            "SPV_KHR_ray_tracing".into(),
        )],
    ));
    module.header = Some(rspirv::dr::ModuleHeader::new(5));
    let void = rspirv::dr::Operand::IdRef(1);
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(2),
        None,
        vec![void.clone()],
    ));
    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(3),
        vec![
            rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            rspirv::dr::Operand::IdRef(2),
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
    let error = validate_module(&binary, TargetEnv::WebGpu0).unwrap_err();
    assert_eq!(
        error,
        ValidationError::DisallowedExtension {
            extension: ExtensionName::from("KHR_ray_tracing"),
            env: TargetEnv::WebGpu0
        }
    );
}

#[test]
fn conditional_extension_after_functions_rejected_for_ordering() {
    // Hand-rolled module: capability + memory model, then types/functions,
    // then a conditional extension placed after the function to trigger
    // layout ordering validation.
    let binary = vec![
        0x0723_0203,
        0x0001_0600,
        0,
        5, // bound
        0,
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
        op(3, Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        MemoryModel::GLSL450 as u32,
        op(2, Op::TypeVoid as u16),
        1,
        op(3, Op::TypeFunction as u16),
        2, // result id
        1, // return type
        op(5, Op::Function as u16),
        1, // result type (void)
        3, // result id
        FunctionControl::NONE.bits(),
        2, // fn type
        op(2, Op::Label as u16),
        4, // label
        op(1, Op::Return as u16),
        op(1, Op::FunctionEnd as u16),
        op(6, Op::ConditionalExtensionINTEL as u16),
        0x5f565053, // "SPV_"
        0x5f52484b, // "KHR_"
        0x5f796172, // "ray_"
        0x63617274, // "trac"
        0x00676e69, // "ing\0"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn ext_inst_import_requires_memory_model() {
    // OpExtInstImport before OpMemoryModel should be reported as a memory-model ordering violation.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        3,          // bound (ids up to 2)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        0x0006000b, // OpExtInstImport %1 "GLSL.std.450"
        1,
        0x4c53_4c47, // "GLSL"
        0x6474_732e, // ".std"
        0x3035_342e, // ".450"
        0,           // padding/null terminator
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(error, ValidationError::MissingMemoryModel);
}

#[test]
fn conditional_extension_must_precede_types_and_globals() {
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        3,          // bound (ids up to 2)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1 (types/globals)
        1,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (after types -> error)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_capability_must_precede_types_and_globals() {
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        3,          // bound (ids up to 2)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1 (types/globals)
        1,
        op(3, 6250), // OpConditionalCapabilityINTEL %2 Linkage (after types -> error)
        2,
        rspirv::spirv::Capability::Linkage as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn capability_after_annotations_is_rejected() {
    // Place a capability after a decorate (Annotations section) to trigger a layout error.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        3,          // bound
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 59), // OpDecorate %1 Block
        1,
        rspirv::spirv::Decoration::Block as u32,
        0,
        op(2, 17), // OpCapability Linkage (after annotations -> error)
        rspirv::spirv::Capability::Linkage as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn extension_after_functions_is_rejected() {
    // Emit a minimal function, then relocate the extension to appear after functions to trigger a layout error.
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_KHR_shader_clock\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%func = OpTypeFunction %void",
        "%main = OpFunction %void None %func",
        "%lbl = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let mut words = assemble_text(&text).expect("assemble extension module");
    // Pull the OpExtension instruction to the end of the module.
    let mut idx = 5; // skip header
    let mut ext_slice: Option<(usize, usize)> = None;
    while idx < words.len() {
        let wc = (words[idx] >> 16) as usize;
        let opcode = (words[idx] & 0xffff) as u16;
        if opcode == rspirv::spirv::Op::Extension as u16 {
            ext_slice = Some((idx, wc));
            break;
        }
        idx += wc;
    }
    let (start, len) = ext_slice.expect("extension instruction present");
    let extension: Vec<u32> = words.drain(start..start + len).collect();
    words.extend(extension);
    let error = validate_module(&words, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn conditional_capability_after_functions_is_rejected() {
    // Place OpConditionalCapabilityINTEL after the function to trigger layout ordering error.
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0600, // version 1.6
        0,           // generator
        6,           // bound (ids up to 5)
        0,           // schema
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
        op(3, Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        MemoryModel::GLSL450 as u32,
        op(2, Op::TypeVoid as u16),
        1,
        op(3, Op::TypeFunction as u16),
        2, // result id
        1, // return type
        op(5, Op::Function as u16),
        1, // result type
        3, // result id
        FunctionControl::NONE.bits(),
        2, // fn type
        op(2, Op::Label as u16),
        4, // label id
        op(1, Op::Return as u16),
        op(1, Op::FunctionEnd as u16),
        op(3, Op::ConditionalCapabilityINTEL as u16),
        5, // result id
        Capability::Shader as u32,
    ];
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn conditional_extension_after_functions_is_rejected() {
    // Place OpConditionalExtensionINTEL after the function to trigger layout ordering error.
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0600, // version 1.6
        0,           // generator
        6,           // bound (ids up to 5)
        0,           // schema
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
        op(3, Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        MemoryModel::GLSL450 as u32,
        op(2, Op::TypeVoid as u16),
        1,
        op(3, Op::TypeFunction as u16),
        2, // result id
        1, // return type
        op(5, Op::Function as u16),
        1, // result type
        3, // result id
        FunctionControl::NONE.bits(),
        2, // fn type
        op(2, Op::Label as u16),
        4, // label id
        op(1, Op::Return as u16),
        op(1, Op::FunctionEnd as u16),
        op(8, Op::ConditionalExtensionINTEL as u16),
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
    ];
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn extension_after_annotations_is_rejected() {
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        2,           // bound (ids up to 1)
        0,           // schema
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
        op(3, Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        MemoryModel::GLSL450 as u32,
        op(3, Op::Decorate as u16),
        1, // target id
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(8, Op::Extension as u16),
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        op(4, Op::TypeInt as u16),
        1, // result id
        32,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn extension_after_ext_inst_import_is_rejected() {
    let clock_ext = [
        1599492179, 1599227979, 1684105331, 1667199589, 1801678700, 0,
    ];
    let glsl_ext = [0x4c53_4c47, 0x6474_732e, 0x3035_342e, 0];
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        2,           // bound (ids up to 1)
        0,           // schema
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
        op(8, Op::Extension as u16),
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        op(6, Op::ExtInstImport as u16),
        1, // result id
        glsl_ext[0],
        glsl_ext[1],
        glsl_ext[2],
        glsl_ext[3],
        op(7, Op::Extension as u16),
        clock_ext[0],
        clock_ext[1],
        clock_ext[2],
        clock_ext[3],
        clock_ext[4],
        clock_ext[5],
        op(3, Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        MemoryModel::GLSL450 as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn extension_after_execution_mode_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%lbl = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "OpEntryPoint Fragment %main \"main\"",
        "OpExecutionMode %main OriginUpperLeft",
        "OpExtension \"SPV_KHR_shader_clock\"",
    ]
    .join("\n");
    let mut words = assemble_text(&text).expect("assemble extension-after-execution-mode module");
    // Move the OpExtension to the end (after execution mode).
    let mut idx = 5; // skip header
    let mut slice = None;
    while idx < words.len() {
        let wc = (words[idx] >> 16) as usize;
        let opcode = (words[idx] & 0xffff) as u16;
        if opcode == rspirv::spirv::Op::Extension as u16 {
            slice = Some((idx, wc));
            break;
        }
        idx += wc;
    }
    let (start, len) = slice.expect("extension present");
    let ext: Vec<u32> = words.drain(start..start + len).collect();
    words.extend(ext);
    let error = validate_module(&words, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn extension_inside_function_is_rejected() {
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        4,           // bound (ids up to 3)
        0,           // schema
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
        op(3, Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        MemoryModel::GLSL450 as u32,
        op(2, Op::TypeVoid as u16),
        1,
        op(3, Op::TypeFunction as u16),
        2, // result id
        1, // return type
        op(5, Op::Function as u16),
        1, // result type
        3, // result id
        FunctionControl::NONE.bits(),
        2, // fn type
        op(2, Op::Label as u16),
        4, // label
        op(8, Op::Extension as u16),
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        op(1, Op::Return as u16),
        op(1, Op::FunctionEnd as u16),
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn capability_after_extension_is_rejected() {
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        1,           // bound
        0,           // schema
        op(8, Op::Extension as u16),
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn extension_after_memory_model_is_rejected() {
    let clock_ext = [
        1599492179, 1599227979, 1684105331, 1667199589, 1801678700, 0,
    ];
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        1,           // bound
        0,           // schema
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
        op(3, Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        MemoryModel::GLSL450 as u32,
        op(7, Op::Extension as u16),
        clock_ext[0],
        clock_ext[1],
        clock_ext[2],
        clock_ext[3],
        clock_ext[4],
        clock_ext[5],
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn names_section_must_follow_debug_section() {
    // OpName (Names section) precedes OpSource (Debug section), which should trigger an ordering error.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        0x00030005, // OpName %1 "x" (names)
        1,
        0x0000_0078,
        op(3, 3), // OpSource Unknown 0 (debug section after names -> error)
        0,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Source
        }
    );
}

#[test]
fn annotations_must_follow_names() {
    // OpDecorate (Annotations) placed before OpName (Names) should trigger ordering diagnostics.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 71), // OpDecorate %1 RelaxedPrecision (annotations after names -> error)
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        0x00030005, // OpName %1 "x" (names)
        1,
        0x0000_0078,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Name
        }
    );
}

#[test]
fn decorations_cannot_follow_functions() {
    use rspirv::{binary::Assemble, dr::Builder, spirv::Decoration, spirv::Op};
    let mut builder = Builder::new();
    builder.set_version(1, 0);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let fn_type = builder.type_function(void, std::iter::empty::<u32>());
    let fn_id = builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let mut words = builder.module().assemble();
    words.push(op(3, Op::Decorate as u16));
    words.push(fn_id);
    words.push(Decoration::RelaxedPrecision as u32);
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Decorate
        }
    );
}

#[test]
fn decorations_cannot_follow_types_and_globals() {
    // Annotations must appear before the types-and-globals section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        2,          // bound (ids up to 1)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, rspirv::spirv::Op::TypeStruct as u16), // %1 = OpTypeStruct
        1,
        op(3, rspirv::spirv::Op::Decorate as u16), // OpDecorate %1 Block (after types -> error)
        1,
        rspirv::spirv::Decoration::Block as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Decorate
        }
    );
}

#[test]
fn decoration_group_cannot_follow_types_and_globals() {
    // Annotation section opcodes such as OpDecorationGroup must appear before the types/globals section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        2,          // bound (ids up to 1)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, rspirv::spirv::Op::TypeStruct as u16), // %1 = OpTypeStruct
        1,
        op(2, rspirv::spirv::Op::DecorationGroup as u16), // OpDecorationGroup %1 (misordered)
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorationGroup
        }
    );
}

#[test]
fn group_decorate_cannot_follow_types_and_globals() {
    // OpGroupDecorate must remain in the annotations section; it is invalid after types/globals.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        3,          // bound (ids up to 2)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations section)
        1,
        op(2, rspirv::spirv::Op::TypeStruct as u16), // %2 = OpTypeStruct
        2,
        op(3, rspirv::spirv::Op::GroupDecorate as u16), // OpGroupDecorate %1 %2 (after types -> error)
        1,
        2,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::GroupDecorate
        }
    );
}

#[test]
fn group_member_decorate_cannot_follow_types_and_globals() {
    // OpGroupMemberDecorate must also stay in the annotations section before types/globals.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        3,          // bound (ids up to 2)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations)
        1,
        op(2, rspirv::spirv::Op::TypeStruct as u16), // %2 = OpTypeStruct
        2,
        op(4, rspirv::spirv::Op::GroupMemberDecorate as u16), // OpGroupMemberDecorate %1 %2 0 (after types -> error)
        1,
        2,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::GroupMemberDecorate
        }
    );
}

#[test]
fn decorate_id_cannot_follow_types_and_globals() {
    // OpDecorateId belongs to the annotations section; placing it after globals is invalid.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, rspirv::spirv::Op::TypeInt as u16), // %1 = OpTypeInt 32 0
        1,
        32,
        0,
        op(4, rspirv::spirv::Op::TypePointer as u16), // %2 = OpTypePointer Function %1
        2,
        rspirv::spirv::StorageClass::Function as u32,
        1,
        op(4, rspirv::spirv::Op::Constant as u16), // %3 = OpConstant %1 4
        1,
        3,
        4,
        op(4, rspirv::spirv::Op::Variable as u16), // %4 = OpVariable %2 Function
        2,
        4,
        rspirv::spirv::StorageClass::Function as u32,
        op(4, rspirv::spirv::Op::DecorateId as u16), // OpDecorateId %4 AlignmentId %3 (after types/globals -> error)
        4,
        rspirv::spirv::Decoration::AlignmentId as u32,
        3,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorateId
        }
    );
}

#[test]
fn decorate_string_cannot_follow_types_and_globals() {
    // OpDecorateString must appear in the annotations section before types/globals.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, rspirv::spirv::Op::TypeInt as u16), // %1 = OpTypeInt 32 0
        1,
        32,
        0,
        op(4, rspirv::spirv::Op::TypePointer as u16), // %2 = OpTypePointer Function %1
        2,
        rspirv::spirv::StorageClass::Function as u32,
        1,
        op(4, rspirv::spirv::Op::Variable as u16), // %3 = OpVariable %2 Function
        2,
        3,
        rspirv::spirv::StorageClass::Function as u32,
        op(4, rspirv::spirv::Op::DecorateString as u16), // OpDecorateString %3 UserSemantic "foo" (after globals -> error)
        3,
        rspirv::spirv::Decoration::UserSemantic as u32,
        0x006f_6f66, // "foo"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorateString
        }
    );
}

#[test]
fn member_decorate_string_cannot_follow_types_and_globals() {
    // OpMemberDecorateString also belongs to the annotations section and must not follow types/globals.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        4,          // bound (ids up to 3)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, rspirv::spirv::Op::TypeInt as u16), // %1 = OpTypeInt 32 0
        1,
        32,
        0,
        op(3, rspirv::spirv::Op::TypeStruct as u16), // %2 = OpTypeStruct %1
        2,
        1,
        op(5, rspirv::spirv::Op::MemberDecorateString as u16), // OpMemberDecorateString %2 0 UserSemantic "foo" (after type -> error)
        2,
        0,
        rspirv::spirv::Decoration::UserSemantic as u32,
        0x006f_6f66, // "foo"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemberDecorateString
        }
    );
}

#[test]
fn decorations_must_follow_entry_points() {
    // Annotations must not precede the entry-point section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, rspirv::spirv::Op::Decorate as u16), // OpDecorate %1 RelaxedPrecision (before entry point -> error)
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(2, rspirv::spirv::Op::TypeVoid as u16), // %1 = OpTypeVoid
        1,
        op(3, rspirv::spirv::Op::TypeFunction as u16), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, rspirv::spirv::Op::Function as u16), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253),                                  // OpReturn
        op(1, 56),                                   // OpFunctionEnd
        op(5, rspirv::spirv::Op::EntryPoint as u16), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::EntryPoint
        }
    );
}

#[test]
fn extensions_cannot_follow_entry_points() {
    // OpExtension must appear before the entry-point section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %1 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        1,
        0x6e69_616d, // "main"
        0,
        op(8, rspirv::spirv::Op::Extension as u16), // OpExtension "SPV_GOOGLE_decorate_string" (after entry point -> error)
        0x5f56_5053,                                // "SPV_"
        0x474f_4f47,                                // "GOOG"
        0x645f_454c,                                // "LE_d"
        0x726f_6365,                                // "ecor"
        0x5f65_7461,                                // "ate_"
        0x6972_7473,                                // "stri"
        0x0000_676e,                                // "ng\0"
        op(2, 19),                                  // OpTypeVoid %2
        2,
        op(3, 33), // OpTypeFunction %3 %2
        3,
        2,
        op(5, 54), // OpFunction %1 None %3
        2,
        1,
        0,
        3,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn decorations_cannot_appear_inside_functions() {
    // Hand-built binary with a decoration inside the function body to ensure layout checking
    // rejects annotations in the function section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
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
        op(5, 54),  // OpFunction %1 %4 None %2
        1,          // result type
        4,          // result id
        0,          // FunctionControl None
        2,          // function type
        op(2, 248), // OpLabel %3
        3,
        op(3, 71), // OpDecorate %3 RelaxedPrecision (illegal in function section)
        3,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Decorate
        }
    );
}

#[test]
fn decorations_cannot_precede_memory_model() {
    // Missing OpMemoryModel with a decoration recorded before any other violation should
    // surface an InstructionBeforeMemoryModel error referencing the decoration opcode.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        2,          // bound (ids up to 1)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 71), // OpDecorate %1 RelaxedPrecision
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(2, 19), // OpTypeVoid %1 (appears after the decoration but still before memory model)
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::InstructionBeforeMemoryModel {
            opcode: rspirv::spirv::Op::Decorate
        }
    );
}

#[test]
fn member_decorations_cannot_appear_inside_functions() {
    // MemberDecorate belongs to the annotations section; placing it inside a function should
    // be rejected by layout validation.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        7,          // bound (ids up to 6)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %2 32 0
        2,
        32,
        0,
        op(3, 30), // OpTypeStruct %1 %2
        1,
        2,
        op(2, 19), // OpTypeVoid %3
        3,
        op(3, 33), // OpTypeFunction %4 %3
        4,
        3,
        op(5, 54), // OpFunction %3 %5 None %4
        3,
        5,
        0,
        4,
        op(2, 248), // OpLabel %6
        6,
        op(4, 72), // OpMemberDecorate %1 0 RowMajor (inside function -> error)
        1,
        0,
        rspirv::spirv::Decoration::RowMajor as u32,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemberDecorate
        }
    );
}

#[test]
fn member_decorate_cannot_follow_functions() {
    // MemberDecorate must appear in the annotations section; placing it after functions should
    // trigger a layout error.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        7,          // bound (ids up to 6)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %2 32 0
        2,
        32,
        0,
        op(3, 30), // OpTypeStruct %1 %2
        1,
        2,
        op(2, 19), // OpTypeVoid %3
        3,
        op(3, 33), // OpTypeFunction %4 %3
        4,
        3,
        op(5, 54), // OpFunction %3 %5 None %4
        3,
        5,
        0,
        4,
        op(2, 248), // OpLabel %6
        6,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(4, 72),  // OpMemberDecorate %1 0 RowMajor (after functions -> error)
        1,
        0,
        rspirv::spirv::Decoration::RowMajor as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemberDecorate
        }
    );
}

#[test]
fn decoration_group_cannot_appear_inside_functions() {
    // OpDecorationGroup belongs to the annotations section; ensure it is rejected when placed
    // inside a function body.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(2, 73), // OpDecorationGroup %5 (illegal inside function section)
        5,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorationGroup
        }
    );
}

#[test]
fn decoration_group_cannot_follow_functions() {
    // OpDecorationGroup must appear in the annotations section before functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(2, 73),  // OpDecorationGroup %5 (after functions -> error)
        5,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorationGroup
        }
    );
}

#[test]
fn group_decorate_cannot_follow_functions() {
    // OpGroupDecorate must stay in the annotations section; placing it after functions should
    // be rejected by the layout pass.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations section)
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
        op(1, 253),                                     // OpReturn
        op(1, 56),                                      // OpFunctionEnd
        op(3, rspirv::spirv::Op::GroupDecorate as u16), // OpGroupDecorate %1 %4 (after functions -> error)
        1,
        4,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::GroupDecorate
        }
    );
}

#[test]
fn group_decorate_cannot_appear_inside_functions() {
    // OpGroupDecorate is an annotation and must not appear in the function section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        7,          // bound (ids up to 6)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations section)
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
        op(3, rspirv::spirv::Op::GroupDecorate as u16), // OpGroupDecorate %1 %4 (inside function -> error)
        1,
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::GroupDecorate
        }
    );
}

#[test]
fn group_member_decorate_cannot_follow_functions() {
    // OpGroupMemberDecorate must remain in the annotations section; placing it after functions
    // should be rejected.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        8,          // bound (ids up to 7)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1
        1,
        op(4, 21), // OpTypeInt %3 32 0
        3,
        32,
        0,
        op(3, 30), // OpTypeStruct %2 %3
        2,
        3,
        op(2, 19), // OpTypeVoid %4
        4,
        op(3, 33), // OpTypeFunction %5 %4
        5,
        4,
        op(5, 54), // OpFunction %4 %6 None %5
        4,
        6,
        0,
        5,
        op(2, 248), // OpLabel %7
        7,
        op(1, 253),                                           // OpReturn
        op(1, 56),                                            // OpFunctionEnd
        op(4, rspirv::spirv::Op::GroupMemberDecorate as u16), // OpGroupMemberDecorate %1 %2 0 (after functions -> error)
        1,
        2,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::GroupMemberDecorate
        }
    );
}

#[test]
fn member_names_cannot_appear_inside_functions() {
    // OpMemberName belongs to the names section; placing it inside a function should be
    // rejected by layout validation.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        7,          // bound (ids up to 6)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %1 32 0
        1,
        32,
        0,
        op(3, 30), // OpTypeStruct %2 %1
        2,
        1,
        op(2, 19), // OpTypeVoid %3
        3,
        op(3, 33), // OpTypeFunction %4 %3
        4,
        3,
        op(5, 54), // OpFunction %3 %5 None %4
        3,
        5,
        0,
        4,
        op(2, 248), // OpLabel %6
        6,
        op(4, 6), // OpMemberName %2 0 "f" (inside function -> error)
        2,
        0,
        0x0000_0066, // "f"
        op(1, 253),  // OpReturn
        op(1, 56),   // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemberName
        }
    );
}

#[test]
fn decorate_string_cannot_appear_inside_functions() {
    // OpDecorateString is an annotation and must not appear in a function body.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        9,          // bound (ids up to 8)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %1 32 0
        1,
        32,
        0,
        op(4, 32), // OpTypePointer %2 Function %1
        2,
        rspirv::spirv::StorageClass::Function as u32,
        1,
        op(2, 19), // OpTypeVoid %3
        3,
        op(4, 33), // OpTypeFunction %4 %3 %2
        4,
        3,
        2,
        op(5, 54), // OpFunction %3 %5 None %4
        3,
        5,
        0,
        4,
        op(2, 248), // OpLabel %6
        6,
        op(4, 59), // OpVariable %2 %7 Function
        2,         // result type
        7,         // result id
        rspirv::spirv::StorageClass::Function as u32,
        op(4, rspirv::spirv::Op::DecorateString as u16), // OpDecorateString %7 UserSemantic "foo" (inside function -> error)
        7,
        rspirv::spirv::Decoration::UserSemantic as u32,
        0x006f_6f66, // "foo"
        op(1, 253),  // OpReturn
        op(1, 56),   // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorateString
        }
    );
}

#[test]
fn decorate_string_cannot_follow_functions() {
    // OpDecorateString must appear in the annotations section before functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        9,          // bound (ids up to 8)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %1 32 0
        1,
        32,
        0,
        op(4, 32), // OpTypePointer %2 Function %1
        2,
        rspirv::spirv::StorageClass::Function as u32,
        1,
        op(2, 19), // OpTypeVoid %3
        3,
        op(4, 33), // OpTypeFunction %4 %3 %2
        4,
        3,
        2,
        op(5, 54), // OpFunction %3 %5 None %4
        3,
        5,
        0,
        4,
        op(2, 248), // OpLabel %6
        6,
        op(4, 59), // OpVariable %2 %7 Function
        2,         // result type
        7,         // result id
        rspirv::spirv::StorageClass::Function as u32,
        op(1, 253),                                      // OpReturn
        op(1, 56),                                       // OpFunctionEnd
        op(4, rspirv::spirv::Op::DecorateString as u16), // OpDecorateString %7 UserSemantic "foo" (after functions -> error)
        7,
        rspirv::spirv::Decoration::UserSemantic as u32,
        0x006f_6f66, // "foo"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorateString
        }
    );
}

#[test]
fn decorate_id_cannot_appear_inside_functions() {
    // OpDecorateId is an annotation and must not appear in the function section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        10,         // bound (ids up to 9)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %1 32 0
        1,
        32,
        0,
        op(4, 32), // OpTypePointer %2 Function %1
        2,
        rspirv::spirv::StorageClass::Function as u32,
        1,
        op(4, 43), // OpConstant %1 4 -> %3
        1,
        3,
        4,
        op(2, 19), // OpTypeVoid %4
        4,
        op(4, 33), // OpTypeFunction %5 %4 %2
        5,
        4,
        2,
        op(5, 54), // OpFunction %4 %6 None %5
        4,
        6,
        0,
        5,
        op(2, 248), // OpLabel %7
        7,
        op(4, 59), // OpVariable %2 %8 Function
        2,
        8,
        rspirv::spirv::StorageClass::Function as u32,
        op(4, rspirv::spirv::Op::DecorateId as u16), // OpDecorateId %8 AlignmentId %3 (inside function -> error)
        8,
        rspirv::spirv::Decoration::AlignmentId as u32,
        3,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorateId
        }
    );
}

#[test]
fn decorate_id_cannot_follow_functions() {
    // OpDecorateId must remain in the annotations section ahead of functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        10,         // bound (ids up to 9)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %1 32 0
        1,
        32,
        0,
        op(4, 32), // OpTypePointer %2 Function %1
        2,
        rspirv::spirv::StorageClass::Function as u32,
        1,
        op(4, 43), // OpConstant %1 4 -> %3
        1,
        3,
        4,
        op(2, 19), // OpTypeVoid %4
        4,
        op(4, 33), // OpTypeFunction %5 %4 %2
        5,
        4,
        2,
        op(5, 54), // OpFunction %4 %6 None %5
        4,
        6,
        0,
        5,
        op(2, 248), // OpLabel %7
        7,
        op(4, 59), // OpVariable %2 %8 Function
        2,
        8,
        rspirv::spirv::StorageClass::Function as u32,
        op(1, 253),                                  // OpReturn
        op(1, 56),                                   // OpFunctionEnd
        op(4, rspirv::spirv::Op::DecorateId as u16), // OpDecorateId %8 AlignmentId %3 (after functions -> error)
        8,
        rspirv::spirv::Decoration::AlignmentId as u32,
        3,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorateId
        }
    );
}

#[test]
fn member_decorate_string_cannot_follow_functions() {
    // OpMemberDecorateString is an annotation and must appear before functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        7,          // bound (ids up to 6)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %1 32 0
        1,
        32,
        0,
        op(3, 30), // OpTypeStruct %2 %1
        2,
        1,
        op(2, 19), // OpTypeVoid %3
        3,
        op(3, 33), // OpTypeFunction %4 %3
        4,
        3,
        op(5, 54), // OpFunction %3 %5 None %4
        3,
        5,
        0,
        4,
        op(2, 248), // OpLabel %6
        6,
        op(1, 253),                                            // OpReturn
        op(1, 56),                                             // OpFunctionEnd
        op(5, rspirv::spirv::Op::MemberDecorateString as u16), // OpMemberDecorateString %2 0 UserSemantic "foo"
        2,                                                     // target
        0,                                                     // member index
        rspirv::spirv::Decoration::UserSemantic as u32,
        0x006f_6f66, // "foo"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemberDecorateString
        }
    );
}

#[test]
fn group_member_decorate_cannot_appear_inside_functions() {
    // OpGroupMemberDecorate is an annotation and must not appear in the function section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        8,          // bound (ids up to 7)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1
        1,
        op(4, 21), // OpTypeInt %3 32 0
        3,
        32,
        0,
        op(3, 30), // OpTypeStruct %2 %3
        2,
        3,
        op(2, 19), // OpTypeVoid %4
        4,
        op(3, 33), // OpTypeFunction %5 %4
        5,
        4,
        op(5, 54), // OpFunction %4 %6 None %5
        4,
        6,
        0,
        5,
        op(2, 248), // OpLabel %7
        7,
        op(4, rspirv::spirv::Op::GroupMemberDecorate as u16), // OpGroupMemberDecorate %1 %2 0 (inside function -> error)
        1,
        2,
        0,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::GroupMemberDecorate
        }
    );
}

#[test]
fn capability_cannot_appear_inside_functions() {
    // Capabilities belong to the module header; placing one in the function section should
    // trigger a layout error.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
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
        op(5, 54),  // OpFunction %1 %3 None %2
        1,          // result type
        3,          // result id
        0,          // FunctionControl None
        2,          // function type
        op(2, 248), // OpLabel %4
        4,
        op(2, 17), // OpCapability Kernel (illegal inside function)
        rspirv::spirv::Capability::Kernel as u32,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn extension_cannot_appear_inside_functions() {
    // Extensions belong to the early module sections; reject an extension in the function body.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
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
        op(5, 54),  // OpFunction %1 %3 None %2
        1,          // result type
        3,          // result id
        0,          // FunctionControl None
        2,          // function type
        op(2, 248), // OpLabel %4
        4,
        op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (illegal inside function)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn extension_cannot_follow_functions() {
    // Extensions must appear before functions; placing one after functions should be rejected.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(8, 10),  // OpExtension "SPV_GOOGLE_decorate_string" (after functions -> error)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn source_cannot_appear_inside_functions() {
    // Debug/Source instructions must not appear inside function bodies.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        op(5, 54),  // OpFunction %1 %3 None %2
        1,          // result type
        3,          // result id
        0,          // FunctionControl None
        2,          // function type
        op(2, 248), // OpLabel %4
        4,
        op(3, 3), // OpSource GLSL 450 (illegal inside function)
        rspirv::spirv::SourceLanguage::GLSL as u32,
        450,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Source
        }
    );
}

#[test]
fn source_cannot_follow_functions() {
    // Debug/Source instructions must stay in the debug section before functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(3, 3),   // OpSource GLSL 450 (after functions -> error)
        rspirv::spirv::SourceLanguage::GLSL as u32,
        450,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Source
        }
    );
}

#[test]
fn source_extension_cannot_appear_inside_functions() {
    // OpSourceExtension must remain in the Debug1 section, not inside functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(2, 4), // OpSourceExtension "ext" (illegal inside function)
        0x0074_7865,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::SourceExtension
        }
    );
}

#[test]
fn source_extension_cannot_follow_functions() {
    // OpSourceExtension belongs to Debug1 and must appear before functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(2, 4),   // OpSourceExtension "ext" (after functions -> error)
        0x0074_7865,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::SourceExtension
        }
    );
}

#[test]
fn source_continued_cannot_appear_inside_functions() {
    // OpSourceContinued must remain in the Debug1 section, not inside functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(2, 2), // OpSourceContinued "c" (illegal inside function)
        0x0000_0063,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::SourceContinued
        }
    );
}

#[test]
fn source_continued_cannot_follow_functions() {
    // OpSourceContinued belongs to Debug1 and must appear before functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(2, 2),   // OpSourceContinued "c" (after functions -> error)
        0x0000_0063,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::SourceContinued
        }
    );
}

#[test]
fn memory_model_cannot_appear_inside_functions() {
    // OpMemoryModel must appear before functions; reject it inside a function body.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54),  // OpFunction %1 %3 None %2
        1,          // result type
        3,          // result id
        0,          // FunctionControl None
        2,          // function type
        op(2, 248), // OpLabel %4
        4,
        op(3, 14), // OpMemoryModel Logical GLSL450 (illegal inside function)
        0,
        1,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(error, ValidationError::FunctionBeforeMemoryModel);
}

#[test]
fn ext_inst_import_cannot_appear_inside_functions() {
    // Imported instruction sets must be declared before functions; reject occurrences in the
    // function section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
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
        op(5, 54),  // OpFunction %1 %3 None %2
        1,          // result type
        3,          // result id
        0,          // FunctionControl None
        2,          // function type
        op(2, 248), // OpLabel %4
        4,
        op(3, rspirv::spirv::Op::ExtInstImport as u16), // OpExtInstImport %5 "G" (illegal inside function)
        5,
        0x0000_0047, // "G"
        op(1, 253),  // OpReturn
        op(1, 56),   // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn ext_inst_import_cannot_follow_functions() {
    // Imported instruction sets must precede functions; reject when placed after function
    // definitions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253),                                     // OpReturn
        op(1, 56),                                      // OpFunctionEnd
        op(3, rspirv::spirv::Op::ExtInstImport as u16), // OpExtInstImport %5 "G" (after functions -> error)
        5,
        0x0000_0047, // "G"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn ext_inst_import_cannot_follow_types_and_globals() {
    // Imported instruction sets must appear before the types-and-globals section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // %1 = OpTypeVoid (enters TypesGlobals)
        1,
        op(3, rspirv::spirv::Op::ExtInstImport as u16), // %2 = OpExtInstImport "G" (misordered after types)
        2,
        0x0000_0047, // "G"
        op(3, 33),   // %3 = OpTypeFunction %1
        3,
        1,
        op(5, 54), // %4 = OpFunction %1 None %3
        1,
        4,
        0,
        3,
        op(2, 248), // %5 = OpLabel
        5,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("OpExtInstImport must precede types/globals");
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn member_decorate_string_cannot_appear_inside_functions() {
    // OpMemberDecorateString must stay in the annotations section; placing it inside a
    // function body should be rejected.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        8,          // bound (ids up to 7)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %1 32 0
        1,
        32,
        0,
        op(3, 30), // OpTypeStruct %2 %1
        2,
        1,
        op(2, 19), // OpTypeVoid %3
        3,
        op(3, 33), // OpTypeFunction %4 %3
        4,
        3,
        op(5, 54), // OpFunction %3 %5 None %4
        3,
        5,
        0,
        4,
        op(2, 248), // OpLabel %6
        6,
        op(5, rspirv::spirv::Op::MemberDecorateString as u16), // OpMemberDecorateString %2 0 UserSemantic "foo" (inside function -> error)
        2,
        0,
        rspirv::spirv::Decoration::UserSemantic as u32,
        0x006f_6f66, // "foo"
        op(1, 253),  // OpReturn
        op(1, 56),   // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemberDecorateString
        }
    );
}

#[test]
fn member_name_cannot_follow_functions() {
    // OpMemberName must remain in the names section; placing it after functions should be
    // rejected by layout validation.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        7,          // bound (ids up to 6)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 21), // OpTypeInt %1 32 0
        1,
        32,
        0,
        op(3, 30), // OpTypeStruct %2 %1
        2,
        1,
        op(2, 19), // OpTypeVoid %3
        3,
        op(3, 33), // OpTypeFunction %4 %3
        4,
        3,
        op(5, 54), // OpFunction %3 %5 None %4
        3,
        5,
        0,
        4,
        op(2, 248), // OpLabel %6
        6,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(4, 6),   // OpMemberName %2 0 "f" (after functions -> error)
        2,
        0,
        0x0000_0066, // "f"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemberName
        }
    );
}

#[test]
fn debug_names_cannot_appear_inside_functions() {
    // Hand-built binary with OpName in the function section to ensure it is rejected.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(3, 5), // OpName %4 "fn" (invalid inside function section)
        4,
        0x006e_0066, // "fn"
        op(1, 253),  // OpReturn
        op(1, 56),   // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Name
        }
    );
}

#[test]
fn execution_mode_must_follow_entry_point() {
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 16), // OpExecutionMode %1 OriginUpperLeft (before EntryPoint)
        1,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(5, 15), // OpEntryPoint Fragment %1 "main"
        rspirv::spirv::ExecutionModel::Fragment as u32,
        1,
        0x6e69616d,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::EntryPoint
        }
    );
}

#[test]
fn capability_cannot_follow_entry_points() {
    // Capabilities must be declared before entry points.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(2, 17), // OpCapability Geometry (misordered after entry point)
        rspirv::spirv::Capability::Geometry as u32,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn ext_inst_import_cannot_follow_entry_points() {
    // Imported instruction sets must appear before entry points.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        0x0006000b, // OpExtInstImport %5 "GLSL.std.450" (misordered after entry point)
        5,
        0x4c53_4c47, // "GLSL"
        0x6474_732e, // ".std"
        0x3035_342e, // ".450"
        0,           // null terminator
        op(2, 19),   // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn memory_model_cannot_follow_entry_points() {
    // OpMemoryModel must be declared before the entry-point section.
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        5,           // bound (ids up to 4)
        0,           // schema
        op(2, 17),   // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(5, 15), // OpEntryPoint Vertex %3 "main" (before memory model -> error)
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::MemoryModel
        }
    );
}

#[test]
fn entry_points_must_precede_debug_names() {
    // OpEntryPoint must appear before the debug/names sections.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(4, 5), // OpName %3 "main" (debug names before entry point)
        3,
        0x6e69_616d, // "main"
        0,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::EntryPoint
        }
    );
}

#[test]
fn execution_modes_must_precede_debug_names() {
    // OpExecutionMode must appear before the debug/names sections.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(4, 5), // OpName %3 "main" (debug names before execution mode)
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft (after debug names -> error)
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExecutionMode
        }
    );
}

#[test]
fn entry_points_must_precede_debug_instructions() {
    // Debug/source instructions must not appear before entry points.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 3), // OpSource Unknown 0 (debug instruction before entry point -> error)
        0,
        0,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::EntryPoint
        }
    );
}

#[test]
fn execution_modes_must_precede_debug_instructions() {
    // Debug/source instructions must not appear before execution modes.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 3), // OpSource Unknown 0 (debug instruction before execution mode -> error)
        0,
        0,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExecutionMode
        }
    );
}

#[test]
fn entry_points_cannot_follow_types_and_globals() {
    // Types/globals belong after entry points.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(5, 15),  // OpEntryPoint Vertex %3 "main" (misordered after types/globals)
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::EntryPoint
        }
    );
}

#[test]
fn execution_modes_cannot_follow_annotations() {
    // Execution modes must appear before the annotations section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 71), // OpDecorate %3 RelaxedPrecision (annotations before execution mode)
        3,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft (misordered after annotations)
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExecutionMode
        }
    );
}

#[test]
fn execution_modes_cannot_follow_types_and_globals() {
    // Execution modes must appear before the types/globals section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2 (types/globals section already begun)
        1,
        3,
        0,
        2,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft (misordered after types/globals)
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExecutionMode
        }
    );
}

#[test]
fn execution_modes_cannot_follow_functions() {
    // Execution modes must appear before function bodies.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(3, 16),  // OpExecutionMode %3 OriginUpperLeft (after functions -> error)
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExecutionMode
        }
    );
}

#[test]
fn capability_cannot_follow_execution_modes() {
    // Capabilities must precede the execution-mode section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(2, 17), // OpCapability Geometry (misordered after execution mode)
        rspirv::spirv::Capability::Geometry as u32,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn extension_cannot_follow_execution_modes() {
    // Extensions must precede the execution-mode section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered after execution mode)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn ext_inst_import_cannot_follow_execution_modes() {
    // Imported instruction sets must precede the execution-mode section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        7,          // bound (ids up to 6)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        0x0006000b, // OpExtInstImport %5 "GLSL.std.450" (misordered after execution mode)
        5,
        0x4c53_4c47, // "GLSL"
        0x6474_732e, // ".std"
        0x3035_342e, // ".450"
        0,           // null terminator
        op(2, 19),   // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn conditional_extension_cannot_follow_execution_modes() {
    // Conditional extensions must precede the execution-mode section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(3, rspirv::spirv::Op::ConditionalExtensionINTEL as u16), // misordered after execution mode
        0x0000_0058,                                                // "X"
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_capability_cannot_follow_execution_modes() {
    // Conditional capabilities must also precede execution modes.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, 16), // OpExecutionMode %3 OriginUpperLeft
        3,
        rspirv::spirv::ExecutionMode::OriginUpperLeft as u32,
        op(3, rspirv::spirv::Op::ConditionalCapabilityINTEL as u16), // misordered after execution mode
        rspirv::spirv::Capability::InputAttachment as u32,
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn entry_points_cannot_follow_annotations() {
    // Entry points must appear before the annotations section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 71), // OpDecorate %3 RelaxedPrecision (annotations before entry point)
        3,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(5, 15), // OpEntryPoint Vertex %3 "main" (misordered after annotations)
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::EntryPoint
        }
    );
}

#[test]
fn entry_points_cannot_follow_functions() {
    // Entry points cannot trail function definitions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(5, 15),  // OpEntryPoint Vertex %3 "main" (misordered after functions)
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::EntryPoint
        }
    );
}

#[test]
fn conditional_capability_cannot_follow_entry_points() {
    // Conditional capabilities must also be declared before entry points.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, rspirv::spirv::Op::ConditionalCapabilityINTEL as u16), // misordered after entry point
        rspirv::spirv::Capability::InputAttachment as u32,
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn conditional_extension_cannot_follow_entry_points() {
    // Conditional extensions must be declared before entry points.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint Vertex %3 "main"
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(3, rspirv::spirv::Op::ConditionalExtensionINTEL as u16), // misordered after entry point
        0x0000_0058,                                                // "X"
        0,
        op(2, 19), // %1 = OpTypeVoid
        1,
        op(3, 33), // %2 = OpTypeFunction %1
        2,
        1,
        op(5, 54), // %3 = OpFunction %1 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn sampler_image_address_mode_must_precede_entry_points() {
    // The text assembler rejects this ordering, so keep a hand-crafted binary with
    // OpSamplerImageAddressingModeNV placed after OpEntryPoint to exercise the validator.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version 1.0
        0,          // generator
        5,          // bound (ids 1..4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(2, 17), // OpCapability BindlessTextureNV
        rspirv::spirv::Capability::BindlessTextureNV as u32,
        op(7, 10), // OpExtension "SPV_NV_bindless_texture"
        0x5f56_5053,
        0x625f_564e,
        0x6c64_6e69,
        0x5f73_7365,
        0x7478_6574,
        0x0065_7275,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(5, 15), // OpEntryPoint GLCompute %3 "main"
        rspirv::spirv::ExecutionModel::GLCompute as u32,
        3,
        0x6e69616d,
        0,
        op(2, rspirv::spirv::Op::SamplerImageAddressingModeNV as u16), // OpSamplerImageAddressingModeNV 64 (misordered after entry point)
        64,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::SamplerImageAddressingModeNV
        }
    );
}

#[test]
fn sampler_image_address_mode_is_required_when_bindless_capability_declared() {
    // BindlessTextureNV requires a single SamplerImageAddressingModeNV declaration.
    let text = [
        "OpCapability Shader",
        "OpCapability BindlessTextureNV",
        "OpExtension \"SPV_NV_bindless_texture\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %func \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%func = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let expected = ValidationError::MissingSamplerImageAddressingMode;
    let text_error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("sampler image address mode is required for bindless capability");
    assert_eq!(text_error, expected);
    let binary = assemble_text(&text).expect("assemble");
    let binary_error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("binary should also require sampler image address mode");
    assert_eq!(binary_error, expected);
}

#[test]
fn sampler_image_address_mode_rejects_invalid_bit_width() {
    // The assembler enforces valid bit widths, so use a hand-built binary with an invalid value.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version 1.0
        0,          // generator
        5,          // bound (ids 1..4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(2, 17), // OpCapability BindlessTextureNV
        rspirv::spirv::Capability::BindlessTextureNV as u32,
        op(7, 10), // OpExtension "SPV_NV_bindless_texture"
        0x5f56_5053,
        0x625f_564e,
        0x6c64_6e69,
        0x5f73_7365,
        0x7478_6574,
        0x0065_7275,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, rspirv::spirv::Op::SamplerImageAddressingModeNV as u16), // invalid bit width
        16,
        op(5, 15), // OpEntryPoint GLCompute %3 "main"
        rspirv::spirv::ExecutionModel::GLCompute as u32,
        3,
        0x6e69616d,
        0,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let expected = ValidationError::InvalidSamplerImageAddressingModeBitWidth { bit_width: 16 };
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(error, expected);
}

#[test]
fn sampler_image_address_mode_rejects_duplicates() {
    // Keep two declarations in the binary to bypass assembler canonicalization.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version 1.0
        0,          // generator
        6,          // bound (ids 1..5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(2, 17), // OpCapability BindlessTextureNV
        rspirv::spirv::Capability::BindlessTextureNV as u32,
        op(7, 10), // OpExtension "SPV_NV_bindless_texture"
        0x5f56_5053,
        0x625f_564e,
        0x6c64_6e69,
        0x5f73_7365,
        0x7478_6574,
        0x0065_7275,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, rspirv::spirv::Op::SamplerImageAddressingModeNV as u16), // first declaration
        64,
        op(2, rspirv::spirv::Op::SamplerImageAddressingModeNV as u16), // duplicate
        64,
        op(5, 15), // OpEntryPoint GLCompute %3 "main"
        rspirv::spirv::ExecutionModel::GLCompute as u32,
        3,
        0x6e69616d,
        0,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(error, ValidationError::DuplicateSamplerImageAddressingMode);
}

#[test]
fn validate_module_detects_duplicate_result_ids() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeVoid",
        "%1 = OpTypeInt 32 0",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateResultId {
            id: Id::new(NonZeroU32::new(1).unwrap())
        }
    );
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateResultId {
            id: Id::new(NonZeroU32::new(1).unwrap())
        }
    );
}

#[test]
fn capability_cannot_follow_memory_model() {
    // Capabilities must be declared before the memory model section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 17), // OpCapability Shader (misordered after memory model)
        rspirv::spirv::Capability::Shader as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn extension_must_precede_types_and_globals() {
    // Keep the extension misordered in binary form; the assembler canonicalizes this section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        6,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn extension_cannot_follow_memory_model() {
    // Extensions must appear before the memory model section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered after memory model)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn capability_cannot_follow_extension_section() {
    // Extensions must precede additional capabilities; a capability after an extension is out of order.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string"
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
        op(2, 17), // OpCapability Shader (out of order after extension)
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn capability_cannot_follow_debug_names() {
    // Once debug names begin, capabilities are no longer allowed.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        0x0003_0005, // OpName %1 "x"
        1,
        0x0000_0078,
        op(2, 17), // OpCapability Float64 (misordered after debug section)
        rspirv::spirv::Capability::Float64 as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn capability_cannot_follow_annotations() {
    // Capabilities must be declared before annotations.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations section)
        1,
        op(2, 17), // OpCapability Geometry (misordered after annotations)
        rspirv::spirv::Capability::Geometry as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn conditional_capability_cannot_follow_debug_names() {
    // OpConditionalCapabilityINTEL (capabilities section) cannot be placed after debug names.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 5), // OpName %1 "x" (names/debug2 section)
        1,
        0x0000_0078,
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Shader (misordered after names)
        1,
        rspirv::spirv::Capability::Shader as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn conditional_capability_cannot_follow_annotations() {
    // Conditional capabilities must be declared before annotations.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations)
        1,
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry (misordered after annotations)
        1,
        rspirv::spirv::Capability::Geometry as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn conditional_capability_cannot_follow_extensions() {
    // Conditional capabilities must appear before the extensions/imports sections.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(2, 10), // OpExtension "e"
        0x0000_0065,
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry (misordered after extensions)
        1,
        rspirv::spirv::Capability::Geometry as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn conditional_capability_cannot_follow_ext_inst_import() {
    // Conditional capabilities must precede extension imports.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(3, 11), // OpExtInstImport %1 "G" (imports section)
        1,
        0x0000_0047, // "G"
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry (misordered after imports)
        1,
        rspirv::spirv::Capability::Geometry as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn conditional_capability_cannot_follow_memory_model() {
    // Conditional capabilities must be declared before the memory model.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry (misordered after memory model)
        1,
        rspirv::spirv::Capability::Geometry as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn conditional_capability_cannot_appear_inside_functions() {
    // Conditional capabilities belong to the capabilities section, not inside functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Shader (inside function -> error)
        1,
        rspirv::spirv::Capability::Shader as u32,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn conditional_capability_cannot_follow_functions() {
    // Conditional capabilities must appear before functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253),  // OpReturn
        op(1, 56),   // OpFunctionEnd
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Shader (after functions -> error)
        1,
        rspirv::spirv::Capability::Shader as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn duplicate_conditional_capability_is_rejected() {
    // Duplicate conditional capabilities should be rejected just like regular capabilities.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        2,          // bound (ids up to 1)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry
        1,
        rspirv::spirv::Capability::Geometry as u32,
        op(3, 6250), // duplicate conditional capability
        1,
        rspirv::spirv::Capability::Geometry as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateCapability {
            capability: rspirv::spirv::Capability::Geometry
        }
    );
}

#[test]
fn extension_cannot_follow_debug_names() {
    // Extensions must appear before debug/names/annotations sections.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        0x0003_0005, // OpName %1 "x"
        1,
        0x0000_0078,
        op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered after debug)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn conditional_extension_cannot_follow_debug_names() {
    // OpConditionalExtensionINTEL must appear before debug/names/annotations sections.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        0x0003_0005, // OpName %1 "x"
        1,
        0x0000_0078,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (misordered after debug)
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_extension_cannot_appear_inside_functions() {
    // Conditional extensions belong to the extensions section, not inside functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (inside function -> error)
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_extension_cannot_follow_functions() {
    // Conditional extensions must appear before functions.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253),  // OpReturn
        op(1, 56),   // OpFunctionEnd
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (after functions -> error)
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_extension_cannot_follow_annotations() {
    // OpConditionalExtensionINTEL (extensions section) must not appear after annotations.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations)
        1,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (misordered)
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_extension_cannot_follow_ext_inst_import() {
    // Conditional extensions must precede imported instruction sets.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(6, 11), // OpExtInstImport %1 "GLSL.std.450"
        1,
        0x4c53_4c47, // "GLSL"
        0x2e74_7364, // ".std"
        0x2e30_3534, // ".450"
        0,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (misordered after import)
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_extension_cannot_follow_memory_model() {
    // Conditional extensions must appear before the memory model.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (misordered after memory model)
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn extension_cannot_follow_annotations() {
    // Extensions must appear before annotations.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations)
        1,
        op(8, 10), // OpExtension "SPV_GOOGLE_decorate_string" (misordered after annotations)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ];
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn extension_cannot_follow_ext_inst_import() {
    // Extensions must appear before imports; a later extension is out of order.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 11), // OpExtInstImport %1 "GLSL.std.450"
        1,
        0x004c_5347, // "GLS"
        op(8, 10),   // OpExtension "SPV_GOOGLE_decorate_string" (misordered after imports)
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
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn names_cannot_follow_annotations() {
    // Names/debug instructions must precede annotations.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(3, 5), // OpName %1 "x" (misordered after annotations)
        1,
        0x0000_0078, // "x"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Decorate
        }
    );
}

#[test]
fn capability_cannot_follow_ext_inst_import() {
    // Capabilities must precede extension imports.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(3, 11), // OpExtInstImport %1 "GLSL.std.450" (imports section)
        1,
        0x004c_5347,
        op(2, 17), // OpCapability Geometry (misordered after imports)
        rspirv::spirv::Capability::Geometry as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn ext_inst_import_cannot_follow_debug_names() {
    // OpExtInstImport belongs to the extensions/imports section and cannot follow debug names.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 5), // OpName %1 "x" (names/debug2 section)
        1,
        0x0000_0078,
        op(3, 11), // OpExtInstImport %1 "GLSL.std.450" (misordered after names)
        1,
        0x004c_5347, // "GLS"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn ext_inst_import_cannot_follow_annotations() {
    // OpExtInstImport must precede the annotations section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotations)
        1,
        op(3, 11), // OpExtInstImport %1 "GLSL.std.450" (misordered after annotations)
        1,
        0x004c_5347, // "GLS"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn debug_instructions_cannot_follow_annotations() {
    // OpSource (debug) must not appear after the annotations section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(3, 3), // OpSource Unknown 0 (misordered after annotations)
        0,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Source
        }
    );
}

#[test]
fn string_cannot_follow_annotations() {
    // OpString (debug) must not appear after the annotations section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1 (annotation section)
        1,
        op(3, 7), // OpString %2 "s" (misordered after annotations)
        2,
        0x0000_0073,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::String
        }
    );
}

#[test]
fn string_cannot_follow_names() {
    // Debug1 instructions (OpString) must precede the Names section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 5), // OpName %1 "x" (names section)
        1,
        0x0000_0078,
        op(3, 7), // OpString %2 "s" (misordered after names section)
        2,
        0x0000_0073,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::String
        }
    );
}

#[test]
fn source_extension_cannot_follow_annotations() {
    // OpSourceExtension (debug) must not appear after the annotations section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(2, 4), // OpSourceExtension "ext" (misordered after annotations)
        0x0074_7865,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::SourceExtension
        }
    );
}

#[test]
fn source_continued_cannot_follow_annotations() {
    // OpSourceContinued (debug) must not appear after the annotations section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        2,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 3), // OpSource Unknown 0 (establish debug section)
        0,
        0,
        op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(2, 2), // OpSourceContinued "c" (misordered after annotations)
        0x0000_0063,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::SourceContinued
        }
    );
}

#[test]
fn module_processed_must_precede_annotations() {
    // OpModuleProcessed belongs to the debug section and must precede annotations.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 71), // OpDecorate %1 RelaxedPrecision (annotation section)
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(2, 330),  // OpModuleProcessed "tag" (misordered after annotations)
        0x0067_6174, // "tag"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Decorate
        }
    );
}

#[test]
fn module_processed_must_follow_names() {
    // OpModuleProcessed (Debug3) must appear after the Names section.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 330),  // OpModuleProcessed "tag" (appearing before names)
        0x0067_6174, // "tag"
        op(3, 5),    // OpName %1 "x" (out of order after ModuleProcessed)
        1,
        0x0000_0078,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Name
        }
    );
}

#[test]
fn module_processed_must_precede_types_and_globals() {
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(2, 330),  // OpModuleProcessed "tag" (too late after types/globals)
        0x0067_6174, // "tag"
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ModuleProcessed
        }
    );
}

#[test]
fn ext_inst_import_must_precede_types_and_globals() {
    // Place OpExtInstImport after a type to trigger layout ordering.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        0x0006000b, // OpExtInstImport %2 "GLSL.std.450" (misordered)
        2,
        0x4c53_4c47,
        0x2e73_7464,
        0x3035_342e,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn ext_inst_import_cannot_follow_memory_model() {
    // The assembler canonicalizes layout, so construct the binary manually to keep
    // OpExtInstImport after OpMemoryModel and ensure the validator flags it.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        3,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        0x0006000b, // OpExtInstImport %2 "GLSL.std.450" (too late)
        2,
        0x4c53_4c47,
        0x2e73_7464,
        0x3035_342e,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn validate_module_rejects_duplicate_capability() {
    // The assembler deduplicates capabilities; construct the binary manually to keep both.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(2, 17), // Duplicate OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::DuplicateCapability {
            capability: rspirv::spirv::Capability::Shader
        }
    );
}

#[test]
fn capabilities_cannot_appear_inside_functions_even_when_layout_skipped() {
    // Capabilities must remain in the capabilities section even if block layout checks are
    // skipped.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound
        0,          // schema
        op(3, 14),  // OpMemoryModel Logical GLSL450
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
        op(2, rspirv::spirv::Op::Capability as u16), // OpCapability Shader (inside function)
        rspirv::spirv::Capability::Shader as u32,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let options = ValidationOptions {
        skip_block_layout: true,
        ..ValidationOptions::default()
    };
    let error =
        validate_module_with_options(&binary, TargetEnv::Universal1_5, options).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn conditional_extension_inside_function_is_rejected_even_when_layout_skipped() {
    // Conditional extensions belong in the extensions section, not inside functions.
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
        op(9, rspirv::spirv::Op::ConditionalExtensionINTEL as u16), // OpConditionalExtensionINTEL %1 "SPV_GOOGLE_decorate_string"
        1,
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[0],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[1],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[2],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[3],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[4],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[5],
        EXT_SPV_GOOGLE_DECORATE_STRING_WORDS[6],
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let options = ValidationOptions {
        skip_block_layout: true,
        ..ValidationOptions::default()
    };
    let error =
        validate_module_with_options(&binary, TargetEnv::Universal1_6, options).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_capability_inside_function_is_rejected_even_when_layout_skipped() {
    // Conditional capabilities must also remain in the capability section.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound
        0,          // schema
        op(3, 14),  // OpMemoryModel Logical GLSL450
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
        op(3, rspirv::spirv::Op::ConditionalCapabilityINTEL as u16), // OpConditionalCapabilityINTEL %1 Shader
        1,
        rspirv::spirv::Capability::Shader as u32,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let options = ValidationOptions {
        skip_block_layout: true,
        ..ValidationOptions::default()
    };
    let error =
        validate_module_with_options(&binary, TargetEnv::Universal1_6, options).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalCapabilityINTEL
        }
    );
}

#[test]
fn extinst_import_cannot_appear_after_functions() {
    // OpExtInstImport must precede functions; placing it after a function should trigger a
    // layout error.
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        6,           // bound (ids up to 5)
        0,           // schema
        op(2, 17),   // OpCapability Shader
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
        op(1, 253),                                            // OpReturn
        op(1, 56),                                             // OpFunctionEnd
        (6 << 16) | (rspirv::spirv::Op::ExtInstImport as u32), // OpExtInstImport %5 "GLSL.std.450" (after functions)
        5,
        0x4c53_4c47, // "GLSL"
        0x6474_732e, // ".std"
        0x3035_342e, // ".450"
        0,           // string terminator padding
    ];
    let err = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn extension_cannot_appear_after_types() {
    // Extensions belong in the extensions section; placing one after types/globals should
    // trigger a layout error.
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_GOOGLE_decorate_string\"",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
    ]
    .join("\n");
    let mut words = assemble_text(&text).expect("assemble");
    // Move the extension to appear after the type to trigger layout ordering.
    let mut idx = 5;
    let mut ext_slice: Option<(usize, usize)> = None;
    while idx < words.len() {
        let wc = (words[idx] >> 16) as usize;
        let opcode = words[idx] & 0xffff;
        if opcode == rspirv::spirv::Op::Extension as u32 {
            ext_slice = Some((idx, wc));
            break;
        }
        idx += wc;
    }
    let (start, len) = ext_slice.expect("extension present");
    let ext_inst: Vec<u32> = words.drain(start..start + len).collect();
    words.extend(ext_inst);
    let err = validate_module(&words, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn capability_cannot_appear_after_annotations() {
    // Capabilities must precede debug/names/annotations; relocating a capability past
    // annotations should be rejected as an ordering violation.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%one = OpConstantTrue %void", // malformed on purpose to get an id to decorate
        "OpDecorate %one RelaxedPrecision",
    ]
    .join("\n");
    let mut words = assemble_text(&text).expect("assemble");
    // Move the capability to the end of the module to violate ordering.
    let mut idx = 5;
    let mut cap_slice: Option<(usize, usize)> = None;
    while idx < words.len() {
        let wc = (words[idx] >> 16) as usize;
        let opcode = words[idx] & 0xffff;
        if opcode == rspirv::spirv::Op::Capability as u32 {
            cap_slice = Some((idx, wc));
            break;
        }
        idx += wc;
    }
    let (start, len) = cap_slice.expect("capability present");
    let capability: Vec<u32> = words.drain(start..start + len).collect();
    words.extend(capability);
    let err = validate_module(&words, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn decorate_string_cannot_appear_after_functions() {
    // OpDecorateString instructions belong in the annotations section; placing them after
    // functions should be rejected.
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        5,           // bound (ids up to 4)
        0,           // schema
        op(2, 17),   // OpCapability Shader
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
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253),                                      // OpReturn
        op(1, 56),                                       // OpFunctionEnd
        op(4, rspirv::spirv::Op::DecorateString as u16), // OpDecorateString %3 UserSemantic "foo" (after functions)
        3,
        rspirv::spirv::Decoration::UserSemantic as u32,
        0x006f_6f66, // "foo"
    ];
    let err = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::DecorateString
        }
    );
}

#[test]
fn conditional_extension_cannot_appear_after_annotations() {
    // Conditional extensions belong in the extensions section; placing them after annotations
    // should trigger a layout error.
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        5,           // bound (ids up to 4)
        0,           // schema
        op(2, 17),   // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 71), // OpDecorate %1 RelaxedPrecision (annotations)
        1,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
        op(2, 19), // OpTypeVoid %1 (types)
        1,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (after annotations/types -> error)
        0x5f56_5053, // "SPV_"
        0x474f_4f47, // "GOOG"
        0x645f_454c, // "LE_d"
        0x726f_6365, // "ecor"
        0x5f65_7461, // "ate_"
        0x6972_7473, // "stri"
        0x0000_676e, // "ng\0"
    ];
    let err = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn extension_cannot_appear_after_annotations() {
    // Regular extensions must also precede annotations/names/types/globals.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpName %1 \"x\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
    ]
    .join("\n");
    let mut words = assemble_text(&text).expect("assemble");
    // Append an extension after the annotations/types to trigger ordering.
    words.extend_from_slice(&[
        0x0008_000a, // OpExtension "SPV_GOOGLE_decorate_string"
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ]);
    let err = validate_module(&words, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn conditional_extension_cannot_appear_after_names() {
    // Conditional extensions must not trail the names section.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpName %1 \"x\"",
        "%void = OpTypeVoid",
    ]
    .join("\n");
    let mut words = assemble_text(&text).expect("assemble");
    words.extend_from_slice(&[
        0x0008_1868, // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string"
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ]);
    let err = validate_module(&words, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn extension_cannot_appear_after_names() {
    // Extensions belong before the names section; placing them after OpName should fail.
    let text = [
        "OpCapability Shader",
        "OpExtension \"SPV_GOOGLE_decorate_string\"",
        "OpMemoryModel Logical GLSL450",
        "OpName %1 \"x\"",
        "%void = OpTypeVoid",
    ]
    .join("\n");
    let mut words = assemble_text(&text).expect("assemble");
    // Move extension after the name/type to trigger ordering.
    let mut ext_slice: Option<(usize, usize)> = None;
    let mut idx = 5;
    while idx < words.len() {
        let wc = (words[idx] >> 16) as usize;
        let opcode = words[idx] & 0xffff;
        if opcode == rspirv::spirv::Op::Extension as u32 {
            ext_slice = Some((idx, wc));
            break;
        }
        idx += wc;
    }
    let (start, wc) = ext_slice.expect("extension present");
    let ext: Vec<u32> = words.drain(start..start + wc).collect();
    words.extend(ext);
    let err = validate_module(&words, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Extension
        }
    );
}

#[test]
fn conditional_extension_cannot_appear_after_debug() {
    // Conditional extensions must not trail debug/source instructions.
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        3,           // bound (ids up to 2)
        0,           // schema
        op(2, 17),   // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 3), // OpSource Unknown 0 (debug section)
        0,
        0,
        op(8, 6248), // OpConditionalExtensionINTEL "SPV_GOOGLE_decorate_string" (after debug -> error)
        0x5f56_5053,
        0x474f_4f47,
        0x645f_454c,
        0x726f_6365,
        0x5f65_7461,
        0x6972_7473,
        0x0000_676e,
    ];
    let err = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalExtensionINTEL
        }
    );
}

#[test]
fn conditional_capability_disallowed_in_env() {
    // Conditional capabilities must still respect the target environment allowlist.
    let binary = vec![
        0x07230203,  // magic
        0x00010000,  // version
        0,           // generator
        6,           // bound
        0,           // schema
        op(3, 6250), // OpConditionalCapabilityINTEL %1 Geometry
        1,
        rspirv::spirv::Capability::Geometry as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
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
        ValidationError::DisallowedCapability {
            capability: rspirv::spirv::Capability::Geometry,
            env: TargetEnv::WebGpu0
        }
    );
}
