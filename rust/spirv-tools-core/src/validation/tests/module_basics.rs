use super::*;

#[test]
fn validate_module_accepts_valid_binary() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .expect("valid module");
    MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .expect("valid module");
}

#[test]
fn validated_module_exposes_module_version() {
    use super::effective_spirv_version;
    let binary = vec![
        0x07230203, // magic number
        SpirvVersion::new(1, 5).to_word(),
        0,         // generator
        1,         // bound
        0,         // schema
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    let module = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .expect("module should validate");
    assert_eq!(module.module_version(), SpirvVersion::new(1, 5));
    assert_eq!(module.header().version(), SpirvVersion::new(1, 5));
    assert_eq!(
        module.effective_version(),
        effective_spirv_version(TargetEnv::Universal1_6, SpirvVersion::new(1, 5))
    );
}

#[test]
fn effective_version_reflects_env_clamp_on_valid_module() {
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
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let module = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_0)
        .expect("module should validate with env clamp");
    assert_eq!(module.module_version(), SpirvVersion::new(1, 6));
    assert_eq!(
        module.effective_version(),
        TargetEnv::Vulkan1_0.spirv_version()
    );
}

#[test]
fn validate_module_checks_operand_ids_against_bound() {
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
    let mut binary = assemble_text(&text).expect("assemble");
    // Force a bound that is too small for the function type/result ids.
    binary[3] = 2;
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::IdExceedsBound {
            id: Id::new(NonZeroU32::new(2).unwrap()),
            bound: CheckedBound::new(DeclaredBound(2)).unwrap(),
        }
    );
}

#[test]
fn validate_module_rejects_memory_model_after_function() {
    // The assembler canonicalizes layout, so build a binary with OpMemoryModel placed after the
    // function body to exercise the layout check directly.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version 1.0
        0,          // generator
        5,          // bound
        0,          // schema
        op(2, 17),  // OpCapability Shader
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
        op(3, 14),  // OpMemoryModel Logical GLSL450 (misordered)
        0,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(error, ValidationError::FunctionBeforeMemoryModel);
}

#[test]
fn validate_module_rejects_duplicate_memory_model() {
    // The text path drops duplicate memory models, so keep a hand-built binary to assert the
    // validator rejects them.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        op(2, 17), // OpCapability Shader
        1,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(3, 14), // Duplicate OpMemoryModel
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(error, ValidationError::DuplicateMemoryModel);
}

#[test]
fn function_requires_entry_label() {
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        7,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(4, 21), // OpTypeInt %2 32 0
        2,
        32,
        0,
        op(4, 33), // OpTypeFunction %3 %1 %2
        3,
        1,
        2,
        op(5, 54), // OpFunction %4 None %3 (missing OpLabel)
        1,
        4,
        0,
        3,
        op(3, 55), // OpFunctionParameter %5 %2
        2,
        5,
        op(1, 56), // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::MissingFunctionEntryBlock {
            function: Id::try_from(4).unwrap()
        }
    );
}

#[test]
fn validate_module_rejects_zero_operand_id() {
    // Text assembly forbids %0 operands, so build the binary directly to cover the check.
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
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %0 (invalid operand id)
        2,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::ZeroId {
            kind: IdKind::Operand,
            opcode: rspirv::spirv::Op::TypeFunction
        }
    );
}

#[test]
fn validate_module_rejects_non_zero_schema() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
    ]
    .join("\n");
    let mut binary = assemble_text(&text).expect("assemble");
    // Reserved word must be zero; flip it to a non-zero value to trigger the validation error.
    binary[4] = 1;
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(error, ValidationError::InvalidReservedWord { reserved: 1 });
}

#[test]
fn valid_module_shares_words_without_copying() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let valid = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("valid module");
    let handle = valid.words_handle();
    assert_eq!(handle.as_slice(), binary.as_slice());
    let arc_from_handle = handle.shared();
    let arc_from_valid = valid.words_handle().shared();
    assert_eq!(
        Arc::as_ptr(&arc_from_handle),
        Arc::as_ptr(&arc_from_valid),
        "validated modules should share backing storage"
    );
    let module_words: ModuleWords = ModuleWords::from(arc_from_handle);
    assert_eq!(module_words.as_slice(), binary.as_slice());
}

#[test]
fn block_struct_unused_no_offset_required() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        // Block-decorated struct exists but is never used in a variable
        "OpDecorate %UnusedData Block",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        // No Offset decoration, but that's fine since it's unused
        "%UnusedData = OpTypeStruct %vec4",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    // Should pass - unused Block struct doesn't require offsets
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}
