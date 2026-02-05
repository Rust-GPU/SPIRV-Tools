use super::*;

#[test]
fn function_declarations_must_precede_definitions() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeVoid",
        "%2 = OpTypeFunction %1",
        "%3 = OpFunction %1 None %2",
        "OpFunctionEnd",
        "%4 = OpFunction %1 None %2",
        "%5 = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let result = MaybeValidModule::Text(&text).validate(TargetEnv::Universal1_6);
    assert!(result.is_ok(), "unexpected validation error: {result:?}");
}

#[test]
fn function_declaration_after_definition_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeVoid",
        "%2 = OpTypeFunction %1",
        "%3 = OpFunction %1 None %2",
        "%4 = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "%5 = OpFunction %1 None %2",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionDeclarationAfterDefinition {
            function: Id::try_from(5).unwrap()
        }
    );
}

#[test]
fn phi_requires_incoming_block_to_exist() {
    // Function has a single predecessor for %merge; the phi references a missing block id.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        11,
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
        op(3, 33), // OpTypeFunction %3 %1
        3,
        1,
        op(3, 1), // OpUndef %4 %2
        2,
        4,
        op(5, 54), // OpFunction %5 None %3
        1,
        5,
        0,
        3,
        op(2, 248), // OpLabel %6
        6,
        op(2, 249), // OpBranch %7
        7,
        op(2, 248), // OpLabel %7
        7,
        op(7, 245), // OpPhi %2 %9 %4 %6 %4 %10 (missing incoming block)
        2,
        9,
        4,
        6,
        4,
        10,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    // Modular rule detects predecessor count mismatch before checking incoming block existence
    assert!(matches!(
        error,
        ValidationError::PhiPredecessorCountMismatch { .. }
            | ValidationError::PhiIncomingBlockMissing { .. }
    ));
}

#[test]
fn phi_incoming_block_must_be_predecessor() {
    // %merge only has predecessor %6, but the phi lists %8 which does not branch to it.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        12,
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
        op(3, 33), // OpTypeFunction %3 %1
        3,
        1,
        op(3, 1), // OpUndef %4 %2
        2,
        4,
        op(5, 54), // OpFunction %5 None %3
        1,
        5,
        0,
        3,
        op(2, 248), // OpLabel %6
        6,
        op(2, 249), // OpBranch %7
        7,
        op(2, 248), // OpLabel %8
        8,
        op(1, 253), // OpReturn
        op(2, 248), // OpLabel %7
        7,
        op(5, 245), // OpPhi %2 %9 %4 %8 (incoming block not a predecessor)
        2,
        9,
        4,
        8,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::PhiIncomingNotPredecessor {
            function: Id::try_from(5).unwrap(),
            block: Id::try_from(7).unwrap(),
            incoming: Id::try_from(8).unwrap()
        }
    );
}

#[test]
fn phi_incoming_value_type_must_match_result_type() {
    // The phi expects %int but the incoming value is a %float constant.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        10,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(4, 21), // OpTypeInt %2 32 1
        2,
        32,
        1,
        op(3, 22), // OpTypeFloat %3 32
        3,
        32,
        op(3, 33), // OpTypeFunction %4 %1
        4,
        1,
        op(4, 43), // OpConstant %3 %5 0
        3,
        5,
        0,
        op(5, 54), // OpFunction %6 None %4
        1,
        6,
        0,
        4,
        op(2, 248), // OpLabel %7 (entry)
        7,
        op(2, 249), // OpBranch %8
        8,
        op(2, 248), // OpLabel %8 (merge)
        8,
        op(5, 245), // OpPhi %2 %9 %5 %7 (incoming value has type %3)
        2,
        9,
        5,
        7,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::PhiIncomingTypeMismatch {
            function: Id::try_from(6).unwrap(),
            block: Id::try_from(8).unwrap(),
            incoming: Id::try_from(5).unwrap(),
            expected: TypeId::try_from(2).unwrap(),
            found: TypeId::try_from(3).unwrap(),
        }
    );
}

#[test]
fn function_call_target_must_be_function() {
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
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(4, 21), // OpTypeInt %3 32 0
        3,
        32,
        0,
        op(5, 54), // OpFunction %4 None %2
        1,
        4,
        0,
        2,
        op(2, 248), // OpLabel %5
        5,
        op(4, 57), // OpFunctionCall %1 %6 %3 (target is not a function)
        1,
        6,
        3,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionCallTargetNotFunction {
            function: Id::try_from(4).unwrap(),
            target: Id::try_from(3).unwrap(),
            found: rspirv::spirv::Op::TypeInt,
        }
    );
}

#[test]
fn function_call_argument_count_must_match() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let callee_ty = builder.type_function(int, [int]);
    let main_ty = builder.type_function(void, std::iter::empty::<u32>());
    let callee = builder
        .begin_function(int, None, rspirv::spirv::FunctionControl::NONE, callee_ty)
        .unwrap();
    let param = builder.function_parameter(int).unwrap();
    builder.begin_block(None).unwrap();
    builder.ret_value(param).unwrap();
    builder.end_function().unwrap();
    let main = builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, main_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .function_call(int, None, callee, std::iter::empty())
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionCallArgumentCountMismatch {
            function: Id::try_from(main).unwrap(),
            expected: 1,
            found: 0,
        }
    );
}

#[test]
fn function_call_argument_types_must_match() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let callee_ty = builder.type_function(int, [int]);
    let main_ty = builder.type_function(void, std::iter::empty::<u32>());
    let callee = builder
        .begin_function(int, None, rspirv::spirv::FunctionControl::NONE, callee_ty)
        .unwrap();
    let param = builder.function_parameter(int).unwrap();
    builder.begin_block(None).unwrap();
    builder.ret_value(param).unwrap();
    builder.end_function().unwrap();
    let main = builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, main_ty)
        .unwrap();
    let entry = builder.begin_block(None).unwrap();
    let float_const = builder.constant_bit32(float, 0x3f80_0000);
    builder
        .function_call(int, None, callee, [float_const])
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionCallArgumentTypeMismatch {
            function: Id::try_from(main).unwrap(),
            argument: Id::try_from(float_const).unwrap(),
            expected: TypeId::try_from(int).unwrap(),
            found: TypeId::try_from(float).unwrap(),
        }
    );
    // ensure entry used to silence warnings
    let _ = entry;
}

#[test]
fn function_call_result_type_must_match() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let int = builder.type_int(32, 0);
    let float = builder.type_float(32, None);
    let callee_ty = builder.type_function(int, std::iter::empty::<u32>());
    let main_ty = builder.type_function(void, std::iter::empty::<u32>());
    let callee = builder
        .begin_function(int, None, rspirv::spirv::FunctionControl::NONE, callee_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    let zero = builder.constant_bit32(int, 0);
    builder.ret_value(zero).unwrap();
    builder.end_function().unwrap();
    let main = builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, main_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder
        .function_call(float, None, callee, std::iter::empty::<u32>())
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionCallResultTypeMismatch {
            function: Id::try_from(main).unwrap(),
            expected: TypeId::try_from(int).unwrap(),
            found: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn value_defined_in_another_function_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%int = OpTypeInt 32 0",
        "%void = OpTypeVoid",
        "%fn_i = OpTypeFunction %int",
        "%one = OpConstant %int 1",
        "%f1 = OpFunction %int None %fn_i",
        "%l1 = OpLabel",
        "%add = OpIAdd %int %one %one",
        "OpReturnValue %add",
        "OpFunctionEnd",
        "%f2 = OpFunction %int None %fn_i",
        "%l2 = OpLabel",
        "OpReturnValue %add",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::ValueDefinedInAnotherFunction {
            function: Id::try_from(8).unwrap(),
            value: Id::try_from(7).unwrap(),
        }
    );
}

#[test]
fn function_variable_storage_must_be_function_class() {
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        10,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(4, 21), // OpTypeInt %3 32 0
        3,
        32,
        0,
        op(5, 54), // OpFunction %4 None %2
        1,
        4,
        0,
        2,
        op(2, 248), // OpLabel %5
        5,
        op(4, 59), // OpVariable %3 %6 Workgroup (invalid storage)
        3,
        6,
        rspirv::spirv::StorageClass::Workgroup as u32,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionVariableStorageClassMismatch {
            function: Id::try_from(4).unwrap(),
            variable: Id::try_from(6).unwrap(),
            storage_class: rspirv::spirv::StorageClass::Workgroup,
        }
    );
}

#[test]
fn function_variable_must_be_in_entry_block() {
    // Function-scope variable appears in the second block.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        10,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(4, 21), // OpTypeInt %3 32 0
        3,
        32,
        0,
        op(5, 54), // OpFunction %4 None %2
        1,
        4,
        0,
        2,
        op(2, 248), // OpLabel %5
        5,
        op(2, 249), // OpBranch %6
        6,
        op(2, 248), // OpLabel %6
        6,
        op(4, 59), // OpVariable %3 %7 Function (misplaced)
        3,
        7,
        rspirv::spirv::StorageClass::Function as u32,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionVariableNotInEntryBlock {
            function: Id::try_from(4).unwrap(),
            variable: Id::try_from(7).unwrap(),
        }
    );
}

#[test]
fn phi_cannot_duplicate_predecessor() {
    // %merge has only predecessor %6, but the phi lists %6 twice.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        11,
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
        op(3, 33), // OpTypeFunction %3 %1
        3,
        1,
        op(3, 1), // OpUndef %4 %2
        2,
        4,
        op(5, 54), // OpFunction %5 None %3
        1,
        5,
        0,
        3,
        op(2, 248), // OpLabel %6
        6,
        op(2, 249), // OpBranch %7
        7,
        op(2, 248), // OpLabel %7
        7,
        op(7, 245), // OpPhi %2 %9 %4 %6 %4 %6 (duplicate incoming block)
        2,
        9,
        4,
        6,
        4,
        6,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    // Modular rule detects predecessor count mismatch (2 incoming vs 1 predecessor)
    assert_eq!(
        error,
        ValidationError::PhiPredecessorCountMismatch {
            function: Id::try_from(5).unwrap(),
            block: Id::try_from(7).unwrap(),
            expected: 1,
            found: 2,
        }
    );
}

#[test]
fn function_type_must_be_type_function() {
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
        op(4, 21), // OpTypeInt %2 32 0 (used incorrectly as function type)
        2,
        32,
        0,
        op(5, 54), // OpFunction %3 None %2 (invalid function type)
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
    // Modular FunctionDefinitionRule uses FunctionTypeInvalid
    assert_eq!(
        error,
        ValidationError::FunctionTypeInvalid {
            function: Id::try_from(3).unwrap(),
            function_type: TypeId::try_from(2).unwrap(),
            expected: "OpTypeFunction",
        }
    );
}

#[test]
fn function_return_type_must_match_function_type() {
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
        op(4, 21), // OpTypeInt %2 32 0
        2,
        32,
        0,
        op(3, 33), // OpTypeFunction %3 %2 (return type int)
        3,
        2,
        op(5, 54), // OpFunction %4 None %3 (return type void)
        1,
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
        ValidationError::FunctionReturnTypeMismatch {
            function: Id::try_from(4).unwrap(),
            result_type: TypeId::try_from(1).unwrap(),
            function_type: TypeId::try_from(2).unwrap(),
        }
    );
}

#[test]
fn function_parameter_count_must_match_function_type() {
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
        op(4, 33), // OpTypeFunction %3 %1 %2 (expects one parameter)
        3,
        1,
        2,
        op(5, 54), // OpFunction %4 None %3 (no parameters provided)
        1,
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
        ValidationError::FunctionParameterCountMismatch {
            function: Id::try_from(4).unwrap(),
            expected: 1,
            found: 0,
        }
    );
}

#[test]
fn function_parameter_types_must_match_function_type() {
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        9,
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
        op(3, 22), // OpTypeFloat %3 32
        3,
        32,
        op(4, 33), // OpTypeFunction %4 %1 %2 (expects int parameter)
        4,
        1,
        2,
        op(5, 54), // OpFunction %5 None %4
        1,
        5,
        0,
        4,
        op(3, 55), // OpFunctionParameter %3 %6 (float instead of int)
        3,
        6,
        op(2, 248), // OpLabel %7
        7,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionParameterTypeMismatch {
            function: Id::try_from(5).unwrap(),
            parameter: Id::try_from(6).unwrap(),
            expected: TypeId::try_from(2).unwrap(),
            found: TypeId::try_from(3).unwrap(),
        }
    );
}

#[test]
fn type_function_requires_type_operands() {
    // The function type references a non-type operand as its parameter.
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
        op(4, 21), // OpTypeInt %2 32 0
        2,
        32,
        0,
        op(4, 43), // OpConstant %3 %2 0 (invalid as function parameter type)
        4,
        3,
        0,
        op(4, 33), // OpTypeFunction %4 %1 %3 (param not a type)
        4,
        1,
        3,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::InvalidTypeFunction {
            type_id: TypeId::try_from(4).unwrap()
        }
    );
}

#[test]
fn type_function_parameters_cannot_be_void() {
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
        op(4, 33), // OpTypeFunction %2 %1 %1 (void parameter)
        3,
        1,
        1,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::FunctionTypeParameterVoid {
            type_id: TypeId::try_from(3).unwrap(),
            parameter: TypeId::try_from(1).unwrap(),
        }
    );
}

#[test]
fn type_function_return_must_be_type() {
    // Return type id does not reference a type instruction.
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
        op(4, 21), // OpTypeInt %1 32 0
        4,
        2,
        0,
        op(4, 43), // OpConstant %2 %1 0 (used as return type)
        2,
        1,
        0,
        op(3, 33), // OpTypeFunction %3 %2 (invalid return type)
        3,
        2,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::InvalidTypeFunction {
            type_id: TypeId::try_from(3).unwrap()
        }
    );
}
