use super::*;

#[test]
fn opcode_helpers_classify_capabilities_and_extensions() {
    use super::instruction_layout::{is_capability_opcode, is_extension_opcode};

    assert!(is_capability_opcode(Op::Capability));
    assert!(is_capability_opcode(Op::ConditionalCapabilityINTEL));
    assert!(is_extension_opcode(Op::Extension));
    assert!(is_extension_opcode(Op::ConditionalExtensionINTEL));
    assert!(!is_capability_opcode(Op::Extension));
    assert!(!is_extension_opcode(Op::Capability));
    assert!(!is_extension_opcode(Op::ExtInstImport));
}

#[test]
fn valid_module_cache_accounts_for_options() {
    use crate::validation::ValidationOptions;
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
    let binary = assemble_text(&text).expect("assemble");
    let mut cache = ValidModuleCache::default();
    let first = cache
        .validate_words_with_options(
            &binary,
            TargetEnv::Universal1_6,
            ValidationOptions::default(),
        )
        .expect("first validation");
    let mut relaxed = ValidationOptions {
        relax_struct_store: true,
        ..ValidationOptions::default()
    };
    relaxed.limits.insert(7, 42);
    let second = cache
        .validate_words_with_options(&binary, TargetEnv::Universal1_6, relaxed)
        .expect("validation with options");
    assert_ne!(
        Arc::as_ptr(&first),
        Arc::as_ptr(&second),
        "options should participate in the cache key"
    );
}

#[test]
fn global_variable_limit_enforced() {
    use crate::validation::{ValidationOptions, LIMIT_MAX_GLOBAL_VARIABLES};
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%ptr = OpTypePointer Uniform %void",
        "%g0 = OpVariable %ptr Uniform",
        "%g1 = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let mut options = ValidationOptions::default();
    options.limits.insert(LIMIT_MAX_GLOBAL_VARIABLES, 1);
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect_err("global variable limit should be enforced");
    assert_eq!(
        err,
        ValidationError::LimitExceeded {
            limit_kind: LIMIT_MAX_GLOBAL_VARIABLES,
            limit: 1,
            found: 2
        }
    );
}

#[test]
fn local_variable_limit_enforced() {
    use crate::validation::{ValidationOptions, LIMIT_MAX_LOCAL_VARIABLES};
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%ptr = OpTypePointer Function %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%l0 = OpVariable %ptr Function",
        "%l1 = OpVariable %ptr Function",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let mut options = ValidationOptions::default();
    options.limits.insert(LIMIT_MAX_LOCAL_VARIABLES, 1);
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect_err("local variable limit should be enforced");
    assert_eq!(
        err,
        ValidationError::LimitExceeded {
            limit_kind: LIMIT_MAX_LOCAL_VARIABLES,
            limit: 1,
            found: 2
        }
    );
}

#[test]
fn control_flow_depth_limit_enforced() {
    use crate::validation::{ValidationOptions, LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH};
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        "OpBranchConditional %bool %then %merge",
        "%then = OpLabel",
        "OpReturn",
        "%merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let mut options = ValidationOptions::default();
    options
        .limits
        .insert(LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH, 0);
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect_err("control flow nesting limit should be enforced");
    assert_eq!(
        err,
        ValidationError::LimitExceeded {
            limit_kind: LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH,
            limit: 0,
            found: 1
        }
    );
}

#[test]
fn selection_merge_requires_conditional_terminator() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        "OpBranch %merge",
        "%merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("selection merges must pair with conditional terminators");
    assert_eq!(
        err,
        ValidationError::InvalidMergeTerminator {
            function: Id::try_from(3).unwrap(),
            block: Id::try_from(4).unwrap(),
            terminator: rspirv::spirv::Op::Branch
        }
    );
}

#[test]
fn bitwise_operands_must_match_result_type() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let bool_ty = b.type_bool();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let bool_const = b.constant_true(bool_ty);
    let iconst = b.constant_bit32(int, 1);
    b.bitwise_and(int, None, bool_const, iconst).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("bitwise operands must match result type");
    assert_eq!(
        err,
        ValidationError::BitwiseOperandTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::BitwiseAnd,
            operand_index: 0,
            result_type: TypeId::try_from(int).unwrap(),
            expected: "int scalar or vector",
        }
    );
}

#[test]
fn logical_ops_require_bool_types() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let iconst = b.constant_bit32(int, 1);
    b.logical_and(int, None, iconst, iconst).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("logical ops require bool operands and result");
    assert_eq!(
        err,
        ValidationError::LogicalResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::LogicalAnd,
            result_type: TypeId::try_from(int).unwrap(),
            expected: "bool scalar or vector",
        }
    );
}

#[test]
fn shift_operands_must_match_result_type() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let float = b.type_float(32, None);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let f = b.constant_bit32(float, 0x3f80_0000);
    let i = b.constant_bit32(int, 1);
    // Mismatched first operand.
    b.shift_left_logical(int, None, f, i).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("shift operands must match result type");
    assert_eq!(
        err,
        ValidationError::BitwiseOperandTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::ShiftLeftLogical,
            operand_index: 0,
            result_type: TypeId::try_from(int).unwrap(),
            expected: "int scalar or vector",
        }
    );
}

#[test]
fn shift_count_can_have_different_bit_width() {
    // Per SPIR-V spec, the Shift operand only needs to match the dimension
    // (scalar vs vector with same component count), not the bit width.
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Int16);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let int32 = b.type_int(32, 0);
    let int16 = b.type_int(16, 0);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let _main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let _header = b.begin_block(None).unwrap();
    let lhs = b.constant_bit32(int32, 1);
    let count = b.constant_bit32(int16, 1);
    b.shift_left_logical(int32, None, lhs, count).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    // This should pass - different bit width for shift count is allowed
    words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("shift count with different bit width should be valid");
}

#[test]
fn shift_count_vector_shape_must_match_value() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let vec_ty = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let zero = b.constant_bit32(int, 0);
    let vec = b.constant_composite(vec_ty, [zero, zero]);
    let scalar_count = b.constant_bit32(int, 1);
    b.shift_right_logical(vec_ty, None, vec, scalar_count)
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("shift count shape must match vector value");
    assert_eq!(
        err,
        ValidationError::BitwiseDimensionMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::ShiftRightLogical,
            operand_name: "Shift",
            result_type: TypeId::try_from(vec_ty).unwrap(),
        }
    );
}

#[test]
fn bitwise_ops_require_integer_types() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let float = b.type_float(32, None);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let fconst = b.constant_bit32(float, 0x3f80_0000);
    b.bitwise_or(float, None, fconst, fconst).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("bitwise ops require integer operands and result");
    assert_eq!(
        err,
        ValidationError::BitwiseResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::BitwiseOr,
            result_type: TypeId::try_from(float).unwrap(),
            expected: "int scalar or vector",
        }
    );
}

#[test]
fn logical_ops_reject_non_bool_result_type() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let bool_ty = b.type_bool();
    let int = b.type_int(32, 0);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let t = b.constant_true(bool_ty);
    b.logical_not(int, None, t).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("logical ops require bool result types");
    assert_eq!(
        err,
        ValidationError::LogicalResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::LogicalNot,
            result_type: TypeId::try_from(int).unwrap(),
            expected: "bool scalar or vector",
        }
    );
}

#[test]
fn integer_compare_requires_bool_result_and_matching_operands() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let float = b.type_float(32, None);
    // Int compare but boolean result type is int (invalid) and operand types differ.
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let iconst = b.constant_bit32(int, 1);
    let fconst = b.constant_bit32(float, 0x3f80_0000);
    b.i_equal(int, None, iconst, fconst).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("integer compares require bool result and matching operand types");
    assert_eq!(
        err,
        ValidationError::LogicalResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::IEqual,
            result_type: TypeId::try_from(int).unwrap(),
            expected: "bool scalar or vector",
        }
    );
}

#[test]
fn vector_compare_requires_vector_bool_result() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let float = b.type_float(32, None);
    let vec2 = b.type_vector(float, 2);
    let bool_ty = b.type_bool();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let zero = b.constant_bit32(float, 0);
    let vec_a = b.constant_composite(vec2, [zero, zero]);
    let vec_b = b.constant_composite(vec2, [zero, zero]);
    // Wrong result type: scalar bool instead of vector<bool, 2>.
    b.f_ord_equal(bool_ty, None, vec_a, vec_b).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector compares require vector<bool> result matching operand shape");
    assert_eq!(
        err,
        ValidationError::LogicalResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::FOrdEqual,
            result_type: TypeId::try_from(bool_ty).unwrap(),
            expected: "bool scalar or vector",
        }
    );
}

#[test]
fn compare_operands_must_match_each_other() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.capability(rspirv::spirv::Capability::Int16); // Allow 16-bit integers
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let bool_ty = b.type_bool();
    let int32 = b.type_int(32, 1);
    let int16 = b.type_int(16, 1);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let lhs = b.constant_bit32(int32, 1);
    let rhs = b.constant_bit32(int16, 1);
    b.i_equal(bool_ty, None, lhs, rhs).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("compare operands must have the same type");
    assert_eq!(
        err,
        ValidationError::LogicalBitWidthMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::IEqual,
            result_type: TypeId::try_from(bool_ty).unwrap(),
        }
    );
}

#[test]
fn compare_operands_cannot_mix_signed_and_unsigned_ints() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let bool_ty = b.type_bool();
    let int = b.type_int(32, 1);
    let uint = b.type_int(32, 0);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let lhs = b.constant_bit32(int, 1);
    let rhs = b.constant_bit32(uint, 1);
    b.i_equal(bool_ty, None, lhs, rhs).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("compare operands must have identical types, not just width");
    assert_eq!(
        err,
        ValidationError::LogicalOperandTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::IEqual,
            result_type: TypeId::try_from(bool_ty).unwrap(),
        }
    );
}

#[test]
fn localsizeid_allowed_with_option() {
    use crate::validation::ValidationOptions;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{ExecutionMode, ExecutionModel, FunctionControl};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let fn_ty = builder.type_function(void, []);
    let uint = builder.type_int(32, 0);
    let local_size = builder.constant_bit32(uint, 1);
    let entry_point = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    builder.entry_point(ExecutionModel::GLCompute, entry_point, "main", []);
    builder.execution_mode_id(
        entry_point,
        ExecutionMode::LocalSizeId,
        [local_size, local_size, local_size],
    );
    let words = builder.module().assemble();
    let options = ValidationOptions {
        allow_localsizeid: true,
        ..ValidationOptions::default()
    };
    words
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_1, options)
        .expect("LocalSizeId should be allowed when option is enabled");
}

#[test]
fn offset_texture_operand_disallowed_by_default_in_vulkan() {
    use crate::validation::ValidationOptions;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{
        AddressingModel, Capability, Dim, ExecutionModel, FunctionControl, ImageFormat,
        ImageOperands, MemoryModel,
    };
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(Capability::Shader);
    builder.capability(Capability::ImageGatherExtended);
    builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let void = builder.type_void();
    let float = builder.type_float(32, None);
    let v2float = builder.type_vector(float, 2);
    let i32 = builder.type_int(32, 1);
    let v2i = builder.type_vector(i32, 2);
    let float_zero = builder.constant_bit32(float, 0.0f32.to_bits());
    let int_zero = builder.constant_bit32(i32, 0);
    let zero_offset = builder.constant_composite(v2i, [int_zero, int_zero]);
    let image = builder.type_image(float, Dim::Dim2D, 0, 0, 0, 1, ImageFormat::Unknown, None);
    let sampled_image = builder.type_sampled_image(image);
    let fn_ty = builder.type_function(void, [sampled_image, v2float]);
    let entry = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let image_param = builder.function_parameter(sampled_image).unwrap();
    let coord_param = builder.function_parameter(v2float).unwrap();
    builder.begin_block(None).unwrap();
    builder
        .image_sample_explicit_lod(
            v2float,
            None,
            image_param,
            coord_param,
            ImageOperands::LOD | ImageOperands::OFFSET,
            [
                rspirv::dr::Operand::IdRef(float_zero),
                rspirv::dr::Operand::IdRef(zero_offset),
            ],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    builder.entry_point(ExecutionModel::Fragment, entry, "main", []);
    let binary = builder.module().assemble();
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_2, ValidationOptions::default())
        .expect_err("Offset operand should be restricted to gather ops in Vulkan by default");
    assert_eq!(
        err,
        ValidationError::OffsetTextureOperandDisallowed {
            opcode: rspirv::spirv::Op::ImageSampleExplicitLod
        }
    );
}

#[test]
fn offset_texture_operand_allowed_with_option() {
    use crate::validation::ValidationOptions;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{
        AddressingModel, Capability, Decoration, Dim, ExecutionMode, ExecutionModel,
        FunctionControl, ImageFormat, ImageOperands, MemoryModel, StorageClass,
    };
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(Capability::Shader);
    builder.capability(Capability::ImageGatherExtended);
    builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let void = builder.type_void();
    let float = builder.type_float(32, None);
    let v2float = builder.type_vector(float, 2);
    let v4float = builder.type_vector(float, 4);
    let i32 = builder.type_int(32, 1);
    let v2i = builder.type_vector(i32, 2);
    let float_zero = builder.constant_bit32(float, 0.0f32.to_bits());
    let int_zero = builder.constant_bit32(i32, 0);
    let zero_offset = builder.constant_composite(v2i, [int_zero, int_zero]);
    let coord_init = builder.constant_composite(v2float, [float_zero, float_zero]);
    let image = builder.type_image(float, Dim::Dim2D, 0, 0, 0, 1, ImageFormat::Unknown, None);
    let sampled_image = builder.type_sampled_image(image);
    // Use UniformConstant for sampler/image
    let ptr_sampled_image =
        builder.type_pointer(None, StorageClass::UniformConstant, sampled_image);
    let sampler_var =
        builder.variable(ptr_sampled_image, None, StorageClass::UniformConstant, None);
    builder.decorate(
        sampler_var,
        Decoration::DescriptorSet,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    builder.decorate(
        sampler_var,
        Decoration::Binding,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    let fn_ty = builder.type_function(void, []);
    let entry = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    let sampler_val = builder
        .load(sampled_image, None, sampler_var, None, [])
        .unwrap();
    builder
        .image_sample_explicit_lod(
            v4float,
            None,
            sampler_val,
            coord_init,
            ImageOperands::LOD | ImageOperands::OFFSET,
            [
                rspirv::dr::Operand::IdRef(float_zero),
                rspirv::dr::Operand::IdRef(zero_offset),
            ],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    builder.entry_point(ExecutionModel::Fragment, entry, "main", [sampler_var]);
    builder.execution_mode(entry, ExecutionMode::OriginUpperLeft, []);
    let binary = builder.module().assemble();
    let options = ValidationOptions {
        allow_offset_texture_operand: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_2, options)
        .expect("Offset operand should be allowed when option is enabled");
}

#[test]
fn offset_texture_operand_allowed_before_hlsl_legalization() {
    use crate::validation::ValidationOptions;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{
        AddressingModel, Capability, Decoration, Dim, ExecutionMode, ExecutionModel,
        FunctionControl, ImageFormat, ImageOperands, MemoryModel, StorageClass,
    };
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(Capability::Shader);
    builder.capability(Capability::ImageGatherExtended);
    builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let void = builder.type_void();
    let float = builder.type_float(32, None);
    let v2float = builder.type_vector(float, 2);
    let v4float = builder.type_vector(float, 4);
    let i32 = builder.type_int(32, 1);
    let v2i = builder.type_vector(i32, 2);
    let float_zero = builder.constant_bit32(float, 0.0f32.to_bits());
    let int_zero = builder.constant_bit32(i32, 0);
    let zero_offset = builder.constant_composite(v2i, [int_zero, int_zero]);
    let coord_init = builder.constant_composite(v2float, [float_zero, float_zero]);
    let image = builder.type_image(float, Dim::Dim2D, 0, 0, 0, 1, ImageFormat::Unknown, None);
    let sampled_image = builder.type_sampled_image(image);
    // Use UniformConstant for sampler/image
    let ptr_sampled_image =
        builder.type_pointer(None, StorageClass::UniformConstant, sampled_image);
    let sampler_var =
        builder.variable(ptr_sampled_image, None, StorageClass::UniformConstant, None);
    builder.decorate(
        sampler_var,
        Decoration::DescriptorSet,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    builder.decorate(
        sampler_var,
        Decoration::Binding,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    let fn_ty = builder.type_function(void, []);
    let entry = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    let sampler_val = builder
        .load(sampled_image, None, sampler_var, None, [])
        .unwrap();
    builder
        .image_sample_explicit_lod(
            v4float,
            None,
            sampler_val,
            coord_init,
            ImageOperands::LOD | ImageOperands::OFFSET,
            [
                rspirv::dr::Operand::IdRef(float_zero),
                rspirv::dr::Operand::IdRef(zero_offset),
            ],
        )
        .unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    builder.entry_point(ExecutionModel::Fragment, entry, "main", [sampler_var]);
    builder.execution_mode(entry, ExecutionMode::OriginUpperLeft, []);
    let binary = builder.module().assemble();
    let options = ValidationOptions {
        before_hlsl_legalization: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_2, options)
        .expect("Offset operand should be allowed when using the pre-HLSL legalization option");
}

#[test]
fn bit_field_ops_require_32bit_in_vulkan_by_default() {
    // Vulkan restricts bit field operations (BitFieldInsert, BitFieldSExtract,
    // BitFieldUExtract, BitReverse, BitCount) to 32-bit integers.
    // Basic bitwise ops (Or, Xor, And, Not) and shift ops are NOT restricted.
    use crate::validation::ValidationOptions;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(Capability::Shader);
    builder.capability(Capability::Int64);
    builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let u64_ty = builder.type_int(64, 0);
    let fn_ty = builder.type_function(u64_ty, [u64_ty]);
    builder
        .begin_function(u64_ty, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let a = builder.function_parameter(u64_ty).unwrap();
    builder.begin_block(None).unwrap();
    // BitReverse is one of the restricted operations
    let rev = builder.bit_reverse(u64_ty, None, a).unwrap();
    builder.ret_value(rev).unwrap();
    builder.end_function().unwrap();
    let binary = builder.module().assemble();
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_1, ValidationOptions::default())
        .expect_err("64-bit bit field ops should be disallowed by default in Vulkan");
    assert_eq!(
        err,
        ValidationError::VulkanBitwiseRequires32Bit {
            opcode: rspirv::spirv::Op::BitReverse,
            bit_width: 64
        }
    );
}

#[test]
fn basic_bitwise_ops_allow_64bit_in_vulkan() {
    // Basic bitwise operations (BitwiseOr, BitwiseXor, BitwiseAnd, Not) and
    // shift operations are NOT restricted to 32-bit in Vulkan.
    use crate::validation::ValidationOptions;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(Capability::Shader);
    builder.capability(Capability::Int64);
    builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let u64_ty = builder.type_int(64, 0);
    let fn_ty = builder.type_function(u64_ty, [u64_ty, u64_ty]);
    builder
        .begin_function(u64_ty, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let a = builder.function_parameter(u64_ty).unwrap();
    let b = builder.function_parameter(u64_ty).unwrap();
    builder.begin_block(None).unwrap();
    let or_result = builder.bitwise_or(u64_ty, None, a, b).unwrap();
    builder.ret_value(or_result).unwrap();
    builder.end_function().unwrap();
    let binary = builder.module().assemble();
    // Should pass without any special option - basic bitwise ops are not restricted
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_1, ValidationOptions::default())
        .expect("64-bit basic bitwise ops should be allowed in Vulkan");
}

#[test]
fn bit_field_ops_allow_non_32bit_when_option_enabled() {
    use crate::validation::ValidationOptions;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(Capability::Shader);
    builder.capability(Capability::Int64);
    builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let u64_ty = builder.type_int(64, 0);
    let fn_ty = builder.type_function(u64_ty, [u64_ty]);
    builder
        .begin_function(u64_ty, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let a = builder.function_parameter(u64_ty).unwrap();
    builder.begin_block(None).unwrap();
    let rev = builder.bit_reverse(u64_ty, None, a).unwrap();
    builder.ret_value(rev).unwrap();
    builder.end_function().unwrap();
    let binary = builder.module().assemble();
    let options = ValidationOptions {
        allow_vulkan_32_bit_bitwise: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_1, options)
        .expect("64-bit bit field ops should be allowed when option is enabled");
}

#[test]
fn friendly_name_helpers_format_ids_and_members() {
    use crate::validation::ValidationOptions;
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpName %func \"friendly\"",
        "OpName %S \"Struct\"",
        "OpMemberName %S 0 \"field\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%S = OpTypeStruct %uint",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let options = ValidationOptions {
        use_friendly_names: true,
        ..ValidationOptions::default()
    };
    let module = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect("validation should succeed");
    let names = module.friendly_names().expect("friendly names present");
    let (&named_func_id, _) = names
        .id_names()
        .iter()
        .find(|(_, name)| name.as_str() == "friendly")
        .expect("function name should be present");
    let formatted_func = names.format_id(named_func_id);
    assert!(
        formatted_func.contains("(friendly)"),
        "expected friendly suffix, got {formatted_func}"
    );
    let (&struct_id, _) = names
        .id_names()
        .iter()
        .find(|(_, name)| name.as_str() == "Struct")
        .expect("struct name should be present");
    let formatted_member = names.format_member(struct_id, MemberIndex(0));
    assert!(
        formatted_member.contains("(field)"),
        "expected member friendly suffix, got {formatted_member}"
    );
}

#[test]
fn layout_relaxation_flags_are_accepted() {
    use crate::validation::ValidationOptions;
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
    let binary = assemble_text(&text).expect("assemble");
    let options = ValidationOptions {
        relax_struct_store: true,
        relax_logical_pointer: true,
        relax_block_layout: true,
        uniform_buffer_standard_layout: true,
        scalar_block_layout: true,
        workgroup_scalar_block_layout: true,
        skip_block_layout: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect("layout relaxation flags should be accepted");
}

#[test]
fn skip_block_layout_does_not_bypass_global_layout_ordering() {
    // skip_block_layout only affects block layout checks, not module section ordering.
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version
        0,           // generator
        6,           // bound
        0,           // schema
        op(2, 17),   // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(3, 11), // OpExtInstImport %3 "GLSL.std.450" (misordered after types)
        3,
        0x4c5347,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
    ];
    use crate::validation::ValidationOptions;
    let err = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert!(matches!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    ));
    let options = ValidationOptions {
        skip_block_layout: true,
        ..ValidationOptions::default()
    };
    let err = validate_module_with_options(&binary, TargetEnv::Universal1_6, options).unwrap_err();
    assert_eq!(
        err,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ExtInstImport
        }
    );
}

#[test]
fn logical_pointer_disallows_pointee_storage_class_without_relaxation() {
    let text = [
        "OpCapability Shader",
        "OpCapability VectorComputeINTEL",
        "OpCapability VectorAnyINTEL",
        "OpExtension \"SPV_INTEL_vector_compute\"",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%ptr_uniform_float = OpTypePointer Uniform %float",
        "%ptr_private_ptr_uniform = OpTypePointer Private %ptr_uniform_float",
        "%var = OpVariable %ptr_private_ptr_uniform Private",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("logical pointer rules should reject pointers to Input");
    if let ValidationError::LogicalPointerPointeeStorageClassInvalid {
        pointee_storage_class: rspirv::spirv::StorageClass::Uniform,
        ..
    } = err
    {
    } else {
        panic!("unexpected error: {err:?}");
    }
    let options = ValidationOptions {
        relax_logical_pointer: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect("relax_logical_pointer should permit pointer-to-pointer");
}

#[test]
fn logical_pointer_requires_variable_pointer_capabilities() {
    let text = [
        "OpCapability Shader",
        "OpCapability VectorComputeINTEL",
        "OpCapability VectorAnyINTEL",
        "OpExtension \"SPV_INTEL_vector_compute\"",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpMemoryModel Logical GLSL450",
        "%int = OpTypeInt 32 0",
        "%ptr_sb_int = OpTypePointer StorageBuffer %int",
        "%ptr_private_ptr_sb = OpTypePointer Private %ptr_sb_int",
        "%var = OpVariable %ptr_private_ptr_sb Private",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("missing VariablePointersStorageBuffer capability should error");
    if let ValidationError::LogicalPointerMissingCapability {
        required_capability: rspirv::spirv::Capability::VariablePointersStorageBuffer,
        ..
    } = err
    {
    } else {
        panic!("unexpected error: {err:?}");
    }
    let with_capability = [
        "OpCapability Shader",
        "OpCapability VectorComputeINTEL",
        "OpCapability VectorAnyINTEL",
        "OpCapability VariablePointersStorageBuffer",
        "OpExtension \"SPV_INTEL_vector_compute\"",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpMemoryModel Logical GLSL450",
        "%int = OpTypeInt 32 0",
        "%ptr_sb_int = OpTypePointer StorageBuffer %int",
        "%ptr_private_ptr_sb = OpTypePointer Private %ptr_sb_int",
        "%var = OpVariable %ptr_private_ptr_sb Private",
    ]
    .join("\n");
    assemble_text(&with_capability)
        .expect("assemble")
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("declaring capability should permit pointer-to-pointer");
}

#[test]
fn logical_pointer_rejects_non_function_or_private_storage_class() {
    let text = [
        "OpCapability Shader",
        "OpCapability VariablePointersStorageBuffer",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%ptr_sb_float = OpTypePointer StorageBuffer %float",
        "%ptr_sb_ptr = OpTypePointer StorageBuffer %ptr_sb_float",
        "%var = OpVariable %ptr_sb_ptr StorageBuffer",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate_with_options(
            TargetEnv::Universal1_6,
            ValidationOptions {
                // Skip block layout so we exercise logical-pointer rules directly.
                skip_block_layout: true,
                ..ValidationOptions::default()
            },
        )
        .expect_err("logical pointer should reject non-Function/Private storage class");
    assert!(
        matches!(
            err,
            ValidationError::LogicalPointerInvalidStorageClass {
                storage_class: rspirv::spirv::StorageClass::StorageBuffer,
                ..
            }
        ),
        "expected storage-class diagnostic, got {err:?}"
    );
}

#[test]
fn opload_rejects_pointer_from_composite_extract() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let u32_type = b.type_int(32, 0);
    let fn_type = b.type_function(void, vec![]);
    let ptr_func_u32 = b.type_pointer(None, rspirv::spirv::StorageClass::Function, u32_type);
    // A struct containing a pointer
    let struct_type = b.type_struct(vec![ptr_func_u32]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    let _var = b.variable(
        ptr_func_u32,
        None,
        rspirv::spirv::StorageClass::Function,
        None,
    );
    let ptr_ptr_type = b.type_pointer(None, rspirv::spirv::StorageClass::Function, struct_type);
    let struct_var = b.variable(
        ptr_ptr_type,
        None,
        rspirv::spirv::StorageClass::Function,
        None,
    );
    let struct_val = b.load(struct_type, None, struct_var, None, vec![]).unwrap();
    // Extract a pointer from the struct composite
    let extracted_ptr = b
        .composite_extract(ptr_func_u32, None, struct_val, vec![0])
        .unwrap();
    // Try to load using the extracted pointer - this should fail in Logical addressing
    b.load(u32_type, None, extracted_ptr, None, vec![]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect_err("loading from composite-extracted pointer should fail");
    assert!(
        matches!(
            err,
            ValidationError::NotALogicalPointer {
                instruction: rspirv::spirv::Op::Load,
                source_opcode: rspirv::spirv::Op::CompositeExtract,
                ..
            }
        ),
        "expected NotALogicalPointer error from CompositeExtract, got {err:?}"
    );
}

#[test]
fn opload_accepts_valid_logical_pointer_sources() {
    // From Variable
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Function %u32",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr Function",
        "%val = OpLoad %u32 %var",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("loading from Variable should work");

    // From AccessChain
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%struct = OpTypeStruct %u32",
        "%ptr_struct = OpTypePointer Function %struct",
        "%ptr_u32 = OpTypePointer Function %u32",
        "%c0 = OpConstant %u32 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_struct Function",
        "%chain = OpAccessChain %ptr_u32 %var %c0",
        "%val = OpLoad %u32 %chain",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("loading from AccessChain should work");
}

#[test]
fn opstore_rejects_non_logical_pointer() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let u32_type = b.type_int(32, 0);
    let fn_type = b.type_function(void, vec![]);
    let ptr_func_u32 = b.type_pointer(None, rspirv::spirv::StorageClass::Function, u32_type);
    let struct_type = b.type_struct(vec![ptr_func_u32]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    let ptr_ptr_type = b.type_pointer(None, rspirv::spirv::StorageClass::Function, struct_type);
    let struct_var = b.variable(
        ptr_ptr_type,
        None,
        rspirv::spirv::StorageClass::Function,
        None,
    );
    let struct_val = b.load(struct_type, None, struct_var, None, vec![]).unwrap();
    let extracted_ptr = b
        .composite_extract(ptr_func_u32, None, struct_val, vec![0])
        .unwrap();
    let constant_val = b.constant_bit32(u32_type, 42);
    // Try to store to the extracted pointer
    b.store(extracted_ptr, constant_val, None, vec![]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect_err("storing to composite-extracted pointer should fail");
    assert!(
        matches!(
            err,
            ValidationError::NotALogicalPointer {
                instruction: rspirv::spirv::Op::Store,
                source_opcode: rspirv::spirv::Op::CompositeExtract,
                ..
            }
        ),
        "expected NotALogicalPointer error for Store, got {err:?}"
    );
}

#[test]
fn friendly_names_populated_on_valid_module() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpName %struct \"MyStruct\"",
        "OpMemberName %struct 0 \"field0\"",
        "%uint = OpTypeInt 32 0",
        "%struct = OpTypeStruct %uint",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let valid = binary
        .as_slice()
        .validate_with_options(
            TargetEnv::Universal1_6,
            ValidationOptions {
                use_friendly_names: true,
                ..ValidationOptions::default()
            },
        )
        .expect("validation should succeed");
    let names = valid
        .friendly_names()
        .expect("friendly names should be populated when enabled");
    let struct_id = valid
        .module()
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == rspirv::spirv::Op::TypeStruct)
        .and_then(|inst| inst.result_id)
        .expect("struct should have a result id");
    assert_eq!(names.id(struct_id), Some("MyStruct"));
    assert_eq!(
        names.member(struct_id, MemberIndex(0)),
        Some("field0"),
        "member names should be recorded"
    );
}

#[test]
fn id_bound_limit_is_enforced() {
    use crate::validation::{ValidationOptions, LIMIT_MAX_ID_BOUND};
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
    let binary = assemble_text(&text).expect("assemble");
    let mut options = ValidationOptions::default();
    options.limits.insert(LIMIT_MAX_ID_BOUND, 3);
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect_err("id bound should exceed configured limit");
    assert_eq!(
        err,
        ValidationError::IdBoundExceedsLimit {
            declared: DeclaredBound(5),
            limit: 3
        }
    );
}

#[test]
fn execution_mode_requires_entry_point() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "OpExecutionMode %main OriginUpperLeft",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect_err("execution modes must target entry points");
    assert_eq!(
        error,
        ValidationError::ExecutionModeWithoutEntryPoint {
            function: Id::try_from(3).unwrap()
        }
    );
}

#[test]
fn execution_mode_must_target_entry_point_function() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%helper = OpFunction %void None %fn",
        "%hentry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "OpExecutionMode %helper OriginUpperLeft",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect_err("execution mode should target entry point");
    assert!(matches!(
        error,
        ValidationError::ExecutionModeWithoutEntryPoint { function }
            if function == Id::try_from(4).unwrap()
    ));
}

#[test]
fn execution_mode_accepts_entry_point() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "OpExecutionMode %main OriginUpperLeft",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("execution mode targets entry point");
}

#[test]
fn conditional_entry_point_must_precede_debug_names() {
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
        op(4, 5), // OpName %4 "main"
        4,
        0x6e69_616d,
        0,
        op(6, 6249), // OpConditionalEntryPointINTEL %5 Vertex %4 "main"
        5,
        rspirv::spirv::ExecutionModel::Vertex as u32,
        4,
        0x6e69_616d,
        0,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %1 %4 None %2
        1,
        4,
        0,
        2,
        op(2, 248), // OpLabel %6
        6,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalEntryPointINTEL
        }
    );
}

#[test]
fn conditional_entry_point_cannot_follow_functions() {
    let intel_function_variants_ext = [
        1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
    ];
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        7,          // bound (ids up to 6)
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
        op(6, 6249), // OpConditionalEntryPointINTEL %5 Vertex %3 "main"
        5,
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d,
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalEntryPointINTEL
        }
    );
}

#[test]
fn conditional_entry_point_must_reference_function() {
    let intel_function_variants_ext = [
        1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
    ];
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        8,          // bound (ids up to 7)
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
        op(6, 6249), // OpConditionalEntryPointINTEL %5 Vertex %5 "main"
        5,
        rspirv::spirv::ExecutionModel::Vertex as u32,
        5,
        0x6e69_616d,
        0,
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
        op(5, 54), // OpFunction %1 %6 None %2
        1,
        6,
        0,
        2,
        op(2, 248), // OpLabel %7
        7,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::InvalidEntryPointTarget {
            target: Id::try_from(5).unwrap(),
            opcode: rspirv::spirv::Op::ConstantTrue
        }
    );
}

#[test]
fn conditional_entry_point_cannot_follow_annotations() {
    let intel_function_variants_ext = [
        1599492179, 1163152969, 1969643340, 1769235310, 1985965679, 1634300513, 7566446,
    ];
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
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
        op(2, 73), // OpDecorationGroup %1 (annotations)
        1,
        op(6, 6249), // OpConditionalEntryPointINTEL %2 Vertex %3 "main"
        2,
        rspirv::spirv::ExecutionModel::Vertex as u32,
        3,
        0x6e69_616d, // "main"
        0,
        op(2, 19), // OpTypeVoid %4
        4,
        op(3, 33), // OpTypeFunction %5 %4
        5,
        4,
        op(5, 54), // OpFunction %4 %3 None %5
        4,
        3,
        0,
        5,
        op(2, 248), // OpLabel %6
        6,
        op(1, 253), // OpReturn
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalEntryPointINTEL
        }
    );
}

#[test]
fn conditional_entry_point_cannot_follow_execution_modes() {
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        4,          // bound (ids up to 3)
        0,          // schema
        op(2, Op::Capability as u16),
        Capability::Shader as u32,
        op(3, Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        MemoryModel::GLSL450 as u32,
        op(6, Op::ExecutionMode as u16),
        3, // function id
        ExecutionMode::LocalSize as u32,
        1,
        1,
        1,
        op(6, Op::ConditionalEntryPointINTEL as u16),
        3, // function id
        rspirv::spirv::ExecutionModel::Fragment as u32,
        3,           // function id again
        0x6e69_616d, // "main"
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::ConditionalEntryPointINTEL
        }
    );
}

#[test]
fn extension_before_capability_is_rejected() {
    let binary = vec![
        0x0723_0203,                 // magic
        0x0001_0000,                 // version
        0,                           // generator
        1,                           // bound (no ids)
        0,                           // schema
        op(8, Op::Extension as u16), // OpExtension "SPV_GOOGLE_decorate_string" (before capabilities)
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
fn instruction_requires_capability() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 5);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    builder.type_acceleration_structure_khr();
    let module = builder.module();
    let words = module.assemble();
    let error = words
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("missing RayTracingKHR capability");
    assert_eq!(
        error,
        ValidationError::MissingInstructionCapability {
            opcode: rspirv::spirv::Op::TypeAccelerationStructureKHR,
            required_capability: rspirv::spirv::Capability::RayTracingNV
        }
    );
}

#[test]
fn operand_requires_extension() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%i32 = OpTypeInt 32 0",
        "%ptr_fn_i32 = OpTypePointer Function %i32",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_fn_i32 Function",
        "%val = OpLoad %i32 %var NonPrivatePointer",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("missing vulkan memory model capability");
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
fn validate_module_rejects_zero_result_id() {
    // The assembler never emits id 0; keep this binary hand-crafted to drive the zero-id path.
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
        op(2, 19), // OpTypeVoid with a zero result id
        0,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::ZeroId {
            kind: IdKind::Result,
            opcode: rspirv::spirv::Op::TypeVoid
        }
    );
}

#[test]
fn linkage_attributes_require_function_or_variable() {
    let text = [
        "OpCapability Shader",
        "OpCapability Linkage",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeInt 32 0",
        "OpDecorate %1 LinkageAttributes \"foo\" Import",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble LinkageAttributes decoration");
    let expected = ValidationError::InvalidDecorationTargetKind {
        decoration: rspirv::spirv::Decoration::LinkageAttributes,
        target: Id::try_from(1).unwrap(),
        found: rspirv::spirv::Op::TypeInt,
        expected: DecorationTargetKind::FunctionOrVariable,
    };
    for module in [
        MaybeValidModule::Text(text.as_str()),
        MaybeValidModule::Binary(binary.as_slice()),
    ] {
        let error = module
            .validate(TargetEnv::Universal1_6)
            .expect_err("LinkageAttributes must target functions or variables");
        assert_eq!(error, expected);
    }
}

#[test]
fn mesh_output_primitives_execution_mode_is_accepted() {
    let text = [
        "OpCapability Shader",
        "OpCapability MeshShadingNV",
        "OpExtension \"SPV_NV_mesh_shader\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint MeshNV %main \"main\"",
        "OpExecutionMode %main OutputTrianglesEXT",
        "OpExecutionMode %main OutputVertices 3",
        "OpExecutionMode %main OutputPrimitivesEXT 2",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .expect("Mesh OutputTriangles/OutputVertices/OutputPrimitivesEXT should validate");
}

#[test]
fn output_vertices_requires_geometry_or_tess_or_mesh() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert!(matches!(
        error,
        ValidationError::ExecutionModeRequiresExecutionModel {
            mode: rspirv::spirv::ExecutionMode::OutputVertices,
            execution_model: rspirv::spirv::ExecutionModel::Vertex,
            ..
        }
    ));
}

#[test]
fn output_primitives_ext_requires_mesh_execution_model() {
    let text = [
        "OpCapability Shader",
        "OpCapability Geometry",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Geometry %main \"main\"",
        "OpExecutionMode %main OutputTriangleStrip",
        "OpExecutionMode %main OutputPrimitivesEXT 2",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert!(matches!(
        error,
        ValidationError::ExecutionModeRequiresExecutionModel {
            mode: rspirv::spirv::ExecutionMode::OutputPrimitivesEXT,
            execution_model: rspirv::spirv::ExecutionModel::Geometry,
            ..
        }
    ));
}

#[test]
fn component_spill_overlaps_across_locations_are_rejected_in_vulkan() {
    // First variable occupies location 0 component 3 and spills into location 1 component 0.
    // Second variable explicitly targets location 1 component 0, so they overlap.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %a %b",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%ptr = OpTypePointer Input %vec2",
        "%ptr_scalar = OpTypePointer Input %float",
        "%a = OpVariable %ptr Input",
        "%b = OpVariable %ptr_scalar Input",
        "OpDecorate %a Location 0",
        "OpDecorate %a Component 3",
        "OpDecorate %b Location 1",
        "OpDecorate %b Component 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_err();
    assert!(matches!(
        error,
        ValidationError::EntryPointInterfaceLocationConflict {
            storage_class: rspirv::spirv::StorageClass::Input,
            location: 1,
            component: 0,
            ..
        }
    ));
}

#[test]
fn storage_buffer_16bit_requires_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability VariablePointers",
        "OpCapability VariablePointersStorageBuffer",
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %var",
        "OpName %main \"main\"",
        "OpName %var \"var\"",
        "OpDecorate %buf Block",
        "OpMemberDecorate %buf 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%buf = OpTypeStruct %u16",
        "%ptr = OpTypePointer StorageBuffer %buf",
        "%var = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate_with_env(text, TargetEnv::Universal1_5)
        .expect_err("expected missing capability");
    assert_eq!(
        error,
        ValidationError::SmallTypeMissingCapability {
            bit_width: 16,
            storage_class: StorageClass::StorageBuffer,
            required_capability: Capability::StorageBuffer16BitAccess
        }
    );
}

#[test]
fn validatable_trait_covers_text_and_binary() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
    ]
    .join("\n");
    let valid_text = text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("valid text");
    assert_eq!(valid_text.header().schema(), Schema::ZERO);
    let binary = assemble_text(&text).expect("assemble");
    let valid_binary = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("valid binary");
    assert_eq!(
        valid_binary.header().bound().declared(),
        DeclaredBound(binary[3])
    );
}

#[test]
fn bitwise_valid_shift_left() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let u32_type = b.type_int(32, 0);
    let fn_type = b.type_function(void, vec![]);

    let base = b.constant_bit32(u32_type, 255); // 0xFF
    let shift = b.constant_bit32(u32_type, 4);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.shift_left_logical(u32_type, None, base, shift).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn bitwise_valid_and() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let u32_type = b.type_int(32, 0);
    let fn_type = b.type_function(void, vec![]);

    let a = b.constant_bit32(u32_type, 0xFF00);
    let bb = b.constant_bit32(u32_type, 0x0F0F);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.bitwise_and(u32_type, None, a, bb).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn bitwise_valid_or() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let i32_type = b.type_int(32, 1);
    let fn_type = b.type_function(void, vec![]);

    let a = b.constant_bit32(i32_type, 0x00FF);
    let bb = b.constant_bit32(i32_type, 0xFF00);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.bitwise_or(i32_type, None, a, bb).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn bitwise_valid_not() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let u32_type = b.type_int(32, 0);
    let fn_type = b.type_function(void, vec![]);

    let a = b.constant_bit32(u32_type, 0xFF);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.not(u32_type, None, a).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn bitwise_valid_vector_xor() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let u32_type = b.type_int(32, 0);
    let uvec4 = b.type_vector(u32_type, 4);
    let fn_type = b.type_function(void, vec![]);

    let c = b.constant_bit32(u32_type, 0xAAAA);
    let v = b.constant_composite(uvec4, vec![c, c, c, c]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.bitwise_xor(uvec4, None, v, v).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn logical_valid_and() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let bool_type = b.type_bool();
    let fn_type = b.type_function(void, vec![]);

    let true_val = b.constant_true(bool_type);
    let false_val = b.constant_false(bool_type);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.logical_and(bool_type, None, true_val, false_val).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn logical_valid_or() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let bool_type = b.type_bool();
    let fn_type = b.type_function(void, vec![]);

    let true_val = b.constant_true(bool_type);
    let false_val = b.constant_false(bool_type);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.logical_or(bool_type, None, true_val, false_val).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn logical_valid_not() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let bool_type = b.type_bool();
    let fn_type = b.type_function(void, vec![]);

    let true_val = b.constant_true(bool_type);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.logical_not(bool_type, None, true_val).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn logical_valid_float_comparison() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let bool_type = b.type_bool();
    let fn_type = b.type_function(void, vec![]);

    let a = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let bb = b.constant_bit32(f32_type, 0x40000000); // 2.0f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.f_ord_less_than(bool_type, None, a, bb).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn logical_valid_int_comparison() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let i32_type = b.type_int(32, 1);
    let bool_type = b.type_bool();
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(i32_type, 5);
    let c2 = b.constant_bit32(i32_type, 10);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.s_greater_than(bool_type, None, c1, c2).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn logical_valid_scalar_float_comparison() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let bool_type = b.type_bool();
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let c2 = b.constant_bit32(f32_type, 0x40000000); // 2.0f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.f_ord_equal(bool_type, None, c1, c2).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn cooperative_matrix_length_khr_valid() {
    // Using rspirv builder since our text assembler doesn't support OpCooperativeMatrixLengthKHR.
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.capability(Capability::CooperativeMatrixKHR);
    b.capability(Capability::VulkanMemoryModel);
    b.extension("SPV_KHR_cooperative_matrix");
    b.extension("SPV_KHR_vulkan_memory_model");
    b.memory_model(AddressingModel::Logical, MemoryModel::Vulkan);

    let void_type = b.type_void();
    let u32_type = b.type_int(32, 0);
    let f32_type = b.type_float(32, None);
    let scope = b.constant_bit32(u32_type, 3); // Subgroup
    let rows = b.constant_bit32(u32_type, 8);
    let cols = b.constant_bit32(u32_type, 8);
    let usage = b.constant_bit32(u32_type, 0); // MatrixA
    let coop_mat_type = b.type_cooperative_matrix_khr(f32_type, scope, rows, cols, usage);
    let fn_type = b.type_function(void_type, vec![]);

    let main_id = b
        .begin_function(void_type, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_id, "main", vec![]);
    b.execution_mode(main_id, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.cooperative_matrix_length_khr(u32_type, None, coop_mat_type)
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn cooperative_matrix_length_khr_wrong_result_type() {
    // Using rspirv builder since our text assembler doesn't support OpCooperativeMatrixLengthKHR.
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.capability(Capability::CooperativeMatrixKHR);
    b.capability(Capability::VulkanMemoryModel);
    b.extension("SPV_KHR_cooperative_matrix");
    b.extension("SPV_KHR_vulkan_memory_model");
    b.memory_model(AddressingModel::Logical, MemoryModel::Vulkan);

    let void_type = b.type_void();
    let u32_type = b.type_int(32, 0);
    let i32_type = b.type_int(32, 1); // signed int
    let f32_type = b.type_float(32, None);
    let scope = b.constant_bit32(u32_type, 3); // Subgroup
    let rows = b.constant_bit32(u32_type, 8);
    let cols = b.constant_bit32(u32_type, 8);
    let usage = b.constant_bit32(u32_type, 0); // MatrixA
    let coop_mat_type = b.type_cooperative_matrix_khr(f32_type, scope, rows, cols, usage);
    let fn_type = b.type_function(void_type, vec![]);

    let main_id = b
        .begin_function(void_type, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_id, "main", vec![]);
    b.execution_mode(main_id, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    // Using signed int as result type - should fail (needs unsigned)
    b.cooperative_matrix_length_khr(i32_type, None, coop_mat_type)
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert!(
        matches!(
            err,
            ValidationError::CooperativeMatrixLengthResultTypeMismatch { .. }
        ),
        "Expected CooperativeMatrixLengthResultTypeMismatch error, got: {err:?}"
    );
}

#[test]
fn validation_error_with_spans_contains_source_location() {
    use crate::assembly::assemble_text_with_spans;
    use crate::validation::{span::LabelKind, validate_module_with_spans};

    // Create an invalid SPIR-V module: struct with too many members for a limit
    let text = r#"OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%struct = OpTypeStruct %int %int %int %int %int"#;

    // Assemble with span tracking
    let result = assemble_text_with_spans(text).expect("assembly should succeed");

    // Create options with a limit of 3 members per struct
    let mut options = ValidationOptions::default();
    options
        .limits
        .insert(crate::validation::LIMIT_MAX_STRUCT_MEMBERS, 3);

    // Validate with span map
    let err = validate_module_with_spans(
        &result.words,
        TargetEnv::Universal1_6,
        options,
        &result.span_map,
    )
    .expect_err("validation should fail due to struct member limit");

    // The error should have a primary span pointing to the struct definition
    let primary = err
        .spans
        .iter()
        .find(|s| s.label.kind == LabelKind::Primary);
    assert!(primary.is_some(), "error should have a primary span");

    // The span should be on line 4 (0-indexed), where %struct is defined
    if let Some(span) = primary {
        let pos = span.span.text_position();
        assert!(pos.is_some(), "span should be a text position");
        if let Some(pos) = pos {
            assert_eq!(pos.line(), 4, "struct should be on line 4");
        }
    }
}

#[test]
fn validation_without_spans_still_works() {
    // Same invalid module but without span tracking
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int %int %int %int %int",
    ]
    .join("\n");

    let words = assemble_text(&text).expect("assembly should succeed");

    // Create options with a limit of 3 members per struct
    let mut options = ValidationOptions::default();
    options
        .limits
        .insert(crate::validation::LIMIT_MAX_STRUCT_MEMBERS, 3);

    // Validate without spans - should still work and fail appropriately
    let result =
        crate::validation::validate_module_with_options(&words, TargetEnv::Universal1_6, options);
    assert!(
        result.is_err(),
        "validation should fail due to struct member limit"
    );
}

#[test]
fn u_convert_16bit_with_int16_passes() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int16",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "OpExecutionMode %main OriginUpperLeft",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u16 = OpTypeInt 16 0",
        "%u32 = OpTypeInt 32 0",
        "%c32 = OpConstant %u32 255",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%result = OpUConvert %u16 %c32",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("UConvert to 16-bit with Int16 capability should pass");
}

#[test]
fn convert_s_to_f_16bit_with_int16_passes() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int16",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "OpExecutionMode %main OriginUpperLeft",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%i16 = OpTypeInt 16 1",
        "%f32 = OpTypeFloat 32",
        "%c16 = OpConstant %i16 1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%result = OpConvertSToF %f32 %c16",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("ConvertSToF with 16-bit input with Int16 capability should pass");
}

#[test]
fn convert_s_to_f_8bit_input_with_int8_passes() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int8",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "OpExecutionMode %main OriginUpperLeft",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%i8 = OpTypeInt 8 1",
        "%f32 = OpTypeFloat 32",
        "%c8 = OpConstant %i8 1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%result = OpConvertSToF %f32 %c8",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("ConvertSToF with 8-bit input with Int8 capability should pass");
}

#[test]
fn lifetime_start_non_function_storage_class_rejected() {
    // OpLifetimeStart with non-Function storage class should fail
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
    // %3 = OpTypePointer CrossWorkgroup %2  (NOT Function)
    module.types_global_values.push(Instruction::new(
        Op::TypePointer,
        Some(3),
        None,
        vec![
            Operand::StorageClass(rspirv::spirv::StorageClass::CrossWorkgroup),
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
    // %8 = OpVariable %3 CrossWorkgroup (global variable)
    module.types_global_values.push(Instruction::new(
        Op::Variable,
        Some(3),
        Some(8),
        vec![Operand::StorageClass(
            rspirv::spirv::StorageClass::CrossWorkgroup,
        )],
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
    // OpLifetimeStart %8 0  (using CrossWorkgroup pointer - should fail)
    block.instructions.push(Instruction::new(
        Op::LifetimeStart,
        None,
        None,
        vec![Operand::IdRef(8), Operand::LiteralBit32(0)],
    ));
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
            ValidationError::LifetimePointerNotFunctionStorageClass { .. }
        ),
        "Expected LifetimePointerNotFunctionStorageClass, got: {error:?}"
    );
}

#[test]
fn group_any_valid_kernel_module() {
    // OpGroupAny with bool result and bool predicate should pass
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op, Scope};

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
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Groups)],
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
            Operand::IdRef(7),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    // Types
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    module
        .types_global_values
        .push(Instruction::new(Op::TypeBool, Some(2), None, vec![]));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(3),
        None,
        vec![Operand::IdRef(1)],
    ));
    // Scope constant: %4 = OpConstant %u32 Workgroup(2)
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(4),
        None,
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(4),
        Some(5),
        vec![Operand::LiteralBit32(Scope::Workgroup as u32)],
    ));
    // Bool constant for predicate
    module
        .types_global_values
        .push(Instruction::new(Op::ConstantTrue, Some(2), Some(6), vec![]));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(7),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(3),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(8), None, vec![]));
    // %9 = OpGroupAny %bool %scope %predicate
    block.instructions.push(Instruction::new(
        Op::GroupAny,
        Some(2),
        Some(9),
        vec![Operand::IdRef(5), Operand::IdRef(6)],
    ));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Universal1_6).expect("Valid OpGroupAny should pass");
}

#[test]
fn group_any_non_bool_result_rejected() {
    // OpGroupAny with non-bool result should fail
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op, Scope};

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
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Groups)],
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
            Operand::IdRef(8),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    module
        .types_global_values
        .push(Instruction::new(Op::TypeBool, Some(2), None, vec![]));
    // %3 = OpTypeInt 32 0  (NOT bool - will be used as result type)
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(3),
        None,
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(4),
        None,
        vec![Operand::IdRef(1)],
    ));
    // Scope constant
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(5),
        vec![Operand::LiteralBit32(Scope::Workgroup as u32)],
    ));
    // Bool constant for predicate
    module
        .types_global_values
        .push(Instruction::new(Op::ConstantTrue, Some(2), Some(6), vec![]));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(8),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(4),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(9), None, vec![]));
    // %10 = OpGroupAny %int %scope %predicate  (int result, should fail)
    block.instructions.push(Instruction::new(
        Op::GroupAny,
        Some(3),
        Some(10),
        vec![Operand::IdRef(5), Operand::IdRef(6)],
    ));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert!(
        matches!(error, ValidationError::GroupResultMustBeBoolScalar { .. }),
        "Expected GroupResultMustBeBoolScalar, got: {error:?}"
    );
}

#[test]
fn group_fadd_valid_kernel_module() {
    // OpGroupFAdd with float result and matching X should pass
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{
        AddressingModel, Capability, ExecutionModel, GroupOperation, MemoryModel, Op, Scope,
    };

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
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Groups)],
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
            Operand::IdRef(8),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    // %2 = OpTypeFloat 32
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        Some(2),
        None,
        vec![Operand::LiteralBit32(32)],
    ));
    // %3 = OpTypeInt 32 0
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(3),
        None,
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(4),
        None,
        vec![Operand::IdRef(1)],
    ));
    // Scope constant
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(5),
        vec![Operand::LiteralBit32(Scope::Workgroup as u32)],
    ));
    // Float constant for X
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(2),
        Some(6),
        vec![Operand::LiteralBit32(0x3f800000)], // 1.0f
    ));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(8),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(4),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(9), None, vec![]));
    // %10 = OpGroupFAdd %float %scope Reduce %x
    block.instructions.push(Instruction::new(
        Op::GroupFAdd,
        Some(2),
        Some(10),
        vec![
            Operand::IdRef(5),
            Operand::GroupOperation(GroupOperation::Reduce),
            Operand::IdRef(6),
        ],
    ));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Universal1_6).expect("Valid OpGroupFAdd should pass");
}

#[test]
fn group_fadd_non_float_result_rejected() {
    // OpGroupFAdd with int result should fail
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{
        AddressingModel, Capability, ExecutionModel, GroupOperation, MemoryModel, Op, Scope,
    };

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
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Groups)],
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
            Operand::IdRef(8),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    // %2 = OpTypeFloat 32
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        Some(2),
        None,
        vec![Operand::LiteralBit32(32)],
    ));
    // %3 = OpTypeInt 32 0 (will be used as WRONG result type)
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(3),
        None,
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(4),
        None,
        vec![Operand::IdRef(1)],
    ));
    // Scope constant
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(5),
        vec![Operand::LiteralBit32(Scope::Workgroup as u32)],
    ));
    // Float constant for X
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(2),
        Some(6),
        vec![Operand::LiteralBit32(0x3f800000)],
    ));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(8),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(4),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(9), None, vec![]));
    // %10 = OpGroupFAdd %int %scope Reduce %x  (int result type - should fail)
    block.instructions.push(Instruction::new(
        Op::GroupFAdd,
        Some(3),
        Some(10),
        vec![
            Operand::IdRef(5),
            Operand::GroupOperation(GroupOperation::Reduce),
            Operand::IdRef(6),
        ],
    ));
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
            ValidationError::GroupResultMustBeFloatScalarOrVector { .. }
        ),
        "Expected GroupResultMustBeFloatScalarOrVector, got: {error:?}"
    );
}

// ============================================================================
// Commit 4: GLSL Refract eta type check
// ============================================================================

#[test]
fn glsl_refract_vec3_f32_with_f32_eta_passes() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%vec3 = OpTypeVector %f32 3
%ptr_out = OpTypePointer Output %vec3
%out = OpVariable %ptr_out Output
%glsl = OpExtInstImport "GLSL.std.450"
%f32_1 = OpConstant %f32 1.0
%vec3_val = OpConstantComposite %vec3 %f32_1 %f32_1 %f32_1
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpExtInst %vec3 %glsl 72 %vec3_val %vec3_val %f32_1
OpStore %out %result
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("Refract vec3<f32> with f32 eta should pass");
}

#[test]
fn glsl_refract_vec3_f32_with_f64_eta_fails() {
    // Use binary construction to precisely test the eta type mismatch
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Module, Operand};

    let mut module = Module::new();
    module.header = Some(rspirv::dr::ModuleHeader {
        magic_number: rspirv::spirv::MAGIC_NUMBER,
        version: (1 << 16) | (5 << 8),
        generator: 0,
        bound: 20,
        reserved_word: 0,
    });

    // OpCapability Shader
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(rspirv::spirv::Capability::Shader)],
    ));
    // OpCapability Float64
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(rspirv::spirv::Capability::Float64)],
    ));

    // OpExtInstImport %1 "GLSL.std.450"
    module.ext_inst_imports.push(Instruction::new(
        Op::ExtInstImport,
        None,
        Some(1),
        vec![Operand::LiteralString("GLSL.std.450".to_string())],
    ));

    // OpMemoryModel Logical GLSL450
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
            Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
        ],
    ));

    // OpEntryPoint Fragment %10 "main" %15
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Fragment),
            Operand::IdRef(10),
            Operand::LiteralString("main".to_string()),
            Operand::IdRef(15),
        ],
    ));

    // OpExecutionMode %10 OriginUpperLeft
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(10),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::OriginUpperLeft),
        ],
    ));

    // Types and constants
    // %2 = OpTypeVoid
    module.types_global_values.push(Instruction::new(
        Op::TypeVoid,
        None,
        Some(2),
        vec![],
    ));
    // %3 = OpTypeFunction %void
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        None,
        Some(3),
        vec![Operand::IdRef(2)],
    ));
    // %4 = OpTypeFloat 32
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        None,
        Some(4),
        vec![Operand::LiteralBit32(32)],
    ));
    // %5 = OpTypeFloat 64
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        None,
        Some(5),
        vec![Operand::LiteralBit32(64)],
    ));
    // %6 = OpTypeVector %4 3
    module.types_global_values.push(Instruction::new(
        Op::TypeVector,
        None,
        Some(6),
        vec![Operand::IdRef(4), Operand::LiteralBit32(3)],
    ));
    // %7 = OpTypePointer Output %6
    module.types_global_values.push(Instruction::new(
        Op::TypePointer,
        None,
        Some(7),
        vec![
            Operand::StorageClass(rspirv::spirv::StorageClass::Output),
            Operand::IdRef(6),
        ],
    ));
    // %8 = OpConstant %4 1.0
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(4),
        Some(8),
        vec![Operand::LiteralBit32(0x3F80_0000)], // 1.0f32
    ));
    // %9 = OpConstant %5 1.0 (64-bit needs two words)
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(5),
        Some(9),
        vec![
            Operand::LiteralBit32(0x0000_0000),
            Operand::LiteralBit32(0x3FF0_0000),
        ], // 1.0f64
    ));
    // %11 = OpConstantComposite %6 %8 %8 %8
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(6),
        Some(11),
        vec![Operand::IdRef(8), Operand::IdRef(8), Operand::IdRef(8)],
    ));
    // %15 = OpVariable %7 Output
    module.types_global_values.push(Instruction::new(
        Op::Variable,
        Some(7),
        Some(15),
        vec![Operand::StorageClass(
            rspirv::spirv::StorageClass::Output,
        )],
    ));

    // Function
    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(2),
        Some(10),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(3),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, None, Some(12), vec![]));
    // %13 = OpExtInst %6 %1 72 %11 %11 %9  (Refract with f64 eta)
    block.instructions.push(Instruction::new(
        Op::ExtInst,
        Some(6),
        Some(13),
        vec![
            Operand::IdRef(1),                     // GLSL import
            Operand::LiteralExtInstInteger(72),     // Refract opcode
            Operand::IdRef(11),                     // I
            Operand::IdRef(11),                     // N
            Operand::IdRef(9),                      // eta (f64!)
        ],
    ));
    block.instructions.push(Instruction::new(
        Op::Store,
        None,
        None,
        vec![Operand::IdRef(15), Operand::IdRef(13)],
    ));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect_err("Refract vec3<f32> with f64 eta should fail");
    assert!(
        matches!(err, ValidationError::ExtInstEtaTypeMismatch { .. }),
        "expected ExtInstEtaTypeMismatch, got {err:?}"
    );
}

// ============================================================================
// Commit 2: OutputVertices with TessellationEvaluation
// ============================================================================

#[test]
fn output_vertices_accepted_with_tess_evaluation() {
    let text = [
        "OpCapability Shader",
        "OpCapability Tessellation",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint TessellationEvaluation %main \"main\"",
        "OpExecutionMode %main Triangles",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("OutputVertices should be accepted with TessellationEvaluation");
}

#[test]
fn output_vertices_rejected_with_fragment() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "OpExecutionMode %main OriginUpperLeft",
        "OpExecutionMode %main OutputVertices 3",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect_err("OutputVertices should not be accepted with Fragment");
    assert!(matches!(
        error,
        ValidationError::ExecutionModeRequiresExecutionModel {
            mode: rspirv::spirv::ExecutionMode::OutputVertices,
            execution_model: rspirv::spirv::ExecutionModel::Fragment,
            ..
        }
    ));
}

// ============================================================================
// Commit 3: Switch branch count off-by-one fix
// ============================================================================

#[test]
fn switch_at_exact_limit_passes() {
    use crate::validation::rules::limits::SwitchBranchLimitRule;
    use crate::validation::{TestContextData, ValidationRule, LIMIT_MAX_SWITCH_BRANCHES};
    use rspirv::dr::{Instruction, Operand};

    // Build a switch with 2 case branches
    let switch_inst = Instruction::new(
        rspirv::spirv::Op::Switch,
        None,
        None,
        vec![
            Operand::IdRef(1),
            Operand::IdRef(2), // default
            Operand::LiteralBit32(0),
            Operand::IdRef(3), // case 0
            Operand::LiteralBit32(1),
            Operand::IdRef(4), // case 1
        ],
    );
    let block = rspirv::dr::Block {
        label: None,
        instructions: vec![switch_inst],
    };
    let function = rspirv::dr::Function {
        def: None,
        parameters: Vec::new(),
        blocks: vec![block],
        end: None,
    };

    let mut test_data = TestContextData::default();
    test_data.module.functions.push(function);
    test_data
        .options
        .limits
        .insert(LIMIT_MAX_SWITCH_BRANCHES, 2); // limit = 2, cases = 2 -> should pass

    let ctx = test_data.as_context();
    let rule = SwitchBranchLimitRule;
    rule.validate(&ctx)
        .expect("switch with 2 cases at limit of 2 should pass");
}

#[test]
fn switch_over_limit_fails() {
    use crate::validation::rules::limits::SwitchBranchLimitRule;
    use crate::validation::{TestContextData, ValidationRule, LIMIT_MAX_SWITCH_BRANCHES};
    use rspirv::dr::{Instruction, Operand};

    // Build a switch with 3 case branches
    let switch_inst = Instruction::new(
        rspirv::spirv::Op::Switch,
        None,
        None,
        vec![
            Operand::IdRef(1),
            Operand::IdRef(2), // default
            Operand::LiteralBit32(0),
            Operand::IdRef(3), // case 0
            Operand::LiteralBit32(1),
            Operand::IdRef(4), // case 1
            Operand::LiteralBit32(2),
            Operand::IdRef(5), // case 2
        ],
    );
    let block = rspirv::dr::Block {
        label: None,
        instructions: vec![switch_inst],
    };
    let function = rspirv::dr::Function {
        def: None,
        parameters: Vec::new(),
        blocks: vec![block],
        end: None,
    };

    let mut test_data = TestContextData::default();
    test_data.module.functions.push(function);
    test_data
        .options
        .limits
        .insert(LIMIT_MAX_SWITCH_BRANCHES, 2); // limit = 2, cases = 3 -> should fail

    let ctx = test_data.as_context();
    let rule = SwitchBranchLimitRule;
    let err = rule
        .validate(&ctx)
        .expect_err("switch with 3 cases at limit of 2 should fail");
    assert_eq!(
        err.error,
        ValidationError::LimitExceeded {
            limit_kind: LIMIT_MAX_SWITCH_BRANCHES,
            limit: 2,
            found: 3
        }
    );
}

// ============================================================================
// Commit 6: Transpose dimension validation
// ============================================================================

/// Helper to build a minimal binary module with OpTranspose.
/// `input_mat_type`: (col_count, row_count) of input matrix
/// `result_mat_type`: (col_count, row_count) of result matrix
fn build_transpose_module(
    input_cols: u32,
    input_rows: u32,
    result_cols: u32,
    result_rows: u32,
) -> Vec<u32> {
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Module, Operand};

    let mut module = Module::new();
    module.header = Some(rspirv::dr::ModuleHeader {
        magic_number: rspirv::spirv::MAGIC_NUMBER,
        version: (1 << 16) | (5 << 8),
        generator: 0,
        bound: 30,
        reserved_word: 0,
    });
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
        vec![Operand::Capability(Capability::Matrix)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
            Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Fragment),
            Operand::IdRef(20),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(20),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::OriginUpperLeft),
        ],
    ));

    // %1 = OpTypeVoid
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, None, Some(1), vec![]));
    // %2 = OpTypeFunction %void
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        None,
        Some(2),
        vec![Operand::IdRef(1)],
    ));
    // %3 = OpTypeFloat 32
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        None,
        Some(3),
        vec![Operand::LiteralBit32(32)],
    ));
    // %4 = OpTypeVector %f32 <input_rows>
    module.types_global_values.push(Instruction::new(
        Op::TypeVector,
        None,
        Some(4),
        vec![Operand::IdRef(3), Operand::LiteralBit32(input_rows)],
    ));
    // %5 = OpTypeVector %f32 <result_rows>
    module.types_global_values.push(Instruction::new(
        Op::TypeVector,
        None,
        Some(5),
        vec![Operand::IdRef(3), Operand::LiteralBit32(result_rows)],
    ));
    // %6 = OpTypeMatrix %vec_input_rows <input_cols>
    module.types_global_values.push(Instruction::new(
        Op::TypeMatrix,
        None,
        Some(6),
        vec![Operand::IdRef(4), Operand::LiteralBit32(input_cols)],
    ));
    // %7 = OpTypeMatrix %vec_result_rows <result_cols>
    module.types_global_values.push(Instruction::new(
        Op::TypeMatrix,
        None,
        Some(7),
        vec![Operand::IdRef(5), Operand::LiteralBit32(result_cols)],
    ));
    // %8 = OpConstant %f32 1.0
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(8),
        vec![Operand::LiteralBit32(0x3F80_0000)],
    ));
    // Build input matrix constant
    let mut col_operands = Vec::new();
    for _ in 0..input_rows {
        col_operands.push(Operand::IdRef(8));
    }
    // %9 = OpConstantComposite %vec_input_rows ...
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(4),
        Some(9),
        col_operands,
    ));
    let mut mat_operands = Vec::new();
    for _ in 0..input_cols {
        mat_operands.push(Operand::IdRef(9));
    }
    // %10 = OpConstantComposite %mat_input ...
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(6),
        Some(10),
        mat_operands,
    ));

    // Function
    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(20),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(2),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, None, Some(21), vec![]));
    // %22 = OpTranspose %mat_result %mat_input
    block.instructions.push(Instruction::new(
        Op::Transpose,
        Some(7),
        Some(22),
        vec![Operand::IdRef(10)],
    ));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    module.assemble()
}

#[test]
fn transpose_mat2x3_to_mat3x2_passes() {
    // mat2x3 (2 cols of vec3) transposed = mat3x2 (3 cols of vec2) - valid
    let binary = build_transpose_module(2, 3, 3, 2);
    validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect("Transpose mat2x3 -> mat3x2 should pass");
}

#[test]
fn transpose_mat2x3_to_mat2x3_fails() {
    // mat2x3 transposed should be mat3x2, NOT mat2x3 - dimensions don't match
    let binary = build_transpose_module(2, 3, 2, 3);
    let err = validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect_err("Transpose mat2x3 -> mat2x3 should fail");
    assert!(
        matches!(err, ValidationError::TransposeDimensionMismatch { .. }),
        "expected TransposeDimensionMismatch, got {err:?}"
    );
}

// ============================================================================
// Commit 5: CopyLogical structural match
// ============================================================================

/// Helper to build a binary module with OpCopyLogical between two struct types.
fn build_copy_logical_module(
    struct_a_members: &[u32],
    struct_b_members: &[u32],
) -> Vec<u32> {
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Module, Operand};

    let mut module = Module::new();
    module.header = Some(rspirv::dr::ModuleHeader {
        magic_number: rspirv::spirv::MAGIC_NUMBER,
        version: (1 << 16) | (5 << 8),
        generator: 0,
        bound: 30,
        reserved_word: 0,
    });
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Shader)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
            Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Fragment),
            Operand::IdRef(20),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(20),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::OriginUpperLeft),
        ],
    ));

    // %1 = OpTypeVoid
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, None, Some(1), vec![]));
    // %2 = OpTypeFunction %void
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        None,
        Some(2),
        vec![Operand::IdRef(1)],
    ));
    // %3 = OpTypeFloat 32
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        None,
        Some(3),
        vec![Operand::LiteralBit32(32)],
    ));
    // %4 = OpTypeInt 32 1
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        None,
        Some(4),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(1)],
    ));

    // %5 = OpTypeStruct <struct_a_members>
    let a_ops: Vec<_> = struct_a_members.iter().map(|id| Operand::IdRef(*id)).collect();
    module.types_global_values.push(Instruction::new(
        Op::TypeStruct,
        None,
        Some(5),
        a_ops,
    ));
    // %6 = OpTypeStruct <struct_b_members>
    let b_ops: Vec<_> = struct_b_members.iter().map(|id| Operand::IdRef(*id)).collect();
    module.types_global_values.push(Instruction::new(
        Op::TypeStruct,
        None,
        Some(6),
        b_ops,
    ));

    // Build constant for struct_a
    let mut const_ops = Vec::new();
    for (i, &member_type_id) in struct_a_members.iter().enumerate() {
        let const_id = 10 + i as u32;
        // Create a constant of the appropriate type
        if member_type_id == 3 {
            // float
            module.types_global_values.push(Instruction::new(
                Op::Constant,
                Some(3),
                Some(const_id),
                vec![Operand::LiteralBit32(0x3F80_0000)],
            ));
        } else {
            // int
            module.types_global_values.push(Instruction::new(
                Op::Constant,
                Some(4),
                Some(const_id),
                vec![Operand::LiteralBit32(1)],
            ));
        }
        const_ops.push(Operand::IdRef(const_id));
    }
    // %15 = OpConstantComposite %struct_a ...
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(5),
        Some(15),
        const_ops,
    ));

    // Function
    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(20),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(2),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, None, Some(21), vec![]));
    // %22 = OpCopyLogical %struct_b %val_a
    block.instructions.push(Instruction::new(
        Op::CopyLogical,
        Some(6),
        Some(22),
        vec![Operand::IdRef(15)],
    ));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    module.assemble()
}

#[test]
fn copy_logical_matching_structs_passes() {
    // Both structs have {f32, f32} - logically matching
    let binary = build_copy_logical_module(&[3, 3], &[3, 3]);
    validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect("CopyLogical between structurally matching structs should pass");
}

#[test]
fn copy_logical_mismatched_member_count_fails() {
    // struct_a has {f32, f32}, struct_b has {f32, f32, i32}
    let binary = build_copy_logical_module(&[3, 3], &[3, 3, 4]);
    let err = validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect_err("CopyLogical between structs with different member counts should fail");
    assert!(
        matches!(
            err,
            ValidationError::CopyLogicalTypesNotLogicallyMatching { .. }
        ),
        "expected CopyLogicalTypesNotLogicallyMatching, got {err:?}"
    );
}

#[test]
fn copy_logical_mismatched_member_types_fails() {
    // struct_a has {f32, f32}, struct_b has {f32, i32}
    let binary = build_copy_logical_module(&[3, 3], &[3, 4]);
    let err = validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect_err("CopyLogical between structs with different member types should fail");
    assert!(
        matches!(
            err,
            ValidationError::CopyLogicalTypesNotLogicallyMatching { .. }
        ),
        "expected CopyLogicalTypesNotLogicallyMatching, got {err:?}"
    );
}

// ============================================================================
// Commit 7: ConstantComposite constituent type checks
// ============================================================================

#[test]
fn constant_composite_struct_correct_types_passes() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%i32 = OpTypeInt 32 1
%struct = OpTypeStruct %f32 %i32
%f32_1 = OpConstant %f32 1.0
%i32_1 = OpConstant %i32 1
%val = OpConstantComposite %struct %f32_1 %i32_1
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Vulkan1_2)
        .expect("ConstantComposite struct with correct member types should pass");
}

#[test]
fn constant_composite_struct_wrong_member_type_fails() {
    // struct has {f32, i32} but we provide {f32, f32}
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Module, Operand};

    let mut module = Module::new();
    module.header = Some(rspirv::dr::ModuleHeader {
        magic_number: rspirv::spirv::MAGIC_NUMBER,
        version: (1 << 16) | (5 << 8),
        generator: 0,
        bound: 20,
        reserved_word: 0,
    });

    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Shader)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
            Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Fragment),
            Operand::IdRef(10),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(10),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::OriginUpperLeft),
        ],
    ));

    // %1 = OpTypeVoid
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, None, Some(1), vec![]));
    // %2 = OpTypeFunction %void
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        None,
        Some(2),
        vec![Operand::IdRef(1)],
    ));
    // %3 = OpTypeFloat 32
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        None,
        Some(3),
        vec![Operand::LiteralBit32(32)],
    ));
    // %4 = OpTypeInt 32 1
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        None,
        Some(4),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(1)],
    ));
    // %5 = OpTypeStruct %f32 %i32
    module.types_global_values.push(Instruction::new(
        Op::TypeStruct,
        None,
        Some(5),
        vec![Operand::IdRef(3), Operand::IdRef(4)],
    ));
    // %6 = OpConstant %f32 1.0
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(6),
        vec![Operand::LiteralBit32(0x3F80_0000)],
    ));
    // %7 = OpConstant %f32 2.0  (wrong type for member 1 which expects i32)
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(7),
        vec![Operand::LiteralBit32(0x4000_0000)],
    ));
    // %8 = OpConstantComposite %struct %f32_val %f32_val (member 1 should be i32!)
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(5),
        Some(8),
        vec![Operand::IdRef(6), Operand::IdRef(7)],
    ));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(10),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(2),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, None, Some(11), vec![]));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect_err("ConstantComposite with wrong member type should fail");
    assert!(
        matches!(
            err,
            ValidationError::ConstantCompositeConstituentTypeMismatch { index: 1 }
        ),
        "expected ConstantCompositeConstituentTypeMismatch at index 1, got {err:?}"
    );
}

#[test]
fn constant_composite_matrix_wrong_column_type_fails() {
    // Matrix expects vec3 columns but we provide vec2
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Module, Operand};

    let mut module = Module::new();
    module.header = Some(rspirv::dr::ModuleHeader {
        magic_number: rspirv::spirv::MAGIC_NUMBER,
        version: (1 << 16) | (5 << 8),
        generator: 0,
        bound: 20,
        reserved_word: 0,
    });

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
        vec![Operand::Capability(Capability::Matrix)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
            Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Fragment),
            Operand::IdRef(15),
            Operand::LiteralString("main".to_string()),
        ],
    ));
    module.execution_modes.push(Instruction::new(
        Op::ExecutionMode,
        None,
        None,
        vec![
            Operand::IdRef(15),
            Operand::ExecutionMode(rspirv::spirv::ExecutionMode::OriginUpperLeft),
        ],
    ));

    // %1 = OpTypeVoid
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, None, Some(1), vec![]));
    // %2 = OpTypeFunction %void
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        None,
        Some(2),
        vec![Operand::IdRef(1)],
    ));
    // %3 = OpTypeFloat 32
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        None,
        Some(3),
        vec![Operand::LiteralBit32(32)],
    ));
    // %4 = OpTypeVector %f32 3
    module.types_global_values.push(Instruction::new(
        Op::TypeVector,
        None,
        Some(4),
        vec![Operand::IdRef(3), Operand::LiteralBit32(3)],
    ));
    // %5 = OpTypeVector %f32 2
    module.types_global_values.push(Instruction::new(
        Op::TypeVector,
        None,
        Some(5),
        vec![Operand::IdRef(3), Operand::LiteralBit32(2)],
    ));
    // %6 = OpTypeMatrix %vec3 2
    module.types_global_values.push(Instruction::new(
        Op::TypeMatrix,
        None,
        Some(6),
        vec![Operand::IdRef(4), Operand::LiteralBit32(2)],
    ));
    // %7 = OpConstant %f32 1.0
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(7),
        vec![Operand::LiteralBit32(0x3F80_0000)],
    ));
    // %8 = OpConstantComposite %vec2 %f32_1 %f32_1 (wrong column type)
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(5),
        Some(8),
        vec![Operand::IdRef(7), Operand::IdRef(7)],
    ));
    // %9 = OpConstantComposite %vec2 %f32_1 %f32_1 (wrong column type)
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(5),
        Some(9),
        vec![Operand::IdRef(7), Operand::IdRef(7)],
    ));
    // %10 = OpConstantComposite %mat2x3 %vec2_a %vec2_b (columns are vec2, should be vec3)
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(6),
        Some(10),
        vec![Operand::IdRef(8), Operand::IdRef(9)],
    ));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(15),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(2),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, None, Some(16), vec![]));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect_err("ConstantComposite matrix with wrong column type should fail");
    assert!(
        matches!(
            err,
            ValidationError::ConstantCompositeConstituentTypeMismatch { .. }
        ),
        "expected ConstantCompositeConstituentTypeMismatch, got {err:?}"
    );
}
