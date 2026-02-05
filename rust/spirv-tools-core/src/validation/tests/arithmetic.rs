use super::*;

#[test]
fn integer_add_operands_must_match_result_type() {
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
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let fconst = b.constant_bit32(float, 0x3f80_0000);
    let iconst = b.constant_bit32(int, 1);
    b.i_add(int, None, fconst, iconst).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("integer add operands must match result type");
    assert_eq!(
        err,
        ValidationError::ArithmeticResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::IAdd,
            result_type: TypeId::try_from(int).unwrap(),
            expected: "int scalar or vector",
        }
    );
}

#[test]
fn float_arithmetic_valid_fadd() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    // Build module using rspirv's builder
    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let fn_type = b.type_function(void, vec![]);

    // Create float constants (1.0f and 2.0f as raw bits)
    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let c2 = b.constant_bit32(f32_type, 0x40000000); // 2.0f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.f_add(f32_type, None, c1, c2).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn float_arithmetic_valid_vector_fmul() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec4_type = b.type_vector(f32_type, 4);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let v1 = b.constant_composite(vec4_type, vec![c1, c1, c1, c1]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.f_mul(vec4_type, None, v1, v1).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn int_arithmetic_valid_iadd() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%i32 = OpTypeInt 32 1",
        "%c1 = OpConstant %i32 5",
        "%c2 = OpConstant %i32 10",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%sum = OpIAdd %i32 %c1 %c2",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn int_arithmetic_valid_imul() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%c1 = OpConstant %u32 3",
        "%c2 = OpConstant %u32 7",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%prod = OpIMul %u32 %c1 %c2",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn dot_product_valid() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec3_type = b.type_vector(f32_type, 3);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let v1 = b.constant_composite(vec3_type, vec![c1, c1, c1]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.dot(f32_type, None, v1, v1).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn int_arithmetic_allows_signed_unsigned_mismatch_isub() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%i32 = OpTypeInt 32 1",
        "%unsigned_55 = OpConstant %u32 55",
        "%signed_10 = OpConstant %i32 10",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        // Result is unsigned, operand 0 is signed, operand 1 is unsigned
        "%result = OpISub %u32 %signed_10 %unsigned_55",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect("should be valid: signedness mismatch allowed for ISub");
}

#[test]
fn int_arithmetic_allows_signed_unsigned_mismatch_iadd() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%i32 = OpTypeInt 32 1",
        "%unsigned_5 = OpConstant %u32 5",
        "%signed_10 = OpConstant %i32 10",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        // Result is signed, operands are mixed
        "%result = OpIAdd %i32 %unsigned_5 %signed_10",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect("should be valid: signedness mismatch allowed for IAdd");
}

#[test]
fn int_arithmetic_allows_signed_unsigned_mismatch_imul() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%i32 = OpTypeInt 32 1",
        "%unsigned_3 = OpConstant %u32 3",
        "%signed_7 = OpConstant %i32 7",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%result = OpIMul %u32 %unsigned_3 %signed_7",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect("should be valid: signedness mismatch allowed for IMul");
}

#[test]
fn int_arithmetic_rejects_different_bit_width() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int64",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%u64 = OpTypeInt 64 0",
        "%c32 = OpConstant %u32 5",
        "%c64 = OpConstant %u64 10",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%result = OpIAdd %u32 %c32 %c64",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    let result = validate_module(&binary, TargetEnv::Vulkan1_2);
    assert!(result.is_err(), "should reject different bit widths");
}

#[test]
fn bitwise_allows_signed_unsigned_mismatch() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let u32_type = b.type_int(32, 0); // unsigned
    let i32_type = b.type_int(32, 1); // signed
    let fn_type = b.type_function(void, vec![]);

    let unsigned_mask = b.constant_bit32(u32_type, 0xFF);
    let signed_val = b.constant_bit32(i32_type, 1234);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    // Result is unsigned, operands are mixed signed/unsigned
    b.bitwise_and(u32_type, None, signed_val, unsigned_mask)
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2)
        .expect("should be valid: signedness mismatch allowed for BitwiseAnd");
}

#[test]
fn conversion_valid_float_to_int() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let i32_type = b.type_int(32, 1);
    let fn_type = b.type_function(void, vec![]);

    let fval = b.constant_bit32(f32_type, 0x4048f5c3); // 3.14f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.convert_f_to_s(i32_type, None, fval).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn conversion_valid_float_to_uint() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let u32_type = b.type_int(32, 0);
    let fn_type = b.type_function(void, vec![]);

    let fval = b.constant_bit32(f32_type, 0x41280000); // 10.5f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.convert_f_to_u(u32_type, None, fval).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn conversion_valid_int_to_float() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let i32_type = b.type_int(32, 1);
    let fn_type = b.type_function(void, vec![]);

    let ival = b.constant_bit32(i32_type, 42);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.convert_s_to_f(f32_type, None, ival).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn conversion_valid_vector_convert() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.capability(Capability::Float64);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let f64_type = b.type_float(64, None);
    let vec2f32 = b.type_vector(f32_type, 2);
    let vec2f64 = b.type_vector(f64_type, 2);
    let fn_type = b.type_function(void, vec![]);

    let c = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let v = b.constant_composite(vec2f32, vec![c, c]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.f_convert(vec2f64, None, v).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn iadd_carry_valid() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let uint = b.type_int(32, 0);
    let struct_ty = b.type_struct([uint, uint]);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let a = b.undef(uint, None);
    let bb = b.undef(uint, None);
    b.i_add_carry(struct_ty, None, a, bb).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("Valid IAddCarry should pass validation");
}

#[test]
fn iadd_carry_result_must_be_struct() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let uint = b.type_int(32, 0);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let a = b.undef(uint, None);
    let bb = b.undef(uint, None);
    // Use wrong result type (uint instead of struct)
    b.i_add_carry(uint, None, a, bb).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("IAddCarry with non-struct result should fail");
    assert!(
        matches!(
            err,
            ValidationError::ExtendedArithmeticResultNotStruct { .. }
        ),
        "Expected ExtendedArithmeticResultNotStruct, got: {err:?}"
    );
}

#[test]
fn iadd_carry_struct_must_have_two_members() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let uint = b.type_int(32, 0);
    let struct_ty = b.type_struct([uint]); // Only 1 member, should be 2
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let a = b.undef(uint, None);
    let bb = b.undef(uint, None);
    b.i_add_carry(struct_ty, None, a, bb).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("IAddCarry with 1-member struct should fail");
    assert!(
        matches!(
            err,
            ValidationError::ExtendedArithmeticStructMemberCount { found: 1, .. }
        ),
        "Expected ExtendedArithmeticStructMemberCount with found=1, got: {err:?}"
    );
}

#[test]
fn iadd_carry_struct_members_must_be_identical() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let uint = b.type_int(32, 0);
    let uint64 = b.type_int(64, 0);
    let struct_ty = b.type_struct([uint, uint64]); // Different types
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let a = b.undef(uint, None);
    let bb = b.undef(uint, None);
    b.i_add_carry(struct_ty, None, a, bb).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("IAddCarry with different struct member types should fail");
    assert!(
        matches!(
            err,
            ValidationError::ExtendedArithmeticStructMembersNotIdentical { .. }
        ),
        "Expected ExtendedArithmeticStructMembersNotIdentical, got: {err:?}"
    );
}

#[test]
fn sdot_valid_with_int_vectors() {
    // OpSDot with int scalar result and matching int vectors should pass
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op};

    let mut module = rspirv::dr::Module::new();
    module.header = Some(rspirv::dr::ModuleHeader::new(20));
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
        vec![Operand::Capability(Capability::DotProduct)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::DotProductInput4x8Bit)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Int8)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(AddressingModel::Logical),
            Operand::MemoryModel(MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(ExecutionModel::Fragment),
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
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    // %2 = OpTypeInt 32 1 (signed)
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(2),
        None,
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(1)],
    ));
    // %3 = OpTypeInt 8 1 (signed 8-bit)
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(3),
        None,
        vec![Operand::LiteralBit32(8), Operand::LiteralBit32(1)],
    ));
    // %4 = OpTypeVector %3 4
    module.types_global_values.push(Instruction::new(
        Op::TypeVector,
        Some(4),
        None,
        vec![Operand::IdRef(3), Operand::LiteralBit32(4)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(5),
        None,
        vec![Operand::IdRef(1)],
    ));
    // Constants
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(6),
        vec![Operand::LiteralBit32(1)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(4),
        Some(7),
        vec![
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
        ],
    ));
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(4),
        Some(8),
        vec![
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
        ],
    ));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(10),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(5),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(11), None, vec![]));
    // %12 = OpSDot %i32 %vec1 %vec2
    block.instructions.push(Instruction::new(
        Op::SDot,
        Some(2),
        Some(12),
        vec![Operand::IdRef(7), Operand::IdRef(8)],
    ));
    block
        .instructions
        .push(Instruction::new(Op::Return, None, None, vec![]));
    func.blocks.push(block);
    func.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
    module.functions.push(func);

    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("Valid OpSDot with int vectors should pass");
}

#[test]
fn sdot_non_int_result_rejected() {
    // OpSDot with float result type should fail
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op};

    let mut module = rspirv::dr::Module::new();
    module.header = Some(rspirv::dr::ModuleHeader::new(20));
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
        vec![Operand::Capability(Capability::DotProduct)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::DotProductInput4x8Bit)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Int8)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(AddressingModel::Logical),
            Operand::MemoryModel(MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(ExecutionModel::Fragment),
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
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    // %2 = OpTypeFloat 32 (WRONG - should be int)
    module.types_global_values.push(Instruction::new(
        Op::TypeFloat,
        Some(2),
        None,
        vec![Operand::LiteralBit32(32)],
    ));
    // %3 = OpTypeInt 8 1
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(3),
        None,
        vec![Operand::LiteralBit32(8), Operand::LiteralBit32(1)],
    ));
    // %4 = OpTypeVector %3 4
    module.types_global_values.push(Instruction::new(
        Op::TypeVector,
        Some(4),
        None,
        vec![Operand::IdRef(3), Operand::LiteralBit32(4)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(5),
        None,
        vec![Operand::IdRef(1)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(6),
        vec![Operand::LiteralBit32(1)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(4),
        Some(7),
        vec![
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
        ],
    ));
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(4),
        Some(8),
        vec![
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
        ],
    ));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(10),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(5),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(11), None, vec![]));
    // %12 = OpSDot %float %vec1 %vec2  (float result - should fail)
    block.instructions.push(Instruction::new(
        Op::SDot,
        Some(2),
        Some(12),
        vec![Operand::IdRef(7), Operand::IdRef(8)],
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
        matches!(error, ValidationError::DotProductResultNotIntScalar { .. }),
        "Expected DotProductResultNotIntScalar, got: {error:?}"
    );
}

#[test]
fn udot_signed_result_rejected() {
    // OpUDot requires unsigned result type
    use rspirv::binary::Assemble;
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel, Op};

    let mut module = rspirv::dr::Module::new();
    module.header = Some(rspirv::dr::ModuleHeader::new(20));
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
        vec![Operand::Capability(Capability::DotProduct)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::DotProductInput4x8Bit)],
    ));
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(Capability::Int8)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            Operand::AddressingModel(AddressingModel::Logical),
            Operand::MemoryModel(MemoryModel::GLSL450),
        ],
    ));
    module.entry_points.push(Instruction::new(
        Op::EntryPoint,
        None,
        None,
        vec![
            Operand::ExecutionModel(ExecutionModel::Fragment),
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
    module
        .types_global_values
        .push(Instruction::new(Op::TypeVoid, Some(1), None, vec![]));
    // %2 = OpTypeInt 32 1 (SIGNED - OpUDot requires unsigned)
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(2),
        None,
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(1)],
    ));
    // %3 = OpTypeInt 8 0 (unsigned 8-bit)
    module.types_global_values.push(Instruction::new(
        Op::TypeInt,
        Some(3),
        None,
        vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
    ));
    // %4 = OpTypeVector %3 4
    module.types_global_values.push(Instruction::new(
        Op::TypeVector,
        Some(4),
        None,
        vec![Operand::IdRef(3), Operand::LiteralBit32(4)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::TypeFunction,
        Some(5),
        None,
        vec![Operand::IdRef(1)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(3),
        Some(6),
        vec![Operand::LiteralBit32(1)],
    ));
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(4),
        Some(7),
        vec![
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
        ],
    ));
    module.types_global_values.push(Instruction::new(
        Op::ConstantComposite,
        Some(4),
        Some(8),
        vec![
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
            Operand::IdRef(6),
        ],
    ));

    let mut func = rspirv::dr::Function::new();
    func.def = Some(Instruction::new(
        Op::Function,
        Some(1),
        Some(10),
        vec![
            Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
            Operand::IdRef(5),
        ],
    ));
    let mut block = rspirv::dr::Block::new();
    block.label = Some(Instruction::new(Op::Label, Some(11), None, vec![]));
    // %12 = OpUDot %i32_signed %vec1 %vec2  (signed result - should fail)
    block.instructions.push(Instruction::new(
        Op::UDot,
        Some(2),
        Some(12),
        vec![Operand::IdRef(7), Operand::IdRef(8)],
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
            ValidationError::DotProductResultNotUnsignedIntScalar { .. }
        ),
        "Expected DotProductResultNotUnsignedIntScalar, got: {error:?}"
    );
}

#[test]
fn capability_with_alternative_extensions_accepts_any_single_extension() {
    // Should succeed with only SPV_NV_viewport_array2 (not both extensions)
    let text_nv = [
        "OpCapability Shader",
        "OpCapability Geometry",
        "OpCapability MultiViewport",
        "OpCapability ShaderViewportIndexLayerEXT",
        "OpExtension \"SPV_NV_viewport_array2\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text_nv
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("Capability with one of its alternative extensions should be accepted");

    // Should succeed with only SPV_EXT_shader_viewport_index_layer
    let text_ext = [
        "OpCapability Shader",
        "OpCapability Geometry",
        "OpCapability MultiViewport",
        "OpCapability ShaderViewportIndexLayerEXT",
        "OpExtension \"SPV_EXT_shader_viewport_index_layer\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text_ext
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("Capability with the other alternative extension should be accepted");
}

#[test]
fn capability_with_alternative_extensions_rejects_when_none_declared() {
    // Should fail when neither alternative extension is declared
    let text = [
        "OpCapability Shader",
        "OpCapability Geometry",
        "OpCapability MultiViewport",
        "OpCapability ShaderViewportIndexLayerEXT",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
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
        .validate(TargetEnv::Vulkan1_1)
        .expect_err("Capability without any required extension should be rejected");
    assert!(
        matches!(
            error,
            ValidationError::DisallowedCapabilityMissingExtension { .. }
        ),
        "Expected DisallowedCapabilityMissingExtension, got: {error:?}"
    );
}

#[test]
fn instance_id_builtin_rejected_in_vulkan_vertex_shader() {
    // InstanceId is not allowed in Vulkan vertex shaders (use InstanceIndex instead)
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %iid",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 1",
        "%ptr_input_int = OpTypePointer Input %int",
        "%iid = OpVariable %ptr_input_int Input",
        "OpDecorate %iid BuiltIn InstanceId",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect_err("InstanceId should be rejected in Vulkan vertex shader");
    assert!(
        matches!(error, ValidationError::BuiltInDisallowedForEnv { .. }),
        "Expected BuiltInDisallowedForEnv, got: {error:?}"
    );
}

#[test]
fn instance_id_builtin_accepted_in_vulkan_ray_tracing_shader() {
    // InstanceId IS allowed in Vulkan ray tracing shaders (ClosestHit, AnyHit, Intersection)
    let text = [
        "OpCapability Shader",
        "OpCapability RayTracingKHR",
        "OpExtension \"SPV_KHR_ray_tracing\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint ClosestHitKHR %main \"main\" %iid",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 1",
        "%ptr_input_int = OpTypePointer Input %int",
        "%iid = OpVariable %ptr_input_int Input",
        "OpDecorate %iid BuiltIn InstanceId",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("InstanceId should be accepted in Vulkan ray tracing shader");
}

#[test]
fn clip_distance_input_rejected_in_vertex_shader() {
    // ClipDistance as Input is not allowed in Vertex shaders
    let text = [
        "OpCapability Shader",
        "OpCapability ClipDistance",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\" %clip",
        "%void = OpTypeVoid",
        "%float = OpTypeFloat 32",
        "%arr = OpTypeArray %float %uint_1",
        "%uint = OpTypeInt 32 0",
        "%uint_1 = OpConstant %uint 1",
        "%ptr_input_arr = OpTypePointer Input %arr",
        "%clip = OpVariable %ptr_input_arr Input",
        "OpDecorate %clip BuiltIn ClipDistance",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect_err("ClipDistance Input should be rejected in Vertex shader");
    assert!(
        matches!(
            error,
            ValidationError::BuiltInWrongStorageClassForExecutionModel { .. }
        ),
        "Expected BuiltInWrongStorageClassForExecutionModel, got: {error:?}"
    );
}

#[test]
fn clip_distance_input_accepted_in_fragment_with_separate_vertex_entry() {
    // ClipDistance as Input is allowed in Fragment shaders, even when
    // the module also has a Vertex entry point (that doesn't use it)
    let text = [
        "OpCapability Shader",
        "OpCapability ClipDistance",
        "OpMemoryModel Logical GLSL450",
        // Vertex entry point does NOT list %clip in its interface
        "OpEntryPoint Vertex %vmain \"vmain\"",
        // Fragment entry point DOES list %clip in its interface
        "OpEntryPoint Fragment %fmain \"fmain\" %clip",
        "OpExecutionMode %fmain OriginUpperLeft",
        "%void = OpTypeVoid",
        "%float = OpTypeFloat 32",
        "%uint = OpTypeInt 32 0",
        "%uint_1 = OpConstant %uint 1",
        "%arr = OpTypeArray %float %uint_1",
        "%ptr_input_arr = OpTypePointer Input %arr",
        "%clip = OpVariable %ptr_input_arr Input",
        "OpDecorate %clip BuiltIn ClipDistance",
        "%fn = OpTypeFunction %void",
        "%vmain = OpFunction %void None %fn",
        "%ventry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "%fmain = OpFunction %void None %fn",
        "%fentry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("ClipDistance Input should be accepted in Fragment shader");
}

#[test]
fn frag_size_ext_rejects_float_type() {
    // FragSizeEXT must be vec2<i32>, not vec2<f32>
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentDensityEXT",
        "OpExtension \"SPV_EXT_fragment_invocation_density\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %fs",
        "OpExecutionMode %main OriginUpperLeft",
        "%void = OpTypeVoid",
        "%float = OpTypeFloat 32",
        "%v2f = OpTypeVector %float 2",
        "%ptr_input_v2f = OpTypePointer Input %v2f",
        "%fs = OpVariable %ptr_input_v2f Input",
        "OpDecorate %fs BuiltIn FragSizeEXT",
        "OpDecorate %fs Flat",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let error = text
        .as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect_err("FragSizeEXT with vec2<f32> should be rejected");
    assert!(
        matches!(error, ValidationError::InvalidBuiltInType { .. }),
        "Expected InvalidBuiltInType, got: {error:?}"
    );
}

#[test]
fn frag_size_ext_accepts_integer_vec2_type() {
    // FragSizeEXT must be vec2<i32/u32>
    let text = [
        "OpCapability Shader",
        "OpCapability FragmentDensityEXT",
        "OpExtension \"SPV_EXT_fragment_invocation_density\"",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %fs",
        "OpExecutionMode %main OriginUpperLeft",
        "%void = OpTypeVoid",
        "%uint = OpTypeInt 32 0",
        "%v2u = OpTypeVector %uint 2",
        "%ptr_input_v2u = OpTypePointer Input %v2u",
        "%fs = OpVariable %ptr_input_v2u Input",
        "OpDecorate %fs BuiltIn FragSizeEXT",
        "OpDecorate %fs Flat",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("FragSizeEXT with vec2<u32> should be accepted");
}

#[test]
fn nested_struct_misalignment_rejected() {
    // An outer Block struct contains an inner struct. The inner struct has a
    // member at offset 2, which is not aligned to 4 (uint alignment under
    // std140). Recursive validation should catch this.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %outer Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %outer 0 Offset 0",
        "OpMemberDecorate %inner 0 Offset 0",
        "OpMemberDecorate %inner 1 Offset 2",
        "%int = OpTypeInt 32 0",
        "%inner = OpTypeStruct %int %int",
        "%outer = OpTypeStruct %inner",
        "%ptr = OpTypePointer Uniform %outer",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("nested struct with misaligned member should fail");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn nested_struct_correct_alignment_accepted() {
    // Same structure but with proper alignment for the inner struct members.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %outer Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %outer 0 Offset 0",
        "OpMemberDecorate %inner 0 Offset 0",
        "OpMemberDecorate %inner 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%inner = OpTypeStruct %int %int",
        "%outer = OpTypeStruct %inner",
        "%ptr = OpTypePointer Uniform %outer",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("nested struct with correct alignment should pass");
}

#[test]
fn nested_array_bad_stride_rejected() {
    // An array-of-arrays: outer array contains inner arrays. Under std140
    // (Uniform+Block), the inner array stride must be aligned to 16 (extended
    // alignment rounding for arrays). A stride of 8 is not aligned to 16.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %outer Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %outer 0 Offset 0",
        "OpDecorate %outer_arr ArrayStride 32",
        "OpDecorate %inner_arr ArrayStride 8",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%inner_arr = OpTypeArray %int %two",
        "%outer_arr = OpTypeArray %inner_arr %two",
        "%outer = OpTypeStruct %outer_arr",
        "%ptr = OpTypePointer Uniform %outer",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("nested array with bad stride should fail");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn nested_array_correct_stride_accepted() {
    // Same array-of-arrays but with the inner array stride properly aligned
    // to 16 (std140 extended alignment).
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %outer Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %outer 0 Offset 0",
        "OpDecorate %outer_arr ArrayStride 32",
        "OpDecorate %inner_arr ArrayStride 16",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%inner_arr = OpTypeArray %int %two",
        "%outer_arr = OpTypeArray %inner_arr %two",
        "%outer = OpTypeStruct %outer_arr",
        "%ptr = OpTypePointer Uniform %outer",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("nested array with correct stride should pass");
}

#[test]
fn row_major_matrix_bad_alignment_rejected() {
    // A row-major mat4x2 (4 columns of vec2<f32>) has alignment equal to a
    // virtual vec4<f32> (4*4=16), NOT the column vector alignment (vec2=8).
    // Placing it at offset 8 should fail because 8 % 16 != 0.
    // Uses Uniform+BufferBlock for std430 rules (no extended alignment rounding).
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%v2 = OpTypeVector %float 2",
        "%mat = OpTypeMatrix %v2 4",
        "%struct = OpTypeStruct %v2 %mat",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %struct BufferBlock",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 8",
        "OpMemberDecorate %struct 1 RowMajor",
        "OpMemberDecorate %struct 1 MatrixStride 16",
    ]
    .join("\n");
    let words = assemble_text(&text).expect("assemble");
    let err = validate_module(&words, TargetEnv::Vulkan1_0)
        .expect_err("row-major matrix at misaligned offset should fail");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn row_major_matrix_correct_alignment_accepted() {
    // Same row-major mat4x2 but placed at offset 16 (aligned to 16). Should pass.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%v2 = OpTypeVector %float 2",
        "%mat = OpTypeMatrix %v2 4",
        "%struct = OpTypeStruct %v2 %mat",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %struct BufferBlock",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 16",
        "OpMemberDecorate %struct 1 RowMajor",
        "OpMemberDecorate %struct 1 MatrixStride 16",
    ]
    .join("\n");
    let words = assemble_text(&text).expect("assemble");
    validate_module(&words, TargetEnv::Vulkan1_0)
        .expect("row-major matrix at aligned offset should pass");
}

#[test]
fn row_major_matrix_large_column_no_straddle_rejection() {
    // A row-major mat2x4 (2 columns of vec4<f64>) with col_size=32 bytes.
    // The Vulkan spec only defines straddle checks for vectors, NOT matrices.
    // Under relaxed layout, this should pass (the old overly strict straddle
    // check for row-major matrices has been removed).
    let text = [
        "OpCapability Shader",
        "OpCapability Float64",
        "OpMemoryModel Logical GLSL450",
        "%f64 = OpTypeFloat 64",
        "%v4 = OpTypeVector %f64 4",
        "%mat = OpTypeMatrix %v4 2",
        "%struct = OpTypeStruct %v4 %mat",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %struct BufferBlock",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 32",
        "OpMemberDecorate %struct 1 RowMajor",
        "OpMemberDecorate %struct 1 MatrixStride 32",
    ]
    .join("\n");
    let words = assemble_text(&text).expect("assemble");
    let opts = ValidationOptions {
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    validate_module_with_options(&words, TargetEnv::Vulkan1_0, opts)
        .expect("row-major matrix should not be rejected for straddle");
}
