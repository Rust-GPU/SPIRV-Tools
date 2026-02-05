use super::*;

#[test]
fn type_int_8bit_requires_int8_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%u8 = OpTypeInt 8 0",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("8-bit int without Int8 capability should fail");
    assert!(
        matches!(err, ValidationError::TypeIntRequiresInt8Capability { .. }),
        "Expected TypeIntRequiresInt8Capability, got: {err:?}"
    );
}

#[test]
fn type_int_8bit_passes_with_int8_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int8",
        "OpMemoryModel Logical GLSL450",
        "%u8 = OpTypeInt 8 0",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("8-bit int with Int8 capability should pass");
}

#[test]
fn type_int_16bit_requires_int16_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%u16 = OpTypeInt 16 0",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("16-bit int without Int16 capability should fail");
    assert!(
        matches!(err, ValidationError::TypeIntRequiresInt16Capability { .. }),
        "Expected TypeIntRequiresInt16Capability, got: {err:?}"
    );
}

#[test]
fn type_int_16bit_passes_with_int16_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int16",
        "OpMemoryModel Logical GLSL450",
        "%u16 = OpTypeInt 16 0",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("16-bit int with Int16 capability should pass");
}

#[test]
fn type_int_32bit_always_valid() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%u32 = OpTypeInt 32 0",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6).expect("32-bit int should always be valid");
}

#[test]
fn type_int_64bit_requires_int64_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%u64 = OpTypeInt 64 0",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("64-bit int without Int64 capability should fail");
    assert!(
        matches!(err, ValidationError::TypeIntRequiresInt64Capability { .. }),
        "Expected TypeIntRequiresInt64Capability, got: {err:?}"
    );
}

#[test]
fn type_int_64bit_passes_with_int64_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability Int64",
        "OpMemoryModel Logical GLSL450",
        "%u64 = OpTypeInt 64 0",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("64-bit int with Int64 capability should pass");
}

#[test]
fn type_float_16bit_requires_float16_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%f16 = OpTypeFloat 16",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("16-bit float without Float16 capability should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeFloatRequiresFloat16Capability { .. }
        ),
        "Expected TypeFloatRequiresFloat16Capability, got: {err:?}"
    );
}

#[test]
fn type_float_16bit_passes_with_float16_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability Float16",
        "OpMemoryModel Logical GLSL450",
        "%f16 = OpTypeFloat 16",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("16-bit float with Float16 capability should pass");
}

#[test]
fn type_float_16bit_passes_with_float16buffer_capability() {
    // Float16Buffer requires Kernel capability
    let text = [
        "OpCapability Kernel",
        "OpCapability Float16Buffer",
        "OpCapability Addresses",
        "OpMemoryModel Physical64 OpenCL",
        "%f16 = OpTypeFloat 16",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("16-bit float with Float16Buffer capability should pass");
}

#[test]
fn type_float_32bit_always_valid() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%f32 = OpTypeFloat 32",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6).expect("32-bit float should always be valid");
}

#[test]
fn type_float_64bit_requires_float64_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%f64 = OpTypeFloat 64",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("64-bit float without Float64 capability should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeFloatRequiresFloat64Capability { .. }
        ),
        "Expected TypeFloatRequiresFloat64Capability, got: {err:?}"
    );
}

#[test]
fn type_float_64bit_passes_with_float64_capability() {
    let text = [
        "OpCapability Shader",
        "OpCapability Float64",
        "OpMemoryModel Logical GLSL450",
        "%f64 = OpTypeFloat 64",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("64-bit float with Float64 capability should pass");
}

#[test]
fn type_vector_8_components_requires_vector16_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%f32 = OpTypeFloat 32",
        "%vec8 = OpTypeVector %f32 8",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("8-component vector without Vector16 capability should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeVectorRequiresVector16Capability {
                component_count: 8,
                ..
            }
        ),
        "Expected TypeVectorRequiresVector16Capability with count 8, got: {err:?}"
    );
}

#[test]
fn type_vector_16_components_requires_vector16_capability() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%f32 = OpTypeFloat 32",
        "%vec16 = OpTypeVector %f32 16",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("16-component vector without Vector16 capability should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeVectorRequiresVector16Capability {
                component_count: 16,
                ..
            }
        ),
        "Expected TypeVectorRequiresVector16Capability with count 16, got: {err:?}"
    );
}

#[test]
fn type_vector_8_components_passes_with_vector16_capability() {
    // Vector16 requires Kernel capability
    let text = [
        "OpCapability Kernel",
        "OpCapability Vector16",
        "OpCapability Addresses",
        "OpMemoryModel Physical64 OpenCL",
        "%f32 = OpTypeFloat 32",
        "%vec8 = OpTypeVector %f32 8",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("8-component vector with Vector16 capability should pass");
}

#[test]
fn type_vector_2_to_4_components_always_valid() {
    for count in [2, 3, 4] {
        let text = format!(
            "OpCapability Shader\n\
             OpMemoryModel Logical GLSL450\n\
             %f32 = OpTypeFloat 32\n\
             %vec = OpTypeVector %f32 {count}"
        );
        let binary = assemble_text(&text).expect("assemble");
        validate_module(&binary, TargetEnv::Universal1_6)
            .unwrap_or_else(|e| panic!("{count}-component vector should be valid: {e:?}"));
    }
}

#[test]
fn type_vector_invalid_component_count() {
    // Vector with 5 components should fail (valid: 2, 3, 4, 8, 16)
    // Vector16 requires Kernel capability
    let text = [
        "OpCapability Kernel",
        "OpCapability Vector16",
        "OpCapability Addresses",
        "OpMemoryModel Physical64 OpenCL",
        "%f32 = OpTypeFloat 32",
        "%vec5 = OpTypeVector %f32 5",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = validate_module(&binary, TargetEnv::Universal1_6)
        .expect_err("5-component vector should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeVectorInvalidComponentCount {
                component_count: 5,
                ..
            }
        ),
        "Expected TypeVectorInvalidComponentCount with count 5, got: {err:?}"
    );
}

#[test]
fn type_vector_component_must_be_scalar() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let f32_ty = b.type_float(32, None);
    let vec2 = b.type_vector(f32_ty, 2);
    // Try to create vector of vectors (invalid)
    b.type_vector(vec2, 2);
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("Vector of vectors should fail");
    assert!(
        matches!(err, ValidationError::TypeVectorComponentNotScalar { .. }),
        "Expected TypeVectorComponentNotScalar, got: {err:?}"
    );
}

#[test]
fn type_matrix_column_must_be_vector() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let f32_ty = b.type_float(32, None);
    // Try to create matrix with scalar column (invalid)
    b.type_matrix(f32_ty, 2);
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("Matrix with scalar column should fail");
    assert!(
        matches!(err, ValidationError::TypeMatrixColumnNotVector { .. }),
        "Expected TypeMatrixColumnNotVector, got: {err:?}"
    );
}

#[test]
fn type_matrix_component_must_be_float() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let i32_ty = b.type_int(32, 1);
    let ivec2 = b.type_vector(i32_ty, 2);
    // Try to create matrix with integer vector column (invalid)
    b.type_matrix(ivec2, 2);
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("Matrix with integer vector column should fail");
    assert!(
        matches!(err, ValidationError::TypeMatrixComponentNotFloat { .. }),
        "Expected TypeMatrixComponentNotFloat, got: {err:?}"
    );
}

#[test]
fn type_matrix_valid_column_counts() {
    use rspirv::{binary::Assemble, dr::Builder};
    for cols in [2, 3, 4] {
        let mut b = Builder::new();
        b.set_version(1, 6);
        b.capability(rspirv::spirv::Capability::Shader);
        b.capability(rspirv::spirv::Capability::Matrix);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        let f32_ty = b.type_float(32, None);
        let vec2 = b.type_vector(f32_ty, 2);
        b.type_matrix(vec2, cols);
        let binary = b.module().assemble();
        binary
            .as_slice()
            .validate(TargetEnv::Universal1_6)
            .unwrap_or_else(|e| panic!("Matrix with {cols} columns should be valid: {e:?}"));
    }
}

#[test]
fn type_matrix_invalid_column_count() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Matrix);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let f32_ty = b.type_float(32, None);
    let vec2 = b.type_vector(f32_ty, 2);
    // 5 columns is invalid
    b.type_matrix(vec2, 5);
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("Matrix with 5 columns should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeMatrixInvalidColumnCount {
                column_count: 5,
                ..
            }
        ),
        "Expected TypeMatrixInvalidColumnCount with count 5, got: {err:?}"
    );
}

#[test]
fn type_array_element_cannot_be_void() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let i32_ty = b.type_int(32, 0);
    let const_1 = b.constant_bit32(i32_ty, 1);
    // Try to create array of void (invalid)
    b.type_array(void, const_1);
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("Array of void should fail");
    assert!(
        matches!(err, ValidationError::TypeArrayElementVoid { .. }),
        "Expected TypeArrayElementVoid, got: {err:?}"
    );
}

#[test]
fn type_array_length_must_be_positive() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let i32_ty = b.type_int(32, 0);
    let const_0 = b.constant_bit32(i32_ty, 0);
    // Try to create array with length 0 (invalid)
    b.type_array(i32_ty, const_0);
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("Array with length 0 should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeArrayLengthInvalid { length: 0, .. }
        ),
        "Expected TypeArrayLengthInvalid with length 0, got: {err:?}"
    );
}

#[test]
fn type_array_valid() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let i32_ty = b.type_int(32, 0);
    let const_10 = b.constant_bit32(i32_ty, 10);
    b.type_array(i32_ty, const_10);
    let binary = b.module().assemble();
    binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("Valid array type should pass");
}

#[test]
fn type_runtime_array_element_cannot_be_void() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = b.type_void();
    // Try to create runtime array of void (invalid)
    b.type_runtime_array(void);
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("Runtime array of void should fail");
    assert!(
        matches!(err, ValidationError::TypeRuntimeArrayElementVoid { .. }),
        "Expected TypeRuntimeArrayElementVoid, got: {err:?}"
    );
}

#[test]
fn type_runtime_array_valid() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let i32_ty = b.type_int(32, 0);
    b.type_runtime_array(i32_ty);
    let binary = b.module().assemble();
    binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("Valid runtime array type should pass");
}

#[test]
fn type_int_invalid_signedness() {
    // Signedness must be 0 or 1
    // We need to construct this manually since rspirv only allows valid values
    use rspirv::{binary::Assemble, dr::Builder, dr::Instruction, dr::Operand};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );

    // Manually add an OpTypeInt with invalid signedness (2)
    let mut module = b.module();
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeInt,
        None,
        Some(1),
        vec![
            Operand::LiteralBit32(32), // width
            Operand::LiteralBit32(2),  // signedness (invalid - should be 0 or 1)
        ],
    ));
    // Update the ID bound
    if let Some(ref mut header) = module.header {
        header.bound = 2;
    }

    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("OpTypeInt with signedness 2 should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeIntInvalidSignedness { signedness: 2, .. }
        ),
        "Expected TypeIntInvalidSignedness with signedness 2, got: {err:?}"
    );
}

#[test]
fn type_cooperative_matrix_khr_valid() {
    // Valid OpTypeCooperativeMatrixKHR with float component type
    use rspirv::{binary::Assemble, dr::Builder, dr::Instruction, dr::Operand};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::CooperativeMatrixKHR);
    b.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    b.extension("SPV_KHR_cooperative_matrix");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );

    let mut module = b.module();

    // IDs: 1=float, 2=scope_constant_type, 3=scope, 4=rows, 5=cols, 6=use, 7=matrix_type
    // Add OpTypeFloat %1 32
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeFloat,
        None,
        Some(1),
        vec![Operand::LiteralBit32(32)],
    ));
    // Add OpTypeInt %2 32 0
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeInt,
        None,
        Some(2),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    // Add OpConstant %3 = %2 3 (Scope::Workgroup = 3)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(3),
        vec![Operand::LiteralBit32(3)],
    ));
    // Add OpConstant %4 = %2 16 (rows)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(4),
        vec![Operand::LiteralBit32(16)],
    ));
    // Add OpConstant %5 = %2 16 (cols)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(5),
        vec![Operand::LiteralBit32(16)],
    ));
    // Add OpConstant %6 = %2 0 (Use::MatrixA = 0)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(6),
        vec![Operand::LiteralBit32(0)],
    ));
    // Add OpTypeCooperativeMatrixKHR %7 %1 %3 %4 %5 %6
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeCooperativeMatrixKHR,
        None,
        Some(7),
        vec![
            Operand::IdRef(1), // Component Type (float)
            Operand::IdRef(3), // Scope
            Operand::IdRef(4), // Rows
            Operand::IdRef(5), // Columns
            Operand::IdRef(6), // Use
        ],
    ));

    if let Some(ref mut header) = module.header {
        header.bound = 8;
    }

    let binary = module.assemble();
    binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_3)
        .expect("Valid OpTypeCooperativeMatrixKHR should pass");
}

#[test]
fn type_cooperative_matrix_khr_component_not_scalar() {
    // OpTypeCooperativeMatrixKHR with vector component type (invalid)
    use rspirv::{binary::Assemble, dr::Builder, dr::Instruction, dr::Operand};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::CooperativeMatrixKHR);
    b.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    b.extension("SPV_KHR_cooperative_matrix");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );

    let mut module = b.module();

    // IDs: 1=float, 2=vec4, 3=int, 4=scope, 5=rows, 6=cols, 7=use, 8=matrix_type
    // Add OpTypeFloat %1 32
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeFloat,
        None,
        Some(1),
        vec![Operand::LiteralBit32(32)],
    ));
    // Add OpTypeVector %2 %1 4 (vec4<f32>)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeVector,
        None,
        Some(2),
        vec![Operand::IdRef(1), Operand::LiteralBit32(4)],
    ));
    // Add OpTypeInt %3 32 0
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeInt,
        None,
        Some(3),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    // Add OpConstant %4 = %3 3 (Scope::Workgroup = 3)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(3),
        Some(4),
        vec![Operand::LiteralBit32(3)],
    ));
    // Add OpConstant %5 = %3 16 (rows)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(3),
        Some(5),
        vec![Operand::LiteralBit32(16)],
    ));
    // Add OpConstant %6 = %3 16 (cols)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(3),
        Some(6),
        vec![Operand::LiteralBit32(16)],
    ));
    // Add OpConstant %7 = %3 0 (Use::MatrixA = 0)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(3),
        Some(7),
        vec![Operand::LiteralBit32(0)],
    ));
    // Add OpTypeCooperativeMatrixKHR %8 %2 %4 %5 %6 %7 (component is vec4, invalid)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeCooperativeMatrixKHR,
        None,
        Some(8),
        vec![
            Operand::IdRef(2), // Component Type (vec4 - INVALID)
            Operand::IdRef(4), // Scope
            Operand::IdRef(5), // Rows
            Operand::IdRef(6), // Columns
            Operand::IdRef(7), // Use
        ],
    ));

    if let Some(ref mut header) = module.header {
        header.bound = 9;
    }

    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_3)
        .expect_err("OpTypeCooperativeMatrixKHR with vector component should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeCooperativeMatrixComponentNotScalar { .. }
        ),
        "Expected TypeCooperativeMatrixComponentNotScalar, got: {err:?}"
    );
}

#[test]
fn type_cooperative_matrix_khr_rows_not_positive() {
    // OpTypeCooperativeMatrixKHR with rows = 0 (invalid)
    use rspirv::{binary::Assemble, dr::Builder, dr::Instruction, dr::Operand};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::CooperativeMatrixKHR);
    b.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    b.extension("SPV_KHR_cooperative_matrix");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );

    let mut module = b.module();

    // IDs: 1=float, 2=int, 3=scope, 4=rows(0), 5=cols, 6=use, 7=matrix_type
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeFloat,
        None,
        Some(1),
        vec![Operand::LiteralBit32(32)],
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeInt,
        None,
        Some(2),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(3),
        vec![Operand::LiteralBit32(3)], // Scope::Workgroup
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(4),
        vec![Operand::LiteralBit32(0)], // Rows = 0 (INVALID)
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(5),
        vec![Operand::LiteralBit32(16)], // Columns
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(6),
        vec![Operand::LiteralBit32(0)], // Use::MatrixA
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeCooperativeMatrixKHR,
        None,
        Some(7),
        vec![
            Operand::IdRef(1), // Component Type
            Operand::IdRef(3), // Scope
            Operand::IdRef(4), // Rows (0 - INVALID)
            Operand::IdRef(5), // Columns
            Operand::IdRef(6), // Use
        ],
    ));

    if let Some(ref mut header) = module.header {
        header.bound = 8;
    }

    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_3)
        .expect_err("OpTypeCooperativeMatrixKHR with rows=0 should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeCooperativeMatrixRowsNotPositive { value: 0, .. }
        ),
        "Expected TypeCooperativeMatrixRowsNotPositive with value 0, got: {err:?}"
    );
}

#[test]
fn type_cooperative_matrix_khr_columns_not_positive() {
    // OpTypeCooperativeMatrixKHR with columns = 0 (invalid)
    use rspirv::{binary::Assemble, dr::Builder, dr::Instruction, dr::Operand};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::CooperativeMatrixKHR);
    b.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    b.extension("SPV_KHR_cooperative_matrix");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );

    let mut module = b.module();

    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeFloat,
        None,
        Some(1),
        vec![Operand::LiteralBit32(32)],
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeInt,
        None,
        Some(2),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(3),
        vec![Operand::LiteralBit32(3)], // Scope
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(4),
        vec![Operand::LiteralBit32(16)], // Rows
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(5),
        vec![Operand::LiteralBit32(0)], // Columns = 0 (INVALID)
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(6),
        vec![Operand::LiteralBit32(0)], // Use
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeCooperativeMatrixKHR,
        None,
        Some(7),
        vec![
            Operand::IdRef(1),
            Operand::IdRef(3),
            Operand::IdRef(4),
            Operand::IdRef(5), // Columns = 0
            Operand::IdRef(6),
        ],
    ));

    if let Some(ref mut header) = module.header {
        header.bound = 8;
    }

    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_3)
        .expect_err("OpTypeCooperativeMatrixKHR with columns=0 should fail");
    assert!(
        matches!(
            err,
            ValidationError::TypeCooperativeMatrixColumnsNotPositive { value: 0, .. }
        ),
        "Expected TypeCooperativeMatrixColumnsNotPositive with value 0, got: {err:?}"
    );
}

#[test]
fn type_cooperative_matrix_nv_valid() {
    // Valid OpTypeCooperativeMatrixNV (no Use operand)
    use rspirv::{binary::Assemble, dr::Builder, dr::Instruction, dr::Operand};
    let mut b = Builder::new();
    b.set_version(1, 5);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::CooperativeMatrixNV);
    b.extension("SPV_NV_cooperative_matrix");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );

    let mut module = b.module();

    // IDs: 1=float, 2=int, 3=scope, 4=rows, 5=cols, 6=matrix_type
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeFloat,
        None,
        Some(1),
        vec![Operand::LiteralBit32(32)],
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeInt,
        None,
        Some(2),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(3),
        vec![Operand::LiteralBit32(3)], // Scope::Workgroup
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(4),
        vec![Operand::LiteralBit32(16)], // Rows
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(2),
        Some(5),
        vec![Operand::LiteralBit32(16)], // Columns
    ));
    // OpTypeCooperativeMatrixNV %6 %1 %3 %4 %5 (no Use operand)
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeCooperativeMatrixNV,
        None,
        Some(6),
        vec![
            Operand::IdRef(1), // Component Type
            Operand::IdRef(3), // Scope
            Operand::IdRef(4), // Rows
            Operand::IdRef(5), // Columns
        ],
    ));

    if let Some(ref mut header) = module.header {
        header.bound = 7;
    }

    let binary = module.assemble();
    binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_2)
        .expect("Valid OpTypeCooperativeMatrixNV should pass");
}

#[test]
fn type_cooperative_matrix_with_int_component() {
    // Valid OpTypeCooperativeMatrixKHR with int component type
    use rspirv::{binary::Assemble, dr::Builder, dr::Instruction, dr::Operand};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::CooperativeMatrixKHR);
    b.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    b.extension("SPV_KHR_cooperative_matrix");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::VulkanKHR,
    );

    let mut module = b.module();

    // IDs: 1=int, 2=scope, 3=rows, 4=cols, 5=use, 6=matrix_type
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeInt,
        None,
        Some(1),
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(1),
        Some(2),
        vec![Operand::LiteralBit32(3)], // Scope::Workgroup
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(1),
        Some(3),
        vec![Operand::LiteralBit32(8)], // Rows
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(1),
        Some(4),
        vec![Operand::LiteralBit32(8)], // Columns
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::Constant,
        Some(1),
        Some(5),
        vec![Operand::LiteralBit32(2)], // Use::MatrixAccumulator = 2
    ));
    module.types_global_values.push(Instruction::new(
        rspirv::spirv::Op::TypeCooperativeMatrixKHR,
        None,
        Some(6),
        vec![
            Operand::IdRef(1), // Component Type (int)
            Operand::IdRef(2), // Scope
            Operand::IdRef(3), // Rows
            Operand::IdRef(4), // Columns
            Operand::IdRef(5), // Use
        ],
    ));

    if let Some(ref mut header) = module.header {
        header.bound = 7;
    }

    let binary = module.assemble();
    binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_3)
        .expect("Valid OpTypeCooperativeMatrixKHR with int component should pass");
}
