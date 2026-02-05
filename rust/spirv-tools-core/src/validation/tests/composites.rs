use super::*;

#[test]
fn vector_shuffle_operands_must_be_vectors() {
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
    let vec2 = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let lhs = b.constant_bit32(int, 0);
    let rhs = b.constant_bit32(int, 1);
    b.vector_shuffle(vec2, None, lhs, rhs, [0u32, 1]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector shuffle operands must be vectors");
    assert_eq!(
        err,
        ValidationError::VectorShuffleOperandNotVector {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            operand: 0,
            found: TypeId::try_from(int).unwrap(),
        }
    );
}

#[test]
fn vector_shuffle_component_types_must_match() {
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
    let v2i = b.type_vector(int, 2);
    let v2f = b.type_vector(float, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let v2i_id = b.undef(v2i, None);
    let v2f_id = b.undef(v2f, None);
    b.vector_shuffle(v2i, None, v2i_id, v2f_id, [0u32, 1])
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector shuffle operands must share the same component type");
    assert_eq!(
        err,
        ValidationError::VectorShuffleComponentTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            first: TypeId::try_from(int).unwrap(),
            second: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn vector_shuffle_result_length_must_match_components() {
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
    let v2 = b.type_vector(int, 2);
    let v3 = b.type_vector(int, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let v2_id = b.undef(v2, None);
    b.vector_shuffle(v3, None, v2_id, v2_id, [0u32, 1]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector shuffle result length must match literal component count");
    assert_eq!(
        err,
        ValidationError::VectorShuffleComponentCountMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            operand_components: 2,
            result_components: 3,
        }
    );
}

#[test]
fn vector_shuffle_indices_must_be_in_range() {
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
    let v2 = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let v2_id = b.undef(v2, None);
    b.vector_shuffle(v2, None, v2_id, v2_id, [0u32, 4]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector shuffle component indices must be in range or undef");
    assert_eq!(
        err,
        ValidationError::VectorShuffleComponentOutOfRange {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            value: 4,
            max: 3,
        }
    );
}

#[test]
fn vector_extract_dynamic_operand_must_be_vector() {
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
    let _vec2 = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let scalar = b.constant_bit32(int, 1);
    let index = b.constant_bit32(int, 0);
    b.vector_extract_dynamic(int, None, scalar, index).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector extract dynamic requires a vector operand");
    assert_eq!(
        err,
        ValidationError::VectorOperandNotVector {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            instruction: rspirv::spirv::Op::VectorExtractDynamic,
            operand: 0,
            found: TypeId::try_from(int).unwrap(),
        }
    );
}

#[test]
fn vector_extract_dynamic_result_type_must_match_component() {
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
    let vec2 = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let index = b.constant_bit32(int, 0);
    b.vector_extract_dynamic(float, None, vector, index)
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector extract dynamic result type must match component type");
    assert_eq!(
        err,
        ValidationError::InstructionResultTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            instruction: rspirv::spirv::Op::VectorExtractDynamic,
            expected: TypeId::try_from(int).unwrap(),
            found: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn vector_extract_dynamic_index_must_be_integer() {
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
    let vec2 = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let index = b.constant_bit32(float, 0.0f32.to_bits());
    b.vector_extract_dynamic(int, None, vector, index).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector extract dynamic index must be an integer scalar");
    assert_eq!(
        err,
        ValidationError::VectorIndexTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            instruction: rspirv::spirv::Op::VectorExtractDynamic,
            operand_index: 1,
            found: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn vector_insert_dynamic_result_type_must_match_vector() {
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
    let vec2 = b.type_vector(int, 2);
    let vec3 = b.type_vector(int, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let value = b.constant_bit32(int, 1);
    let index = b.constant_bit32(int, 0);
    b.vector_insert_dynamic(vec3, None, vector, value, index)
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector insert dynamic result type must match vector operand type");
    assert_eq!(
        err,
        ValidationError::InstructionResultTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            instruction: rspirv::spirv::Op::VectorInsertDynamic,
            expected: TypeId::try_from(vec2).unwrap(),
            found: TypeId::try_from(vec3).unwrap(),
        }
    );
}

#[test]
fn vector_insert_dynamic_component_type_must_match_vector() {
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
    let vec2 = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let value = b.constant_bit32(float, 1.0f32.to_bits());
    let index = b.constant_bit32(int, 0);
    b.vector_insert_dynamic(vec2, None, vector, value, index)
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector insert dynamic component type must match vector component");
    assert_eq!(
        err,
        ValidationError::OperandTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            instruction: rspirv::spirv::Op::VectorInsertDynamic,
            operand_index: 1,
            expected: TypeId::try_from(int).unwrap(),
            found: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn vector_insert_dynamic_index_must_be_integer() {
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
    let vec2 = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let value = b.constant_bit32(int, 1);
    let index = b.constant_bit32(float, 0.0f32.to_bits());
    b.vector_insert_dynamic(vec2, None, vector, value, index)
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector insert dynamic index must be an integer scalar");
    assert_eq!(
        err,
        ValidationError::VectorIndexTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            instruction: rspirv::spirv::Op::VectorInsertDynamic,
            operand_index: 2,
            found: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn vector_times_scalar_scalar_type_must_match_component() {
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
    let vec2 = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let scalar = b.constant_bit32(float, 0.0f32.to_bits());
    b.vector_times_scalar(vec2, None, vector, scalar).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector times scalar requires scalar to match vector component type");
    assert_eq!(
        err,
        ValidationError::ArithmeticResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            opcode: rspirv::spirv::Op::VectorTimesScalar,
            result_type: TypeId::try_from(vec2).unwrap(),
            expected: "float vector",
        }
    );
}

#[test]
fn vector_times_scalar_result_type_must_match_vector() {
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
    let vec2 = b.type_vector(int, 2);
    let v2f = b.type_vector(float, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let scalar = b.constant_bit32(int, 2);
    b.vector_times_scalar(v2f, None, vector, scalar).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector times scalar result type must match vector operand");
    assert_eq!(
        err,
        ValidationError::ArithmeticResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            opcode: rspirv::spirv::Op::VectorTimesScalar,
            result_type: TypeId::try_from(v2f).unwrap(),
            expected: "float vector",
        }
    );
}

#[test]
fn matrix_times_vector_component_type_must_match() {
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
    let vec2 = b.type_vector(int, 2);
    let vec2f = b.type_vector(float, 2);
    let mat2 = b.type_matrix(vec2, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let matrix = b.undef(mat2, None);
    let vector = b.undef(vec2f, None);
    b.matrix_times_vector(vec2, None, matrix, vector).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("matrix/vector multiply requires matching component types");
    assert_eq!(
        err,
        ValidationError::MatrixTimesVectorComponentTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            matrix_component: TypeId::try_from(int).unwrap(),
            vector_component: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn matrix_times_vector_dimensions_must_match() {
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
    let vec2 = b.type_vector(int, 2);
    let mat3x2 = b.type_matrix(vec2, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let matrix = b.undef(mat3x2, None);
    let vector = b.undef(vec2, None);
    b.matrix_times_vector(vec2, None, matrix, vector).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("matrix/vector multiply requires vector components to match matrix columns");
    assert_eq!(
        err,
        ValidationError::MatrixTimesVectorDimensionMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            matrix_columns: 3,
            vector_components: 2,
        }
    );
}

#[test]
fn matrix_times_vector_result_type_must_match_column() {
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
    let vec2 = b.type_vector(int, 2);
    let vec3 = b.type_vector(int, 3);
    let mat2 = b.type_matrix(vec2, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let matrix = b.undef(mat2, None);
    let vector = b.undef(vec2, None);
    b.matrix_times_vector(vec3, None, matrix, vector).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("matrix/vector multiply result type must match matrix column type");
    assert_eq!(
        err,
        ValidationError::ArithmeticResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            opcode: rspirv::spirv::Op::MatrixTimesVector,
            result_type: TypeId::try_from(vec3).unwrap(),
            expected: "matching column vector",
        }
    );
}

#[test]
fn vector_times_matrix_component_type_must_match() {
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
    let vec2 = b.type_vector(float, 2);
    let vec3 = b.type_vector(float, 3);
    let vec2i = b.type_vector(int, 2);
    let mat2x3 = b.type_matrix(vec2i, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let matrix = b.undef(mat2x3, None);
    b.vector_times_matrix(vec3, None, vector, matrix).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector/matrix multiply requires matching component types");
    assert_eq!(
        err,
        ValidationError::VectorTimesMatrixComponentTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            vector_component: TypeId::try_from(float).unwrap(),
            matrix_component: TypeId::try_from(int).unwrap(),
        }
    );
}

#[test]
fn vector_times_matrix_dimensions_must_match() {
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
    let vec3 = b.type_vector(int, 3);
    let vec4 = b.type_vector(int, 4);
    let mat2x3 = b.type_matrix(vec3, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec4, None);
    let matrix = b.undef(mat2x3, None);
    b.vector_times_matrix(vec3, None, vector, matrix).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector/matrix multiply requires vector length to match matrix rows");
    assert_eq!(
        err,
        ValidationError::VectorTimesMatrixDimensionMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            vector_components: 4,
            matrix_rows: 3,
        }
    );
}

#[test]
fn vector_times_matrix_result_dimensions_must_match_columns() {
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
    let vec2 = b.type_vector(int, 2);
    let vec4 = b.type_vector(int, 4);
    let mat2x3 = b.type_matrix(vec2, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let matrix = b.undef(mat2x3, None);
    b.vector_times_matrix(vec4, None, vector, matrix).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector/matrix multiply result length must equal matrix columns");
    assert_eq!(
        err,
        ValidationError::VectorTimesMatrixResultDimensionMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            expected_components: 3,
            found_components: 4,
        }
    );
}

#[test]
fn vector_times_matrix_result_component_type_must_match() {
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
    let vec2 = b.type_vector(int, 2);
    let vec3f = b.type_vector(float, 3);
    let mat2x3 = b.type_matrix(vec2, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let vector = b.undef(vec2, None);
    let matrix = b.undef(mat2x3, None);
    b.vector_times_matrix(vec3f, None, vector, matrix).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("vector/matrix multiply result component must match matrix component");
    assert_eq!(
        err,
        ValidationError::VectorTimesMatrixResultComponentTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            expected: TypeId::try_from(int).unwrap(),
            found: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn matrix_times_matrix_dimensions_must_match() {
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
    let vec2 = b.type_vector(int, 2);
    let mat2x3 = b.type_matrix(vec2, 3);
    let mat2x2 = b.type_matrix(vec2, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let left = b.undef(mat2x3, None);
    let right = b.undef(mat2x2, None);
    b.matrix_times_matrix(mat2x3, None, left, right).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("matrix/matrix multiply requires left columns to equal right rows");
    assert_eq!(
        err,
        ValidationError::MatrixTimesMatrixDimensionMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            left_columns: 3,
            right_rows: 2,
        }
    );
}

#[test]
fn matrix_times_matrix_component_types_must_match() {
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
    let vec2i = b.type_vector(int, 2);
    let vec2f = b.type_vector(float, 2);
    let mat2x2i = b.type_matrix(vec2i, 2);
    let mat2x2f = b.type_matrix(vec2f, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let left = b.undef(mat2x2i, None);
    let right = b.undef(mat2x2f, None);
    b.matrix_times_matrix(mat2x2i, None, left, right).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("matrix/matrix multiply requires matching component types");
    assert_eq!(
        err,
        ValidationError::MatrixTimesMatrixComponentTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            left_component: TypeId::try_from(int).unwrap(),
            right_component: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn matrix_times_matrix_result_shape_must_match_operands() {
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
    let vec2 = b.type_vector(int, 2);
    let vec3 = b.type_vector(int, 3);
    let mat2x3 = b.type_matrix(vec2, 3);
    let mat3x4 = b.type_matrix(vec3, 4);
    let mat3x3 = b.type_matrix(vec3, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let left = b.undef(mat2x3, None);
    let right = b.undef(mat3x4, None);
    b.matrix_times_matrix(mat3x3, None, left, right).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("matrix/matrix multiply result shape must match left rows and right columns");
    assert_eq!(
        err,
        ValidationError::MatrixTimesMatrixResultShapeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            expected_columns: 4,
            expected_rows: 2,
        }
    );
}

#[test]
fn matrix_times_matrix_result_component_type_must_match_operands() {
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
    let vec2i = b.type_vector(int, 2);
    let vec3i = b.type_vector(int, 3);
    let vec2f = b.type_vector(float, 2);
    let mat2x3 = b.type_matrix(vec2i, 3);
    let mat3x2 = b.type_matrix(vec3i, 2);
    let mat2x2f = b.type_matrix(vec2f, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let left = b.undef(mat2x3, None);
    let right = b.undef(mat3x2, None);
    b.matrix_times_matrix(mat2x2f, None, left, right).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err(
            "matrix/matrix multiply result component must match the operand component type",
        );
    assert_eq!(
        err,
        ValidationError::MatrixTimesMatrixResultComponentTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
            expected: TypeId::try_from(int).unwrap(),
            found: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn unreachable_definition_used_in_entry_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 1",
        "%fn = OpTypeFunction %int",
        "%one = OpConstant %int 1",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        "OpReturnValue %dead",
        "%deadblock = OpLabel",
        "%dead = OpIAdd %int %one %one",
        "OpReturnValue %dead",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("uses of values defined only in unreachable blocks must be rejected");
    // Value from unreachable block doesn't dominate the use
    assert!(matches!(err, ValidationError::ValueNotDominated { .. }));
}

#[test]
fn composite_extract_indexes_are_checked() {
    use rspirv::{binary::Assemble, dr::Builder};

    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );

    let void = b.type_void();
    let int = b.type_int(32, 1);
    let vec_ty = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let zero = b.constant_bit32(int, 0);
    let composite = b.constant_composite(vec_ty, [zero, zero]);
    b.composite_extract(int, None, composite, [3]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("composite extract indexes must be in range");
    assert_eq!(
        err,
        ValidationError::CompositeIndexOutOfBounds {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            instruction: rspirv::spirv::Op::CompositeExtract,
            composite_type: TypeId::try_from(vec_ty).unwrap(),
            index_position: 0,
            index: 3,
            bound: 2,
        }
    );
}

#[test]
fn composite_insert_requires_component_type() {
    use rspirv::{binary::Assemble, dr::Builder};

    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );

    let void = b.type_void();
    let int = b.type_int(32, 1);
    let uint = b.type_int(32, 0);
    let vec_ty = b.type_vector(int, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let object = b.constant_bit32(uint, 1);
    let zero = b.constant_bit32(int, 0);
    let composite = b.constant_composite(vec_ty, [zero, zero]);
    b.composite_insert(vec_ty, None, object, composite, [0])
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("composite insert requires component type to match object");
    assert_eq!(
        err,
        ValidationError::CompositeOperandTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::CompositeInsert,
            operand_index: 0,
            result_type: TypeId::try_from(vec_ty).unwrap(),
            expected: "matching component type",
        }
    );
}

#[test]
fn composite_insert_result_type_matches_composite() {
    use rspirv::{binary::Assemble, dr::Builder};

    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );

    let void = b.type_void();
    let int = b.type_int(32, 1);
    let vec2 = b.type_vector(int, 2);
    let vec3 = b.type_vector(int, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let zero = b.constant_bit32(int, 0);
    let composite = b.constant_composite(vec2, [zero, zero]);
    b.composite_insert(vec3, None, zero, composite, [1])
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("composite insert result type must match composite operand type");
    assert_eq!(
        err,
        ValidationError::CompositeResultTypeInvalid {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::CompositeInsert,
            result_type: TypeId::try_from(vec3).unwrap(),
            expected: "same type as composite operand",
        }
    );
}

#[test]
fn copy_object_result_type_matches_operand() {
    use rspirv::{binary::Assemble, dr::Builder};

    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
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
    let value = b.constant_bit32(int, 0);
    b.copy_object(float, None, value).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let binary = b.module().assemble();

    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("copy object result type must match operand type");
    assert_eq!(
        err,
        ValidationError::CompositeOperandTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            opcode: rspirv::spirv::Op::CopyObject,
            result_type: TypeId::try_from(float).unwrap(),
        }
    );
}

#[test]
fn load_result_type_matches_pointer_pointee() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeVoid",
        "%2 = OpTypeFunction %1",
        "%3 = OpTypeInt 32 0",
        "%4 = OpTypeFloat 32",
        "%5 = OpTypePointer Function %3",
        "%6 = OpFunction %1 None %2",
        "%7 = OpLabel",
        "%8 = OpVariable %5 Function",
        "%9 = OpLoad %4 %8",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("load result type must match pointer pointee type");
    assert_eq!(
        err,
        ValidationError::LoadResultTypeMismatch {
            result_type: TypeId::try_from(4).unwrap(),
            pointee_type: TypeId::try_from(3).unwrap(),
        }
    );
}

#[test]
fn composite_valid_vector_shuffle() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec4 = b.type_vector(f32_type, 4);
    let vec2 = b.type_vector(f32_type, 2);
    let fn_type = b.type_function(void, vec![]);

    let c = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let v = b.constant_composite(vec4, vec![c, c, c, c]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.vector_shuffle(vec2, None, v, v, vec![0, 2]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_valid_extract() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec4 = b.type_vector(f32_type, 4);
    let fn_type = b.type_function(void, vec![]);

    let c = b.constant_bit32(f32_type, 0x40000000); // 2.0f
    let v = b.constant_composite(vec4, vec![c, c, c, c]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.composite_extract(f32_type, None, v, vec![2]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_valid_insert() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec4 = b.type_vector(f32_type, 4);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let c2 = b.constant_bit32(f32_type, 0x42c60000); // 99.0f
    let v = b.constant_composite(vec4, vec![c1, c1, c1, c1]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.composite_insert(vec4, None, c2, v, vec![1]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_valid_construct() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec3 = b.type_vector(f32_type, 3);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let c2 = b.constant_bit32(f32_type, 0x40000000); // 2.0f
    let c3 = b.constant_bit32(f32_type, 0x40400000); // 3.0f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.composite_construct(vec3, None, vec![c1, c2, c3]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_valid_transpose() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec3 = b.type_vector(f32_type, 3);
    let vec4 = b.type_vector(f32_type, 4);
    let mat3x4 = b.type_matrix(vec4, 3); // 3 columns of vec4
    let mat4x3 = b.type_matrix(vec3, 4); // 4 columns of vec3 (transpose result)
    let fn_type = b.type_function(void, vec![]);

    let c = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let col = b.constant_composite(vec4, vec![c, c, c, c]);
    let m = b.constant_composite(mat3x4, vec![col, col, col]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.transpose(mat4x3, None, m).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_valid_copy_object() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let fn_type = b.type_function(void, vec![]);

    let val = b.constant_bit32(f32_type, 0x4048f5c3); // 3.14f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.copy_object(f32_type, None, val).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_construct_valid_vector() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec3 = b.type_vector(f32_type, 3);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let c2 = b.constant_bit32(f32_type, 0x40000000); // 2.0f
    let c3 = b.constant_bit32(f32_type, 0x40400000); // 3.0f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.composite_construct(vec3, None, vec![c1, c2, c3]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_construct_valid_vector_mixed() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec2 = b.type_vector(f32_type, 2);
    let vec4 = b.type_vector(f32_type, 4);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let c2 = b.constant_bit32(f32_type, 0x40000000); // 2.0f
    let vec2_const = b.constant_composite(vec2, vec![c1, c2]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    // Construct vec4 from vec2 + scalar + scalar = 4 components
    b.composite_construct(vec4, None, vec![vec2_const, c1, c2])
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_construct_valid_matrix() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec3 = b.type_vector(f32_type, 3);
    let mat2x3 = b.type_matrix(vec3, 2); // 2 columns of vec3
    let fn_type = b.type_function(void, vec![]);

    let c = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let col = b.constant_composite(vec3, vec![c, c, c]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.composite_construct(mat2x3, None, vec![col, col]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_construct_valid_array() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let u32_type = b.type_int(32, 0);
    let arr_len = b.constant_bit32(u32_type, 3);
    let arr_type = b.type_array(u32_type, arr_len);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(u32_type, 1);
    let c2 = b.constant_bit32(u32_type, 2);
    let c3 = b.constant_bit32(u32_type, 3);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.composite_construct(arr_type, None, vec![c1, c2, c3])
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_construct_valid_struct() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let u32_type = b.type_int(32, 0);
    let struct_type = b.type_struct(vec![f32_type, u32_type]);
    let fn_type = b.type_function(void, vec![]);

    let f_val = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let u_val = b.constant_bit32(u32_type, 42);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    b.composite_construct(struct_type, None, vec![f_val, u_val])
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn composite_construct_vector_too_few_constituents() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec3 = b.type_vector(f32_type, 3);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    // Only 1 constituent for vec3 - should fail
    b.composite_construct(vec3, None, vec![c1]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert!(
        matches!(
            err,
            ValidationError::CompositeConstructVectorTooFewConstituents { .. }
        ),
        "Expected CompositeConstructVectorTooFewConstituents error, got: {err:?}"
    );
}

#[test]
fn composite_construct_vector_component_count_mismatch() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec3 = b.type_vector(f32_type, 3);
    let fn_type = b.type_function(void, vec![]);

    let c1 = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let c2 = b.constant_bit32(f32_type, 0x40000000); // 2.0f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    // 2 scalars for vec3 - should fail (need 3)
    b.composite_construct(vec3, None, vec![c1, c2]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert!(
        matches!(
            err,
            ValidationError::CompositeConstructVectorComponentCountMismatch {
                expected: 3,
                given: 2,
                ..
            }
        ),
        "Expected CompositeConstructVectorComponentCountMismatch error, got: {err:?}"
    );
}

#[test]
fn composite_construct_matrix_column_count_mismatch() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let vec3 = b.type_vector(f32_type, 3);
    let mat3x3 = b.type_matrix(vec3, 3); // 3 columns
    let fn_type = b.type_function(void, vec![]);

    let c = b.constant_bit32(f32_type, 0x3f800000); // 1.0f
    let col = b.constant_composite(vec3, vec![c, c, c]);

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    // Only 2 columns for mat3x3 - should fail
    b.composite_construct(mat3x3, None, vec![col, col]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert!(
        matches!(
            err,
            ValidationError::CompositeConstructMatrixColumnCountMismatch {
                expected: 3,
                given: 2,
                ..
            }
        ),
        "Expected CompositeConstructMatrixColumnCountMismatch error, got: {err:?}"
    );
}

#[test]
fn composite_construct_struct_member_count_mismatch() {
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let f32_type = b.type_float(32, None);
    let u32_type = b.type_int(32, 0);
    let struct_type = b.type_struct(vec![f32_type, u32_type]); // 2 members
    let fn_type = b.type_function(void, vec![]);

    let f_val = b.constant_bit32(f32_type, 0x3f800000); // 1.0f

    let main_fn = b
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_fn, "main", vec![]);
    b.execution_mode(main_fn, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    // Only 1 constituent for struct with 2 members - should fail
    b.composite_construct(struct_type, None, vec![f_val])
        .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert!(
        matches!(
            err,
            ValidationError::CompositeConstructStructMemberCountMismatch {
                expected: 2,
                given: 1,
                ..
            }
        ),
        "Expected CompositeConstructStructMemberCountMismatch error, got: {err:?}"
    );
}

#[test]
fn copy_logical_same_types_fails() {
    // CopyLogical requires source and result types to be different but logically matching.
    // Using rspirv builder since our text assembler doesn't support OpCopyLogical.
    use rspirv::binary::Assemble;
    use rspirv::spirv::{AddressingModel, ExecutionMode as SpvExecutionMode, ExecutionModel};

    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void_type = b.type_void();
    let u32_type = b.type_int(32, 0);
    let c3 = b.constant_bit32(u32_type, 3);
    let arr3_type = b.type_array(u32_type, c3);
    let c1 = b.constant_bit32(u32_type, 1);
    let c2 = b.constant_bit32(u32_type, 2);
    let arr_val = b.constant_composite(arr3_type, vec![c1, c2, c3]);
    let fn_type = b.type_function(void_type, vec![]);

    let main_id = b
        .begin_function(void_type, None, FunctionControl::NONE, fn_type)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main_id, "main", vec![]);
    b.execution_mode(main_id, SpvExecutionMode::LocalSize, vec![1, 1, 1]);

    b.begin_block(None).unwrap();
    // CopyLogical with same type should fail
    b.copy_logical(arr3_type, None, arr_val).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert!(
        matches!(err, ValidationError::CopyLogicalTypesEqual { .. }),
        "Expected CopyLogicalTypesEqual error, got: {err:?}"
    );
}

#[test]
fn matrix_times_scalar_valid() {
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
    let mat2x2 = b.type_matrix(vec2, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let matrix = b.undef(mat2x2, None);
    let scalar = b.undef(float, None);
    b.matrix_times_scalar(mat2x2, None, matrix, scalar).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("Valid matrix times scalar should pass validation");
}

#[test]
fn matrix_times_scalar_result_must_be_matrix() {
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
    let mat2x2 = b.type_matrix(vec2, 2);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let matrix = b.undef(mat2x2, None);
    let scalar = b.undef(float, None);
    // Use wrong result type (float instead of matrix)
    b.matrix_times_scalar(float, None, matrix, scalar).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("MatrixTimesScalar with non-matrix result should fail");
    assert!(
        matches!(err, ValidationError::ArithmeticResultTypeInvalid { .. }),
        "Expected ArithmeticResultTypeInvalid, got: {err:?}"
    );
}

#[test]
fn outer_product_valid() {
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
    let vec3 = b.type_vector(float, 3);
    let mat2x3 = b.type_matrix(vec2, 3); // 2 rows, 3 columns
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let left = b.undef(vec2, None); // rows = 2
    let right = b.undef(vec3, None); // cols = 3
    b.outer_product(mat2x3, None, left, right).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("Valid outer product should pass validation");
}

#[test]
fn outer_product_result_must_be_matrix() {
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
    let vec3 = b.type_vector(float, 3);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let left = b.undef(vec2, None);
    let right = b.undef(vec3, None);
    // Use wrong result type (float instead of matrix)
    b.outer_product(float, None, left, right).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("OuterProduct with non-matrix result should fail");
    assert!(
        matches!(err, ValidationError::ArithmeticResultTypeInvalid { .. }),
        "Expected ArithmeticResultTypeInvalid, got: {err:?}"
    );
}
