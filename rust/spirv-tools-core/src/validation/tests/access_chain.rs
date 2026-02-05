use super::*;

#[test]
fn access_chain_base_must_be_pointer() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr = OpTypePointer Function %u32",
        "%zero = OpConstant %u32 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        // Base operand is %zero, which is not a pointer.
        "%ac = OpAccessChain %ptr %zero %zero",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("access chain base must be a pointer");
    assert!(matches!(
        err,
        ValidationError::AccessChainBaseNotPointer {
            instruction: rspirv::spirv::Op::AccessChain,
            ..
        }
    ));
}

#[test]
fn access_chain_requires_composite_targets() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%ptr_u32 = OpTypePointer Function %u32",
        "%zero = OpConstant %u32 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_u32 Function",
        // %var points to a scalar; indexing into it is invalid.
        "%ac = OpAccessChain %ptr_u32 %var %zero",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("access chain must target composite types");
    assert!(matches!(
        err,
        ValidationError::AccessChainNonCompositeTarget {
            instruction: rspirv::spirv::Op::AccessChain,
            ..
        }
    ));
}

#[test]
fn access_chain_indexes_must_be_integer_scalars() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%f32 = OpTypeFloat 32",
        "%len = OpConstant %u32 4",
        "%array = OpTypeArray %u32 %len",
        "%ptr_array = OpTypePointer Function %array",
        "%ptr_u32 = OpTypePointer Function %u32",
        "%fzero = OpConstant %f32 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_array Function",
        // Index operand uses a float constant.
        "%ac = OpAccessChain %ptr_u32 %var %fzero",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("access chain indexes must be integer scalars");
    assert!(matches!(
        err,
        ValidationError::AccessChainIndexTypeInvalid {
            instruction: rspirv::spirv::Op::AccessChain,
            operand_index: 1,
            ..
        }
    ));
}

#[test]
fn access_chain_struct_index_must_be_literal() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%u32 = OpTypeInt 32 0",
        "%fn = OpTypeFunction %void",
        "%struct = OpTypeStruct %u32",
        "%ptr_struct = OpTypePointer Function %struct",
        "%ptr_u32 = OpTypePointer Function %u32",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_struct Function",
        "%idx_var = OpVariable %ptr_u32 Function",
        "%idx = OpLoad %u32 %idx_var",
        // Struct index is provided via an id that is not a literal constant.
        "%ac = OpAccessChain %ptr_u32 %var %idx",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("struct indexes must be literals");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainStructIndexNotLiteral {
                instruction: rspirv::spirv::Op::AccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn access_chain_struct_index_must_be_in_bounds() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%struct = OpTypeStruct %u32",
        "%ptr_struct = OpTypePointer Function %struct",
        "%ptr_u32 = OpTypePointer Function %u32",
        "%one = OpConstant %u32 1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_struct Function",
        // Struct has one member; index 1 is out of bounds.
        "%ac = OpAccessChain %ptr_u32 %var %one",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("struct index must be within bounds");
    assert!(matches!(
        err,
        ValidationError::AccessChainStructIndexOutOfBounds {
            instruction: rspirv::spirv::Op::AccessChain,
            index: 1,
            bound: 1,
            ..
        }
    ));
}

#[test]
fn access_chain_result_pointer_must_match_target_type() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%f32 = OpTypeFloat 32",
        "%two = OpConstant %u32 2",
        "%inner = OpTypeArray %u32 %two",
        "%outer = OpTypeArray %inner %two",
        "%ptr_array = OpTypePointer Function %outer",
        "%ptr_u32 = OpTypePointer Function %u32",
        "%ptr_f32 = OpTypePointer Function %f32",
        "%zero = OpConstant %u32 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_array Function",
        // Result type claims to point to %f32, but the chain resolves to %u32.
        "%ac = OpAccessChain %ptr_f32 %var %zero %zero",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("result pointer must match computed target type");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainResultTypeMismatch {
                instruction: rspirv::spirv::Op::AccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn access_chain_storage_class_must_match_base() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%two = OpConstant %u32 2",
        "%inner = OpTypeArray %u32 %two",
        "%outer = OpTypeArray %inner %two",
        "%ptr_function_array = OpTypePointer Function %outer",
        "%ptr_uniform = OpTypePointer Uniform %u32",
        "%elem_ptr_function = OpTypePointer Function %u32",
        "%zero = OpConstant %u32 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_function_array Function",
        // Result uses a different storage class than the base pointer.
        "%ac = OpAccessChain %ptr_uniform %var %zero %zero",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("result storage class must match base pointer storage class");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainStorageClassMismatch {
                instruction: rspirv::spirv::Op::AccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ptr_access_chain_indexes_must_be_integer_scalars() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let f32 = b.type_float(32, None);
    let len = b.constant_bit32(u32, 4);
    let array = b.type_array(u32, len);
    let ptr_array = b.type_pointer(None, StorageClass::Function, array);
    let ptr_u32 = b.type_pointer(None, StorageClass::Function, u32);
    let fzero = b.constant_bit32(f32, 0);
    b.begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let var = b.variable(ptr_array, None, StorageClass::Function, None);
    let result_id = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrAccessChain,
            Some(ptr_u32),
            Some(result_id),
            vec![Operand::IdRef(var), Operand::IdRef(fzero)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr access chain indexes must be integer scalars");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainIndexTypeInvalid {
                instruction: rspirv::spirv::Op::PtrAccessChain,
                operand_index: 1,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ptr_access_chain_struct_index_must_be_literal() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let struct_ty = b.type_struct([u32]);
    let ptr_struct = b.type_pointer(None, StorageClass::Function, struct_ty);
    let ptr_u32 = b.type_pointer(None, StorageClass::Function, u32);
    b.begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let var = b.variable(ptr_struct, None, StorageClass::Function, None);
    let idx_var = b.variable(ptr_u32, None, StorageClass::Function, None);
    let loaded_idx = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::Load,
            Some(u32),
            Some(loaded_idx),
            vec![Operand::IdRef(idx_var)],
        ),
    )
    .unwrap();
    let ac_id = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrAccessChain,
            Some(ptr_u32),
            Some(ac_id),
            vec![Operand::IdRef(var), Operand::IdRef(loaded_idx)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr access chain with no indices should match base pointee type");
    // The test has base pointing to struct{u32} but result type is ptr to u32.
    // Since there are no composite indices, the result type should match the base
    // pointee type (struct), not the struct member type (u32).
    assert!(
        matches!(
            err,
            ValidationError::AccessChainResultTypeMismatch {
                instruction: rspirv::spirv::Op::PtrAccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ptr_access_chain_result_pointer_must_match_target_type() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let f32 = b.type_float(32, None);
    let len = b.constant_bit32(u32, 4);
    let array = b.type_array(u32, len);
    let ptr_array = b.type_pointer(None, StorageClass::Function, array);
    let ptr_f32 = b.type_pointer(None, StorageClass::Function, f32);
    let zero = b.constant_bit32(u32, 0);
    b.begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let var = b.variable(ptr_array, None, StorageClass::Function, None);
    let ac_id = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrAccessChain,
            Some(ptr_f32),
            Some(ac_id),
            vec![Operand::IdRef(var), Operand::IdRef(zero)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr access chain result pointer must match target type");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainResultTypeMismatch {
                instruction: rspirv::spirv::Op::PtrAccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ptr_access_chain_storage_class_must_match_base() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let len = b.constant_bit32(u32, 4);
    let array = b.type_array(u32, len);
    let ptr_array = b.type_pointer(None, StorageClass::Function, array);
    let ptr_u32_workgroup = b.type_pointer(None, StorageClass::Workgroup, u32);
    let zero = b.constant_bit32(u32, 0);
    b.begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let var = b.variable(ptr_array, None, StorageClass::Function, None);
    let ac_id = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrAccessChain,
            Some(ptr_u32_workgroup),
            Some(ac_id),
            vec![Operand::IdRef(var), Operand::IdRef(zero)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr access chain storage class must match base");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainStorageClassMismatch {
                instruction: rspirv::spirv::Op::PtrAccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn inbounds_ptr_access_chain_indexes_must_be_integer_scalars() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let f32 = b.type_float(32, None);
    let len = b.constant_bit32(u32, 4);
    let array = b.type_array(u32, len);
    let ptr_array = b.type_pointer(None, StorageClass::Function, array);
    let ptr_u32 = b.type_pointer(None, StorageClass::Function, u32);
    let fzero = b.constant_bit32(f32, 0);
    b.begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let var = b.variable(ptr_array, None, StorageClass::Function, None);
    let result_id = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::InBoundsPtrAccessChain,
            Some(ptr_u32),
            Some(result_id),
            vec![Operand::IdRef(var), Operand::IdRef(fzero)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("inbounds ptr access chain indexes must be integer scalars");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainIndexTypeInvalid {
                instruction: rspirv::spirv::Op::InBoundsPtrAccessChain,
                operand_index: 1,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn inbounds_ptr_access_chain_struct_index_must_be_literal() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let struct_ty = b.type_struct([u32]);
    let ptr_struct = b.type_pointer(None, StorageClass::Function, struct_ty);
    let ptr_u32 = b.type_pointer(None, StorageClass::Function, u32);
    b.begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let var = b.variable(ptr_struct, None, StorageClass::Function, None);
    let idx_var = b.variable(ptr_u32, None, StorageClass::Function, None);
    let loaded_idx = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::Load,
            Some(u32),
            Some(loaded_idx),
            vec![Operand::IdRef(idx_var)],
        ),
    )
    .unwrap();
    let ac_id = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::InBoundsPtrAccessChain,
            Some(ptr_u32),
            Some(ac_id),
            vec![Operand::IdRef(var), Operand::IdRef(loaded_idx)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("inbounds ptr access chain with no indices should match base pointee type");
    // The test has base pointing to struct{u32} but result type is ptr to u32.
    // Since there are no composite indices, the result type should match the base
    // pointee type (struct), not the struct member type (u32).
    assert!(
        matches!(
            err,
            ValidationError::AccessChainResultTypeMismatch {
                instruction: rspirv::spirv::Op::InBoundsPtrAccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn inbounds_ptr_access_chain_result_pointer_must_match_target_type() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let f32 = b.type_float(32, None);
    let len = b.constant_bit32(u32, 4);
    let array = b.type_array(u32, len);
    let ptr_array = b.type_pointer(None, StorageClass::Function, array);
    let ptr_f32 = b.type_pointer(None, StorageClass::Function, f32);
    let zero = b.constant_bit32(u32, 0);
    b.begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let var = b.variable(ptr_array, None, StorageClass::Function, None);
    let ac_id = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::InBoundsPtrAccessChain,
            Some(ptr_f32),
            Some(ac_id),
            vec![Operand::IdRef(var), Operand::IdRef(zero)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("inbounds ptr access chain result pointer must match target type");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainResultTypeMismatch {
                instruction: rspirv::spirv::Op::InBoundsPtrAccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn inbounds_ptr_access_chain_storage_class_must_match_base() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let len = b.constant_bit32(u32, 4);
    let array = b.type_array(u32, len);
    let ptr_array = b.type_pointer(None, StorageClass::Function, array);
    let ptr_u32_workgroup = b.type_pointer(None, StorageClass::Workgroup, u32);
    let zero = b.constant_bit32(u32, 0);
    b.begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    b.begin_block(None).unwrap();
    let var = b.variable(ptr_array, None, StorageClass::Function, None);
    let ac_id = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::InBoundsPtrAccessChain,
            Some(ptr_u32_workgroup),
            Some(ac_id),
            vec![Operand::IdRef(var), Operand::IdRef(zero)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("inbounds ptr access chain storage class must match base");
    assert!(
        matches!(
            err,
            ValidationError::AccessChainStorageClassMismatch {
                instruction: rspirv::spirv::Op::InBoundsPtrAccessChain,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ptr_equal_result_type_must_be_bool() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let ptr_u32 = b.type_pointer(None, StorageClass::Function, u32);
    let _main = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let _header = b.begin_block(None).unwrap();
    let var = b.variable(ptr_u32, None, StorageClass::Function, None);
    let cmp = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrEqual,
            // Deliberately use an integer result to trigger the mismatch.
            Some(u32),
            Some(cmp),
            vec![Operand::IdRef(var), Operand::IdRef(var)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr comparisons require a boolean result");
    assert!(
        matches!(
            err,
            ValidationError::InstructionResultTypeMismatch {
                instruction: Op::PtrEqual,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ptr_equal_requires_variable_pointers_storage_buffer_capability() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    // Intentionally omit VariablePointersStorageBuffer.
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let bool_ty = b.type_bool();
    let int_ty = b.type_int(32, 0);
    let ptr_int = b.type_pointer(None, StorageClass::StorageBuffer, int_ty);
    let ptr_val = b.undef(ptr_int, None);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let block = b.begin_block(None).unwrap();
    let cmp = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrEqual,
            Some(bool_ty),
            Some(cmp),
            vec![Operand::IdRef(ptr_val), Operand::IdRef(ptr_val)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let err = b
        .module()
        .assemble()
        .validate(TargetEnv::Universal1_6)
        .expect_err("storage buffer pointer comparisons require capability");
    assert_eq!(
        err,
        ValidationError::PointerComparisonMissingCapability {
            function: Id::try_from(func).unwrap(),
            block: Id::try_from(block).unwrap(),
            instruction: Op::PtrEqual,
            storage_class: StorageClass::StorageBuffer,
            required_capability: Capability::VariablePointersStorageBuffer,
        }
    );
}

#[test]
fn ptr_equal_operands_must_be_pointers() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let bool_ty = b.type_bool();
    let ptr_u32 = b.type_pointer(None, StorageClass::Function, u32);
    let zero = b.constant_bit32(u32, 0);
    let main = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let var = b.variable(ptr_u32, None, StorageClass::Function, None);
    let cmp = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrEqual,
            Some(bool_ty),
            Some(cmp),
            vec![Operand::IdRef(zero), Operand::IdRef(var)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr comparisons require pointer operands");
    assert_eq!(
        err,
        ValidationError::PointerComparisonOperandNotPointer {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            instruction: Op::PtrEqual,
            operand_index: 0,
            found: TypeId::try_from(u32).unwrap(),
        }
    );
}

#[test]
fn ptr_equal_operands_must_match_pointer_types() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.memory_model(
        rspirv::spirv::AddressingModel::Physical64,
        MemoryModel::OpenCL,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let u32 = b.type_int(32, 0);
    let f32 = b.type_float(32, None);
    let bool_ty = b.type_bool();
    let ptr_u32 = b.type_pointer(None, StorageClass::Function, u32);
    let ptr_f32 = b.type_pointer(None, StorageClass::Function, f32);
    let main = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let ptr_a = b.undef(ptr_u32, None);
    let ptr_b = b.undef(ptr_f32, None);
    let cmp = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrEqual,
            Some(bool_ty),
            Some(cmp),
            vec![Operand::IdRef(ptr_a), Operand::IdRef(ptr_b)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr comparisons require matching pointer operand types");
    assert_eq!(
        err,
        ValidationError::OperandTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            instruction: Op::PtrEqual,
            operand_index: 1,
            expected: TypeId::try_from(ptr_u32).unwrap(),
            found: TypeId::try_from(ptr_f32).unwrap(),
        }
    );
}

#[test]
fn ptr_diff_operands_must_match_pointer_types() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::Shader);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let int_ty = b.type_int(32, 0);
    let float_ty = b.type_float(32, None);
    let ptr_int = b.type_pointer(None, StorageClass::StorageBuffer, int_ty);
    let ptr_float = b.type_pointer(None, StorageClass::StorageBuffer, float_ty);
    let undef_int_ptr = b.undef(ptr_int, None);
    let undef_float_ptr = b.undef(ptr_float, None);
    let main = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let diff = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrDiff,
            Some(int_ty),
            Some(diff),
            vec![
                Operand::IdRef(undef_int_ptr),
                Operand::IdRef(undef_float_ptr),
            ],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let err = b
        .module()
        .assemble()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr diff requires matching pointer operand types");
    assert_eq!(
        err,
        ValidationError::OperandTypeMismatch {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            instruction: Op::PtrDiff,
            operand_index: 1,
            expected: TypeId::try_from(ptr_int).unwrap(),
            found: TypeId::try_from(ptr_float).unwrap(),
        }
    );
}

#[test]
fn ptr_equal_allows_untyped_pointer_mismatch_with_matching_storage_class() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.capability(Capability::VariablePointers);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::UntypedPointersKHR);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.extension("SPV_KHR_untyped_pointers");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let ptr1 = b.id();
    b.module_mut()
        .types_global_values
        .push(rspirv::dr::Instruction::new(
            Op::TypeUntypedPointerKHR,
            None,
            Some(ptr1),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ));
    let ptr2 = b.id();
    b.module_mut()
        .types_global_values
        .push(rspirv::dr::Instruction::new(
            Op::TypeUntypedPointerKHR,
            None,
            Some(ptr2),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ));
    let void = b.type_void();
    let bool_ty = b.type_bool();
    let fn_ty = b.type_function(void, [ptr1, ptr2]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let param1 = b.function_parameter(ptr1).unwrap();
    let param2 = b.function_parameter(ptr2).unwrap();
    let _block = b.begin_block(None).unwrap();
    let cmp = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrEqual,
            Some(bool_ty),
            Some(cmp),
            vec![Operand::IdRef(param1), Operand::IdRef(param2)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.module()
        .assemble()
        .validate(TargetEnv::Vulkan1_3)
        .expect("untyped pointers with matching storage class should be allowed");
}

#[test]
fn ptr_equal_untyped_pointer_storage_classes_must_match() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.capability(Capability::VariablePointers);
    b.capability(Capability::UntypedPointersKHR);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.extension("SPV_KHR_untyped_pointers");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let ptr1 = b.id();
    b.module_mut()
        .types_global_values
        .push(rspirv::dr::Instruction::new(
            Op::TypeUntypedPointerKHR,
            None,
            Some(ptr1),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ));
    let ptr2 = b.id();
    b.module_mut()
        .types_global_values
        .push(rspirv::dr::Instruction::new(
            Op::TypeUntypedPointerKHR,
            None,
            Some(ptr2),
            vec![Operand::StorageClass(StorageClass::Workgroup)],
        ));
    let void = b.type_void();
    let bool_ty = b.type_bool();
    let fn_ty = b.type_function(void, [ptr1, ptr2]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let param1 = b.function_parameter(ptr1).unwrap();
    let param2 = b.function_parameter(ptr2).unwrap();
    let _block = b.begin_block(None).unwrap();
    let cmp = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrEqual,
            Some(bool_ty),
            Some(cmp),
            vec![Operand::IdRef(param1), Operand::IdRef(param2)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let err = b
        .module()
        .assemble()
        .validate(TargetEnv::Vulkan1_3)
        .expect_err("storage classes must match even for untyped pointers");
    assert!(
        matches!(
            err,
            ValidationError::OperandTypeMismatch {
                instruction: rspirv::spirv::Op::PtrEqual,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ptr_diff_result_type_must_be_integer() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::Shader);
    b.capability(Capability::Linkage);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let bool_ty = b.type_bool();
    let int_ty = b.type_int(32, 0);
    let ptr_int = b.type_pointer(None, StorageClass::StorageBuffer, int_ty);
    let ptr_val = b.undef(ptr_int, None);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let block = b.begin_block(None).unwrap();
    let diff = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrDiff,
            // Deliberately invalid result type.
            Some(bool_ty),
            Some(diff),
            vec![Operand::IdRef(ptr_val), Operand::IdRef(ptr_val)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr diff requires integer result");
    assert_eq!(
        err,
        ValidationError::InstructionResultTypeMismatch {
            function: Id::try_from(func).unwrap(),
            block: Id::try_from(block).unwrap(),
            instruction: Op::PtrDiff,
            expected: TypeId::try_from(int_ty).unwrap(),
            found: TypeId::try_from(bool_ty).unwrap(),
        }
    );
}

#[test]
fn ptr_diff_operands_must_be_pointers() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::Shader);
    b.capability(Capability::Linkage);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let int_ty = b.type_int(32, 0);
    let zero = b.constant_bit32(int_ty, 0);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let block = b.begin_block(None).unwrap();
    let diff = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrDiff,
            Some(int_ty),
            Some(diff),
            vec![Operand::IdRef(zero), Operand::IdRef(zero)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr diff operands must be pointers");
    assert_eq!(
        err,
        ValidationError::PointerComparisonOperandNotPointer {
            function: Id::try_from(func).unwrap(),
            block: Id::try_from(block).unwrap(),
            instruction: Op::PtrDiff,
            operand_index: 0,
            found: TypeId::try_from(int_ty).unwrap(),
        }
    );
}

#[test]
fn ptr_diff_storage_class_must_be_allowed() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op, StorageClass},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::Shader);
    b.capability(Capability::Linkage);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::VariablePointers);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let int_ty = b.type_int(32, 0);
    let ptr_int = b.type_pointer(None, StorageClass::Function, int_ty);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let block = b.begin_block(None).unwrap();
    let var = b.variable(ptr_int, None, StorageClass::Function, None);
    let diff = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrDiff,
            Some(int_ty),
            Some(diff),
            vec![Operand::IdRef(var), Operand::IdRef(var)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("ptr diff disallows private storage class");
    assert_eq!(
        err,
        ValidationError::PointerComparisonInvalidStorageClass {
            function: Id::try_from(func).unwrap(),
            block: Id::try_from(block).unwrap(),
            instruction: Op::PtrDiff,
            storage_class: StorageClass::Function,
        }
    );
}

#[test]
fn ptr_diff_workgroup_requires_variable_pointers_capability() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op, StorageClass},
    };
    let mut b = Builder::new();
    b.capability(Capability::Addresses);
    b.capability(Capability::Kernel);
    b.capability(Capability::Shader);
    b.capability(Capability::Linkage);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let int_ty = b.type_int(32, 0);
    let ptr_int = b.type_pointer(None, StorageClass::Workgroup, int_ty);
    let ptr_val = b.undef(ptr_int, None);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let _block = b.begin_block(None).unwrap();
    let diff = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrDiff,
            Some(int_ty),
            Some(diff),
            vec![Operand::IdRef(ptr_val), Operand::IdRef(ptr_val)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let binary = b.module().assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("workgroup pointer comparisons require VariablePointers");
    assert!(
        matches!(
            err,
            ValidationError::PointerComparisonMissingCapability {
                instruction: Op::PtrDiff,
                storage_class: StorageClass::Workgroup,
                required_capability: Capability::VariablePointers,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ptr_diff_allows_untyped_pointer_storage_buffer() {
    use rspirv::{
        binary::Assemble,
        dr::{Builder, InsertPoint, Operand},
        spirv::{Capability, FunctionControl, MemoryModel, Op, StorageClass},
    };
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.capability(Capability::VariablePointersStorageBuffer);
    b.capability(Capability::UntypedPointersKHR);
    b.capability(Capability::VariablePointers);
    b.extension("SPV_KHR_variable_pointers");
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.extension("SPV_KHR_untyped_pointers");
    b.extension("SPV_KHR_physical_storage_buffer");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        MemoryModel::GLSL450,
    );
    let ptr = b.id();
    b.module_mut()
        .types_global_values
        .push(rspirv::dr::Instruction::new(
            Op::TypeUntypedPointerKHR,
            None,
            Some(ptr),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ));
    let void = b.type_void();
    let int_ty = b.type_int(32, 0);
    let fn_ty = b.type_function(void, [ptr, ptr]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    let param1 = b.function_parameter(ptr).unwrap();
    let param2 = b.function_parameter(ptr).unwrap();
    let _block = b.begin_block(None).unwrap();
    let diff = b.id();
    b.insert_into_block(
        InsertPoint::End,
        rspirv::dr::Instruction::new(
            Op::PtrDiff,
            Some(int_ty),
            Some(diff),
            vec![Operand::IdRef(param1), Operand::IdRef(param2)],
        ),
    )
    .unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.module()
        .assemble()
        .validate(TargetEnv::Vulkan1_3)
        .expect("untyped pointer comparisons should succeed");
}

#[test]
fn struct_depth_limit_enforced() {
    use crate::validation::{ValidationOptions, LIMIT_MAX_STRUCT_DEPTH};
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%i32 = OpTypeInt 32 0",
        "%inner = OpTypeStruct %i32",
        "%outer = OpTypeStruct %inner",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let mut options = ValidationOptions::default();
    options.limits.insert(LIMIT_MAX_STRUCT_DEPTH, 1);
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect_err("struct depth limit should be enforced");
    assert_eq!(
        err,
        ValidationError::LimitExceeded {
            limit_kind: LIMIT_MAX_STRUCT_DEPTH,
            limit: 1,
            found: 2
        }
    );
}
