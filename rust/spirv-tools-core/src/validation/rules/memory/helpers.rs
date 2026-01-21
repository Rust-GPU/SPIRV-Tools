//! Helper functions for memory operation validation.

use std::collections::HashMap;

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Decoration, Op, StorageClass};

use crate::validation::types::{Id, ResultId, TypeId};

/// Helper to convert a u32 to Id, returning a default Id(1) for zero values.
/// This is used for error reporting where we need an Id even if the actual
/// instruction had an invalid zero id.
pub fn id_from_u32(value: u32) -> Id {
    Id::try_from(value).unwrap_or_else(|_| Id::try_from(1).unwrap())
}

/// Helper to convert a u32 to TypeId, returning a default TypeId(1) for zero values.
pub fn type_id_from_u32(value: u32) -> TypeId {
    TypeId::try_from(value).unwrap_or_else(|_| TypeId::try_from(1).unwrap())
}

/// Helper to convert a u32 to ResultId, returning a default ResultId(1) for zero values.
pub fn result_id_from_u32(value: u32) -> ResultId {
    ResultId::try_from(value).unwrap_or_else(|_| ResultId::try_from(1).unwrap())
}

/// Gets the storage class from a pointer type instruction.
pub fn get_pointer_storage_class(type_inst: &Instruction) -> Option<StorageClass> {
    if type_inst.class.opcode != Op::TypePointer
        && type_inst.class.opcode != Op::TypeUntypedPointerKHR
    {
        return None;
    }
    match type_inst.operands.first() {
        Some(Operand::StorageClass(sc)) => Some(*sc),
        _ => None,
    }
}

/// Gets the pointee type ID from a pointer type instruction.
pub fn get_pointee_type(type_inst: &Instruction) -> Option<u32> {
    if type_inst.class.opcode != Op::TypePointer {
        return None;
    }
    match type_inst.operands.get(1) {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

/// Checks if a type contains OpTypeBool anywhere in it (recursively).
pub fn contains_bool(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
    visited: &mut std::collections::HashSet<u32>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }

    let Some(result_id) = ResultId::try_from(type_id).ok() else {
        return false;
    };
    let Some(inst) = definitions.get(&result_id) else {
        return false;
    };

    match inst.class.opcode {
        Op::TypeBool => true,
        Op::TypeVector | Op::TypeMatrix | Op::TypeArray | Op::TypeRuntimeArray => {
            if let Some(Operand::IdRef(elem_id)) = inst.operands.first() {
                contains_bool(*elem_id, definitions, visited)
            } else {
                false
            }
        }
        Op::TypeStruct => {
            for op in &inst.operands {
                if let Operand::IdRef(member_id) = op {
                    if contains_bool(*member_id, definitions, visited) {
                        return true;
                    }
                }
            }
            false
        }
        Op::TypePointer => {
            if let Some(Operand::IdRef(pointee_id)) = inst.operands.get(1) {
                contains_bool(*pointee_id, definitions, visited)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Checks if a type contains a runtime array anywhere in it.
pub fn contains_runtime_array(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
    visited: &mut std::collections::HashSet<u32>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }

    let Some(result_id) = ResultId::try_from(type_id).ok() else {
        return false;
    };
    let Some(inst) = definitions.get(&result_id) else {
        return false;
    };

    match inst.class.opcode {
        Op::TypeRuntimeArray => true,
        Op::TypeArray => {
            if let Some(Operand::IdRef(elem_id)) = inst.operands.first() {
                contains_runtime_array(*elem_id, definitions, visited)
            } else {
                false
            }
        }
        Op::TypeStruct => {
            for op in &inst.operands {
                if let Operand::IdRef(member_id) = op {
                    if contains_runtime_array(*member_id, definitions, visited) {
                        return true;
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Checks if the instruction produces a logical pointer.
pub fn is_logical_pointer_producer(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Variable
            | Op::AccessChain
            | Op::InBoundsAccessChain
            | Op::FunctionParameter
            | Op::ImageTexelPointer
            | Op::CopyObject
            | Op::Select
            | Op::Phi
            | Op::FunctionCall
            | Op::PtrAccessChain
            | Op::InBoundsPtrAccessChain
            | Op::Load
            | Op::ConstantNull
            | Op::Bitcast
            | Op::UntypedVariableKHR
            | Op::UntypedAccessChainKHR
            | Op::UntypedInBoundsAccessChainKHR
            | Op::UntypedPtrAccessChainKHR
            | Op::UntypedInBoundsPtrAccessChainKHR
    )
}

/// Checks if a storage class is read-only.
pub fn is_readonly_storage_class(sc: StorageClass) -> bool {
    matches!(
        sc,
        StorageClass::UniformConstant | StorageClass::Input | StorageClass::PushConstant
    )
}

/// Checks if the storage class allows NonPrivatePointer memory access.
pub fn allows_non_private_pointer(sc: StorageClass) -> bool {
    matches!(
        sc,
        StorageClass::Uniform
            | StorageClass::Workgroup
            | StorageClass::CrossWorkgroup
            | StorageClass::Generic
            | StorageClass::Image
            | StorageClass::StorageBuffer
            | StorageClass::PhysicalStorageBuffer
    )
}

/// Check if the instruction has a specific decoration.
pub fn has_decoration(module: &Module, id: u32, dec: Decoration) -> bool {
    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate {
            if let (Some(Operand::IdRef(target)), Some(Operand::Decoration(d))) =
                (inst.operands.first(), inst.operands.get(1))
            {
                if *target == id && *d == dec {
                    return true;
                }
            }
        }
    }
    false
}

/// Returns the size in bytes of the largest scalar type within a type.
///
/// For scalar types (int, float, bool), returns the size in bytes.
/// For composite types (struct, array, vector), returns the largest scalar size found recursively.
///
/// This is used to validate that PhysicalStorageBuffer Aligned values are
/// at least as large as the largest scalar type being accessed.
pub fn get_largest_scalar_type(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
    visited: &mut std::collections::HashSet<u32>,
) -> u32 {
    // Avoid infinite recursion with cyclic types
    if !visited.insert(type_id) {
        return 0;
    }

    let Some(result_id) = ResultId::try_from(type_id).ok() else {
        return 0;
    };
    let Some(inst) = definitions.get(&result_id) else {
        return 0;
    };

    match inst.class.opcode {
        Op::TypeStruct => {
            // Find the largest scalar among all struct members
            let mut max_size = 0u32;
            for op in &inst.operands {
                if let Operand::IdRef(member_id) = op {
                    let member_size = get_largest_scalar_type(*member_id, definitions, visited);
                    max_size = max_size.max(member_size);
                }
            }
            max_size
        }
        Op::TypeArray | Op::TypeRuntimeArray => {
            // Get the element type's largest scalar
            if let Some(Operand::IdRef(elem_id)) = inst.operands.first() {
                get_largest_scalar_type(*elem_id, definitions, visited)
            } else {
                0
            }
        }
        Op::TypeVector | Op::TypeMatrix => {
            // Get the component type's largest scalar
            if let Some(Operand::IdRef(comp_id)) = inst.operands.first() {
                get_largest_scalar_type(*comp_id, definitions, visited)
            } else {
                0
            }
        }
        Op::TypeInt | Op::TypeFloat => {
            // Return bit width / 8 (size in bytes)
            if let Some(Operand::LiteralBit32(width)) = inst.operands.first() {
                *width / 8
            } else {
                0
            }
        }
        Op::TypeBool => {
            // Bool is typically 1 byte, but SPIR-V doesn't define its size explicitly
            // The C++ code uses GetBitWidth which returns the abstract width, not physical
            // For bool, we return 0 to match the C++ behavior (GetBitWidth returns 0 for bool)
            0
        }
        Op::TypePointer => {
            // For pointers, we look at the pointee type
            if let Some(Operand::IdRef(pointee_id)) = inst.operands.get(1) {
                get_largest_scalar_type(*pointee_id, definitions, visited)
            } else {
                0
            }
        }
        _ => 0,
    }
}
