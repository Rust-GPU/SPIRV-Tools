//! Extension traits for rspirv types to support type introspection.
//!
//! This module provides extension traits on rspirv's `Instruction` type to query
//! SPIR-V type properties without creating intermediate data structures.

use std::collections::HashMap;

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{Op, StorageClass};

use super::types::ResultId;

/// Extension trait for type introspection on SPIR-V type instructions.
pub trait TypeInstructionExt {
    /// Returns true if this instruction defines a type.
    fn is_type_instruction(&self) -> bool;

    /// Returns true if this instruction defines OpTypeVoid.
    fn is_void_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeBool.
    fn is_bool_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeInt.
    fn is_int_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeFloat.
    fn is_float_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeVector.
    fn is_vector_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeMatrix.
    fn is_matrix_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeArray.
    fn is_array_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeRuntimeArray.
    fn is_runtime_array_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeStruct.
    fn is_struct_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypePointer.
    fn is_pointer_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeImage.
    fn is_image_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeSampler.
    fn is_sampler_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeSampledImage.
    fn is_sampled_image_type(&self) -> bool;

    /// Returns true if this instruction defines OpTypeFunction.
    fn is_function_type(&self) -> bool;

    /// Returns true if this instruction defines a cooperative matrix type.
    fn is_cooperative_matrix_type(&self) -> bool;

    /// Returns the bit width if this is OpTypeInt or OpTypeFloat.
    fn numeric_bit_width(&self) -> Option<u32>;

    /// Returns the signedness if this is OpTypeInt (0 = unsigned, 1 = signed).
    fn int_signedness(&self) -> Option<u32>;

    /// Returns true if this is an unsigned integer type.
    fn is_unsigned_int_type(&self) -> bool;

    /// Returns true if this is a signed integer type.
    fn is_signed_int_type(&self) -> bool;

    /// Returns the vector component count if this is OpTypeVector.
    fn vector_component_count(&self) -> Option<u32>;

    /// Returns the vector component type ID if this is OpTypeVector.
    fn vector_component_type_id(&self) -> Option<u32>;

    /// Returns the matrix column count if this is OpTypeMatrix.
    fn matrix_column_count(&self) -> Option<u32>;

    /// Returns the matrix column type ID (a vector) if this is OpTypeMatrix.
    fn matrix_column_type_id(&self) -> Option<u32>;

    /// Returns the storage class if this is OpTypePointer.
    fn pointer_storage_class(&self) -> Option<StorageClass>;

    /// Returns the pointee type ID if this is OpTypePointer.
    fn pointer_pointee_type_id(&self) -> Option<u32>;

    /// Returns the struct member type IDs if this is OpTypeStruct.
    fn struct_member_type_ids(&self) -> Option<Vec<u32>>;

    /// Returns the array element type ID if this is OpTypeArray or OpTypeRuntimeArray.
    fn array_element_type_id(&self) -> Option<u32>;

    /// Returns the function return type ID if this is OpTypeFunction.
    fn function_return_type_id(&self) -> Option<u32>;

    /// Returns the function parameter type IDs if this is OpTypeFunction.
    fn function_parameter_type_ids(&self) -> Option<Vec<u32>>;
}

impl TypeInstructionExt for Instruction {
    fn is_type_instruction(&self) -> bool {
        matches!(
            self.class.opcode,
            Op::TypeVoid
                | Op::TypeBool
                | Op::TypeInt
                | Op::TypeFloat
                | Op::TypeVector
                | Op::TypeMatrix
                | Op::TypeImage
                | Op::TypeSampler
                | Op::TypeSampledImage
                | Op::TypeArray
                | Op::TypeRuntimeArray
                | Op::TypeStruct
                | Op::TypeOpaque
                | Op::TypePointer
                | Op::TypeUntypedPointerKHR
                | Op::TypeFunction
                | Op::TypeEvent
                | Op::TypeDeviceEvent
                | Op::TypeReserveId
                | Op::TypeQueue
                | Op::TypePipe
                | Op::TypeForwardPointer
                | Op::TypePipeStorage
                | Op::TypeNamedBarrier
                | Op::TypeAccelerationStructureKHR
                | Op::TypeCooperativeMatrixKHR
                | Op::TypeCooperativeMatrixNV
                | Op::TypeRayQueryKHR
                | Op::TypeHitObjectNV
        )
    }

    fn is_void_type(&self) -> bool {
        self.class.opcode == Op::TypeVoid
    }

    fn is_bool_type(&self) -> bool {
        self.class.opcode == Op::TypeBool
    }

    fn is_int_type(&self) -> bool {
        self.class.opcode == Op::TypeInt
    }

    fn is_float_type(&self) -> bool {
        self.class.opcode == Op::TypeFloat
    }

    fn is_vector_type(&self) -> bool {
        self.class.opcode == Op::TypeVector
    }

    fn is_matrix_type(&self) -> bool {
        self.class.opcode == Op::TypeMatrix
    }

    fn is_array_type(&self) -> bool {
        self.class.opcode == Op::TypeArray
    }

    fn is_runtime_array_type(&self) -> bool {
        self.class.opcode == Op::TypeRuntimeArray
    }

    fn is_struct_type(&self) -> bool {
        self.class.opcode == Op::TypeStruct
    }

    fn is_pointer_type(&self) -> bool {
        matches!(
            self.class.opcode,
            Op::TypePointer | Op::TypeUntypedPointerKHR
        )
    }

    fn is_image_type(&self) -> bool {
        self.class.opcode == Op::TypeImage
    }

    fn is_sampler_type(&self) -> bool {
        self.class.opcode == Op::TypeSampler
    }

    fn is_sampled_image_type(&self) -> bool {
        self.class.opcode == Op::TypeSampledImage
    }

    fn is_function_type(&self) -> bool {
        self.class.opcode == Op::TypeFunction
    }

    fn is_cooperative_matrix_type(&self) -> bool {
        matches!(
            self.class.opcode,
            Op::TypeCooperativeMatrixKHR | Op::TypeCooperativeMatrixNV
        )
    }

    fn numeric_bit_width(&self) -> Option<u32> {
        if !self.is_int_type() && !self.is_float_type() {
            return None;
        }
        self.operands.first().and_then(literal_u32)
    }

    fn int_signedness(&self) -> Option<u32> {
        if !self.is_int_type() {
            return None;
        }
        self.operands.get(1).and_then(literal_u32)
    }

    fn is_unsigned_int_type(&self) -> bool {
        self.is_int_type() && self.int_signedness() == Some(0)
    }

    fn is_signed_int_type(&self) -> bool {
        self.is_int_type() && self.int_signedness() != Some(0)
    }

    fn vector_component_count(&self) -> Option<u32> {
        if !self.is_vector_type() {
            return None;
        }
        self.operands.get(1).and_then(literal_u32)
    }

    fn vector_component_type_id(&self) -> Option<u32> {
        if !self.is_vector_type() {
            return None;
        }
        self.operands.first().and_then(id_ref)
    }

    fn matrix_column_count(&self) -> Option<u32> {
        if !self.is_matrix_type() {
            return None;
        }
        self.operands.get(1).and_then(literal_u32)
    }

    fn matrix_column_type_id(&self) -> Option<u32> {
        if !self.is_matrix_type() {
            return None;
        }
        self.operands.first().and_then(id_ref)
    }

    fn pointer_storage_class(&self) -> Option<StorageClass> {
        if !self.is_pointer_type() {
            return None;
        }
        self.operands.first().and_then(|op| match op {
            Operand::StorageClass(sc) => Some(*sc),
            _ => None,
        })
    }

    fn pointer_pointee_type_id(&self) -> Option<u32> {
        if self.class.opcode == Op::TypeUntypedPointerKHR {
            return None; // Untyped pointers have no pointee
        }
        if !self.is_pointer_type() {
            return None;
        }
        self.operands.get(1).and_then(id_ref)
    }

    fn struct_member_type_ids(&self) -> Option<Vec<u32>> {
        if !self.is_struct_type() {
            return None;
        }
        Some(self.operands.iter().filter_map(id_ref).collect())
    }

    fn array_element_type_id(&self) -> Option<u32> {
        if !self.is_array_type() && !self.is_runtime_array_type() {
            return None;
        }
        self.operands.first().and_then(id_ref)
    }

    fn function_return_type_id(&self) -> Option<u32> {
        if !self.is_function_type() {
            return None;
        }
        self.operands.first().and_then(id_ref)
    }

    fn function_parameter_type_ids(&self) -> Option<Vec<u32>> {
        if !self.is_function_type() {
            return None;
        }
        Some(self.operands.iter().skip(1).filter_map(id_ref).collect())
    }
}

/// Extension trait for resolving type properties through a definitions map.
pub trait TypeResolver {
    /// Returns true if the type is a float scalar type.
    fn is_float_scalar(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool;

    /// Returns true if the type is a float scalar or vector type.
    fn is_float_scalar_or_vector(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> bool;

    /// Returns true if the type is an int scalar type.
    fn is_int_scalar(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool;

    /// Returns true if the type is an int scalar or vector type.
    fn is_int_scalar_or_vector(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> bool;

    /// Returns true if the type is an unsigned int scalar or vector type.
    fn is_unsigned_int_scalar_or_vector(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> bool;

    /// Returns true if the type is a bool scalar type.
    fn is_bool_scalar(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool;

    /// Returns true if the type is a bool scalar or vector type.
    fn is_bool_scalar_or_vector(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> bool;

    /// Returns the bit width if the type is a numeric scalar or the component of a numeric vector.
    fn get_bit_width(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>)
        -> Option<u32>;

    /// Returns the dimension (1 for scalar, N for vecN).
    fn get_dimension(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> u32;
}

/// Default type resolver that uses a definitions map.
pub struct DefaultTypeResolver;

impl TypeResolver for DefaultTypeResolver {
    fn is_float_scalar(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
        get_type_instruction(type_id, definitions)
            .map(|inst| inst.is_float_type())
            .unwrap_or(false)
    }

    fn is_float_scalar_or_vector(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> bool {
        let Some(inst) = get_type_instruction(type_id, definitions) else {
            return false;
        };

        if inst.is_float_type() {
            return true;
        }

        if inst.is_vector_type() {
            if let Some(comp_id) = inst.vector_component_type_id() {
                return self.is_float_scalar(comp_id, definitions);
            }
        }

        false
    }

    fn is_int_scalar(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
        get_type_instruction(type_id, definitions)
            .map(|inst| inst.is_int_type())
            .unwrap_or(false)
    }

    fn is_int_scalar_or_vector(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> bool {
        let Some(inst) = get_type_instruction(type_id, definitions) else {
            return false;
        };

        if inst.is_int_type() {
            return true;
        }

        if inst.is_vector_type() {
            if let Some(comp_id) = inst.vector_component_type_id() {
                return self.is_int_scalar(comp_id, definitions);
            }
        }

        false
    }

    fn is_unsigned_int_scalar_or_vector(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> bool {
        let Some(inst) = get_type_instruction(type_id, definitions) else {
            return false;
        };

        if inst.is_unsigned_int_type() {
            return true;
        }

        if inst.is_vector_type() {
            if let Some(comp_id) = inst.vector_component_type_id() {
                return get_type_instruction(comp_id, definitions)
                    .map(|i| i.is_unsigned_int_type())
                    .unwrap_or(false);
            }
        }

        false
    }

    fn is_bool_scalar(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
        get_type_instruction(type_id, definitions)
            .map(|inst| inst.is_bool_type())
            .unwrap_or(false)
    }

    fn is_bool_scalar_or_vector(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> bool {
        let Some(inst) = get_type_instruction(type_id, definitions) else {
            return false;
        };

        if inst.is_bool_type() {
            return true;
        }

        if inst.is_vector_type() {
            if let Some(comp_id) = inst.vector_component_type_id() {
                return self.is_bool_scalar(comp_id, definitions);
            }
        }

        false
    }

    fn get_bit_width(
        &self,
        type_id: u32,
        definitions: &HashMap<ResultId, Instruction>,
    ) -> Option<u32> {
        let inst = get_type_instruction(type_id, definitions)?;

        if let Some(width) = inst.numeric_bit_width() {
            return Some(width);
        }

        if inst.is_vector_type() {
            if let Some(comp_id) = inst.vector_component_type_id() {
                return self.get_bit_width(comp_id, definitions);
            }
        }

        None
    }

    fn get_dimension(&self, type_id: u32, definitions: &HashMap<ResultId, Instruction>) -> u32 {
        let Some(inst) = get_type_instruction(type_id, definitions) else {
            return 1;
        };

        if inst.is_vector_type() {
            inst.vector_component_count().unwrap_or(1)
        } else {
            1
        }
    }
}

// Helper functions

fn get_type_instruction(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<&Instruction> {
    ResultId::try_from(type_id)
        .ok()
        .and_then(|rid| definitions.get(&rid))
}

fn literal_u32(operand: &Operand) -> Option<u32> {
    match operand {
        Operand::LiteralBit32(v) => Some(*v),
        Operand::LiteralBit64(v) => Some(*v as u32),
        _ => None,
    }
}

fn id_ref(operand: &Operand) -> Option<u32> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_instruction_ext_int() {
        let inst = Instruction::new(
            Op::TypeInt,
            None,
            Some(1),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(1)],
        );

        assert!(inst.is_type_instruction());
        assert!(inst.is_int_type());
        assert!(!inst.is_float_type());
        assert_eq!(inst.numeric_bit_width(), Some(32));
        assert_eq!(inst.int_signedness(), Some(1));
        assert!(inst.is_signed_int_type());
        assert!(!inst.is_unsigned_int_type());
    }

    #[test]
    fn test_type_instruction_ext_unsigned_int() {
        let inst = Instruction::new(
            Op::TypeInt,
            None,
            Some(1),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        );

        assert!(inst.is_unsigned_int_type());
        assert!(!inst.is_signed_int_type());
    }

    #[test]
    fn test_type_instruction_ext_float() {
        let inst = Instruction::new(
            Op::TypeFloat,
            None,
            Some(1),
            vec![Operand::LiteralBit32(32)],
        );

        assert!(inst.is_type_instruction());
        assert!(inst.is_float_type());
        assert!(!inst.is_int_type());
        assert_eq!(inst.numeric_bit_width(), Some(32));
    }

    #[test]
    fn test_type_instruction_ext_vector() {
        let inst = Instruction::new(
            Op::TypeVector,
            None,
            Some(2),
            vec![Operand::IdRef(1), Operand::LiteralBit32(4)],
        );

        assert!(inst.is_vector_type());
        assert_eq!(inst.vector_component_type_id(), Some(1));
        assert_eq!(inst.vector_component_count(), Some(4));
    }

    #[test]
    fn test_type_instruction_ext_pointer() {
        let inst = Instruction::new(
            Op::TypePointer,
            None,
            Some(3),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(1),
            ],
        );

        assert!(inst.is_pointer_type());
        assert_eq!(inst.pointer_storage_class(), Some(StorageClass::Function));
        assert_eq!(inst.pointer_pointee_type_id(), Some(1));
    }
}
