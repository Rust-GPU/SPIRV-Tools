//! Helper functions for type validation rules.
//!
//! This module contains shared helper functions used across type validation rules.

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::ValidationContext;
use crate::validation::types::ResultId;

/// Checks if an opcode is a scalar type (bool, int, or float).
pub fn is_scalar_type(opcode: Op) -> bool {
    matches!(opcode, Op::TypeBool | Op::TypeInt | Op::TypeFloat)
}

/// Check if opcode is a scalar numeric type (int or float, but not bool).
pub fn is_scalar_numeric_type(opcode: Op) -> bool {
    matches!(opcode, Op::TypeInt | Op::TypeFloat)
}

/// Checks if an opcode produces a constant value.
pub fn is_constant_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Constant
            | Op::ConstantTrue
            | Op::ConstantFalse
            | Op::ConstantNull
            | Op::ConstantComposite
            | Op::ConstantSampler
            | Op::SpecConstant
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstantComposite
            | Op::SpecConstantOp
    )
}

/// Try to extract a constant integer value from a constant instruction.
pub fn get_constant_int_value(
    inst: &rspirv::dr::Instruction,
    ctx: &ValidationContext<'_>,
) -> Option<i64> {
    if inst.class.opcode != Op::Constant {
        return None;
    }

    // Get the type to determine signedness and bit width
    let type_id = inst.result_type?;
    let type_result_id = ResultId::try_from(type_id).ok()?;
    let type_inst = ctx.definitions.get(&type_result_id)?;

    if type_inst.class.opcode != Op::TypeInt {
        return None;
    }

    let width = match type_inst.operands.first() {
        Some(Operand::LiteralBit32(w)) => *w,
        _ => return None,
    };

    let signedness = match type_inst.operands.get(1) {
        Some(Operand::LiteralBit32(s)) => *s,
        _ => return None,
    };

    // Get the constant value
    let value = match inst.operands.first() {
        Some(Operand::LiteralBit32(v)) => *v as u64,
        Some(Operand::LiteralBit64(v)) => *v,
        _ => return None,
    };

    // Convert based on signedness and width
    if signedness != 0 {
        // Signed integer
        match width {
            8 => Some(value as i8 as i64),
            16 => Some(value as i16 as i64),
            32 => Some(value as i32 as i64),
            64 => Some(value as i64),
            _ => None,
        }
    } else {
        // Unsigned integer (treat as positive)
        Some(value as i64)
    }
}

/// Checks if an opcode is a type instruction.
pub fn is_type_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
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
