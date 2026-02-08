//! Parsing for arithmetic, comparison, and logical operations.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

use super::util::{parse_binary_args, parse_ternary_args, parse_unary_arg};

/// Binary integer and comparison operations.
const BINARY_OPS: &[(&str, Op)] = &[
    // Integer arithmetic
    ("Add", Op::IAdd),
    ("Sub", Op::ISub),
    ("Mul", Op::IMul),
    ("SDiv", Op::SDiv),
    ("UDiv", Op::UDiv),
    ("SRem", Op::SRem),
    ("SMod", Op::SMod),
    ("UMod", Op::UMod),
    // Shifts
    ("Shl", Op::ShiftLeftLogical),
    ("ShrU", Op::ShiftRightLogical),
    ("ShrS", Op::ShiftRightArithmetic),
    // Bitwise
    ("BitAnd", Op::BitwiseAnd),
    ("BitOr", Op::BitwiseOr),
    ("BitXor", Op::BitwiseXor),
    // Integer comparisons
    ("Eq", Op::IEqual),
    ("Ne", Op::INotEqual),
    ("SLt", Op::SLessThan),
    ("SLe", Op::SLessThanEqual),
    ("SGt", Op::SGreaterThan),
    ("SGe", Op::SGreaterThanEqual),
    ("ULt", Op::ULessThan),
    ("ULe", Op::ULessThanEqual),
    ("UGt", Op::UGreaterThan),
    ("UGe", Op::UGreaterThanEqual),
    // Logical
    ("LogAnd", Op::LogicalAnd),
    ("LogOr", Op::LogicalOr),
    ("LogEq", Op::LogicalEqual),
    ("LogNe", Op::LogicalNotEqual),
];

/// Unary integer operations.
const UNARY_OPS: &[(&str, Op)] = &[
    ("Neg", Op::SNegate),
    ("BitNot", Op::Not),
    ("BitReverse", Op::BitReverse),
    ("LogNot", Op::LogicalNot),
    ("BitCount", Op::BitCount),
    ("CopyObject", Op::CopyObject),
    // Typed Copy variants all emit CopyObject
    ("CopyI", Op::CopyObject),
    ("CopyF", Op::CopyObject),
    ("CopyB", Op::CopyObject),
    ("Any", Op::Any),
    ("All", Op::All),
];

/// All Select/Gamma/If variants (typed and untyped) map to Op::Select.
const SELECT_VARIANTS: &[&str] = &[
    "Select", "SelectI", "SelectF", "SelectB",
    "Gamma", "GammaI", "GammaF", "GammaB",
    "If", "IfI", "IfF", "IfB",
];

/// Try to parse an arithmetic, comparison, or logical operation.
pub fn try_parse_arithmetic(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
) -> Option<Instruction> {
    // Try binary operations
    for (name, opcode) in BINARY_OPS {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some((lhs, rhs)) = parse_binary_args(rest, id_map) {
                return Some(Instruction::new(
                    *opcode,
                    Some(result_type),
                    Some(result_id),
                    vec![
                        rspirv::dr::Operand::IdRef(lhs),
                        rspirv::dr::Operand::IdRef(rhs),
                    ],
                ));
            }
        }
    }

    // Try unary operations
    for (name, opcode) in UNARY_OPS {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(operand) = parse_unary_arg(rest, id_map) {
                return Some(Instruction::new(
                    *opcode,
                    Some(result_type),
                    Some(result_id),
                    vec![rspirv::dr::Operand::IdRef(operand)],
                ));
            }
        }
    }

    // Parse all Select/Gamma/If variants (typed and untyped) → Op::Select
    for variant in SELECT_VARIANTS {
        let prefix = format!("({} ", variant);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some((cond, t, f)) = parse_ternary_args(rest, id_map) {
                return Some(Instruction::new(
                    Op::Select,
                    Some(result_type),
                    Some(result_id),
                    vec![
                        rspirv::dr::Operand::IdRef(cond),
                        rspirv::dr::Operand::IdRef(t),
                        rspirv::dr::Operand::IdRef(f),
                    ],
                ));
            }
        }
    }

    None
}
