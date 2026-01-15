//! Parsing for floating-point operations.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

use super::util::{parse_binary_args, parse_unary_arg};

/// Binary floating-point operations.
const FP_BINARY_OPS: &[(&str, Op)] = &[
    // Arithmetic
    ("FAdd", Op::FAdd),
    ("FSub", Op::FSub),
    ("FMul", Op::FMul),
    ("FDiv", Op::FDiv),
    ("FRem", Op::FRem),
    ("FMod", Op::FMod),
    // Ordered comparisons
    ("FOrdEq", Op::FOrdEqual),
    ("FOrdNe", Op::FOrdNotEqual),
    ("FOrdLt", Op::FOrdLessThan),
    ("FOrdLe", Op::FOrdLessThanEqual),
    ("FOrdGt", Op::FOrdGreaterThan),
    ("FOrdGe", Op::FOrdGreaterThanEqual),
    // Unordered comparisons
    ("FUnordEq", Op::FUnordEqual),
    ("FUnordNe", Op::FUnordNotEqual),
    ("FUnordLt", Op::FUnordLessThan),
    ("FUnordLe", Op::FUnordLessThanEqual),
    ("FUnordGt", Op::FUnordGreaterThan),
    ("FUnordGe", Op::FUnordGreaterThanEqual),
    // Dot product
    ("Dot", Op::Dot),
];

/// Unary floating-point operations.
const FP_UNARY_OPS: &[(&str, Op)] = &[
    ("FNeg", Op::FNegate),
    ("IsNan", Op::IsNan),
    ("IsInf", Op::IsInf),
    ("QuantizeToF16", Op::QuantizeToF16),
];

/// Conversion operations.
const CONVERSION_OPS: &[(&str, Op)] = &[
    ("ConvertFToU", Op::ConvertFToU),
    ("ConvertFToS", Op::ConvertFToS),
    ("ConvertSToF", Op::ConvertSToF),
    ("ConvertUToF", Op::ConvertUToF),
    ("SConvert", Op::SConvert),
    ("UConvert", Op::UConvert),
    ("FConvert", Op::FConvert),
    ("Bitcast", Op::Bitcast),
];

/// Derivative operations (fragment shader).
const DERIVATIVE_OPS: &[(&str, Op)] = &[
    ("DPdx", Op::DPdx),
    ("DPdy", Op::DPdy),
    ("Fwidth", Op::Fwidth),
    ("DPdxFine", Op::DPdxFine),
    ("DPdyFine", Op::DPdyFine),
    ("FwidthFine", Op::FwidthFine),
    ("DPdxCoarse", Op::DPdxCoarse),
    ("DPdyCoarse", Op::DPdyCoarse),
    ("FwidthCoarse", Op::FwidthCoarse),
];

/// Try to parse a floating-point operation.
pub fn try_parse_floating(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
) -> Option<Instruction> {
    // Try binary FP operations
    for (name, opcode) in FP_BINARY_OPS {
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

    // Try unary FP operations
    for (name, opcode) in FP_UNARY_OPS {
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

    // Try conversion operations
    for (name, opcode) in CONVERSION_OPS {
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

    // Try derivative operations
    for (name, opcode) in DERIVATIVE_OPS {
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

    None
}
