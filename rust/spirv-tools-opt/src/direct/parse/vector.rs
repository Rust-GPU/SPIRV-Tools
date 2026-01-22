//! Parsing for vector and composite operations.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

use super::util::{
    parse_binary_args, parse_expr_list, parse_ternary_args, resolve_term_to_id, split_terms,
};

/// Matrix binary operations.
const MATRIX_BINARY_OPS: &[(&str, Op)] = &[
    ("MatTimesScalar", Op::MatrixTimesScalar),
    ("MatTimesVec", Op::MatrixTimesVector),
    ("VecTimesMat", Op::VectorTimesMatrix),
    ("MatTimesMat", Op::MatrixTimesMatrix),
    ("OuterProduct", Op::OuterProduct),
    ("VectorExtractDynamic", Op::VectorExtractDynamic),
];

/// Matrix unary operations.
const MATRIX_UNARY_OPS: &[(&str, Op)] = &[("Transpose", Op::Transpose)];

/// Try to parse a vector or composite operation.
pub fn try_parse_vector(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
) -> Option<Instruction> {
    // Matrix binary operations
    for (name, opcode) in MATRIX_BINARY_OPS {
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

    // Matrix unary operations
    for (name, opcode) in MATRIX_UNARY_OPS {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(inner) = rest.strip_suffix(')') {
                if let Some(operand) = resolve_term_to_id(inner.trim(), id_map) {
                    return Some(Instruction::new(
                        *opcode,
                        Some(result_type),
                        Some(result_id),
                        vec![rspirv::dr::Operand::IdRef(operand)],
                    ));
                }
            }
        }
    }

    // VectorInsertDynamic (ternary: vector, component, index)
    if let Some(rest) = term.strip_prefix("(VectorInsertDynamic ") {
        if let Some((vec, component, idx)) = parse_ternary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::VectorInsertDynamic,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(vec),
                    rspirv::dr::Operand::IdRef(component),
                    rspirv::dr::Operand::IdRef(idx),
                ],
            ));
        }
    }

    // CompositeExtract (composite, index as literal)
    if let Some(rest) = term.strip_prefix("(CompositeExtract ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if terms.len() >= 2 {
                if let Some(composite_id) = resolve_term_to_id(&terms[0], id_map) {
                    if let Ok(index) = terms[1].parse::<u32>() {
                        return Some(Instruction::new(
                            Op::CompositeExtract,
                            Some(result_type),
                            Some(result_id),
                            vec![
                                rspirv::dr::Operand::IdRef(composite_id),
                                rspirv::dr::Operand::LiteralBit32(index),
                            ],
                        ));
                    }
                }
            }
        }
    }

    // VecExtract (same as CompositeExtract for vectors)
    if let Some(rest) = term.strip_prefix("(VecExtract ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if terms.len() >= 2 {
                if let Some(vec_id) = resolve_term_to_id(&terms[0], id_map) {
                    if let Ok(index) = terms[1].parse::<u32>() {
                        return Some(Instruction::new(
                            Op::CompositeExtract,
                            Some(result_type),
                            Some(result_id),
                            vec![
                                rspirv::dr::Operand::IdRef(vec_id),
                                rspirv::dr::Operand::LiteralBit32(index),
                            ],
                        ));
                    }
                }
            }
        }
    }

    // CompositeInsert (composite, object, index as literal)
    if let Some(rest) = term.strip_prefix("(CompositeInsert ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if terms.len() >= 3 {
                if let (Some(composite_id), Some(object_id)) = (
                    resolve_term_to_id(&terms[0], id_map),
                    resolve_term_to_id(&terms[1], id_map),
                ) {
                    if let Ok(index) = terms[2].parse::<u32>() {
                        return Some(Instruction::new(
                            Op::CompositeInsert,
                            Some(result_type),
                            Some(result_id),
                            vec![
                                rspirv::dr::Operand::IdRef(object_id),
                                rspirv::dr::Operand::IdRef(composite_id),
                                rspirv::dr::Operand::LiteralBit32(index),
                            ],
                        ));
                    }
                }
            }
        }
    }

    // VecInsert (same as CompositeInsert for vectors)
    if let Some(rest) = term.strip_prefix("(VecInsert ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if terms.len() >= 3 {
                if let (Some(vec_id), Some(scalar_id)) = (
                    resolve_term_to_id(&terms[0], id_map),
                    resolve_term_to_id(&terms[1], id_map),
                ) {
                    if let Ok(index) = terms[2].parse::<u32>() {
                        return Some(Instruction::new(
                            Op::CompositeInsert,
                            Some(result_type),
                            Some(result_id),
                            vec![
                                rspirv::dr::Operand::IdRef(scalar_id),
                                rspirv::dr::Operand::IdRef(vec_id),
                                rspirv::dr::Operand::LiteralBit32(index),
                            ],
                        ));
                    }
                }
            }
        }
    }

    // CompositeConstruct (ECons/ENil list of components)
    if let Some(rest) = term.strip_prefix("(CompositeConstruct ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let components = parse_expr_list(inner.trim(), id_map);
            if !components.is_empty() {
                let operands: Vec<rspirv::dr::Operand> = components
                    .into_iter()
                    .map(rspirv::dr::Operand::IdRef)
                    .collect();
                return Some(Instruction::new(
                    Op::CompositeConstruct,
                    Some(result_type),
                    Some(result_id),
                    operands,
                ));
            }
        }
    }

    // Vec2, Vec3, Vec4 constructors -> CompositeConstruct
    if let Some(rest) = term.strip_prefix("(Vec2 ") {
        if let Some((a, b)) = parse_binary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::CompositeConstruct,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
            ));
        }
    }

    if let Some(rest) = term.strip_prefix("(Vec3 ") {
        if let Some((a, b, c)) = parse_ternary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::CompositeConstruct,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(a),
                    rspirv::dr::Operand::IdRef(b),
                    rspirv::dr::Operand::IdRef(c),
                ],
            ));
        }
    }

    if let Some(rest) = term.strip_prefix("(Vec4 ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if terms.len() >= 4 {
                if let (Some(a), Some(b), Some(c), Some(d)) = (
                    resolve_term_to_id(&terms[0], id_map),
                    resolve_term_to_id(&terms[1], id_map),
                    resolve_term_to_id(&terms[2], id_map),
                    resolve_term_to_id(&terms[3], id_map),
                ) {
                    return Some(Instruction::new(
                        Op::CompositeConstruct,
                        Some(result_type),
                        Some(result_id),
                        vec![
                            rspirv::dr::Operand::IdRef(a),
                            rspirv::dr::Operand::IdRef(b),
                            rspirv::dr::Operand::IdRef(c),
                            rspirv::dr::Operand::IdRef(d),
                        ],
                    ));
                }
            }
        }
    }

    // VectorShuffle variants (VecShuffle2, VecShuffle3, VecShuffle4)
    for (prefix, num_indices) in [
        ("(VecShuffle2 ", 2),
        ("(VecShuffle3 ", 3),
        ("(VecShuffle4 ", 4),
    ] {
        if let Some(rest) = term.strip_prefix(prefix) {
            if let Some(inner) = rest.strip_suffix(')') {
                let terms = split_terms(inner);
                if terms.len() >= 2 + num_indices {
                    if let (Some(v1), Some(v2)) = (
                        resolve_term_to_id(&terms[0], id_map),
                        resolve_term_to_id(&terms[1], id_map),
                    ) {
                        let mut operands = vec![
                            rspirv::dr::Operand::IdRef(v1),
                            rspirv::dr::Operand::IdRef(v2),
                        ];
                        for i in 0..num_indices {
                            if let Ok(idx) = terms[2 + i].parse::<u32>() {
                                operands.push(rspirv::dr::Operand::LiteralBit32(idx));
                            }
                        }
                        if operands.len() == 2 + num_indices {
                            return Some(Instruction::new(
                                Op::VectorShuffle,
                                Some(result_type),
                                Some(result_id),
                                operands,
                            ));
                        }
                    }
                }
            }
        }
    }

    None
}
