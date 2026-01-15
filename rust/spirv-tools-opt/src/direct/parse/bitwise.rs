//! Parsing for bitfield operations.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

use super::util::{parse_ternary_args, resolve_term_to_id, split_terms};

/// Try to parse a bitfield operation (BitFieldInsert, BitFieldSExtract, BitFieldUExtract).
pub fn try_parse_bitfield(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
) -> Option<Instruction> {
    // Parse BitFieldSExtract (base, offset, count) -> OpBitFieldSExtract
    if let Some(rest) = term.strip_prefix("(BitFieldSExtract ") {
        if let Some((base, offset, count)) = parse_ternary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::BitFieldSExtract,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(base),
                    rspirv::dr::Operand::IdRef(offset),
                    rspirv::dr::Operand::IdRef(count),
                ],
            ));
        }
    }

    // Parse BitFieldUExtract (base, offset, count) -> OpBitFieldUExtract
    if let Some(rest) = term.strip_prefix("(BitFieldUExtract ") {
        if let Some((base, offset, count)) = parse_ternary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::BitFieldUExtract,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(base),
                    rspirv::dr::Operand::IdRef(offset),
                    rspirv::dr::Operand::IdRef(count),
                ],
            ));
        }
    }

    // Parse BitFieldInsert (base, insert, offset, count) -> OpBitFieldInsert
    if let Some(rest) = term.strip_prefix("(BitFieldInsert ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if terms.len() >= 4 {
                if let (Some(base), Some(insert), Some(offset), Some(count)) = (
                    resolve_term_to_id(&terms[0], id_map),
                    resolve_term_to_id(&terms[1], id_map),
                    resolve_term_to_id(&terms[2], id_map),
                    resolve_term_to_id(&terms[3], id_map),
                ) {
                    return Some(Instruction::new(
                        Op::BitFieldInsert,
                        Some(result_type),
                        Some(result_id),
                        vec![
                            rspirv::dr::Operand::IdRef(base),
                            rspirv::dr::Operand::IdRef(insert),
                            rspirv::dr::Operand::IdRef(offset),
                            rspirv::dr::Operand::IdRef(count),
                        ],
                    ));
                }
            }
        }
    }

    None
}
