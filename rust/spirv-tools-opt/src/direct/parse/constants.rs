//! Parsing for constants and symbol references.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

/// Try to parse a constant term (Const, Const64, or Sym).
pub fn try_parse_constant(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
    type_width: Option<u32>,
) -> Option<Instruction> {
    // Parse (Const64 N)
    if let Some(rest) = term.strip_prefix("(Const64 ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                return Some(Instruction::new(
                    Op::Constant,
                    Some(result_type),
                    Some(result_id),
                    vec![rspirv::dr::Operand::LiteralBit64(value as u64)],
                ));
            }
        }
    }

    // Parse (Const N)
    if let Some(rest) = term.strip_prefix("(Const ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                // For boolean types (width 1), emit ConstantTrue/ConstantFalse
                if type_width == Some(1) {
                    if value == 0 {
                        return Some(Instruction::new(
                            Op::ConstantFalse,
                            Some(result_type),
                            Some(result_id),
                            vec![],
                        ));
                    } else {
                        return Some(Instruction::new(
                            Op::ConstantTrue,
                            Some(result_type),
                            Some(result_id),
                            vec![],
                        ));
                    }
                }
                // For integer types, emit Op::Constant with appropriate width
                let operand = if type_width == Some(64) {
                    rspirv::dr::Operand::LiteralBit64(value as u64)
                } else {
                    rspirv::dr::Operand::LiteralBit32(value as u32)
                };
                return Some(Instruction::new(
                    Op::Constant,
                    Some(result_type),
                    Some(result_id),
                    vec![operand],
                ));
            }
        }
    }

    // Parse (Sym "idN")
    if let Some(rest) = term.strip_prefix("(Sym \"") {
        if let Some(sym_name) = rest.strip_suffix("\")") {
            if let Some(&ref_id) = id_map.get(sym_name) {
                return Some(Instruction::new(
                    Op::CopyObject,
                    Some(result_type),
                    Some(result_id),
                    vec![rspirv::dr::Operand::IdRef(ref_id)],
                ));
            }
        }
    }

    None
}

/// Find all (Const N) and (Const64 N) subterms in an extracted term.
/// Returns a list of (is_64bit, value) tuples.
pub fn find_inline_constants(term: &str) -> Vec<(bool, i64)> {
    let mut constants = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = term.chars().collect();

    while i < chars.len() {
        // Look for "(Const " or "(Const64 "
        if i + 7 <= chars.len() {
            let slice: String = chars[i..i + 7].iter().collect();
            if slice == "(Const " {
                // Find the closing paren
                let start = i + 7;
                let mut end = start;
                while end < chars.len() && chars[end] != ')' {
                    end += 1;
                }
                if end < chars.len() {
                    let num_str: String = chars[start..end].iter().collect();
                    if let Ok(value) = num_str.trim().parse::<i64>() {
                        constants.push((false, value));
                    }
                }
                i = end;
                continue;
            }
        }
        if i + 9 <= chars.len() {
            let slice: String = chars[i..i + 9].iter().collect();
            if slice == "(Const64 " {
                let start = i + 9;
                let mut end = start;
                while end < chars.len() && chars[end] != ')' {
                    end += 1;
                }
                if end < chars.len() {
                    let num_str: String = chars[start..end].iter().collect();
                    if let Ok(value) = num_str.trim().parse::<i64>() {
                        constants.push((true, value));
                    }
                }
                i = end;
                continue;
            }
        }
        i += 1;
    }

    constants
}
