//! Parsing for constants and symbol references.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

/// Try to parse a constant term (Const, Const64, FConst, BoolConst, Sym, ISym, FSym, BSym).
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

    // Parse (FConst N.N) - float constants
    if let Some(rest) = term.strip_prefix("(FConst ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<f64>() {
                // Convert back to IEEE bit pattern for the appropriate width
                let operand = if type_width == Some(64) {
                    rspirv::dr::Operand::LiteralBit64(value.to_bits())
                } else {
                    // Default to 32-bit float
                    rspirv::dr::Operand::LiteralBit32((value as f32).to_bits())
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

    // Parse (BoolConst N) - boolean constants
    if let Some(rest) = term.strip_prefix("(BoolConst ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
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
        }
    }

    // Parse typed and untyped Sym variants — all produce CopyObject
    for prefix in &["(Sym \"", "(ISym \"", "(FSym \"", "(BSym \""] {
        if let Some(rest) = term.strip_prefix(prefix) {
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
    }

    None
}

/// Inline constant kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineConstKind {
    /// 32-bit integer constant
    Int32,
    /// 64-bit integer constant
    Int64,
    /// Boolean constant
    Bool,
    /// Float constant (value stored as f64 bits reinterpreted as i64)
    Float,
}

/// Find all (Const N), (Const64 N), (BoolConst N), and (FConst N.N) subterms in an extracted term.
/// Returns a list of (kind, value) tuples. For Float, the i64 value contains the f64 bit pattern.
pub fn find_inline_constants(term: &str) -> Vec<(InlineConstKind, i64)> {
    let mut constants = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = term.chars().collect();

    while i < chars.len() {
        // Look for "(BoolConst " (must check before "(Const " since it's longer)
        if i + 11 <= chars.len() {
            let slice: String = chars[i..i + 11].iter().collect();
            if slice == "(BoolConst " {
                let start = i + 11;
                let mut end = start;
                while end < chars.len() && chars[end] != ')' {
                    end += 1;
                }
                if end < chars.len() {
                    let num_str: String = chars[start..end].iter().collect();
                    if let Ok(value) = num_str.trim().parse::<i64>() {
                        constants.push((InlineConstKind::Bool, value));
                    }
                }
                i = end;
                continue;
            }
        }
        // Look for "(FConst " (must check before "(Const " since it starts differently but let's be safe)
        if i + 8 <= chars.len() {
            let slice: String = chars[i..i + 8].iter().collect();
            if slice == "(FConst " {
                let start = i + 8;
                let mut end = start;
                while end < chars.len() && chars[end] != ')' {
                    end += 1;
                }
                if end < chars.len() {
                    let num_str: String = chars[start..end].iter().collect();
                    if let Ok(value) = num_str.trim().parse::<f64>() {
                        // Store f64 bits as i64
                        constants.push((InlineConstKind::Float, value.to_bits() as i64));
                    }
                }
                i = end;
                continue;
            }
        }
        // Look for "(Const64 " (must check before "(Const " since it's longer)
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
                        constants.push((InlineConstKind::Int64, value));
                    }
                }
                i = end;
                continue;
            }
        }
        // Look for "(Const "
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
                        constants.push((InlineConstKind::Int32, value));
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
