//! Utility functions for term parsing.

use rspirv::spirv::Word;
use std::collections::HashMap;

/// Split a term string into its component terms, respecting parentheses nesting.
pub fn split_terms(s: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            ' ' | '\t' | '\n' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    terms.push(trimmed);
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        terms.push(trimmed);
    }
    terms
}

/// Resolve a term to its corresponding SPIR-V ID.
pub fn resolve_term_to_id(term: &str, id_map: &HashMap<String, Word>) -> Option<Word> {
    let term = term.trim();

    // Handle typed and untyped Sym variants
    for prefix in &["(Sym \"", "(ISym \"", "(FSym \"", "(BSym \""] {
        if let Some(rest) = term.strip_prefix(prefix) {
            if let Some(sym_name) = rest.strip_suffix("\")") {
                return id_map.get(sym_name).copied();
            }
        }
    }

    if term.starts_with("id") {
        return id_map.get(term).copied();
    }

    // Handle inline constants - look up by const_N key
    if let Some(rest) = term.strip_prefix("(Const ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                let const_key = format!("const_{}", value);
                return id_map.get(&const_key).copied();
            }
        }
    }
    if let Some(rest) = term.strip_prefix("(Const64 ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                let const_key = format!("const64_{}", value);
                return id_map.get(&const_key).copied();
            }
        }
    }
    if let Some(rest) = term.strip_prefix("(BoolConst ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<i64>() {
                let const_key = format!("boolconst_{}", value);
                return id_map.get(&const_key).copied();
            }
        }
    }
    // Handle FConst - look up by fconst_BITS key
    if let Some(rest) = term.strip_prefix("(FConst ") {
        if let Some(num_str) = rest.strip_suffix(')') {
            if let Ok(value) = num_str.trim().parse::<f64>() {
                let const_key = format!("fconst_{}", value.to_bits());
                return id_map.get(&const_key).copied();
            }
        }
    }

    // Handle bridge constructors as transparent wrappers
    for prefix in &[
        "(IntToExpr ",
        "(FloatToExpr ",
        "(BoolToExpr ",
        "(ExprToInt ",
        "(ExprToFloat ",
        "(ExprToBool ",
    ] {
        if let Some(rest) = term.strip_prefix(prefix) {
            if let Some(inner) = rest.strip_suffix(')') {
                return resolve_term_to_id(inner.trim(), id_map);
            }
        }
    }

    None
}

/// Parse binary operation arguments from a term suffix.
pub fn parse_binary_args(rest: &str, id_map: &HashMap<String, Word>) -> Option<(Word, Word)> {
    let terms = split_terms(rest.strip_suffix(')')?);
    if terms.len() >= 2 {
        let lhs_id = resolve_term_to_id(&terms[0], id_map)?;
        let rhs_id = resolve_term_to_id(&terms[1], id_map)?;
        Some((lhs_id, rhs_id))
    } else {
        None
    }
}

/// Parse a unary operation argument from a term suffix.
pub fn parse_unary_arg(rest: &str, id_map: &HashMap<String, Word>) -> Option<Word> {
    let term = rest.strip_suffix(')')?;
    resolve_term_to_id(term.trim(), id_map)
}

/// Parse ternary operation arguments from a term suffix.
pub fn parse_ternary_args(
    rest: &str,
    id_map: &HashMap<String, Word>,
) -> Option<(Word, Word, Word)> {
    let terms = split_terms(rest.strip_suffix(')')?);
    if terms.len() >= 3 {
        let a = resolve_term_to_id(&terms[0], id_map)?;
        let b = resolve_term_to_id(&terms[1], id_map)?;
        let c = resolve_term_to_id(&terms[2], id_map)?;
        Some((a, b, c))
    } else {
        None
    }
}

/// Parse an ECons/ENil expression list into a vector of IDs.
pub fn parse_expr_list(term: &str, id_map: &HashMap<String, Word>) -> Vec<Word> {
    let term = term.trim();
    if term == "(ENil)" {
        return Vec::new();
    }
    if let Some(rest) = term.strip_prefix("(ECons ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if terms.len() >= 2 {
                let mut result = Vec::new();
                if let Some(id) = resolve_term_to_id(&terms[0], id_map) {
                    result.push(id);
                }
                result.extend(parse_expr_list(&terms[1], id_map));
                return result;
            }
        }
    }
    Vec::new()
}
