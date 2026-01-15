//! Term parsing and instruction reconstruction from egglog results.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

/// Convert egglog term back to instruction.
/// `type_width` is the bit width of the result type (1 for bool, 32/64 for int).
pub fn term_to_instruction(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
    type_width: Option<u32>,
) -> Option<Instruction> {
    let term = term.trim();

    // Parse (Const N) or (Const64 N)
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

    // Parse binary operations
    let binary_ops = [
        ("Add", Op::IAdd),
        ("Sub", Op::ISub),
        ("Mul", Op::IMul),
        ("SDiv", Op::SDiv),
        ("UDiv", Op::UDiv),
        ("SRem", Op::SRem),
        ("SMod", Op::SMod),
        ("UMod", Op::UMod),
        ("Shl", Op::ShiftLeftLogical),
        ("ShrU", Op::ShiftRightLogical),
        ("ShrS", Op::ShiftRightArithmetic),
        ("BitAnd", Op::BitwiseAnd),
        ("BitOr", Op::BitwiseOr),
        ("BitXor", Op::BitwiseXor),
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
        ("LogAnd", Op::LogicalAnd),
        ("LogOr", Op::LogicalOr),
        ("LogEq", Op::LogicalEqual),
        ("LogNe", Op::LogicalNotEqual),
        // Floating-point operations
        ("FAdd", Op::FAdd),
        ("FSub", Op::FSub),
        ("FMul", Op::FMul),
        ("FDiv", Op::FDiv),
        ("FRem", Op::FRem),
        ("FMod", Op::FMod),
        // Floating-point comparisons (ordered)
        ("FOrdEq", Op::FOrdEqual),
        ("FOrdNe", Op::FOrdNotEqual),
        ("FOrdLt", Op::FOrdLessThan),
        ("FOrdLe", Op::FOrdLessThanEqual),
        ("FOrdGt", Op::FOrdGreaterThan),
        ("FOrdGe", Op::FOrdGreaterThanEqual),
        // Floating-point comparisons (unordered)
        ("FUnordEq", Op::FUnordEqual),
        ("FUnordNe", Op::FUnordNotEqual),
        ("FUnordLt", Op::FUnordLessThan),
        ("FUnordLe", Op::FUnordLessThanEqual),
        ("FUnordGt", Op::FUnordGreaterThan),
        ("FUnordGe", Op::FUnordGreaterThanEqual),
        // Vector operations
        ("VectorExtractDynamic", Op::VectorExtractDynamic),
    ];

    for (name, opcode) in &binary_ops {
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

    // Parse unary operations
    let unary_ops = [
        ("Neg", Op::SNegate),
        ("BitNot", Op::Not),
        ("BitReverse", Op::BitReverse),
        ("LogNot", Op::LogicalNot),
        // Floating-point unary
        ("FNeg", Op::FNegate),
        // Conversion operations
        ("ConvertFToU", Op::ConvertFToU),
        ("ConvertFToS", Op::ConvertFToS),
        ("ConvertSToF", Op::ConvertSToF),
        ("ConvertUToF", Op::ConvertUToF),
        // Derivative operations (fragment shader)
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

    for (name, opcode) in &unary_ops {
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

    // Parse Select
    if let Some(rest) = term.strip_prefix("(Select ") {
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

    // Parse VectorInsertDynamic (ternary: vector, component, index)
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

    // Parse CompositeExtract (binary: composite, index as literal)
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

    // Parse CompositeInsert (ternary: composite, object, index as literal)
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

    // Parse CompositeConstruct (ECons/ENil list of components)
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

    None
}

/// Parse an ECons/ENil expression list into a vector of IDs.
fn parse_expr_list(term: &str, id_map: &HashMap<String, Word>) -> Vec<Word> {
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

fn parse_binary_args(rest: &str, id_map: &HashMap<String, Word>) -> Option<(Word, Word)> {
    let terms = split_terms(rest.strip_suffix(')')?);
    if terms.len() >= 2 {
        let lhs_id = resolve_term_to_id(&terms[0], id_map)?;
        let rhs_id = resolve_term_to_id(&terms[1], id_map)?;
        Some((lhs_id, rhs_id))
    } else {
        None
    }
}

fn parse_unary_arg(rest: &str, id_map: &HashMap<String, Word>) -> Option<Word> {
    let term = rest.strip_suffix(')')?;
    resolve_term_to_id(term.trim(), id_map)
}

fn parse_ternary_args(rest: &str, id_map: &HashMap<String, Word>) -> Option<(Word, Word, Word)> {
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

fn split_terms(s: &str) -> Vec<String> {
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

fn resolve_term_to_id(term: &str, id_map: &HashMap<String, Word>) -> Option<Word> {
    let term = term.trim();

    if let Some(rest) = term.strip_prefix("(Sym \"") {
        if let Some(sym_name) = rest.strip_suffix("\")") {
            return id_map.get(sym_name).copied();
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
            let slice: String = chars[i..i+7].iter().collect();
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
            let slice: String = chars[i..i+9].iter().collect();
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

/// Parse the extract result from egglog.
pub fn parse_extract_result(s: &str) -> Option<String> {
    let s = s.trim();

    if !s.starts_with("ExtractBest(") {
        return Some(s.to_string());
    }

    if let Some(nodes_start) = s.find("nodes: {") {
        let rest = &s[nodes_start + 8..];
        if let Some(nodes_end) = rest.find('}') {
            let nodes_str = &rest[..nodes_end];
            let nodes = parse_term_dag_nodes(nodes_str);
            if !nodes.is_empty() {
                return Some(nodes.last()?.clone());
            }
        }
    }

    None
}

fn parse_term_dag_nodes(nodes_str: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut depth = 0;
    let mut current = String::new();

    for c in nodes_str.chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                let node = current.trim().to_string();
                if !node.is_empty() {
                    if let Some(term) = term_dag_node_to_term(&node, &result) {
                        result.push(term);
                    }
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }

    let node = current.trim().to_string();
    if !node.is_empty() {
        if let Some(term) = term_dag_node_to_term(&node, &result) {
            result.push(term);
        }
    }

    result
}

fn term_dag_node_to_term(node: &str, previous_nodes: &[String]) -> Option<String> {
    let node = node.trim();

    if node.starts_with("Lit(Int(") && node.ends_with("))") {
        let val = &node[8..node.len() - 2];
        return Some(format!("(Const {})", val));
    }

    if node.starts_with("Lit(String(") && node.ends_with("))") {
        let sym = &node[11..node.len() - 2];
        return Some(format!("(Sym {})", sym));
    }

    if node.starts_with("App(") && node.ends_with(")") {
        let inner = &node[4..node.len() - 1];

        if let Some(quote_start) = inner.find('"') {
            if let Some(quote_end) = inner[quote_start + 1..].find('"') {
                let op_name = &inner[quote_start + 1..quote_start + 1 + quote_end];

                if let Some(bracket_start) = inner.find('[') {
                    if let Some(bracket_end) = inner.rfind(']') {
                        let indices_str = &inner[bracket_start + 1..bracket_end];
                        let indices: Vec<usize> = indices_str
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();

                        if indices.is_empty() {
                            return Some(format!("({})", op_name));
                        }

                        let children: Vec<String> = indices
                            .iter()
                            .map(|&i| {
                                if i < previous_nodes.len() {
                                    previous_nodes[i].clone()
                                } else {
                                    format!("?{}", i)
                                }
                            })
                            .collect();

                        return Some(format!("({} {})", op_name, children.join(" ")));
                    }
                }
            }
        }
    }

    None
}
