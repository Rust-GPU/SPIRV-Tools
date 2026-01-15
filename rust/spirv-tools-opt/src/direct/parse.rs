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
    term_to_instruction_with_ext(term, result_id, result_type, id_map, type_width, None)
}

/// Convert egglog term back to instruction with extended instruction set support.
pub fn term_to_instruction_with_ext(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
    type_width: Option<u32>,
    glsl_ext_id: Option<Word>,
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
        // Dot product
        ("Dot", Op::Dot),
        // Matrix operations
        ("MatTimesScalar", Op::MatrixTimesScalar),
        ("MatTimesVec", Op::MatrixTimesVector),
        ("VecTimesMat", Op::VectorTimesMatrix),
        ("MatTimesMat", Op::MatrixTimesMatrix),
        ("OuterProduct", Op::OuterProduct),
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
        // Additional conversions
        ("SConvert", Op::SConvert),
        ("UConvert", Op::UConvert),
        ("FConvert", Op::FConvert),
        ("Bitcast", Op::Bitcast),
        ("QuantizeToF16", Op::QuantizeToF16),
        // FP predicates
        ("IsNan", Op::IsNan),
        ("IsInf", Op::IsInf),
        // Bit operations
        ("BitCount", Op::BitCount),
        // Matrix unary
        ("Transpose", Op::Transpose),
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

    // Parse VectorShuffle variants (VecShuffle2, VecShuffle3, VecShuffle4)
    for (prefix, num_indices) in [("(VecShuffle2 ", 2), ("(VecShuffle3 ", 3), ("(VecShuffle4 ", 4)] {
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

    // Parse GLSL.std.450 extended instructions (if ext ID is available)
    if let Some(ext_id) = glsl_ext_id {
        if let Some(inst) = parse_glsl_ext_instruction(term, result_id, result_type, id_map, ext_id) {
            return Some(inst);
        }
    }

    None
}

/// Parse a GLSL.std.450 extended instruction term.
fn parse_glsl_ext_instruction(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
    ext_id: Word,
) -> Option<Instruction> {
    // GLSL.std.450 unary operations
    let glsl_unary = [
        ("Sin", 13u32), ("Cos", 14), ("Tan", 15),
        ("Asin", 16), ("Acos", 17), ("Atan", 18),
        ("Sinh", 19), ("Cosh", 20), ("Tanh", 21),
        ("Asinh", 22), ("Acosh", 23), ("Atanh", 24),
        ("Exp", 27), ("Log", 28), ("Exp2", 29), ("Log2", 30),
        ("Sqrt", 31), ("InverseSqrt", 32),
        ("Determinant", 33), ("MatInverse", 34),
        ("FAbs", 4), ("SAbs", 5), ("FSign", 6), ("Sign", 7),
        ("FFloor", 8), ("FCeil", 9), ("Fract", 10),
        ("Radians", 11), ("Degrees", 12),
        ("FRound", 1), ("FTrunc", 3),
        ("Length", 66), ("Normalize", 69),
        ("FindILsb", 73), ("FindSMsb", 74), ("FindUMsb", 75),
        ("PackSnorm4x8", 54), ("PackUnorm4x8", 55),
        ("PackSnorm2x16", 56), ("PackUnorm2x16", 57),
        ("PackHalf2x16", 58), ("PackDouble2x32", 59),
        ("UnpackSnorm2x16", 60), ("UnpackUnorm2x16", 61),
        ("UnpackHalf2x16", 62), ("UnpackSnorm4x8", 63),
        ("UnpackUnorm4x8", 64), ("UnpackDouble2x32", 65),
        ("ModfStruct", 35), ("Modf", 36),
        ("FrexpStruct", 51), ("Frexp", 52),
    ];

    for (name, opcode) in &glsl_unary {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(inner) = rest.strip_suffix(')') {
                if let Some(arg) = resolve_term_to_id(inner.trim(), id_map) {
                    return Some(Instruction::new(
                        Op::ExtInst,
                        Some(result_type),
                        Some(result_id),
                        vec![
                            rspirv::dr::Operand::IdRef(ext_id),
                            rspirv::dr::Operand::LiteralExtInstInteger(*opcode),
                            rspirv::dr::Operand::IdRef(arg),
                        ],
                    ));
                }
            }
        }
    }

    // GLSL.std.450 binary operations
    let glsl_binary = [
        ("Pow", 26u32), ("Atan2", 25),
        ("FMin", 37), ("UMin", 38), ("SMin", 39),
        ("FMax", 40), ("UMax", 41), ("SMax", 42),
        ("Step", 48), ("Distance", 67), ("Cross", 68),
        ("Reflect", 71), ("Ldexp", 53),
        ("NMin", 79), ("NMax", 80),
    ];

    for (name, opcode) in &glsl_binary {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(inner) = rest.strip_suffix(')') {
                let terms = split_terms(inner);
                if terms.len() >= 2 {
                    if let (Some(a), Some(b)) = (
                        resolve_term_to_id(&terms[0], id_map),
                        resolve_term_to_id(&terms[1], id_map),
                    ) {
                        return Some(Instruction::new(
                            Op::ExtInst,
                            Some(result_type),
                            Some(result_id),
                            vec![
                                rspirv::dr::Operand::IdRef(ext_id),
                                rspirv::dr::Operand::LiteralExtInstInteger(*opcode),
                                rspirv::dr::Operand::IdRef(a),
                                rspirv::dr::Operand::IdRef(b),
                            ],
                        ));
                    }
                }
            }
        }
    }

    // GLSL.std.450 ternary operations
    let glsl_ternary = [
        ("FClamp", 43u32), ("UClamp", 44), ("SClamp", 45),
        ("FMix", 46), ("SmoothStep", 49), ("Fma", 50),
        ("FaceForward", 70), ("Refract", 72),
        ("NClamp", 81),
    ];

    for (name, opcode) in &glsl_ternary {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(inner) = rest.strip_suffix(')') {
                let terms = split_terms(inner);
                if terms.len() >= 3 {
                    if let (Some(a), Some(b), Some(c)) = (
                        resolve_term_to_id(&terms[0], id_map),
                        resolve_term_to_id(&terms[1], id_map),
                        resolve_term_to_id(&terms[2], id_map),
                    ) {
                        return Some(Instruction::new(
                            Op::ExtInst,
                            Some(result_type),
                            Some(result_id),
                            vec![
                                rspirv::dr::Operand::IdRef(ext_id),
                                rspirv::dr::Operand::LiteralExtInstInteger(*opcode),
                                rspirv::dr::Operand::IdRef(a),
                                rspirv::dr::Operand::IdRef(b),
                                rspirv::dr::Operand::IdRef(c),
                            ],
                        ));
                    }
                }
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
