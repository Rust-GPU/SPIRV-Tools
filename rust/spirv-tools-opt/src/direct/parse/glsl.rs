//! Parsing for GLSL.std.450 extended instructions.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

use super::util::{resolve_term_to_id, split_terms};

/// GLSL.std.450 unary operations with their opcode numbers.
const GLSL_UNARY: &[(&str, u32)] = &[
    // Trigonometric
    ("Sin", 13),
    ("Cos", 14),
    ("Tan", 15),
    ("Asin", 16),
    ("Acos", 17),
    ("Atan", 18),
    ("Sinh", 19),
    ("Cosh", 20),
    ("Tanh", 21),
    ("Asinh", 22),
    ("Acosh", 23),
    ("Atanh", 24),
    // Exponential
    ("Exp", 27),
    ("Log", 28),
    ("Exp2", 29),
    ("Log2", 30),
    ("Sqrt", 31),
    ("InverseSqrt", 32),
    // Matrix
    ("Determinant", 33),
    ("MatInverse", 34),
    // Common
    ("FAbs", 4),
    ("SAbs", 5),
    ("FSign", 6),
    ("Sign", 7),
    ("FFloor", 8),
    ("FCeil", 9),
    ("Fract", 10),
    ("Radians", 11),
    ("Degrees", 12),
    ("FRound", 1),
    ("FTrunc", 3),
    // Geometry
    ("Length", 66),
    ("Normalize", 69),
    // Integer
    ("FindILsb", 73),
    ("FindSMsb", 74),
    ("FindUMsb", 75),
    // Pack/Unpack
    ("PackSnorm4x8", 54),
    ("PackUnorm4x8", 55),
    ("PackSnorm2x16", 56),
    ("PackUnorm2x16", 57),
    ("PackHalf2x16", 58),
    ("PackDouble2x32", 59),
    ("UnpackSnorm2x16", 60),
    ("UnpackUnorm2x16", 61),
    ("UnpackHalf2x16", 62),
    ("UnpackSnorm4x8", 63),
    ("UnpackUnorm4x8", 64),
    ("UnpackDouble2x32", 65),
    // Modf/Frexp
    ("ModfStruct", 35),
    ("Modf", 36),
    ("FrexpStruct", 51),
    ("Frexp", 52),
];

/// GLSL.std.450 binary operations with their opcode numbers.
const GLSL_BINARY: &[(&str, u32)] = &[
    ("Pow", 26),
    ("Atan2", 25),
    ("FMin", 37),
    ("UMin", 38),
    ("SMin", 39),
    ("FMax", 40),
    ("UMax", 41),
    ("SMax", 42),
    ("Step", 48),
    ("Distance", 67),
    ("Cross", 68),
    ("Reflect", 71),
    ("Ldexp", 53),
    ("NMin", 79),
    ("NMax", 80),
];

/// GLSL.std.450 ternary operations with their opcode numbers.
const GLSL_TERNARY: &[(&str, u32)] = &[
    ("FClamp", 43),
    ("UClamp", 44),
    ("SClamp", 45),
    ("FMix", 46),
    ("SmoothStep", 49),
    ("Fma", 50),
    ("FaceForward", 70),
    ("Refract", 72),
    ("NClamp", 81),
];

/// Try to parse a GLSL.std.450 extended instruction.
pub fn try_parse_glsl(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
    ext_id: Word,
) -> Option<Instruction> {
    // Try unary GLSL operations
    for (name, opcode) in GLSL_UNARY {
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

    // Try binary GLSL operations
    for (name, opcode) in GLSL_BINARY {
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

    // Try ternary GLSL operations
    for (name, opcode) in GLSL_TERNARY {
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
