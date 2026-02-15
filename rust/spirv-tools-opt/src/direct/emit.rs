//! Unified term emission: S-expression parsing and SPIR-V instruction generation.
//!
//! Replaces the old two-path system (term_to_instruction + materialize_term)
//! with a single recursive tree-walking emitter backed by a unified ops table.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

use super::TypeClass;

// ---------------------------------------------------------------------------
// Term tree
// ---------------------------------------------------------------------------

/// A parsed S-expression term from egglog extraction output.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// Atom: bare identifier (`id5`), number (`42`, `3.14`), or quoted string content (`id5` from `"id5"`).
    Atom(String),
    /// Application: `(OpName arg1 arg2 ...)`
    App { op: String, args: Vec<Term> },
}

// ---------------------------------------------------------------------------
// S-expression parser
// ---------------------------------------------------------------------------

/// Parse an S-expression string into a Term tree.
///
/// Handles:
/// - Bare atoms: `id5`, `42`, `-3`, `3.14`
/// - Quoted strings: `"id5"` → Atom("id5")
/// - Applications: `(Add (Sym "id5") (Const 3))`
/// - Nested expressions of arbitrary depth
pub fn parse_sexpr(s: &str) -> Option<Term> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (term, rest) = parse_one(s)?;
    if !rest.trim().is_empty() {
        return None; // trailing garbage
    }
    Some(term)
}

/// Parse one term from the front of `s`, returning (term, remaining).
fn parse_one(s: &str) -> Option<(Term, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }
    let first = s.as_bytes()[0];
    match first {
        b'(' => parse_app(s),
        b'"' => parse_string_lit(s),
        _ => parse_atom(s),
    }
}

/// Parse `(OpName arg1 arg2 ...)`.
fn parse_app(s: &str) -> Option<(Term, &str)> {
    let s = s.strip_prefix('(')?.trim_start();

    // Parse the operator name (first atom before space or paren)
    let op_end = s
        .find(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .unwrap_or(s.len());
    if op_end == 0 {
        return None;
    }
    let op = s[..op_end].to_string();
    let mut rest = s[op_end..].trim_start();

    // Parse arguments until closing paren
    let mut args = Vec::new();
    loop {
        if rest.is_empty() {
            return None; // unclosed paren
        }
        if rest.starts_with(')') {
            rest = &rest[1..];
            break;
        }
        let (arg, remaining) = parse_one(rest)?;
        args.push(arg);
        rest = remaining.trim_start();
    }
    Some((Term::App { op, args }, rest))
}

/// Parse a quoted string `"..."`, returning Atom with the unquoted content.
fn parse_string_lit(s: &str) -> Option<(Term, &str)> {
    let s = s.strip_prefix('"')?;
    // Find closing quote (no escape handling needed for egglog output)
    let end = s.find('"')?;
    let content = s[..end].to_string();
    let rest = &s[end + 1..];
    Some((Term::Atom(content), rest))
}

/// Parse a bare atom (identifier, number, negative number).
fn parse_atom(s: &str) -> Option<(Term, &str)> {
    let end = s
        .find(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == '"')
        .unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    let atom = s[..end].to_string();
    Some((Term::Atom(atom), &s[end..]))
}

// ---------------------------------------------------------------------------
// Emit context
// ---------------------------------------------------------------------------

/// Bundles mutable state needed during recursive term emission.
pub struct EmitCtx<'a> {
    pub id_map: &'a mut HashMap<String, Word>,
    pub next_id: &'a mut Word,
    pub int32_type: Option<Word>,
    pub int64_type: Option<Word>,
    pub float32_type: Option<Word>,
    pub float64_type: Option<Word>,
    pub bool_type: Option<Word>,
    pub type_classes: &'a HashMap<Word, TypeClass>,
    pub glsl_ext_id: Option<Word>,
    pub type_widths: &'a HashMap<Word, u32>,
    pub id_to_type: &'a HashMap<Word, Word>,
}

// ---------------------------------------------------------------------------
// Emit pattern enum
// ---------------------------------------------------------------------------

/// How to emit a SPIR-V instruction for a given constructor.
#[derive(Debug, Clone, Copy)]
enum EmitPattern {
    /// Unary: 1 IdRef operand. (opcode, result_class, operand_class)
    Unary(Op, TypeClass, TypeClass),
    /// Binary: 2 IdRef operands. (opcode, result_class, operand_class)
    Binary(Op, TypeClass, TypeClass),
    /// Ternary: 3 IdRef operands. (opcode, result_class, operand_class)
    Ternary(Op, TypeClass, TypeClass),
    /// Quaternary: 4 IdRef operands. (opcode, result_class, operand_class)
    Quaternary(Op, TypeClass, TypeClass),
    /// GLSL unary: ExtInst with 1 operand. (glsl_opcode, result_class, operand_class)
    GlslUnary(u32, TypeClass, TypeClass),
    /// GLSL binary: ExtInst with 2 operands. (glsl_opcode, result_class, operand_class)
    GlslBinary(u32, TypeClass, TypeClass),
    /// GLSL binary with heterogeneous operands. (glsl_opcode, result_class, op1_class, op2_class)
    GlslBinaryMixed(u32, TypeClass, TypeClass, TypeClass),
    /// GLSL ternary: ExtInst with 3 operands. (glsl_opcode, result_class, operand_class)
    GlslTernary(u32, TypeClass, TypeClass),
    /// Select/Gamma/If: cond is Bool, arms match the given TypeClass.
    Select(TypeClass),
    /// Bridge constructor: transparent wrapper, recurse into child.
    Bridge,
    /// Load: emit OpLoad from first arg (pointer), ignore second arg (memory token).
    Load,
    /// Extract with literal index: emit OpCompositeExtract(composite, literal_index).
    ExtractLiteral(Op),
    /// Insert with swapped operands + literal index: emit Op(object, composite, literal_index).
    /// Egglog order: (composite, object, index). SPIR-V order: (object, composite, index).
    InsertSwapped(Op),
    /// CompositeConstruct from N positional args. (count)
    CompositeN(u32),
    /// CompositeConstruct from ECons/ENil list.
    CompositeList,
    /// VectorShuffle with N literal indices after 2 vector operands. (num_indices)
    ShuffleN(u32),
    /// Image op with operand mask: emit Op(sampled_image, coord, mask, extra_operand).
    ImageWithMask(Op, u32),
}

// ---------------------------------------------------------------------------
// Unified ops table
// ---------------------------------------------------------------------------

/// Unified ops table mapping constructor names to emission patterns.
/// This single table replaces all the duplicated tables across parse submodules
/// and materialize_term.
const OPS_TABLE: &[(&str, EmitPattern)] = &[
    // ===== Integer arithmetic (IntExpr) =====
    (
        "Add",
        EmitPattern::Binary(Op::IAdd, TypeClass::Int, TypeClass::Int),
    ),
    (
        "Sub",
        EmitPattern::Binary(Op::ISub, TypeClass::Int, TypeClass::Int),
    ),
    (
        "Mul",
        EmitPattern::Binary(Op::IMul, TypeClass::Int, TypeClass::Int),
    ),
    (
        "SDiv",
        EmitPattern::Binary(Op::SDiv, TypeClass::Int, TypeClass::Int),
    ),
    (
        "UDiv",
        EmitPattern::Binary(Op::UDiv, TypeClass::Int, TypeClass::Int),
    ),
    (
        "SRem",
        EmitPattern::Binary(Op::SRem, TypeClass::Int, TypeClass::Int),
    ),
    (
        "SMod",
        EmitPattern::Binary(Op::SMod, TypeClass::Int, TypeClass::Int),
    ),
    (
        "UMod",
        EmitPattern::Binary(Op::UMod, TypeClass::Int, TypeClass::Int),
    ),
    // Shifts
    (
        "Shl",
        EmitPattern::Binary(Op::ShiftLeftLogical, TypeClass::Int, TypeClass::Int),
    ),
    (
        "ShrU",
        EmitPattern::Binary(Op::ShiftRightLogical, TypeClass::Int, TypeClass::Int),
    ),
    (
        "ShrS",
        EmitPattern::Binary(Op::ShiftRightArithmetic, TypeClass::Int, TypeClass::Int),
    ),
    // Bitwise
    (
        "BitAnd",
        EmitPattern::Binary(Op::BitwiseAnd, TypeClass::Int, TypeClass::Int),
    ),
    (
        "BitOr",
        EmitPattern::Binary(Op::BitwiseOr, TypeClass::Int, TypeClass::Int),
    ),
    (
        "BitXor",
        EmitPattern::Binary(Op::BitwiseXor, TypeClass::Int, TypeClass::Int),
    ),
    // Integer unary
    (
        "Neg",
        EmitPattern::Unary(Op::SNegate, TypeClass::Int, TypeClass::Int),
    ),
    (
        "BitNot",
        EmitPattern::Unary(Op::Not, TypeClass::Int, TypeClass::Int),
    ),
    (
        "BitReverse",
        EmitPattern::Unary(Op::BitReverse, TypeClass::Int, TypeClass::Int),
    ),
    (
        "BitCount",
        EmitPattern::Unary(Op::BitCount, TypeClass::Int, TypeClass::Int),
    ),
    // Integer comparisons: result Bool, operands Int
    (
        "Eq",
        EmitPattern::Binary(Op::IEqual, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "Ne",
        EmitPattern::Binary(Op::INotEqual, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "SLt",
        EmitPattern::Binary(Op::SLessThan, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "SLe",
        EmitPattern::Binary(Op::SLessThanEqual, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "SGt",
        EmitPattern::Binary(Op::SGreaterThan, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "SGe",
        EmitPattern::Binary(Op::SGreaterThanEqual, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "ULt",
        EmitPattern::Binary(Op::ULessThan, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "ULe",
        EmitPattern::Binary(Op::ULessThanEqual, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "UGt",
        EmitPattern::Binary(Op::UGreaterThan, TypeClass::Bool, TypeClass::Int),
    ),
    (
        "UGe",
        EmitPattern::Binary(Op::UGreaterThanEqual, TypeClass::Bool, TypeClass::Int),
    ),
    // ===== Logical (BoolExpr) =====
    (
        "LogAnd",
        EmitPattern::Binary(Op::LogicalAnd, TypeClass::Bool, TypeClass::Bool),
    ),
    (
        "LogOr",
        EmitPattern::Binary(Op::LogicalOr, TypeClass::Bool, TypeClass::Bool),
    ),
    (
        "LogEq",
        EmitPattern::Binary(Op::LogicalEqual, TypeClass::Bool, TypeClass::Bool),
    ),
    (
        "LogNe",
        EmitPattern::Binary(Op::LogicalNotEqual, TypeClass::Bool, TypeClass::Bool),
    ),
    (
        "LogNot",
        EmitPattern::Unary(Op::LogicalNot, TypeClass::Bool, TypeClass::Bool),
    ),
    (
        "Any",
        EmitPattern::Unary(Op::Any, TypeClass::Bool, TypeClass::Bool),
    ),
    (
        "All",
        EmitPattern::Unary(Op::All, TypeClass::Bool, TypeClass::Bool),
    ),
    // ===== Floating-point arithmetic (FloatExpr) =====
    (
        "FAdd",
        EmitPattern::Binary(Op::FAdd, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FSub",
        EmitPattern::Binary(Op::FSub, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FMul",
        EmitPattern::Binary(Op::FMul, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FDiv",
        EmitPattern::Binary(Op::FDiv, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FRem",
        EmitPattern::Binary(Op::FRem, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FMod",
        EmitPattern::Binary(Op::FMod, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FNeg",
        EmitPattern::Unary(Op::FNegate, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Dot",
        EmitPattern::Binary(Op::Dot, TypeClass::Float, TypeClass::Float),
    ),
    // Float comparisons (ordered): result Bool, operands Float
    (
        "FOrdEq",
        EmitPattern::Binary(Op::FOrdEqual, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FOrdNe",
        EmitPattern::Binary(Op::FOrdNotEqual, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FOrdLt",
        EmitPattern::Binary(Op::FOrdLessThan, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FOrdLe",
        EmitPattern::Binary(Op::FOrdLessThanEqual, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FOrdGt",
        EmitPattern::Binary(Op::FOrdGreaterThan, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FOrdGe",
        EmitPattern::Binary(Op::FOrdGreaterThanEqual, TypeClass::Bool, TypeClass::Float),
    ),
    // Float comparisons (unordered)
    (
        "FUnordEq",
        EmitPattern::Binary(Op::FUnordEqual, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FUnordNe",
        EmitPattern::Binary(Op::FUnordNotEqual, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FUnordLt",
        EmitPattern::Binary(Op::FUnordLessThan, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FUnordLe",
        EmitPattern::Binary(Op::FUnordLessThanEqual, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FUnordGt",
        EmitPattern::Binary(Op::FUnordGreaterThan, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "FUnordGe",
        EmitPattern::Binary(
            Op::FUnordGreaterThanEqual,
            TypeClass::Bool,
            TypeClass::Float,
        ),
    ),
    // Float queries: result Bool, operand Float
    (
        "IsNan",
        EmitPattern::Unary(Op::IsNan, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "IsInf",
        EmitPattern::Unary(Op::IsInf, TypeClass::Bool, TypeClass::Float),
    ),
    (
        "QuantizeToF16",
        EmitPattern::Unary(Op::QuantizeToF16, TypeClass::Float, TypeClass::Float),
    ),
    // ===== Conversions (cross-sort) =====
    (
        "ConvertFToU",
        EmitPattern::Unary(Op::ConvertFToU, TypeClass::Int, TypeClass::Float),
    ),
    (
        "ConvertFToS",
        EmitPattern::Unary(Op::ConvertFToS, TypeClass::Int, TypeClass::Float),
    ),
    (
        "ConvertSToF",
        EmitPattern::Unary(Op::ConvertSToF, TypeClass::Float, TypeClass::Int),
    ),
    (
        "ConvertUToF",
        EmitPattern::Unary(Op::ConvertUToF, TypeClass::Float, TypeClass::Int),
    ),
    (
        "SConvert",
        EmitPattern::Unary(Op::SConvert, TypeClass::Int, TypeClass::Int),
    ),
    (
        "UConvert",
        EmitPattern::Unary(Op::UConvert, TypeClass::Int, TypeClass::Int),
    ),
    (
        "FConvert",
        EmitPattern::Unary(Op::FConvert, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Bitcast",
        EmitPattern::Unary(Op::Bitcast, TypeClass::Other, TypeClass::Other),
    ),
    // ===== Copy variants =====
    (
        "CopyObject",
        EmitPattern::Unary(Op::CopyObject, TypeClass::Other, TypeClass::Other),
    ),
    (
        "CopyI",
        EmitPattern::Unary(Op::CopyObject, TypeClass::Int, TypeClass::Int),
    ),
    (
        "CopyF",
        EmitPattern::Unary(Op::CopyObject, TypeClass::Float, TypeClass::Float),
    ),
    (
        "CopyB",
        EmitPattern::Unary(Op::CopyObject, TypeClass::Bool, TypeClass::Bool),
    ),
    // ===== Derivative operations =====
    (
        "DPdx",
        EmitPattern::Unary(Op::DPdx, TypeClass::Float, TypeClass::Float),
    ),
    (
        "DPdy",
        EmitPattern::Unary(Op::DPdy, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Fwidth",
        EmitPattern::Unary(Op::Fwidth, TypeClass::Float, TypeClass::Float),
    ),
    (
        "DPdxFine",
        EmitPattern::Unary(Op::DPdxFine, TypeClass::Float, TypeClass::Float),
    ),
    (
        "DPdyFine",
        EmitPattern::Unary(Op::DPdyFine, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FwidthFine",
        EmitPattern::Unary(Op::FwidthFine, TypeClass::Float, TypeClass::Float),
    ),
    (
        "DPdxCoarse",
        EmitPattern::Unary(Op::DPdxCoarse, TypeClass::Float, TypeClass::Float),
    ),
    (
        "DPdyCoarse",
        EmitPattern::Unary(Op::DPdyCoarse, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FwidthCoarse",
        EmitPattern::Unary(Op::FwidthCoarse, TypeClass::Float, TypeClass::Float),
    ),
    // ===== Bitfield operations =====
    (
        "BitFieldSExtract",
        EmitPattern::Ternary(Op::BitFieldSExtract, TypeClass::Int, TypeClass::Int),
    ),
    (
        "BitFieldUExtract",
        EmitPattern::Ternary(Op::BitFieldUExtract, TypeClass::Int, TypeClass::Int),
    ),
    (
        "BitFieldInsert",
        EmitPattern::Quaternary(Op::BitFieldInsert, TypeClass::Int, TypeClass::Int),
    ),
    // ===== Matrix operations =====
    (
        "MatTimesScalar",
        EmitPattern::Binary(Op::MatrixTimesScalar, TypeClass::Other, TypeClass::Other),
    ),
    (
        "MatTimesVec",
        EmitPattern::Binary(Op::MatrixTimesVector, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecTimesMat",
        EmitPattern::Binary(Op::VectorTimesMatrix, TypeClass::Other, TypeClass::Other),
    ),
    (
        "MatTimesMat",
        EmitPattern::Binary(Op::MatrixTimesMatrix, TypeClass::Other, TypeClass::Other),
    ),
    (
        "OuterProduct",
        EmitPattern::Binary(Op::OuterProduct, TypeClass::Other, TypeClass::Other),
    ),
    (
        "Transpose",
        EmitPattern::Unary(Op::Transpose, TypeClass::Other, TypeClass::Other),
    ),
    // ===== Vector arithmetic (Expr-sort, component-wise) =====
    (
        "VecAdd",
        EmitPattern::Binary(Op::IAdd, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecSub",
        EmitPattern::Binary(Op::ISub, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecMul",
        EmitPattern::Binary(Op::IMul, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecDiv",
        EmitPattern::Binary(Op::SDiv, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecSDiv",
        EmitPattern::Binary(Op::SDiv, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecUDiv",
        EmitPattern::Binary(Op::UDiv, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecSRem",
        EmitPattern::Binary(Op::SRem, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecSMod",
        EmitPattern::Binary(Op::SMod, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecUMod",
        EmitPattern::Binary(Op::UMod, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecFAdd",
        EmitPattern::Binary(Op::FAdd, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecFSub",
        EmitPattern::Binary(Op::FSub, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecFMul",
        EmitPattern::Binary(Op::FMul, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecFDiv",
        EmitPattern::Binary(Op::FDiv, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecFRem",
        EmitPattern::Binary(Op::FRem, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecFMod",
        EmitPattern::Binary(Op::FMod, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecTimesScalar",
        EmitPattern::Binary(Op::VectorTimesScalar, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecNeg",
        EmitPattern::Unary(Op::SNegate, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VecFNeg",
        EmitPattern::Unary(Op::FNegate, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VectorExtractDynamic",
        EmitPattern::Binary(Op::VectorExtractDynamic, TypeClass::Other, TypeClass::Other),
    ),
    (
        "VectorInsertDynamic",
        EmitPattern::Ternary(Op::VectorInsertDynamic, TypeClass::Other, TypeClass::Other),
    ),
    // ===== Memory operations =====
    // AccessChainDyn is handled specially in emit_app (not via OPS_TABLE)
    // because the base pointer type differs from the result pointer type.
    // ===== Image query operations =====
    (
        "ImageQuerySize",
        EmitPattern::Unary(Op::ImageQuerySize, TypeClass::Other, TypeClass::Other),
    ),
    (
        "ImageQueryLevels",
        EmitPattern::Unary(Op::ImageQueryLevels, TypeClass::Other, TypeClass::Other),
    ),
    (
        "ImageQuerySamples",
        EmitPattern::Unary(Op::ImageQuerySamples, TypeClass::Other, TypeClass::Other),
    ),
    (
        "ImageSample",
        EmitPattern::Binary(
            Op::ImageSampleImplicitLod,
            TypeClass::Other,
            TypeClass::Other,
        ),
    ),
    (
        "ImageFetch",
        EmitPattern::Binary(Op::ImageFetch, TypeClass::Other, TypeClass::Other),
    ),
    (
        "ImageQuerySizeLod",
        EmitPattern::Binary(Op::ImageQuerySizeLod, TypeClass::Other, TypeClass::Other),
    ),
    (
        "ImageQueryLod",
        EmitPattern::Binary(Op::ImageQueryLod, TypeClass::Other, TypeClass::Other),
    ),
    // ===== Select / Gamma / If variants =====
    ("Select", EmitPattern::Select(TypeClass::Other)),
    ("Gamma", EmitPattern::Select(TypeClass::Other)),
    ("If", EmitPattern::Select(TypeClass::Other)),
    ("SelectI", EmitPattern::Select(TypeClass::Int)),
    ("GammaI", EmitPattern::Select(TypeClass::Int)),
    ("IfI", EmitPattern::Select(TypeClass::Int)),
    ("SelectF", EmitPattern::Select(TypeClass::Float)),
    ("GammaF", EmitPattern::Select(TypeClass::Float)),
    ("IfF", EmitPattern::Select(TypeClass::Float)),
    ("SelectB", EmitPattern::Select(TypeClass::Bool)),
    ("GammaB", EmitPattern::Select(TypeClass::Bool)),
    ("IfB", EmitPattern::Select(TypeClass::Bool)),
    ("VecSelect", EmitPattern::Select(TypeClass::Other)),
    // ===== Bridge constructors =====
    ("IntToExpr", EmitPattern::Bridge),
    ("FloatToExpr", EmitPattern::Bridge),
    ("BoolToExpr", EmitPattern::Bridge),
    ("ExprToInt", EmitPattern::Bridge),
    ("ExprToFloat", EmitPattern::Bridge),
    ("ExprToBool", EmitPattern::Bridge),
    // ===== GLSL.std.450 unary =====
    (
        "Sin",
        EmitPattern::GlslUnary(13, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Cos",
        EmitPattern::GlslUnary(14, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Tan",
        EmitPattern::GlslUnary(15, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Asin",
        EmitPattern::GlslUnary(16, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Acos",
        EmitPattern::GlslUnary(17, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Atan",
        EmitPattern::GlslUnary(18, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Sinh",
        EmitPattern::GlslUnary(19, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Cosh",
        EmitPattern::GlslUnary(20, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Tanh",
        EmitPattern::GlslUnary(21, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Asinh",
        EmitPattern::GlslUnary(22, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Acosh",
        EmitPattern::GlslUnary(23, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Atanh",
        EmitPattern::GlslUnary(24, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Exp",
        EmitPattern::GlslUnary(27, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Log",
        EmitPattern::GlslUnary(28, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Exp2",
        EmitPattern::GlslUnary(29, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Log2",
        EmitPattern::GlslUnary(30, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Sqrt",
        EmitPattern::GlslUnary(31, TypeClass::Float, TypeClass::Float),
    ),
    (
        "InverseSqrt",
        EmitPattern::GlslUnary(32, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Determinant",
        EmitPattern::GlslUnary(33, TypeClass::Float, TypeClass::Float),
    ),
    (
        "MatInverse",
        EmitPattern::GlslUnary(34, TypeClass::Other, TypeClass::Other),
    ),
    (
        "FAbs",
        EmitPattern::GlslUnary(4, TypeClass::Float, TypeClass::Float),
    ),
    (
        "SAbs",
        EmitPattern::GlslUnary(5, TypeClass::Int, TypeClass::Int),
    ),
    (
        "FSign",
        EmitPattern::GlslUnary(6, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Sign",
        EmitPattern::GlslUnary(7, TypeClass::Int, TypeClass::Int),
    ),
    (
        "FFloor",
        EmitPattern::GlslUnary(8, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FCeil",
        EmitPattern::GlslUnary(9, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Fract",
        EmitPattern::GlslUnary(10, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Radians",
        EmitPattern::GlslUnary(11, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Degrees",
        EmitPattern::GlslUnary(12, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FRound",
        EmitPattern::GlslUnary(1, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FTrunc",
        EmitPattern::GlslUnary(3, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Length",
        EmitPattern::GlslUnary(66, TypeClass::Float, TypeClass::Other),
    ),
    (
        "Normalize",
        EmitPattern::GlslUnary(69, TypeClass::Other, TypeClass::Other),
    ),
    (
        "FindILsb",
        EmitPattern::GlslUnary(73, TypeClass::Int, TypeClass::Int),
    ),
    (
        "FindSMsb",
        EmitPattern::GlslUnary(74, TypeClass::Int, TypeClass::Int),
    ),
    (
        "FindUMsb",
        EmitPattern::GlslUnary(75, TypeClass::Int, TypeClass::Int),
    ),
    // Pack/Unpack
    (
        "PackSnorm4x8",
        EmitPattern::GlslUnary(54, TypeClass::Int, TypeClass::Other),
    ),
    (
        "PackUnorm4x8",
        EmitPattern::GlslUnary(55, TypeClass::Int, TypeClass::Other),
    ),
    (
        "PackSnorm2x16",
        EmitPattern::GlslUnary(56, TypeClass::Int, TypeClass::Other),
    ),
    (
        "PackUnorm2x16",
        EmitPattern::GlslUnary(57, TypeClass::Int, TypeClass::Other),
    ),
    (
        "PackHalf2x16",
        EmitPattern::GlslUnary(58, TypeClass::Int, TypeClass::Other),
    ),
    (
        "PackDouble2x32",
        EmitPattern::GlslUnary(59, TypeClass::Float, TypeClass::Other),
    ),
    (
        "UnpackSnorm2x16",
        EmitPattern::GlslUnary(60, TypeClass::Other, TypeClass::Int),
    ),
    (
        "UnpackUnorm2x16",
        EmitPattern::GlslUnary(61, TypeClass::Other, TypeClass::Int),
    ),
    (
        "UnpackHalf2x16",
        EmitPattern::GlslUnary(62, TypeClass::Other, TypeClass::Int),
    ),
    (
        "UnpackSnorm4x8",
        EmitPattern::GlslUnary(63, TypeClass::Other, TypeClass::Int),
    ),
    (
        "UnpackUnorm4x8",
        EmitPattern::GlslUnary(64, TypeClass::Other, TypeClass::Int),
    ),
    (
        "UnpackDouble2x32",
        EmitPattern::GlslUnary(65, TypeClass::Other, TypeClass::Float),
    ),
    // Modf/Frexp
    (
        "ModfStruct",
        EmitPattern::GlslUnary(35, TypeClass::Other, TypeClass::Float),
    ),
    (
        "Modf",
        EmitPattern::GlslUnary(36, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FrexpStruct",
        EmitPattern::GlslUnary(51, TypeClass::Other, TypeClass::Float),
    ),
    (
        "Frexp",
        EmitPattern::GlslUnary(52, TypeClass::Float, TypeClass::Float),
    ),
    // ===== GLSL.std.450 binary =====
    (
        "Pow",
        EmitPattern::GlslBinary(26, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Atan2",
        EmitPattern::GlslBinary(25, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FMin",
        EmitPattern::GlslBinary(37, TypeClass::Float, TypeClass::Float),
    ),
    (
        "UMin",
        EmitPattern::GlslBinary(38, TypeClass::Int, TypeClass::Int),
    ),
    (
        "SMin",
        EmitPattern::GlslBinary(39, TypeClass::Int, TypeClass::Int),
    ),
    (
        "FMax",
        EmitPattern::GlslBinary(40, TypeClass::Float, TypeClass::Float),
    ),
    (
        "UMax",
        EmitPattern::GlslBinary(41, TypeClass::Int, TypeClass::Int),
    ),
    (
        "SMax",
        EmitPattern::GlslBinary(42, TypeClass::Int, TypeClass::Int),
    ),
    (
        "Step",
        EmitPattern::GlslBinary(48, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Distance",
        EmitPattern::GlslBinary(67, TypeClass::Float, TypeClass::Other),
    ),
    (
        "Cross",
        EmitPattern::GlslBinary(68, TypeClass::Other, TypeClass::Other),
    ),
    (
        "Reflect",
        EmitPattern::GlslBinary(71, TypeClass::Other, TypeClass::Other),
    ),
    (
        "Ldexp",
        EmitPattern::GlslBinaryMixed(53, TypeClass::Float, TypeClass::Float, TypeClass::Int),
    ),
    (
        "NMin",
        EmitPattern::GlslBinary(79, TypeClass::Float, TypeClass::Float),
    ),
    (
        "NMax",
        EmitPattern::GlslBinary(80, TypeClass::Float, TypeClass::Float),
    ),
    // ===== GLSL.std.450 ternary =====
    (
        "FClamp",
        EmitPattern::GlslTernary(43, TypeClass::Float, TypeClass::Float),
    ),
    (
        "UClamp",
        EmitPattern::GlslTernary(44, TypeClass::Int, TypeClass::Int),
    ),
    (
        "SClamp",
        EmitPattern::GlslTernary(45, TypeClass::Int, TypeClass::Int),
    ),
    (
        "FMix",
        EmitPattern::GlslTernary(46, TypeClass::Float, TypeClass::Float),
    ),
    (
        "VecFMix",
        EmitPattern::GlslTernary(46, TypeClass::Other, TypeClass::Other),
    ),
    (
        "SmoothStep",
        EmitPattern::GlslTernary(49, TypeClass::Float, TypeClass::Float),
    ),
    (
        "Fma",
        EmitPattern::GlslTernary(50, TypeClass::Float, TypeClass::Float),
    ),
    (
        "FaceForward",
        EmitPattern::GlslTernary(70, TypeClass::Other, TypeClass::Other),
    ),
    (
        "Refract",
        EmitPattern::GlslTernary(72, TypeClass::Other, TypeClass::Other),
    ),
    (
        "NClamp",
        EmitPattern::GlslTernary(81, TypeClass::Float, TypeClass::Float),
    ),
    // ===== Memory operations (special) =====
    ("Load", EmitPattern::Load),
    // ===== Composite extract/insert =====
    (
        "CompositeExtract",
        EmitPattern::ExtractLiteral(Op::CompositeExtract),
    ),
    (
        "VecExtract",
        EmitPattern::ExtractLiteral(Op::CompositeExtract),
    ),
    (
        "CompositeInsert",
        EmitPattern::InsertSwapped(Op::CompositeInsert),
    ),
    ("VecInsert", EmitPattern::InsertSwapped(Op::CompositeInsert)),
    // ===== CompositeConstruct variants =====
    ("Vec2", EmitPattern::CompositeN(2)),
    ("Vec3", EmitPattern::CompositeN(3)),
    ("Vec4", EmitPattern::CompositeN(4)),
    ("CompositeConstruct", EmitPattern::CompositeList),
    // ===== VectorShuffle variants =====
    ("VecShuffle2", EmitPattern::ShuffleN(2)),
    ("VecShuffle3", EmitPattern::ShuffleN(3)),
    ("VecShuffle4", EmitPattern::ShuffleN(4)),
    // ===== Image operations with operand masks =====
    (
        "ImageSampleOffset",
        EmitPattern::ImageWithMask(Op::ImageSampleImplicitLod, 16),
    ), // OFFSET = 16
    (
        "ImageSampleConstOffset",
        EmitPattern::ImageWithMask(Op::ImageSampleImplicitLod, 8),
    ), // CONST_OFFSET = 8
    (
        "ImageFetchOffset",
        EmitPattern::ImageWithMask(Op::ImageFetch, 16),
    ), // OFFSET = 16
    (
        "ImageFetchConstOffset",
        EmitPattern::ImageWithMask(Op::ImageFetch, 8),
    ), // CONST_OFFSET = 8
];

// ---------------------------------------------------------------------------
// Resolve type for TypeClass
// ---------------------------------------------------------------------------

/// Resolve the result type for an instruction.
///
/// The egraph's IType/FType/BType propagation should ensure the incoming
/// `result_type` already matches `class`. A fallback here indicates
/// incomplete type propagation in the egraph rules.
fn resolve_result_type(class: TypeClass, result_type: Word, ctx: &mut EmitCtx) -> Word {
    let original_class = ctx
        .type_classes
        .get(&result_type)
        .copied()
        .unwrap_or(TypeClass::Other);
    if original_class == class || class == TypeClass::Other {
        return result_type;
    }
    // Fallback: egraph type propagation missed this case.
    debug_assert!(
        false,
        "resolve_result_type fallback: expected {:?} but got {:?} for type id {}. \
         This indicates incomplete IType/FType/BType propagation in datatypes.egg.",
        class, original_class, result_type
    );
    match class {
        TypeClass::Int => ctx.int32_type.unwrap_or(result_type),
        TypeClass::Float => ctx.float32_type.unwrap_or(result_type),
        TypeClass::Bool => ctx.bool_type.unwrap_or(result_type),
        TypeClass::Other => result_type,
    }
}

/// Derive the operand type from the instruction's result type.
///
/// For same-class operations (e.g., IAdd: Int->Int), this returns result_type
/// unchanged. For cross-class operations (e.g., IEqual: Bool result, Int operands),
/// this legitimately falls back to the canonical type for the operand class.
/// This is by design, not a propagation gap.
fn resolve_operand_type(class: TypeClass, result_type: Word, ctx: &EmitCtx) -> Word {
    let original_class = ctx
        .type_classes
        .get(&result_type)
        .copied()
        .unwrap_or(TypeClass::Other);
    if original_class == class || class == TypeClass::Other {
        return result_type;
    }
    match class {
        TypeClass::Int => ctx.int32_type.unwrap_or(result_type),
        TypeClass::Float => ctx.float32_type.unwrap_or(result_type),
        TypeClass::Bool => ctx.bool_type.unwrap_or(result_type),
        TypeClass::Other => result_type,
    }
}

// ---------------------------------------------------------------------------
// Core emission
// ---------------------------------------------------------------------------

/// Recursively emit SPIR-V instructions from a parsed term tree.
///
/// Returns `(result_id, synthesized_instructions)` where:
/// - `result_id` is the ID that holds the computed value
/// - `synthesized_instructions` is a list of new instructions created
///   (may be empty if the term resolved to an existing ID)
pub fn emit_term(
    term: &Term,
    result_type: Word,
    ctx: &mut EmitCtx,
) -> Option<(Word, Vec<Instruction>)> {
    match term {
        Term::Atom(s) => resolve_atom(s, ctx),
        Term::App { op, args } => emit_app(op, args, result_type, ctx),
    }
}

/// Resolve an atom to an existing ID.
fn resolve_atom(s: &str, ctx: &EmitCtx) -> Option<(Word, Vec<Instruction>)> {
    let &id = ctx.id_map.get(s)?;
    Some((id, Vec::new()))
}

/// Try to resolve a term to an existing ID and look up its SPIR-V type.
/// Returns the type ID if the term is a symbol reference with a known type.
fn resolve_term_type(term: &Term, ctx: &EmitCtx) -> Option<Word> {
    match term {
        Term::Atom(s) => {
            let &id = ctx.id_map.get(s.as_str())?;
            ctx.id_to_type.get(&id).copied()
        }
        Term::App { op, args } => match op.as_str() {
            "Sym" | "ISym" | "FSym" | "BSym" => {
                let name = match args.first() {
                    Some(Term::Atom(n)) => n.as_str(),
                    _ => return None,
                };
                let &id = ctx.id_map.get(name)?;
                ctx.id_to_type.get(&id).copied()
            }
            // Bridge constructors are transparent — recurse into the inner term
            "IntToExpr" | "FloatToExpr" | "BoolToExpr" | "ExprToInt" | "ExprToFloat"
            | "ExprToBool" => args.first().and_then(|inner| resolve_term_type(inner, ctx)),
            _ => None,
        },
    }
}

/// Emit an application node.
fn emit_app(
    op: &str,
    args: &[Term],
    result_type: Word,
    ctx: &mut EmitCtx,
) -> Option<(Word, Vec<Instruction>)> {
    // --- Sym variants: resolve symbol to ID ---
    match op {
        "Sym" | "ISym" | "FSym" | "BSym" => {
            if let Some(Term::Atom(name)) = args.first() {
                if let Some(&id) = ctx.id_map.get(name.as_str()) {
                    return Some((id, Vec::new()));
                }
            }
            return None;
        }
        _ => {}
    }

    // --- Constants ---
    match op {
        "Const" => return emit_const_int(args, false, result_type, ctx),
        "Const64" => return emit_const_int(args, true, result_type, ctx),
        "BoolConst" => return emit_bool_const(args, ctx),
        "FConst" => return emit_float_const(args, result_type, ctx),
        _ => {}
    }

    // --- AccessChain: base pointer type differs from result pointer type ---
    // When the base is a nested expression (not a Sym), we can't determine
    // its correct pointer type. Bail out to preserve the original instruction
    // rather than synthesizing an AccessChain with a wrong intermediate type.
    if op == "AccessChainDyn" {
        if args.len() < 2 {
            return None;
        }
        let mut synth = Vec::new();
        let base_type = resolve_term_type(&args[0], ctx);
        let (base, mut s) = emit_term(&args[0], base_type.unwrap_or(result_type), ctx)?;
        if !s.is_empty() && base_type.is_none() {
            return None;
        }
        synth.append(&mut s);
        let idx_type = resolve_term_type(&args[1], ctx).unwrap_or(result_type);
        let (idx, mut s) = emit_term(&args[1], idx_type, ctx)?;
        synth.append(&mut s);
        let id = alloc_id(ctx);
        synth.push(Instruction::new(
            Op::AccessChain,
            Some(result_type),
            Some(id),
            vec![
                rspirv::dr::Operand::IdRef(base),
                rspirv::dr::Operand::IdRef(idx),
            ],
        ));
        return Some((id, synth));
    }

    // --- Lookup in unified ops table ---
    use std::sync::OnceLock;
    static OPS_MAP: OnceLock<HashMap<&'static str, EmitPattern>> = OnceLock::new();
    let map = OPS_MAP.get_or_init(|| OPS_TABLE.iter().copied().collect());
    if let Some(pattern) = map.get(op) {
        return emit_pattern(pattern, args, result_type, ctx);
    }

    #[cfg(debug_assertions)]
    eprintln!(
        "emit_app: unrecognized constructor '{}' — missing from OPS_TABLE",
        op
    );
    None
}

/// Emit a pattern from the ops table.
fn emit_pattern(
    pattern: &EmitPattern,
    args: &[Term],
    result_type: Word,
    ctx: &mut EmitCtx,
) -> Option<(Word, Vec<Instruction>)> {
    match *pattern {
        EmitPattern::Unary(opcode, result_class, operand_class)
        | EmitPattern::Binary(opcode, result_class, operand_class)
        | EmitPattern::Ternary(opcode, result_class, operand_class)
        | EmitPattern::Quaternary(opcode, result_class, operand_class) => {
            let arity = match *pattern {
                EmitPattern::Unary(..) => 1,
                EmitPattern::Binary(..) => 2,
                EmitPattern::Ternary(..) => 3,
                EmitPattern::Quaternary(..) => 4,
                _ => unreachable!(),
            };
            if args.len() < arity {
                return None;
            }
            let op_result_type = resolve_result_type(result_class, result_type, ctx);
            let operand_type = resolve_operand_type(operand_class, result_type, ctx);
            let mut synth = Vec::new();
            let mut operand_ids = Vec::with_capacity(arity);
            for arg in &args[..arity] {
                let (arg_id, mut s) = emit_term(arg, operand_type, ctx)?;
                synth.append(&mut s);
                operand_ids.push(rspirv::dr::Operand::IdRef(arg_id));
            }
            let id = alloc_id(ctx);
            synth.push(Instruction::new(
                opcode,
                Some(op_result_type),
                Some(id),
                operand_ids,
            ));
            Some((id, synth))
        }
        EmitPattern::GlslUnary(glsl_opcode, result_class, operand_class)
        | EmitPattern::GlslBinary(glsl_opcode, result_class, operand_class)
        | EmitPattern::GlslTernary(glsl_opcode, result_class, operand_class) => {
            let ext_id = ctx.glsl_ext_id?;
            let arity = match *pattern {
                EmitPattern::GlslUnary(..) => 1,
                EmitPattern::GlslBinary(..) => 2,
                EmitPattern::GlslTernary(..) => 3,
                _ => unreachable!(),
            };
            if args.len() < arity {
                return None;
            }
            let op_result_type = resolve_result_type(result_class, result_type, ctx);
            let operand_type = resolve_operand_type(operand_class, result_type, ctx);
            let mut synth = Vec::new();
            let mut operands = vec![
                rspirv::dr::Operand::IdRef(ext_id),
                rspirv::dr::Operand::LiteralExtInstInteger(glsl_opcode),
            ];
            for arg in &args[..arity] {
                let (arg_id, mut s) = emit_term(arg, operand_type, ctx)?;
                synth.append(&mut s);
                operands.push(rspirv::dr::Operand::IdRef(arg_id));
            }
            let id = alloc_id(ctx);
            synth.push(Instruction::new(
                Op::ExtInst,
                Some(op_result_type),
                Some(id),
                operands,
            ));
            Some((id, synth))
        }
        EmitPattern::GlslBinaryMixed(glsl_opcode, result_class, op1_class, op2_class) => {
            let ext_id = ctx.glsl_ext_id?;
            if args.len() < 2 {
                return None;
            }
            let op_result_type = resolve_result_type(result_class, result_type, ctx);
            let op1_type = resolve_operand_type(op1_class, result_type, ctx);
            let op2_type = resolve_operand_type(op2_class, result_type, ctx);
            let mut synth = Vec::new();
            let (a, mut s) = emit_term(&args[0], op1_type, ctx)?;
            synth.append(&mut s);
            let (b, mut s) = emit_term(&args[1], op2_type, ctx)?;
            synth.append(&mut s);
            let id = alloc_id(ctx);
            synth.push(Instruction::new(
                Op::ExtInst,
                Some(op_result_type),
                Some(id),
                vec![
                    rspirv::dr::Operand::IdRef(ext_id),
                    rspirv::dr::Operand::LiteralExtInstInteger(glsl_opcode),
                    rspirv::dr::Operand::IdRef(a),
                    rspirv::dr::Operand::IdRef(b),
                ],
            ));
            Some((id, synth))
        }
        EmitPattern::Select(type_class) => {
            if args.len() < 3 {
                return None;
            }
            let select_type = resolve_result_type(type_class, result_type, ctx);
            let cond_type = resolve_operand_type(TypeClass::Bool, result_type, ctx);
            let mut synth = Vec::new();
            let (cond, mut s) = emit_term(&args[0], cond_type, ctx)?;
            synth.append(&mut s);
            let (then_val, mut s) = emit_term(&args[1], select_type, ctx)?;
            synth.append(&mut s);
            let (else_val, mut s) = emit_term(&args[2], select_type, ctx)?;
            synth.append(&mut s);
            let id = alloc_id(ctx);
            synth.push(Instruction::new(
                Op::Select,
                Some(select_type),
                Some(id),
                vec![
                    rspirv::dr::Operand::IdRef(cond),
                    rspirv::dr::Operand::IdRef(then_val),
                    rspirv::dr::Operand::IdRef(else_val),
                ],
            ));
            Some((id, synth))
        }
        EmitPattern::Bridge => {
            if args.is_empty() {
                return None;
            }
            // Bridges are transparent — pass the parent's type through unchanged
            emit_term(&args[0], result_type, ctx)
        }
        EmitPattern::Load => {
            if args.is_empty() {
                return None;
            }
            let mut synth = Vec::new();
            // Pointer operand type differs from result_type (loaded value type).
            let ptr_type = resolve_term_type(&args[0], ctx).unwrap_or(result_type);
            let (ptr, mut s) = emit_term(&args[0], ptr_type, ctx)?;
            synth.append(&mut s);
            let id = alloc_id(ctx);
            synth.push(Instruction::new(
                Op::Load,
                Some(result_type),
                Some(id),
                vec![rspirv::dr::Operand::IdRef(ptr)],
            ));
            Some((id, synth))
        }
        EmitPattern::ExtractLiteral(opcode) => {
            if args.len() < 2 {
                return None;
            }
            let mut synth = Vec::new();
            // Composite operand type differs from result_type (element type).
            let composite_type = resolve_term_type(&args[0], ctx).unwrap_or(result_type);
            let (composite, mut s) = emit_term(&args[0], composite_type, ctx)?;
            synth.append(&mut s);
            let index = term_as_u32(&args[1])?;
            let id = alloc_id(ctx);
            synth.push(Instruction::new(
                opcode,
                Some(result_type),
                Some(id),
                vec![
                    rspirv::dr::Operand::IdRef(composite),
                    rspirv::dr::Operand::LiteralBit32(index),
                ],
            ));
            Some((id, synth))
        }
        EmitPattern::InsertSwapped(opcode) => {
            if args.len() < 3 {
                return None;
            }
            let mut synth = Vec::new();
            let (composite, mut s) = emit_term(&args[0], result_type, ctx)?;
            synth.append(&mut s);
            // Object operand type differs from result_type (composite type).
            let object_type = resolve_term_type(&args[1], ctx).unwrap_or(result_type);
            let (object, mut s) = emit_term(&args[1], object_type, ctx)?;
            synth.append(&mut s);
            let index = term_as_u32(&args[2])?;
            let id = alloc_id(ctx);
            // SPIR-V operand order: object, composite, index
            synth.push(Instruction::new(
                opcode,
                Some(result_type),
                Some(id),
                vec![
                    rspirv::dr::Operand::IdRef(object),
                    rspirv::dr::Operand::IdRef(composite),
                    rspirv::dr::Operand::LiteralBit32(index),
                ],
            ));
            Some((id, synth))
        }
        EmitPattern::CompositeN(count) => {
            emit_composite_construct(args, count as usize, result_type, ctx)
        }
        EmitPattern::CompositeList => {
            if args.is_empty() {
                return None;
            }
            let (components, mut synth) = flatten_expr_list(&args[0], result_type, ctx)?;
            if components.is_empty() {
                return None;
            }
            let id = alloc_id(ctx);
            let operands = components
                .into_iter()
                .map(rspirv::dr::Operand::IdRef)
                .collect();
            synth.push(Instruction::new(
                Op::CompositeConstruct,
                Some(result_type),
                Some(id),
                operands,
            ));
            Some((id, synth))
        }
        EmitPattern::ShuffleN(count) => emit_vector_shuffle(args, count as usize, result_type, ctx),
        EmitPattern::ImageWithMask(opcode, mask_bits) => {
            if args.len() < 3 {
                return None;
            }
            let mask = rspirv::spirv::ImageOperands::from_bits(mask_bits).unwrap_or_else(|| {
                panic!("invalid image operand mask {:#x} in OPS_TABLE", mask_bits)
            });
            let mut synth = Vec::new();
            // Image, coordinate, and offset operand types differ from result_type.
            let image_type = resolve_term_type(&args[0], ctx).unwrap_or(result_type);
            let (image, mut s) = emit_term(&args[0], image_type, ctx)?;
            synth.append(&mut s);
            let coord_type = resolve_term_type(&args[1], ctx).unwrap_or(result_type);
            let (coord, mut s) = emit_term(&args[1], coord_type, ctx)?;
            synth.append(&mut s);
            let offset_type = resolve_term_type(&args[2], ctx).unwrap_or(result_type);
            let (offset, mut s) = emit_term(&args[2], offset_type, ctx)?;
            synth.append(&mut s);
            let id = alloc_id(ctx);
            synth.push(Instruction::new(
                opcode,
                Some(result_type),
                Some(id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(coord),
                    rspirv::dr::Operand::ImageOperands(mask),
                    rspirv::dr::Operand::IdRef(offset),
                ],
            ));
            Some((id, synth))
        }
    }
}

// ---------------------------------------------------------------------------
// Constant emission
// ---------------------------------------------------------------------------

/// Emit a 32-bit or 64-bit integer constant.
/// For (Const N): uses result_type's width to decide encoding (supports bool, 32-bit, 64-bit).
/// For (Const64 N): always uses 64-bit encoding.
fn emit_const_int(
    args: &[Term],
    is_64: bool,
    result_type: Word,
    ctx: &mut EmitCtx,
) -> Option<(Word, Vec<Instruction>)> {
    let value_str = match args.first() {
        Some(Term::Atom(s)) => s,
        _ => return None,
    };
    let value: i64 = value_str.parse().ok()?;

    // Determine actual bit width from result type
    let type_width = ctx.type_widths.get(&result_type).copied();
    let actual_64 = is_64 || type_width == Some(64);

    // Boolean constants are handled by BoolConst → emit_bool_const().
    // Const is an IntExpr constructor, so it should never extract with a boolean type.
    debug_assert!(
        type_width != Some(1),
        "emit_const_int called with boolean type (width=1, value={}). \
         This should be handled by BoolConst/emit_bool_const instead.",
        value
    );

    let const_key = format!("const_{}_{}", result_type, value);

    // Return existing constant if available
    if let Some(&id) = ctx.id_map.get(&const_key) {
        return Some((id, Vec::new()));
    }

    // Synthesize new constant
    let ty = if actual_64 {
        ctx.int64_type.or(ctx.int32_type)?
    } else {
        ctx.int32_type?
    };
    let id = alloc_id(ctx);
    let operand = if actual_64 {
        rspirv::dr::Operand::LiteralBit64(value as u64)
    } else {
        rspirv::dr::Operand::LiteralBit32(value as u32)
    };
    let inst = Instruction::new(Op::Constant, Some(ty), Some(id), vec![operand]);
    ctx.id_map.insert(const_key, id);
    Some((id, vec![inst]))
}

/// Emit a boolean constant.
fn emit_bool_const(args: &[Term], ctx: &mut EmitCtx) -> Option<(Word, Vec<Instruction>)> {
    let value_str = match args.first() {
        Some(Term::Atom(s)) => s,
        _ => return None,
    };
    let value: i64 = value_str.parse().ok()?;

    let const_key = format!("boolconst_{}", value);
    if let Some(&id) = ctx.id_map.get(&const_key) {
        return Some((id, Vec::new()));
    }

    let ty = ctx.bool_type?;
    let id = alloc_id(ctx);
    let op = if value != 0 {
        Op::ConstantTrue
    } else {
        Op::ConstantFalse
    };
    let inst = Instruction::new(op, Some(ty), Some(id), vec![]);
    ctx.id_map.insert(const_key, id);
    Some((id, vec![inst]))
}

/// Emit a float constant.
fn emit_float_const(
    args: &[Term],
    result_type: Word,
    ctx: &mut EmitCtx,
) -> Option<(Word, Vec<Instruction>)> {
    let value_str = match args.first() {
        Some(Term::Atom(s)) => s,
        _ => return None,
    };
    let value: f64 = value_str.parse().ok()?;

    let type_width = ctx.type_widths.get(&result_type).copied();
    let const_key = format!("fconst_{}_{}", result_type, value.to_bits());
    if let Some(&id) = ctx.id_map.get(&const_key) {
        return Some((id, Vec::new()));
    }

    let ty = if type_width == Some(64) {
        ctx.float64_type.or(ctx.float32_type)?
    } else {
        ctx.float32_type?
    };
    let id = alloc_id(ctx);
    let operand = if type_width == Some(64) {
        rspirv::dr::Operand::LiteralBit64(value.to_bits())
    } else {
        rspirv::dr::Operand::LiteralBit32((value as f32).to_bits())
    };
    let inst = Instruction::new(Op::Constant, Some(ty), Some(id), vec![operand]);
    ctx.id_map.insert(const_key, id);
    Some((id, vec![inst]))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Allocate a new SPIR-V ID and register it in the id_map.
fn alloc_id(ctx: &mut EmitCtx) -> Word {
    let id = *ctx.next_id;
    *ctx.next_id += 1;
    ctx.id_map.insert(format!("id{}", id), id);
    id
}

/// Extract a u32 literal from an Atom term.
fn term_as_u32(term: &Term) -> Option<u32> {
    match term {
        Term::Atom(s) => s.parse().ok(),
        _ => None,
    }
}

/// Emit a CompositeConstruct from N positional arguments.
fn emit_composite_construct(
    args: &[Term],
    expected: usize,
    result_type: Word,
    ctx: &mut EmitCtx,
) -> Option<(Word, Vec<Instruction>)> {
    if args.len() < expected {
        return None;
    }
    let mut synth = Vec::new();
    let mut operands = Vec::new();
    for arg in &args[..expected] {
        // Each component's type differs from result_type (composite type).
        let component_type = resolve_term_type(arg, ctx).unwrap_or(result_type);
        let (arg_id, mut s) = emit_term(arg, component_type, ctx)?;
        synth.append(&mut s);
        operands.push(rspirv::dr::Operand::IdRef(arg_id));
    }
    let id = alloc_id(ctx);
    synth.push(Instruction::new(
        Op::CompositeConstruct,
        Some(result_type),
        Some(id),
        operands,
    ));
    Some((id, synth))
}

/// Flatten an ECons/ENil list into a vector of resolved IDs and any synthesized instructions.
fn flatten_expr_list(
    term: &Term,
    result_type: Word,
    ctx: &mut EmitCtx,
) -> Option<(Vec<Word>, Vec<Instruction>)> {
    match term {
        Term::App { op, .. } if op == "ENil" => Some((Vec::new(), Vec::new())),
        Term::App { op, args } if op == "ECons" && args.len() >= 2 => {
            // Each element's type differs from result_type (composite type).
            let head_type = resolve_term_type(&args[0], ctx).unwrap_or(result_type);
            let (head_id, head_synth) = emit_term(&args[0], head_type, ctx)?;
            let (mut rest_ids, mut rest_synth) = flatten_expr_list(&args[1], result_type, ctx)?;
            let mut ids = vec![head_id];
            ids.append(&mut rest_ids);
            let mut synth = head_synth;
            synth.append(&mut rest_synth);
            Some((ids, synth))
        }
        _ => None,
    }
}

/// Emit a VectorShuffle instruction with N literal indices.
fn emit_vector_shuffle(
    args: &[Term],
    num_indices: usize,
    result_type: Word,
    ctx: &mut EmitCtx,
) -> Option<(Word, Vec<Instruction>)> {
    if args.len() < 2 + num_indices {
        return None;
    }
    let mut synth = Vec::new();
    // Input vector types may differ from result_type (output vector type).
    let v1_type = resolve_term_type(&args[0], ctx).unwrap_or(result_type);
    let (v1, mut s) = emit_term(&args[0], v1_type, ctx)?;
    synth.append(&mut s);
    let v2_type = resolve_term_type(&args[1], ctx).unwrap_or(result_type);
    let (v2, mut s) = emit_term(&args[1], v2_type, ctx)?;
    synth.append(&mut s);
    let mut operands = vec![
        rspirv::dr::Operand::IdRef(v1),
        rspirv::dr::Operand::IdRef(v2),
    ];
    for i in 0..num_indices {
        let idx = term_as_u32(&args[2 + i])?;
        operands.push(rspirv::dr::Operand::LiteralBit32(idx));
    }
    let id = alloc_id(ctx);
    synth.push(Instruction::new(
        Op::VectorShuffle,
        Some(result_type),
        Some(id),
        operands,
    ));
    Some((id, synth))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_atom() {
        assert_eq!(parse_sexpr("id5"), Some(Term::Atom("id5".into())));
        assert_eq!(parse_sexpr("42"), Some(Term::Atom("42".into())));
        assert_eq!(parse_sexpr("-3"), Some(Term::Atom("-3".into())));
        assert_eq!(parse_sexpr("3.14"), Some(Term::Atom("3.14".into())));
    }

    #[test]
    fn parse_quoted_string() {
        assert_eq!(parse_sexpr("\"id5\""), Some(Term::Atom("id5".into())));
        assert_eq!(
            parse_sexpr("\"hello world\""),
            Some(Term::Atom("hello world".into()))
        );
    }

    #[test]
    fn parse_simple_app() {
        assert_eq!(
            parse_sexpr("(Const 42)"),
            Some(Term::App {
                op: "Const".into(),
                args: vec![Term::Atom("42".into())]
            })
        );
    }

    #[test]
    fn parse_binary_app() {
        assert_eq!(
            parse_sexpr("(Add id5 id6)"),
            Some(Term::App {
                op: "Add".into(),
                args: vec![Term::Atom("id5".into()), Term::Atom("id6".into())]
            })
        );
    }

    #[test]
    fn parse_nested_app() {
        let result = parse_sexpr("(Add (Sym \"id5\") (Const 3))");
        assert_eq!(
            result,
            Some(Term::App {
                op: "Add".into(),
                args: vec![
                    Term::App {
                        op: "Sym".into(),
                        args: vec![Term::Atom("id5".into())]
                    },
                    Term::App {
                        op: "Const".into(),
                        args: vec![Term::Atom("3".into())]
                    },
                ]
            })
        );
    }

    #[test]
    fn parse_deeply_nested() {
        let result = parse_sexpr("(IntToExpr (Add (Sym \"id5\") (Mul (Sym \"id6\") (Const 2))))");
        assert!(result.is_some());
        if let Some(Term::App { op, args }) = &result {
            assert_eq!(op, "IntToExpr");
            assert_eq!(args.len(), 1);
            if let Term::App { op, args } = &args[0] {
                assert_eq!(op, "Add");
                assert_eq!(args.len(), 2);
            }
        }
    }

    #[test]
    fn parse_empty_app() {
        assert_eq!(
            parse_sexpr("(ENil)"),
            Some(Term::App {
                op: "ENil".into(),
                args: vec![]
            })
        );
    }

    #[test]
    fn parse_with_whitespace() {
        let result = parse_sexpr("  (Add   id5   id6  )  ");
        assert_eq!(
            result,
            Some(Term::App {
                op: "Add".into(),
                args: vec![Term::Atom("id5".into()), Term::Atom("id6".into())]
            })
        );
    }

    #[test]
    fn parse_negative_number() {
        assert_eq!(
            parse_sexpr("(Const -7)"),
            Some(Term::App {
                op: "Const".into(),
                args: vec![Term::Atom("-7".into())]
            })
        );
    }

    #[test]
    fn parse_returns_none_for_trailing_garbage() {
        assert_eq!(parse_sexpr("id5 extra"), None);
    }

    #[test]
    fn parse_returns_none_for_empty() {
        assert_eq!(parse_sexpr(""), None);
        assert_eq!(parse_sexpr("  "), None);
    }

    #[test]
    fn emit_resolves_sym() {
        let mut id_map = HashMap::new();
        id_map.insert("id5".to_string(), 5);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = EmitCtx {
            id_map: &mut id_map,
            next_id: &mut next_id,
            int32_type: Some(10),
            int64_type: None,
            float32_type: Some(11),
            float64_type: None,
            bool_type: Some(12),
            type_classes: &type_classes,
            glsl_ext_id: None,
            type_widths: &type_widths,
            id_to_type: &HashMap::new(),
        };
        let term = parse_sexpr("(Sym \"id5\")").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 5);
        assert!(synth.is_empty());
    }

    #[test]
    fn emit_synthesizes_constant() {
        let mut id_map = HashMap::new();
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = EmitCtx {
            id_map: &mut id_map,
            next_id: &mut next_id,
            int32_type: Some(10),
            int64_type: None,
            float32_type: Some(11),
            float64_type: None,
            bool_type: Some(12),
            type_classes: &type_classes,
            glsl_ext_id: None,
            type_widths: &type_widths,
            id_to_type: &HashMap::new(),
        };
        let term = parse_sexpr("(Const 42)").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::Constant);
    }

    #[test]
    fn emit_binary_op_with_nested_constants() {
        let mut id_map = HashMap::new();
        let mut next_id = 100;
        let mut type_classes = HashMap::new();
        type_classes.insert(10u32, TypeClass::Int);
        let type_widths = HashMap::new();
        let mut ctx = EmitCtx {
            id_map: &mut id_map,
            next_id: &mut next_id,
            int32_type: Some(10),
            int64_type: None,
            float32_type: Some(11),
            float64_type: None,
            bool_type: Some(12),
            type_classes: &type_classes,
            glsl_ext_id: None,
            type_widths: &type_widths,
            id_to_type: &HashMap::new(),
        };
        let term = parse_sexpr("(Add (Const 3) (Const 5))").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        // Should synthesize: const 3, const 5, Add
        assert_eq!(synth.len(), 3);
        assert_eq!(synth[0].class.opcode, Op::Constant);
        assert_eq!(synth[1].class.opcode, Op::Constant);
        assert_eq!(synth[2].class.opcode, Op::IAdd);
        assert_eq!(result_id, 102); // 100=const3, 101=const5, 102=add
    }

    #[test]
    fn emit_bridge_is_transparent() {
        let mut id_map = HashMap::new();
        id_map.insert("id5".to_string(), 5);
        let mut next_id = 100;
        let mut type_classes = HashMap::new();
        type_classes.insert(10u32, TypeClass::Int);
        let type_widths = HashMap::new();
        let mut ctx = EmitCtx {
            id_map: &mut id_map,
            next_id: &mut next_id,
            int32_type: Some(10),
            int64_type: None,
            float32_type: Some(11),
            float64_type: None,
            bool_type: Some(12),
            type_classes: &type_classes,
            glsl_ext_id: None,
            type_widths: &type_widths,
            id_to_type: &HashMap::new(),
        };
        let term = parse_sexpr("(IntToExpr (Sym \"id5\"))").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 5);
        assert!(synth.is_empty());
    }

    /// Helper to create an EmitCtx for tests with some ids pre-registered.
    fn make_test_ctx<'a>(
        id_map: &'a mut HashMap<String, Word>,
        next_id: &'a mut Word,
        type_classes: &'a HashMap<Word, TypeClass>,
        type_widths: &'a HashMap<Word, u32>,
    ) -> EmitCtx<'a> {
        static EMPTY_ID_TO_TYPE: std::sync::OnceLock<HashMap<Word, Word>> =
            std::sync::OnceLock::new();
        EmitCtx {
            id_map,
            next_id,
            int32_type: Some(10),
            int64_type: None,
            float32_type: Some(11),
            float64_type: None,
            bool_type: Some(12),
            type_classes,
            glsl_ext_id: None,
            type_widths,
            id_to_type: EMPTY_ID_TO_TYPE.get_or_init(HashMap::new),
        }
    }

    // ===== Load =====

    #[test]
    fn emit_load_from_pointer() {
        let mut id_map = HashMap::new();
        id_map.insert("ptr1".to_string(), 1);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(Load (Sym \"ptr1\") (Sym \"mem0\"))").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::Load);
    }

    #[test]
    fn emit_load_no_args_returns_none() {
        let mut id_map = HashMap::new();
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = Term::App {
            op: "Load".into(),
            args: vec![],
        };
        assert!(emit_term(&term, 10, &mut ctx).is_none());
    }

    // ===== CompositeExtract / VecExtract =====

    #[test]
    fn emit_composite_extract_with_literal_index() {
        let mut id_map = HashMap::new();
        id_map.insert("vec1".to_string(), 1);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(CompositeExtract (Sym \"vec1\") 2)").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::CompositeExtract);
    }

    #[test]
    fn emit_vec_extract_uses_composite_extract_opcode() {
        let mut id_map = HashMap::new();
        id_map.insert("vec1".to_string(), 1);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(VecExtract (Sym \"vec1\") 0)").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::CompositeExtract);
    }

    #[test]
    fn emit_composite_extract_too_few_args_returns_none() {
        let mut id_map = HashMap::new();
        id_map.insert("vec1".to_string(), 1);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(CompositeExtract (Sym \"vec1\"))").unwrap();
        assert!(emit_term(&term, 10, &mut ctx).is_none());
    }

    // ===== CompositeInsert / VecInsert =====

    #[test]
    fn emit_composite_insert_swaps_operands() {
        let mut id_map = HashMap::new();
        id_map.insert("vec1".to_string(), 1);
        id_map.insert("val1".to_string(), 2);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        // Egglog order: (CompositeInsert composite object index)
        let term = parse_sexpr("(CompositeInsert (Sym \"vec1\") (Sym \"val1\") 1)").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::CompositeInsert);
        // SPIR-V operand order: object (val1=2), composite (vec1=1), index (1)
        assert_eq!(synth[0].operands[0], rspirv::dr::Operand::IdRef(2));
        assert_eq!(synth[0].operands[1], rspirv::dr::Operand::IdRef(1));
        assert_eq!(synth[0].operands[2], rspirv::dr::Operand::LiteralBit32(1));
    }

    #[test]
    fn emit_composite_insert_too_few_args_returns_none() {
        let mut id_map = HashMap::new();
        id_map.insert("vec1".to_string(), 1);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(CompositeInsert (Sym \"vec1\") (Sym \"vec1\"))").unwrap();
        assert!(emit_term(&term, 10, &mut ctx).is_none());
    }

    // ===== Vec2/Vec3/Vec4 (CompositeN) =====

    #[test]
    fn emit_vec2_composite_construct() {
        let mut id_map = HashMap::new();
        id_map.insert("a".to_string(), 1);
        id_map.insert("b".to_string(), 2);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(Vec2 (Sym \"a\") (Sym \"b\"))").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::CompositeConstruct);
        assert_eq!(synth[0].operands.len(), 2);
    }

    #[test]
    fn emit_vec3_composite_construct() {
        let mut id_map = HashMap::new();
        id_map.insert("a".to_string(), 1);
        id_map.insert("b".to_string(), 2);
        id_map.insert("c".to_string(), 3);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(Vec3 (Sym \"a\") (Sym \"b\") (Sym \"c\"))").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::CompositeConstruct);
        assert_eq!(synth[0].operands.len(), 3);
    }

    #[test]
    fn emit_vec4_too_few_args_returns_none() {
        let mut id_map = HashMap::new();
        id_map.insert("a".to_string(), 1);
        id_map.insert("b".to_string(), 2);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(Vec4 (Sym \"a\") (Sym \"b\"))").unwrap();
        assert!(emit_term(&term, 10, &mut ctx).is_none());
    }

    // ===== CompositeConstruct (CompositeList) =====

    #[test]
    fn emit_composite_construct_from_econs_list() {
        let mut id_map = HashMap::new();
        id_map.insert("x".to_string(), 1);
        id_map.insert("y".to_string(), 2);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term =
            parse_sexpr("(CompositeConstruct (ECons (Sym \"x\") (ECons (Sym \"y\") (ENil))))")
                .unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::CompositeConstruct);
        assert_eq!(synth[0].operands.len(), 2);
    }

    #[test]
    fn emit_composite_construct_empty_list_returns_none() {
        let mut id_map = HashMap::new();
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(CompositeConstruct (ENil))").unwrap();
        assert!(emit_term(&term, 10, &mut ctx).is_none());
    }

    // ===== VecShuffle (ShuffleN) =====

    #[test]
    fn emit_vec_shuffle2() {
        let mut id_map = HashMap::new();
        id_map.insert("v1".to_string(), 1);
        id_map.insert("v2".to_string(), 2);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(VecShuffle2 (Sym \"v1\") (Sym \"v2\") 0 3)").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::VectorShuffle);
        // 2 IdRef + 2 LiteralBit32
        assert_eq!(synth[0].operands.len(), 4);
    }

    #[test]
    fn emit_vec_shuffle4() {
        let mut id_map = HashMap::new();
        id_map.insert("v1".to_string(), 1);
        id_map.insert("v2".to_string(), 2);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(VecShuffle4 (Sym \"v1\") (Sym \"v2\") 0 1 4 5)").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::VectorShuffle);
        // 2 IdRef + 4 LiteralBit32
        assert_eq!(synth[0].operands.len(), 6);
    }

    #[test]
    fn emit_vec_shuffle_too_few_indices_returns_none() {
        let mut id_map = HashMap::new();
        id_map.insert("v1".to_string(), 1);
        id_map.insert("v2".to_string(), 2);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        // VecShuffle3 needs 2 vectors + 3 indices = 5 args, only 4 given
        let term = parse_sexpr("(VecShuffle3 (Sym \"v1\") (Sym \"v2\") 0 1)").unwrap();
        assert!(emit_term(&term, 10, &mut ctx).is_none());
    }

    // ===== ImageWithMask =====

    #[test]
    fn emit_image_sample_with_offset() {
        let mut id_map = HashMap::new();
        id_map.insert("img".to_string(), 1);
        id_map.insert("coord".to_string(), 2);
        id_map.insert("off".to_string(), 3);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term =
            parse_sexpr("(ImageSampleOffset (Sym \"img\") (Sym \"coord\") (Sym \"off\"))").unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::ImageSampleImplicitLod);
        // operands: image, coord, mask, offset
        assert_eq!(synth[0].operands.len(), 4);
        assert_eq!(
            synth[0].operands[2],
            rspirv::dr::Operand::ImageOperands(rspirv::spirv::ImageOperands::OFFSET)
        );
    }

    #[test]
    fn emit_image_fetch_const_offset() {
        let mut id_map = HashMap::new();
        id_map.insert("img".to_string(), 1);
        id_map.insert("coord".to_string(), 2);
        id_map.insert("off".to_string(), 3);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term =
            parse_sexpr("(ImageFetchConstOffset (Sym \"img\") (Sym \"coord\") (Sym \"off\"))")
                .unwrap();
        let (result_id, synth) = emit_term(&term, 10, &mut ctx).unwrap();
        assert_eq!(result_id, 100);
        assert_eq!(synth.len(), 1);
        assert_eq!(synth[0].class.opcode, Op::ImageFetch);
        assert_eq!(
            synth[0].operands[2],
            rspirv::dr::Operand::ImageOperands(rspirv::spirv::ImageOperands::CONST_OFFSET)
        );
    }

    #[test]
    fn emit_image_with_mask_too_few_args_returns_none() {
        let mut id_map = HashMap::new();
        id_map.insert("img".to_string(), 1);
        id_map.insert("coord".to_string(), 2);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(ImageSampleOffset (Sym \"img\") (Sym \"coord\"))").unwrap();
        assert!(emit_term(&term, 10, &mut ctx).is_none());
    }

    // ===== Unknown op returns None =====

    #[test]
    fn emit_unknown_op_returns_none() {
        let mut id_map = HashMap::new();
        id_map.insert("x".to_string(), 1);
        let mut next_id = 100;
        let type_classes = HashMap::new();
        let type_widths = HashMap::new();
        let mut ctx = make_test_ctx(&mut id_map, &mut next_id, &type_classes, &type_widths);
        let term = parse_sexpr("(CompletelyBogusOp (Sym \"x\"))").unwrap();
        assert!(emit_term(&term, 10, &mut ctx).is_none());
    }
}
