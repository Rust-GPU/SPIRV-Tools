use crate::{ConstValue, SpirvLang};
use egg::{Id, RecExpr, Symbol};
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use rspirv::spirv::Word;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors surfaced when translating SPIR-V into optimizer expressions.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TranslateError {
    /// An instruction opcode is not supported by the arithmetic translator.
    #[error("unsupported opcode {0:?} in arithmetic translation")]
    UnsupportedOp(Op),
    /// An instruction is missing a result id.
    #[error("instruction {0:?} is missing a result id")]
    MissingResultId(Op),
    /// An operand id could not be resolved to a previously defined value.
    #[error("operand id {id} for {opcode:?} was not defined earlier in the block")]
    UnknownOperand { id: u32, opcode: Op },
    /// A constant operand had an unexpected layout.
    #[error("constant for id {id} is missing or not a 32-bit literal")]
    InvalidConstant { id: u32 },
    /// A rebuilt node was missing an original result id.
    #[error("optimized node {node:?} is missing original result id for rebuild")]
    MissingOriginalId { node: SpirvLang },
    /// A rebuilt node was missing a result type.
    #[error("optimized node {node:?} is missing result type for rebuild")]
    MissingResultType { node: SpirvLang },
    /// An operand is not dominated by a prior definition when dominance is enforced.
    #[error("operand id {id} for {opcode:?} is not dominated in the linear stream")]
    UndominatedOperand { id: u32, opcode: Op },
}

/// Extract integer type widths from a parsed module.
pub fn type_widths_from_module(module: &rspirv::dr::Module) -> HashMap<Word, u32> {
    module
        .types_global_values
        .iter()
        .filter_map(|inst| match inst.class.opcode {
            Op::TypeInt => inst.result_id.and_then(|id| {
                inst.operands.first().and_then(|op| match op {
                    rspirv::dr::Operand::LiteralBit32(bits) => Some((id, *bits)),
                    _ => None,
                })
            }),
            Op::TypeBool => inst.result_id.map(|id| (id, 1)),
            _ => None,
        })
        .collect()
}

/// The result of translating a linear block of arithmetic SPIR-V instructions.
#[derive(Debug, PartialEq, Eq)]
pub struct TranslatedExpr {
    /// The e-graph friendly expression built from the instructions.
    pub expr: RecExpr<SpirvLang>,
    /// The root e-class id corresponding to the last instruction's result.
    pub root: Id,
    /// The result type id corresponding to the root (if any).
    pub result_type: Option<Word>,
    /// The SPIR-V result id associated with the root.
    pub result_id: Option<Word>,
    /// Mapping from expression node index to original result ids (when available).
    pub original_ids: Vec<Option<Word>>,
    /// Width hints for symbolic operands derived from type information.
    pub symbol_widths: HashMap<Symbol, u8>,
    /// Result type ids per node (when available).
    pub node_types: Vec<Option<Word>>,
}

fn intern_operand(
    operand_id: Word,
    ids: &mut HashMap<Word, Id>,
    expr: &mut RecExpr<SpirvLang>,
    node_to_id: &mut Vec<Option<Word>>,
    node_types: &mut Vec<Option<Word>>,
    symbol_widths: &mut HashMap<Symbol, u8>,
    id_widths: &HashMap<Word, u8>,
) -> Id {
    if let Some(existing) = ids.get(&operand_id) {
        return *existing;
    }
    let sym = make_symbol(operand_id);
    let sym_id = expr.add(SpirvLang::Symbol(sym));
    if let Some(width) = id_widths.get(&operand_id) {
        symbol_widths.insert(sym, *width);
    }
    ids.insert(operand_id, sym_id);
    node_to_id.push(Some(operand_id));
    node_types.push(None);
    sym_id
}

fn symbol_id(sym: &egg::Symbol) -> Option<Word> {
    sym.as_str().strip_prefix("id")?.parse().ok()
}

fn make_symbol(id: Word) -> Symbol {
    Symbol::from(format!("id{id}"))
}

/// Translate a sequence of arithmetic instructions into an e-graph expression.
///
/// Supported ops:
/// - `OpConstant` (32/64-bit), `OpConstantTrue/False/Null`
/// - Integer arithmetic/bitwise ops (`OpIAdd`, `OpIMul`, `OpISub`, `OpSRem`, `OpSMod`, `OpUMod`, shifts, bitwise, `OpBitReverse`)
/// - Integer comparisons (`OpIEqual`, `OpINotEqual`, `OpSLessThan`, etc.)
/// - Logical ops (`OpLogicalNot`, `OpLogicalAnd`, `OpLogicalOr`, logical eq/ne)
/// - `OpSelect`, `OpCopyObject`, and unary `OpSNegate`
///
/// The caller is responsible for providing instructions in dominance order so
/// operands are defined before use. Type declarations are ignored; only value
/// instructions are expected.
pub fn translate_arith(instructions: &[Instruction]) -> Result<TranslatedExpr, TranslateError> {
    translate_arith_with_types(instructions, &HashMap::new())
}

pub fn translate_arith_with_types(
    instructions: &[Instruction],
    type_widths: &HashMap<Word, u32>,
) -> Result<TranslatedExpr, TranslateError> {
    let mut expr = RecExpr::default();
    let mut ids: HashMap<Word, Id> = HashMap::new();
    let mut root = None;
    let mut root_type = None;
    let mut root_id = None;
    let mut node_to_id = Vec::new();
    let mut symbol_widths: HashMap<Symbol, u8> = HashMap::new();
    let mut node_types: Vec<Option<Word>> = Vec::new();

    let mut type_widths: HashMap<Word, u32> = type_widths.clone();
    for inst in instructions {
        match inst.class.opcode {
            Op::TypeInt => {
                if let (Some(result_id), Some(rspirv::dr::Operand::LiteralBit32(bits))) =
                    (inst.result_id, inst.operands.first())
                {
                    type_widths.entry(result_id).or_insert(*bits);
                }
            }
            Op::TypeBool => {
                if let Some(result_id) = inst.result_id {
                    type_widths.entry(result_id).or_insert(1);
                }
            }
            _ => {}
        }
    }

    let mut id_widths: HashMap<Word, u8> = HashMap::new();
    for inst in instructions {
        let Some(result_id) = inst.result_id else {
            continue;
        };
        let opcode = inst.class.opcode;
        if matches!(opcode, Op::TypeInt | Op::TypeBool) {
            continue;
        }
        let literal_width = if opcode == Op::Constant {
            inst.operands.iter().find_map(|op| match op {
                rspirv::dr::Operand::LiteralBit64(_) => Some(64),
                rspirv::dr::Operand::LiteralBit32(_) => Some(32),
                _ => None,
            })
        } else {
            None
        };
        let width_bits = inst
            .result_type
            .and_then(|ty| type_widths.get(&ty).copied())
            .or(literal_width)
            .unwrap_or(32);
        let width = width_bits.min(64) as u8;
        id_widths.entry(result_id).or_insert(width);
        match opcode {
            Op::Select => {
                if let Some(cond_id) = inst.operands.first().and_then(|op| op.id_ref_any()) {
                    id_widths.entry(cond_id).or_insert(1);
                }
                for operand in inst
                    .operands
                    .iter()
                    .skip(1)
                    .filter_map(|op| op.id_ref_any())
                {
                    id_widths.entry(operand).or_insert(width);
                }
            }
            Op::LogicalNot
            | Op::LogicalAnd
            | Op::LogicalOr
            | Op::LogicalEqual
            | Op::LogicalNotEqual => {
                for operand in inst.operands.iter().filter_map(|op| op.id_ref_any()) {
                    id_widths.entry(operand).or_insert(1);
                }
            }
            Op::IEqual
            | Op::INotEqual
            | Op::SLessThan
            | Op::SLessThanEqual
            | Op::SGreaterThan
            | Op::SGreaterThanEqual
            | Op::ULessThan
            | Op::ULessThanEqual
            | Op::UGreaterThan
            | Op::UGreaterThanEqual => {}
            _ => {
                for operand in inst.operands.iter().filter_map(|op| op.id_ref_any()) {
                    id_widths.entry(operand).or_insert(width);
                }
            }
        }
    }

    let defined_ids: HashSet<Word> = instructions
        .iter()
        .filter_map(|inst| inst.result_id)
        .collect();
    for inst in instructions {
        for operand in inst.operands.iter().filter_map(|op| op.id_ref_any()) {
            if !defined_ids.contains(&operand) && !ids.contains_key(&operand) {
                let _ = intern_operand(
                    operand,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
            }
        }
    }

    for inst in instructions {
        let opcode = inst.class.opcode;
        if opcode == Op::TypeInt {
            continue;
        }
        let result_id = inst
            .result_id
            .ok_or(TranslateError::MissingResultId(opcode))?;
        let result_width = id_widths.get(&result_id).copied();
        let node_id = match opcode {
            Op::Constant => {
                let (value, literal_width) = inst
                    .operands
                    .iter()
                    .find_map(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some((*v as u64, 32u8)),
                        rspirv::dr::Operand::LiteralBit64(v) => Some((*v, 64u8)),
                        _ => None,
                    })
                    .ok_or(TranslateError::InvalidConstant { id: result_id })?;
                let width = result_width.unwrap_or(literal_width);
                expr.add(SpirvLang::Const(ConstValue::new_with_width(value, width)))
            }
            Op::ConstantTrue => {
                let width = result_width.unwrap_or(1);
                expr.add(SpirvLang::Const(ConstValue::new_with_width(1, width)))
            }
            Op::ConstantFalse => {
                let width = result_width.unwrap_or(1);
                expr.add(SpirvLang::Const(ConstValue::new_with_width(0, width)))
            }
            Op::ConstantNull => {
                let width = result_width.unwrap_or(32);
                expr.add(SpirvLang::Const(ConstValue::new_with_width(0, width)))
            }
            Op::IAdd => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::Add([lhs, rhs]))
            }
            Op::IMul => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::Mul([lhs, rhs]))
            }
            Op::ISub => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::Sub([lhs, rhs]))
            }
            Op::SNegate => {
                let operand = inst
                    .operands
                    .iter()
                    .find_map(|op| op.id_ref_any())
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let id = intern_operand(
                    operand,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::Neg(id))
            }
            Op::SDiv | Op::UDiv => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let node = if opcode == Op::SDiv {
                    SpirvLang::SDiv([lhs, rhs])
                } else {
                    SpirvLang::UDiv([lhs, rhs])
                };
                expr.add(node)
            }
            Op::Select => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let cond_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let true_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let false_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                id_widths.entry(cond_id).or_insert(1);
                symbol_widths.entry(make_symbol(cond_id)).or_insert(1);
                let cond = intern_operand(
                    cond_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let true_val = intern_operand(
                    true_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let false_val = intern_operand(
                    false_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::Select([cond, true_val, false_val]))
            }
            Op::IEqual
            | Op::INotEqual
            | Op::SLessThan
            | Op::SLessThanEqual
            | Op::SGreaterThan
            | Op::SGreaterThanEqual
            | Op::ULessThan
            | Op::ULessThanEqual
            | Op::UGreaterThan
            | Op::UGreaterThanEqual => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let node = match opcode {
                    Op::IEqual => SpirvLang::Eq([lhs, rhs]),
                    Op::INotEqual => SpirvLang::Ne([lhs, rhs]),
                    Op::SLessThan => SpirvLang::SLt([lhs, rhs]),
                    Op::SLessThanEqual => SpirvLang::SLe([lhs, rhs]),
                    Op::SGreaterThan => SpirvLang::SGt([lhs, rhs]),
                    Op::SGreaterThanEqual => SpirvLang::SGe([lhs, rhs]),
                    Op::ULessThan => SpirvLang::ULt([lhs, rhs]),
                    Op::ULessThanEqual => SpirvLang::ULe([lhs, rhs]),
                    Op::UGreaterThan => SpirvLang::UGt([lhs, rhs]),
                    Op::UGreaterThanEqual => SpirvLang::UGe([lhs, rhs]),
                    _ => return Err(TranslateError::UnsupportedOp(opcode)),
                };
                expr.add(node)
            }
            Op::LogicalNot => {
                let operand = inst
                    .operands
                    .iter()
                    .find_map(|op| op.id_ref_any())
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let id = intern_operand(
                    operand,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::LogNot(id))
            }
            Op::LogicalAnd | Op::LogicalOr | Op::LogicalEqual | Op::LogicalNotEqual => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let node = match opcode {
                    Op::LogicalAnd => SpirvLang::LogAnd([lhs, rhs]),
                    Op::LogicalOr => SpirvLang::LogOr([lhs, rhs]),
                    Op::LogicalEqual => SpirvLang::LogEq([lhs, rhs]),
                    Op::LogicalNotEqual => SpirvLang::LogNe([lhs, rhs]),
                    _ => return Err(TranslateError::UnsupportedOp(opcode)),
                };
                expr.add(node)
            }
            Op::BitwiseAnd => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::BitAnd([lhs, rhs]))
            }
            Op::SRem | Op::SMod | Op::UMod => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let node = match opcode {
                    Op::SRem => SpirvLang::SRem([lhs, rhs]),
                    Op::SMod => SpirvLang::SMod([lhs, rhs]),
                    Op::UMod => SpirvLang::UMod([lhs, rhs]),
                    _ => return Err(TranslateError::UnsupportedOp(opcode)),
                };
                expr.add(node)
            }
            Op::ShiftLeftLogical | Op::ShiftRightLogical | Op::ShiftRightArithmetic => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let node = match opcode {
                    Op::ShiftLeftLogical => SpirvLang::Shl([lhs, rhs]),
                    Op::ShiftRightLogical => SpirvLang::ShrU([lhs, rhs]),
                    _ => SpirvLang::ShrS([lhs, rhs]),
                };
                expr.add(node)
            }
            Op::BitwiseOr | Op::BitwiseXor => {
                let mut ops = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .peekable();
                let lhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let rhs_id = ops
                    .next()
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let lhs = intern_operand(
                    lhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let rhs = intern_operand(
                    rhs_id,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                let node = if opcode == Op::BitwiseOr {
                    SpirvLang::BitOr([lhs, rhs])
                } else {
                    SpirvLang::BitXor([lhs, rhs])
                };
                expr.add(node)
            }
            Op::Not => {
                let operand = inst
                    .operands
                    .iter()
                    .find_map(|op| op.id_ref_any())
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let id = intern_operand(
                    operand,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::BitNot(id))
            }
            Op::BitReverse => {
                let operand = inst
                    .operands
                    .iter()
                    .find_map(|op| op.id_ref_any())
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let id = intern_operand(
                    operand,
                    &mut ids,
                    &mut expr,
                    &mut node_to_id,
                    &mut node_types,
                    &mut symbol_widths,
                    &id_widths,
                );
                expr.add(SpirvLang::BitReverse(id))
            }
            Op::CopyObject => {
                let operand = inst
                    .operands
                    .iter()
                    .find_map(|op| op.id_ref_any())
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let sym = make_symbol(operand);
                if let Some(width) = id_widths.get(&operand) {
                    symbol_widths.insert(sym, *width);
                }
                expr.add(SpirvLang::Symbol(sym))
            }
            other => return Err(TranslateError::UnsupportedOp(other)),
        };
        ids.insert(result_id, node_id);
        root = Some(node_id);
        root_type = inst.result_type;
        root_id = Some(result_id);
        node_to_id.push(Some(result_id));
        node_types.push(inst.result_type);
    }

    let root = root.unwrap_or_else(|| Id::from(0));
    Ok(TranslatedExpr {
        expr,
        root,
        result_type: root_type,
        result_id: root_id,
        original_ids: node_to_id,
        symbol_widths,
        node_types,
    })
}

/// Translate while enforcing that operands are defined earlier in the linear stream.
pub fn translate_arith_with_types_dominated(
    instructions: &[Instruction],
    type_widths: &HashMap<Word, u32>,
) -> Result<TranslatedExpr, TranslateError> {
    check_linear_dominance(instructions)?;
    translate_arith_with_types(instructions, type_widths)
}

fn check_linear_dominance(instructions: &[Instruction]) -> Result<(), TranslateError> {
    let defined: HashSet<Word> = instructions
        .iter()
        .filter_map(|inst| inst.result_id)
        .collect();
    let mut seen: HashSet<Word> = HashSet::new();
    for inst in instructions {
        let opcode = inst.class.opcode;
        if matches!(opcode, Op::TypeInt | Op::TypeBool) {
            if let Some(id) = inst.result_id {
                seen.insert(id);
            }
            continue;
        }
        for operand in inst.operands.iter().filter_map(|op| op.id_ref_any()) {
            if defined.contains(&operand) && !seen.contains(&operand) {
                return Err(TranslateError::UndominatedOperand {
                    id: operand,
                    opcode,
                });
            }
        }
        if let Some(id) = inst.result_id {
            seen.insert(id);
        }
    }
    Ok(())
}

/// Optimize a straight-line arithmetic block; if it reduces to a constant,
/// return a single `OpConstant` instruction with the original root id/type.
pub fn optimize_arith_block(
    instructions: &[Instruction],
) -> Result<Vec<Instruction>, TranslateError> {
    optimize_arith_block_with_types(instructions, &HashMap::new())
}

/// Rebuild a stream of arithmetic instructions from an optimized expression, reusing
/// original result ids/types. Fails if the optimized expression contains nodes without
/// corresponding original ids or result types.
pub fn rebuild_arith_with_original_ids(
    optimized: &RecExpr<SpirvLang>,
    translated: &TranslatedExpr,
) -> Result<Vec<Instruction>, TranslateError> {
    let mut assigned_ids = Vec::with_capacity(optimized.as_ref().len());
    let mut assigned_types = Vec::with_capacity(optimized.as_ref().len());
    for (idx, _node) in optimized.as_ref().iter().enumerate() {
        let Some(id) = translated.original_ids.get(idx).and_then(|id| *id) else {
            return Err(TranslateError::MissingOriginalId {
                node: optimized.as_ref()[idx].clone(),
            });
        };
        let Some(ty) = translated.node_types.get(idx).and_then(|t| *t) else {
            return Err(TranslateError::MissingResultType {
                node: optimized.as_ref()[idx].clone(),
            });
        };
        assigned_ids.push(id);
        assigned_types.push(ty);
    }

    let mut out = Vec::new();
    for (idx, node) in optimized.as_ref().iter().enumerate() {
        let result_id = assigned_ids[idx];
        let result_type = assigned_types[idx];
        let inst = match node {
            SpirvLang::Const(val) => {
                if val.width_bits() == 1 {
                    let opcode = if val.get_u64() == 0 {
                        Op::ConstantFalse
                    } else {
                        Op::ConstantTrue
                    };
                    Instruction::new(opcode, Some(result_type), Some(result_id), Vec::new())
                } else {
                    let operands = match val.width_bits() {
                        64 => vec![rspirv::dr::Operand::LiteralBit64(val.get_u64())],
                        _ => vec![rspirv::dr::Operand::LiteralBit32(val.get())],
                    };
                    Instruction::new(Op::Constant, Some(result_type), Some(result_id), operands)
                }
            }
            SpirvLang::Add([a, b]) => Instruction::new(
                Op::IAdd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Mul([a, b]) => Instruction::new(
                Op::IMul,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Sub([a, b]) => Instruction::new(
                Op::ISub,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SDiv([a, b]) => Instruction::new(
                Op::SDiv,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UDiv([a, b]) => Instruction::new(
                Op::UDiv,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SRem([a, b]) => Instruction::new(
                Op::SRem,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SMod([a, b]) => Instruction::new(
                Op::SMod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UMod([a, b]) => Instruction::new(
                Op::UMod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Shl([a, b]) => Instruction::new(
                Op::ShiftLeftLogical,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ShrS([a, b]) => Instruction::new(
                Op::ShiftRightArithmetic,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ShrU([a, b]) => Instruction::new(
                Op::ShiftRightLogical,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Eq([a, b]) => Instruction::new(
                Op::IEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Ne([a, b]) => Instruction::new(
                Op::INotEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SLt([a, b]) => Instruction::new(
                Op::SLessThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SLe([a, b]) => Instruction::new(
                Op::SLessThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SGt([a, b]) => Instruction::new(
                Op::SGreaterThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SGe([a, b]) => Instruction::new(
                Op::SGreaterThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ULt([a, b]) => Instruction::new(
                Op::ULessThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ULe([a, b]) => Instruction::new(
                Op::ULessThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UGt([a, b]) => Instruction::new(
                Op::UGreaterThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UGe([a, b]) => Instruction::new(
                Op::UGreaterThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogNot(a) => Instruction::new(
                Op::LogicalNot,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)])],
            ),
            SpirvLang::LogAnd([a, b]) => Instruction::new(
                Op::LogicalAnd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogOr([a, b]) => Instruction::new(
                Op::LogicalOr,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogEq([a, b]) => Instruction::new(
                Op::LogicalEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogNe([a, b]) => Instruction::new(
                Op::LogicalNotEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitAnd([a, b]) => Instruction::new(
                Op::BitwiseAnd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitOr([a, b]) => Instruction::new(
                Op::BitwiseOr,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitXor([a, b]) => Instruction::new(
                Op::BitwiseXor,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitNot(a) => Instruction::new(
                Op::Not,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)])],
            ),
            SpirvLang::BitReverse(a) => Instruction::new(
                Op::BitReverse,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)])],
            ),
            SpirvLang::Neg(a) => Instruction::new(
                Op::SNegate,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)])],
            ),
            SpirvLang::Select([c, t, f]) => Instruction::new(
                Op::Select,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*c)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*t)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*f)]),
                ],
            ),
            SpirvLang::Symbol(sym) => Instruction::new(
                Op::CopyObject,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(
                    symbol_id(sym).expect("symbol id"),
                )],
            ),
            SpirvLang::RotL(_)
            | SpirvLang::RotR(_)
            | SpirvLang::If(_)
            | SpirvLang::Merge(_)
            | SpirvLang::Ret
            | SpirvLang::RetVal(_)
            | SpirvLang::Phi(_)
            | SpirvLang::Pair(_) => continue,
        };
        out.push(inst);
    }

    Ok(out)
}

pub fn optimize_arith_block_with_types(
    instructions: &[Instruction],
    type_widths: &HashMap<Word, u32>,
) -> Result<Vec<Instruction>, TranslateError> {
    let translated = translate_arith_with_types(instructions, type_widths)?;
    let optimized = crate::optimize_translated(&translated);
    if optimized == translated.expr {
        return Ok(instructions.to_vec());
    }
    let Some(result_type) = translated.result_type else {
        return Ok(instructions.to_vec());
    };
    let Some(root_id) = translated.result_id else {
        return Ok(instructions.to_vec());
    };

    if optimized.as_ref().is_empty() {
        return Ok(instructions.to_vec());
    }

    let original_cost = expr_cost(&translated.expr);
    let optimized_cost = expr_cost(&optimized);
    if optimized_cost >= original_cost {
        return Ok(instructions.to_vec());
    }

    // Assign result ids to the optimized expression, preferring original ids when available.
    let mut available_ids: Vec<_> = translated
        .original_ids
        .iter()
        .flatten()
        .copied()
        .filter(|id| Some(*id) != translated.result_id)
        .collect();
    available_ids.sort_unstable();
    available_ids.dedup();
    let reserved_symbol_ids: HashSet<_> = optimized
        .as_ref()
        .iter()
        .filter_map(|node| match node {
            SpirvLang::Symbol(sym) => symbol_id(sym),
            _ => None,
        })
        .collect();
    available_ids.retain(|id| !reserved_symbol_ids.contains(id));
    let mut assigned_ids = Vec::with_capacity(optimized.as_ref().len());
    let mut pool_iter = available_ids.into_iter();
    for (idx, node) in optimized.as_ref().iter().enumerate() {
        let is_root = idx == optimized.as_ref().len() - 1;
        if is_root {
            assigned_ids.push(root_id);
            continue;
        }
        if let SpirvLang::Symbol(sym) = node {
            if let Some(id) = symbol_id(sym) {
                assigned_ids.push(id);
                continue;
            }
        }
        let next_id = pool_iter
            .next()
            .unwrap_or_else(|| root_id + (idx as u32) + 1);
        assigned_ids.push(next_id);
        let _ = node;
    }

    let mut output = Vec::with_capacity(optimized.as_ref().len());
    for (idx, node) in optimized.as_ref().iter().enumerate() {
        let result_id = assigned_ids[idx];
        let inst = match node {
            SpirvLang::Const(val) => {
                if val.width_bits() == 1 {
                    let opcode = if val.get_u64() == 0 {
                        Op::ConstantFalse
                    } else {
                        Op::ConstantTrue
                    };
                    Instruction::new(opcode, Some(result_type), Some(result_id), Vec::new())
                } else {
                    let operands = match val.width_bits() {
                        64 => vec![rspirv::dr::Operand::LiteralBit64(val.get_u64())],
                        _ => vec![rspirv::dr::Operand::LiteralBit32(val.get())],
                    };
                    Instruction::new(Op::Constant, Some(result_type), Some(result_id), operands)
                }
            }
            SpirvLang::Add([a, b]) => Instruction::new(
                Op::IAdd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Mul([a, b]) => Instruction::new(
                Op::IMul,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Sub([a, b]) => Instruction::new(
                Op::ISub,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SDiv([a, b]) => Instruction::new(
                Op::SDiv,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UDiv([a, b]) => Instruction::new(
                Op::UDiv,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SRem([a, b]) => Instruction::new(
                Op::SRem,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SMod([a, b]) => Instruction::new(
                Op::SMod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UMod([a, b]) => Instruction::new(
                Op::UMod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitAnd([a, b]) => Instruction::new(
                Op::BitwiseAnd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitOr([a, b]) => Instruction::new(
                Op::BitwiseOr,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitXor([a, b]) => Instruction::new(
                Op::BitwiseXor,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::BitNot(x) => Instruction::new(
                Op::Not,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*x)])],
            ),
            SpirvLang::BitReverse(x) => Instruction::new(
                Op::BitReverse,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*x)])],
            ),
            SpirvLang::Select([c, t, f]) => Instruction::new(
                Op::Select,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*c)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*t)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*f)]),
                ],
            ),
            SpirvLang::RotL(_)
            | SpirvLang::RotR(_)
            | SpirvLang::If(_)
            | SpirvLang::Merge(_)
            | SpirvLang::Ret
            | SpirvLang::RetVal(_)
            | SpirvLang::Phi(_)
            | SpirvLang::Pair(_) => continue,
            SpirvLang::Shl([a, b]) => Instruction::new(
                Op::ShiftLeftLogical,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ShrS([a, b]) => Instruction::new(
                Op::ShiftRightArithmetic,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ShrU([a, b]) => Instruction::new(
                Op::ShiftRightLogical,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Eq([a, b]) => Instruction::new(
                Op::IEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Ne([a, b]) => Instruction::new(
                Op::INotEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SLt([a, b]) => Instruction::new(
                Op::SLessThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SLe([a, b]) => Instruction::new(
                Op::SLessThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SGt([a, b]) => Instruction::new(
                Op::SGreaterThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::SGe([a, b]) => Instruction::new(
                Op::SGreaterThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ULt([a, b]) => Instruction::new(
                Op::ULessThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::ULe([a, b]) => Instruction::new(
                Op::ULessThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UGt([a, b]) => Instruction::new(
                Op::UGreaterThan,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::UGe([a, b]) => Instruction::new(
                Op::UGreaterThanEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogNot(a) => Instruction::new(
                Op::LogicalNot,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)])],
            ),
            SpirvLang::LogAnd([a, b]) => Instruction::new(
                Op::LogicalAnd,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogOr([a, b]) => Instruction::new(
                Op::LogicalOr,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogEq([a, b]) => Instruction::new(
                Op::LogicalEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::LogNe([a, b]) => Instruction::new(
                Op::LogicalNotEqual,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)]),
                    rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*b)]),
                ],
            ),
            SpirvLang::Neg(a) => Instruction::new(
                Op::SNegate,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)])],
            ),
            SpirvLang::Symbol(sym) => Instruction::new(
                Op::CopyObject,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(
                    symbol_id(sym).expect("symbol id"),
                )],
            ),
        };
        output.push(inst);
    }

    Ok(output)
}

fn expr_cost(expr: &RecExpr<SpirvLang>) -> usize {
    let mut costs = Vec::with_capacity(expr.as_ref().len());
    for node in expr.as_ref().iter() {
        let cost = match node {
            SpirvLang::Const(_) | SpirvLang::Symbol(_) => 1,
            SpirvLang::Neg(a) => 1 + costs[usize::from(*a)],
            SpirvLang::LogNot(a) => 1 + costs[usize::from(*a)],
            SpirvLang::Mul([a, b]) => 2 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::SDiv([a, b]) | SpirvLang::UDiv([a, b]) => {
                8 + costs[usize::from(*a)] + costs[usize::from(*b)]
            }
            SpirvLang::Add([a, b])
            | SpirvLang::Sub([a, b])
            | SpirvLang::SRem([a, b])
            | SpirvLang::SMod([a, b])
            | SpirvLang::UMod([a, b])
            | SpirvLang::Shl([a, b])
            | SpirvLang::ShrS([a, b])
            | SpirvLang::ShrU([a, b])
            | SpirvLang::BitOr([a, b])
            | SpirvLang::BitXor([a, b])
            | SpirvLang::Eq([a, b])
            | SpirvLang::Ne([a, b])
            | SpirvLang::SLt([a, b])
            | SpirvLang::SLe([a, b])
            | SpirvLang::SGt([a, b])
            | SpirvLang::SGe([a, b])
            | SpirvLang::ULt([a, b])
            | SpirvLang::ULe([a, b])
            | SpirvLang::UGt([a, b])
            | SpirvLang::UGe([a, b])
            | SpirvLang::LogAnd([a, b])
            | SpirvLang::LogOr([a, b])
            | SpirvLang::LogEq([a, b])
            | SpirvLang::LogNe([a, b])
            | SpirvLang::RotL([a, b])
            | SpirvLang::RotR([a, b]) => 1 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::BitAnd([a, b]) => 2 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::BitNot(x) | SpirvLang::BitReverse(x) => 1 + costs[usize::from(*x)],
            SpirvLang::If([a, b, c]) => {
                1 + costs[usize::from(*a)] + costs[usize::from(*b)] + costs[usize::from(*c)]
            }
            SpirvLang::Select([a, b, c]) => {
                1 + costs[usize::from(*a)] + costs[usize::from(*b)] + costs[usize::from(*c)]
            }
            SpirvLang::Merge([a, b]) => 1 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::Ret => 1,
            SpirvLang::RetVal(a) => 1 + costs[usize::from(*a)],
            SpirvLang::Phi([a, b]) => 1 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::Pair([a, b]) => 1 + costs[usize::from(*a)] + costs[usize::from(*b)],
        };
        costs.push(cost);
    }
    costs.last().copied().unwrap_or(0)
}
