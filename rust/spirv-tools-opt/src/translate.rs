use crate::{ConstValue, SpirvLang};
use egg::{Id, RecExpr};
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
}

fn intern_operand(
    operand_id: Word,
    ids: &mut HashMap<Word, Id>,
    expr: &mut RecExpr<SpirvLang>,
    node_to_id: &mut Vec<Option<Word>>,
) -> Id {
    if let Some(existing) = ids.get(&operand_id) {
        return *existing;
    }
    let sym = expr.add(SpirvLang::Symbol(egg::Symbol::from(format!(
        "id{operand_id}"
    ))));
    ids.insert(operand_id, sym);
    node_to_id.push(Some(operand_id));
    sym
}

fn symbol_id(sym: &egg::Symbol) -> Option<Word> {
    sym.as_str().strip_prefix("id")?.parse().ok()
}

/// Translate a sequence of arithmetic instructions into an e-graph expression.
///
/// Supported ops:
/// - `OpConstant` with a single 32-bit literal (treated as unsigned for folding)
/// - `OpIAdd` and `OpIMul` with id operands
///
/// The caller is responsible for providing instructions in dominance order so
/// operands are defined before use. Type declarations are ignored; only value
/// instructions are expected.
pub fn translate_arith(instructions: &[Instruction]) -> Result<TranslatedExpr, TranslateError> {
    let mut expr = RecExpr::default();
    let mut ids: HashMap<Word, Id> = HashMap::new();
    let mut root = None;
    let mut root_type = None;
    let mut root_id = None;
    let mut node_to_id = Vec::new();

    let defined_ids: HashSet<Word> = instructions
        .iter()
        .filter_map(|inst| inst.result_id)
        .collect();
    for inst in instructions {
        for operand in inst.operands.iter().filter_map(|op| op.id_ref_any()) {
            if !defined_ids.contains(&operand) && !ids.contains_key(&operand) {
                let sym = expr.add(SpirvLang::Symbol(egg::Symbol::from(format!("id{operand}"))));
                ids.insert(operand, sym);
                node_to_id.push(Some(operand));
            }
        }
    }

    for inst in instructions {
        let opcode = inst.class.opcode;
        let result_id = inst
            .result_id
            .ok_or(TranslateError::MissingResultId(opcode))?;
        let node_id = match opcode {
            Op::Constant => {
                let literal = inst
                    .operands
                    .iter()
                    .find_map(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => {
                            Some(ConstValue::new_with_width(*v as u64, 32))
                        }
                        rspirv::dr::Operand::LiteralBit64(v) => {
                            Some(ConstValue::new_with_width(*v, 64))
                        }
                        _ => None,
                    })
                    .ok_or(TranslateError::InvalidConstant { id: result_id })?;
                expr.add(SpirvLang::Const(literal))
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
                let lhs = intern_operand(lhs_id, &mut ids, &mut expr, &mut node_to_id);
                let rhs = intern_operand(rhs_id, &mut ids, &mut expr, &mut node_to_id);
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
                let lhs = intern_operand(lhs_id, &mut ids, &mut expr, &mut node_to_id);
                let rhs = intern_operand(rhs_id, &mut ids, &mut expr, &mut node_to_id);
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
                let lhs = intern_operand(lhs_id, &mut ids, &mut expr, &mut node_to_id);
                let rhs = intern_operand(rhs_id, &mut ids, &mut expr, &mut node_to_id);
                expr.add(SpirvLang::Sub([lhs, rhs]))
            }
            Op::SNegate => {
                let operand = inst
                    .operands
                    .iter()
                    .find_map(|op| op.id_ref_any())
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let id = intern_operand(operand, &mut ids, &mut expr, &mut node_to_id);
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
                let lhs = intern_operand(lhs_id, &mut ids, &mut expr, &mut node_to_id);
                let rhs = intern_operand(rhs_id, &mut ids, &mut expr, &mut node_to_id);
                let node = if opcode == Op::SDiv {
                    SpirvLang::SDiv([lhs, rhs])
                } else {
                    SpirvLang::UDiv([lhs, rhs])
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
                let lhs = intern_operand(lhs_id, &mut ids, &mut expr, &mut node_to_id);
                let rhs = intern_operand(rhs_id, &mut ids, &mut expr, &mut node_to_id);
                expr.add(SpirvLang::BitAnd([lhs, rhs]))
            }
            Op::SRem | Op::UMod => {
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
                let lhs = intern_operand(lhs_id, &mut ids, &mut expr, &mut node_to_id);
                let rhs = intern_operand(rhs_id, &mut ids, &mut expr, &mut node_to_id);
                let node = if opcode == Op::SRem {
                    SpirvLang::SRem([lhs, rhs])
                } else {
                    SpirvLang::UMod([lhs, rhs])
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
                let lhs = intern_operand(lhs_id, &mut ids, &mut expr, &mut node_to_id);
                let rhs = intern_operand(rhs_id, &mut ids, &mut expr, &mut node_to_id);
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
                let lhs = intern_operand(lhs_id, &mut ids, &mut expr, &mut node_to_id);
                let rhs = intern_operand(rhs_id, &mut ids, &mut expr, &mut node_to_id);
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
                let id = intern_operand(operand, &mut ids, &mut expr, &mut node_to_id);
                expr.add(SpirvLang::BitNot(id))
            }
            other => return Err(TranslateError::UnsupportedOp(other)),
        };
        ids.insert(result_id, node_id);
        root = Some(node_id);
        root_type = inst.result_type;
        root_id = Some(result_id);
        node_to_id.push(Some(result_id));
    }

    let root = root.unwrap_or_else(|| Id::from(0));
    Ok(TranslatedExpr {
        expr,
        root,
        result_type: root_type,
        result_id: root_id,
        original_ids: node_to_id,
    })
}

/// Optimize a straight-line arithmetic block; if it reduces to a constant,
/// return a single `OpConstant` instruction with the original root id/type.
pub fn optimize_arith_block(
    instructions: &[Instruction],
) -> Result<Vec<Instruction>, TranslateError> {
    let translated = translate_arith(instructions)?;
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
                let operands = match val.width_bits() {
                    64 => vec![rspirv::dr::Operand::LiteralBit64(val.get_u64())],
                    _ => vec![rspirv::dr::Operand::LiteralBit32(val.get())],
                };
                Instruction::new(Op::Constant, Some(result_type), Some(result_id), operands)
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
            SpirvLang::RotL(_) | SpirvLang::RotR(_) => continue,
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
            SpirvLang::Neg(a) => Instruction::new(
                Op::SNegate,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::IdRef(assigned_ids[usize::from(*a)])],
            ),
            SpirvLang::Symbol(_) => continue,
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
            SpirvLang::Mul([a, b]) => 2 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::SDiv([a, b]) | SpirvLang::UDiv([a, b]) => {
                8 + costs[usize::from(*a)] + costs[usize::from(*b)]
            }
            SpirvLang::Add([a, b])
            | SpirvLang::Sub([a, b])
            | SpirvLang::SRem([a, b])
            | SpirvLang::UMod([a, b])
            | SpirvLang::Shl([a, b])
            | SpirvLang::ShrS([a, b])
            | SpirvLang::ShrU([a, b])
            | SpirvLang::BitOr([a, b])
            | SpirvLang::BitXor([a, b])
            | SpirvLang::RotL([a, b])
            | SpirvLang::RotR([a, b]) => 1 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::BitAnd([a, b]) => 2 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::BitNot(x) => 1 + costs[usize::from(*x)],
        };
        costs.push(cost);
    }
    costs.last().copied().unwrap_or(0)
}
