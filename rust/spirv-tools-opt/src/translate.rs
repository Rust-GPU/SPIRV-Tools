use crate::{ConstValue, SpirvLang};
use egg::{Id, RecExpr};
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use rspirv::spirv::Word;
use std::collections::HashMap;
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
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .ok_or(TranslateError::InvalidConstant { id: result_id })?;
                expr.add(SpirvLang::Const(ConstValue::new(literal)))
            }
            Op::IAdd => {
                let mut ops = inst.operands.iter().filter_map(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => Some(*id),
                    _ => None,
                });
                let lhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .first()
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let rhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .get(1)
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                expr.add(SpirvLang::Add([lhs, rhs]))
            }
            Op::IMul => {
                let mut ops = inst.operands.iter().filter_map(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => Some(*id),
                    _ => None,
                });
                let lhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .first()
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let rhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .get(1)
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                expr.add(SpirvLang::Mul([lhs, rhs]))
            }
            Op::ISub => {
                let mut ops = inst.operands.iter().filter_map(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => Some(*id),
                    _ => None,
                });
                let lhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .first()
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let rhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .get(1)
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                expr.add(SpirvLang::Sub([lhs, rhs]))
            }
            Op::SNegate => {
                let operand = inst
                    .operands
                    .iter()
                    .find_map(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => Some(*id),
                        _ => None,
                    })
                    .ok_or(TranslateError::UnknownOperand { id: 0, opcode })?;
                let id = ids
                    .get(&operand)
                    .copied()
                    .ok_or(TranslateError::UnknownOperand {
                        id: operand,
                        opcode,
                    })?;
                expr.add(SpirvLang::Neg(id))
            }
            Op::SDiv | Op::UDiv => {
                let mut ops = inst.operands.iter().filter_map(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => Some(*id),
                    _ => None,
                });
                let lhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .first()
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let rhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .get(1)
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let node = if opcode == Op::SDiv {
                    SpirvLang::SDiv([lhs, rhs])
                } else {
                    SpirvLang::UDiv([lhs, rhs])
                };
                expr.add(node)
            }
            Op::BitwiseAnd => {
                let mut ops = inst.operands.iter().filter_map(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => Some(*id),
                    _ => None,
                });
                let lhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .first()
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let rhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .get(1)
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                expr.add(SpirvLang::BitAnd([lhs, rhs]))
            }
            Op::SRem | Op::UMod => {
                let mut ops = inst.operands.iter().filter_map(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => Some(*id),
                    _ => None,
                });
                let lhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .first()
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let rhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .get(1)
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let node = if opcode == Op::SRem {
                    SpirvLang::SRem([lhs, rhs])
                } else {
                    SpirvLang::UMod([lhs, rhs])
                };
                expr.add(node)
            }
            Op::ShiftLeftLogical | Op::ShiftRightLogical | Op::ShiftRightArithmetic => {
                let mut ops = inst.operands.iter().filter_map(|op| match op {
                    rspirv::dr::Operand::IdRef(id) => Some(*id),
                    _ => None,
                });
                let lhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .first()
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let rhs = ops.next().and_then(|id| ids.get(&id).copied()).ok_or(
                    TranslateError::UnknownOperand {
                        id: inst
                            .operands
                            .get(1)
                            .and_then(|op| op.id_ref_any())
                            .unwrap_or(0),
                        opcode,
                    },
                )?;
                let node = match opcode {
                    Op::ShiftLeftLogical => SpirvLang::Shl([lhs, rhs]),
                    Op::ShiftRightLogical => SpirvLang::ShrU([lhs, rhs]),
                    _ => SpirvLang::ShrS([lhs, rhs]),
                };
                expr.add(node)
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
    let mut assigned_ids = Vec::with_capacity(optimized.as_ref().len());
    let mut pool_iter = available_ids.into_iter();
    for (idx, node) in optimized.as_ref().iter().enumerate() {
        let is_root = idx == optimized.as_ref().len() - 1;
        if is_root {
            assigned_ids.push(root_id);
            continue;
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
            SpirvLang::Const(val) => Instruction::new(
                Op::Constant,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::LiteralBit32(val.get())],
            ),
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
            | SpirvLang::ShrU([a, b]) => 1 + costs[usize::from(*a)] + costs[usize::from(*b)],
            SpirvLang::BitAnd([a, b]) => 2 + costs[usize::from(*a)] + costs[usize::from(*b)],
        };
        costs.push(cost);
    }
    costs.last().copied().unwrap_or(0)
}
