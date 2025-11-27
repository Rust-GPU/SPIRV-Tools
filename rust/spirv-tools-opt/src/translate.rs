use crate::{ConstValue, SpirvLang};
use egg::{Id, RecExpr};
use rspirv::dr::Instruction;
use rspirv::spirv::Op;
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
    pub result_type: Option<u32>,
    /// The SPIR-V result id associated with the root.
    pub result_id: Option<u32>,
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
    let mut ids: HashMap<u32, Id> = HashMap::new();
    let mut root = None;
    let mut root_type = None;
    let mut root_id = None;

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
            other => return Err(TranslateError::UnsupportedOp(other)),
        };
        ids.insert(result_id, node_id);
        root = Some(node_id);
        root_type = inst.result_type;
        root_id = Some(result_id);
    }

    let root = root.unwrap_or_else(|| Id::from(0));
    Ok(TranslatedExpr {
        expr,
        root,
        result_type: root_type,
        result_id: root_id,
    })
}

/// Optimize a straight-line arithmetic block; if it reduces to a constant,
/// return a single `OpConstant` instruction with the original root id/type.
pub fn optimize_arith_block(
    instructions: &[Instruction],
) -> Result<Vec<Instruction>, TranslateError> {
    let translated = translate_arith(instructions)?;
    let optimized = crate::optimize_translated(&translated);
    if optimized.as_ref().len() == 1 {
        if let SpirvLang::Const(ConstValue(value)) = &optimized.as_ref()[0] {
            let result_id = translated
                .result_id
                .ok_or(TranslateError::MissingResultId(Op::Constant))?;
            let result_type = translated
                .result_type
                .ok_or(TranslateError::UnsupportedOp(Op::Constant))?;
            let inst = Instruction::new(
                Op::Constant,
                Some(result_type),
                Some(result_id),
                vec![rspirv::dr::Operand::LiteralBit32(*value)],
            );
            return Ok(vec![inst]);
        }
    }
    Ok(instructions.to_vec())
}
