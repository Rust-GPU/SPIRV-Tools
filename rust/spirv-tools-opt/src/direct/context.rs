//! Context for building egglog expressions from SPIR-V instructions.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

/// Context for building egglog expressions from SPIR-V instructions.
pub struct EgglogContext {
    /// Maps result_id -> egglog term string
    pub id_to_term: HashMap<Word, String>,
    /// Maps result_id -> result_type
    pub id_to_type: HashMap<Word, Word>,
    /// Type widths
    type_widths: HashMap<Word, u32>,
    /// All root IDs (instructions we're optimizing)
    pub root_ids: Vec<Word>,
}

impl EgglogContext {
    pub fn new(type_widths: &HashMap<Word, u32>) -> Self {
        Self {
            id_to_term: HashMap::new(),
            id_to_type: HashMap::new(),
            type_widths: type_widths.clone(),
            root_ids: Vec::new(),
        }
    }

    /// Add an instruction to the context.
    pub fn add_instruction(&mut self, inst: &Instruction) {
        if let Some(result_id) = inst.result_id {
            if let Some(result_type) = inst.result_type {
                self.id_to_type.insert(result_id, result_type);
            }

            if let Some(term) = self.instruction_to_term(inst) {
                self.id_to_term.insert(result_id, term);
                self.root_ids.push(result_id);
            }
        }
    }

    /// Get or create a term for an operand ID.
    fn get_or_create_term(&mut self, id: Word) -> String {
        if let Some(term) = self.id_to_term.get(&id) {
            term.clone()
        } else {
            format!("(Sym \"id{}\")", id)
        }
    }

    /// Convert an instruction to an egglog term.
    fn instruction_to_term(&mut self, inst: &Instruction) -> Option<String> {
        let term = match inst.class.opcode {
            Op::Constant => {
                let value = inst.operands.iter().find_map(|op| match op {
                    rspirv::dr::Operand::LiteralBit32(v) => Some(*v as i64),
                    rspirv::dr::Operand::LiteralBit64(v) => Some(*v as i64),
                    _ => None,
                })?;
                let width = inst
                    .result_type
                    .and_then(|ty| self.type_widths.get(&ty))
                    .copied()
                    .unwrap_or(32);
                if width == 64 {
                    format!("(Const64 {})", value)
                } else {
                    format!("(Const {})", value)
                }
            }
            Op::ConstantTrue => "(Const 1)".to_string(),
            Op::ConstantFalse => "(Const 0)".to_string(),

            Op::IAdd => self.binary_op("Add", inst)?,
            Op::ISub => self.binary_op("Sub", inst)?,
            Op::IMul => self.binary_op("Mul", inst)?,
            Op::SDiv => self.binary_op("SDiv", inst)?,
            Op::UDiv => self.binary_op("UDiv", inst)?,
            Op::SRem => self.binary_op("SRem", inst)?,
            Op::SMod => self.binary_op("SMod", inst)?,
            Op::UMod => self.binary_op("UMod", inst)?,
            Op::SNegate => self.unary_op("Neg", inst)?,
            Op::ShiftLeftLogical => self.binary_op("Shl", inst)?,
            Op::ShiftRightLogical => self.binary_op("ShrU", inst)?,
            Op::ShiftRightArithmetic => self.binary_op("ShrS", inst)?,
            Op::BitwiseAnd => self.binary_op("BitAnd", inst)?,
            Op::BitwiseOr => self.binary_op("BitOr", inst)?,
            Op::BitwiseXor => self.binary_op("BitXor", inst)?,
            Op::Not => self.unary_op("BitNot", inst)?,
            Op::BitReverse => self.unary_op("BitReverse", inst)?,
            Op::IEqual => self.binary_op("Eq", inst)?,
            Op::INotEqual => self.binary_op("Ne", inst)?,
            Op::SLessThan => self.binary_op("SLt", inst)?,
            Op::SLessThanEqual => self.binary_op("SLe", inst)?,
            Op::SGreaterThan => self.binary_op("SGt", inst)?,
            Op::SGreaterThanEqual => self.binary_op("SGe", inst)?,
            Op::ULessThan => self.binary_op("ULt", inst)?,
            Op::ULessThanEqual => self.binary_op("ULe", inst)?,
            Op::UGreaterThan => self.binary_op("UGt", inst)?,
            Op::UGreaterThanEqual => self.binary_op("UGe", inst)?,
            Op::LogicalNot => self.unary_op("LogNot", inst)?,
            Op::LogicalAnd => self.binary_op("LogAnd", inst)?,
            Op::LogicalOr => self.binary_op("LogOr", inst)?,
            Op::LogicalEqual => self.binary_op("LogEq", inst)?,
            Op::LogicalNotEqual => self.binary_op("LogNe", inst)?,
            Op::Select => {
                let ops: Vec<Word> = inst.operands.iter().filter_map(|op| op.id_ref_any()).collect();
                if ops.len() >= 3 {
                    let cond = self.get_or_create_term(ops[0]);
                    let t = self.get_or_create_term(ops[1]);
                    let f = self.get_or_create_term(ops[2]);
                    format!("(Select {} {} {})", cond, t, f)
                } else {
                    return None;
                }
            }
            Op::CopyObject => {
                // CopyObject is just a reference to another value
                let operand_id = inst.operands.iter().find_map(|op| op.id_ref_any())?;
                self.get_or_create_term(operand_id)
            }
            Op::Phi => {
                // For Phi, check if all incoming values are the same
                // Phi operands are pairs: (value, block) repeated
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                // Extract just the values (even indices)
                let values: Vec<Word> = ops.iter().step_by(2).copied().collect();
                if values.is_empty() {
                    return None;
                }
                // If all values are the same, the phi simplifies to that value
                // Use a Sym reference so that after extraction we get a CopyObject
                let first = values[0];
                if values.iter().all(|&v| v == first) {
                    // Always use Sym for phi simplification to preserve the reference
                    format!("(Sym \"id{}\")", first)
                } else {
                    // Can't optimize a phi with different values in the e-graph yet
                    return None;
                }
            }
            _ => return None,
        };

        Some(term)
    }

    fn binary_op(&mut self, op: &str, inst: &Instruction) -> Option<String> {
        let ops: Vec<Word> = inst.operands.iter().filter_map(|op| op.id_ref_any()).collect();
        if ops.len() >= 2 {
            let lhs = self.get_or_create_term(ops[0]);
            let rhs = self.get_or_create_term(ops[1]);
            Some(format!("({} {} {})", op, lhs, rhs))
        } else {
            None
        }
    }

    fn unary_op(&mut self, op: &str, inst: &Instruction) -> Option<String> {
        let operand_id = inst.operands.iter().find_map(|op| op.id_ref_any())?;
        let operand = self.get_or_create_term(operand_id);
        Some(format!("({} {})", op, operand))
    }
}
