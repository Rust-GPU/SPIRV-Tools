//! Parsing for memory operations (Load, AccessChain).

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

use super::util::{resolve_term_to_id, split_terms};

/// Try to parse a memory operation.
pub fn try_parse_memory(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
) -> Option<Instruction> {
    // Parse Load (Load ptr mem) - extract just the pointer
    if let Some(rest) = term.strip_prefix("(Load ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if !terms.is_empty() {
                if let Some(ptr) = resolve_term_to_id(&terms[0], id_map) {
                    return Some(Instruction::new(
                        Op::Load,
                        Some(result_type),
                        Some(result_id),
                        vec![rspirv::dr::Operand::IdRef(ptr)],
                    ));
                }
            }
        }
    }

    // Parse AccessChainDyn (dynamic index)
    if let Some(rest) = term.strip_prefix("(AccessChainDyn ") {
        if let Some(inner) = rest.strip_suffix(')') {
            let terms = split_terms(inner);
            if terms.len() >= 2 {
                if let (Some(base), Some(idx)) = (
                    resolve_term_to_id(&terms[0], id_map),
                    resolve_term_to_id(&terms[1], id_map),
                ) {
                    return Some(Instruction::new(
                        Op::AccessChain,
                        Some(result_type),
                        Some(result_id),
                        vec![
                            rspirv::dr::Operand::IdRef(base),
                            rspirv::dr::Operand::IdRef(idx),
                        ],
                    ));
                }
            }
        }
    }

    // Note: Static AccessChain1/2/3 require constant pool access which we don't have
    // in this context. They would need to be handled at a higher level where we can
    // create constant instructions for the literal indices.

    None
}
