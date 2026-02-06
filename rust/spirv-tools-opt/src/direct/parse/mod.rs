//! Term parsing and instruction reconstruction from egglog results.
//!
//! This module converts egglog terms back to SPIR-V instructions.
//! It's organized into submodules by operation category:
//!
//! - `constants`: Constant values and symbol references
//! - `arithmetic`: Integer arithmetic, comparison, and logical operations
//! - `bitwise`: Bitfield operations (insert, extract)
//! - `floating`: Floating-point operations and conversions
//! - `vector`: Vector/composite construction and manipulation
//! - `memory`: Load, store, and access chain operations
//! - `image`: Image query operations
//! - `glsl`: GLSL.std.450 extended instruction set
//! - `extract`: Parsing egglog extraction results
//! - `util`: Shared parsing utilities

mod arithmetic;
mod bitwise;
mod constants;
mod extract;
mod floating;
mod glsl;
mod image;
mod memory;
mod util;
mod vector;

use rspirv::dr::Instruction;
use rspirv::spirv::Word;
use std::collections::HashMap;

// Re-export public items
pub use constants::{find_inline_constants, InlineConstKind};
pub use extract::parse_extract_result;

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

    // Try each category in order of likelihood

    // Constants and symbols (very common)
    if let Some(inst) =
        constants::try_parse_constant(term, result_id, result_type, id_map, type_width)
    {
        return Some(inst);
    }

    // Arithmetic and logical operations (most common)
    if let Some(inst) = arithmetic::try_parse_arithmetic(term, result_id, result_type, id_map) {
        return Some(inst);
    }

    // Floating-point operations
    if let Some(inst) = floating::try_parse_floating(term, result_id, result_type, id_map) {
        return Some(inst);
    }

    // Vector and composite operations
    if let Some(inst) = vector::try_parse_vector(term, result_id, result_type, id_map) {
        return Some(inst);
    }

    // Bitfield operations
    if let Some(inst) = bitwise::try_parse_bitfield(term, result_id, result_type, id_map) {
        return Some(inst);
    }

    // Memory operations
    if let Some(inst) = memory::try_parse_memory(term, result_id, result_type, id_map) {
        return Some(inst);
    }

    // Image operations
    if let Some(inst) = image::try_parse_image(term, result_id, result_type, id_map) {
        return Some(inst);
    }

    // GLSL extended instructions (if ext ID is available)
    if let Some(ext_id) = glsl_ext_id {
        if let Some(inst) = glsl::try_parse_glsl(term, result_id, result_type, id_map, ext_id) {
            return Some(inst);
        }
    }

    None
}
