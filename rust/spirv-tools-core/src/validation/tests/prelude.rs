//! Common imports and helper functions for validation tests.
//!
//! This module provides shared utilities for validation tests. Tests should
//! import directly from `crate::validation` for any private/internal functions.

// Re-export public validation types
pub use crate::validation::{
    format_validation_error, format_validation_error_from_words, validate_module,
    validate_module_with_options, CheckedBound, DeclaredBound, DecorationTargetId,
    DecorationTargetKind, ExtensionName, FriendlyNames, Id, IdKind, MaybeValidModule,
    MemberDecorationTargetId, MemberIndex, MergeTargetKind, ModuleWords, OperandId, ResultId,
    Schema, TypeId, ValidModuleCache, ValidatableModule, ValidationError, ValidationOptions,
};

// Re-export common crate types
pub use crate::assembly::assemble_text;
pub use crate::target_env::TargetEnv;
pub use crate::version::SpirvVersion;

// Re-export rspirv types
pub use rspirv::spirv::{
    Capability, ExecutionMode, FunctionControl, MemoryModel, Op, StorageClass,
};

// Re-export std types
pub use std::collections::HashMap;
pub use std::num::NonZeroU32;
pub use std::sync::Arc;

/// Constructs a SPIR-V instruction header word from word count and opcode.
pub fn op(word_count: u16, opcode: u16) -> u32 {
    ((word_count as u32) << 16) | opcode as u32
}

/// Reorders an instruction with the given opcode to the end of the module.
pub fn reorder_opcode_to_end(mut words: Vec<u32>, opcode: Op) -> Vec<u32> {
    let mut idx = 5; // skip header
    while idx < words.len() {
        let wc = (words[idx] >> 16) as usize;
        let op_word = words[idx] & 0xffff;
        if op_word == opcode as u32 {
            let inst: Vec<u32> = words.drain(idx..idx + wc).collect();
            words.extend(inst);
            break;
        }
        idx += wc;
    }
    words
}

/// Extension name bytes for SPV_GOOGLE_decorate_string.
pub const EXT_SPV_GOOGLE_DECORATE_STRING_WORDS: [u32; 7] = [
    0x5f56_5053,
    0x474f_4f47,
    0x645f_454c,
    0x726f_6365,
    0x5f65_7461,
    0x6972_7473,
    0x0000_676e,
];
