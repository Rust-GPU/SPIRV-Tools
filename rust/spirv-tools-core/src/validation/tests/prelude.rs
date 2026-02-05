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

/// Constructs a minimal valid SPIR-V module text with a Shader capability and
/// the given extension name.
pub fn module_with_extension(extension: &str) -> String {
    module_with_extension_custom(
        extension,
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
    )
}

/// Constructs a minimal valid SPIR-V module text with Kernel/Addresses
/// capabilities and the given extension name.
pub fn opencl_module_with_extension(extension: &str) -> String {
    // Note: OpenCL requires Physical32/Physical64 addressing model
    [
        "OpCapability Kernel",
        "OpCapability Addresses",
        &format!("OpExtension \"{extension}\""),
        "OpMemoryModel Physical64 OpenCL",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n")
}

/// Constructs a minimal valid SPIR-V module text with custom capability,
/// extension, and memory model strings.
pub fn module_with_extension_custom(
    extension: &str,
    capability: &str,
    memory_model: &str,
) -> String {
    [
        capability,
        &format!("OpExtension \"{extension}\""),
        memory_model,
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n")
}

/// Asserts that the given extension name is accepted in Vulkan but rejected
/// in OpenCL and OpenGL environments.
pub fn assert_vulkan_only_extension(name: &str) {
    let text = module_with_extension(name);
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .unwrap_or_else(|_| panic!("{name} should be accepted for Vulkan targets"));
    for env in [TargetEnv::OpenCl2_2, TargetEnv::OpenGl4_5] {
        let error = text
            .as_str()
            .validate(env)
            .expect_err("{name} should be rejected outside Vulkan");
        assert_eq!(
            error,
            ValidationError::DisallowedExtension {
                extension: ExtensionName::from(name),
                env
            }
        );
    }
}

/// Assembles the given text and validates it with a specific target environment.
pub fn assemble_and_validate_with_env(
    text: impl AsRef<str>,
    env: TargetEnv,
) -> Result<(), ValidationError> {
    let binary = assemble_text(text.as_ref()).expect("assemble text");
    validate_module(&binary, env)
}

/// Assembles the given text and validates it with `Universal1_3`.
pub fn assemble_and_validate(text: impl AsRef<str>) -> Result<(), ValidationError> {
    assemble_and_validate_with_env(text, TargetEnv::Universal1_3)
}
