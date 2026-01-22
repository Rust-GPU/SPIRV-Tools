#![warn(missing_docs)]
// Allow some clippy lints that would require significant refactoring
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::module_inception)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::doc_overindented_list_items)]

//! Core, type-safe building blocks for the SPIRV-Tools Rust port.

/// Assembly/disassembly helper structures.
pub mod assembly;
/// Diagnostic helpers shared across crates.
pub mod diagnostic;
/// Disassembly helpers built on rspirv.
pub mod disassembly;
/// Endianness modeling utilities.
pub mod endian;
/// Message severity abstractions.
pub mod message;
/// Result codes and helpers.
pub mod result;
/// String literal parsing/formatting helpers shared across the port.
pub mod string_literal;
/// Target environment modeling.
pub mod target_env;
/// Validation helpers for SPIR-V modules.
pub mod validation;
/// Version helpers for SPIR-V and client APIs.
pub mod version;

pub use assembly::{
    assemble_text, assemble_text_with_env, assemble_text_with_options, parse_instruction,
    AssemblyTranslator, BinaryToTextOptions, IdRef, InstructionLayout, LexError, Lexer,
    LiteralNumber, ModuleBuilder, NamedId, OperandDescriptor, OperandValue, ParseError,
    ParsedInstruction, ParsedOperand, Punctuation, ResultId, Span, SpirvId, StringLiteral,
    TextToBinaryOptions, Token, TokenKind, TypeId, WordToken,
};
pub use diagnostic::{DiagnosticMessage, MessagePosition};
pub use disassembly::{disassemble_binary, DisassemblyError};
pub use endian::Endianness;
pub use message::MessageLevel;
pub use result::{InvalidSpvResult, SpvResult};
pub use target_env::TargetEnv;
pub use validation::{validate_module, ValidationError};
pub use version::{SpirvVersion, VulkanVersion};
