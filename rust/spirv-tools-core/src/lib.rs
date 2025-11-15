#![warn(missing_docs)]

//! Core, type-safe building blocks for the SPIRV-Tools Rust port.

/// Assembly/disassembly helper structures.
pub mod assembly;
/// Diagnostic helpers shared across crates.
pub mod diagnostic;
/// Endianness modeling utilities.
pub mod endian;
/// Message severity abstractions.
pub mod message;
/// Result codes and helpers.
pub mod result;
/// Target environment modeling.
pub mod target_env;
/// Version helpers for SPIR-V and client APIs.
pub mod version;

pub use assembly::{
    parse_instruction, AssemblyTranslator, BinaryToTextOptions, IdRef, InstructionLayout, LexError,
    Lexer, LiteralNumber, ModuleBuilder, NamedId, OperandDescriptor, OperandValue, ParseError,
    ParsedInstruction, ParsedOperand, Punctuation, ResultId, Span, SpirvId, StringLiteral,
    TextToBinaryOptions, Token, TokenKind, TypeId, WordToken,
};
pub use diagnostic::{DiagnosticMessage, MessagePosition};
pub use endian::Endianness;
pub use message::MessageLevel;
pub use result::{InvalidSpvResult, SpvResult};
pub use target_env::TargetEnv;
pub use version::{SpirvVersion, VulkanVersion};
