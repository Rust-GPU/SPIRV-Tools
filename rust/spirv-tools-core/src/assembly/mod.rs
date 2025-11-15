//! Assembly-related utilities.

/// Grammar-backed instruction metadata and operand newtypes.
pub mod instruction;
/// Tokenization primitives for SPIR-V assembly text.
pub mod lexer;
/// Bitflag wrappers for assembler/disassembler options.
pub mod options;

pub use instruction::{
    IdRef, InstructionLayout, LiteralNumber, OperandDescriptor, ResultId, SpirvId, TypeId,
};
pub use lexer::{
    LexError, Lexer, NamedId, Punctuation, Span, StringLiteral, Token, TokenKind, WordToken,
};
pub use options::{BinaryToTextOptions, TextToBinaryOptions};
