//! Assembly-related utilities.

/// Tokenization primitives for SPIR-V assembly text.
pub mod lexer;
/// Bitflag wrappers for assembler/disassembler options.
pub mod options;

pub use lexer::{
    LexError, Lexer, NamedId, Punctuation, Span, StringLiteral, Token, TokenKind, WordToken,
};
pub use options::{BinaryToTextOptions, TextToBinaryOptions};
