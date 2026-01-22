//! Assembly-related utilities.

/// Translator that turns parsed instructions into DR modules.
pub mod assembler;
/// Decoration operand metadata derived from the SPIR-V grammar.
pub mod decoration;
/// Extended instruction helpers.
pub mod ext_inst;
/// Grammar-backed instruction metadata and operand newtypes.
pub mod instruction;
/// Tokenization primitives for SPIR-V assembly text.
pub mod lexer;
/// Bitflag wrappers for assembler/disassembler options.
pub mod options;
/// Parser and builder utilities for SPIR-V assembly instructions.
pub mod parser;

pub use assembler::{
    assemble_text, assemble_text_with_env, assemble_text_with_options, assemble_text_with_spans,
    assemble_text_with_spans_and_env, assemble_text_with_spans_full, AssemblyError,
    AssemblyTranslator, AssemblyWithSpans, ModuleBuilder,
};
pub use ext_inst::{
    lookup_custom_ext_inst_name, lookup_custom_ext_inst_opcode, ExtInstImportInfo, ExtInstSetKind,
    ResolvedExtInst,
};
pub use instruction::{
    IdRef, InstructionLayout, LiteralNumber, OperandDescriptor, ResultId, SpirvId, TypeId,
};
pub use lexer::{
    LexError, Lexer, NamedId, Punctuation, Span, StringLiteral, Token, TokenKind, WordToken,
};
pub use options::{BinaryToTextOptions, TextToBinaryOptions};
pub use parser::{parse_instruction, OperandValue, ParseError, ParsedInstruction, ParsedOperand};
