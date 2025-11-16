use std::borrow::Cow;

use rspirv::spirv;

use super::instruction::{
    IdRef, InstructionLayout, LiteralNumber, OperandDescriptor, ResultId, SpirvId, TypeId,
};
use super::lexer::{Lexer, Punctuation, Span, Token, TokenKind, WordToken};
use crate::diagnostic::{DiagnosticMessage, MessagePosition};
use crate::message::MessageLevel;
use rspirv::grammar::{OperandKind, OperandQuantifier};

/// Fully parsed instruction with typed operands.
#[derive(Debug)]
pub struct ParsedInstruction<'a> {
    layout: InstructionLayout,
    result_type: Option<TypeId<'a>>,
    result_id: Option<ResultId<'a>>,
    operands: Vec<ParsedOperand<'a>>,
}

impl<'a> ParsedInstruction<'a> {
    /// Returns the opcode for this instruction.
    pub fn opcode(&self) -> spirv::Op {
        self.layout.opcode()
    }

    /// Returns the parsed result type identifier.
    pub fn result_type(&self) -> Option<TypeId<'a>> {
        self.result_type
    }

    /// Returns the parsed result identifier.
    pub fn result_id(&self) -> Option<ResultId<'a>> {
        self.result_id
    }

    /// Returns the parsed operands including descriptor metadata.
    pub fn operands(&self) -> &[ParsedOperand<'a>] {
        &self.operands
    }
}

/// Typed operand paired with its grammar descriptor.
#[derive(Debug)]
pub struct ParsedOperand<'a> {
    descriptor: OperandDescriptor,
    value: OperandValue<'a>,
    span: Span,
}

impl<'a> ParsedOperand<'a> {
    /// Returns the grammar metadata describing this operand.
    pub fn descriptor(&self) -> OperandDescriptor {
        self.descriptor
    }

    /// Returns the parsed operand value.
    pub fn value(&self) -> &OperandValue<'a> {
        &self.value
    }

    /// Returns the span covering this operand in the input source.
    pub fn span(&self) -> Span {
        self.span
    }
}

/// Supported operand values surfaced by the typed parser.
#[derive(Clone, Debug, PartialEq)]
pub enum OperandValue<'a> {
    /// References another ID within the module.
    Id(IdRef<'a>),
    /// Numeric literal parsed from the text stream.
    Literal(LiteralNumber),
    /// String literal operand (without quotes).
    String(&'a str),
    /// Raw word token for operand kinds we do not specialize yet.
    Word(WordToken<'a>),
    /// Structured memory access operand (mask + auxiliary parameters).
    MemoryAccess(MemoryAccessOperand<'a>),
}

/// Memory access operand parsed from the text stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryAccessOperand<'a> {
    mask: spirv::MemoryAccess,
    alignment: Option<LiteralNumber>,
    make_pointer_available_scope: Option<IdRef<'a>>,
    make_pointer_visible_scope: Option<IdRef<'a>>,
}

impl<'a> MemoryAccessOperand<'a> {
    /// Creates a parsed memory access operand capturing the mask and auxiliary fields.
    pub const fn new(
        mask: spirv::MemoryAccess,
        alignment: Option<LiteralNumber>,
        make_pointer_available_scope: Option<IdRef<'a>>,
        make_pointer_visible_scope: Option<IdRef<'a>>,
    ) -> Self {
        Self {
            mask,
            alignment,
            make_pointer_available_scope,
            make_pointer_visible_scope,
        }
    }

    /// Returns the mask bits encoded by this operand.
    pub const fn mask(&self) -> spirv::MemoryAccess {
        self.mask
    }

    /// Returns the optional alignment literal when the `Aligned` bit is present.
    pub const fn alignment(&self) -> Option<&LiteralNumber> {
        self.alignment.as_ref()
    }

    /// Returns the optional scope when `MakePointerAvailable*` is present.
    pub const fn make_pointer_available_scope(&self) -> Option<IdRef<'a>> {
        self.make_pointer_available_scope
    }

    /// Returns the optional scope when `MakePointerVisible*` is present.
    pub const fn make_pointer_visible_scope(&self) -> Option<IdRef<'a>> {
        self.make_pointer_visible_scope
    }
}

/// Parser errors emitted with diagnostic metadata so they can be reported through the context consumer.
#[derive(Debug)]
pub struct ParseError {
    diagnostic: DiagnosticMessage<'static>,
}

impl ParseError {
    fn new(position: MessagePosition, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            diagnostic: DiagnosticMessage::new(MessageLevel::Error, position, message)
                .with_source("assembler"),
        }
    }

    /// Returns the diagnostic payload suitable for emission.
    pub fn diagnostic(&self) -> &DiagnosticMessage<'static> {
        &self.diagnostic
    }

    /// Consumes this error and returns the contained diagnostic.
    pub fn into_diagnostic(self) -> DiagnosticMessage<'static> {
        self.diagnostic
    }
}

/// High-level entry point: parse a single instruction from the provided text.
pub fn parse_instruction(text: &str) -> Result<ParsedInstruction<'_>, ParseError> {
    Parser::new(text).parse_instruction()
}

struct Parser<'a> {
    stream: TokenStream<'a>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            let token = lexer
                .next_token()
                .expect("lexer errors handled in ParseError");
            let at_end = matches!(token.kind(), TokenKind::EndOfFile);
            tokens.push(token);
            if at_end {
                break;
            }
        }
        Self {
            stream: TokenStream::new(tokens),
        }
    }

    fn parse_instruction(mut self) -> Result<ParsedInstruction<'a>, ParseError> {
        let first_word = self
            .stream
            .expect_word("instruction or result identifier")?;
        let (result_id, opcode_token) = if self.stream.consume_equals() {
            let id = parse_identifier(first_word.word, first_word.span, "result id")?;
            let opcode = self.stream.expect_word("opcode")?;
            (Some(ResultId::new(id)), opcode)
        } else {
            (None, first_word)
        };

        let opcode = opcode_token
            .word
            .opcode()
            .ok_or_else(|| ParseError::new(opcode_token.span.start(), "Unknown opcode"))?;
        let layout = InstructionLayout::lookup(opcode).ok_or_else(|| {
            ParseError::new(opcode_token.span.start(), "Opcode not available in grammar")
        })?;

        if layout.result_id().is_some() && result_id.is_none() {
            return Err(ParseError::new(
                opcode_token.span.start(),
                "Instruction requires a result id assignment",
            ));
        }
        if layout.result_id().is_none() && result_id.is_some() {
            return Err(ParseError::new(
                opcode_token.span.start(),
                "Instruction does not define a result id",
            ));
        }

        let mut parsed_result_type = None;
        if layout.result_type().is_some() {
            let type_token = self.stream.expect_word("result type")?;
            let type_id = parse_identifier(type_token.word, type_token.span, "result type")?;
            parsed_result_type = Some(TypeId::new(type_id));
        }

        let mut operands = Vec::new();
        for descriptor in layout.operands() {
            match descriptor.quantifier() {
                OperandQuantifier::One => {
                    operands.push(self.parse_operand(descriptor)?);
                }
                OperandQuantifier::ZeroOrOne => {
                    if self.stream.peek_is_end() {
                        continue;
                    }
                    operands.push(self.parse_operand(descriptor)?);
                }
                OperandQuantifier::ZeroOrMore => {
                    while !self.stream.peek_is_end() {
                        operands.push(self.parse_operand(descriptor)?);
                    }
                }
            }
        }

        if !self.stream.peek_is_end() {
            if matches!(
                layout.opcode(),
                spirv::Op::ExecutionMode | spirv::Op::ExecutionModeId | spirv::Op::LoopMerge
            ) {
                self.parse_trailing_literals(&mut operands)?;
            } else {
                let extra = self.stream.next().expect("peek checked");
                return Err(ParseError::new(
                    extra.span().start(),
                    "Unexpected tokens after instruction",
                ));
            }
        }

        Ok(ParsedInstruction {
            layout,
            result_type: parsed_result_type,
            result_id,
            operands,
        })
    }

    fn parse_trailing_literals(
        &mut self,
        operands: &mut Vec<ParsedOperand<'a>>,
    ) -> Result<(), ParseError> {
        let descriptor =
            OperandDescriptor::new(OperandKind::LiteralInteger, OperandQuantifier::ZeroOrMore);
        while !self.stream.peek_is_end() {
            operands.push(self.parse_operand(descriptor)?);
        }
        Ok(())
    }

    fn parse_operand(
        &mut self,
        descriptor: OperandDescriptor,
    ) -> Result<ParsedOperand<'a>, ParseError> {
        let token = self.stream.expect_any("operand")?;
        let span = token.span();
        let value = match token.kind() {
            TokenKind::Word(word) => match descriptor.kind() {
                OperandKind::IdRef | OperandKind::IdResult | OperandKind::IdResultType => {
                    OperandValue::Id(IdRef::new(parse_identifier(word, span, "id")?))
                }
                OperandKind::LiteralInteger | OperandKind::LiteralContextDependentNumber => {
                    OperandValue::Literal(parse_integer(word, span)?)
                }
                OperandKind::MemoryAccess => self.parse_memory_access_operand(word, span)?,
                _ => OperandValue::Word(word),
            },
            TokenKind::StringLiteral(lit) => {
                if descriptor.kind() != OperandKind::LiteralString {
                    return Err(ParseError::new(
                        span.start(),
                        "String literal not allowed for this operand",
                    ));
                }
                OperandValue::String(lit.value())
            }
            TokenKind::Punctuation(_) | TokenKind::EndOfFile => {
                return Err(ParseError::new(
                    span.start(),
                    "Unexpected token in operand list",
                ));
            }
        };

        Ok(ParsedOperand {
            descriptor,
            value,
            span,
        })
    }
}

impl<'a> Parser<'a> {
    fn parse_memory_access_operand(
        &mut self,
        word: WordToken<'a>,
        span: Span,
    ) -> Result<OperandValue<'a>, ParseError> {
        let mask = parse_memory_access_mask(word.as_str(), span)?;
        let alignment = if mask.contains(spirv::MemoryAccess::ALIGNED) {
            Some(self.parse_alignment_literal()?)
        } else {
            None
        };
        let make_pointer_available_scope = if mask
            .contains(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE)
            || mask.contains(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE_KHR)
        {
            Some(self.parse_scope_operand("MakePointerAvailable scope")?)
        } else {
            None
        };
        let make_pointer_visible_scope = if mask.contains(spirv::MemoryAccess::MAKE_POINTER_VISIBLE)
            || mask.contains(spirv::MemoryAccess::MAKE_POINTER_VISIBLE_KHR)
        {
            Some(self.parse_scope_operand("MakePointerVisible scope")?)
        } else {
            None
        };

        Ok(OperandValue::MemoryAccess(MemoryAccessOperand::new(
            mask,
            alignment,
            make_pointer_available_scope,
            make_pointer_visible_scope,
        )))
    }

    fn parse_alignment_literal(&mut self) -> Result<LiteralNumber, ParseError> {
        let token = self.stream.expect_any("alignment literal")?;
        match token.kind() {
            TokenKind::Word(word) => parse_integer(word, token.span()),
            _ => Err(ParseError::new(
                token.span().start(),
                "Alignment must be a literal integer",
            )),
        }
    }

    fn parse_scope_operand(&mut self, label: &str) -> Result<IdRef<'a>, ParseError> {
        let located = self.stream.expect_word(label)?;
        let id = parse_identifier(located.word, located.span, label)?;
        Ok(IdRef::new(id))
    }
}

struct TokenStream<'a> {
    tokens: Vec<Token<'a>>,
    index: usize,
}

impl<'a> TokenStream<'a> {
    fn new(tokens: Vec<Token<'a>>) -> Self {
        Self { tokens, index: 0 }
    }

    fn next(&mut self) -> Option<Token<'a>> {
        let token = self.tokens.get(self.index).copied();
        if token.is_some() {
            self.index += 1;
        }
        token
    }

    fn peek(&self) -> Option<Token<'a>> {
        self.tokens.get(self.index).copied()
    }

    fn peek_is_end(&self) -> bool {
        matches!(self.peek().map(|t| t.kind()), Some(TokenKind::EndOfFile))
    }

    fn expect_word(&mut self, what: &str) -> Result<LocatedWord<'a>, ParseError> {
        match self.next() {
            Some(token) => match token.kind() {
                TokenKind::Word(word) => Ok(LocatedWord {
                    word,
                    span: token.span(),
                }),
                _ => Err(ParseError::new(
                    token.span().start(),
                    format!("Expected {what}, found unexpected token"),
                )),
            },
            None => Err(ParseError::new(
                MessagePosition::default(),
                format!("Unexpected end of instruction while expecting {what}"),
            )),
        }
    }

    fn expect_any(&mut self, what: &str) -> Result<Token<'a>, ParseError> {
        self.next().ok_or_else(|| {
            ParseError::new(
                MessagePosition::default(),
                format!("Unexpected end of instruction while expecting {what}"),
            )
        })
    }

    fn consume_equals(&mut self) -> bool {
        match self.peek() {
            Some(token) if matches!(token.kind(), TokenKind::Punctuation(Punctuation::Equals)) => {
                self.next();
                true
            }
            _ => false,
        }
    }
}

struct LocatedWord<'a> {
    word: WordToken<'a>,
    span: Span,
}

fn parse_identifier<'a>(
    word: WordToken<'a>,
    span: Span,
    label: &str,
) -> Result<SpirvId<'a>, ParseError> {
    let named = word.named_id().ok_or_else(|| {
        ParseError::new(span.start(), format!("Expected {label} beginning with '%'"))
    })?;
    Ok(SpirvId::named(named))
}

fn parse_integer(word: WordToken<'_>, span: Span) -> Result<LiteralNumber, ParseError> {
    let text = word.as_str();
    if let Ok(value) = text.parse::<i64>() {
        if value < 0 {
            Ok(LiteralNumber::signed(value))
        } else {
            Ok(LiteralNumber::unsigned(value as u64))
        }
    } else if let Ok(value) = text.parse::<u64>() {
        Ok(LiteralNumber::unsigned(value))
    } else {
        Err(ParseError::new(
            span.start(),
            "Failed to parse integer literal",
        ))
    }
}

fn parse_memory_access_mask(text: &str, span: Span) -> Result<spirv::MemoryAccess, ParseError> {
    if text == "None" {
        return Ok(spirv::MemoryAccess::empty());
    }
    if let Ok(bits) = text.parse::<u32>() {
        return Ok(spirv::MemoryAccess::from_bits_truncate(bits));
    }

    let mut mask = spirv::MemoryAccess::empty();
    for part in text.split('|').map(str::trim) {
        if part.is_empty() || part == "None" {
            continue;
        }
        if let Some(flag) = memory_access_flag(part) {
            mask |= flag;
        } else if let Ok(bits) = part.parse::<u32>() {
            mask |= spirv::MemoryAccess::from_bits_truncate(bits);
        } else {
            return Err(ParseError::new(
                span.start(),
                format!("Unknown memory access flag '{part}'"),
            ));
        }
    }
    Ok(mask)
}

fn memory_access_flag(name: &str) -> Option<spirv::MemoryAccess> {
    match name {
        "Volatile" => Some(spirv::MemoryAccess::VOLATILE),
        "Aligned" => Some(spirv::MemoryAccess::ALIGNED),
        "Nontemporal" => Some(spirv::MemoryAccess::NONTEMPORAL),
        "MakePointerAvailable" => Some(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE),
        "MakePointerAvailableKHR" => Some(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE_KHR),
        "MakePointerVisible" => Some(spirv::MemoryAccess::MAKE_POINTER_VISIBLE),
        "MakePointerVisibleKHR" => Some(spirv::MemoryAccess::MAKE_POINTER_VISIBLE_KHR),
        "NonPrivatePointer" => Some(spirv::MemoryAccess::NON_PRIVATE_POINTER),
        "NonPrivatePointerKHR" => Some(spirv::MemoryAccess::NON_PRIVATE_POINTER_KHR),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_instruction, OperandValue};
    use crate::assembly::instruction::LiteralNumber;
    use rspirv::spirv;

    #[test]
    fn parses_op_type_int() {
        let parsed = parse_instruction("%2 = OpTypeInt 32 1").expect("parse");
        assert_eq!(parsed.opcode(), spirv::Op::TypeInt);
        assert!(parsed.result_type().is_none());
        assert_eq!(
            parsed
                .result_id()
                .unwrap()
                .as_spirv_id()
                .as_named()
                .unwrap()
                .name(),
            "2"
        );
        assert!(matches!(
            parsed.operands()[0].value(),
            OperandValue::Literal(_)
        ));
    }

    #[test]
    fn rejects_missing_result_id() {
        let error = parse_instruction("OpTypeInt 32 1").unwrap_err();
        assert!(error
            .diagnostic()
            .message()
            .contains("result id assignment"));
    }

    #[test]
    fn parses_memory_access_alignment_and_scope() {
        let parsed =
            parse_instruction("%val = OpLoad %uint %ptr Aligned|MakePointerVisible 16 %scope")
                .expect("parse");
        let operand = parsed.operands().last().expect("memory operand");
        match operand.value() {
            OperandValue::MemoryAccess(memory) => {
                assert!(memory.mask().contains(spirv::MemoryAccess::ALIGNED));
                assert!(memory
                    .mask()
                    .contains(spirv::MemoryAccess::MAKE_POINTER_VISIBLE));
                assert_eq!(memory.alignment().unwrap(), &LiteralNumber::unsigned(16));
                let scope = memory
                    .make_pointer_visible_scope()
                    .unwrap()
                    .as_spirv_id()
                    .as_named()
                    .unwrap();
                assert_eq!(scope.name(), "scope");
            }
            other => panic!("Expected memory access operand, got {other:?}"),
        }
    }
}
