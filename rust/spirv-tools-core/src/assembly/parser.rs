use std::borrow::Cow;
use std::num::NonZeroU32;

use rspirv::spirv;

use super::instruction::{
    IdRef, InstructionLayout, LiteralNumber, OperandDescriptor, ResultId, SpirvId, TypeId,
};
use super::lexer::{LexError, Lexer, Punctuation, Span, Token, TokenKind, WordToken};
use crate::diagnostic::{DiagnosticMessage, MessagePosition};
use crate::message::MessageLevel;
use rspirv::grammar::{OperandKind, OperandQuantifier};

/// Fully parsed instruction with typed operands.
#[derive(Debug)]
pub struct ParsedInstruction<'a> {
    layout: InstructionLayout,
    opcode_span: Span,
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

    /// Returns the span of the opcode token for this instruction.
    pub fn opcode_span(&self) -> Span {
        self.opcode_span
    }

    /// Returns the starting position of the opcode token.
    pub fn opcode_position(&self) -> MessagePosition {
        self.opcode_span.start()
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
    /// A pair of ID references (used for instructions like OpPhi).
    IdPair(IdRef<'a>, IdRef<'a>),
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

    fn from_lex(error: LexError) -> Self {
        match error {
            LexError::UnterminatedString { position } => {
                Self::new(position, "unterminated string literal")
            }
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
    Parser::new(text)?.parse_instruction()
}

/// Parses a single instruction using the provided source origin for diagnostics.
pub fn parse_instruction_with_origin(
    text: &str,
    origin: MessagePosition,
) -> Result<ParsedInstruction<'_>, ParseError> {
    Parser::with_origin(text, origin)?.parse_instruction()
}

struct Parser<'a> {
    stream: TokenStream<'a>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Result<Self, ParseError> {
        Self::with_origin(input, MessagePosition::default())
    }

    fn with_origin(input: &'a str, origin: MessagePosition) -> Result<Self, ParseError> {
        let mut lexer = Lexer::with_origin(input, origin);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token().map_err(ParseError::from_lex)?;
            let at_end = matches!(token.kind(), TokenKind::EndOfFile);
            tokens.push(token);
            if at_end {
                break;
            }
        }
        Ok(Self {
            stream: TokenStream::new(tokens),
        })
    }

    fn parse_instruction(mut self) -> Result<ParsedInstruction<'a>, ParseError> {
        let first_word = self
            .stream
            .expect_word("instruction or result identifier")?;
        let (result_id, opcode_token) = if self.stream.consume_equals() {
            let id = parse_identifier(first_word.word, first_word.span, "result id")?;
            let opcode = self.stream.expect_word("opcode")?;
            (Some(ResultId::new(id, first_word.span)), opcode)
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
            parsed_result_type = Some(TypeId::new(type_id, type_token.span));
        }

        let mut operands = Vec::new();
        for descriptor in layout.operands() {
            match descriptor.quantifier() {
                OperandQuantifier::One => {
                    operands.push(self.parse_operand(descriptor, opcode)?);
                }
                OperandQuantifier::ZeroOrOne => {
                    if self.stream.peek_is_end() {
                        continue;
                    }
                    operands.push(self.parse_operand(descriptor, opcode)?);
                }
                OperandQuantifier::ZeroOrMore => {
                    while !self.stream.peek_is_end() {
                        operands.push(self.parse_operand(descriptor, opcode)?);
                    }
                }
            }
        }

        if !self.stream.peek_is_end() {
            match layout.opcode() {
                spirv::Op::ExecutionMode | spirv::Op::ExecutionModeId | spirv::Op::LoopMerge => {
                    self.parse_trailing_literals(&mut operands, opcode)?;
                }
                spirv::Op::Decorate
                | spirv::Op::DecorateId
                | spirv::Op::DecorateString
                | spirv::Op::MemberDecorate
                | spirv::Op::MemberDecorateString => {
                    self.parse_annotation_operands(&mut operands)?;
                }
                _ => {
                    let extra = self.stream.next().expect("peek checked");
                    return Err(ParseError::new(
                        extra.span().start(),
                        "Unexpected tokens after instruction",
                    ));
                }
            }
        }

        Ok(ParsedInstruction {
            layout,
            opcode_span: opcode_token.span,
            result_type: parsed_result_type,
            result_id,
            operands,
        })
    }

    fn parse_trailing_literals(
        &mut self,
        operands: &mut Vec<ParsedOperand<'a>>,
        opcode: spirv::Op,
    ) -> Result<(), ParseError> {
        let descriptor =
            OperandDescriptor::new(OperandKind::LiteralInteger, OperandQuantifier::ZeroOrMore);
        while !self.stream.peek_is_end() {
            operands.push(self.parse_operand(descriptor, opcode)?);
        }
        Ok(())
    }

    fn parse_annotation_operands(
        &mut self,
        operands: &mut Vec<ParsedOperand<'a>>,
    ) -> Result<(), ParseError> {
        while !self.stream.peek_is_end() {
            let token = self.stream.expect_any("decoration operand")?;
            let span = token.span();
            let (descriptor, value) = match token.kind() {
                TokenKind::Word(word) => {
                    if let Some(named) = word.named_id() {
                        (
                            OperandDescriptor::new(
                                OperandKind::IdRef,
                                OperandQuantifier::ZeroOrMore,
                            ),
                            OperandValue::Id(IdRef::new(SpirvId::named(named), span)),
                        )
                    } else if let Some(literal) = parse_loose_integer(word.as_str()) {
                        (
                            OperandDescriptor::new(
                                OperandKind::LiteralInteger,
                                OperandQuantifier::ZeroOrMore,
                            ),
                            OperandValue::Literal(literal),
                        )
                    } else {
                        (
                            OperandDescriptor::new(
                                OperandKind::LiteralContextDependentNumber,
                                OperandQuantifier::ZeroOrMore,
                            ),
                            OperandValue::Word(word),
                        )
                    }
                }
                TokenKind::StringLiteral(literal) => (
                    OperandDescriptor::new(
                        OperandKind::LiteralString,
                        OperandQuantifier::ZeroOrMore,
                    ),
                    OperandValue::String(literal.value()),
                ),
                TokenKind::Punctuation(_) | TokenKind::EndOfFile => {
                    return Err(ParseError::new(
                        span.start(),
                        "Unexpected token in decoration operand list",
                    ));
                }
            };
            operands.push(ParsedOperand {
                descriptor,
                value,
                span,
            });
        }
        Ok(())
    }

    fn parse_operand(
        &mut self,
        descriptor: OperandDescriptor,
        opcode: spirv::Op,
    ) -> Result<ParsedOperand<'a>, ParseError> {
        let token = self.stream.expect_any("operand")?;
        let span = token.span();
        let allow_ext_inst_operand = opcode == spirv::Op::ExtInst
            && descriptor.kind() == OperandKind::IdRef
            && descriptor.quantifier() == OperandQuantifier::ZeroOrMore;
        let value = match token.kind() {
            TokenKind::Word(word) => match descriptor.kind() {
                OperandKind::IdRef | OperandKind::IdResult | OperandKind::IdResultType => {
                    if allow_ext_inst_operand {
                        if let Some(named) = word.named_id() {
                            OperandValue::Id(IdRef::new(SpirvId::named(named), span))
                        } else if let Ok(literal) = parse_integer(word, span) {
                            OperandValue::Literal(literal)
                        } else {
                            OperandValue::Word(word)
                        }
                    } else {
                        let id = parse_identifier(word, span, "id")?;
                        OperandValue::Id(IdRef::new(id, span))
                    }
                }
                OperandKind::LiteralInteger => {
                    OperandValue::Literal(parse_integer(word, span)?)
                }
                OperandKind::LiteralContextDependentNumber => {
                    // Context-dependent numbers may be integer or float text
                    // depending on the result type (e.g. OpConstant %float 42.5).
                    // Fall back to Word so the assembler can do type-aware parsing.
                    match parse_integer(word, span) {
                        Ok(literal) => OperandValue::Literal(literal),
                        Err(_) => OperandValue::Word(word),
                    }
                }
                OperandKind::LiteralExtInstInteger => match parse_integer(word, span) {
                    Ok(literal) => OperandValue::Literal(literal),
                    Err(_) => OperandValue::Word(word),
                },
                OperandKind::PairIdRefIdRef => {
                    let first = IdRef::new(parse_identifier(word, span, "id")?, span);
                    let second = self.stream.expect_word("second id in pair")?;
                    let second_id = IdRef::new(
                        parse_identifier(second.word, second.span, "id")?,
                        second.span,
                    );
                    OperandValue::IdPair(first, second_id)
                }
                OperandKind::MemoryAccess => self.parse_memory_access_operand(word, span)?,
                _ => OperandValue::Word(word),
            },
            TokenKind::StringLiteral(lit) => {
                if descriptor.kind() != OperandKind::LiteralString && !allow_ext_inst_operand {
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
        let make_pointer_available_scope =
            if mask.contains(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE) {
                Some(self.parse_scope_operand("MakePointerAvailable scope")?)
            } else {
                None
            };
        let make_pointer_visible_scope = if mask.contains(spirv::MemoryAccess::MAKE_POINTER_VISIBLE)
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
        Ok(IdRef::new(id, located.span))
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
    let name = named.name();
    if let Ok(value) = name.parse::<u32>() {
        let Some(nonzero) = NonZeroU32::new(value) else {
            return Err(ParseError::new(span.start(), "Result ids cannot be 0"));
        };
        Ok(SpirvId::numeric(nonzero))
    } else {
        Ok(SpirvId::named(named))
    }
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

fn parse_loose_integer(text: &str) -> Option<LiteralNumber> {
    if text.starts_with('-') {
        text.parse::<i64>().ok().map(LiteralNumber::signed)
    } else {
        text.parse::<u64>().ok().map(LiteralNumber::unsigned)
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
        "MakePointerAvailable" | "MakePointerAvailableKHR" => {
            Some(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE)
        }
        "MakePointerVisible" | "MakePointerVisibleKHR" => {
            Some(spirv::MemoryAccess::MAKE_POINTER_VISIBLE)
        }
        "NonPrivatePointer" | "NonPrivatePointerKHR" => {
            Some(spirv::MemoryAccess::NON_PRIVATE_POINTER)
        }
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
        let numeric = parsed
            .result_id()
            .unwrap()
            .as_spirv_id()
            .as_numeric()
            .unwrap();
        assert_eq!(numeric.get(), 2);
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
