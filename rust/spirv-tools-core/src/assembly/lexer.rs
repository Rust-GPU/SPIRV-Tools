use core::convert::TryFrom;
use std::collections::HashMap;

use once_cell::sync::Lazy;
use rspirv::{grammar::CoreInstructionTable, spirv};
use thiserror::Error;

use crate::diagnostic::MessagePosition;

static OPCODE_BY_NAME: Lazy<HashMap<&'static str, spirv::Op>> = Lazy::new(|| {
    CoreInstructionTable::iter()
        .map(|inst| (inst.opname, inst.opcode))
        .collect()
});

/// Tokenized punctuation characters recognized by the SPIR-V assembler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Punctuation {
    /// `,`
    Comma,
    /// `(`
    ParenOpen,
    /// `)`
    ParenClose,
    /// `=`
    Equals,
}

impl Punctuation {
    fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            b',' => Some(Self::Comma),
            b'(' => Some(Self::ParenOpen),
            b')' => Some(Self::ParenClose),
            b'=' => Some(Self::Equals),
            _ => None,
        }
    }
}

/// Errors that can be produced during lexing.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LexError {
    /// A string literal was opened but never closed before the input ended.
    #[error("unterminated string literal")]
    UnterminatedString {
        /// Start position of the string literal missing its closing quote.
        position: MessagePosition,
    },
}

/// Half-open span covering a token inside the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    start: MessagePosition,
    end: MessagePosition,
}

impl Span {
    fn new(start: MessagePosition, end: MessagePosition) -> Self {
        Self { start, end }
    }

    /// Returns the starting source location.
    pub fn start(&self) -> MessagePosition {
        self.start
    }

    /// Returns the exclusive ending source location.
    pub fn end(&self) -> MessagePosition {
        self.end
    }
}

/// Borrowed string literal contents without the surrounding quotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringLiteral<'a> {
    value: &'a str,
}

impl<'a> StringLiteral<'a> {
    fn new(value: &'a str) -> Self {
        Self { value }
    }

    /// Returns the literal contents, without quotes.
    pub fn value(&self) -> &'a str {
        self.value
    }
}

/// A named id, e.g. `%1` or `%my_var`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamedId<'a> {
    raw: &'a str,
}

impl<'a> NamedId<'a> {
    fn new(text: &'a str) -> Option<Self> {
        text.starts_with('%').then_some(Self { raw: text })
    }

    /// Returns the identifier including the leading `%`.
    pub fn as_str(&self) -> &'a str {
        self.raw
    }

    /// Returns the identifier without the leading `%`.
    pub fn name(&self) -> &'a str {
        &self.raw[1..]
    }
}

/// A bare word token. Consumers can classify it into richer categories on
/// demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordToken<'a> {
    text: &'a str,
}

impl<'a> WordToken<'a> {
    fn new(text: &'a str) -> Self {
        debug_assert!(!text.is_empty());
        Self { text }
    }

    /// Returns the raw textual contents for this word.
    pub fn as_str(&self) -> &'a str {
        self.text
    }

    /// Returns the parsed opcode if this word is an `Op*` instruction name.
    pub fn opcode(&self) -> Option<spirv::Op> {
        let opname = self.text.strip_prefix("Op")?;
        OPCODE_BY_NAME.get(opname).copied()
    }

    /// Returns the named-id view if the word starts with `%`.
    pub fn named_id(&self) -> Option<NamedId<'a>> {
        NamedId::new(self.text)
    }
}

/// Token kinds that can be produced by the lexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind<'a> {
    /// Ordinary words (identifiers, immediates, opcodes, etc.).
    Word(WordToken<'a>),
    /// String literal contents.
    StringLiteral(StringLiteral<'a>),
    /// Simple punctuation.
    Punctuation(Punctuation),
    /// End-of-file sentinel.
    EndOfFile,
}

/// A token along with its input span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    kind: TokenKind<'a>,
    span: Span,
}

impl<'a> Token<'a> {
    fn new(kind: TokenKind<'a>, span: Span) -> Self {
        Self { kind, span }
    }

    /// Returns the token kind.
    pub fn kind(&self) -> TokenKind<'a> {
        self.kind
    }

    /// Returns the span for this token.
    pub fn span(&self) -> Span {
        self.span
    }
}

/// Streaming lexer that tokenizes SPIR-V assembly text.
pub struct Lexer<'a> {
    input: &'a str,
    bytes: &'a [u8],
    offset: usize,
    line: u32,
    column: u32,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given assembly string.
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            offset: 0,
            line: 0,
            column: 0,
        }
    }

    /// Returns the next token in the stream.
    pub fn next_token(&mut self) -> Result<Token<'a>, LexError> {
        self.skip_whitespace_and_comments();
        let start_pos = self.position();
        if self.offset >= self.bytes.len() {
            return Ok(Token::new(
                TokenKind::EndOfFile,
                Span::new(start_pos, start_pos),
            ));
        }

        let byte = self.peek_byte().expect("offset bounds checked above");
        if let Some(punctuation) = Punctuation::from_byte(byte) {
            self.advance_byte();
            let span = Span::new(start_pos, self.position());
            return Ok(Token::new(TokenKind::Punctuation(punctuation), span));
        }

        if byte == b'"' {
            return self.lex_string_literal(start_pos);
        }

        Ok(self.lex_word(start_pos))
    }

    fn lex_word(&mut self, start_pos: MessagePosition) -> Token<'a> {
        let start_offset = self.offset;
        while let Some(byte) = self.peek_byte() {
            if is_word_breaker(byte) {
                break;
            }
            self.advance_byte();
        }
        let end_pos = self.position();
        let text = &self.input[start_offset..self.offset];
        Token::new(
            TokenKind::Word(WordToken::new(text)),
            Span::new(start_pos, end_pos),
        )
    }

    fn lex_string_literal(&mut self, start_pos: MessagePosition) -> Result<Token<'a>, LexError> {
        // Skip the opening quote.
        self.advance_byte();
        let content_start = self.offset;
        let mut escaping = false;
        while let Some(byte) = self.peek_byte() {
            self.advance_byte();
            match byte {
                b'\\' => {
                    escaping = !escaping;
                    continue;
                }
                b'"' if !escaping => {
                    let content = &self.input[content_start..self.offset - 1];
                    let span = Span::new(start_pos, self.position());
                    return Ok(Token::new(
                        TokenKind::StringLiteral(StringLiteral::new(content)),
                        span,
                    ));
                }
                _ => {
                    escaping = false;
                }
            }
        }

        Err(LexError::UnterminatedString {
            position: start_pos,
        })
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek_byte() {
                Some(b' ' | b'\t' | b'\r') => {
                    self.advance_byte();
                }
                Some(b'\n') => {
                    self.advance_byte();
                }
                Some(b';') => {
                    self.advance_byte();
                    self.skip_comment();
                }
                Some(0) => {
                    self.advance_byte();
                }
                _ => break,
            }
        }
    }

    fn skip_comment(&mut self) {
        while let Some(byte) = self.peek_byte() {
            self.advance_byte();
            if byte == b'\n' {
                break;
            }
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.bytes.get(self.offset).copied()
    }

    fn advance_byte(&mut self) -> Option<u8> {
        let byte = *self.bytes.get(self.offset)?;
        self.offset += 1;
        if byte == b'\n' {
            self.line += 1;
            self.column = 0;
        } else {
            self.column += 1;
        }
        Some(byte)
    }

    fn position(&self) -> MessagePosition {
        let index = u32::try_from(self.offset).unwrap_or(u32::MAX);
        MessagePosition::new(self.line, self.column, index)
    }
}

fn is_word_breaker(byte: u8) -> bool {
    matches!(
        byte,
        b' ' | b'\t' | b'\r' | b'\n' | b';' | b',' | b'(' | b')' | b'='
    )
}

#[cfg(test)]
mod tests {
    use super::{LexError, Lexer, Punctuation, TokenKind};
    use rspirv::spirv;

    fn collect(text: &str) -> Vec<TokenKind<'_>> {
        let mut lexer = Lexer::new(text);
        let mut kinds = Vec::new();
        loop {
            let token = lexer.next_token().expect("lex");
            let kind = token.kind();
            let done = matches!(kind, TokenKind::EndOfFile);
            kinds.push(kind);
            if done {
                break;
            }
        }
        kinds
    }

    #[test]
    fn lexes_basic_sequence() {
        let kinds = collect("%1 = OpTypeInt 32 1");
        assert!(matches!(kinds[0], TokenKind::Word(word) if word.named_id().is_some()));
        assert!(matches!(
            kinds[1],
            TokenKind::Punctuation(Punctuation::Equals)
        ));
        assert!(
            matches!(kinds[2], TokenKind::Word(word) if word.opcode() == Some(spirv::Op::TypeInt))
        );
        assert!(matches!(kinds[3], TokenKind::Word(word) if word.as_str() == "32"));
        assert!(matches!(kinds[4], TokenKind::Word(word) if word.as_str() == "1"));
        assert!(matches!(kinds[5], TokenKind::EndOfFile));
    }

    #[test]
    fn skips_comments_and_whitespace() {
        let kinds = collect("OpCapability Shader\n; comment\nOpMemoryModel Logical GLSL450");
        assert!(
            matches!(kinds[0], TokenKind::Word(word) if word.opcode() == Some(spirv::Op::Capability))
        );
        assert!(matches!(kinds[1], TokenKind::Word(word) if word.as_str() == "Shader"));
        assert!(
            matches!(kinds[2], TokenKind::Word(word) if word.opcode() == Some(spirv::Op::MemoryModel))
        );
        assert!(matches!(kinds[3], TokenKind::Word(word) if word.as_str() == "Logical"));
        assert!(matches!(kinds[4], TokenKind::Word(word) if word.as_str() == "GLSL450"));
        assert!(matches!(kinds[5], TokenKind::EndOfFile));
    }

    #[test]
    fn parses_string_literals() {
        let mut lexer = Lexer::new("OpName %main \"main\"");
        assert!(lexer.next_token().unwrap().kind().is_word());
        assert!(lexer.next_token().unwrap().kind().is_word());
        match lexer.next_token().unwrap().kind() {
            TokenKind::StringLiteral(lit) => assert_eq!(lit.value(), "main"),
            other => panic!("unexpected token: {other:?}"),
        }
    }

    #[test]
    fn reports_unterminated_strings() {
        let mut lexer = Lexer::new("OpName %main \"unterminated");
        lexer.next_token().unwrap(); // OpName
        lexer.next_token().unwrap(); // %main
        let error = lexer.next_token().unwrap_err();
        assert!(matches!(error, LexError::UnterminatedString { .. }));
    }

    trait TokenKindExt<'a> {
        fn is_word(&self) -> bool;
    }

    impl<'a> TokenKindExt<'a> for TokenKind<'a> {
        fn is_word(&self) -> bool {
            matches!(self, TokenKind::Word(_))
        }
    }
}
