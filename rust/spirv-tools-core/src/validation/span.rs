//! Source span information for rich validation error reporting.
//!
//! This module provides types for tracking source locations in SPIR-V modules,
//! enabling rustc-style error messages with multiple labeled spans pointing to
//! relevant source locations.
//!
//! # Example Usage
//!
//! ```ignore
//! use spirv_tools_core::validation::span::{SpannedError, SourceSpan, SpanLabel};
//!
//! // Create a validation error with multiple spans
//! let error = SpannedError::new(ValidationError::TypeMismatch { ... })
//!     .with_span(SourceSpan::text(10, 5, 10, 20), SpanLabel::primary("type mismatch here"))
//!     .with_span(SourceSpan::text(5, 10, 5, 25), SpanLabel::secondary("expected type defined here"));
//! ```

use std::fmt;

use crate::diagnostic::MessagePosition;

/// A source location that can represent either text or binary positions.
///
/// For text sources (SPIR-V assembly), this contains line/column information.
/// For binary sources, this contains word offset information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceLocation {
    /// Position in text source (SPIR-V assembly).
    Text(MessagePosition),
    /// Word offset in binary source.
    Binary {
        /// Zero-based word offset from the start of the module.
        word_offset: u32,
    },
    /// Position within an instruction (for operand-level errors).
    Instruction {
        /// Word offset of the instruction start.
        instruction_offset: u32,
        /// Zero-based operand index within the instruction.
        operand_index: Option<u32>,
    },
}

impl SourceLocation {
    /// Creates a text source location from line and column (zero-based).
    pub const fn text(line: u32, column: u32) -> Self {
        Self::Text(MessagePosition::new(line, column, 0))
    }

    /// Creates a text source location from a `MessagePosition`.
    pub const fn from_position(position: MessagePosition) -> Self {
        Self::Text(position)
    }

    /// Creates a binary source location from a word offset.
    pub const fn binary(word_offset: u32) -> Self {
        Self::Binary { word_offset }
    }

    /// Creates an instruction-level source location.
    pub const fn instruction(instruction_offset: u32, operand_index: Option<u32>) -> Self {
        Self::Instruction {
            instruction_offset,
            operand_index,
        }
    }
}

impl Default for SourceLocation {
    fn default() -> Self {
        Self::Text(MessagePosition::default())
    }
}

/// A span covering a range in the source.
///
/// Spans are half-open intervals `[start, end)` in the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    /// Starting location (inclusive).
    pub start: SourceLocation,
    /// Ending location (exclusive).
    pub end: SourceLocation,
}

impl SourceSpan {
    /// Creates a new span from start and end locations.
    pub const fn new(start: SourceLocation, end: SourceLocation) -> Self {
        Self { start, end }
    }

    /// Creates a span from text positions (zero-based line/column).
    pub const fn text(
        start_line: u32,
        start_column: u32,
        end_line: u32,
        end_column: u32,
    ) -> Self {
        Self {
            start: SourceLocation::text(start_line, start_column),
            end: SourceLocation::text(end_line, end_column),
        }
    }

    /// Creates a span from binary word offsets.
    pub const fn binary(start_offset: u32, end_offset: u32) -> Self {
        Self {
            start: SourceLocation::binary(start_offset),
            end: SourceLocation::binary(end_offset),
        }
    }

    /// Creates a span covering a single instruction.
    pub const fn instruction(offset: u32, word_count: u32) -> Self {
        Self {
            start: SourceLocation::binary(offset),
            end: SourceLocation::binary(offset + word_count),
        }
    }

    /// Creates a point span (start == end) at a single location.
    pub const fn point(location: SourceLocation) -> Self {
        Self {
            start: location,
            end: location,
        }
    }

    /// Returns the word offset if this is a binary span.
    pub fn word_offset(&self) -> Option<u32> {
        match self.start {
            SourceLocation::Binary { word_offset } => Some(word_offset),
            SourceLocation::Instruction {
                instruction_offset, ..
            } => Some(instruction_offset),
            _ => None,
        }
    }

    /// Returns the text position if this is a text span.
    pub fn text_position(&self) -> Option<MessagePosition> {
        match self.start {
            SourceLocation::Text(pos) => Some(pos),
            _ => None,
        }
    }

    /// Creates a span from an assembly lexer span.
    ///
    /// This converts the text-based span from the SPIR-V assembler into
    /// a validation source span for error reporting.
    pub fn from_assembly_span(span: crate::assembly::lexer::Span) -> Self {
        Self {
            start: SourceLocation::from_position(span.start()),
            end: SourceLocation::from_position(span.end()),
        }
    }
}

impl Default for SourceSpan {
    fn default() -> Self {
        Self {
            start: SourceLocation::default(),
            end: SourceLocation::default(),
        }
    }
}

/// The kind of span label, affecting how it's rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LabelKind {
    /// Primary span - the main location of the error (typically rendered with `^^^`).
    #[default]
    Primary,
    /// Secondary span - related location (typically rendered with `---`).
    Secondary,
    /// Note span - additional context (typically rendered differently).
    Note,
    /// Help span - suggestion for fixing the error.
    Help,
}

/// A labeled span for display in error messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanLabel {
    /// The kind of label (primary, secondary, note, help).
    pub kind: LabelKind,
    /// Human-readable message for this span.
    pub message: String,
}

impl SpanLabel {
    /// Creates a new span label.
    pub fn new(kind: LabelKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Creates a primary (main error location) label.
    pub fn primary(message: impl Into<String>) -> Self {
        Self::new(LabelKind::Primary, message)
    }

    /// Creates a secondary (related location) label.
    pub fn secondary(message: impl Into<String>) -> Self {
        Self::new(LabelKind::Secondary, message)
    }

    /// Creates a note label.
    pub fn note(message: impl Into<String>) -> Self {
        Self::new(LabelKind::Note, message)
    }

    /// Creates a help label.
    pub fn help(message: impl Into<String>) -> Self {
        Self::new(LabelKind::Help, message)
    }
}

/// A span with its associated label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledSpan {
    /// The source span.
    pub span: SourceSpan,
    /// The label for this span.
    pub label: SpanLabel,
    /// Optional disassembled instruction text (for binary validation).
    pub instruction_text: Option<String>,
}

impl LabeledSpan {
    /// Creates a new labeled span.
    pub fn new(span: SourceSpan, label: SpanLabel) -> Self {
        Self {
            span,
            label,
            instruction_text: None,
        }
    }

    /// Creates a labeled span with a primary label.
    pub fn primary(span: SourceSpan, message: impl Into<String>) -> Self {
        Self::new(span, SpanLabel::primary(message))
    }

    /// Creates a labeled span with a secondary label.
    pub fn secondary(span: SourceSpan, message: impl Into<String>) -> Self {
        Self::new(span, SpanLabel::secondary(message))
    }

    /// Adds instruction text to this labeled span.
    pub fn with_instruction_text(mut self, text: impl Into<String>) -> Self {
        self.instruction_text = Some(text.into());
        self
    }
}

/// A validation error enriched with source span information.
///
/// This wrapper pairs a `ValidationError` with one or more labeled spans,
/// enabling rich error messages like those produced by rustc.
///
/// # Example
///
/// ```ignore
/// error[E0308]: type mismatch
///   --> shader.spvasm:10:5
///    |
/// 10 |     %result = OpFAdd %float %a %b
///    |               ^^^^^^^^^^^^^^^^^^^^^ expected i32, found f32
///    |
/// note: expected type defined here
///   --> shader.spvasm:5:10
///    |
///  5 |     %result_type = OpTypeInt 32 1
///    |                    ^^^^^^^^^^^^^^
/// ```
#[derive(Debug, Clone)]
pub struct SpannedError<E> {
    /// The underlying validation error.
    pub error: E,
    /// Labeled spans associated with this error.
    pub spans: Vec<LabeledSpan>,
    /// Additional notes that don't have specific source locations.
    pub notes: Vec<String>,
    /// Suggested fixes or help messages.
    pub help: Vec<String>,
}

impl<E> SpannedError<E> {
    /// Creates a new spanned error from an underlying error.
    pub fn new(error: E) -> Self {
        Self {
            error,
            spans: Vec::new(),
            notes: Vec::new(),
            help: Vec::new(),
        }
    }

    /// Adds a labeled span to this error.
    pub fn with_labeled_span(mut self, labeled_span: LabeledSpan) -> Self {
        self.spans.push(labeled_span);
        self
    }

    /// Adds a span with a label to this error.
    pub fn with_span(mut self, span: SourceSpan, label: SpanLabel) -> Self {
        self.spans.push(LabeledSpan::new(span, label));
        self
    }

    /// Adds a primary span to this error.
    pub fn with_primary_span(mut self, span: SourceSpan, message: impl Into<String>) -> Self {
        self.spans.push(LabeledSpan::primary(span, message));
        self
    }

    /// Adds a secondary span to this error.
    pub fn with_secondary_span(mut self, span: SourceSpan, message: impl Into<String>) -> Self {
        self.spans.push(LabeledSpan::secondary(span, message));
        self
    }

    /// Adds a note without a specific source location.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Adds a help message.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help.push(help.into());
        self
    }

    /// Returns the primary span, if any.
    pub fn primary_span(&self) -> Option<&LabeledSpan> {
        self.spans
            .iter()
            .find(|s| s.label.kind == LabelKind::Primary)
    }

    /// Returns all spans of a given kind.
    pub fn spans_of_kind(&self, kind: LabelKind) -> impl Iterator<Item = &LabeledSpan> {
        self.spans.iter().filter(move |s| s.label.kind == kind)
    }

    /// Maps the underlying error to a different type.
    pub fn map_error<F, E2>(self, f: F) -> SpannedError<E2>
    where
        F: FnOnce(E) -> E2,
    {
        SpannedError {
            error: f(self.error),
            spans: self.spans,
            notes: self.notes,
            help: self.help,
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SpannedError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl<E: fmt::Display> fmt::Display for SpannedError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display the main error message
        write!(f, "{}", self.error)?;

        // Display span information
        for labeled_span in &self.spans {
            let kind_str = match labeled_span.label.kind {
                LabelKind::Primary => "",
                LabelKind::Secondary => "note: ",
                LabelKind::Note => "note: ",
                LabelKind::Help => "help: ",
            };

            match labeled_span.span.start {
                SourceLocation::Text(pos) => {
                    write!(
                        f,
                        "\n  --> {}:{}: {}{}",
                        pos.line() + 1,
                        pos.column() + 1,
                        kind_str,
                        labeled_span.label.message
                    )?;
                }
                SourceLocation::Binary { word_offset } => {
                    write!(
                        f,
                        "\n  at word {}: {}{}",
                        word_offset, kind_str, labeled_span.label.message
                    )?;
                }
                SourceLocation::Instruction {
                    instruction_offset,
                    operand_index,
                } => {
                    if let Some(idx) = operand_index {
                        write!(
                            f,
                            "\n  at instruction word {}, operand {}: {}{}",
                            instruction_offset, idx, kind_str, labeled_span.label.message
                        )?;
                    } else {
                        write!(
                            f,
                            "\n  at instruction word {}: {}{}",
                            instruction_offset, kind_str, labeled_span.label.message
                        )?;
                    }
                }
            }

            // Display the instruction text if available
            if let Some(ref text) = labeled_span.instruction_text {
                write!(f, "\n  {}", text)?;
            }
        }

        // Display notes
        for note in &self.notes {
            write!(f, "\n  = note: {}", note)?;
        }

        // Display help
        for help in &self.help {
            write!(f, "\n  = help: {}", help)?;
        }

        Ok(())
    }
}

/// Maps SPIR-V IDs to their source spans.
///
/// This is populated during parsing/assembly and used during validation
/// to look up the definition site of IDs for error reporting.
///
/// The SpanMap stores only position information (line/column indices).
/// The caller retains ownership of the source text and uses span positions
/// to extract source snippets for error display.
#[derive(Debug, Clone, Default)]
pub struct SpanMap {
    /// Maps instruction word offsets to their source spans.
    instruction_spans: std::collections::HashMap<u32, SourceSpan>,
    /// Maps result IDs to their defining instruction's span.
    id_spans: std::collections::HashMap<u32, SourceSpan>,
}

impl SpanMap {
    /// Creates a new empty span map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the span for an instruction at a given word offset.
    pub fn record_instruction(&mut self, word_offset: u32, span: SourceSpan) {
        self.instruction_spans.insert(word_offset, span);
    }

    /// Records the span for a result ID.
    pub fn record_id(&mut self, id: u32, span: SourceSpan) {
        self.id_spans.insert(id, span);
    }

    /// Looks up the span for an instruction at a given word offset.
    pub fn get_instruction_span(&self, word_offset: u32) -> Option<SourceSpan> {
        self.instruction_spans.get(&word_offset).copied()
    }

    /// Looks up the span where an ID was defined.
    pub fn get_id_span(&self, id: u32) -> Option<SourceSpan> {
        self.id_spans.get(&id).copied()
    }

    /// Returns the number of recorded instruction spans.
    pub fn instruction_count(&self) -> usize {
        self.instruction_spans.len()
    }

    /// Returns the number of recorded ID spans.
    pub fn id_count(&self) -> usize {
        self.id_spans.len()
    }

    /// Returns true if the span map is empty.
    pub fn is_empty(&self) -> bool {
        self.instruction_spans.is_empty() && self.id_spans.is_empty()
    }
}

/// Extracts a source snippet from source text using a span.
///
/// This is a helper function for callers to extract source lines for error display.
/// The caller provides the source text they own, and this function extracts the
/// relevant lines based on the span positions.
pub fn extract_source_snippet<'a>(source: &'a str, span: &SourceSpan) -> Option<SourceSnippet<'a>> {
    let (start_line, start_col, end_line, end_col) = match (span.start, span.end) {
        (SourceLocation::Text(start), SourceLocation::Text(end)) => {
            (start.line(), start.column(), end.line(), end.column())
        }
        _ => return None,
    };

    let source_lines: Vec<&str> = source.lines().collect();
    if source_lines.is_empty() {
        return None;
    }

    let start_line = start_line as usize;
    let end_line = end_line as usize;

    if start_line >= source_lines.len() {
        return None;
    }

    let lines: Vec<(usize, &str)> = (start_line..=end_line.min(source_lines.len() - 1))
        .map(|i| (i, source_lines[i]))
        .collect();

    Some(SourceSnippet {
        lines,
        start_column: start_col as usize,
        end_column: end_col as usize,
        is_multiline: start_line != end_line,
    })
}

/// A source code snippet for display in error messages.
#[derive(Debug, Clone)]
pub struct SourceSnippet<'a> {
    /// The source lines with their line numbers (zero-indexed).
    pub lines: Vec<(usize, &'a str)>,
    /// Starting column of the span (zero-indexed).
    pub start_column: usize,
    /// Ending column of the span (zero-indexed).
    pub end_column: usize,
    /// Whether the span crosses multiple lines.
    pub is_multiline: bool,
}

impl<'a> SourceSnippet<'a> {
    /// Formats this snippet for display, with line numbers and underlines.
    ///
    /// Returns a string like:
    /// ```text
    ///  10 |     %result = OpFAdd %float %a %b
    ///     |               ^^^^^^^^^^^^^^^^^^^^
    /// ```
    pub fn format(&self, label: Option<&str>) -> String {
        if self.lines.is_empty() {
            return String::new();
        }

        let max_line_num = self.lines.last().map(|(n, _)| n + 1).unwrap_or(1);
        let line_num_width = max_line_num.to_string().len().max(2);

        let mut result = String::new();

        for (i, (line_num, line_text)) in self.lines.iter().enumerate() {
            // Format line number (1-indexed for display)
            result.push_str(&format!(
                "{:>width$} | {}\n",
                line_num + 1,
                line_text,
                width = line_num_width
            ));

            // Add underline for the first line (or only line)
            if i == 0 {
                let underline_start = self.start_column;
                let underline_end = if self.is_multiline {
                    line_text.len()
                } else {
                    self.end_column.max(underline_start + 1)
                };
                let underline_len = underline_end.saturating_sub(underline_start).max(1);

                result.push_str(&format!(
                    "{:>width$} | {}{}\n",
                    "",
                    " ".repeat(underline_start),
                    "^".repeat(underline_len),
                    width = line_num_width
                ));

                // Add label on the underline line if provided
                if let Some(label) = label {
                    if !label.is_empty() {
                        result.push_str(&format!(
                            "{:>width$} | {}{} {}\n",
                            "",
                            " ".repeat(underline_start),
                            "|",
                            label,
                            width = line_num_width
                        ));
                    }
                }
            }
        }

        result
    }
}

// ============================================================================
// Error enrichment utilities
// ============================================================================

/// Extension trait for enriching errors with span information.
///
/// This trait provides convenient methods to wrap errors with source spans,
/// enabling rich rustc-style error messages.
pub trait WithSpan: Sized {
    /// Wraps this error in a `SpannedError`.
    fn with_span(self) -> SpannedError<Self> {
        SpannedError::new(self)
    }

    /// Wraps this error with a primary span.
    fn with_primary_span_at(self, span: SourceSpan, message: impl Into<String>) -> SpannedError<Self> {
        SpannedError::new(self).with_primary_span(span, message)
    }
}

// Blanket implementation for all types
impl<E> WithSpan for E {}

/// Type alias for validation results with rich span information.
pub type SpannedResult<T, E> = Result<T, SpannedError<E>>;

/// Type alias specifically for SPIR-V validation errors with spans.
pub type SpannedValidationError = SpannedError<super::error::ValidationError>;

/// Type alias for validation rule results with span information.
pub type ValidationResult = Result<(), SpannedValidationError>;

// Implement From<ValidationError> for SpannedValidationError
// This allows using `?` with ValidationError in functions that return ValidationResult
impl From<super::error::ValidationError> for SpannedValidationError {
    fn from(error: super::error::ValidationError) -> Self {
        SpannedError::new(error)
    }
}

// Implement From<SpannedValidationError> for ValidationError
// This allows extracting the inner error (discarding span info) when needed for
// functions that return Result<_, ValidationError>
impl From<SpannedValidationError> for super::error::ValidationError {
    fn from(spanned: SpannedValidationError) -> Self {
        spanned.error
    }
}

/// Extension trait for ValidationError to easily convert to spanned errors.
pub trait ValidationErrorExt {
    /// Wraps this error as a spanned validation error (with no spans).
    fn into_spanned(self) -> SpannedValidationError;

    /// Wraps this error as a spanned validation error with a primary span at an ID.
    fn at_id(self, id: u32, message: impl Into<String>, span_map: Option<&SpanMap>) -> SpannedValidationError;

    /// Wraps this error with a primary span at an ID using the context's span map.
    fn at_id_ctx(self, id: impl Into<u32>, message: impl Into<String>, ctx: &super::context::ValidationContext<'_>) -> SpannedValidationError;

    /// Wraps this error with a primary span at an ID, and a secondary span at another ID.
    fn at_ids(
        self,
        primary_id: impl Into<u32>,
        primary_msg: impl Into<String>,
        secondary_id: impl Into<u32>,
        secondary_msg: impl Into<String>,
        ctx: &super::context::ValidationContext<'_>,
    ) -> SpannedValidationError;
}

impl ValidationErrorExt for super::error::ValidationError {
    fn into_spanned(self) -> SpannedValidationError {
        SpannedError::new(self)
    }

    fn at_id(self, id: u32, message: impl Into<String>, span_map: Option<&SpanMap>) -> SpannedValidationError {
        let mut spanned = SpannedError::new(self);
        if let Some(map) = span_map {
            if let Some(span) = map.get_id_span(id) {
                spanned = spanned.with_primary_span(span, message);
            }
        }
        spanned
    }

    fn at_id_ctx(self, id: impl Into<u32>, message: impl Into<String>, ctx: &super::context::ValidationContext<'_>) -> SpannedValidationError {
        self.at_id(id.into(), message, ctx.span_map)
    }

    fn at_ids(
        self,
        primary_id: impl Into<u32>,
        primary_msg: impl Into<String>,
        secondary_id: impl Into<u32>,
        secondary_msg: impl Into<String>,
        ctx: &super::context::ValidationContext<'_>,
    ) -> SpannedValidationError {
        let mut spanned = SpannedError::new(self);
        if let Some(map) = ctx.span_map {
            if let Some(span) = map.get_id_span(primary_id.into()) {
                spanned = spanned.with_primary_span(span, primary_msg);
            }
            if let Some(span) = map.get_id_span(secondary_id.into()) {
                spanned = spanned.with_secondary_span(span, secondary_msg);
            }
        }
        spanned
    }
}

/// Helper to create a spanned validation error.
///
/// # Example
///
/// ```ignore
/// use spirv_tools_core::validation::span::{spanned_err, SourceSpan};
/// use spirv_tools_core::validation::ValidationError;
///
/// let error = spanned_err(
///     ValidationError::TypeMismatch { ... },
///     SourceSpan::text(10, 5, 10, 25),
///     "mismatched types in operands",
/// );
/// ```
pub fn spanned_err<E>(error: E, span: SourceSpan, message: impl Into<String>) -> SpannedError<E> {
    SpannedError::new(error).with_primary_span(span, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location_variants() {
        let text_loc = SourceLocation::text(10, 5);
        assert!(matches!(text_loc, SourceLocation::Text(_)));

        let binary_loc = SourceLocation::binary(100);
        assert!(matches!(binary_loc, SourceLocation::Binary { word_offset: 100 }));

        let inst_loc = SourceLocation::instruction(50, Some(2));
        assert!(matches!(
            inst_loc,
            SourceLocation::Instruction {
                instruction_offset: 50,
                operand_index: Some(2)
            }
        ));
    }

    #[test]
    fn test_source_span_creation() {
        let text_span = SourceSpan::text(1, 0, 1, 10);
        assert!(text_span.text_position().is_some());
        assert!(text_span.word_offset().is_none());

        let binary_span = SourceSpan::binary(0, 5);
        assert!(binary_span.word_offset().is_some());
        assert!(binary_span.text_position().is_none());
    }

    #[test]
    fn test_labeled_span() {
        let span = SourceSpan::text(5, 10, 5, 20);
        let labeled = LabeledSpan::primary(span, "type mismatch");
        assert_eq!(labeled.label.kind, LabelKind::Primary);
        assert_eq!(labeled.label.message, "type mismatch");
    }

    #[test]
    fn test_spanned_error() {
        #[derive(Debug, Clone)]
        struct TestError(&'static str);
        impl fmt::Display for TestError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl std::error::Error for TestError {}

        let error = SpannedError::new(TestError("type mismatch"))
            .with_primary_span(SourceSpan::text(10, 5, 10, 25), "found f32, expected i32")
            .with_secondary_span(SourceSpan::text(5, 10, 5, 20), "expected type defined here")
            .with_note("types must be identical for arithmetic operations")
            .with_help("consider using OpConvertFToS to convert f32 to i32");

        assert_eq!(error.spans.len(), 2);
        assert_eq!(error.notes.len(), 1);
        assert_eq!(error.help.len(), 1);

        // Test display
        let display = format!("{}", error);
        assert!(display.contains("type mismatch"));
        assert!(display.contains("found f32, expected i32"));
    }

    #[test]
    fn test_span_map() {
        let mut map = SpanMap::new();
        assert!(map.is_empty());

        map.record_instruction(10, SourceSpan::binary(10, 15));
        map.record_id(1, SourceSpan::binary(10, 15));

        assert!(!map.is_empty());
        assert_eq!(map.instruction_count(), 1);
        assert_eq!(map.id_count(), 1);

        assert!(map.get_instruction_span(10).is_some());
        assert!(map.get_instruction_span(20).is_none());

        assert!(map.get_id_span(1).is_some());
        assert!(map.get_id_span(2).is_none());
    }
}
