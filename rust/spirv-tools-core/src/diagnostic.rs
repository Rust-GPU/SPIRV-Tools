//! Diagnostic primitives shared across the Rust port.

use core::fmt;
use std::borrow::Cow;

use crate::MessageLevel;

/// Zero-based source position used for diagnostics.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct MessagePosition {
    line: u32,
    column: u32,
    index: u32,
}

impl MessagePosition {
    /// Creates a new position using zero-based line, column, and byte index values.
    pub const fn new(line: u32, column: u32, index: u32) -> Self {
        Self {
            line,
            column,
            index,
        }
    }

    /// Returns the zero-based line.
    pub const fn line(self) -> u32 {
        self.line
    }

    /// Returns the zero-based column.
    pub const fn column(self) -> u32 {
        self.column
    }

    /// Returns the zero-based byte index in the input stream.
    pub const fn index(self) -> u32 {
        self.index
    }
}

/// Structured diagnostic message destined for the context's consumer.
#[derive(Clone, PartialEq, Eq)]
pub struct DiagnosticMessage<'a> {
    level: MessageLevel,
    source: Option<Cow<'a, str>>,
    position: MessagePosition,
    message: Cow<'a, str>,
}

impl<'a> DiagnosticMessage<'a> {
    /// Creates a new diagnostic for the given severity level.
    pub fn new(
        level: MessageLevel,
        position: MessagePosition,
        message: impl Into<Cow<'a, str>>,
    ) -> Self {
        Self {
            level,
            source: None,
            position,
            message: message.into(),
        }
    }

    /// Sets the source label describing the subsystem emitting this diagnostic.
    pub fn with_source(mut self, source: impl Into<Cow<'a, str>>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Returns the diagnostic severity level.
    pub const fn level(&self) -> MessageLevel {
        self.level
    }

    /// Returns the associated source label, if any.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Returns the recorded source position.
    pub const fn position(&self) -> MessagePosition {
        self.position
    }

    /// Returns the diagnostic payload.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Debug for DiagnosticMessage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiagnosticMessage")
            .field("level", &self.level)
            .field("source", &self.source)
            .field("position", &self.position)
            .field("message", &self.message)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{DiagnosticMessage, MessagePosition};
    use crate::MessageLevel;

    #[test]
    fn position_accessors_round_trip() {
        let position = MessagePosition::new(10, 20, 42);
        assert_eq!(position.line(), 10);
        assert_eq!(position.column(), 20);
        assert_eq!(position.index(), 42);
    }

    #[test]
    fn diagnostic_builders_capture_fields() {
        let message = DiagnosticMessage::new(
            MessageLevel::Warning,
            MessagePosition::new(1, 2, 3),
            "test message",
        )
        .with_source("assembler");

        assert_eq!(message.level(), MessageLevel::Warning);
        assert_eq!(message.source(), Some("assembler"));
        assert_eq!(message.position().line(), 1);
        assert_eq!(message.position().column(), 2);
        assert_eq!(message.position().index(), 3);
        assert_eq!(message.message(), "test message");
    }
}
