//! Shared string-literal helpers for the SPIR-V text format.

/// Parses a quoted string literal payload from assembly text, stripping escape markers.
///
/// The SPIR-V assembler treats `\` as an escape prefix but does not interpret C-style
/// sequences. Instead, the backslash is removed and the following character is emitted
/// verbatim. This mirrors the legacy C++ `spvTextToLiteral` behaviour so round-trips
/// canonicalize sequences like `\"` to `"`, `\\` to `\`, and `\foo` to `foo`.
pub fn parse_string_literal(raw: &str) -> String {
    let mut decoded = String::with_capacity(raw.len());
    let mut escaping = false;
    for ch in raw.chars() {
        if ch == '\\' && !escaping {
            escaping = true;
            continue;
        }
        decoded.push(ch);
        escaping = false;
    }
    decoded
}

/// Renders a string literal for disassembly output, escaping quotes and backslashes.
pub fn render_string_literal(value: &str) -> String {
    let mut formatted = String::with_capacity(value.len() + 2);
    formatted.push('"');
    for ch in value.chars() {
        if matches!(ch, '"' | '\\') {
            formatted.push('\\');
        }
        formatted.push(ch);
    }
    formatted.push('"');
    formatted
}

#[cfg(test)]
mod tests {
    use super::{parse_string_literal, render_string_literal};

    #[test]
    fn parse_strips_escape_prefixes() {
        assert_eq!(parse_string_literal("\\foo"), "foo");
        assert_eq!(parse_string_literal("\\\\foo"), "\\foo");
        assert_eq!(parse_string_literal("\\\"quoted\\\""), "\"quoted\"");
    }

    #[test]
    fn parse_preserves_newlines_and_utf8() {
        assert_eq!(parse_string_literal("\\\nfoo"), "\nfoo");
        assert_eq!(parse_string_literal("\\亲"), "亲");
    }

    #[test]
    fn render_re_escapes_quotes_and_backslashes() {
        assert_eq!(render_string_literal("foo"), "\"foo\"");
        assert_eq!(render_string_literal("\"quoted\""), "\"\\\"quoted\\\"\"");
        assert_eq!(render_string_literal("\\path"), "\"\\\\path\"");
    }
}
