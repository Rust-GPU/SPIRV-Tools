#![allow(missing_docs)]

use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[doc = "Options controlling the SPIR-V text-to-binary assembler pipeline.\n\nThese mirror `spv_text_to_binary_options_t`. The numeric values match the\nC definitions exactly so they can be passed across FFI boundaries without\nconversion."]
    pub struct TextToBinaryOptions: u32 {
        #[doc = "No optional behaviours are enabled."]
        const NONE = 1 << 0;
        #[doc = "Preserve explicit numeric IDs from the source text."]
        const PRESERVE_NUMERIC_IDS = 1 << 1;
    }
}

impl From<TextToBinaryOptions> for u32 {
    fn from(value: TextToBinaryOptions) -> Self {
        value.bits()
    }
}

impl From<u32> for TextToBinaryOptions {
    fn from(value: u32) -> Self {
        TextToBinaryOptions::from_bits_truncate(value)
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[doc = "Options controlling the SPIR-V binary-to-text disassembler pipeline.\n\nThese mirror `spv_binary_to_text_options_t`. Values mirror the C\nbitfield so they can be exchanged over FFI without conversion."]
    pub struct BinaryToTextOptions: u32 {
        const NONE = 1 << 0;
        const PRINT = 1 << 1;
        const COLOR = 1 << 2;
        const INDENT = 1 << 3;
        const SHOW_BYTE_OFFSET = 1 << 4;
        const NO_HEADER = 1 << 5;
        const FRIENDLY_NAMES = 1 << 6;
        const COMMENT = 1 << 7;
        const NESTED_INDENT = 1 << 8;
        const REORDER_BLOCKS = 1 << 9;
        const HEX = 1 << 10;
    }
}

impl From<BinaryToTextOptions> for u32 {
    fn from(value: BinaryToTextOptions) -> Self {
        value.bits()
    }
}

impl From<u32> for BinaryToTextOptions {
    fn from(value: u32) -> Self {
        BinaryToTextOptions::from_bits_truncate(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryToTextOptions, TextToBinaryOptions};

    #[test]
    fn text_to_binary_round_trip() {
        let flags = TextToBinaryOptions::NONE | TextToBinaryOptions::PRESERVE_NUMERIC_IDS;
        let raw: u32 = flags.into();
        assert_eq!(TextToBinaryOptions::from(raw), flags);
    }

    #[test]
    fn binary_to_text_round_trip() {
        let flags = BinaryToTextOptions::COMMENT
            | BinaryToTextOptions::FRIENDLY_NAMES
            | BinaryToTextOptions::SHOW_BYTE_OFFSET;
        let raw: u32 = flags.into();
        assert_eq!(BinaryToTextOptions::from(raw), flags);
    }

    #[test]
    fn binary_to_text_defaults_do_not_enable_friendly_names() {
        let raw = BinaryToTextOptions::INDENT.bits()
            | BinaryToTextOptions::NESTED_INDENT.bits()
            | BinaryToTextOptions::NO_HEADER.bits()
            | BinaryToTextOptions::COMMENT.bits();
        let flags = BinaryToTextOptions::from(raw);
        assert!(!flags.contains(BinaryToTextOptions::FRIENDLY_NAMES));
    }

    #[test]
    fn unknown_bits_are_dropped() {
        let raw = 0xFFFF_FFFF;
        let flags = BinaryToTextOptions::from(raw);
        assert_eq!(u32::from(flags), BinaryToTextOptions::all().bits());
    }
}
