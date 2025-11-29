#![allow(
    clippy::single_match,
    clippy::match_like_matches_macro,
    clippy::bad_bit_mask,
    clippy::erasing_op
)]
// Helper mapping of operand capability/extension requirements generated from the SPIR-V grammar.
include!(concat!(env!("OUT_DIR"), "/operand_requirements.rs"));
