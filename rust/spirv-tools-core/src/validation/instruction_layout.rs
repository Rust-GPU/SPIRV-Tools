#![allow(clippy::match_like_matches_macro)]
// Helper mapping for mode-setting instructions generated from the SPIR-V grammar.
include!(concat!(env!("OUT_DIR"), "/instruction_layout.rs"));
