#![cfg(feature = "ffi")]

use std::ffi::CString;
use std::ptr;

use spirv_tools_ffi::{
    assemble, disassemble, AssemblerOptions, DisassemblerOptions, MessageConsumer,
    Result as FfiResult,
};

fn simple_module_lines() -> [&'static str; 9] {
    [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
}

fn simple_module_text() -> String {
    simple_module_lines().join("\n")
}

fn suppress_messages() -> MessageConsumer {
    MessageConsumer::new(|_, _, _, _| {})
}

#[test]
fn assemble_then_disassemble_round_trip_matches_original() {
    let text = simple_module_text();
    let c_text = CString::new(text.clone()).expect("c string");
    let mut asm_opts = AssemblerOptions::default();
    let mut dis_opts = DisassemblerOptions::default();

    let assembled: FfiResult<Vec<u32>> = assemble(&c_text, &mut asm_opts, suppress_messages());
    let words = assembled.expect("assemble");

    let disassembled =
        disassemble(&words, &mut dis_opts, suppress_messages()).expect("disassemble");
    let round_trip = disassembled.to_string_lossy();

    assert_eq!(
        text.trim(),
        round_trip.trim(),
        "FFI asm/dis round trip differed"
    );
}

#[test]
fn assembler_rejects_invalid_text_like_cpp() {
    // Missing OpMemoryModel line to force an error.
    let bad = "OpCapability Shader";
    let c_text = CString::new(bad).expect("c string");
    let mut asm_opts = AssemblerOptions::default();

    let result = assemble(&c_text, &mut asm_opts, suppress_messages());
    assert!(
        result.is_err(),
        "expected assembler to fail on invalid text"
    );
}

#[test]
fn disassembler_rejects_invalid_binary_like_cpp() {
    // Not enough words to be a valid SPIR-V binary.
    let invalid_words: [u32; 2] = [0x0302_023, 0xDEAD_BEEFu32];
    let mut dis_opts = DisassemblerOptions::default();

    let result = disassemble(&invalid_words, &mut dis_opts, suppress_messages());
    assert!(
        result.is_err(),
        "expected disassembler to fail on invalid binary"
    );
}
