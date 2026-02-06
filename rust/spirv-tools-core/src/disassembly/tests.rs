use super::{disassemble_binary, FRIENDLY_NAME_SAMPLE_BINARY};
use crate::assembly::{
    assemble_text, assemble_text_with_options, BinaryToTextOptions, TextToBinaryOptions,
};
use crate::target_env::TargetEnv;
use rspirv::binary::Assemble;
use rspirv::dr::{self, Builder};
use rspirv::spirv::{
    AccessQualifier, AddressingModel, BuiltIn, Capability, Decoration, ExecutionModel,
    FunctionControl, MemoryModel, SelectionControl, StorageClass,
};

const CONDITIONAL_EXTENSION_SAMPLE_BINARY: &[u32] = &[
    0x07230203, 0x00010600, 0x00000000, 0x00000003, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
    0x00000000, 0x00000001, 0x00020014, 0x00000001, 0x00030031, 0x00000001, 0x00000002, 0x00091868,
    0x00000002, 0x5f565053, 0x45544e49, 0x75665f4c, 0x6974636e, 0x765f6e6f, 0x61697261, 0x0073746e,
];

fn disassemble_with_options(words: &[u32], options: BinaryToTextOptions) -> String {
    disassemble_binary(words, options).expect("disassemble")
}

fn entry_point_module(name_literal: &str) -> String {
    format!(
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %3 {name_literal}
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd
"
    )
}

fn encode_and_decode_fixture(
    text: &str,
    disassemble_options: BinaryToTextOptions,
    assemble_options: TextToBinaryOptions,
) -> String {
    let binary = assemble_text_with_options(text, TargetEnv::Universal1_0, assemble_options)
        .expect("assemble");
    disassemble_binary(
        &binary,
        disassemble_options | BinaryToTextOptions::NO_HEADER,
    )
    .expect("disassemble")
}

fn round_trip_entry_point_literal(name_literal: &str, expected_literal: &str) {
    let before = entry_point_module(name_literal);
    let binary = assemble_text(&before).expect("assemble");
    let text = disassemble_binary(&binary, BinaryToTextOptions::NO_HEADER).expect("disassemble");
    assert!(
        text.contains("OpEntryPoint Vertex"),
        "disassembly missing entry point: {text:?}"
    );
    assert!(
        text.contains(expected_literal),
        "expected literal {expected_literal:?} in {text:?}"
    );
    assert!(
        !text.contains(name_literal),
        "disassembly retained escape prefix: {text:?}"
    );
}

#[test]
fn disassembles_simple_module() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble text");
    let options =
        BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::COMMENT | BinaryToTextOptions::INDENT;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(disassembled.trim_start().starts_with("OpCapability Shader"));
    assert!(disassembled.contains("OpFunctionEnd"));
}

#[test]
fn disassembly_respects_no_header_option() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble text");
    let options = BinaryToTextOptions::NONE | BinaryToTextOptions::NO_HEADER;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(!disassembled.starts_with(";"));
    assert!(disassembled.starts_with("OpCapability"));
}

#[test]
fn invalid_binary_reports_parse_error() {
    let binary = vec![0xDEAD_BEEFu32];
    let error = disassemble_binary(&binary, BinaryToTextOptions::NO_HEADER)
        .expect_err("expected parse error");
    match error {
        super::DisassemblyError::Parse { message, .. } => assert!(!message.is_empty()),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn disassembly_accepts_print_option() {
    let text = "OpCapability Shader";
    let binary = assemble_text(text).expect("assemble text");
    let options = BinaryToTextOptions::PRINT | BinaryToTextOptions::NO_HEADER;
    // PRINT is handled by the caller; the disassembler should still succeed.
    let _ = disassemble_binary(&binary, options).expect("disassemble");
}

#[test]
fn supports_options_covers_zero_and_no_header() {
    assert!(super::supports_options(BinaryToTextOptions::empty()));
    assert!(super::supports_options(BinaryToTextOptions::NO_HEADER));
    assert!(super::supports_options(BinaryToTextOptions::INDENT));
    assert!(super::supports_options(BinaryToTextOptions::FRIENDLY_NAMES));
    assert!(super::supports_options(BinaryToTextOptions::COMMENT));
    assert!(super::supports_options(BinaryToTextOptions::REORDER_BLOCKS));
    assert!(super::supports_options(BinaryToTextOptions::COLOR));
    assert!(super::supports_options(BinaryToTextOptions::HEX));
}

#[test]
fn disassembly_appends_byte_offsets() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical Simple",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble text");
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::SHOW_BYTE_OFFSET;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    let expected = [
        "OpCapability Shader                                 ; 0x00000014",
        "OpMemoryModel Logical Simple                        ; 0x0000001c",
        "%1 = OpTypeVoid                                     ; 0x00000028",
        "%2 = OpTypeFunction %1                              ; 0x00000030",
    ]
    .join("\n")
        + "\n";
    assert_eq!(disassembled, expected);
}

#[test]
fn disassembly_applies_indent_formatting() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical Simple",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble text");
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::INDENT;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");

    let expected = [
        "               OpCapability Shader",
        "               OpMemoryModel Logical Simple",
        "          %1 = OpTypeVoid",
        "          %2 = OpTypeFunction %1",
        "          %3 = OpFunction %1 None %2",
        "          %4 = OpLabel",
        "               OpReturn",
        "               OpFunctionEnd",
    ]
    .join("\n")
        + "\n";
    assert_eq!(disassembled, expected);
}

#[test]
fn disassembly_uses_friendly_names_when_available() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let void_fn = builder.type_function(void, vec![]);
    let main = builder
        .begin_function(void, None, FunctionControl::NONE, void_fn)
        .expect("begin function");
    builder.name(main, "my_main");
    builder.entry_point(ExecutionModel::Vertex, main, "main", Vec::new());
    builder.begin_block(None).expect("block");
    builder.ret().expect("return");
    builder.end_function().expect("end function");
    let module = builder.module();
    let binary = module.assemble();
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(disassembled.contains("%my_main = OpFunction"));
    assert!(disassembled.contains("OpEntryPoint Vertex %my_main \"main\""));
}

#[test]
fn disassembly_nested_indent_tracks_block_depth() {
    let binary = build_selection_module(false);
    let options = BinaryToTextOptions::NO_HEADER
        | BinaryToTextOptions::INDENT
        | BinaryToTextOptions::NESTED_INDENT
        | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    let lines: Vec<&str> = disassembled.lines().collect();
    let entry_idx = lines
        .iter()
        .position(|line| line.contains("%entry ="))
        .expect("entry label");
    let then_idx = lines
        .iter()
        .position(|line| line.contains("%then ="))
        .expect("then label");
    let entry_spaces = spaces_after_equals(lines[entry_idx]).expect("entry spaces");
    let then_spaces = spaces_after_equals(lines[then_idx]).expect("then spaces");
    assert!(then_spaces > entry_spaces);
    let body_idx = lines
        .iter()
        .enumerate()
        .skip(then_idx + 1)
        .find(|(_, line)| {
            !line.trim().is_empty() && !line.contains("OpLabel") && line.contains("Op")
        })
        .map(|(idx, _)| idx)
        .expect("body instruction");
    assert!(leading_spaces(lines[body_idx]) > leading_spaces(lines[then_idx]));
}

#[test]
fn disassembly_emits_decoration_comments() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let float = builder.type_float(32, None);
    let ptr = builder.type_pointer(None, StorageClass::UniformConstant, float);
    let var = builder.variable(ptr, None, StorageClass::UniformConstant, None);
    builder.decorate(
        var,
        Decoration::DescriptorSet,
        [dr::Operand::LiteralBit32(0)],
    );
    builder.decorate(var, Decoration::Binding, [dr::Operand::LiteralBit32(1)]);
    let void = builder.type_void();
    let void_fn = builder.type_function(void, vec![]);
    builder
        .begin_function(void, None, FunctionControl::NONE, void_fn)
        .expect("function");
    builder.begin_block(None).expect("block");
    builder.ret().expect("return");
    builder.end_function().expect("end");
    let module = builder.module();
    let binary = module.assemble();
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::COMMENT;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(disassembled.contains("DescriptorSet 0"));
    assert!(disassembled.contains("Binding 1"));
}

#[test]
fn disassembly_applies_color_formatting() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let void_fn = builder.type_function(void, vec![]);
    builder
        .begin_function(void, None, FunctionControl::NONE, void_fn)
        .expect("function");
    builder.begin_block(None).expect("block");
    builder.ret().expect("return");
    builder.end_function().expect("end");
    let module = builder.module();
    let binary = module.assemble();
    let options = BinaryToTextOptions::NO_HEADER
        | BinaryToTextOptions::INDENT
        | BinaryToTextOptions::COLOR
        | BinaryToTextOptions::COMMENT
        | BinaryToTextOptions::SHOW_BYTE_OFFSET;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(disassembled.contains(super::COLOR_BLUE));
    assert!(disassembled.contains(super::COLOR_GREY));
}

#[test]
fn disassembly_reorders_blocks_when_requested() {
    let binary = build_selection_module(false);
    let base_options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let unreordered =
        disassemble_binary(&binary, base_options).expect("disassemble without reorder");
    let reordered = disassemble_binary(&binary, base_options | BinaryToTextOptions::REORDER_BLOCKS)
        .expect("disassemble with reorder");

    let label_index = |text: &str, needle: &str| -> usize {
        text.lines()
            .position(|line| line.contains(needle) && line.contains("OpLabel"))
            .unwrap_or(usize::MAX)
    };

    let unreordered_then = label_index(&unreordered, "%then");
    let unreordered_merge = label_index(&unreordered, "%merge");
    assert!(unreordered_merge < unreordered_then);

    let reordered_entry = label_index(&reordered, "%entry");
    let reordered_then = label_index(&reordered, "%then");
    let reordered_merge = label_index(&reordered, "%merge");
    assert!(reordered_then < reordered_merge);
    assert!(reordered_entry < reordered_then);
}

#[test]
fn disassembly_matches_indent_fixture_sample() {
    let input = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeInt 32 0",
        "%2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10 ; force IDs into double digits",
        "%11 = OpConstant %1 42",
        "OpStore %2 %3 Aligned|Volatile 4 ; bogus, but not indented",
    ]
    .join("\n");
    let expected = [
        "               OpCapability Shader",
        "               OpMemoryModel Logical GLSL450",
        "          %1 = OpTypeInt 32 0",
        "          %2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10",
        "         %11 = OpConstant %1 42",
        "               OpStore %2 %3 Volatile|Aligned 4",
    ]
    .join("\n")
        + "\n";
    let output = encode_and_decode_fixture(
        &input,
        BinaryToTextOptions::INDENT,
        TextToBinaryOptions::NONE,
    );
    assert_eq!(output, expected);
}

#[test]
fn disassembly_matches_indent_fixture_nested_if() {
    let input = [
        "OpCapability Shader",
        "OpMemoryModel Logical Simple",
        "OpEntryPoint Fragment %100 \"main\"",
        "OpExecutionMode %100 OriginUpperLeft",
        "OpName %var \"var\"",
        "%void = OpTypeVoid",
        "%3 = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%5 = OpConstantNull %bool",
        "%true = OpConstantTrue %bool",
        "%false = OpConstantFalse %bool",
        "%uint = OpTypeInt 32 0",
        "%int = OpTypeInt 32 1",
        "%uint_42 = OpConstant %uint 42",
        "%int_42 = OpConstant %int 42",
        "%13 = OpTypeFunction %uint",
        "%uint_0 = OpConstant %uint 0",
        "%uint_1 = OpConstant %uint 1",
        "%uint_2 = OpConstant %uint 2",
        "%uint_3 = OpConstant %uint 3",
        "%uint_4 = OpConstant %uint 4",
        "%uint_5 = OpConstant %uint 5",
        "%uint_6 = OpConstant %uint 6",
        "%uint_7 = OpConstant %uint 7",
        "%uint_8 = OpConstant %uint 8",
        "%uint_10 = OpConstant %uint 10",
        "%uint_20 = OpConstant %uint 20",
        "%uint_30 = OpConstant %uint 30",
        "%uint_40 = OpConstant %uint 40",
        "%uint_50 = OpConstant %uint 50",
        "%uint_90 = OpConstant %uint 90",
        "%uint_99 = OpConstant %uint 99",
        "%_ptr_Private_uint = OpTypePointer Private %uint",
        "%var = OpVariable %_ptr_Private_uint Private",
        "%uint_999 = OpConstant %uint 999",
        "%100 = OpFunction %void None %3",
        "%10 = OpLabel",
        "OpStore %var %uint_0",
        "OpSelectionMerge %99 None",
        "OpBranchConditional %5 %30 %40",
        "%30 = OpLabel",
        "OpStore %var %uint_1",
        "OpBranch %99",
        "%40 = OpLabel",
        "OpStore %var %uint_2",
        "OpBranch %99",
        "%99 = OpLabel",
        "OpStore %var %uint_999",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let expected = [
        "               OpCapability Shader",
        "               OpMemoryModel Logical Simple",
        "               OpEntryPoint Fragment %100 \"main\"",
        "               OpExecutionMode %100 OriginUpperLeft",
        "               OpName %1 \"var\"",
        "          %2 = OpTypeVoid",
        "          %3 = OpTypeFunction %2",
        "          %4 = OpTypeBool",
        "          %5 = OpConstantNull %4",
        "          %6 = OpConstantTrue %4",
        "          %7 = OpConstantFalse %4",
        "          %8 = OpTypeInt 32 0",
        "          %9 = OpTypeInt 32 1",
        "         %11 = OpConstant %8 42",
        "         %12 = OpConstant %9 42",
        "         %13 = OpTypeFunction %8",
        "         %14 = OpConstant %8 0",
        "         %15 = OpConstant %8 1",
        "         %16 = OpConstant %8 2",
        "         %17 = OpConstant %8 3",
        "         %18 = OpConstant %8 4",
        "         %19 = OpConstant %8 5",
        "         %20 = OpConstant %8 6",
        "         %21 = OpConstant %8 7",
        "         %22 = OpConstant %8 8",
        "         %23 = OpConstant %8 10",
        "         %24 = OpConstant %8 20",
        "         %25 = OpConstant %8 30",
        "         %26 = OpConstant %8 40",
        "         %27 = OpConstant %8 50",
        "         %28 = OpConstant %8 90",
        "         %29 = OpConstant %8 99",
        "         %31 = OpTypePointer Private %8",
        "          %1 = OpVariable %31 Private",
        "         %32 = OpConstant %8 999",
        "        %100 = OpFunction %2 None %3",
        "",
        "         %10 = OpLabel",
        "                 OpStore %1 %14",
        "                 OpSelectionMerge %99 None",
        "                 OpBranchConditional %5 %30 %40",
        "",
        "         %30 =     OpLabel",
        "                     OpStore %1 %15",
        "                     OpBranch %99",
        "",
        "         %40 =     OpLabel",
        "                     OpStore %1 %16",
        "                     OpBranch %99",
        "",
        "         %99 = OpLabel",
        "                 OpStore %1 %32",
        "                 OpReturn",
        "               OpFunctionEnd",
    ]
    .join("\n")
        + "\n";
    let output = encode_and_decode_fixture(
        &input,
        BinaryToTextOptions::INDENT | BinaryToTextOptions::NESTED_INDENT,
        TextToBinaryOptions::PRESERVE_NUMERIC_IDS,
    );
    assert_eq!(output, expected);
}

#[test]
fn disassembly_matches_indent_fixture_reordered_if() {
    let input = [
        "               OpCapability Shader",
        "               OpMemoryModel Logical Simple",
        "               OpEntryPoint Fragment %100 \"main\"",
        "               OpExecutionMode %100 OriginUpperLeft",
        "               OpName %1 \"var\"",
        "          %2 = OpTypeVoid",
        "          %3 = OpTypeFunction %2",
        "          %4 = OpTypeBool",
        "          %5 = OpConstantNull %4",
        "          %6 = OpConstantTrue %4",
        "          %7 = OpConstantFalse %4",
        "          %8 = OpTypeInt 32 0",
        "          %9 = OpTypeInt 32 1",
        "         %11 = OpConstant %8 42",
        "         %12 = OpConstant %9 42",
        "         %13 = OpTypeFunction %8",
        "         %14 = OpConstant %8 0",
        "         %15 = OpConstant %8 1",
        "         %16 = OpConstant %8 2",
        "         %17 = OpConstant %8 3",
        "         %18 = OpConstant %8 4",
        "         %19 = OpConstant %8 5",
        "         %21 = OpConstant %8 6",
        "         %22 = OpConstant %8 7",
        "         %23 = OpConstant %8 8",
        "         %24 = OpConstant %8 10",
        "         %25 = OpConstant %8 20",
        "         %26 = OpConstant %8 30",
        "         %27 = OpConstant %8 40",
        "         %28 = OpConstant %8 50",
        "         %29 = OpConstant %8 90",
        "         %31 = OpConstant %8 99",
        "         %32 = OpTypePointer Private %8",
        "          %1 = OpVariable %32 Private",
        "         %33 = OpConstant %8 999",
        "        %100 = OpFunction %2 None %3",
        "         %10 = OpLabel",
        "               OpSelectionMerge %99 None",
        "               OpBranchConditional %5 %20 %50",
        "         %99 = OpLabel",
        "               OpReturn",
        "         %20 = OpLabel",
        "               OpSelectionMerge %49 None",
        "               OpBranchConditional %5 %30 %40",
        "         %49 = OpLabel",
        "               OpBranch %99",
        "         %40 = OpLabel",
        "               OpBranch %49",
        "         %30 = OpLabel",
        "               OpBranch %49",
        "         %50 = OpLabel",
        "               OpSelectionMerge %79 None",
        "               OpBranchConditional %5 %60 %70",
        "         %79 = OpLabel",
        "               OpBranch %99",
        "         %60 = OpLabel",
        "               OpBranch %79",
        "         %70 = OpLabel",
        "               OpBranch %79",
        "               OpFunctionEnd",
    ]
    .join("\n");
    let expected = [
        "               OpCapability Shader",
        "               OpMemoryModel Logical Simple",
        "               OpEntryPoint Fragment %100 \"main\"",
        "               OpExecutionMode %100 OriginUpperLeft",
        "               OpName %1 \"var\"",
        "          %2 = OpTypeVoid",
        "          %3 = OpTypeFunction %2",
        "          %4 = OpTypeBool",
        "          %5 = OpConstantNull %4",
        "          %6 = OpConstantTrue %4",
        "          %7 = OpConstantFalse %4",
        "          %8 = OpTypeInt 32 0",
        "          %9 = OpTypeInt 32 1",
        "         %11 = OpConstant %8 42",
        "         %12 = OpConstant %9 42",
        "         %13 = OpTypeFunction %8",
        "         %14 = OpConstant %8 0",
        "         %15 = OpConstant %8 1",
        "         %16 = OpConstant %8 2",
        "         %17 = OpConstant %8 3",
        "         %18 = OpConstant %8 4",
        "         %19 = OpConstant %8 5",
        "         %21 = OpConstant %8 6",
        "         %22 = OpConstant %8 7",
        "         %23 = OpConstant %8 8",
        "         %24 = OpConstant %8 10",
        "         %25 = OpConstant %8 20",
        "         %26 = OpConstant %8 30",
        "         %27 = OpConstant %8 40",
        "         %28 = OpConstant %8 50",
        "         %29 = OpConstant %8 90",
        "         %31 = OpConstant %8 99",
        "         %32 = OpTypePointer Private %8",
        "          %1 = OpVariable %32 Private",
        "         %33 = OpConstant %8 999",
        "        %100 = OpFunction %2 None %3",
        "         %10 = OpLabel",
        "               OpSelectionMerge %99 None",
        "               OpBranchConditional %5 %20 %50",
        "         %20 = OpLabel",
        "               OpSelectionMerge %49 None",
        "               OpBranchConditional %5 %30 %40",
        "         %30 = OpLabel",
        "               OpBranch %49",
        "         %40 = OpLabel",
        "               OpBranch %49",
        "         %49 = OpLabel",
        "               OpBranch %99",
        "         %50 = OpLabel",
        "               OpSelectionMerge %79 None",
        "               OpBranchConditional %5 %60 %70",
        "         %60 = OpLabel",
        "               OpBranch %79",
        "         %70 = OpLabel",
        "               OpBranch %79",
        "         %79 = OpLabel",
        "               OpBranch %99",
        "         %99 = OpLabel",
        "               OpReturn",
        "               OpFunctionEnd",
    ]
    .join("\n")
        + "\n";
    let output = encode_and_decode_fixture(
        &input,
        BinaryToTextOptions::INDENT | BinaryToTextOptions::REORDER_BLOCKS,
        TextToBinaryOptions::PRESERVE_NUMERIC_IDS,
    );
    assert_eq!(output, expected);
}

const REORDER_FIXTURE_BINARY: &[u32] = &[
    0x07230203, 0x00010600, 0x00070000, 0x0000002a, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
    0x00000000, 0x00000000, 0x0005000f, 0x00000004, 0x00000001, 0x6e69616d, 0x00000000, 0x00030010,
    0x00000001, 0x00000007, 0x00030005, 0x00000002, 0x00007261, 0x00020013, 0x00000003, 0x00030021,
    0x00000004, 0x00000003, 0x00020014, 0x00000005, 0x0003002e, 0x00000005, 0x00000006, 0x00030029,
    0x00000005, 0x00000007, 0x0003002a, 0x00000005, 0x00000008, 0x00040015, 0x00000009, 0x00000020,
    0x00000000, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x0004002b, 0x00000009, 0x0000000b,
    0x0000002a, 0x0004002b, 0x0000000a, 0x0000000c, 0x0000002a, 0x00030021, 0x0000000d, 0x00000009,
    0x0004002b, 0x00000009, 0x0000000e, 0x00000000, 0x0004002b, 0x00000009, 0x0000000f, 0x00000001,
    0x0004002b, 0x00000009, 0x00000010, 0x00000002, 0x0004002b, 0x00000009, 0x00000011, 0x00000003,
    0x0004002b, 0x00000009, 0x00000012, 0x00000004, 0x0004002b, 0x00000009, 0x00000013, 0x00000005,
    0x0004002b, 0x00000009, 0x00000014, 0x00000006, 0x0004002b, 0x00000009, 0x00000015, 0x00000007,
    0x0004002b, 0x00000009, 0x00000016, 0x00000008, 0x0004002b, 0x00000009, 0x00000017, 0x0000000a,
    0x0004002b, 0x00000009, 0x00000018, 0x00000014, 0x0004002b, 0x00000009, 0x00000019, 0x0000001e,
    0x0004002b, 0x00000009, 0x0000001a, 0x00000028, 0x0004002b, 0x00000009, 0x0000001b, 0x00000032,
    0x0004002b, 0x00000009, 0x0000001c, 0x0000005a, 0x0004002b, 0x00000009, 0x0000001d, 0x00000063,
    0x00040020, 0x0000001e, 0x00000006, 0x00000009, 0x0004003b, 0x0000001e, 0x00000002, 0x00000006,
    0x0004002b, 0x00000009, 0x0000001f, 0x000003e7, 0x00050036, 0x00000003, 0x00000064, 0x00000000,
    0x00000004, 0x000200f8, 0x00000020, 0x000300f7, 0x00000021, 0x00000000, 0x000400fa, 0x00000006,
    0x00000022, 0x00000023, 0x000200f8, 0x00000022, 0x000300f7, 0x00000024, 0x00000000, 0x000400fa,
    0x00000006, 0x00000025, 0x00000026, 0x000200f8, 0x00000025, 0x000200f9, 0x00000024, 0x000200f8,
    0x00000026, 0x000200f9, 0x00000024, 0x000200f8, 0x00000024, 0x000200f9, 0x00000021, 0x000200f8,
    0x00000023, 0x000300f7, 0x00000027, 0x00000000, 0x000400fa, 0x00000006, 0x00000028, 0x00000029,
    0x000200f8, 0x00000028, 0x000200f9, 0x00000027, 0x000200f8, 0x00000029, 0x000200f9, 0x00000027,
    0x000200f8, 0x00000027, 0x000200f9, 0x00000021, 0x000200f8, 0x00000021, 0x000100fd, 0x00010038,
];

#[test]
fn reorder_blocks_matches_indent_fixture() {
    let module = rspirv::dr::load_words(REORDER_FIXTURE_BINARY).expect("load module");
    let function = module.functions.first().expect("function");
    let order = super::block_analysis::reorder_function_blocks(function);
    let expected: Vec<usize> = (0..function.blocks.len()).collect();
    assert_eq!(order, expected);
}

#[test]
fn disassembly_print_option_writes_stdout() {
    let _ = take_print_log();
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let void_fn = builder.type_function(void, vec![]);
    builder
        .begin_function(void, None, FunctionControl::NONE, void_fn)
        .expect("function");
    builder.begin_block(None).expect("entry block");
    builder.ret().expect("return");
    builder.end_function().expect("end function");
    let module = builder.module();
    let binary = module.assemble();
    let options = BinaryToTextOptions::NO_HEADER
        | BinaryToTextOptions::PRINT
        | BinaryToTextOptions::FRIENDLY_NAMES;
    let text = disassemble_binary(&binary, options).expect("disassemble");
    assert!(text.is_empty());
    let printed = take_print_log();
    assert!(!printed.is_empty());
    assert!(printed.iter().any(|entry| entry.contains("OpFunction")));
}

#[test]
fn disassembly_formats_literals_as_hex() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%uint = OpTypeInt 32 0",
        "%val = OpConstant %uint 42",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::HEX;
    let output = disassemble_binary(&binary, options).expect("disassemble");
    assert!(output.contains("0x0000002a"), "{output}");
    assert!(!output.contains(" 42"));
}

#[test]
fn format_literal_string_preserves_escape_sequences() {
    let formatted = super::formatting::format_literal_string("foo\\nbar");
    assert_eq!(formatted, r#""foo\\nbar""#);
}

#[test]
fn format_literal_string_reescapes_quotes() {
    let formatted = super::formatting::format_literal_string("say \"hi\"");
    assert_eq!(formatted, r#""say \"hi\"""#);
}

#[test]
fn round_trips_string_literal_stripping_escape_prefix() {
    round_trip_entry_point_literal("\"\\foo\"", "\"foo\"");
}

#[test]
fn round_trips_string_literal_with_leading_newline() {
    round_trip_entry_point_literal("\"\\\nfoo\"", "\"\nfoo\"");
}

#[test]
fn round_trips_string_literal_with_utf8_escape_prefix() {
    round_trip_entry_point_literal("\"\\亲\"", "\"亲\"");
}

#[test]
fn disassembly_formats_literal_strings_with_embedded_newlines() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let fn_ty = builder.type_function(void, vec![]);
    let func = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .expect("function");
    builder.begin_block(None).expect("entry block");
    builder.ret().expect("return");
    builder.end_function().expect("end function");
    builder.name(func, "foo\nbar");
    let module = builder.module();
    let binary = module.assemble();
    let text = disassemble_binary(&binary, BinaryToTextOptions::NO_HEADER).expect("disassemble");
    assert!(text.contains("\"foo\nbar\""), "{text:?}");
}

#[test]
fn friendly_name_builder_assigns_type_names() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let int = builder.type_int(32, 1);
    builder.constant_bit32(int, 1);
    let module = builder.module();
    let type_table = super::types::TypeTable::from_module(&module);
    let names = super::names::FriendlyNameTable::from_module(&module, &type_table);
    assert_eq!(names.lookup(void), Some("void"));
    assert_eq!(names.lookup(int), Some("int"));
}

#[test]
fn friendly_names_match_opt_fixture() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let int = builder.type_int(32, 1);
    let ptr = builder.type_pointer(None, StorageClass::Function, int);
    let fn_type = builder.type_function(void, vec![ptr]);
    let main = builder
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .expect("function");
    builder.name(main, "main");
    builder.function_parameter(ptr).expect("param");
    builder.begin_block(None).expect("block");
    builder.ret().expect("ret");
    builder.end_function().expect("end");
    let binary = builder.module().assemble();
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(
        disassembled.contains("%int = OpTypeInt 32 1"),
        "{disassembled}"
    );
    assert!(
        disassembled.contains("%_ptr_Function_int = OpTypePointer Function %int"),
        "{disassembled}"
    );
}

#[test]
fn friendly_names_match_binary_to_text_fixture() {
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled =
        disassemble_binary(FRIENDLY_NAME_SAMPLE_BINARY, options).expect("disassemble");
    assert!(
        disassembled.contains("%uint = OpTypeInt 32 0"),
        "{disassembled}"
    );
    assert!(
        disassembled.contains("%uint_42 = OpConstant %uint 42"),
        "{disassembled}"
    );
}

#[test]
fn disassembly_respects_raw_id_option() {
    let options = BinaryToTextOptions::NO_HEADER;
    let disassembled =
        disassemble_binary(FRIENDLY_NAME_SAMPLE_BINARY, options).expect("disassemble");
    assert!(
        disassembled.contains("%1 = OpTypeInt 32 0"),
        "{disassembled}"
    );
}

#[test]
fn disassembly_omits_newline_when_output_empty() {
    let binary = vec![0x0723_0203, 0x0001_0000, 0, 1, 0];
    let options = BinaryToTextOptions::NO_HEADER;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert_eq!(disassembled, "");
}

#[test]
fn indent_alignment_matches_legacy_formatter() {
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::INDENT;
    let disassembled =
        disassemble_binary(FRIENDLY_NAME_SAMPLE_BINARY, options).expect("disassemble");
    let type_line = disassembled
        .lines()
        .find(|line| line.contains("OpTypeInt 32 0"))
        .expect("type line present");
    assert!(type_line.starts_with("          %1 ="), "{type_line}");
}

#[test]
fn friendly_names_match_pass_fixture_sample() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    builder.name(void, "void_t");
    let fn_ty = builder.type_function(void, vec![]);
    builder.name(fn_ty, "fn_t");
    let bool_ty = builder.type_bool();
    let out_ptr = builder.type_pointer(None, StorageClass::Output, bool_ty);
    let flag = builder.variable(out_ptr, None, StorageClass::Output, None);
    builder.name(flag, "flag");
    builder.decorate(flag, Decoration::RelaxedPrecision, []);

    let main_fn = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .expect("function");
    builder.name(main_fn, "main");
    builder.begin_block(None).expect("block");
    builder.ret().expect("return");
    builder.end_function().expect("end");
    builder.entry_point(ExecutionModel::Fragment, main_fn, "main", vec![flag]);
    builder.decorate(main_fn, Decoration::RelaxedPrecision, []);

    let module = builder.module();
    let binary = module.assemble();
    let options = BinaryToTextOptions::NO_HEADER
        | BinaryToTextOptions::COMMENT
        | BinaryToTextOptions::INDENT
        | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    let expected = r#"               OpCapability Shader
               OpMemoryModel Logical Simple
               OpEntryPoint Fragment %main "main" %flag

               ; Debug Information
               OpName %void_t "void_t"              ; id %1
               OpName %fn_t "fn_t"                  ; id %2
               OpName %flag "flag"                  ; id %5
               OpName %main "main"                  ; id %6

               ; Annotations
               OpDecorate %flag RelaxedPrecision
               OpDecorate %main RelaxedPrecision

               ; Types, variables and constants
     %void_t = OpTypeVoid
       %fn_t = OpTypeFunction %void_t
       %bool = OpTypeBool
%_ptr_Output_bool = OpTypePointer Output %bool
       %flag = OpVariable %_ptr_Output_bool Output  ; RelaxedPrecision

               ; Function 6
       %main = OpFunction %void_t None %fn_t        ; RelaxedPrecision
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
"#;
    assert_eq!(disassembled, expected);
}

#[test]
fn friendly_names_include_builtin_decorations() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let fn_type = builder.type_function(void, vec![]);
    let uint = builder.type_int(32, 0);
    let vec3 = builder.type_vector(uint, 3);
    let input_ptr = builder.type_pointer(None, StorageClass::Input, vec3);
    let builtin = builder.variable(input_ptr, None, StorageClass::Input, None);
    builder.decorate(
        builtin,
        Decoration::BuiltIn,
        [dr::Operand::BuiltIn(BuiltIn::GlobalInvocationId)],
    );
    builder
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .expect("function");
    builder.begin_block(None).expect("block");
    builder.ret().expect("ret");
    builder.end_function().expect("end");

    let binary = builder.module().assemble();
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(
        disassembled.contains("%gl_GlobalInvocationID = OpVariable"),
        "{disassembled}"
    );
}

#[test]
fn disassembly_normalizes_execution_model_aliases() {
    let mut builder = Builder::new();
    builder.capability(Capability::RayTracingNV);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let fn_type = builder.type_function(void, vec![]);
    let main = builder
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .expect("function");
    builder.begin_block(None).expect("block");
    builder.ret().expect("ret");
    builder.end_function().expect("end");
    builder.entry_point(ExecutionModel::RayGenerationNV, main, "main", vec![]);
    let binary = builder.module().assemble();
    let options = BinaryToTextOptions::NO_HEADER;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(
        disassembled.contains("OpEntryPoint RayGenerationKHR"),
        "{disassembled}"
    );
}

#[test]
fn disassembly_normalizes_storage_class_aliases() {
    let mut builder = Builder::new();
    builder.capability(Capability::RayTracingNV);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let float = builder.type_float(32, None);
    let ptr = builder.type_pointer(None, StorageClass::CallableDataNV, float);
    let payload = builder.variable(ptr, None, StorageClass::CallableDataNV, None);
    builder.name(payload, "payload");
    let binary = builder.module().assemble();
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(
        disassembled.contains("%_ptr_CallableDataKHR_float = OpTypePointer CallableDataKHR %float"),
        "{disassembled}"
    );
    assert!(
        disassembled.contains("%payload = OpVariable %_ptr_CallableDataKHR_float CallableDataKHR"),
        "{disassembled}"
    );
}

#[test]
fn disassembly_preserves_trailing_opline_order() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let file = builder.string("file.ext");
    let void = builder.type_void();
    let fn_ty = builder.type_function(void, vec![]);
    let func = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .expect("function");
    let label = builder.begin_block(None).expect("block");
    builder.ret().expect("return");
    builder.end_function().expect("end");
    builder.line(file, 1, 0);
    let binary = builder.module().assemble();
    let options = BinaryToTextOptions::NO_HEADER;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    let expected = format!(
            "OpCapability Shader\nOpMemoryModel Logical Simple\n%{} = OpString \"file.ext\"\n%{} = OpTypeVoid\n%{} = OpTypeFunction %{}\n%{} = OpFunction %{} None %{}\n%{} = OpLabel\nOpReturn\nOpFunctionEnd\nOpLine %{} 1 0\n",
            file, void, fn_ty, void, func, void, fn_ty, label, file
        );
    assert_eq!(disassembled, expected);
}

#[test]
fn disassembly_preserves_prefunction_opline_order() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let file = builder.string("file.ext");
    let void = builder.type_void();
    let fn_ty = builder.type_function(void, vec![]);
    let func = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .expect("function");
    builder.line(file, 10, 10);
    builder.line(file, 20, 20);
    let label = builder.begin_block(None).expect("block");
    builder.ret().expect("return");
    builder.end_function().expect("end");
    let module = builder.module();
    let mut binary = module.assemble();
    let mut prelude_words = Vec::new();
    let mut scan = super::HEADER_WORD_COUNT;
    while scan < binary.len() {
        let word = binary[scan];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xFFFF;
        if opcode == rspirv::spirv::Op::Line as u32 {
            prelude_words.extend_from_slice(&binary[scan..scan + word_count]);
            binary.drain(scan..scan + word_count);
        } else {
            scan += word_count;
        }
    }
    let mut insert = super::HEADER_WORD_COUNT;
    while insert < binary.len() {
        let word = binary[insert];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xFFFF;
        insert += word_count;
        if opcode == rspirv::spirv::Op::Function as u32 {
            break;
        }
    }
    binary.splice(insert..insert, prelude_words.iter().cloned());
    let options = BinaryToTextOptions::NO_HEADER;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    let expected = format!(
            "OpCapability Shader\nOpMemoryModel Logical Simple\n%{} = OpString \"file.ext\"\n%{} = OpTypeVoid\n%{} = OpTypeFunction %{}\n%{} = OpFunction %{} None %{}\nOpLine %{} 10 10\nOpLine %{} 20 20\n%{} = OpLabel\nOpReturn\nOpFunctionEnd\n",
            file, void, fn_ty, void, func, void, fn_ty, file, file, label
        );
    assert_eq!(disassembled, expected);
}

#[test]
fn disassembly_formats_integer_constants_using_type() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = builder.type_int(32, 1);
    let uint = builder.type_int(32, 0);
    builder.constant_bit32(int, (-1867i32) as u32);
    builder.constant_bit32(uint, 1867);
    let binary = builder.module().assemble();
    let options = BinaryToTextOptions::NO_HEADER;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(disassembled.contains("-1867"));
    assert!(disassembled.contains(" 1867"));
}

#[test]
fn friendly_names_rename_type_ids() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let void_fn = builder.type_function(void, vec![]);
    builder
        .begin_function(void, None, FunctionControl::NONE, void_fn)
        .expect("function");
    builder.begin_block(None).expect("block");
    builder.ret().expect("return");
    builder.end_function().expect("end");
    let module = builder.module();
    let binary = module.assemble();
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(
        disassembled.contains("%void = OpTypeVoid"),
        "{disassembled}"
    );
}

#[test]
fn friendly_names_deduplicate_opname_collisions() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let fn_type = builder.type_function(void, vec![]);

    let first = builder
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .expect("function");
    builder.name(first, "foo");
    builder.begin_block(None).expect("block");
    builder.ret().expect("ret");
    builder.end_function().expect("end");

    let second = builder
        .begin_function(void, None, FunctionControl::NONE, fn_type)
        .expect("function");
    builder.name(second, "foo");
    builder.begin_block(None).expect("block");
    builder.ret().expect("ret");
    builder.end_function().expect("end");

    let module = builder.module();
    let binary = module.assemble();
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(disassembled.contains("%foo = OpFunction"), "{disassembled}");
    assert!(
        disassembled.contains("%foo_0 = OpFunction"),
        "{disassembled}"
    );
}

#[test]
fn friendly_names_include_pipe_access_qualifier() {
    let mut builder = Builder::new();
    builder.capability(Capability::Pipes);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    builder.type_pipe(AccessQualifier::ReadOnly);
    let module = builder.module();
    let binary = module.assemble();
    let options = BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::FRIENDLY_NAMES;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(
        disassembled.contains("%PipeReadOnly = OpTypePipe ReadOnly"),
        "{disassembled}"
    );
}

#[test]
fn disassembly_formats_float_constants() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let f16 = builder.type_float(16, None);
    let f32 = builder.type_float(32, None);
    builder.constant_bit32(f32, (-3.125f32).to_bits());
    builder.constant_bit32(f16, 0x7e00);
    let binary = builder.module().assemble();
    let options = BinaryToTextOptions::NO_HEADER;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(disassembled.contains("-3.125"), "{disassembled}");
    assert!(
        disassembled
            .lines()
            .any(|line| line.contains("OpConstant") && line.contains("0x")),
        "{disassembled}"
    );
}

#[test]
fn disassembly_formats_special_float_values_as_hex() {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let float = builder.type_float(32, None);
    builder.constant_bit32(float, 0x7fc00000); // NaN
    builder.constant_bit32(float, 0x7f800000); // +Inf
    builder.constant_bit32(float, 0xff800000); // -Inf
    let binary = builder.module().assemble();
    let disassembled =
        disassemble_binary(&binary, BinaryToTextOptions::NO_HEADER).expect("disassemble");
    assert!(disassembled.contains("0x1.8p+128"));
    assert!(disassembled.contains("0x1p+128"));
    assert!(disassembled.contains("-0x1p+128"));
}

#[test]
fn hex_option_overrides_constant_formatting() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%int = OpTypeInt 32 1",
        "%val = OpConstant %int -42",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble text");
    let options = BinaryToTextOptions::HEX | BinaryToTextOptions::NO_HEADER;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    assert!(disassembled.contains("0xffffffd6"), "{disassembled}");
}

#[test]
fn nested_indent_inserts_blank_line_before_labels() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble text");
    let options = BinaryToTextOptions::NO_HEADER
        | BinaryToTextOptions::INDENT
        | BinaryToTextOptions::NESTED_INDENT;
    let disassembled = disassemble_binary(&binary, options).expect("disassemble");
    let lines: Vec<&str> = disassembled.lines().collect();
    let mut has_blank = false;
    for pair in lines.windows(2) {
        if pair[1].contains("OpLabel") && pair[0].trim().is_empty() {
            has_blank = true;
            break;
        }
    }
    assert!(
        has_blank,
        "expected blank line before label:\n{disassembled}"
    );
}

#[test]
fn disassembly_handles_conditional_extension_intel() {
    let text = disassemble_with_options(
        CONDITIONAL_EXTENSION_SAMPLE_BINARY,
        BinaryToTextOptions::INDENT,
    );
    assert!(
        text.contains("OpConditionalExtensionINTEL %2 \"SPV_INTEL_function_variants\""),
        "{text}"
    );
}

fn spaces_after_equals(line: &str) -> Option<usize> {
    let (_, rest) = line.split_once('=')?;
    Some(rest.chars().take_while(|ch| *ch == ' ').count())
}

fn leading_spaces(line: &str) -> usize {
    line.chars().take_while(|ch| ch.is_whitespace()).count()
}

fn build_selection_module(permuted: bool) -> Vec<u32> {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = builder.type_void();
    let bool_type = builder.type_bool();
    let void_fn = builder.type_function(void, vec![]);
    let true_id = builder.constant_true(bool_type);
    builder
        .begin_function(void, None, FunctionControl::NONE, void_fn)
        .expect("function");
    let entry_label = builder.begin_block(None).expect("entry block");
    builder.name(entry_label, "entry");
    let merge_label = builder.id();
    let then_label = builder.id();
    builder.name(merge_label, "merge");
    builder.name(then_label, "then");
    builder
        .selection_merge(merge_label, SelectionControl::NONE)
        .expect("merge");
    builder
        .branch_conditional(true_id, then_label, merge_label, std::iter::empty())
        .expect("branch conditional");
    builder.begin_block(Some(merge_label)).expect("merge block");
    builder.ret().expect("return");
    builder.begin_block(Some(then_label)).expect("then block");
    builder.branch(merge_label).expect("branch");
    builder.end_function().expect("end function");

    let mut module = builder.module();
    if permuted {
        if let Some(function) = module.functions.get_mut(0) {
            if function.blocks.len() >= 3 {
                let merge_block = function.blocks.remove(function.blocks.len() - 1);
                function.blocks.insert(1, merge_block);
            }
        }
    }
    module.assemble()
}

#[cfg(test)]
fn take_print_log() -> Vec<String> {
    super::PRINT_LOG.lock().unwrap().drain(..).collect()
}
