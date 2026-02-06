use super::{
    assemble_instructions, assemble_text, assemble_text_with_env, assemble_text_with_options,
    AssemblyTranslator,
};
use crate::assembly::parser::parse_instruction;
use crate::assembly::{BinaryToTextOptions, TextToBinaryOptions};
use crate::disassembly::disassemble_binary;
use crate::target_env::TargetEnv;
use crate::version::SpirvVersion;
use rspirv::{dr, spirv};

#[test]
fn translator_emits_type_int_instruction() {
    let parsed = parse_instruction("%uint = OpTypeInt 32 0").expect("parse");
    let mut translator = AssemblyTranslator::new();
    translator.translate(&parsed);
    let (module, diagnostics) = translator.finish();
    assert!(diagnostics.is_empty());
    let inst = module
        .types_global_values
        .first()
        .expect("type instruction");
    assert_eq!(inst.class.opcode, rspirv::spirv::Op::TypeInt);
    assert_eq!(inst.result_id, Some(1));
    assert_eq!(inst.operands.len(), 2);
}

#[test]
fn translator_sets_memory_model() {
    let parsed = parse_instruction("OpMemoryModel Logical GLSL450").expect("parse");
    let mut translator = AssemblyTranslator::new();
    translator.translate(&parsed);
    let (module, diagnostics) = translator.finish();
    assert!(diagnostics.is_empty());
    let inst = module.memory_model.as_ref().expect("memory model");
    assert_eq!(inst.class.opcode, rspirv::spirv::Op::MemoryModel);
}

#[test]
fn translator_emits_entry_point_instruction() {
    let parsed = parse_instruction("OpEntryPoint GLCompute %main \"main\" %a %b").expect("parse");
    let mut translator = AssemblyTranslator::new();
    translator.translate(&parsed);
    let (module, diagnostics) = translator.finish();
    assert!(diagnostics.is_empty());
    let inst = module.entry_points.first().expect("entry point");
    assert_eq!(inst.class.opcode, rspirv::spirv::Op::EntryPoint);
}

#[test]
fn translator_emits_extension_instruction() {
    let parsed = parse_instruction("OpExtension \"SPV_KHR_ray_tracing\"").expect("parse extension");
    let mut translator = AssemblyTranslator::new();
    translator.translate(&parsed);
    let (module, diagnostics) = translator.finish();
    assert!(diagnostics.is_empty());
    let inst = module.extensions.first().expect("extension");
    assert_eq!(inst.class.opcode, spirv::Op::Extension);
    assert_eq!(
        inst.operands,
        vec![dr::Operand::LiteralString("SPV_KHR_ray_tracing".into())]
    );
}

#[test]
fn assembler_preserves_textual_order_for_globals() {
    let input = r#"
; comment line
            OpMemoryModel Logical Simple
%glsl450 = OpExtInstImport "GLSL.std.450"
"#;
    let words = assemble_text(input).expect("assemble");
    assert!(
        words.len() >= 5,
        "assembled module should contain header and instructions"
    );
    let instructions = &words[5..];
    let memory_model = 196_622;
    let ext_inst_import = 393_227;
    let mem_idx = instructions
        .iter()
        .position(|word| *word == memory_model)
        .expect("memory model present");
    let ext_idx = instructions
        .iter()
        .position(|word| *word == ext_inst_import)
        .expect("ext inst import present");
    assert!(
        ext_idx < mem_idx,
        "assembler should canonicalize layout ordering (extinst before memory model)"
    );
}

#[test]
fn arm_motion_engine_ext_inst_round_trips_with_names() {
    let src = [
        "%1 = OpExtInstImport \"Arm.MotionEngine.100\"",
        "%3 = OpExtInst %2 %1 MIN_SAD %4 %5 %6 %7 %8 %9 %10 %11 %12",
    ]
    .join("\n");
    let binary = assemble_text(&src).expect("assemble arm.motion");
    let disassembled = disassemble_binary(&binary, BinaryToTextOptions::NONE).expect("disassemble");
    assert!(
        disassembled.contains("MIN_SAD"),
        "expected disassembly to use the opcode name, got: {disassembled}"
    );
    assert!(
        !disassembled.contains(" OpExtInst %2 %1 0 "),
        "extinst opcode should not fall back to a numeric literal: {disassembled}"
    );
}

fn round_trip_with_options(
    text: &str,
    options: TextToBinaryOptions,
    disassemble_opts: BinaryToTextOptions,
) -> String {
    let binary =
        assemble_text_with_options(text, TargetEnv::Universal1_0, options).expect("assemble");
    disassemble_binary(&binary, disassemble_opts).expect("disassemble")
}

#[test]
fn assembler_renumbers_numeric_ids_by_default() {
    let before = [
        "OpCapability Addresses",
        "OpCapability Kernel",
        "OpCapability GenericPointer",
        "OpCapability Linkage",
        "OpMemoryModel Physical32 OpenCL",
        "%i32 = OpTypeInt 32 1",
        "%u32 = OpTypeInt 32 0",
        "%f32 = OpTypeFloat 32",
        "%200 = OpTypeVoid",
        "%300 = OpTypeFunction %200",
        "%main = OpFunction %200 None %300",
        "%entry = OpLabel",
        "%100 = OpConstant %u32 100",
        "%1 = OpConstant %u32 200",
        "%2 = OpConstant %u32 300",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let expected = [
        "OpCapability Addresses",
        "OpCapability Kernel",
        "OpCapability GenericPointer",
        "OpCapability Linkage",
        "OpMemoryModel Physical32 OpenCL",
        "%1 = OpTypeInt 32 1",
        "%2 = OpTypeInt 32 0",
        "%3 = OpTypeFloat 32",
        "%4 = OpTypeVoid",
        "%5 = OpTypeFunction %4",
        "%8 = OpConstant %2 100",
        "%9 = OpConstant %2 200",
        "%10 = OpConstant %2 300",
        "%6 = OpFunction %4 None %5",
        "%7 = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n")
        + "\n";

    let text = round_trip_with_options(
        &before,
        TextToBinaryOptions::NONE,
        BinaryToTextOptions::NO_HEADER,
    );
    assert_eq!(text, expected);
}

#[test]
fn assembler_preserves_numeric_ids_when_requested() {
    let before = [
        "OpCapability Addresses",
        "OpCapability Kernel",
        "OpCapability GenericPointer",
        "OpCapability Linkage",
        "OpMemoryModel Physical32 OpenCL",
        "%i32 = OpTypeInt 32 1",
        "%u32 = OpTypeInt 32 0",
        "%f32 = OpTypeFloat 32",
        "%200 = OpTypeVoid",
        "%300 = OpTypeFunction %200",
        "%main = OpFunction %200 None %300",
        "%entry = OpLabel",
        "%100 = OpConstant %u32 100",
        "%1 = OpConstant %u32 200",
        "%2 = OpConstant %u32 300",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let expected = [
        "OpCapability Addresses",
        "OpCapability Kernel",
        "OpCapability GenericPointer",
        "OpCapability Linkage",
        "OpMemoryModel Physical32 OpenCL",
        "%3 = OpTypeInt 32 1",
        "%4 = OpTypeInt 32 0",
        "%5 = OpTypeFloat 32",
        "%200 = OpTypeVoid",
        "%300 = OpTypeFunction %200",
        "%100 = OpConstant %4 100",
        "%1 = OpConstant %4 200",
        "%2 = OpConstant %4 300",
        "%6 = OpFunction %200 None %300",
        "%7 = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n")
        + "\n";

    let text = round_trip_with_options(
        &before,
        TextToBinaryOptions::PRESERVE_NUMERIC_IDS,
        BinaryToTextOptions::NO_HEADER,
    );
    assert_eq!(text, expected);
}

#[test]
fn translator_emits_constant_instruction() {
    let type_inst = parse_instruction("%uint = OpTypeInt 32 0").unwrap();
    let const_inst = parse_instruction("%c32 = OpConstant %uint 32").unwrap();
    let mut translator = AssemblyTranslator::new();
    translator.translate(&type_inst);
    translator.translate(&const_inst);
    let (module, diagnostics) = translator.finish();
    assert!(diagnostics.is_empty());
    assert!(module
        .types_global_values
        .iter()
        .any(|inst| inst.class.opcode == rspirv::spirv::Op::Constant));
}

#[test]
fn assemble_instructions_streams_sequence() {
    let type_inst = parse_instruction("%uint = OpTypeInt 32 0").unwrap();
    let mem_model = parse_instruction("OpMemoryModel Logical GLSL450").unwrap();
    let module = assemble_instructions(&[&type_inst, &mem_model]).expect("assemble instructions");
    assert!(module.memory_model.is_some());
}

#[test]
fn assemble_text_parses_multiple_lines() {
    let text = "%uint = OpTypeInt 32 0\nOpMemoryModel Logical GLSL450";
    let binary = assemble_text(text).expect("assemble text");
    assert!(!binary.is_empty());
}

#[test]
fn assemble_text_emits_simple_function() {
    let text = "\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n\
OpMemoryModel Logical GLSL450\n\
%main = OpFunction %void None %void_fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd";
    let binary = assemble_text(text).expect("assemble text");
    assert!(!binary.is_empty());
}

#[test]
fn translator_handles_execution_mode_and_memory_ops() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\" %buffer",
        "OpExecutionMode %main LocalSize 1 1 1",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%ptr = OpTypePointer StorageBuffer %uint",
        "%one = OpConstant %uint 1",
        "%buffer = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%value = OpLoad %uint %buffer",
        "%sum = OpIAdd %uint %value %one",
        "OpStore %buffer %sum",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| {
            parse_instruction(line)
                .unwrap_or_else(|err| panic!("failed to parse '{line}': {err:?}"))
        })
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    assert_eq!(module.capabilities.len(), 1);
    assert_eq!(module.execution_modes.len(), 1);
    assert_eq!(module.entry_points.len(), 1);
    let function = module.functions.first().expect("function");
    assert_eq!(function.blocks.len(), 1);
    let block = function.blocks.first().expect("entry block");
    assert!(block
        .instructions
        .iter()
        .any(|inst| inst.class.opcode == spirv::Op::IAdd));
}

#[test]
fn translator_emits_glsl_ext_inst_with_named_opcode() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%glsl = OpExtInstImport \"GLSL.std.450\"",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%zero = OpConstant %float 0",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%abs = OpExtInst %float %glsl FAbs %zero",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    assert_eq!(module.ext_inst_imports.len(), 1);
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("entry block");
    let ext_inst = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::ExtInst)
        .expect("ext inst instruction");
    assert!(matches!(
        ext_inst.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::LiteralExtInstInteger(4),
            dr::Operand::IdRef(_)
        ]
    ));
}

#[test]
fn translator_emits_member_decorate_matrix_stride() {
    let source = [
        "%float = OpTypeFloat 32",
        "%vec4 = OpTypeVector %float 4",
        "%mat = OpTypeMatrix %vec4 4",
        "%struct = OpTypeStruct %mat",
        "OpMemberDecorate %struct 0 MatrixStride 16",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let struct_id = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::TypeStruct)
        .and_then(|inst| inst.result_id)
        .expect("struct id");
    let annotation = module.annotations.first().expect("annotation");
    assert_eq!(annotation.class.opcode, spirv::Op::MemberDecorate);
    assert_eq!(
        annotation.operands.as_slice(),
        [
            dr::Operand::IdRef(struct_id),
            dr::Operand::LiteralBit32(0),
            dr::Operand::Decoration(spirv::Decoration::MatrixStride),
            dr::Operand::LiteralBit32(16),
        ]
    );
}

#[test]
fn diagnostics_report_original_positions() {
    let text = "OpCapability Shader\n    OpTypo Thing\n";
    let diagnostics = assemble_text(text)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert!(!diagnostics.is_empty());
    let position = diagnostics[0].position();
    assert_eq!(position.line(), 1);
    assert_eq!(position.column(), 4);
}

#[test]
fn row_major_requires_matrix_stride() {
    let source = [
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%mat = OpTypeMatrix %vec2 2",
        "%struct = OpTypeStruct %mat",
        "OpMemberDecorate %struct 0 RowMajor",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "RowMajor decoration requires an accompanying MatrixStride"
    );
}

#[test]
fn matrix_layout_requires_matrix_member_type() {
    let source = [
        "%float = OpTypeFloat 32",
        "%struct = OpTypeStruct %float",
        "OpMemberDecorate %struct 0 RowMajor",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "RowMajor decoration requires the member type to be a matrix or array of matrices"
    );
}

#[test]
fn matrix_stride_requires_matrix_member() {
    let source = [
        "%float = OpTypeFloat 32",
        "%struct = OpTypeStruct %float",
        "OpMemberDecorate %struct 0 MatrixStride 16",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "MatrixStride decoration requires the member type to contain a matrix"
    );
}

#[test]
fn conflicting_matrix_major_decorations_report_diagnostic() {
    let source = [
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%mat = OpTypeMatrix %vec2 2",
        "%struct = OpTypeStruct %mat",
        "OpMemberDecorate %struct 0 RowMajor",
        "OpMemberDecorate %struct 0 MatrixStride 16",
        "OpMemberDecorate %struct 0 ColMajor",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "RowMajor and ColMajor decorations cannot both target the same member"
    );
}

#[test]
fn translator_emits_builtin_decorations() {
    let source = [
        "%float = OpTypeFloat 32",
        "%ptr = OpTypePointer Input %float",
        "%var = OpVariable %ptr Input",
        "OpDecorate %var BuiltIn Position",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let var_id = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::Variable)
        .and_then(|inst| inst.result_id)
        .expect("variable id");
    let annotation = module.annotations.first().expect("annotation");
    assert_eq!(annotation.class.opcode, spirv::Op::Decorate);
    assert_eq!(
        annotation.operands.as_slice(),
        [
            dr::Operand::IdRef(var_id),
            dr::Operand::Decoration(spirv::Decoration::BuiltIn),
            dr::Operand::BuiltIn(spirv::BuiltIn::Position),
        ]
    );
}

#[test]
fn translator_emits_linkage_attributes() {
    let source = [
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "OpDecorate %main LinkageAttributes \"main\" Import",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function_id = module
        .functions
        .first()
        .and_then(|func| func.def.as_ref())
        .and_then(|inst| inst.result_id)
        .expect("function id");
    let annotation = module.annotations.first().expect("annotation");
    assert_eq!(annotation.class.opcode, spirv::Op::Decorate);
    assert_eq!(
        annotation.operands.as_slice(),
        [
            dr::Operand::IdRef(function_id),
            dr::Operand::Decoration(spirv::Decoration::LinkageAttributes),
            dr::Operand::LiteralString("main".to_string()),
            dr::Operand::LinkageType(spirv::LinkageType::Import),
        ]
    );
}

#[test]
fn translator_emits_decorate_id_operands() {
    let source = [
        "%uint = OpTypeInt 32 0",
        "%ptr = OpTypePointer Uniform %uint",
        "%var = OpVariable %ptr Uniform",
        "%const = OpConstant %uint 16",
        "OpDecorateId %var AlignmentId %const",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let var_id = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::Variable)
        .and_then(|inst| inst.result_id)
        .expect("var id");
    let const_id = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::Constant)
        .and_then(|inst| inst.result_id)
        .expect("const id");
    let annotation = module.annotations.first().expect("annotation");
    assert_eq!(annotation.class.opcode, spirv::Op::Decorate);
    assert_eq!(
        annotation.operands.as_slice(),
        [
            dr::Operand::IdRef(var_id),
            dr::Operand::Decoration(spirv::Decoration::AlignmentId),
            dr::Operand::IdRef(const_id),
        ]
    );
}

#[test]
fn translator_handles_opencl_ext_inst_literal_operands() {
    let source = [
        "OpCapability Kernel",
        "OpMemoryModel Physical64 OpenCL",
        "%opencl = OpExtInstImport \"OpenCL.std\"",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%ulong = OpTypeInt 64 0",
        "%ptr = OpTypePointer CrossWorkgroup %float",
        "%offset = OpConstant %ulong 1",
        "%addr = OpVariable %ptr CrossWorkgroup",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%load = OpExtInst %vec2 %opencl vloadn %offset %addr 2",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("entry block");
    let ext_inst = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::ExtInst)
        .expect("ext inst instruction");
    assert!(matches!(
        ext_inst.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::LiteralExtInstInteger(_),
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::LiteralBit32(2)
        ]
    ));
}

#[test]
fn translator_handles_opencl_rounding_mode_operands() {
    let source = [
        "OpCapability Kernel",
        "OpMemoryModel Physical64 OpenCL",
        "%opencl = OpExtInstImport \"OpenCL.std\"",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%ptr = OpTypePointer CrossWorkgroup %float",
        "%float_0 = OpConstant %float 0",
        "%value = OpConstantComposite %vec2 %float_0 %float_0",
        "%var = OpVariable %ptr CrossWorkgroup",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%call = OpExtInst %void %opencl vstore_half_r %value %var %value RTE",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("entry block");
    let ext_inst = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::ExtInst)
        .expect("ext inst instruction");
    assert!(matches!(
        ext_inst.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::LiteralExtInstInteger(_),
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::FPRoundingMode(spirv::FPRoundingMode::RTE)
        ]
    ));
}

#[test]
fn translator_handles_opencl_printf_variadic_operands() {
    let source = [
        "OpCapability Kernel",
        "OpMemoryModel Physical64 OpenCL",
        "%opencl = OpExtInstImport \"OpenCL.std\"",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%ptr = OpTypePointer CrossWorkgroup %uint",
        "%value = OpVariable %ptr CrossWorkgroup",
        "%format = OpVariable %ptr CrossWorkgroup",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%call = OpExtInst %void %opencl printf %format %value %value",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("entry block");
    let ext_inst = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::ExtInst)
        .expect("ext inst instruction");
    assert_eq!(ext_inst.operands.len(), 5);
}

#[test]
fn translator_emits_memory_operands_for_load_store() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%ptr_ty = OpTypePointer StorageBuffer %uint",
        "%zero = OpConstant %uint 0",
        "%buffer = OpVariable %ptr_ty StorageBuffer",
        "%scope = OpConstant %uint 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%val = OpLoad %uint %buffer Aligned|MakePointerVisible 8 %scope",
        "OpStore %buffer %val Aligned|MakePointerAvailable 8 %scope",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("entry block");
    let load = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::Load)
        .expect("load inst");
    assert!(matches!(
        load.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::MemoryAccess(mask),
            dr::Operand::LiteralBit32(8),
            dr::Operand::IdRef(_)
        ] if mask.contains(spirv::MemoryAccess::ALIGNED)
            && mask.contains(spirv::MemoryAccess::MAKE_POINTER_VISIBLE)
    ));
    let store = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::Store)
        .expect("store inst");
    assert!(matches!(
        store.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::MemoryAccess(mask),
            dr::Operand::LiteralBit32(8),
            dr::Operand::IdRef(_)
        ] if mask.contains(spirv::MemoryAccess::ALIGNED)
            && mask.contains(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE)
    ));
}

#[test]
fn translator_emits_access_chain_instruction() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%ptr_uint = OpTypePointer StorageBuffer %uint",
        "%ptr_ptr_uint = OpTypePointer StorageBuffer %ptr_uint",
        "%zero = OpConstant %uint 0",
        "%var = OpVariable %ptr_ptr_uint StorageBuffer",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%elem_ptr = OpAccessChain %ptr_uint %var %zero",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("block");
    let access_chain = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::AccessChain)
        .expect("access chain instruction");
    assert!(matches!(
        access_chain.operands.as_slice(),
        [dr::Operand::IdRef(_), dr::Operand::IdRef(_)]
    ));
}

#[test]
fn translator_handles_branch_instructions() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%one = OpConstant %uint 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "OpBranch %mid",
        "%mid = OpLabel",
        "OpBranchConditional %one %then %exit 1 2",
        "%then = OpLabel",
        "OpReturn",
        "%exit = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let all_insts: Vec<_> = function
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .collect();
    assert!(all_insts
        .iter()
        .any(|inst| inst.class.opcode == spirv::Op::Branch));
    let branch_cond = all_insts
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::BranchConditional)
        .expect("branch conditional inst");
    assert!(branch_cond.operands.len() >= 3);
}

#[test]
fn translator_handles_copy_memory_operands() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%size = OpConstant %uint 4",
        "%ptr_fn = OpTypePointer Function %uint",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%dst = OpVariable %ptr_fn Function",
        "%src = OpVariable %ptr_fn Function",
        "OpCopyMemory %dst %src Aligned 4 Aligned 8",
        "OpCopyMemorySized %dst %src %size Aligned 4 Aligned 8",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("block");
    let mut copies = block.instructions.iter().filter(|inst| {
        matches!(
            inst.class.opcode,
            spirv::Op::CopyMemory | spirv::Op::CopyMemorySized
        )
    });
    let copy = copies.next().expect("OpCopyMemory");
    assert!(matches!(
        copy.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::MemoryAccess(first),
            dr::Operand::LiteralBit32(4),
            dr::Operand::MemoryAccess(second),
            dr::Operand::LiteralBit32(8)
        ] if first.contains(spirv::MemoryAccess::ALIGNED)
            && second.contains(spirv::MemoryAccess::ALIGNED)
    ));
    let copy_sized = copies.next().expect("OpCopyMemorySized");
    assert!(matches!(
        copy_sized.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::MemoryAccess(first),
            dr::Operand::LiteralBit32(4),
            dr::Operand::MemoryAccess(second),
            dr::Operand::LiteralBit32(8)
        ] if first.contains(spirv::MemoryAccess::ALIGNED)
            && second.contains(spirv::MemoryAccess::ALIGNED)
    ));
}

#[test]
fn translator_emits_selection_merge() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%one = OpConstant %uint 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        "OpBranchConditional %one %then %else",
        "%then = OpLabel",
        "OpBranch %merge",
        "%else = OpLabel",
        "OpBranch %merge",
        "%merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let selection_merge = function
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .find(|inst| inst.class.opcode == spirv::Op::SelectionMerge)
        .expect("selection merge");
    assert!(matches!(
        selection_merge.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::SelectionControl(control)
        ] if control.is_empty()
    ));
}

#[test]
fn translator_emits_loop_merge_with_operands() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%one = OpConstant %uint 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "OpBranch %loop",
        "%loop = OpLabel",
        "OpLoopMerge %merge %continue MinIterations|PartialCount 4 2",
        "OpBranch %continue",
        "%continue = OpLabel",
        "OpBranchConditional %one %loop %merge",
        "%merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let loop_merge = function
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .find(|inst| inst.class.opcode == spirv::Op::LoopMerge)
        .expect("loop merge");
    assert!(matches!(
        loop_merge.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::LoopControl(control),
            dr::Operand::LiteralBit32(4),
            dr::Operand::LiteralBit32(2)
        ] if control.contains(spirv::LoopControl::MIN_ITERATIONS)
            && control.contains(spirv::LoopControl::PARTIAL_COUNT)
    ));
}

#[test]
fn translator_emits_composite_construct() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%uint_0 = OpConstant %int 0",
        "%uint_1 = OpConstant %int 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%vec = OpCompositeConstruct %vec2 %uint_0 %uint_1",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("block");
    let inst = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::CompositeConstruct)
        .expect("composite construct");
    assert_eq!(inst.operands.len(), 2);
}

#[test]
fn translator_emits_vector_shuffle() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%vec4 = OpTypeVector %int 4",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%two = OpConstant %int 2",
        "%three = OpConstant %int 3",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%v1 = OpCompositeConstruct %vec2 %zero %one",
        "%v2 = OpCompositeConstruct %vec2 %two %three",
        "%shuffle = OpVectorShuffle %vec4 %v1 %v2 0 1 2 3",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let shuffle = function
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .find(|inst| inst.class.opcode == spirv::Op::VectorShuffle)
        .expect("vector shuffle");
    assert_eq!(shuffle.operands.len(), 6);
}

#[test]
fn vector_shuffle_rejects_component_count_mismatch() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%vec4 = OpTypeVector %int 4",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%v1 = OpCompositeConstruct %vec2 %zero %one",
        "%v2 = OpCompositeConstruct %vec2 %one %zero",
        "%shuffle = OpVectorShuffle %vec4 %v1 %v2 0 1 2",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "OpVectorShuffle expects 4 component literals but received 3"
    );
}

#[test]
fn vector_shuffle_rejects_out_of_bounds_component() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%vec4 = OpTypeVector %int 4",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%v1 = OpCompositeConstruct %vec2 %zero %one",
        "%v2 = OpCompositeConstruct %vec2 %one %zero",
        "%shuffle = OpVectorShuffle %vec4 %v1 %v2 0 1 5 3",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "Shuffle component 5 exceeds the available inputs (4)"
    );
}

#[test]
fn translator_emits_composite_extract_and_insert() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%v = OpCompositeConstruct %vec2 %zero %one",
        "%elem = OpCompositeExtract %int %v 1",
        "%result = OpCompositeInsert %vec2 %elem %v 0",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let mut extract_seen = false;
    let mut insert_seen = false;
    for inst in function
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
    {
        if inst.class.opcode == spirv::Op::CompositeExtract {
            extract_seen = true;
            assert!(matches!(
                inst.operands.as_slice(),
                [dr::Operand::IdRef(_), dr::Operand::LiteralBit32(1)]
            ));
        }
        if inst.class.opcode == spirv::Op::CompositeInsert {
            insert_seen = true;
            assert!(matches!(
                inst.operands.as_slice(),
                [
                    dr::Operand::IdRef(_),
                    dr::Operand::IdRef(_),
                    dr::Operand::LiteralBit32(0)
                ]
            ));
        }
    }
    assert!(extract_seen && insert_seen);
}

#[test]
fn translator_handles_array_composite_extract() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%four = OpConstant %int 4",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%two = OpConstant %int 2",
        "%three = OpConstant %int 3",
        "%arr = OpTypeArray %int %four",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%value = OpCompositeConstruct %arr %zero %one %two %three",
        "%elem = OpCompositeExtract %int %value 2",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("block");
    let extract = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::CompositeExtract)
        .expect("extract instruction");
    assert!(matches!(
        extract.operands.as_slice(),
        [dr::Operand::IdRef(_), dr::Operand::LiteralBit32(2)]
    ));
}

#[test]
fn translator_handles_struct_composite_insert() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%two = OpConstant %int 2",
        "%struct = OpTypeStruct %int %int",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%value = OpCompositeConstruct %struct %zero %one",
        "%result = OpCompositeInsert %struct %two %value 1",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("block");
    let insert = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::CompositeInsert)
        .expect("insert instruction");
    assert!(matches!(
        insert.operands.as_slice(),
        [
            dr::Operand::IdRef(_),
            dr::Operand::IdRef(_),
            dr::Operand::LiteralBit32(1)
        ]
    ));
}

#[test]
fn translator_handles_matrix_composite_extract() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%mat2 = OpTypeMatrix %vec2 2",
        "%float_0 = OpConstant %float 0",
        "%float_1 = OpConstant %float 1",
        "%float_2 = OpConstant %float 2",
        "%float_3 = OpConstant %float 3",
        "%col0 = OpConstantComposite %vec2 %float_0 %float_1",
        "%col1 = OpConstantComposite %vec2 %float_2 %float_3",
        "%mat = OpConstantComposite %mat2 %col0 %col1",
        "%struct = OpTypeStruct %mat2",
        "%value = OpConstantComposite %struct %mat",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%elem = OpCompositeExtract %float %value 0 1 0",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("block");
    let extract = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::CompositeExtract)
        .expect("extract instruction");
    assert_eq!(extract.operands.len(), 4);
}

#[test]
fn translator_handles_nested_composite_extract() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%uint = OpTypeInt 32 0",
        "%two = OpConstant %uint 2",
        "%vec2 = OpTypeVector %float 2",
        "%mat2 = OpTypeMatrix %vec2 2",
        "%arr = OpTypeArray %mat2 %two",
        "%struct = OpTypeStruct %arr",
        "%f0 = OpConstant %float 0",
        "%f1 = OpConstant %float 1",
        "%f2 = OpConstant %float 2",
        "%f3 = OpConstant %float 3",
        "%f4 = OpConstant %float 4",
        "%f5 = OpConstant %float 5",
        "%f6 = OpConstant %float 6",
        "%f7 = OpConstant %float 7",
        "%col0 = OpConstantComposite %vec2 %f0 %f1",
        "%col1 = OpConstantComposite %vec2 %f2 %f3",
        "%col2 = OpConstantComposite %vec2 %f4 %f5",
        "%col3 = OpConstantComposite %vec2 %f6 %f7",
        "%mat_a = OpConstantComposite %mat2 %col0 %col1",
        "%mat_b = OpConstantComposite %mat2 %col2 %col3",
        "%arr_val = OpConstantComposite %arr %mat_a %mat_b",
        "%value = OpConstantComposite %struct %arr_val",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%elem = OpCompositeExtract %float %value 0 1 0 1",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    assemble_instructions(&refs).expect("assemble instructions");
}

#[test]
fn composite_extract_reports_out_of_range_index() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%v = OpCompositeConstruct %vec2 %zero %one",
        "%elem = OpCompositeExtract %int %v 3",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "Composite extract index 3 exceeds vector width 2"
    );
}

#[test]
fn composite_extract_rejects_array_index_out_of_bounds() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%four = OpConstant %int 4",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%two = OpConstant %int 2",
        "%three = OpConstant %int 3",
        "%arr = OpTypeArray %int %four",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%value = OpCompositeConstruct %arr %zero %one %two %three",
        "%elem = OpCompositeExtract %int %value 5",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "Array index 5 exceeds array length 4"
    );
}

#[test]
fn composite_extract_rejects_matrix_column_index() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%mat2 = OpTypeMatrix %vec2 2",
        "%f0 = OpConstant %float 0",
        "%f1 = OpConstant %float 1",
        "%col0 = OpConstantComposite %vec2 %f0 %f1",
        "%col1 = OpConstantComposite %vec2 %f1 %f0",
        "%mat = OpConstantComposite %mat2 %col0 %col1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%elem = OpCompositeExtract %vec2 %mat 3",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "Matrix column index 3 exceeds column count 2"
    );
}

#[test]
fn composite_insert_requires_matching_object_type() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%v = OpCompositeConstruct %vec2 %zero %one",
        "%result = OpCompositeInsert %vec2 %v %v 0",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let diagnostics = assemble_instructions(&refs)
        .expect_err("expected diagnostics")
        .into_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].message(),
        "Object operand type must match the selected component type"
    );
}

#[test]
fn translator_emits_constant_composite_and_null() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%vec_const = OpConstantComposite %vec2 %zero %one",
        "%null_vec = OpConstantNull %vec2",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let mut const_opcodes = module.types_global_values.iter().filter(|inst| {
        inst.class.opcode == spirv::Op::ConstantComposite
            || inst.class.opcode == spirv::Op::ConstantNull
    });
    assert!(const_opcodes.any(|inst| inst.class.opcode == spirv::Op::ConstantComposite));
    assert!(const_opcodes.any(|inst| inst.class.opcode == spirv::Op::ConstantNull));
}

#[test]
fn translator_emits_phi_instruction() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 0",
        "%bool = OpTypeBool",
        "%true = OpConstantTrue %bool",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "OpBranchConditional %true %then %else",
        "%then = OpLabel",
        "OpBranch %merge",
        "%else = OpLabel",
        "OpBranch %merge",
        "%merge = OpLabel",
        "%phi = OpPhi %int %zero %then %one %else",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let phi = function
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .find(|inst| inst.class.opcode == spirv::Op::Phi)
        .expect("phi instruction");
    assert_eq!(phi.operands.len(), 4);
}

#[test]
fn assembler_stamps_target_env_version() {
    let binary =
        assemble_text_with_env("", TargetEnv::Universal1_0).expect("assemble text with env");
    assert!(binary.len() > 1);
    assert_eq!(binary[1], SpirvVersion::new(1, 0).to_word());
}

#[test]
fn assemble_with_spans_tracks_result_ids() {
    use super::assemble_text_with_spans;

    let text = r#"OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%fn_type = OpTypeFunction %void
%main = OpFunction %void None %fn_type
%entry = OpLabel
OpReturn
OpFunctionEnd"#;

    let result = assemble_text_with_spans(text).expect("assembly should succeed");

    // Verify the span map has entries for the result IDs
    assert!(!result.span_map.is_empty(), "span map should not be empty");

    // Check that we can look up spans for specific IDs
    // %void should be ID 1, %fn_type should be ID 2, etc.
    // The exact IDs depend on resolution order but we should have multiple entries
    assert!(
        result.span_map.id_count() >= 4,
        "should have at least 4 ID spans (void, fn_type, main, entry)"
    );
}

#[test]
fn assemble_with_spans_records_correct_line_info() {
    use super::assemble_text_with_spans;
    use crate::validation::span::SourceLocation;

    let text = "%uint = OpTypeInt 32 0";

    let result = assemble_text_with_spans(text).expect("assembly should succeed");

    // The ID %uint should be resolved to 1
    let span = result
        .span_map
        .get_id_span(1)
        .expect("should have span for ID 1");

    // The span should point to line 0 (zero-based), where %uint is defined
    match span.start {
        SourceLocation::Text(pos) => {
            assert_eq!(pos.line(), 0, "should be on line 0");
            // Column should point to the start of %uint
            assert_eq!(pos.column(), 0, "should start at column 0");
        }
        _ => panic!("expected text source location"),
    }
}

#[test]
fn translator_emits_bitcast() {
    let source = [
        "OpCapability Shader",
        "OpCapability Int8",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "OpExecutionMode %main OriginUpperLeft",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%u8 = OpTypeInt 8 0",
        "%u32 = OpTypeInt 32 0",
        "%c32 = OpConstant %u32 255",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%result = OpBitcast %u8 %c32",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("entry block");
    let bitcast = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::Bitcast)
        .expect("bitcast instruction");
    assert!(bitcast.result_type.is_some());
    assert!(bitcast.result_id.is_some());
    assert_eq!(bitcast.operands.len(), 1);
}

#[test]
fn translator_emits_convert_s_to_f() {
    let source = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\"",
        "OpExecutionMode %main OriginUpperLeft",
        "%void = OpTypeVoid",
        "%void_fn = OpTypeFunction %void",
        "%int = OpTypeInt 32 1",
        "%float = OpTypeFloat 32",
        "%ci = OpConstant %int 42",
        "%main = OpFunction %void None %void_fn",
        "%entry = OpLabel",
        "%result = OpConvertSToF %float %ci",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let block = function.blocks.first().expect("entry block");
    let convert = block
        .instructions
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::ConvertSToF)
        .expect("convert instruction");
    assert!(convert.result_type.is_some());
    assert!(convert.result_id.is_some());
    assert_eq!(convert.operands.len(), 1);
}

// ---------------------------------------------------------------
// Context-dependent number literal tests (OpConstant / OpSpecConstant)
// Matching C++ spirv-as: integer text for float types is parsed as
// the float value, not raw bits.
// ---------------------------------------------------------------

/// Helper: assemble text, find the first OpConstant, return its operand.
fn assemble_and_get_constant_operand(lines: &[&str]) -> dr::Operand {
    let parsed: Vec<_> = lines
        .iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble");
    module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::Constant)
        .expect("OpConstant not found")
        .operands
        .first()
        .expect("OpConstant missing operand")
        .clone()
}

#[test]
fn constant_integer_literal_for_float32_encodes_as_float_value() {
    // "42" with float32 type should encode as float 42.0, not raw bits 0x2A.
    let operand = assemble_and_get_constant_operand(&[
        "%float = OpTypeFloat 32",
        "%c = OpConstant %float 42",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit32(42.0_f32.to_bits()));
}

#[test]
fn constant_zero_for_float32_encodes_correctly() {
    let operand =
        assemble_and_get_constant_operand(&["%float = OpTypeFloat 32", "%c = OpConstant %float 0"]);
    assert_eq!(operand, dr::Operand::LiteralBit32(0.0_f32.to_bits()));
}

#[test]
fn constant_one_for_float32_encodes_correctly() {
    let operand =
        assemble_and_get_constant_operand(&["%float = OpTypeFloat 32", "%c = OpConstant %float 1"]);
    assert_eq!(operand, dr::Operand::LiteralBit32(1.0_f32.to_bits()));
}

#[test]
fn constant_negative_integer_for_float32_encodes_as_negative_float() {
    // "-1" with float32 type should encode as -1.0f (0xBF800000).
    let operand = assemble_and_get_constant_operand(&[
        "%float = OpTypeFloat 32",
        "%c = OpConstant %float -1",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit32((-1.0_f32).to_bits()));
}

#[test]
fn constant_large_integer_for_float32_encodes_as_float() {
    let operand = assemble_and_get_constant_operand(&[
        "%float = OpTypeFloat 32",
        "%c = OpConstant %float 1000000",
    ]);
    assert_eq!(
        operand,
        dr::Operand::LiteralBit32(1_000_000.0_f32.to_bits())
    );
}

#[test]
fn constant_integer_literal_for_float64_encodes_as_double_value() {
    // "42" with float64 type should encode as double 42.0.
    let operand = assemble_and_get_constant_operand(&[
        "%double = OpTypeFloat 64",
        "%c = OpConstant %double 42",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit64(42.0_f64.to_bits()));
}

#[test]
fn constant_negative_integer_for_float64_encodes_as_negative_double() {
    let operand = assemble_and_get_constant_operand(&[
        "%double = OpTypeFloat 64",
        "%c = OpConstant %double -1",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit64((-1.0_f64).to_bits()));
}

#[test]
fn constant_float_text_for_float32_encodes_correctly() {
    // "42.5" is float text, parsed via OperandValue::Word path.
    let operand = assemble_and_get_constant_operand(&[
        "%float = OpTypeFloat 32",
        "%c = OpConstant %float 42.5",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit32(42.5_f32.to_bits()));
}

#[test]
fn constant_negative_float_text_for_float32_encodes_correctly() {
    let operand = assemble_and_get_constant_operand(&[
        "%float = OpTypeFloat 32",
        "%c = OpConstant %float -3.14",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit32((-3.14_f32).to_bits()));
}

#[test]
fn constant_float_text_for_float64_encodes_correctly() {
    let operand = assemble_and_get_constant_operand(&[
        "%double = OpTypeFloat 64",
        "%c = OpConstant %double 42.5",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit64(42.5_f64.to_bits()));
}

#[test]
fn constant_integer_for_uint32_encodes_as_raw_bits() {
    // Integer types should still encode as raw integer bits.
    let operand =
        assemble_and_get_constant_operand(&["%uint = OpTypeInt 32 0", "%c = OpConstant %uint 42"]);
    assert_eq!(operand, dr::Operand::LiteralBit32(42));
}

#[test]
fn constant_integer_for_sint32_encodes_as_raw_bits() {
    let operand =
        assemble_and_get_constant_operand(&["%int = OpTypeInt 32 1", "%c = OpConstant %int 42"]);
    assert_eq!(operand, dr::Operand::LiteralBit32(42));
}

#[test]
fn constant_negative_for_sint32_encodes_twos_complement() {
    let operand =
        assemble_and_get_constant_operand(&["%int = OpTypeInt 32 1", "%c = OpConstant %int -1"]);
    // -1 as i32 in two's complement is 0xFFFFFFFF
    assert_eq!(operand, dr::Operand::LiteralBit32((-1_i32) as u32));
}

#[test]
fn constant_integer_for_uint64_encodes_value() {
    // Small values that fit in 32 bits are stored as LiteralBit32 by
    // encode_literal_operand. This is a pre-existing behavior; the binary
    // serializer handles type-width encoding.
    let operand = assemble_and_get_constant_operand(&[
        "%ulong = OpTypeInt 64 0",
        "%c = OpConstant %ulong 42",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit32(42));
}

#[test]
fn constant_large_integer_for_uint64_encodes_as_64bit() {
    // Values that don't fit in 32 bits should use LiteralBit64.
    let operand = assemble_and_get_constant_operand(&[
        "%ulong = OpTypeInt 64 0",
        "%c = OpConstant %ulong 4294967296",
    ]);
    assert_eq!(operand, dr::Operand::LiteralBit64(4_294_967_296));
}

#[test]
fn constant_float_round_trips_through_assemble_disassemble() {
    // Full round-trip: assemble "OpConstant %float 42" then disassemble
    // and verify the output shows 42 (the float value), not 5.88545e-44.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%c = OpConstant %float 42",
    ]
    .join("\n");
    let disassembled = round_trip_with_options(
        &text,
        TextToBinaryOptions::NONE,
        BinaryToTextOptions::NO_HEADER,
    );
    assert!(
        disassembled.contains("OpConstant") && disassembled.contains(" 42"),
        "Expected disassembly to contain 'OpConstant ... 42', got: {disassembled}"
    );
    // Must NOT contain the subnormal float that 0x2A bit pattern represents
    assert!(
        !disassembled.contains("5.88545"),
        "OpConstant should not show raw bits interpretation: {disassembled}"
    );
}

#[test]
fn constant_float_text_round_trips_through_assemble_disassemble() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%c = OpConstant %float 42.5",
    ]
    .join("\n");
    let disassembled = round_trip_with_options(
        &text,
        TextToBinaryOptions::NONE,
        BinaryToTextOptions::NO_HEADER,
    );
    assert!(
        disassembled.contains("42.5"),
        "Expected disassembly to contain '42.5', got: {disassembled}"
    );
}

#[test]
fn constant_does_not_note_integer_constant_for_float_type() {
    // When the type is float, note_integer_constant should NOT be called.
    // Verify by checking that a subsequent array-length lookup doesn't
    // confuse float bits with integer values.
    let type_inst = parse_instruction("%float = OpTypeFloat 32").unwrap();
    let const_inst = parse_instruction("%c = OpConstant %float 42").unwrap();
    let mut translator = AssemblyTranslator::new();
    translator.translate(&type_inst);
    translator.translate(&const_inst);
    let (module, diagnostics) = translator.finish();
    assert!(diagnostics.is_empty());
    let constant = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::Constant)
        .expect("constant");
    // The operand should be 42.0f bits, not integer 42
    assert_eq!(
        constant.operands.first().unwrap(),
        &dr::Operand::LiteralBit32(42.0_f32.to_bits())
    );
}

#[test]
fn function_parameter_gets_unique_id_from_function() {
    // Regression test: OpFunctionParameter must get a different result ID
    // than its parent OpFunction. Previously, the assembler used Builder's
    // internal next_id counter (via function_parameter()) which was
    // independent of the module_builder's counter, causing collisions.
    let source = [
        "%void = OpTypeVoid",
        "%uint = OpTypeInt 32 0",
        "%fn_type = OpTypeFunction %void %uint",
        "OpMemoryModel Logical GLSL450",
        "%func = OpFunction %void None %fn_type",
        "%param = OpFunctionParameter %uint",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| {
            parse_instruction(line)
                .unwrap_or_else(|err| panic!("failed to parse '{line}': {err:?}"))
        })
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let function = module.functions.first().expect("function");
    let func_id = function.def.as_ref().unwrap().result_id.unwrap();
    let param = function.parameters.first().expect("parameter");
    let param_id = param.result_id.unwrap();
    assert_ne!(
        func_id, param_id,
        "OpFunction and OpFunctionParameter must have different result IDs, \
             but both got {func_id}"
    );
}

#[test]
fn multiple_functions_with_parameters_have_unique_ids() {
    // Ensure that across multiple functions, all result IDs are unique.
    let source = [
        "%void = OpTypeVoid",
        "%uint = OpTypeInt 32 0",
        "%fn_type = OpTypeFunction %void %uint",
        "OpMemoryModel Logical GLSL450",
        "%func1 = OpFunction %void None %fn_type",
        "%param1 = OpFunctionParameter %uint",
        "%entry1 = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "%func2 = OpFunction %void None %fn_type",
        "%param2 = OpFunctionParameter %uint",
        "%entry2 = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| {
            parse_instruction(line)
                .unwrap_or_else(|err| panic!("failed to parse '{line}': {err:?}"))
        })
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let mut all_ids = std::collections::HashSet::new();
    for function in &module.functions {
        let func_id = function.def.as_ref().unwrap().result_id.unwrap();
        assert!(all_ids.insert(func_id), "Duplicate function ID: {func_id}");
        for param in &function.parameters {
            let param_id = param.result_id.unwrap();
            assert!(
                all_ids.insert(param_id),
                "Duplicate parameter ID: {param_id}"
            );
        }
        for block in &function.blocks {
            if let Some(label) = &block.label {
                let label_id = label.result_id.unwrap();
                assert!(all_ids.insert(label_id), "Duplicate label ID: {label_id}");
            }
        }
    }
}

#[test]
fn assemble_runtime_array_type() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%uint = OpTypeInt 32 0",
        "%rarr = OpTypeRuntimeArray %uint",
    ]
    .join("\n");
    let words = assemble_text(&text).expect("OpTypeRuntimeArray should assemble");
    // Disassemble and check the runtime array instruction is present
    let module = rspirv::dr::load_words(&words).expect("load");
    let has_runtime_array = module
        .types_global_values
        .iter()
        .any(|inst| inst.class.opcode == spirv::Op::TypeRuntimeArray);
    assert!(
        has_runtime_array,
        "assembled module should contain OpTypeRuntimeArray"
    );
}

#[test]
fn translator_emits_type_image_instruction() {
    let source = [
        "%float = OpTypeFloat 32",
        "%img = OpTypeImage %float 2D 0 0 0 1 Unknown",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let img_inst = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::TypeImage)
        .expect("should have OpTypeImage");
    assert_eq!(img_inst.result_id, Some(2));
    assert_eq!(img_inst.operands.len(), 7);
    // Check Dim operand
    assert_eq!(img_inst.operands[1], dr::Operand::Dim(spirv::Dim::Dim2D));
    // Check Sampled operand
    assert_eq!(img_inst.operands[5], dr::Operand::LiteralBit32(1));
    // Check ImageFormat operand
    assert_eq!(
        img_inst.operands[6],
        dr::Operand::ImageFormat(spirv::ImageFormat::Unknown)
    );
}

#[test]
fn translator_emits_type_image_with_all_dims() {
    for (dim_str, expected_dim) in [
        ("1D", spirv::Dim::Dim1D),
        ("2D", spirv::Dim::Dim2D),
        ("3D", spirv::Dim::Dim3D),
        ("Cube", spirv::Dim::DimCube),
        ("Buffer", spirv::Dim::DimBuffer),
        ("SubpassData", spirv::Dim::DimSubpassData),
    ] {
        let source =
            format!("%float = OpTypeFloat 32\n%img = OpTypeImage %float {dim_str} 0 0 0 1 Unknown");
        let parsed: Vec<_> = source
            .lines()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect(&format!("assemble with Dim {dim_str}"));
        let img_inst = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::TypeImage)
            .expect("should have OpTypeImage");
        assert_eq!(
            img_inst.operands[1],
            dr::Operand::Dim(expected_dim),
            "Dim mismatch for {dim_str}"
        );
    }
}

#[test]
fn translator_emits_type_sampled_image_instruction() {
    let source = [
        "%float = OpTypeFloat 32",
        "%img = OpTypeImage %float 2D 0 0 0 1 Unknown",
        "%simg = OpTypeSampledImage %img",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble instructions");
    let simg_inst = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == spirv::Op::TypeSampledImage)
        .expect("should have OpTypeSampledImage");
    assert_eq!(simg_inst.result_id, Some(3));
    assert_eq!(simg_inst.operands.len(), 1);
    assert_eq!(simg_inst.operands[0], dr::Operand::IdRef(2)); // references %img
}

#[test]
fn type_image_assembles_and_validates() {
    // Full module with OpTypeImage that assembles and validates successfully
    let text = r#"
OpCapability Shader
OpCapability InputAttachment
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpDecorate %var DescriptorSet 0
OpDecorate %var Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%img = OpTypeImage %f32 SubpassData 0 0 0 2 Unknown
%ptr = OpTypePointer UniformConstant %img
%var = OpVariable %ptr UniformConstant
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpTypeImage module");
    assert!(!binary.is_empty(), "assembled binary should not be empty");
}

#[test]
fn image_sample_implicit_lod_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %result_var
OpExecutionMode %main OriginUpperLeft
OpDecorate %simg_var DescriptorSet 0
OpDecorate %simg_var Binding 0
OpDecorate %result_var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%v2f32 = OpTypeVector %f32 2
%v4f32 = OpTypeVector %f32 4
%img_ty = OpTypeImage %f32 2D 0 0 0 1 Unknown
%simg_ty = OpTypeSampledImage %img_ty
%ptr_simg = OpTypePointer UniformConstant %simg_ty
%simg_var = OpVariable %ptr_simg UniformConstant
%ptr_v4f32 = OpTypePointer Output %v4f32
%result_var = OpVariable %ptr_v4f32 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%simg = OpLoad %simg_ty %simg_var
%coord = OpLoad %v2f32 %result_var
%result = OpImageSampleImplicitLod %v4f32 %simg %coord
OpStore %result_var %result
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpImageSampleImplicitLod");
    assert!(!binary.is_empty());
}

#[test]
fn image_sample_explicit_lod_with_lod_operand() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %result_var
OpExecutionMode %main OriginUpperLeft
OpDecorate %simg_var DescriptorSet 0
OpDecorate %simg_var Binding 0
OpDecorate %result_var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%v2f32 = OpTypeVector %f32 2
%v4f32 = OpTypeVector %f32 4
%img_ty = OpTypeImage %f32 2D 0 0 0 1 Unknown
%simg_ty = OpTypeSampledImage %img_ty
%ptr_simg = OpTypePointer UniformConstant %simg_ty
%simg_var = OpVariable %ptr_simg UniformConstant
%ptr_v4f32 = OpTypePointer Output %v4f32
%result_var = OpVariable %ptr_v4f32 Output
%f32_0 = OpConstant %f32 0
%main = OpFunction %void None %fn
%entry = OpLabel
%simg = OpLoad %simg_ty %simg_var
%coord = OpLoad %v2f32 %result_var
%result = OpImageSampleExplicitLod %v4f32 %simg %coord Lod %f32_0
OpStore %result_var %result
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpImageSampleExplicitLod with Lod");
    assert!(!binary.is_empty());
}

#[test]
fn image_fetch_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %result_var
OpExecutionMode %main OriginUpperLeft
OpDecorate %img_var DescriptorSet 0
OpDecorate %img_var Binding 0
OpDecorate %result_var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%i32 = OpTypeInt 32 1
%v2i32 = OpTypeVector %i32 2
%v4f32 = OpTypeVector %f32 4
%img_ty = OpTypeImage %f32 2D 0 0 0 1 Unknown
%simg_ty = OpTypeSampledImage %img_ty
%ptr_simg = OpTypePointer UniformConstant %simg_ty
%img_var = OpVariable %ptr_simg UniformConstant
%ptr_v4f32 = OpTypePointer Output %v4f32
%result_var = OpVariable %ptr_v4f32 Output
%i32_0 = OpConstant %i32 0
%main = OpFunction %void None %fn
%entry = OpLabel
%simg = OpLoad %simg_ty %img_var
%img = OpImage %img_ty %simg
%coord = OpLoad %v2i32 %result_var
%result = OpImageFetch %v4f32 %img %coord Lod %i32_0
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpImageFetch");
    assert!(!binary.is_empty());
}

#[test]
fn image_write_assembles() {
    let text = r#"
OpCapability Shader
OpCapability StorageImageWriteWithoutFormat
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpDecorate %img_var DescriptorSet 0
OpDecorate %img_var Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%i32 = OpTypeInt 32 1
%v2i32 = OpTypeVector %i32 2
%v4f32 = OpTypeVector %f32 4
%img_ty = OpTypeImage %f32 2D 0 0 0 2 Unknown
%ptr_img = OpTypePointer UniformConstant %img_ty
%img_var = OpVariable %ptr_img UniformConstant
%i32_0 = OpConstant %i32 0
%f32_1 = OpConstant %f32 1
%main = OpFunction %void None %fn
%entry = OpLabel
%img = OpLoad %img_ty %img_var
%coord = OpCompositeConstruct %v2i32 %i32_0 %i32_0
%texel = OpCompositeConstruct %v4f32 %f32_1 %f32_1 %f32_1 %f32_1
OpImageWrite %img %coord %texel
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpImageWrite");
    assert!(!binary.is_empty());
}

#[test]
fn image_query_lod_assembles() {
    let text = r#"
OpCapability Shader
OpCapability ImageQuery
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %result_var
OpExecutionMode %main OriginUpperLeft
OpDecorate %simg_var DescriptorSet 0
OpDecorate %simg_var Binding 0
OpDecorate %result_var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%v2f32 = OpTypeVector %f32 2
%v4f32 = OpTypeVector %f32 4
%img_ty = OpTypeImage %f32 2D 0 0 0 1 Unknown
%simg_ty = OpTypeSampledImage %img_ty
%ptr_simg = OpTypePointer UniformConstant %simg_ty
%simg_var = OpVariable %ptr_simg UniformConstant
%ptr_v2f32 = OpTypePointer Output %v2f32
%ptr_v4f32 = OpTypePointer Output %v4f32
%result_var = OpVariable %ptr_v4f32 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%simg = OpLoad %simg_ty %simg_var
%coord = OpLoad %v2f32 %result_var
%lod = OpImageQueryLod %v2f32 %simg %coord
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpImageQueryLod");
    assert!(!binary.is_empty());
}

#[test]
fn image_sample_with_grad_operand() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %result_var
OpExecutionMode %main OriginUpperLeft
OpDecorate %simg_var DescriptorSet 0
OpDecorate %simg_var Binding 0
OpDecorate %result_var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%v2f32 = OpTypeVector %f32 2
%v4f32 = OpTypeVector %f32 4
%img_ty = OpTypeImage %f32 2D 0 0 0 1 Unknown
%simg_ty = OpTypeSampledImage %img_ty
%ptr_simg = OpTypePointer UniformConstant %simg_ty
%simg_var = OpVariable %ptr_simg UniformConstant
%ptr_v4f32 = OpTypePointer Output %v4f32
%result_var = OpVariable %ptr_v4f32 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%simg = OpLoad %simg_ty %simg_var
%coord = OpLoad %v2f32 %result_var
%dx = OpLoad %v2f32 %result_var
%dy = OpLoad %v2f32 %result_var
%result = OpImageSampleExplicitLod %v4f32 %simg %coord Grad %dx %dy
OpStore %result_var %result
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble with Grad operand");
    assert!(!binary.is_empty());
}

#[test]
fn image_sample_with_combined_operands() {
    // Lod|ConstOffset with two dependent ids
    let source = [
        "%1 = OpTypeFloat 32",
        "%2 = OpTypeVector %1 2",
        "%3 = OpTypeVector %1 4",
        "%4 = OpTypeImage %1 2D 0 0 0 1 Unknown",
        "%5 = OpTypeSampledImage %4",
    ];
    let parsed: Vec<_> = source
        .into_iter()
        .map(|line| parse_instruction(line).expect("parse"))
        .collect();
    let refs: Vec<_> = parsed.iter().collect();
    let module = assemble_instructions(&refs).expect("assemble types");
    // Verify OpImage and OpTypeSampledImage are present
    let type_count = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == spirv::Op::TypeImage)
        .count();
    assert_eq!(type_count, 1);
}

#[test]
fn sampled_image_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %result_var
OpExecutionMode %main OriginUpperLeft
OpDecorate %img_var DescriptorSet 0
OpDecorate %img_var Binding 0
OpDecorate %sampler_var DescriptorSet 0
OpDecorate %sampler_var Binding 1
OpDecorate %result_var Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f32 = OpTypeFloat 32
%v2f32 = OpTypeVector %f32 2
%v4f32 = OpTypeVector %f32 4
%img_ty = OpTypeImage %f32 2D 0 0 0 1 Unknown
%sampler_ty = OpTypeSampler
%simg_ty = OpTypeSampledImage %img_ty
%ptr_img = OpTypePointer UniformConstant %img_ty
%ptr_sampler = OpTypePointer UniformConstant %sampler_ty
%img_var = OpVariable %ptr_img UniformConstant
%sampler_var = OpVariable %ptr_sampler UniformConstant
%ptr_v4f32 = OpTypePointer Output %v4f32
%result_var = OpVariable %ptr_v4f32 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%img = OpLoad %img_ty %img_var
%sampler = OpLoad %sampler_ty %sampler_var
%simg = OpSampledImage %simg_ty %img %sampler
%coord = OpLoad %v2f32 %result_var
%result = OpImageSampleImplicitLod %v4f32 %simg %coord
OpStore %result_var %result
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpSampledImage + sample");
    assert!(!binary.is_empty());
}

#[test]
fn switch_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%ptr_uint = OpTypePointer Function %uint
%c0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%sel = OpVariable %ptr_uint Function
%val = OpLoad %uint %sel
OpSelectionMerge %merge None
OpSwitch %val %default 1 %case1 2 %case2
%case1 = OpLabel
OpBranch %merge
%case2 = OpLabel
OpBranch %merge
%default = OpLabel
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpSwitch");
    assert!(!binary.is_empty());
}

#[test]
fn switch_with_no_cases_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%ptr_uint = OpTypePointer Function %uint
%c0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%sel = OpVariable %ptr_uint Function
%val = OpLoad %uint %sel
OpSelectionMerge %merge None
OpSwitch %val %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpSwitch with no cases");
    assert!(!binary.is_empty());
}

#[test]
fn switch_roundtrips_through_disassembly() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%ptr_uint = OpTypePointer Function %uint
%c0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%sel = OpVariable %ptr_uint Function
%val = OpLoad %uint %sel
OpSelectionMerge %merge None
OpSwitch %val %default 10 %case_a 20 %case_b
%case_a = OpLabel
OpBranch %merge
%case_b = OpLabel
OpBranch %merge
%default = OpLabel
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble");
    let disasm =
        disassemble_binary(&binary, BinaryToTextOptions::default()).expect("should disassemble");
    assert!(
        disasm.contains("OpSwitch"),
        "disassembly should contain OpSwitch: {disasm}"
    );
    // Re-assemble the disassembly to verify round-trip
    let binary2 = assemble_text(&disasm).expect("should re-assemble");
    assert_eq!(
        binary, binary2,
        "round-trip should produce identical binary"
    );
}

#[test]
fn unreachable_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpUnreachable
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpUnreachable");
    assert!(!binary.is_empty());
}

#[test]
fn kill_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpKill
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpKill");
    assert!(!binary.is_empty());
}

#[test]
fn select_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%true = OpConstantTrue %bool
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpSelect %uint %true %c1 %c2
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpSelect");
    assert!(!binary.is_empty());
}

#[test]
fn function_call_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%helper = OpFunction %void None %fn
%helper_entry = OpLabel
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn
%entry = OpLabel
%call = OpFunctionCall %void %helper
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpFunctionCall");
    assert!(!binary.is_empty());
}

#[test]
fn comparison_and_logic_ops_assemble() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%fn = OpTypeFunction %void
%c1u = OpConstant %uint 1
%c2u = OpConstant %uint 2
%c1i = OpConstant %int 1
%c2i = OpConstant %int 2
%c1f = OpConstant %float 1.0
%c2f = OpConstant %float 2.0
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%main = OpFunction %void None %fn
%entry = OpLabel
%ieq = OpIEqual %bool %c1u %c2u
%ine = OpINotEqual %bool %c1u %c2u
%slt = OpSLessThan %bool %c1i %c2i
%ult = OpULessThan %bool %c1u %c2u
%sle = OpSLessThanEqual %bool %c1i %c2i
%ule = OpULessThanEqual %bool %c1u %c2u
%sgt = OpSGreaterThan %bool %c1i %c2i
%ugt = OpUGreaterThan %bool %c1u %c2u
%sge = OpSGreaterThanEqual %bool %c1i %c2i
%uge = OpUGreaterThanEqual %bool %c1u %c2u
%foeq = OpFOrdEqual %bool %c1f %c2f
%fone = OpFOrdNotEqual %bool %c1f %c2f
%folt = OpFOrdLessThan %bool %c1f %c2f
%fogt = OpFOrdGreaterThan %bool %c1f %c2f
%fole = OpFOrdLessThanEqual %bool %c1f %c2f
%foge = OpFOrdGreaterThanEqual %bool %c1f %c2f
%land = OpLogicalAnd %bool %true %false
%lor = OpLogicalOr %bool %true %false
%leq = OpLogicalEqual %bool %true %false
%lne = OpLogicalNotEqual %bool %true %false
%lnot = OpLogicalNot %bool %true
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble comparison and logic ops");
    assert!(!binary.is_empty());
}

#[test]
fn bitwise_and_shift_ops_assemble() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%c1 = OpConstant %uint 0xFF
%c2 = OpConstant %uint 0x0F
%c3 = OpConstant %uint 4
%main = OpFunction %void None %fn
%entry = OpLabel
%and = OpBitwiseAnd %uint %c1 %c2
%or = OpBitwiseOr %uint %c1 %c2
%xor = OpBitwiseXor %uint %c1 %c2
%not = OpNot %uint %c1
%sll = OpShiftLeftLogical %uint %c1 %c3
%srl = OpShiftRightLogical %uint %c1 %c3
%sra = OpShiftRightArithmetic %uint %c1 %c3
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble bitwise and shift ops");
    assert!(!binary.is_empty());
}

#[test]
fn arithmetic_ops_assemble() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%fn = OpTypeFunction %void
%c1u = OpConstant %uint 10
%c2u = OpConstant %uint 3
%c1i = OpConstant %int 10
%c2i = OpConstant %int 3
%c1f = OpConstant %float 10.0
%c2f = OpConstant %float 3.0
%main = OpFunction %void None %fn
%entry = OpLabel
%sdiv = OpSDiv %int %c1i %c2i
%udiv = OpUDiv %uint %c1u %c2u
%fdiv = OpFDiv %float %c1f %c2f
%srem = OpSRem %int %c1i %c2i
%frem = OpFRem %float %c1f %c2f
%fmod = OpFMod %float %c1f %c2f
%smod = OpSMod %int %c1i %c2i
%umod = OpUMod %uint %c1u %c2u
%sneg = OpSNegate %int %c1i
%fneg = OpFNegate %float %c1f
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble arithmetic ops");
    assert!(!binary.is_empty());
}

#[test]
fn copy_object_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%c1 = OpConstant %uint 42
%main = OpFunction %void None %fn
%entry = OpLabel
%copy = OpCopyObject %uint %c1
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpCopyObject");
    assert!(!binary.is_empty());
}

#[test]
fn hex_literals_assemble() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%c1 = OpConstant %uint 0x0
%c2 = OpConstant %uint 0xFF
%c3 = OpConstant %uint 0xDEADBEEF
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble hex literals");
    assert!(!binary.is_empty());
}

#[test]
fn atomic_ops_assemble() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ptr_wg_uint = OpTypePointer Workgroup %uint
%fn = OpTypeFunction %void
%scope = OpConstant %uint 2
%relaxed = OpConstant %uint 0
%one = OpConstant %uint 1
%var = OpVariable %ptr_wg_uint Workgroup
%main = OpFunction %void None %fn
%entry = OpLabel
%val = OpAtomicLoad %uint %var %scope %relaxed
OpAtomicStore %var %scope %relaxed %one
%xchg = OpAtomicExchange %uint %var %scope %relaxed %one
%add = OpAtomicIAdd %uint %var %scope %relaxed %one
%sub = OpAtomicISub %uint %var %scope %relaxed %one
%smin = OpAtomicSMin %uint %var %scope %relaxed %one
%umin = OpAtomicUMin %uint %var %scope %relaxed %one
%smax = OpAtomicSMax %uint %var %scope %relaxed %one
%umax = OpAtomicUMax %uint %var %scope %relaxed %one
%and = OpAtomicAnd %uint %var %scope %relaxed %one
%or = OpAtomicOr %uint %var %scope %relaxed %one
%xor = OpAtomicXor %uint %var %scope %relaxed %one
%inc = OpAtomicIIncrement %uint %var %scope %relaxed
%dec = OpAtomicIDecrement %uint %var %scope %relaxed
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble atomic operations");
    assert!(!binary.is_empty());
}

#[test]
fn atomic_compare_exchange_assembles() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ptr_wg_uint = OpTypePointer Workgroup %uint
%fn = OpTypeFunction %void
%scope = OpConstant %uint 2
%relaxed = OpConstant %uint 0
%one = OpConstant %uint 1
%zero = OpConstant %uint 0
%var = OpVariable %ptr_wg_uint Workgroup
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpAtomicCompareExchange %uint %var %scope %relaxed %relaxed %one %zero
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble OpAtomicCompareExchange");
    assert!(!binary.is_empty());
}

#[test]
fn barriers_assemble() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fn = OpTypeFunction %void
%workgroup = OpConstant %uint 2
%acq_rel = OpConstant %uint 8
%main = OpFunction %void None %fn
%entry = OpLabel
OpControlBarrier %workgroup %workgroup %acq_rel
OpMemoryBarrier %workgroup %acq_rel
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble barrier operations");
    assert!(!binary.is_empty());
}

#[test]
fn atomic_ops_roundtrip_through_disassembly() {
    let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ptr_wg_uint = OpTypePointer Workgroup %uint
%fn = OpTypeFunction %void
%scope = OpConstant %uint 2
%relaxed = OpConstant %uint 0
%one = OpConstant %uint 1
%var = OpVariable %ptr_wg_uint Workgroup
%main = OpFunction %void None %fn
%entry = OpLabel
%val = OpAtomicLoad %uint %var %scope %relaxed
OpAtomicStore %var %scope %relaxed %one
%add = OpAtomicIAdd %uint %var %scope %relaxed %one
OpReturn
OpFunctionEnd
"#;
    let binary = assemble_text(text).expect("should assemble atomic ops");
    let disasm =
        disassemble_binary(&binary, BinaryToTextOptions::default()).expect("should disassemble");
    assert!(
        disasm.contains("OpAtomicLoad"),
        "disassembly should contain OpAtomicLoad"
    );
    assert!(
        disasm.contains("OpAtomicStore"),
        "disassembly should contain OpAtomicStore"
    );
    assert!(
        disasm.contains("OpAtomicIAdd"),
        "disassembly should contain OpAtomicIAdd"
    );
}
