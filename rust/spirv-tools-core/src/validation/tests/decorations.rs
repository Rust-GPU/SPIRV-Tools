use super::*;

#[test]
fn friendly_names_table_captures_member_name() {
    use crate::validation::ValidationOptions;
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpName %S \"Struct\"",
        "OpMemberName %S 1 \"member\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%S = OpTypeStruct %uint %uint",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let options = ValidationOptions {
        use_friendly_names: true,
        ..ValidationOptions::default()
    };
    let module = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect("validation should succeed");
    let names = module
        .friendly_names()
        .expect("friendly names should be present");
    let struct_id = module
        .module()
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == rspirv::spirv::Op::TypeStruct)
        .and_then(|inst| inst.result_id)
        .expect("struct should have a result id");
    assert_eq!(names.id(struct_id), Some("Struct"));
    assert_eq!(names.member(struct_id, MemberIndex(1)), Some("member"));
}

#[test]
fn localsizeid_disallowed_without_option_in_older_vulkan() {
    use crate::validation::ValidationOptions;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{ExecutionMode, ExecutionModel, FunctionControl};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let fn_ty = builder.type_function(void, []);
    let uint = builder.type_int(32, 0);
    let local_size = builder.constant_bit32(uint, 1);
    let entry_point = builder
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    builder.begin_block(None).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    builder.entry_point(ExecutionModel::GLCompute, entry_point, "main", []);
    builder.execution_mode_id(
        entry_point,
        ExecutionMode::LocalSizeId,
        [local_size, local_size, local_size],
    );
    let words = builder.module().assemble();
    let err = words
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_2, ValidationOptions::default())
        .expect_err("LocalSizeId should be disallowed without the option");
    assert_eq!(
        err,
        ValidationError::LocalSizeIdNotAllowed {
            env: TargetEnv::Vulkan1_2
        }
    );
}

#[test]
fn format_validation_error_uses_friendly_names_when_available() {
    use std::iter::FromIterator;
    let id = 42;
    let names = FriendlyNames::from_parts(
        HashMap::from_iter([(id, "named_func".to_string())]),
        HashMap::new(),
    );
    let error = ValidationError::ExecutionModeWithoutEntryPoint {
        function: Id::try_from(id).unwrap(),
    };
    let rendered = format_validation_error(&error, Some(&names));
    assert!(
        rendered.contains("named_func"),
        "expected friendly name in rendered error, got {rendered}"
    );
    let fallback = format_validation_error(&error, None);
    assert!(
        !fallback.contains("named_func"),
        "fallback should omit friendly name"
    );
}

#[test]
fn format_validation_error_from_words_parses_names() {
    use crate::validation::ValidationOptions;
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpExecutionMode %main LocalSize 1 1 1",
        "OpName %main \"friendly\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let options = ValidationOptions::default();
    let error = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options.clone())
        .expect_err("missing entry point should fail");
    let rendered = format_validation_error_from_words(binary.as_slice(), &options, &error);
    assert!(
        rendered.contains("friendly"),
        "expected rendered error to include friendly name, got {rendered}"
    );
}

#[test]
fn friendly_names_disabled_when_option_off() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpName %struct \"MyStruct\"",
        "%uint = OpTypeInt 32 0",
        "%struct = OpTypeStruct %uint",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let valid = binary
        .as_slice()
        .validate_with_options(
            TargetEnv::Universal1_6,
            ValidationOptions {
                use_friendly_names: false,
                ..ValidationOptions::default()
            },
        )
        .expect("validation should succeed");
    assert!(
        valid.friendly_names().is_none(),
        "friendly names should be omitted when disabled"
    );
}

#[test]
fn relax_struct_store_allows_layout_compatible_structs() {
    use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};
    fn inst(
        opcode: rspirv::spirv::Op,
        result_type: Option<u32>,
        result_id: Option<u32>,
        operands: Vec<rspirv::dr::Operand>,
    ) -> Instruction {
        Instruction::new(opcode, result_type, result_id, operands)
    }
    let mut module = Module::new();
    module.header = Some(ModuleHeader::new(11));
    module.capabilities.push(inst(
        rspirv::spirv::Op::Capability,
        None,
        None,
        vec![rspirv::dr::Operand::Capability(
            rspirv::spirv::Capability::Shader,
        )],
    ));
    module.memory_model = Some(inst(
        rspirv::spirv::Op::MemoryModel,
        None,
        None,
        vec![
            rspirv::dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
            rspirv::dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
        ],
    ));
    module.types_global_values.extend([
        inst(rspirv::spirv::Op::TypeVoid, None, Some(1), vec![]),
        inst(
            rspirv::spirv::Op::TypeInt,
            None,
            Some(2),
            vec![
                rspirv::dr::Operand::LiteralBit32(32),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(3),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(4),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(2)],
        ),
        inst(
            rspirv::spirv::Op::TypePointer,
            None,
            Some(5),
            vec![
                rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Function),
                rspirv::dr::Operand::IdRef(3),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeFunction,
            None,
            Some(6),
            vec![rspirv::dr::Operand::IdRef(1)],
        ),
    ]);
    module.functions.push(rspirv::dr::Function {
        def: Some(inst(
            rspirv::spirv::Op::Function,
            Some(1),
            Some(7),
            vec![
                rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                rspirv::dr::Operand::IdRef(6),
            ],
        )),
        end: Some(inst(rspirv::spirv::Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![rspirv::dr::Block {
            label: Some(inst(rspirv::spirv::Op::Label, None, Some(8), vec![])),
            instructions: vec![
                inst(
                    rspirv::spirv::Op::Variable,
                    Some(5),
                    Some(9),
                    vec![rspirv::dr::Operand::StorageClass(
                        rspirv::spirv::StorageClass::Function,
                    )],
                ),
                inst(rspirv::spirv::Op::Undef, Some(4), Some(10), vec![]),
                inst(
                    rspirv::spirv::Op::Store,
                    None,
                    None,
                    vec![
                        rspirv::dr::Operand::IdRef(9),
                        rspirv::dr::Operand::IdRef(10),
                    ],
                ),
                inst(rspirv::spirv::Op::Return, None, None, vec![]),
            ],
        }],
    });
    module.entry_points.push(inst(
        rspirv::spirv::Op::EntryPoint,
        None,
        None,
        vec![
            rspirv::dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Vertex),
            rspirv::dr::Operand::IdRef(7),
            rspirv::dr::Operand::LiteralString("main".to_string()),
        ],
    ));
    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("struct store types should mismatch by default");
    assert!(matches!(err, ValidationError::StoreTypeMismatch { .. }));
    let options = ValidationOptions {
        relax_struct_store: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect("relax_struct_store should permit layout-compatible structs");
}

#[test]
fn member_decorate_requires_struct_target() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%u32 = OpTypeInt 32 0",
        "OpMemberDecorate %u32 0 Offset 0",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::MemberDecorationTargetNotStruct {
            target: MemberDecorationTargetId::new(
                DecorationTargetId::try_from(1).unwrap(),
                MemberIndex::new(0)
            )
        }
    );
}

#[test]
fn member_decorate_requires_valid_member_index() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%u32 = OpTypeInt 32 0",
        "%vec2 = OpTypeStruct %u32 %u32",
        "OpMemberDecorate %vec2 2 Offset 0",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::MemberDecorationIndexOutOfRange {
            target: DecorationTargetId::try_from(2).unwrap(),
            member: MemberIndex::new(2),
            member_count: 2
        }
    );
}

#[test]
fn offset_requires_member_decorate() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%struct = OpTypeStruct %void",
        "OpDecorate %struct Offset 0",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::MemberOnlyDecorationUsedWithDecorate {
            decoration: rspirv::spirv::Decoration::Offset
        }
    );
}

#[test]
fn matrix_stride_requires_member_decorate() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%mat2 = OpTypeMatrix %vec2 2",
        "OpDecorate %mat2 MatrixStride 8",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::MemberOnlyDecorationUsedWithDecorate {
            decoration: rspirv::spirv::Decoration::MatrixStride
        }
    );
}

#[test]
fn row_major_requires_member_decorate() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%mat2 = OpTypeMatrix %vec2 2",
        "OpDecorate %mat2 RowMajor",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::MemberOnlyDecorationUsedWithDecorate {
            decoration: rspirv::spirv::Decoration::RowMajor
        }
    );
}

#[test]
fn col_major_requires_member_decorate() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%mat2 = OpTypeMatrix %vec2 2",
        "OpDecorate %mat2 ColMajor",
    ]
    .join("\n");
    let error = MaybeValidModule::Text(&text)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(
        error,
        ValidationError::MemberOnlyDecorationUsedWithDecorate {
            decoration: rspirv::spirv::Decoration::ColMajor
        }
    );
}

#[test]
fn group_decorate_requires_declared_group() {
    // The text assembler refuses to emit binaries with invalid decoration groups, so we
    // hand-build the binary to drive the validator directly.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        2,          // bound (ids up to 1)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        0x0006000b, // OpExtInstImport %1 "GLSL.std.450"
        1,
        0x4c53_4c47,
        0x2e73_7464,
        0x3035_342e,
        0,         // null terminator for the import string
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        0x0003004a, // OpGroupDecorate %1 %1 (invalid group id)
        1,
        1,
    ];
    let expected = ValidationError::UnknownDecorationGroup {
        group: Id::try_from(1).unwrap(),
    };
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(error, expected);
}

#[test]
fn decorate_requires_declared_target() {
    // The text assembler enforces target existence up front, so use a binary to ensure the
    // validator catches the missing target.
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        3,          // bound (ids up to 2)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        0x00030047, // OpDecorate %2 RelaxedPrecision (target %2 is undefined)
        2,
        rspirv::spirv::Decoration::RelaxedPrecision as u32,
    ];
    let expected = ValidationError::MissingDecorationTarget {
        target: Id::try_from(2).unwrap(),
    };
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(error, expected);
}

#[test]
fn group_member_decorate_requires_declared_targets() {
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        5,          // bound (ids up to 4)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1
        1,
        op(4, 75), // OpGroupMemberDecorate %1 %4 0 (target %4 is undefined)
        1,
        4,
        0,
        op(2, 19), // OpTypeVoid %2
        2,
        op(3, 33), // OpTypeFunction %3 %2
        3,
        2,
    ];
    let expected = ValidationError::MissingDecorationTarget {
        target: Id::try_from(4).unwrap(),
    };
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(error, expected);
}

#[test]
fn group_member_decorate_requires_struct_targets() {
    let binary = vec![
        0x07230203, // magic
        0x00010000, // version
        0,          // generator
        6,          // bound (ids up to 5)
        0,          // schema
        op(2, 17),  // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 73), // OpDecorationGroup %1
        1,
        op(4, 75), // OpGroupMemberDecorate %1 %4 0 (%4 is not a struct)
        1,
        4,
        0,
        op(4, 21), // OpTypeInt %2 32
        2,
        32,
        0,
        op(3, 33), // OpTypeFunction %3 %2
        3,
        2,
        op(3, 22), // OpTypeFloat %4 32 (non-struct target)
        4,
        32,
    ];
    let expected = ValidationError::MemberDecorationTargetNotStruct {
        target: MemberDecorationTargetId::new(
            DecorationTargetId::new(OperandId::try_from(4u32).unwrap()),
            MemberIndex::new(0),
        ),
    };
    let error = MaybeValidModule::Binary(&binary)
        .validate(TargetEnv::Universal1_6)
        .unwrap_err();
    assert_eq!(error, expected);
}

#[test]
fn spec_id_requires_scalar_specialization_constant() {
    let text = [
        "OpCapability Addresses",
        "OpCapability Kernel",
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeInt 32 1",
        "%2 = OpConstant %1 1",
        "OpDecorate %2 SpecId 7",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble spec id decoration");
    let expected = ValidationError::InvalidDecorationTargetKind {
        decoration: rspirv::spirv::Decoration::SpecId,
        target: Id::try_from(2).unwrap(),
        found: rspirv::spirv::Op::Constant,
        expected: DecorationTargetKind::ScalarSpecConstant,
    };
    for module in [
        MaybeValidModule::Text(text.as_str()),
        MaybeValidModule::Binary(binary.as_slice()),
    ] {
        let error = module
            .validate(TargetEnv::Universal1_6)
            .expect_err("SpecId must target scalar specialization constants");
        assert_eq!(error, expected);
    }
}

#[test]
fn block_requires_struct_type_target() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeInt 32 0",
        "OpDecorate %1 Block",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble block decoration");
    let expected = ValidationError::InvalidDecorationTargetKind {
        decoration: rspirv::spirv::Decoration::Block,
        target: Id::try_from(1).unwrap(),
        found: rspirv::spirv::Op::TypeInt,
        expected: DecorationTargetKind::StructType,
    };
    for module in [
        MaybeValidModule::Text(text.as_str()),
        MaybeValidModule::Binary(binary.as_slice()),
    ] {
        let error = module
            .validate(TargetEnv::Universal1_6)
            .expect_err("Block must target a struct type");
        assert_eq!(error, expected);
    }
}

#[test]
fn array_stride_requires_array_or_pointer_target() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeInt 32 0",
        "OpDecorate %1 ArrayStride 16",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble array stride decoration");
    let expected = ValidationError::InvalidDecorationTargetKind {
        decoration: rspirv::spirv::Decoration::ArrayStride,
        target: Id::try_from(1).unwrap(),
        found: rspirv::spirv::Op::TypeInt,
        expected: DecorationTargetKind::ArrayOrPointerType,
    };
    for module in [
        MaybeValidModule::Text(text.as_str()),
        MaybeValidModule::Binary(binary.as_slice()),
    ] {
        let error = module
            .validate(TargetEnv::Universal1_6)
            .expect_err("ArrayStride must target array/runtime array/pointer types");
        assert_eq!(error, expected);
    }
}

#[test]
fn workgroup_size_builtin_requires_constant_when_shader() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeInt 32 0",
        "%2 = OpTypeVector %1 3",
        "%3 = OpTypePointer Input %2",
        "%4 = OpVariable %3 Input",
        "OpDecorate %4 BuiltIn WorkgroupSize",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble workgroup size builtin");
    let expected = ValidationError::InvalidDecorationTargetKind {
        decoration: rspirv::spirv::Decoration::BuiltIn,
        target: Id::try_from(4).unwrap(),
        found: rspirv::spirv::Op::Variable,
        expected: DecorationTargetKind::Constant,
    };
    for module in [
        MaybeValidModule::Text(text.as_str()),
        MaybeValidModule::Binary(binary.as_slice()),
    ] {
        let error = module
            .validate(TargetEnv::Universal1_6)
            .expect_err("WorkgroupSize must target a constant when Shader is declared");
        assert_eq!(error, expected);
    }
}

#[test]
fn memory_object_decorations_require_memory_objects() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%1 = OpTypeInt 32 0",
        "%2 = OpConstant %1 0",
        "OpDecorate %2 NoPerspective",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble NoPerspective decoration");
    let expected = ValidationError::InvalidDecorationTargetKind {
        decoration: rspirv::spirv::Decoration::NoPerspective,
        target: Id::try_from(2).unwrap(),
        found: rspirv::spirv::Op::Constant,
        expected: DecorationTargetKind::MemoryObjectDeclaration,
    };
    for module in [
        MaybeValidModule::Text(text.as_str()),
        MaybeValidModule::Binary(binary.as_slice()),
    ] {
        let error = module
            .validate(TargetEnv::Universal1_6)
            .expect_err("memory object decorations must target memory object declarations");
        assert_eq!(error, expected);
    }
}

#[test]
fn function_definition_with_import_linkage_is_rejected() {
    // A function definition (with blocks) cannot have Import linkage
    let text = r#"
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %main LinkageAttributes "main" Import
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("function definition with import linkage should be rejected");
    assert!(
        matches!(
            err,
            ValidationError::FunctionDefinitionHasImportLinkage { .. }
        ),
        "Expected FunctionDefinitionHasImportLinkage, got: {err:?}"
    );
}

#[test]
fn function_declaration_without_import_linkage_is_rejected_when_linkage_capability_present() {
    // A function declaration (no blocks) must have Import linkage when Linkage capability is declared
    let text = r#"
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%fn = OpTypeFunction %void
%func = OpFunction %void None %fn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("function declaration without import linkage should be rejected when Linkage capability present");
    assert!(
        matches!(
            err,
            ValidationError::FunctionDeclarationMissingImportLinkage { .. }
        ),
        "Expected FunctionDeclarationMissingImportLinkage, got: {err:?}"
    );
}

#[test]
fn function_declaration_with_import_linkage_is_allowed() {
    // A function declaration (no blocks) with Import linkage is valid
    let text = r#"
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %func LinkageAttributes "external_func" Import
%void = OpTypeVoid
%fn = OpTypeFunction %void
%func = OpFunction %void None %fn
OpFunctionEnd
"#;
    assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect("function declaration with import linkage should be valid");
}

#[test]
fn imported_variable_with_initializer_is_rejected() {
    // An imported variable cannot have an initializer
    let text = r#"
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var LinkageAttributes "external_var" Import
%int = OpTypeInt 32 0
%ptr = OpTypePointer Private %int
%const = OpConstant %int 42
%var = OpVariable %ptr Private %const
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(text, TargetEnv::Universal1_6)
        .expect_err("imported variable with initializer should be rejected");
    assert!(
        matches!(err, ValidationError::ImportedVariableHasInitializer { .. }),
        "Expected ImportedVariableHasInitializer, got: {err:?}"
    );
}

#[test]
fn vulkan_memory_model_deprecates_coherent_decoration() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 5);
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::VulkanMemoryModel);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Vulkan,
    );

    let void = b.type_void();
    let int = b.type_int(32, 0);
    let ptr = b.type_pointer(None, rspirv::spirv::StorageClass::StorageBuffer, int);
    let var = b.variable(ptr, None, rspirv::spirv::StorageClass::StorageBuffer, None);
    b.decorate(var, rspirv::spirv::Decoration::Coherent, []);
    b.decorate(
        var,
        rspirv::spirv::Decoration::DescriptorSet,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );
    b.decorate(
        var,
        rspirv::spirv::Decoration::Binding,
        [rspirv::dr::Operand::LiteralBit32(0)],
    );

    let fn_type = b.type_function(void, std::iter::empty::<u32>());
    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
        .unwrap();
    b.begin_block(None).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    assert!(
        matches!(
            err,
            ValidationError::VulkanMemoryModelDeprecatesDecoration {
                decoration: rspirv::spirv::Decoration::Coherent
            }
        ),
        "Expected VulkanMemoryModelDeprecatesDecoration for Coherent, got: {err:?}"
    );
}

#[test]
fn integer_wrap_decoration_on_non_integer_op_is_rejected() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 4);
    b.capability(rspirv::spirv::Capability::Shader);
    b.extension("SPV_KHR_no_integer_wrap_decoration");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );

    let void = b.type_void();
    let float = b.type_float(32, None);
    let fn_type = b.type_function(void, std::iter::empty::<u32>());
    let f1 = b.constant_bit32(float, 1.0f32.to_bits());
    let f2 = b.constant_bit32(float, 2.0f32.to_bits());

    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
        .unwrap();
    b.begin_block(None).unwrap();
    // FAdd is not an integer operation
    let add_result = b.f_add(float, None, f1, f2).unwrap();
    // Decorate FAdd with NoSignedWrap - should be invalid
    b.decorate(add_result, rspirv::spirv::Decoration::NoSignedWrap, []);
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    let err = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert!(
        matches!(err, ValidationError::IntegerWrapDecorationInvalidOp { .. }),
        "Expected IntegerWrapDecorationInvalidOp, got: {err:?}"
    );
}

#[test]
fn integer_wrap_decoration_on_iadd_is_allowed() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 4);
    b.capability(rspirv::spirv::Capability::Shader);
    b.extension("SPV_KHR_no_integer_wrap_decoration");
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );

    let void = b.type_void();
    let int = b.type_int(32, 0);
    let fn_type = b.type_function(void, std::iter::empty::<u32>());
    let i1 = b.constant_bit32(int, 1);
    let i2 = b.constant_bit32(int, 2);

    b.begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_type)
        .unwrap();
    b.begin_block(None).unwrap();
    let add_result = b.i_add(int, None, i1, i2).unwrap();
    b.decorate(add_result, rspirv::spirv::Decoration::NoSignedWrap, []);
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let binary = module.assemble();
    validate_module(&binary, TargetEnv::Universal1_6)
        .expect("NoSignedWrap on IAdd should be valid");
}

#[test]
fn relaxed_precision_without_shader_capability_is_rejected() {
    // RelaxedPrecision requires Shader capability - caught by the capability grammar check
    let text = r#"
OpCapability Kernel
OpCapability Addresses
OpMemoryModel Physical64 OpenCL
OpDecorate %var RelaxedPrecision
%int = OpTypeInt 32 0
%ptr = OpTypePointer CrossWorkgroup %int
%var = OpVariable %ptr CrossWorkgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let err = assemble_and_validate_with_env(text, TargetEnv::OpenCl2_2)
        .expect_err("RelaxedPrecision without Shader capability should be rejected");
    // The capability grammar validation catches this first
    assert!(
        matches!(
            err,
            ValidationError::MissingOperandCapability {
                required_capability: rspirv::spirv::Capability::Shader,
                ..
            }
        ),
        "Expected MissingOperandCapability for Shader, got: {err:?}"
    );
}
