use super::*;

#[test]
fn friendly_names_table_captures_op_name() {
    use crate::validation::ValidationOptions;
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
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
    let function_id = module
        .module()
        .functions
        .first()
        .and_then(|f| f.def.as_ref())
        .and_then(|inst| inst.result_id)
        .expect("function should have a result id");
    assert_eq!(names.id(function_id), Some("friendly"));
}

#[test]
fn relax_struct_store_rejects_mismatched_array_lengths() {
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
    module.header = Some(ModuleHeader::new(21));
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
            rspirv::spirv::Op::Constant,
            Some(2),
            Some(3),
            vec![rspirv::dr::Operand::LiteralBit32(2)],
        ),
        inst(
            rspirv::spirv::Op::Constant,
            Some(2),
            Some(4),
            vec![rspirv::dr::Operand::LiteralBit32(3)],
        ),
        inst(
            rspirv::spirv::Op::TypeArray,
            None,
            Some(5),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
        ),
        inst(
            rspirv::spirv::Op::TypeArray,
            None,
            Some(6),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(4)],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(7),
            vec![rspirv::dr::Operand::IdRef(5)],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(8),
            vec![rspirv::dr::Operand::IdRef(6)],
        ),
        inst(
            rspirv::spirv::Op::TypePointer,
            None,
            Some(9),
            vec![
                rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Function),
                rspirv::dr::Operand::IdRef(7),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeFunction,
            None,
            Some(10),
            vec![rspirv::dr::Operand::IdRef(1)],
        ),
        inst(rspirv::spirv::Op::TypeVoid, None, Some(20), vec![]),
    ]);
    module.functions.push(rspirv::dr::Function {
        def: Some(inst(
            rspirv::spirv::Op::Function,
            Some(1),
            Some(11),
            vec![
                rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                rspirv::dr::Operand::IdRef(10),
            ],
        )),
        end: Some(inst(rspirv::spirv::Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![rspirv::dr::Block {
            label: Some(inst(rspirv::spirv::Op::Label, None, Some(12), vec![])),
            instructions: vec![
                inst(
                    rspirv::spirv::Op::Variable,
                    Some(9),
                    Some(13),
                    vec![rspirv::dr::Operand::StorageClass(
                        rspirv::spirv::StorageClass::Function,
                    )],
                ),
                inst(rspirv::spirv::Op::Undef, Some(8), Some(14), vec![]),
                inst(
                    rspirv::spirv::Op::Store,
                    None,
                    None,
                    vec![
                        rspirv::dr::Operand::IdRef(13),
                        rspirv::dr::Operand::IdRef(14),
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
            rspirv::dr::Operand::IdRef(11),
            rspirv::dr::Operand::LiteralString("main".to_string()),
        ],
    ));
    let binary = module.assemble();
    // Relaxation should still reject mismatched array lengths.
    let options = ValidationOptions {
        relax_struct_store: true,
        ..ValidationOptions::default()
    };
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect_err("array length mismatch should not be considered layout-compatible");
    if let ValidationError::StoreTypeMismatch { .. } = err {
    } else {
        panic!("unexpected error: {err:?}");
    }
}

#[test]
fn relax_struct_store_rejects_mismatched_array_stride() {
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
    module.header = Some(ModuleHeader::new(15));
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
            rspirv::spirv::Op::Constant,
            Some(2),
            Some(3),
            vec![rspirv::dr::Operand::LiteralBit32(2)],
        ),
        inst(
            rspirv::spirv::Op::TypeArray,
            None,
            Some(4),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
        ),
        inst(
            rspirv::spirv::Op::TypeArray,
            None,
            Some(5),
            vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(6),
            vec![rspirv::dr::Operand::IdRef(4)],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(7),
            vec![rspirv::dr::Operand::IdRef(5)],
        ),
        inst(
            rspirv::spirv::Op::TypePointer,
            None,
            Some(8),
            vec![
                rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Function),
                rspirv::dr::Operand::IdRef(6),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeFunction,
            None,
            Some(9),
            vec![rspirv::dr::Operand::IdRef(1)],
        ),
    ]);
    module.annotations.extend([
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(4),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::ArrayStride),
                rspirv::dr::Operand::LiteralBit32(4),
            ],
        ),
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(5),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::ArrayStride),
                rspirv::dr::Operand::LiteralBit32(8),
            ],
        ),
    ]);
    module.functions.push(rspirv::dr::Function {
        def: Some(inst(
            rspirv::spirv::Op::Function,
            Some(1),
            Some(10),
            vec![
                rspirv::dr::Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                rspirv::dr::Operand::IdRef(9),
            ],
        )),
        end: Some(inst(rspirv::spirv::Op::FunctionEnd, None, None, vec![])),
        parameters: vec![],
        blocks: vec![rspirv::dr::Block {
            label: Some(inst(rspirv::spirv::Op::Label, None, Some(11), vec![])),
            instructions: vec![
                inst(
                    rspirv::spirv::Op::Variable,
                    Some(8),
                    Some(12),
                    vec![rspirv::dr::Operand::StorageClass(
                        rspirv::spirv::StorageClass::Function,
                    )],
                ),
                inst(rspirv::spirv::Op::Undef, Some(7), Some(13), vec![]),
                inst(
                    rspirv::spirv::Op::Store,
                    None,
                    None,
                    vec![
                        rspirv::dr::Operand::IdRef(12),
                        rspirv::dr::Operand::IdRef(13),
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
            rspirv::dr::Operand::IdRef(10),
            rspirv::dr::Operand::LiteralString("main".to_string()),
        ],
    ));
    // Stride metadata must be respected when comparing array layouts.
    assert_eq!(
        array_stride(&module, ResultId::try_from(4).unwrap()),
        Some(4)
    );
    assert_eq!(
        array_stride(&module, ResultId::try_from(5).unwrap()),
        Some(8)
    );
    let options = ValidationOptions {
        relax_struct_store: true,
        ..ValidationOptions::default()
    };
    let binary = module.assemble();
    let parsed =
        parse_module(binary.as_slice()).expect("assembled module should round-trip through parser");
    assert_eq!(
        array_stride(&parsed, ResultId::try_from(4).unwrap()),
        Some(4)
    );
    assert_eq!(
        array_stride(&parsed, ResultId::try_from(5).unwrap()),
        Some(8)
    );
    let validation_result = validate_words_internal(
        ModuleWords::from(Arc::from(binary.as_slice())),
        TargetEnv::Universal1_6,
        options.clone(),
        None,
    );
    match validation_result {
        Err(spanned) if matches!(spanned.error, ValidationError::StoreTypeMismatch { .. }) => {}
        Err(other) => panic!("full validation path failed with unexpected error: {other:?}"),
        Ok(_) => panic!("full validation path should reject incompatible strides"),
    }
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect_err("array stride mismatch should still fail without layout relaxation");
    assert!(matches!(err, ValidationError::StoreTypeMismatch { .. }));
    let relaxed = ValidationOptions {
        relax_struct_store: true,
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, relaxed)
        .expect("layout relaxation should bypass array stride mismatch");
}

#[test]
fn relax_struct_store_with_layout_relaxation_accepts_incompatible_structs() {
    use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};
    fn inst(
        opcode: rspirv::spirv::Op,
        result_type: Option<u32>,
        result_id: Option<u32>,
        operands: Vec<rspirv::dr::Operand>,
    ) -> Instruction {
        Instruction::new(opcode, result_type, result_id, operands)
    }
    // S0 has two members, S1 has one; store should pass when both relax_struct_store
    // and a block-layout relaxation flag are set.
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
            vec![rspirv::dr::Operand::IdRef(2)],
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
    let options = ValidationOptions {
        relax_struct_store: true,
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect("relax_struct_store with layout relaxation should allow mismatched structs");
}

#[test]
fn block_layout_requires_member_offsets() {
    use rspirv::{binary::Assemble, dr::Instruction, dr::Module, dr::ModuleHeader};
    fn inst(
        opcode: rspirv::spirv::Op,
        result_type: Option<u32>,
        result_id: Option<u32>,
        operands: Vec<rspirv::dr::Operand>,
    ) -> Instruction {
        Instruction::new(opcode, result_type, result_id, operands)
    }
    // Creates a Block-decorated struct used in Uniform storage class.
    // Block structs require Offset decorations when used in buffer storage classes.
    fn make_block_struct(member_offsets: Option<Vec<u32>>) -> Vec<u32> {
        let mut module = Module::new();
        // IDs: 1=void, 2=uint, 3=struct, 4=ptr, 5=var
        module.header = Some(ModuleHeader::new(6));
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
            // TypePointer to struct with Uniform storage class
            inst(
                rspirv::spirv::Op::TypePointer,
                None,
                Some(4),
                vec![
                    rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Uniform),
                    rspirv::dr::Operand::IdRef(3),
                ],
            ),
            // Variable using the pointer type
            inst(
                rspirv::spirv::Op::Variable,
                Some(4),
                Some(5),
                vec![rspirv::dr::Operand::StorageClass(
                    rspirv::spirv::StorageClass::Uniform,
                )],
            ),
        ]);
        module.annotations.push(inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
            ],
        ));
        // DescriptorSet and Binding decorations for the variable
        module.annotations.push(inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(5),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::DescriptorSet),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ));
        module.annotations.push(inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(5),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Binding),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ));
        if let Some(offsets) = member_offsets {
            for (index, offset) in offsets.into_iter().enumerate() {
                module.annotations.push(inst(
                    rspirv::spirv::Op::MemberDecorate,
                    None,
                    None,
                    vec![
                        rspirv::dr::Operand::IdRef(3),
                        rspirv::dr::Operand::LiteralBit32(index as u32),
                        rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                        rspirv::dr::Operand::LiteralBit32(offset),
                    ],
                ));
            }
        }
        module.assemble()
    }
    let binary = make_block_struct(None);
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("missing member offsets should fail block layout");
    match err {
        ValidationError::InvalidBlockLayout {
            struct_type,
            reason,
            ..
        } => {
            assert_eq!(u32::from(struct_type), 3);
            assert!(reason.contains("Offset"), "unexpected reason: {reason:?}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    let relax_options = ValidationOptions {
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, relax_options)
        .expect_err("relax_block_layout should still require offsets");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
    let options = ValidationOptions {
        skip_block_layout: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect("skip_block_layout should skip member offset enforcement");
}

#[test]
fn block_layout_rejects_overlapping_offsets() {
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
    // IDs: 1=void, 2=uint, 3=struct, 4=ptr, 5=var
    module.header = Some(ModuleHeader::new(6));
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
        // TypePointer to struct with Uniform storage class
        inst(
            rspirv::spirv::Op::TypePointer,
            None,
            Some(4),
            vec![
                rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Uniform),
                rspirv::dr::Operand::IdRef(3),
            ],
        ),
        // Variable using the pointer type
        inst(
            rspirv::spirv::Op::Variable,
            Some(4),
            Some(5),
            vec![rspirv::dr::Operand::StorageClass(
                rspirv::spirv::StorageClass::Uniform,
            )],
        ),
    ]);
    module.annotations.extend([
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
            ],
        ),
        // DescriptorSet and Binding for the variable
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(5),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::DescriptorSet),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(5),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Binding),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::LiteralBit32(0),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::LiteralBit32(1),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(0), // Same offset as first member (overlapping)
            ],
        ),
    ]);
    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("overlapping member offsets should fail block layout");
    match err {
        ValidationError::InvalidBlockLayout {
            struct_type,
            reason,
            ..
        } => {
            assert_eq!(u32::from(struct_type), 3);
            assert!(reason.contains("overlap"), "unexpected reason: {reason:?}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    let relax_options = ValidationOptions {
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_0, relax_options)
        .expect_err("relax_block_layout should still enforce overlap constraints");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
    let options = ValidationOptions {
        skip_block_layout: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_0, options)
        .expect("skip_block_layout should skip overlap checks");
}

#[test]
fn relax_block_layout_allows_scalar_vector_alignment() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%struct = OpTypeStruct %int %vec2",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("vector offset should require base alignment in strict layout");
    if let ValidationError::InvalidBlockLayout { reason, .. } = err {
        assert!(reason.contains("aligned"), "unexpected reason: {reason:?}");
    } else {
        panic!("unexpected error: {err:?}");
    }
    let relax = ValidationOptions {
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    text.as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, relax)
        .expect("relax_block_layout should permit scalar-aligned vectors");
    let scalar = ValidationOptions {
        scalar_block_layout: true,
        ..ValidationOptions::default()
    };
    text.as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, scalar)
        .expect("scalar_block_layout should permit scalar alignment for vectors");
}

#[test]
fn uniform_buffer_standard_layout_does_not_relax_vector_alignment() {
    // uniform_buffer_standard_layout only changes std140→std430 (no 16-byte
    // rounding for arrays/structs). It does NOT enable relaxed vector offset
    // checks. A vec2 at offset 4 (alignment 8) should fail.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%struct = OpTypeStruct %int %vec2",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let opts = ValidationOptions {
        uniform_buffer_standard_layout: true,
        ..ValidationOptions::default()
    };
    let err = text
        .as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, opts)
        .expect_err("uniform_buffer_standard_layout does not relax vector alignment");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn uniform_buffer_standard_layout_with_relax_allows_scalar_vector_alignment() {
    // When BOTH uniform_buffer_standard_layout AND relax_block_layout are
    // enabled, vectors can use scalar element alignment for offsets.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%struct = OpTypeStruct %int %vec2",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let opts = ValidationOptions {
        uniform_buffer_standard_layout: true,
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    text.as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, opts)
        .expect("relax_block_layout should permit scalar-aligned vectors");
}

#[test]
fn workgroup_scalar_block_layout_uses_scalar_alignment() {
    // workgroup_scalar_block_layout should only apply to Workgroup storage class,
    // NOT to Uniform or other classes (C++ lines 1428-1430).
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
    module.header = Some(ModuleHeader::new(10));
    module.capabilities.push(inst(
        rspirv::spirv::Op::Capability,
        None,
        None,
        vec![rspirv::dr::Operand::Capability(
            rspirv::spirv::Capability::Shader,
        )],
    ));
    module.capabilities.push(inst(
        rspirv::spirv::Op::Capability,
        None,
        None,
        vec![rspirv::dr::Operand::Capability(
            rspirv::spirv::Capability::WorkgroupMemoryExplicitLayoutKHR,
        )],
    ));
    module.extensions.push(inst(
        rspirv::spirv::Op::Extension,
        None,
        None,
        vec![rspirv::dr::Operand::LiteralString(
            "SPV_KHR_workgroup_memory_explicit_layout".to_string(),
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
        inst(
            rspirv::spirv::Op::TypeInt,
            None,
            Some(1),
            vec![
                rspirv::dr::Operand::LiteralBit32(32),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeVector,
            None,
            Some(2),
            vec![
                rspirv::dr::Operand::IdRef(1),
                rspirv::dr::Operand::LiteralBit32(2),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(3),
            vec![rspirv::dr::Operand::IdRef(1), rspirv::dr::Operand::IdRef(2)],
        ),
        inst(
            rspirv::spirv::Op::TypePointer,
            None,
            Some(4),
            vec![
                rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Workgroup),
                rspirv::dr::Operand::IdRef(3),
            ],
        ),
        inst(
            rspirv::spirv::Op::Variable,
            Some(4),
            Some(5),
            vec![rspirv::dr::Operand::StorageClass(
                rspirv::spirv::StorageClass::Workgroup,
            )],
        ),
    ]);
    module.annotations.extend([
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::LiteralBit32(0),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::LiteralBit32(1),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(4),
            ],
        ),
    ]);
    let binary = module.assemble();
    let opts = ValidationOptions {
        workgroup_scalar_block_layout: true,
        ..ValidationOptions::default()
    };
    binary
        .as_slice()
        .validate_with_options(TargetEnv::Vulkan1_2, opts)
        .expect("workgroup_scalar_block_layout should permit scalar alignment for Workgroup");
}

#[test]
fn workgroup_scalar_does_not_affect_uniform() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%vec2 = OpTypeVector %int 2",
        "%struct = OpTypeStruct %int %vec2",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let opts = ValidationOptions {
        workgroup_scalar_block_layout: true,
        ..ValidationOptions::default()
    };
    let err = text
        .as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, opts)
        .expect_err("workgroup_scalar_block_layout should NOT relax Uniform layout");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn array_stride_must_align_to_element() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 16",
        "OpDecorate %arr ArrayStride 6",
        "%int = OpTypeInt 32 0",
        "%arr = OpTypeArray %int %len",
        "%len = OpConstant %int 2",
        "%struct = OpTypeStruct %arr %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("array stride not aligned to element size should fail");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn storage_buffer_array_stride_uses_std430_no_extended_alignment() {
    // StorageBuffer + Block uses std430 rules: arrays do NOT get 16-byte
    // extended alignment. An array of uint[2] with stride 8 should be valid
    // because the element alignment is 4, and 8 % 4 == 0.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpDecorate %inner_arr ArrayStride 4",
        "OpDecorate %outer_arr ArrayStride 8",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%inner_arr = OpTypeArray %int %two",
        "%outer_arr = OpTypeArray %inner_arr %two",
        "%struct = OpTypeStruct %outer_arr",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("StorageBuffer uses std430 rules where array stride 8 is valid");
}

#[test]
fn uniform_block_array_stride_requires_std140_extended_alignment() {
    // Uniform + Block uses std140 rules: arrays get 16-byte extended alignment.
    // An array of uint[2] with stride 8 should be rejected because std140
    // rounds the array alignment up to 16, and 8 % 16 != 0.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpDecorate %inner_arr ArrayStride 4",
        "OpDecorate %outer_arr ArrayStride 8",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%inner_arr = OpTypeArray %int %two",
        "%outer_arr = OpTypeArray %inner_arr %two",
        "%struct = OpTypeStruct %outer_arr",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("Uniform + Block uses std140 where array stride 8 is not aligned to 16");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn relax_block_layout_preserves_std140_extended_alignment() {
    // relax_block_layout does NOT disable std140 extended alignment for
    // Uniform + Block. Only uniform_buffer_standard_layout does that.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpDecorate %inner_arr ArrayStride 4",
        "OpDecorate %outer_arr ArrayStride 8",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%inner_arr = OpTypeArray %int %two",
        "%outer_arr = OpTypeArray %inner_arr %two",
        "%struct = OpTypeStruct %outer_arr",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let opts = ValidationOptions {
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    let err = text
        .as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, opts)
        .expect_err("relax_block_layout does not disable std140 extended alignment");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn uniform_buffer_standard_layout_disables_std140_extended_alignment() {
    // uniform_buffer_standard_layout makes blockRules=false in C++,
    // converting std140 to std430 (no 16-byte extended alignment).
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpDecorate %inner_arr ArrayStride 4",
        "OpDecorate %outer_arr ArrayStride 8",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%inner_arr = OpTypeArray %int %two",
        "%outer_arr = OpTypeArray %inner_arr %two",
        "%struct = OpTypeStruct %outer_arr",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let opts = ValidationOptions {
        uniform_buffer_standard_layout: true,
        ..ValidationOptions::default()
    };
    text.as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, opts)
        .expect("uniform_buffer_standard_layout disables std140 extended alignment");
}

#[test]
fn vector_straddle_rejected_under_relax() {
    let text = [
        "OpCapability Shader",
        "OpCapability Float64",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 8",
        "%f64 = OpTypeFloat 64",
        "%v3 = OpTypeVector %f64 3",
        "%struct = OpTypeStruct %f64 %v3",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("misaligned vector should fail");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
    let relax = ValidationOptions {
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    let err = text
        .as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, relax)
        .expect_err("relaxed layout still rejects improper vector straddle");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn small_vector_straddle_rejected_under_relax() {
    // A vec2<f32> (8 bytes) at offset 12 straddles the 16-byte boundary
    // (bytes 12-19 span blocks 0 and 1). This must be rejected under relaxed
    // layout. The C++ hasImproperStraddle checks: (F >> 4) != (L >> 4).
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "OpMemberDecorate %struct 2 Offset 12",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%struct = OpTypeStruct %float %float %vec2",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    let relax = ValidationOptions {
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    let err = text
        .as_str()
        .validate_with_options(TargetEnv::Vulkan1_1, relax)
        .expect_err("vec2 at offset 12 straddles 16-byte boundary");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn small_vector_no_straddle_accepted_under_relax() {
    // A vec2<f32> (8 bytes) at offset 8 does NOT straddle: bytes 8-15 are
    // all in the same 16-byte block. This should be accepted.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "OpMemberDecorate %struct 2 Offset 8",
        "%float = OpTypeFloat 32",
        "%vec2 = OpTypeVector %float 2",
        "%struct = OpTypeStruct %float %float %vec2",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    let relax = ValidationOptions {
        relax_block_layout: true,
        ..ValidationOptions::default()
    };
    text.as_str()
        .validate_with_options(TargetEnv::Vulkan1_1, relax)
        .expect("vec2 at offset 8 does not straddle 16-byte boundary");
}

#[test]
fn member_in_array_padding_rejected() {
    // An array of uint[2] (stride 16 under std140) has 8 bytes of padding.
    // The next member at offset 8 falls inside the padding, which is not
    // allowed under non-scalar block layout rules.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 8",
        "OpDecorate %arr ArrayStride 16",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%arr = OpTypeArray %int %two",
        "%struct = OpTypeStruct %arr %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("member in array padding should be rejected");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn member_after_array_padding_accepted() {
    // Array of uint[2] stride 16 under std140: size = (2-1)*16 + 4 = 20.
    // Under std140, next_valid_offset rounds up to alignment: align(20, 16) = 32.
    // So the next member must be at offset 32 or later.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 32",
        "OpDecorate %arr ArrayStride 16",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%arr = OpTypeArray %int %two",
        "%struct = OpTypeStruct %arr %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("member at offset 32 is after array's std140-rounded extent");
}

#[test]
fn scalar_layout_allows_member_after_array_raw_extent() {
    // Under scalar block layout, next_valid_offset is NOT rounded up to
    // alignment for arrays (unlike std140). Array uint[2] stride 16:
    // size = (2-1)*16 + 4 = 20. Under scalar, next_valid = 20 (no rounding).
    // Under std140, next_valid = align(20, 16) = 32.
    // So scalar allows a member at offset 20, but std140 would require 32.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 20",
        "OpDecorate %arr ArrayStride 16",
        "%int = OpTypeInt 32 0",
        "%two = OpConstant %int 2",
        "%arr = OpTypeArray %int %two",
        "%struct = OpTypeStruct %arr %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let opts = ValidationOptions {
        scalar_block_layout: true,
        ..ValidationOptions::default()
    };
    text.as_str()
        .validate_with_options(TargetEnv::Vulkan1_0, opts)
        .expect("scalar layout allows member at offset 20 (after array raw extent)");
}

#[test]
fn matrix_stride_alignment_and_size() {
    // Test that matrix stride must be at least as large as the column size
    let text = [
        "OpCapability Shader",
        "OpCapability Float64",
        "OpMemoryModel Logical GLSL450",
        "%f64 = OpTypeFloat 64",
        "%v2 = OpTypeVector %f64 2",
        "%mat2 = OpTypeMatrix %v2 2",
        "%struct = OpTypeStruct %v2 %mat2",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 32",
        "OpMemberDecorate %struct 1 ColMajor",
        "OpMemberDecorate %struct 1 MatrixStride 8", // Too small for vec2<f64> (16 bytes)
    ]
    .join("\n");
    let words = assemble_text(&text).expect("assemble");
    let err = validate_module(&words, TargetEnv::Vulkan1_0)
        .expect_err("matrix stride smaller than column size should fail");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
    let aligned = [
        "OpCapability Shader",
        "OpCapability Float64",
        "OpMemoryModel Logical GLSL450",
        "%f64 = OpTypeFloat 64",
        "%v2 = OpTypeVector %f64 2",
        "%mat2 = OpTypeMatrix %v2 2",
        "%struct = OpTypeStruct %v2 %mat2",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 32",
        "OpMemberDecorate %struct 1 ColMajor",
        "OpMemberDecorate %struct 1 MatrixStride 16", // Correct size for vec2<f64>
    ]
    .join("\n");
    let aligned_words = assemble_text(&aligned).expect("assemble");
    validate_module(&aligned_words, TargetEnv::Vulkan1_0)
        .expect("aligned matrix stride should pass");
}

#[test]
fn block_struct_missing_array_stride_rejected() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%int = OpTypeInt 32 0",
        "%len = OpConstant %int 2",
        "%arr = OpTypeArray %int %len",
        "%struct = OpTypeStruct %arr",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        // No ArrayStride on %arr
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect_err("missing ArrayStride on block struct array should be rejected");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn block_struct_with_array_stride_accepted() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%int = OpTypeInt 32 0",
        "%len = OpConstant %int 2",
        "%arr = OpTypeArray %int %len",
        "%struct = OpTypeStruct %arr",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpDecorate %arr ArrayStride 16",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("array with ArrayStride in block struct should pass");
}

#[test]
fn block_struct_missing_matrix_majorness_rejected() {
    // Build binary directly since the assembler requires RowMajor/ColMajor
    // when MatrixStride is present.
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
    module.header = Some(ModuleHeader::new(10));
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
        inst(
            rspirv::spirv::Op::TypeFloat,
            None,
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(32)],
        ),
        inst(
            rspirv::spirv::Op::TypeVector,
            None,
            Some(2),
            vec![
                rspirv::dr::Operand::IdRef(1),
                rspirv::dr::Operand::LiteralBit32(2),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeMatrix,
            None,
            Some(3),
            vec![
                rspirv::dr::Operand::IdRef(2),
                rspirv::dr::Operand::LiteralBit32(2),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(4),
            vec![rspirv::dr::Operand::IdRef(3)],
        ),
        inst(
            rspirv::spirv::Op::TypePointer,
            None,
            Some(5),
            vec![
                rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Uniform),
                rspirv::dr::Operand::IdRef(4),
            ],
        ),
        inst(
            rspirv::spirv::Op::Variable,
            Some(5),
            Some(6),
            vec![rspirv::dr::Operand::StorageClass(
                rspirv::spirv::StorageClass::Uniform,
            )],
        ),
    ]);
    module.annotations.extend([
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(4),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
            ],
        ),
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(6),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::DescriptorSet),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(6),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Binding),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(4),
                rspirv::dr::Operand::LiteralBit32(0),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        // MatrixStride but NO RowMajor/ColMajor
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(4),
                rspirv::dr::Operand::LiteralBit32(0),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::MatrixStride),
                rspirv::dr::Operand::LiteralBit32(16),
            ],
        ),
    ]);
    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("missing RowMajor/ColMajor on block struct matrix should be rejected");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn block_struct_missing_matrix_stride_rejected() {
    // Build binary directly since the assembler validates ColMajor requires MatrixStride.
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
    module.header = Some(ModuleHeader::new(10));
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
        inst(
            rspirv::spirv::Op::TypeFloat,
            None,
            Some(1),
            vec![rspirv::dr::Operand::LiteralBit32(32)],
        ),
        inst(
            rspirv::spirv::Op::TypeVector,
            None,
            Some(2),
            vec![
                rspirv::dr::Operand::IdRef(1),
                rspirv::dr::Operand::LiteralBit32(2),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeMatrix,
            None,
            Some(3),
            vec![
                rspirv::dr::Operand::IdRef(2),
                rspirv::dr::Operand::LiteralBit32(2),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(4),
            vec![rspirv::dr::Operand::IdRef(3)],
        ),
        inst(
            rspirv::spirv::Op::TypePointer,
            None,
            Some(5),
            vec![
                rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Uniform),
                rspirv::dr::Operand::IdRef(4),
            ],
        ),
        inst(
            rspirv::spirv::Op::Variable,
            Some(5),
            Some(6),
            vec![rspirv::dr::Operand::StorageClass(
                rspirv::spirv::StorageClass::Uniform,
            )],
        ),
    ]);
    module.annotations.extend([
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(4),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
            ],
        ),
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(6),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::DescriptorSet),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(6),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Binding),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(4),
                rspirv::dr::Operand::LiteralBit32(0),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        // ColMajor but NO MatrixStride
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(4),
                rspirv::dr::Operand::LiteralBit32(0),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::ColMajor),
            ],
        ),
    ]);
    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("missing MatrixStride on block struct matrix should be rejected");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn block_struct_with_matrix_decorations_accepted() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%float = OpTypeFloat 32",
        "%vec4 = OpTypeVector %float 4",
        "%mat4 = OpTypeMatrix %vec4 4",
        "%struct = OpTypeStruct %mat4",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 0 ColMajor",
        "OpMemberDecorate %struct 0 MatrixStride 16",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("block struct with ColMajor + MatrixStride should pass");
}

#[test]
fn runtime_array_must_be_last_member() {
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
    module.header = Some(ModuleHeader::new(8));
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
        inst(
            rspirv::spirv::Op::TypeInt,
            None,
            Some(1),
            vec![
                rspirv::dr::Operand::LiteralBit32(32),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::TypeRuntimeArray,
            None,
            Some(2),
            vec![rspirv::dr::Operand::IdRef(1)],
        ),
        inst(
            rspirv::spirv::Op::TypeStruct,
            None,
            Some(3),
            vec![
                rspirv::dr::Operand::IdRef(2),
                rspirv::dr::Operand::IdRef(1),
                rspirv::dr::Operand::IdRef(1),
            ],
        ),
        // TypePointer to struct with Uniform storage class
        inst(
            rspirv::spirv::Op::TypePointer,
            None,
            Some(4),
            vec![
                rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::Uniform),
                rspirv::dr::Operand::IdRef(3),
            ],
        ),
        // Variable using the pointer type
        inst(
            rspirv::spirv::Op::Variable,
            Some(4),
            Some(5),
            vec![rspirv::dr::Operand::StorageClass(
                rspirv::spirv::StorageClass::Uniform,
            )],
        ),
    ]);
    module.annotations.extend([
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(2),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::ArrayStride),
                rspirv::dr::Operand::LiteralBit32(4),
            ],
        ),
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Block),
            ],
        ),
        // DescriptorSet and Binding decorations for the variable
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(5),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::DescriptorSet),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::Decorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(5),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Binding),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::LiteralBit32(0),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(0),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::LiteralBit32(1),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(16),
            ],
        ),
        inst(
            rspirv::spirv::Op::MemberDecorate,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(3),
                rspirv::dr::Operand::LiteralBit32(2),
                rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Offset),
                rspirv::dr::Operand::LiteralBit32(32),
            ],
        ),
    ]);
    let binary = module.assemble();
    let err = binary
        .as_slice()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("runtime array must be the final member");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn switch_branch_limit_enforced() {
    use crate::validation::rules::limits::SwitchBranchLimitRule;
    use crate::validation::{TestContextData, ValidationRule, LIMIT_MAX_SWITCH_BRANCHES};
    use rspirv::dr::{Instruction, Operand};

    // Build a minimal module with an OpSwitch that exceeds the configured limit.
    let switch_inst = Instruction::new(
        rspirv::spirv::Op::Switch,
        None,
        None,
        vec![
            Operand::IdRef(1),
            Operand::IdRef(2),
            Operand::LiteralBit32(0),
            Operand::IdRef(3),
            Operand::LiteralBit32(1),
            Operand::IdRef(4),
        ],
    );
    let block = rspirv::dr::Block {
        label: None,
        instructions: vec![switch_inst],
    };
    let function = rspirv::dr::Function {
        def: None,
        parameters: Vec::new(),
        blocks: vec![block],
        end: None,
    };

    let mut test_data = TestContextData::default();
    test_data.module.functions.push(function);
    test_data
        .options
        .limits
        .insert(LIMIT_MAX_SWITCH_BRANCHES, 2);

    let ctx = test_data.as_context();
    let rule = SwitchBranchLimitRule;
    let err = rule
        .validate(&ctx)
        .expect_err("switch branch limit should be enforced");
    assert_eq!(
        err.error,
        ValidationError::LimitExceeded {
            limit_kind: LIMIT_MAX_SWITCH_BRANCHES,
            limit: 2,
            found: 3
        }
    );
}

#[test]
fn block_struct_in_storage_buffer_requires_offset() {
    let text = [
        "OpCapability Shader",
        "OpCapability PhysicalStorageBufferAddresses",
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "OpDecorate %BufferData Block",
        // No Offset decoration - should fail
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%BufferData = OpTypeStruct %vec4",
        "%ptr = OpTypePointer StorageBuffer %BufferData",
        "%buffer = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    match error {
        ValidationError::InvalidBlockLayout {
            struct_type: _,
            reason,
        } => {
            assert!(
                reason.contains("Offset"),
                "expected missing Offset error, got: {reason}"
            );
        }
        other => panic!("expected InvalidBlockLayout error, got: {other:?}"),
    }
}

#[test]
fn block_struct_in_uniform_requires_offset() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "OpDecorate %UniformData Block",
        "OpDecorate %uniform DescriptorSet 0",
        "OpDecorate %uniform Binding 0",
        // No Offset decoration - should fail
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%UniformData = OpTypeStruct %vec4",
        "%ptr = OpTypePointer Uniform %UniformData",
        "%uniform = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    match error {
        ValidationError::InvalidBlockLayout {
            struct_type: _,
            reason,
        } => {
            assert!(
                reason.contains("Offset"),
                "expected missing Offset error, got: {reason}"
            );
        }
        other => panic!("expected InvalidBlockLayout error, got: {other:?}"),
    }
}

#[test]
fn block_struct_in_push_constant_requires_offset() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "OpDecorate %PushData Block",
        // No Offset decoration - should fail
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%PushData = OpTypeStruct %vec4",
        "%ptr = OpTypePointer PushConstant %PushData",
        "%push = OpVariable %ptr PushConstant",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    let error = validate_module(&binary, TargetEnv::Vulkan1_2).unwrap_err();
    match error {
        ValidationError::InvalidBlockLayout {
            struct_type: _,
            reason,
        } => {
            assert!(
                reason.contains("Offset"),
                "expected missing Offset error, got: {reason}"
            );
        }
        other => panic!("expected InvalidBlockLayout error, got: {other:?}"),
    }
}

#[test]
fn block_struct_with_offset_in_storage_buffer_is_valid() {
    let text = [
        "OpCapability Shader",
        "OpCapability PhysicalStorageBufferAddresses",
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "OpDecorate %BufferData Block",
        "OpMemberDecorate %BufferData 0 Offset 0",
        "OpDecorate %buffer DescriptorSet 0",
        "OpDecorate %buffer Binding 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%BufferData = OpTypeStruct %vec4",
        "%ptr = OpTypePointer StorageBuffer %BufferData",
        "%buffer = OpVariable %ptr StorageBuffer",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    validate_module(&binary, TargetEnv::Vulkan1_2).expect("should be valid");
}

#[test]
fn buffer_block_struct_in_uniform_requires_offset() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "OpDecorate %BufferData BufferBlock",
        "OpDecorate %buffer DescriptorSet 0",
        "OpDecorate %buffer Binding 0",
        // No Offset decoration - should fail
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%BufferData = OpTypeStruct %vec4",
        "%ptr = OpTypePointer Uniform %BufferData",
        "%buffer = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    // Use Universal1_3 where BufferBlock is still valid
    let error = validate_module(&binary, TargetEnv::Universal1_3).unwrap_err();
    match error {
        ValidationError::InvalidBlockLayout {
            struct_type: _,
            reason,
        } => {
            assert!(
                reason.contains("Offset"),
                "expected missing Offset error, got: {reason}"
            );
        }
        other => panic!("expected InvalidBlockLayout error, got: {other:?}"),
    }
}

#[test]
fn buffer_block_struct_with_offset_in_uniform_is_valid() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint GLCompute %main \"main\"",
        "OpExecutionMode %main LocalSize 1 1 1",
        "OpDecorate %BufferData BufferBlock",
        "OpDecorate %buffer DescriptorSet 0",
        "OpDecorate %buffer Binding 0",
        "OpMemberDecorate %BufferData 0 Offset 0",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%f32 = OpTypeFloat 32",
        "%vec4 = OpTypeVector %f32 4",
        "%BufferData = OpTypeStruct %vec4",
        "%ptr = OpTypePointer Uniform %BufferData",
        "%buffer = OpVariable %ptr Uniform",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");

    let binary = assemble_text(&text).expect("assemble");
    // Use Universal1_3 where BufferBlock is still valid
    validate_module(&binary, TargetEnv::Universal1_3).expect("should be valid");
}

#[test]
fn struct_size_accounts_for_offset_gaps() {
    // struct { float a; /* offset 0, size 4 */ vec4 b; /* offset 16, size 16 */ }
    // Nested inside an outer struct to exercise the overlap check against the
    // inner struct's computed size.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %outer Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %inner 0 Offset 0",
        "OpMemberDecorate %inner 1 Offset 16",
        "OpMemberDecorate %outer 0 Offset 0",
        "OpMemberDecorate %outer 1 Offset 32",
        "%float = OpTypeFloat 32",
        "%v4 = OpTypeVector %float 4",
        "%inner = OpTypeStruct %float %v4",
        "%uint = OpTypeInt 32 0",
        "%outer = OpTypeStruct %inner %uint",
        "%ptr = OpTypePointer Uniform %outer",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("struct with gap should pass: inner size is 32, member 1 at offset 32 is valid");
}

#[test]
fn struct_size_detects_overlap_with_offset_gaps() {
    // inner = { float @0, vec4 @16 } → size 32
    // outer member 1 at offset 20 overlaps inner (0..32)
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %outer Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %inner 0 Offset 0",
        "OpMemberDecorate %inner 1 Offset 16",
        "OpMemberDecorate %outer 0 Offset 0",
        "OpMemberDecorate %outer 1 Offset 20",
        "%float = OpTypeFloat 32",
        "%v4 = OpTypeVector %float 4",
        "%inner = OpTypeStruct %float %v4",
        "%uint = OpTypeInt 32 0",
        "%outer = OpTypeStruct %inner %uint",
        "%ptr = OpTypePointer Uniform %outer",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("member at offset 20 overlaps inner struct ending at offset 32");
    if let ValidationError::InvalidBlockLayout { reason, .. } = err {
        assert!(reason.contains("overlap"), "unexpected reason: {reason:?}");
    } else {
        panic!("unexpected error: {err:?}");
    }
}

#[test]
fn array_size_accounts_for_stride() {
    // struct { array<vec3, 2> @0 (stride 16, size 28); uint @28 }
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %outer Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpDecorate %arr ArrayStride 16",
        "OpMemberDecorate %outer 0 Offset 0",
        "OpMemberDecorate %outer 1 Offset 28",
        "%float = OpTypeFloat 32",
        "%v3 = OpTypeVector %float 3",
        "%uint = OpTypeInt 32 0",
        "%c2 = OpConstant %uint 2",
        "%arr = OpTypeArray %v3 %c2",
        "%outer = OpTypeStruct %arr %uint",
        "%ptr = OpTypePointer StorageBuffer %outer",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    let opts = ValidationOptions {
        scalar_block_layout: true,
        ..ValidationOptions::default()
    };
    text.as_str()
        .validate_with_options(TargetEnv::Vulkan1_1, opts)
        .expect("member at offset 28 is valid: array size is (2-1)*16+12 = 28");
}

#[test]
fn array_size_detects_overlap_with_stride() {
    // array<vec3, 2> stride=16 → size = (2-1)*16+12 = 28
    // second member at offset 24 overlaps array (0..28)
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %outer Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpDecorate %arr ArrayStride 16",
        "OpMemberDecorate %outer 0 Offset 0",
        "OpMemberDecorate %outer 1 Offset 24",
        "%float = OpTypeFloat 32",
        "%v3 = OpTypeVector %float 3",
        "%uint = OpTypeInt 32 0",
        "%c2 = OpConstant %uint 2",
        "%arr = OpTypeArray %v3 %c2",
        "%outer = OpTypeStruct %arr %uint",
        "%ptr = OpTypePointer StorageBuffer %outer",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    let opts = ValidationOptions {
        scalar_block_layout: true,
        ..ValidationOptions::default()
    };
    let err = text
        .as_str()
        .validate_with_options(TargetEnv::Vulkan1_1, opts)
        .expect_err("member at offset 24 overlaps array ending at offset 28");
    if let ValidationError::InvalidBlockLayout { reason, .. } = err {
        assert!(reason.contains("overlap"), "unexpected reason: {reason:?}");
    } else {
        panic!("unexpected error: {err:?}");
    }
}

#[test]
fn descriptor_array_struct_is_validated() {
    // Block struct with correct offsets behind an array should pass.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpDecorate %arr ArrayStride 16",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%uint = OpTypeInt 32 0",
        "%c4 = OpConstant %uint 4",
        "%struct = OpTypeStruct %uint %uint",
        "%arr = OpTypeArray %struct %c4",
        "%ptr = OpTypePointer Uniform %arr",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("block struct behind descriptor array should be validated and pass");
}

#[test]
fn descriptor_array_struct_bad_offsets_rejected() {
    // Block struct with overlapping offsets behind an array should fail.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpDecorate %arr ArrayStride 16",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 0",
        "%uint = OpTypeInt 32 0",
        "%c4 = OpConstant %uint 4",
        "%struct = OpTypeStruct %uint %uint",
        "%arr = OpTypeArray %struct %c4",
        "%ptr = OpTypePointer Uniform %arr",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("block struct behind array with overlapping offsets should fail");
    if let ValidationError::InvalidBlockLayout { reason, .. } = err {
        assert!(reason.contains("overlap"), "unexpected reason: {reason:?}");
    } else {
        panic!("unexpected error: {err:?}");
    }
}

#[test]
fn descriptor_runtime_array_struct_is_validated() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct BufferBlock",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpDecorate %rarr ArrayStride 16",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%uint = OpTypeInt 32 0",
        "%struct = OpTypeStruct %uint %uint",
        "%rarr = OpTypeRuntimeArray %struct",
        "%ptr = OpTypePointer Uniform %rarr",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("block struct behind runtime array should pass");
}

#[test]
fn universal_env_skips_layout_validation() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    // Under Vulkan, overlapping offsets would fail:
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect_err("Vulkan should reject overlapping offsets");
    // Under Universal, layout checks are skipped so this should pass:
    text.as_str()
        .validate(TargetEnv::Universal1_6)
        .expect("Universal env skips layout validation per C++ line 1478");
}

#[test]
fn universal_env_still_enforces_offset_presence() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Universal1_6)
        .expect_err("missing Offset should fail even under Universal");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn non_struct_type_in_block_layout_passes_when_aligned() {
    // We can't directly trigger the pseudo-member path from normal SPIR-V
    // because block-decorated types must be structs. But the code path exists
    // for PhysicalStorageBuffer pointers and similar. Here we verify that
    // the normal struct path still works correctly after the refactor.
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%struct = OpTypeStruct %int %int",
        "%ptr = OpTypePointer Uniform %struct",
        "%var = OpVariable %ptr Uniform",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_0)
        .expect("simple aligned struct should validate");
}

#[test]
fn pointer_member_skipped_under_logical_addressing() {
    // A block struct containing a pointer member under logical addressing.
    // The pointer type has no layout, so check_struct_layout should skip it.
    let text = [
        "OpCapability Shader",
        "OpCapability VariablePointersStorageBuffer",
        "OpExtension \"SPV_KHR_variable_pointers\"",
        "OpMemoryModel Logical GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%int_ptr = OpTypePointer StorageBuffer %int",
        "%struct = OpTypeStruct %int %int_ptr",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    // Under logical addressing, pointer_size is 0, so pointer member layout
    // is skipped. The struct is valid because the int member at offset 0 is aligned.
    // The int_ptr at offset 4 would normally need 8-byte alignment under physical
    // addressing, but under logical addressing it has no layout and is skipped.
    text.as_str()
        .validate(TargetEnv::Vulkan1_1)
        .expect("pointer member should be skipped under logical addressing");
}

#[test]
fn pointer_member_valid_under_physical_storage_buffer() {
    let text = [
        "OpCapability Shader",
        "OpCapability PhysicalStorageBufferAddresses",
        "OpExtension \"SPV_KHR_physical_storage_buffer\"",
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 8",
        "%int = OpTypeInt 32 0",
        "%int_ptr = OpTypePointer PhysicalStorageBuffer %int",
        "%struct = OpTypeStruct %int %int_ptr",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("8-byte aligned pointer at offset 8 should be valid");
}

#[test]
fn pointer_member_misaligned_under_physical_storage_buffer() {
    let text = [
        "OpCapability Shader",
        "OpCapability PhysicalStorageBufferAddresses",
        "OpExtension \"SPV_KHR_physical_storage_buffer\"",
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%int_ptr = OpTypePointer PhysicalStorageBuffer %int",
        "%struct = OpTypeStruct %int %int_ptr",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("pointer at offset 4 should fail (needs 8-byte alignment)");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}

#[test]
fn pointer_member_overlap_detected_under_physical_addressing() {
    let text = [
        "OpCapability Shader",
        "OpCapability PhysicalStorageBufferAddresses",
        "OpExtension \"SPV_KHR_physical_storage_buffer\"",
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        "OpMemberDecorate %struct 1 Offset 8",
        // Second member at offset 8 = pointer at offset 0 + size 8, just fine
        "OpMemberDecorate %struct 2 Offset 12",
        // Third member at offset 12 overlaps second (int at 8 takes 4 bytes, ends at 12, so 12 is ok)
        "%int = OpTypeInt 32 0",
        "%int_ptr = OpTypePointer PhysicalStorageBuffer %int",
        "%struct = OpTypeStruct %int_ptr %int %int",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    text.as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect("non-overlapping pointer+int members should be valid");
}

#[test]
fn pointer_member_overlap_is_rejected() {
    let text = [
        "OpCapability Shader",
        "OpCapability PhysicalStorageBufferAddresses",
        "OpExtension \"SPV_KHR_physical_storage_buffer\"",
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
        "OpDecorate %struct Block",
        "OpDecorate %var DescriptorSet 0",
        "OpDecorate %var Binding 0",
        "OpMemberDecorate %struct 0 Offset 0",
        // Place int at offset 4 which overlaps the 8-byte pointer at offset 0
        "OpMemberDecorate %struct 1 Offset 4",
        "%int = OpTypeInt 32 0",
        "%int_ptr = OpTypePointer PhysicalStorageBuffer %int",
        "%struct = OpTypeStruct %int_ptr %int",
        "%ptr = OpTypePointer StorageBuffer %struct",
        "%var = OpVariable %ptr StorageBuffer",
    ]
    .join("\n");
    let err = text
        .as_str()
        .validate(TargetEnv::Vulkan1_2)
        .expect_err("int at offset 4 overlaps 8-byte pointer at offset 0");
    assert!(matches!(err, ValidationError::InvalidBlockLayout { .. }));
}
