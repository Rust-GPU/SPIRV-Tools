use super::*;

#[test]
fn block_requires_terminator() {
    // A block must end with a terminator instruction.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        4,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4 (no terminator follows)
        4,
        op(1, 56), // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    if let ValidationError::Parse(message) = error {
        assert!(
            message.contains("block without terminator"),
            "unexpected parse error: {message}"
        );
    } else {
        panic!("expected parse error, got {error:?}");
    }
}

#[test]
fn block_cannot_have_instructions_after_terminator() {
    // A terminator must end the block; trailing instructions are invalid.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(1, 253), // OpReturn
        op(1, 0),   // OpNop (illegal after terminator)
        op(1, 56),  // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    if let ValidationError::Parse(message) = error {
        assert!(
            message.contains("instruction") && message.contains("not inside block"),
            "unexpected parse error: {message}"
        );
    } else {
        panic!("expected parse error, got {error:?}");
    }
}

#[test]
fn branch_requires_existing_target() {
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        6,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4
        4,
        op(2, 249), // OpBranch %5 (undefined target)
        5,
        op(1, 56), // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::MissingBlockTarget {
            function: Id::try_from(3).unwrap(),
            target: Id::try_from(5).unwrap()
        }
    );
}

#[test]
fn switch_requires_existing_targets() {
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        8,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(4, 21), // OpTypeInt %2 32 0
        2,
        32,
        0,
        op(3, 33), // OpTypeFunction %3 %1
        3,
        1,
        op(5, 54), // OpFunction %4 None %3
        1,
        4,
        0,
        3,
        op(2, 248), // OpLabel %5
        5,
        op(5, 128), // OpSwitch %6 %7 0 %7 (both %6 and %7 undefined)
        6,
        7,
        0,
        7,
        op(1, 56), // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    if let ValidationError::Parse(message) = error {
        assert!(
            message.contains("block") && message.contains("terminator"),
            "unexpected parse error: {message}"
        );
    } else {
        panic!("expected parse error, got {error:?}");
    }
}

#[test]
fn entry_block_cannot_have_predecessors() {
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        7,
        0,
        op(2, 17), // OpCapability Shader
        rspirv::spirv::Capability::Shader as u32,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(3, 33), // OpTypeFunction %2 %1
        2,
        1,
        op(5, 54), // OpFunction %3 None %2
        1,
        3,
        0,
        2,
        op(2, 248), // OpLabel %4 (entry)
        4,
        op(2, 249), // OpBranch %5
        5,
        op(2, 248), // OpLabel %5 (second block)
        5,
        op(2, 249), // OpBranch %4 (back to entry)
        4,
        op(1, 56), // OpFunctionEnd
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::EntryBlockHasPredecessor {
            function: Id::try_from(3).unwrap(),
            entry: Id::try_from(4).unwrap()
        }
    );
}

#[test]
fn unreachable_block_is_allowed() {
    // SPIR-V spec allows unreachable blocks - the C++ validator skips them
    // during structured control flow validation.
    use rspirv::{binary::Assemble, dr::Builder};
    let mut builder = Builder::new();
    builder.set_version(1, 6);
    builder.capability(rspirv::spirv::Capability::Shader);
    builder.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::GLSL450,
    );
    let void = builder.type_void();
    let fn_ty = builder.type_function(void, std::iter::empty::<u32>());
    let _function_id = builder
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let _entry = builder.begin_block(None).unwrap();
    builder.ret().unwrap();
    let _unreachable = builder.begin_block(None).unwrap();
    builder.ret().unwrap();
    builder.end_function().unwrap();
    let words = builder.module().assemble();
    // Should now succeed since unreachable blocks are allowed
    words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("unreachable blocks should be allowed");
}

#[test]
fn capability_must_appear_before_types() {
    // The assembler reorders sections, so preserve the out-of-order capability via binary.
    let binary = vec![
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        op(3, 14), // OpMemoryModel Logical GLSL450
        0,
        1,
        op(2, 19), // OpTypeVoid %1
        1,
        op(2, 17), // OpCapability Shader (out of order)
        rspirv::spirv::Capability::Shader as u32,
    ];
    let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
    assert_eq!(
        error,
        ValidationError::LayoutOutOfOrder {
            opcode: rspirv::spirv::Op::Capability
        }
    );
}

#[test]
fn merge_must_immediately_precede_terminator() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%ptr = OpTypePointer Function %bool",
        "%true = OpConstantTrue %bool",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr Function",
        "OpSelectionMerge %merge None",
        "%tmp = OpLoad %bool %var",
        "OpBranchConditional %true %merge %merge",
        "%merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("merge must sit immediately before terminator");
    assert_eq!(
        err,
        ValidationError::MergeInstructionNotBeforeTerminator {
            function: Id::try_from(6).unwrap(),
            block: Id::try_from(7).unwrap(),
        }
    );
}

#[test]
fn selection_merge_must_precede_switch() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let entry = b.begin_block(None).unwrap();
    let merge_label = b.id();
    let zero = b.constant_bit32(int, 0);
    b.selection_merge(merge_label, rspirv::spirv::SelectionControl::NONE)
        .unwrap();
    b.nop().unwrap(); // Placement violation between merge and switch.
    b.switch(
        zero,
        merge_label,
        vec![(rspirv::dr::Operand::LiteralBit32(0), merge_label)],
    )
    .unwrap();
    b.begin_block(Some(merge_label)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("selection merge must sit immediately before switch");
    assert_eq!(
        err,
        ValidationError::MergeInstructionNotBeforeTerminator {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(entry).unwrap(),
        }
    );
}

#[test]
fn loop_merge_must_precede_branch() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let merge = b.id();
    let cont = b.id();
    b.loop_merge(
        merge,
        cont,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    b.nop().unwrap(); // placement violation
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop merge must sit immediately before terminator");
    assert_eq!(
        err,
        ValidationError::MergeInstructionNotBeforeTerminator {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
        }
    );
}

#[test]
fn loop_merge_cannot_terminate_with_switch() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let merge = b.id();
    let cont = b.id();
    b.loop_merge(
        merge,
        cont,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    let zero = b.constant_bit32(int, 0);
    b.switch(
        zero,
        merge,
        vec![(rspirv::dr::Operand::LiteralBit32(0), cont)],
    )
    .unwrap();
    b.begin_block(Some(cont)).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop merge header cannot end in OpSwitch");
    assert_eq!(
        err,
        ValidationError::InvalidMergeTerminator {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            terminator: rspirv::spirv::Op::Switch
        }
    );
}

#[test]
fn loop_merge_cannot_be_followed_by_return() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let merge = b.id();
    let cont = b.id();
    b.loop_merge(
        merge,
        cont,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    // Invalid terminator after loop merge.
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop header cannot end in OpReturn after loop merge");
    assert_eq!(
        err,
        ValidationError::InvalidMergeTerminator {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            terminator: rspirv::spirv::Op::Return
        }
    );
}

#[test]
fn loop_merge_cannot_be_followed_by_unreachable() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let merge = b.id();
    let cont = b.id();
    b.loop_merge(
        merge,
        cont,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    // Invalid terminator after loop merge.
    b.unreachable().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop header cannot end in OpUnreachable after loop merge");
    assert_eq!(
        err,
        ValidationError::InvalidMergeTerminator {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            terminator: rspirv::spirv::Op::Unreachable
        }
    );
}

#[test]
fn selection_merge_target_must_be_dominated_by_header() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let bool = b.type_bool();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let _entry = b.begin_block(None).unwrap();
    let header = b.id();
    let merge = b.id();
    let cond = b.constant_true(bool);
    b.selection_merge(merge, rspirv::spirv::SelectionControl::NONE)
        .unwrap();
    b.branch_conditional(cond, header, merge, std::iter::empty())
        .unwrap();
    b.begin_block(Some(header)).unwrap();
    b.selection_merge(merge, rspirv::spirv::SelectionControl::NONE)
        .unwrap();
    b.branch_conditional(cond, merge, merge, std::iter::empty())
        .unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("merge target should be dominated by header");
    assert_eq!(
        err,
        ValidationError::MergeTargetNotDominated {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            kind: MergeTargetKind::Merge,
            target: Id::try_from(merge).unwrap()
        }
    );
}

#[test]
fn loop_merge_target_must_be_dominated_by_header() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let bool = b.type_bool();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let _entry = b.begin_block(None).unwrap();
    let header = b.id();
    let merge = b.id();
    let cont = b.id();
    let cond = b.constant_true(bool);
    b.selection_merge(merge, rspirv::spirv::SelectionControl::NONE)
        .unwrap();
    b.branch_conditional(cond, header, merge, std::iter::empty())
        .unwrap();
    b.begin_block(Some(header)).unwrap();
    b.loop_merge(
        merge,
        cont,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    b.branch_conditional(cond, merge, cont, std::iter::empty())
        .unwrap();
    b.begin_block(Some(cont)).unwrap();
    b.branch(header).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop merge target should be dominated by header");
    assert_eq!(
        err,
        ValidationError::MergeTargetNotDominated {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            kind: MergeTargetKind::Merge,
            target: Id::try_from(merge).unwrap()
        }
    );
}

#[test]
fn loop_continue_target_must_be_dominated_by_header() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let bool = b.type_bool();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let _entry = b.begin_block(None).unwrap();
    let header = b.id();
    let merge = b.id();
    let cont = b.id();
    let cond = b.constant_true(bool);
    b.selection_merge(cont, rspirv::spirv::SelectionControl::NONE)
        .unwrap();
    b.branch_conditional(cond, header, cont, std::iter::empty())
        .unwrap();
    b.begin_block(Some(header)).unwrap();
    b.loop_merge(
        merge,
        cont,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    b.branch_conditional(cond, merge, cont, std::iter::empty())
        .unwrap();
    b.begin_block(Some(cont)).unwrap();
    b.branch(header).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop continue target should be dominated by header");
    assert_eq!(
        err,
        ValidationError::MergeTargetNotDominated {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            kind: MergeTargetKind::Continue,
            target: Id::try_from(cont).unwrap()
        }
    );
}

#[test]
fn loop_merge_cannot_be_followed_by_return_value() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let int = b.type_int(32, 1);
    let fn_ty = b.type_function(int, std::iter::empty::<u32>());
    let main = b
        .begin_function(int, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let merge = b.id();
    let cont = b.id();
    b.loop_merge(
        merge,
        cont,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    // Invalid terminator after loop merge.
    let zero = b.constant_bit32(int, 0);
    b.ret_value(zero).unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop header cannot end in OpReturnValue after loop merge");
    assert_eq!(
        err,
        ValidationError::InvalidMergeTerminator {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            terminator: rspirv::spirv::Op::ReturnValue
        }
    );
}

#[test]
fn loop_merge_cannot_be_followed_by_kill() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let merge = b.id();
    let cont = b.id();
    b.loop_merge(
        merge,
        cont,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    // Invalid terminator after loop merge.
    b.kill().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop header cannot end in OpKill after loop merge");
    assert_eq!(
        err,
        ValidationError::InvalidMergeTerminator {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            terminator: rspirv::spirv::Op::Kill
        }
    );
}

#[test]
fn selection_merge_target_must_exist() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let bool_ty = b.type_bool();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let missing = b.constant_true(bool_ty); // not a block id
    let merge_label = b.id();
    b.selection_merge(missing, rspirv::spirv::SelectionControl::NONE)
        .unwrap();
    b.branch_conditional(missing, merge_label, merge_label, None)
        .unwrap();
    b.begin_block(Some(merge_label)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("selection merge target must exist and be a block");
    assert_eq!(
        err,
        ValidationError::MergeTargetMissing {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            kind: MergeTargetKind::Merge,
            target: Id::try_from(missing).unwrap(),
        }
    );
}

#[test]
fn loop_merge_targets_must_exist() {
    // Create a properly structured module where the merge target doesn't exist (is a non-block ID).
    // %entry -> %header with LoopMerge pointing to non-existent block
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%missing_merge = OpConstantTrue %bool", // Use a constant as merge target (invalid)
        "%missing_continue = OpConstantFalse %bool", // Use a constant as continue target (invalid)
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpBranch %header",
        "%header = OpLabel",
        // Merge and continue targets point to constants, not blocks - this is invalid
        "OpLoopMerge %missing_merge %missing_continue None",
        "OpBranch %body",
        "%body = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop merge targets must be blocks in the same function");
    // ID layout: void=1, fn=2, bool=3, missing_merge=4, missing_continue=5, main=6, entry=7, header=8, body=9
    assert_eq!(
        err,
        ValidationError::MergeTargetMissing {
            function: Id::try_from(6).unwrap(),
            block: Id::try_from(8).unwrap(),
            kind: MergeTargetKind::Merge,
            target: Id::try_from(4).unwrap(),
        }
    );
}

#[test]
fn loop_continue_target_must_exist() {
    use rspirv::{binary::Assemble, dr::Builder};
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let bool_ty = b.type_bool();
    let fn_ty = b.type_function(void, std::iter::empty::<u32>());
    let main = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, fn_ty)
        .unwrap();
    let header = b.begin_block(None).unwrap();
    let merge = b.id();
    let missing_continue = b.constant_false(bool_ty);
    let body = b.id();
    b.loop_merge(
        merge,
        missing_continue,
        rspirv::spirv::LoopControl::NONE,
        std::iter::empty::<rspirv::dr::Operand>(),
    )
    .unwrap();
    b.branch(body).unwrap();
    b.begin_block(Some(body)).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let words = b.module().assemble();
    let err = words
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop continue target must be a block in the same function");
    assert_eq!(
        err,
        ValidationError::MergeTargetMissing {
            function: Id::try_from(main).unwrap(),
            block: Id::try_from(header).unwrap(),
            kind: MergeTargetKind::Continue,
            target: Id::try_from(missing_continue).unwrap(),
        }
    );
}

#[test]
fn loop_merge_targets_must_be_distinct_and_exist() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        // merge and continue use the same target, which is invalid.
        "OpLoopMerge %merge %merge None",
        "OpBranch %merge",
        "%merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop merge cannot reuse the merge block as continue target");
    assert_eq!(
        err,
        ValidationError::ContinueTargetMatchesMerge {
            function: Id::try_from(3).unwrap(),
            block: Id::try_from(4).unwrap(),
            target: Id::try_from(5).unwrap()
        }
    );
}

#[test]
fn selection_merge_target_cannot_be_header() {
    // Create a properly structured module where the header block references itself as merge target.
    // %entry -> %header (where header has SelectionMerge %header which is invalid)
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%true = OpConstantTrue %bool",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpBranch %header",
        "%header = OpLabel",
        // Merge target aliases the header block - this is invalid
        "OpSelectionMerge %header None",
        "OpBranchConditional %true %then %else",
        "%then = OpLabel",
        "OpReturn",
        "%else = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("selection merge target cannot equal its own block");
    // ID layout: void=1, fn=2, bool=3, true=4, main=5, entry=6, header=7, then=8, else=9
    assert_eq!(
        err,
        ValidationError::MergeTargetIsBlock {
            function: Id::try_from(5).unwrap(),
            block: Id::try_from(7).unwrap(),
            kind: MergeTargetKind::Merge,
            target: Id::try_from(7).unwrap()
        }
    );
}

#[test]
fn loop_merge_targets_cannot_be_header() {
    // Create a properly structured module where the loop header references itself as merge target.
    // %entry -> %header (where header has LoopMerge %header %header which is invalid)
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpBranch %header",
        "%header = OpLabel",
        // Merge and continue targets both alias the header block - this is invalid
        "OpLoopMerge %header %header None",
        "OpBranch %body",
        "%body = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("loop merge cannot target its own header");
    // ID layout: void=1, fn=2, main=3, entry=4, header=5, body=6
    assert_eq!(
        err,
        ValidationError::MergeTargetIsBlock {
            function: Id::try_from(3).unwrap(),
            block: Id::try_from(5).unwrap(),
            kind: MergeTargetKind::Merge,
            target: Id::try_from(5).unwrap()
        }
    );
}

#[test]
fn value_use_must_be_dominated_by_definition() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%int = OpTypeInt 32 1",
        "%fn = OpTypeFunction %int",
        "%true = OpConstantTrue %bool",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        "OpBranchConditional %true %then %else",
        "%then = OpLabel",
        "OpBranch %merge",
        "%else = OpLabel",
        "%v = OpIAdd %int %zero %one",
        "OpBranch %merge",
        "%merge = OpLabel",
        "OpReturnValue %v",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("value defined in else does not dominate merge");
    assert!(matches!(err, ValidationError::ValueNotDominated { .. }));
}

#[test]
fn phi_incoming_must_be_dominated_along_edge() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%int = OpTypeInt 32 1",
        "%fn = OpTypeFunction %int",
        "%true = OpConstantTrue %bool",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        "OpBranchConditional %true %merge %then",
        "%then = OpLabel",
        "%v = OpIAdd %int %zero %one",
        "OpBranch %merge",
        "%merge = OpLabel",
        "%phi = OpPhi %int %v %entry %v %then",
        "OpReturnValue %phi",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("phi incoming must be dominated along its predecessor edge");
    match err {
        ValidationError::PhiIncomingNotDominated { .. } => {}
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn phi_must_list_all_predecessors() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%int = OpTypeInt 32 1",
        "%fn = OpTypeFunction %int",
        "%true = OpConstantTrue %bool",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        "OpBranchConditional %true %merge %side",
        "%side = OpLabel",
        "%v = OpIAdd %int %zero %one",
        "OpBranch %merge",
        "%merge = OpLabel",
        // Missing incoming pair for predecessor %entry.
        "%phi = OpPhi %int %v %side",
        "OpReturnValue %phi",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("phi must list all predecessors");
    assert_eq!(
        err,
        ValidationError::PhiPredecessorCountMismatch {
            function: Id::try_from(8).unwrap(),
            block: Id::try_from(10).unwrap(),
            expected: 2,
            found: 1,
        }
    );
}

#[test]
fn phi_incoming_types_must_match_result_type() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%int = OpTypeInt 32 1",
        "%uint = OpTypeInt 32 0",
        "%fn = OpTypeFunction %int",
        "%true = OpConstantTrue %bool",
        "%zero_i = OpConstant %int 0",
        "%zero_u = OpConstant %uint 0",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        "OpBranchConditional %true %merge %side",
        "%side = OpLabel",
        "OpBranch %merge",
        "%merge = OpLabel",
        // Incoming value type (%uint) does not match phi result type (%int).
        "%phi = OpPhi %int %zero_u %side %zero_i %entry",
        "OpReturnValue %phi",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("phi incoming types must match result type");
    assert_eq!(
        err,
        ValidationError::PhiIncomingTypeMismatch {
            function: Id::try_from(9).unwrap(),
            block: Id::try_from(11).unwrap(),
            incoming: Id::try_from(8).unwrap(),
            expected: TypeId::try_from(3).unwrap(),
            found: TypeId::try_from(4).unwrap()
        }
    );
}

#[test]
fn phi_cannot_have_extra_predecessors() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%int = OpTypeInt 32 1",
        "%fn = OpTypeFunction %int",
        "%true = OpConstantTrue %bool",
        "%zero = OpConstant %int 0",
        "%one = OpConstant %int 1",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        "OpBranchConditional %true %merge %side",
        "%side = OpLabel",
        "%v = OpIAdd %int %zero %one",
        "OpBranch %merge",
        "%merge = OpLabel",
        // Too many incoming predecessor/value pairs (3 pairs, but only 2 predecessors).
        "%phi = OpPhi %int %v %side %zero %entry %one %entry",
        "OpReturnValue %phi",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("phi cannot list more incoming predecessors than the block has");
    // Modular rule detects predecessor count mismatch (3 incoming vs 2 predecessors)
    assert!(matches!(
        err,
        ValidationError::PhiPredecessorCountMismatch { .. }
            | ValidationError::PhiDuplicatePredecessor { .. }
    ));
}

#[test]
fn unreachable_definition_cannot_be_used_in_reachable_block() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%int = OpTypeInt 32 1",
        "%one = OpConstant %int 1",
        "%fn = OpTypeFunction %int",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        // Use a value defined only in an unreachable block.
        "OpReturnValue %undef",
        "%unreachable = OpLabel",
        "%undef = OpIAdd %int %one %one",
        "OpReturnValue %undef",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("definitions in unreachable blocks cannot be used in reachable code");
    // Value from unreachable block doesn't dominate the use
    assert!(matches!(err, ValidationError::ValueNotDominated { .. }));
}

#[test]
fn value_must_dominate_non_phi_uses() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%bool = OpTypeBool",
        "%int = OpTypeInt 32 1",
        "%fn = OpTypeFunction %int",
        "%true = OpConstantTrue %bool",
        "%zero = OpConstant %int 0",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        "OpSelectionMerge %merge None",
        // Branch to merge and to then; merge has two predecessors.
        "OpBranchConditional %true %then %merge",
        "%then = OpLabel",
        "%v = OpIAdd %int %zero %zero",
        "OpBranch %merge",
        "%merge = OpLabel",
        // Use %v, which is not available along the direct %entry -> %merge edge.
        "%sum = OpIAdd %int %v %zero",
        "OpReturnValue %sum",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("values must dominate all non-phi uses");
    assert_eq!(
        err,
        ValidationError::ValueNotDominated {
            function: Id::try_from(7).unwrap(),
            block: Id::try_from(9).unwrap(),
            value: Id::try_from(11).unwrap()
        }
    );
}

#[test]
fn operand_id_must_be_defined_globally() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%ptr = OpTypePointer Function %missing",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("global operands must reference defined ids");
    assert!(matches!(
        err,
        ValidationError::UndefinedId { function: None, .. }
    ));
}

#[test]
fn operand_id_must_be_defined_in_function_scope() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 1",
        "%fn = OpTypeFunction %int",
        "%main = OpFunction %int None %fn",
        "%entry = OpLabel",
        "OpBranch %merge",
        "%merge = OpLabel",
        "%phi = OpPhi %int %undef %entry",
        "OpReturnValue %phi",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("operands inside functions must reference defined ids");
    assert!(matches!(
        err,
        ValidationError::UndefinedId {
            function: Some(_),
            ..
        }
    ));
}

#[test]
fn result_type_must_be_type_opcode() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%int = OpTypeInt 32 1",
        "%fn = OpTypeFunction %void",
        "%one = OpConstant %int 1",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %one Function",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("result types must reference type opcodes");
    assert!(matches!(
        err,
        ValidationError::ResultTypeNotType {
            instruction: rspirv::spirv::Op::Variable,
            ..
        }
    ));
}

#[test]
fn branch_conditional_to_two_returns_is_valid() {
    // A BranchConditional to two blocks that both return is valid without a merge
    // because there's no reconvergence needed (both branches exit the function).
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%true = OpConstantTrue %bool",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpBranchConditional %true %then %else",
        "%then = OpLabel",
        "OpReturn",
        "%else = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect("BranchConditional to two returns should be valid");
}

#[test]
fn switch_requires_selection_merge() {
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version 1.0
        0,           // generator
        10,          // bound (ids up to 9)
        0,           // schema
        op(2, rspirv::spirv::Op::Capability as u16),
        rspirv::spirv::Capability::Shader as u32,
        op(3, rspirv::spirv::Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        rspirv::spirv::MemoryModel::GLSL450 as u32,
        op(2, rspirv::spirv::Op::TypeVoid as u16),
        1, // %void
        op(3, rspirv::spirv::Op::TypeFunction as u16),
        2, // %fn
        1, // return type
        op(4, rspirv::spirv::Op::TypeInt as u16),
        3, // %i32
        32,
        0,
        op(4, rspirv::spirv::Op::Constant as u16),
        3, // type
        4, // %zero
        0, // literal
        op(5, rspirv::spirv::Op::Function as u16),
        1, // return type
        5, // %main
        rspirv::spirv::FunctionControl::NONE.bits(),
        2, // function type
        op(2, rspirv::spirv::Op::Label as u16),
        6, // %entry
        op(5, rspirv::spirv::Op::Switch as u16),
        4, // selector %zero
        8, // default target
        0, // literal
        7, // case target
        op(2, rspirv::spirv::Op::Label as u16),
        7, // %case
        op(2, rspirv::spirv::Op::Branch as u16),
        9, // %merge
        op(2, rspirv::spirv::Op::Label as u16),
        8, // %default
        op(2, rspirv::spirv::Op::Branch as u16),
        9, // %merge
        op(2, rspirv::spirv::Op::Label as u16),
        9, // %merge
        op(1, rspirv::spirv::Op::Return as u16),
        op(1, rspirv::spirv::Op::FunctionEnd as u16),
    ];
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("structured switch must have selection merge");
    assert_eq!(
        err,
        ValidationError::MissingSelectionMerge {
            function: Id::try_from(5).unwrap(),
            block: Id::try_from(6).unwrap(),
            terminator: rspirv::spirv::Op::Switch
        }
    );
}

#[test]
fn phi_must_precede_non_phi_in_block() {
    let binary = vec![
        0x0723_0203, // magic
        0x0001_0000, // version 1.0
        0,           // generator
        11,          // bound (ids up to 10)
        0,           // schema
        op(2, rspirv::spirv::Op::Capability as u16),
        rspirv::spirv::Capability::Shader as u32,
        op(3, rspirv::spirv::Op::MemoryModel as u16),
        rspirv::spirv::AddressingModel::Logical as u32,
        rspirv::spirv::MemoryModel::GLSL450 as u32,
        op(2, rspirv::spirv::Op::TypeVoid as u16),
        1, // %void
        op(4, rspirv::spirv::Op::TypeInt as u16),
        2, // %i32
        32,
        0,
        op(3, rspirv::spirv::Op::TypeFunction as u16),
        3, // %fn
        1, // return type
        op(4, rspirv::spirv::Op::Constant as u16),
        2, // type
        4, // %c
        0, // literal
        op(5, rspirv::spirv::Op::Function as u16),
        1, // return type
        5, // %main
        rspirv::spirv::FunctionControl::NONE.bits(),
        3, // function type
        op(2, rspirv::spirv::Op::Label as u16),
        6, // %entry
        op(2, rspirv::spirv::Op::Branch as u16),
        8, // %merge
        op(2, rspirv::spirv::Op::Label as u16),
        7, // %side
        op(2, rspirv::spirv::Op::Branch as u16),
        8, // %merge
        op(2, rspirv::spirv::Op::Label as u16),
        8, // %merge
        op(3, rspirv::spirv::Op::Undef as u16),
        2, // type %i32
        9, // %tmp
        // OpPhi %i32 %c %entry %c %side  (word count 7)
        op(7, rspirv::spirv::Op::Phi as u16),
        2,  // type
        10, // result id %phi
        4,  // value %c
        6,  // incoming block %entry
        4,  // value %c
        7,  // incoming block %side
        op(1, rspirv::spirv::Op::Return as u16),
        op(1, rspirv::spirv::Op::FunctionEnd as u16),
    ];
    let err = binary
        .as_slice()
        .validate(TargetEnv::Universal1_6)
        .expect_err("phi must be grouped before non-phi instructions");
    assert_eq!(
        err,
        ValidationError::PhiAfterNonPhi {
            function: Id::try_from(5).unwrap(),
            block: Id::try_from(8).unwrap(),
        }
    );
}

#[test]
fn access_chain_index_limit_enforced() {
    use crate::validation::{ValidationOptions, LIMIT_MAX_ACCESS_CHAIN_INDEXES};
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%u32 = OpTypeInt 32 0",
        "%two = OpConstant %u32 2",
        "%inner = OpTypeArray %u32 %two",
        "%outer = OpTypeArray %inner %two",
        "%ptr_outer = OpTypePointer Function %outer",
        "%elem_ptr = OpTypePointer Function %u32",
        "%zero = OpConstant %u32 0",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%var = OpVariable %ptr_outer Function",
        "%ac = OpAccessChain %elem_ptr %var %zero %zero",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let mut options = ValidationOptions::default();
    options.limits.insert(LIMIT_MAX_ACCESS_CHAIN_INDEXES, 1);
    let err = binary
        .as_slice()
        .validate_with_options(TargetEnv::Universal1_6, options)
        .expect_err("access chain index limit should be enforced");
    assert_eq!(
        err,
        ValidationError::LimitExceeded {
            limit_kind: LIMIT_MAX_ACCESS_CHAIN_INDEXES,
            limit: 1,
            found: 2
        }
    );
}
