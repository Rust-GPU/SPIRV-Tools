use rspirv::dr::Instruction;
use rspirv::spirv::Op;
use spirv_tools_opt::translate::{translate_arith_with_types_dominated, TranslateError};

fn make_const(id: u32, ty: u32, value: u32) -> Instruction {
    Instruction::new(
        Op::Constant,
        Some(ty),
        Some(id),
        vec![rspirv::dr::Operand::LiteralBit32(value)],
    )
}

#[test]
fn dominated_translation_rejects_undefined_operand() {
    // Add uses ids 2 and 3 before they are defined.
    let type_int = Instruction::new(
        Op::TypeInt,
        None,
        Some(10),
        vec![
            rspirv::dr::Operand::LiteralBit32(32),
            rspirv::dr::Operand::LiteralBit32(0),
        ],
    );
    let add = Instruction::new(
        Op::IAdd,
        Some(10),
        Some(1),
        vec![rspirv::dr::Operand::IdRef(2), rspirv::dr::Operand::IdRef(3)],
    );
    let c2 = make_const(2, 10, 4);
    let c3 = make_const(3, 10, 5);
    let stream = vec![type_int, add, c2, c3];

    let res = translate_arith_with_types_dominated(&stream, &Default::default());
    assert!(
        matches!(
            res,
            Err(TranslateError::UndominatedOperand {
                id: 2,
                opcode: Op::IAdd
            })
        ),
        "translation should reject operands that are not yet defined"
    );
}

#[test]
fn dominated_translation_allows_external_operands() {
    // Operand ids 99/100 are not produced in the stream, so they are treated as
    // pre-existing definitions that already dominate.
    let type_int = Instruction::new(
        Op::TypeInt,
        None,
        Some(10),
        vec![
            rspirv::dr::Operand::LiteralBit32(32),
            rspirv::dr::Operand::LiteralBit32(0),
        ],
    );
    let add = Instruction::new(
        Op::IAdd,
        Some(10),
        Some(1),
        vec![
            rspirv::dr::Operand::IdRef(99),
            rspirv::dr::Operand::IdRef(100),
        ],
    );
    let c2 = make_const(2, 10, 4);
    let stream = vec![type_int, add, c2];

    let res = translate_arith_with_types_dominated(&stream, &Default::default());
    assert!(
        res.is_ok(),
        "operands defined outside the stream should be considered dominating"
    );
}
