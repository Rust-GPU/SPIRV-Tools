use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

fn build_const_add_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c2 = b.constant_bit32(int, 2);
    let c3 = b.constant_bit32(int, 3);
    let add = b.i_add(int, None, c2, c3).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

#[test]
fn spirv_opt_cli_folds_add_by_default() {
    let (words, add_id) = build_const_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_add = false;
    let mut saw_const = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IAdd => saw_add = true,
            Op::Constant => {
                if inst.result_id == Some(add_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
                {
                    saw_const = true;
                }
            }
            _ => {}
        }
    }
    assert!(saw_const, "add should fold to constant 5 with same id");
    assert!(!saw_add, "addition should be removed");
}

#[test]
fn spirv_opt_cli_respects_passthrough() {
    let (words, add_id) = build_const_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--passthrough")
        .arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    assert_eq!(
        optimized_words, words,
        "passthrough should leave module unchanged"
    );

    // Double-check the add is present with the expected id.
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_add = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IAdd && inst.result_id == Some(add_id) {
            saw_add = true;
        }
    }
    assert!(saw_add, "passthrough should preserve add instruction");
}

#[test]
fn spirv_opt_cli_folds_add_negation() {
    let (words, add_id) = build_add_negate_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_zero = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IAdd {
            panic!("add should have been folded away");
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            found_zero = true;
        }
    }
    assert!(found_zero, "add+negate should fold to zero with same id");
}

fn build_umod_pow2_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c5 = b.constant_bit32(int, 5);
    let c1 = b.constant_bit32(int, 1);
    let c8 = b.constant_bit32(int, 8);
    let x = b.i_add(int, None, c5, c1).expect("iadd");
    let umod = b.u_mod(int, None, x, c8).expect("umod");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), umod)
}

#[test]
fn spirv_opt_cli_folds_mul_by_one() {
    let (words, mul_id) = build_mul_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IMul {
            panic!("mul should be folded away");
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "mul by one should fold to original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_udiv_by_one() {
    let (words, div_id) = build_udiv_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(inst.class.opcode, Op::UDiv, "udiv by one should be removed");
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(div_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(9)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "udiv by one should fold to original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_sdiv_by_one() {
    let (words, div_id) = build_sdiv_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(inst.class.opcode, Op::SDiv, "sdiv by one should be removed");
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(div_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(9)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "sdiv by one should fold to original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_urem_by_one() {
    let (words, rem_id) = build_urem_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_zero = false;
    for inst in module.all_inst_iter() {
        assert_ne!(inst.class.opcode, Op::UMod, "umod by one should fold");
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(rem_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            found_zero = true;
        }
    }
    assert!(found_zero, "umod by one should fold to zero with same id");
}

#[test]
fn spirv_opt_cli_folds_srem_by_one() {
    let (words, rem_id) = build_srem_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_zero = false;
    for inst in module.all_inst_iter() {
        assert_ne!(inst.class.opcode, Op::SRem, "srem by one should fold");
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(rem_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            found_zero = true;
        }
    }
    assert!(found_zero, "srem by one should fold to zero with same id");
}

#[test]
fn spirv_opt_cli_folds_shift_by_zero() {
    let (words, shl_id) = build_shift_zero_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::ShiftLeftLogical,
            "shift by zero should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(shl_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "shift by zero should rewrite to original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_rotate_pattern() {
    let (words, or_id) = build_rotate_fold_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut has_or = false;
    let mut found_const = false;
    let mut has_shl = false;
    let mut has_shr = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::BitwiseOr => has_or = true,
            Op::ShiftLeftLogical => has_shl = true,
            Op::ShiftRightLogical => has_shr = true,
            Op::Constant
                if inst.result_id == Some(or_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0x90)] =>
            {
                found_const = true;
            }
            _ => {}
        }
    }
    assert!(!has_or, "rotate pattern should fold away bitwise OR");
    assert!(!has_shl, "rotate pattern should fold away left shift");
    assert!(!has_shr, "rotate pattern should fold away right shift");
    assert!(
        found_const,
        "rotate pattern should fold to constant 0x90 with original id"
    );
}

#[test]
fn spirv_opt_cli_folds_rotate_pattern_commuted_or() {
    let (words, or_id) = build_rotate_fold_commuted_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut has_or = false;
    let mut found_const = false;
    let mut has_shl = false;
    let mut has_shr = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::BitwiseOr => has_or = true,
            Op::ShiftLeftLogical => has_shl = true,
            Op::ShiftRightLogical => has_shr = true,
            Op::Constant
                if inst.result_id == Some(or_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0x90)] =>
            {
                found_const = true;
            }
            _ => {}
        }
    }
    assert!(!has_or, "rotate pattern should fold away bitwise OR");
    assert!(!has_shl, "rotate pattern should fold away left shift");
    assert!(!has_shr, "rotate pattern should fold away right shift");
    assert!(
        found_const,
        "rotate pattern should fold to constant 0x90 with original id"
    );
}

#[test]
fn spirv_opt_cli_simplifies_bitand_all_ones() {
    let (words, and_id) = build_bitand_all_ones_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseAnd,
            "bitwise and with all ones should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(and_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0x1234_5678)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "and with all ones should fold to original value"
    );
}

#[test]
fn spirv_opt_cli_simplifies_bitor_all_ones() {
    let (words, or_id) = build_bitor_all_ones_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseOr,
            "bitwise or with all ones should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(or_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(u32::MAX)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "or with all ones should fold to constant all-ones with original id"
    );
}

#[test]
fn spirv_opt_cli_rewrites_bitxor_all_ones_to_not() {
    let (words, xor_id) = build_bitxor_all_ones_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let expected = !11u32; // xor with all ones
    let mut saw_not = false;
    let mut saw_const = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::BitwiseXor => panic!("bitwise xor should be rewritten"),
            Op::Not if inst.result_id == Some(xor_id) => saw_not = true,
            Op::Constant
                if inst.result_id == Some(xor_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(expected)] =>
            {
                saw_const = true;
            }
            _ => {}
        }
    }
    assert!(
        saw_not || saw_const,
        "xor with all ones should become a bitwise not or folded constant with the same id"
    );
}

#[test]
fn spirv_opt_cli_folds_bitand_zero() {
    let (words, and_id) = build_bitand_zero_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut saw_zero = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseAnd,
            "and with zero should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(and_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            saw_zero = true;
        }
    }
    assert!(saw_zero, "and with zero should fold to zero with same id");
}

#[test]
fn spirv_opt_cli_folds_bitor_zero() {
    let (words, or_id) = build_bitor_zero_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseOr,
            "or with zero should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(or_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "or with zero should fold to the original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_bitxor_zero() {
    let (words, xor_id) = build_bitxor_zero_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseXor,
            "xor with zero should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(xor_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(16)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "xor with zero should fold to the original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_bitand_self() {
    let (words, and_id) = build_bitand_self_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut saw_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseAnd,
            "and with self should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(and_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(10)]
        {
            saw_const = true;
        }
    }
    assert!(saw_const, "and with self should fold to the original value");
}

#[test]
fn spirv_opt_cli_folds_bitor_self() {
    let (words, or_id) = build_bitor_self_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseOr,
            "or with self should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(or_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(15)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "or with self should fold to the original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_bitxor_self() {
    let (words, xor_id) = build_bitxor_self_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_zero = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseXor,
            "xor with self should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(xor_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            found_zero = true;
        }
    }
    assert!(found_zero, "xor with self should fold to zero with same id");
}

#[test]
fn spirv_opt_cli_folds_bitand_complement() {
    let (words, and_id) = build_bitand_complement_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut saw_zero = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseAnd,
            "and with complement should be eliminated"
        );
        assert_ne!(inst.class.opcode, Op::Not, "dead not should be removed");
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(and_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            saw_zero = true;
        }
    }
    assert!(
        saw_zero,
        "and with complement should fold to zero with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_bitor_complement() {
    let (words, or_id) = build_bitor_complement_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut saw_ones = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::BitwiseOr,
            "or with complement should be eliminated"
        );
        assert_ne!(inst.class.opcode, Op::Not, "dead not should be removed");
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(or_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(u32::MAX)]
        {
            saw_ones = true;
        }
    }
    assert!(
        saw_ones,
        "or with complement should fold to all ones with same id"
    );
}

#[test]
fn spirv_opt_cli_rewrites_umod_pow2_to_mask() {
    let (words, umod_id) = build_umod_pow2_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_band = false;
    let mut saw_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(inst.class.opcode, Op::UMod, "umod should be rewritten");
        if inst.class.opcode == Op::BitwiseAnd {
            saw_band = true;
            let mask_is_7 = inst
                .operands
                .iter()
                .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(7)));
            assert!(mask_is_7, "expected mask 7 for modulo by 8");
            assert_eq!(
                inst.result_id,
                Some(umod_id),
                "should reuse original result id for bitmask"
            );
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(umod_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(6)]
        {
            saw_const = true;
        }
    }
    assert!(
        saw_band || saw_const,
        "expected bitwise and (or folded constant) to replace umod with power-of-two divisor"
    );
}

#[test]
fn spirv_opt_cli_folds_div_rem_and_negate_and_shifts() {
    let (words, _div_id_unused, _rem_id_unused, _neg_id_unused, _shl_id_unused, _shr_id_unused) =
        build_div_rem_neg_shift_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let _ = module; // Coverage sanity; future parity checks can assert specific folds.
}

static ENV_GUARD: Mutex<()> = Mutex::new(());

#[test]
fn spirv_opt_cli_respects_disable_env() {
    let _guard = ENV_GUARD.lock().unwrap();
    let (words, add_id) = build_const_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.env("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    assert_eq!(optimized_words, words, "env disable should passthrough");
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");

    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_add = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IAdd && inst.result_id == Some(add_id) {
            saw_add = true;
        }
    }
    assert!(saw_add, "env disable should preserve add");
}

#[test]
fn spirv_opt_cli_force_rust_ignores_disable_env() {
    let _guard = ENV_GUARD.lock().unwrap();
    let (words, add_id) = build_const_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.env("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
    cmd.arg("--force-rust")
        .arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_add = false;
    let mut saw_const = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IAdd => saw_add = true,
            Op::Constant => {
                if inst.result_id == Some(add_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
                {
                    saw_const = true;
                }
            }
            _ => {}
        }
    }
    assert!(
        saw_const && !saw_add,
        "force-rust should fold even when disable env is set"
    );
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
}

#[test]
fn spirv_opt_cli_folds_affine_gcd_add() {
    let _guard = ENV_GUARD.lock().unwrap();
    let (words, add_id) = build_affine_gcd_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_const = false;
    let mut saw_ops = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if inst.result_id == Some(add_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(36)]
                {
                    saw_const = true;
                }
            }
            Op::IAdd | Op::IMul if inst.result_id == Some(add_id) => saw_ops = true,
            _ => {}
        }
    }
    assert!(saw_const, "affine gcd add should fold to const 36");
    assert!(!saw_ops, "add/mul should be removed after folding");
}

#[test]
fn spirv_opt_cli_folds_mul_by_neg_one() {
    let (words, mul_id) = build_mul_neg_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IMul && inst.result_id == Some(mul_id) {
            panic!("mul by -1 should be rewritten or folded");
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(6))]
        {
            found = true;
        }
        if inst.class.opcode == Op::SNegate && inst.result_id == Some(mul_id) {
            found = true;
        }
    }
    assert!(found, "mul by -1 should become negate or folded constant");
}

fn cpp_opt_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SPIRV_CPP_OPT") {
        let pb = PathBuf::from(path);
        if pb.is_file() {
            return Some(pb);
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join("spirv-opt");
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

fn run_cli_with_path(
    words: &[u32],
    extra_args: &[&str],
    prepend_path: Option<&PathBuf>,
) -> std::process::Output {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    if let Some(path) = prepend_path {
        if let Some(parent) = path.parent() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!("{}:{}", parent.display(), current_path);
            cmd.env("PATH", new_path);
        }
    }
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(words))
        .expect("write module");
    child.wait_with_output().expect("run spirv-opt")
}

fn assert_cpp_cli_matches_rust(words: &[u32], label: &str) {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let rust_output = run_cli_with_path(words, &[], None);
    assert!(
        rust_output.status.success(),
        "rust cli ({label}) failed: {}",
        String::from_utf8_lossy(&rust_output.stderr)
    );

    let cpp_output = run_cli_with_path(words, &["--cpp"], Some(&cpp_opt));
    assert!(
        cpp_output.status.success(),
        "cpp cli ({label}) failed: {}",
        String::from_utf8_lossy(&cpp_output.stderr)
    );

    assert_eq!(
        rust_output.stdout, cpp_output.stdout,
        "Rust optimizer output should match C++ spirv-opt output ({label})"
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_output() {
    let (words, _) = build_const_add_module();
    assert_cpp_cli_matches_rust(&words, "const add fold");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_umod_pow2_output() {
    let (words, _) = build_umod_pow2_module();
    assert_cpp_cli_matches_rust(&words, "pow2 umod to mask");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_mul_by_one_output() {
    let (words, _) = build_mul_one_module();
    assert_cpp_cli_matches_rust(&words, "mul by one");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_mul_by_zero_output() {
    let (words, _) = build_mul_zero_module();
    assert_cpp_cli_matches_rust(&words, "mul by zero");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_add_negate_output() {
    let (words, _) = build_add_negate_module();
    assert_cpp_cli_matches_rust(&words, "add + negate");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_srem_divisible_output() {
    let (words, _) = build_srem_divisible_module();
    assert_cpp_cli_matches_rust(&words, "srem divisible");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_srem_non_divisible_output() {
    let (words, _) = build_srem_non_divisible_module();
    assert_cpp_cli_matches_rust(&words, "srem non-divisible");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_mul_neg_one_output() {
    let (words, _) = build_mul_neg_one_module();
    assert_cpp_cli_matches_rust(&words, "mul by -1");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_affine_gcd_add_output() {
    let (words, _) = build_affine_gcd_add_module();
    assert_cpp_cli_matches_rust(&words, "affine gcd add");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_pow2_mul_output() {
    let (words, _) = build_mul_pow2_module();
    assert_cpp_cli_matches_rust(&words, "mul by power-of-two");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_pow2_div_output() {
    let (words, _) = build_div_pow2_module();
    assert_cpp_cli_matches_rust(&words, "div by power-of-two");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_udiv_one_output() {
    let (words, _) = build_udiv_one_module();
    assert_cpp_cli_matches_rust(&words, "udiv by one");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_urem_one_output() {
    let (words, _) = build_urem_one_module();
    assert_cpp_cli_matches_rust(&words, "urem by one");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_sdiv_one_output() {
    let (words, _) = build_sdiv_one_module();
    assert_cpp_cli_matches_rust(&words, "sdiv by one");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_srem_one_output() {
    let (words, _) = build_srem_one_module();
    assert_cpp_cli_matches_rust(&words, "srem by one");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_shift_zero_output() {
    let (words, _) = build_shift_zero_module();
    assert_cpp_cli_matches_rust(&words, "shift by zero");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_rotate_output() {
    let (words, _) = build_rotate_fold_module();
    assert_cpp_cli_matches_rust(&words, "rotate fold");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_rotate_u64_output() {
    let (words, _) = build_rotate_fold_u64_module();
    assert_cpp_cli_matches_rust(&words, "rotate fold 64-bit");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_rotate_commuted_output() {
    let (words, _) = build_rotate_fold_commuted_module();
    assert_cpp_cli_matches_rust(&words, "rotate fold with commuted or");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_rotate_u64_commuted_output() {
    let (words, _) = build_rotate_fold_u64_commuted_module();
    assert_cpp_cli_matches_rust(&words, "rotate fold 64-bit with commuted or");
}

fn build_factored_const_mul_sum_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let mul_left = b.i_mul(int, None, lhs, c4).expect("mul left");
    let mul_right = b.i_mul(int, None, rhs, c4).expect("mul right");
    let add = b.i_add(int, None, mul_left, mul_right).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_factored_const_mul_sum_commuted_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let mul_left = b.i_mul(int, None, c4, lhs).expect("mul left");
    let mul_right = b.i_mul(int, None, c4, rhs).expect("mul right");
    let add = b.i_add(int, None, mul_left, mul_right).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_factored_const_mul_sub_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let mul_left = b.i_mul(int, None, lhs, c4).expect("mul left");
    let mul_right = b.i_mul(int, None, rhs, c4).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_factored_const_mul_sub_commuted_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let mul_left = b.i_mul(int, None, c4, lhs).expect("mul left");
    let mul_right = b.i_mul(int, None, c4, rhs).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_factored_const_mul_sum_mixed_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let mul_left = b.i_mul(int, None, lhs, c4).expect("mul left");
    let mul_right = b.i_mul(int, None, c4, rhs).expect("mul right");
    let add = b.i_add(int, None, mul_left, mul_right).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_factored_const_mul_sub_mixed_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let mul_left = b.i_mul(int, None, lhs, c4).expect("mul left");
    let mul_right = b.i_mul(int, None, c4, rhs).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_factored_symbolic_mul_sub_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let base = b.function_parameter(int).expect("base param");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let mul_left = b.i_mul(int, None, base, lhs).expect("mul left");
    let mul_right = b.i_mul(int, None, base, rhs).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, base, lhs, rhs)
}

fn build_factored_symbolic_mul_add_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let base = b.function_parameter(int).expect("base param");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let mul_left = b.i_mul(int, None, base, lhs).expect("mul left");
    let mul_right = b.i_mul(int, None, base, rhs).expect("mul right");
    let add = b.i_add(int, None, mul_left, mul_right).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add, base, lhs, rhs)
}

fn build_factored_symbolic_mul_sub_commuted_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let base = b.function_parameter(int).expect("base param");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let mul_left = b.i_mul(int, None, lhs, base).expect("mul left");
    let mul_right = b.i_mul(int, None, rhs, base).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, base, lhs, rhs)
}

fn build_factored_symbolic_mul_add_mixed_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let base = b.function_parameter(int).expect("base param");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let mul_left = b.i_mul(int, None, base, lhs).expect("mul left");
    let mul_right = b.i_mul(int, None, rhs, base).expect("mul right");
    let add = b.i_add(int, None, mul_left, mul_right).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add, base, lhs, rhs)
}

fn build_factored_symbolic_mul_sub_mixed_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let base = b.function_parameter(int).expect("base param");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let mul_left = b.i_mul(int, None, base, lhs).expect("mul left");
    let mul_right = b.i_mul(int, None, rhs, base).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, base, lhs, rhs)
}

fn build_factored_symbolic_mul_add_commuted_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int, int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let base = b.function_parameter(int).expect("base param");
    let lhs = b.function_parameter(int).expect("lhs param");
    let rhs = b.function_parameter(int).expect("rhs param");
    b.begin_block(None).expect("block");
    let mul_left = b.i_mul(int, None, lhs, base).expect("mul left");
    let mul_right = b.i_mul(int, None, rhs, base).expect("mul right");
    let add = b.i_add(int, None, mul_left, mul_right).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add, base, lhs, rhs)
}

fn build_factored_mul_sum_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("param id");
    b.begin_block(None).expect("block");
    let c2 = b.constant_bit32(int, 2);
    let c3 = b.constant_bit32(int, 3);
    let mul_left = b.i_mul(int, None, param, c2).expect("mul left");
    let mul_right = b.i_mul(int, None, param, c3).expect("mul right");
    let add = b.i_add(int, None, mul_left, mul_right).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add, param)
}

fn build_factored_mixed_const_difference_mul_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let c2 = b.constant_bit32(int, 2);
    let c5 = b.constant_bit32(int, 5);
    let mul_left = b.i_mul(int, None, c2, x).expect("mul left");
    let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_difference_mul_commuted_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let c2 = b.constant_bit32(int, 2);
    let c5 = b.constant_bit32(int, 5);
    let mul_left = b.i_mul(int, None, x, c2).expect("mul left commuted");
    let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_positive_difference_mul_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let c7 = b.constant_bit32(int, 7);
    let c2 = b.constant_bit32(int, 2);
    let mul_left = b.i_mul(int, None, c7, x).expect("mul left");
    let mul_right = b.i_mul(int, None, c2, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_positive_difference_mul_commuted_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let c7 = b.constant_bit32(int, 7);
    let c2 = b.constant_bit32(int, 2);
    let mul_left = b.i_mul(int, None, x, c7).expect("mul left commuted");
    let mul_right = b.i_mul(int, None, c2, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_wrap_negative_difference_mul_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let high = b.constant_bit32(int, u32::MAX - 1); // -2 in two's complement
    let c5 = b.constant_bit32(int, 5);
    let mul_left = b.i_mul(int, None, high, x).expect("mul left");
    let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_wrap_negative_difference_mul_commuted_module() -> (Vec<u32>, u32, u32)
{
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let high = b.constant_bit32(int, u32::MAX - 1); // -2 in two's complement
    let c5 = b.constant_bit32(int, 5);
    let mul_left = b.i_mul(int, None, x, high).expect("mul left commuted");
    let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_wrap_positive_difference_mul_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let neg1 = b.constant_bit32(int, u32::MAX); // -1
    let neg4 = b.constant_bit32(int, u32::MAX - 3); // -4
    let mul_left = b.i_mul(int, None, neg1, x).expect("mul left");
    let mul_right = b.i_mul(int, None, neg4, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_wrap_positive_difference_mul_commuted_module() -> (Vec<u32>, u32, u32)
{
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let neg1 = b.constant_bit32(int, u32::MAX); // -1
    let neg4 = b.constant_bit32(int, u32::MAX - 3); // -4
    let mul_left = b.i_mul(int, None, x, neg1).expect("mul left commuted");
    let mul_right = b.i_mul(int, None, neg4, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_const_equal_difference_mul_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let mul_left = b.i_mul(int, None, c6, x).expect("mul left");
    let mul_right = b.i_mul(int, None, c6, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_factored_const_equal_difference_mul_commuted_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let mul_left = b.i_mul(int, None, x, c6).expect("mul left commuted");
    let mul_right = b.i_mul(int, None, c6, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_factored_const_equal_difference_unsigned_mul_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let mul_left = b.i_mul(int, None, c6, x).expect("mul left");
    let mul_right = b.i_mul(int, None, x, c6).expect("mul right commuted");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_factored_const_difference_unsigned_mul_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let c7 = b.constant_bit32(int, 7);
    let c2 = b.constant_bit32(int, 2);
    let mul_left = b.i_mul(int, None, c7, x).expect("mul left");
    let mul_right = b.i_mul(int, None, c2, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_wrap_positive_difference_unsigned_mul_module() -> (Vec<u32>, u32, u32)
{
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let neg1 = b.constant_bit32(int, u32::MAX);
    let neg4 = b.constant_bit32(int, u32::MAX - 3);
    let mul_left = b.i_mul(int, None, neg1, x).expect("mul left");
    let mul_right = b.i_mul(int, None, neg4, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_wrap_positive_difference_unsigned_mul_commuted_module(
) -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let neg1 = b.constant_bit32(int, u32::MAX);
    let neg4 = b.constant_bit32(int, u32::MAX - 3);
    let mul_left = b.i_mul(int, None, x, neg1).expect("mul left commuted");
    let mul_right = b.i_mul(int, None, neg4, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_wrap_negative_difference_unsigned_mul_module() -> (Vec<u32>, u32, u32)
{
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let high = b.constant_bit32(int, u32::MAX - 1);
    let c5 = b.constant_bit32(int, 5);
    let mul_left = b.i_mul(int, None, high, x).expect("mul left");
    let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_factored_mixed_const_wrap_negative_difference_unsigned_mul_commuted_module(
) -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    b.begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let x = b.function_parameter(int).expect("x param");
    b.begin_block(None).expect("block");
    let high = b.constant_bit32(int, u32::MAX - 1);
    let c5 = b.constant_bit32(int, 5);
    let mul_left = b.i_mul(int, None, x, high).expect("mul left commuted");
    let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
    let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

#[test]
fn spirv_opt_cli_factors_common_multiplicand() {
    let (words, add_id, param_id) = build_factored_mul_sum_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut constants = std::collections::HashMap::new();
    let mut mul_count = 0;
    let mut add_seen = false;
    let mut factored_mul_matches = false;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if let (Some(id), Some(value)) = (
                    inst.result_id,
                    inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }),
                ) {
                    constants.insert(id, value);
                }
            }
            Op::IMul => {
                mul_count += 1;
                let Some(result_id) = inst.result_id else {
                    continue;
                };
                if result_id != add_id {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_param = lhs == param_id || rhs == param_id;
                let const_id = if lhs == param_id { rhs } else { lhs };
                let is_const_five = constants
                    .get(&const_id)
                    .copied()
                    .map(|v| v == 5)
                    .unwrap_or(false);
                factored_mul_matches = uses_param && is_const_five;
            }
            Op::IAdd => add_seen = true,
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave a single multiply");
    assert!(
        factored_mul_matches,
        "factored multiply should reuse add result id and scale param by five"
    );
    assert!(!add_seen, "addition should be removed after factoring");
}

#[test]
fn spirv_opt_cli_factors_shared_constant_from_sum() {
    let (words, add_id) = build_factored_const_mul_sum_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut constants = std::collections::HashMap::new();
    let mut add_result = None;
    let mut scaling_count = 0;
    let mut factored = false;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if let (Some(id), Some(value)) = (
                    inst.result_id,
                    inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }),
                ) {
                    constants.insert(id, value);
                }
            }
            Op::IAdd => {
                add_result = inst.result_id;
            }
            Op::IMul => {
                scaling_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(add_res_id) = add_result else {
                    continue;
                };
                let uses_add = lhs == add_res_id || rhs == add_res_id;
                let const_id = if lhs == add_res_id { rhs } else { lhs };
                let is_const_four = constants
                    .get(&const_id)
                    .copied()
                    .map(|v| v == 4)
                    .unwrap_or(false);
                factored = uses_add && is_const_four;
            }
            Op::ShiftLeftLogical => {
                scaling_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(add_res_id) = add_result else {
                    continue;
                };
                let uses_add = lhs == add_res_id;
                let is_shift_two = constants
                    .get(&rhs)
                    .copied()
                    .map(|v| v == 2)
                    .unwrap_or(false);
                factored = uses_add && is_shift_two;
            }
            _ => {}
        }
    }

    assert_eq!(
        scaling_count, 1,
        "factoring should leave one scaling instruction"
    );
    assert!(
        add_result.is_some(),
        "addition should remain as the inner sum"
    );
    assert!(
        factored,
        "factored multiply should reuse add result and scale by four"
    );
}

#[test]
fn spirv_opt_cli_factors_shared_constant_from_sum_commuted_mul() {
    let (words, add_id) = build_factored_const_mul_sum_commuted_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut constants = std::collections::HashMap::new();
    let mut add_result = None;
    let mut scaling_count = 0;
    let mut factored = false;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if let (Some(id), Some(value)) = (
                    inst.result_id,
                    inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }),
                ) {
                    constants.insert(id, value);
                }
            }
            Op::IAdd => {
                add_result = inst.result_id;
            }
            Op::IMul => {
                scaling_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(add_res_id) = add_result else {
                    continue;
                };
                let uses_add = lhs == add_res_id || rhs == add_res_id;
                let const_id = if lhs == add_res_id { rhs } else { lhs };
                let is_const_four = constants
                    .get(&const_id)
                    .copied()
                    .map(|v| v == 4)
                    .unwrap_or(false);
                factored = uses_add && is_const_four;
            }
            Op::ShiftLeftLogical => {
                scaling_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(add_res_id) = add_result else {
                    continue;
                };
                let uses_add = lhs == add_res_id;
                let is_shift_two = constants
                    .get(&rhs)
                    .copied()
                    .map(|v| v == 2)
                    .unwrap_or(false);
                factored = uses_add && is_shift_two;
            }
            _ => {}
        }
    }

    assert_eq!(
        scaling_count, 1,
        "factoring should leave one scaling instruction"
    );
    assert!(add_result.is_some(), "addition should remain as inner sum");
    assert!(
        factored,
        "factored multiply should reuse add result when constants lead the multiplies"
    );
}

#[test]
fn spirv_opt_cli_factors_shared_constant_from_sum_mixed_mul_order() {
    let (words, add_id) = build_factored_const_mul_sum_mixed_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut constants = std::collections::HashMap::new();
    let mut add_result = None;
    let mut scaling_count = 0;
    let mut factored = false;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if let (Some(id), Some(value)) = (
                    inst.result_id,
                    inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }),
                ) {
                    constants.insert(id, value);
                }
            }
            Op::IAdd => {
                add_result = inst.result_id;
            }
            Op::IMul => {
                scaling_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(add_res_id) = add_result else {
                    continue;
                };
                let uses_add = lhs == add_res_id || rhs == add_res_id;
                let const_id = if lhs == add_res_id { rhs } else { lhs };
                let is_const_four = constants
                    .get(&const_id)
                    .copied()
                    .map(|v| v == 4)
                    .unwrap_or(false);
                factored = uses_add && is_const_four;
            }
            Op::ShiftLeftLogical => {
                scaling_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(add_res_id) = add_result else {
                    continue;
                };
                let uses_add = lhs == add_res_id;
                let is_shift_two = constants
                    .get(&rhs)
                    .copied()
                    .map(|v| v == 2)
                    .unwrap_or(false);
                factored = uses_add && is_shift_two;
            }
            _ => {}
        }
    }

    assert_eq!(
        scaling_count, 1,
        "factoring should leave one scaling instruction"
    );
    assert!(add_result.is_some(), "addition should remain as inner sum");
    assert!(
        factored,
        "factored multiply should reuse add result even when only one multiply commutes the constant"
    );
}

#[test]
fn spirv_opt_cli_factors_shared_constant_from_sub() {
    let (words, sub_id) = build_factored_const_mul_sub_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut constants = std::collections::HashMap::new();
    let mut sub_count = 0;
    let mut scaling_count = 0;
    let mut factored = false;
    let mut inner_sub = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if let (Some(id), Some(value)) = (
                    inst.result_id,
                    inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }),
                ) {
                    constants.insert(id, value);
                }
            }
            Op::IAdd => {
                panic!("addition should not remain after factoring the subtract");
            }
            Op::ISub => {
                sub_count += 1;
                inner_sub = inst.result_id;
            }
            Op::IMul => {
                scaling_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_sub = inner_sub.is_some_and(|sid| lhs == sid || rhs == sid);
                let const_id = if inner_sub.is_some_and(|sid| lhs == sid) {
                    rhs
                } else {
                    lhs
                };
                let is_const_four = constants
                    .get(&const_id)
                    .copied()
                    .map(|v| v == 4)
                    .unwrap_or(false);
                factored = uses_sub && is_const_four;
            }
            Op::ShiftLeftLogical => {
                scaling_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_sub = inner_sub.is_some_and(|sid| lhs == sid);
                let is_shift_two = constants
                    .get(&rhs)
                    .copied()
                    .map(|v| v == 2)
                    .unwrap_or(false);
                factored = uses_sub && is_shift_two;
            }
            _ => {}
        }
    }

    assert_eq!(sub_count, 1, "inner subtract should remain");
    assert_eq!(
        scaling_count, 1,
        "factoring should leave a single scaling op"
    );
    assert!(factored, "scaling should reuse the subtract result id");
}

#[test]
fn spirv_opt_cli_factors_shared_constant_from_sub_commuted_mul() {
    let (words, sub_id) = build_factored_const_mul_sub_commuted_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut constants = std::collections::HashMap::new();
    let mut sub_count = 0;
    let mut scaling_count = 0;
    let mut factored = false;
    let mut inner_sub = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if let (Some(id), Some(value)) = (
                    inst.result_id,
                    inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }),
                ) {
                    constants.insert(id, value);
                }
            }
            Op::IAdd => panic!("addition should not remain after factoring the subtract"),
            Op::ISub => {
                sub_count += 1;
                inner_sub = inst.result_id;
            }
            Op::IMul => {
                scaling_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_sub = inner_sub.is_some_and(|sid| lhs == sid || rhs == sid);
                let const_id = if inner_sub.is_some_and(|sid| lhs == sid) {
                    rhs
                } else {
                    lhs
                };
                let is_const_four = constants
                    .get(&const_id)
                    .copied()
                    .map(|v| v == 4)
                    .unwrap_or(false);
                factored = uses_sub && is_const_four;
            }
            Op::ShiftLeftLogical => {
                scaling_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_sub = inner_sub.is_some_and(|sid| lhs == sid);
                let is_shift_two = constants
                    .get(&rhs)
                    .copied()
                    .map(|v| v == 2)
                    .unwrap_or(false);
                factored = uses_sub && is_shift_two;
            }
            _ => {}
        }
    }

    assert_eq!(sub_count, 1, "inner subtract should remain");
    assert_eq!(
        scaling_count, 1,
        "factoring should leave a single scaling op"
    );
    assert!(
        factored,
        "scaling should reuse the subtract result id when constants lead the multiplies"
    );
}

#[test]
fn spirv_opt_cli_factors_shared_constant_from_sub_mixed_mul_order() {
    let (words, sub_id) = build_factored_const_mul_sub_mixed_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut constants = std::collections::HashMap::new();
    let mut sub_count = 0;
    let mut scaling_count = 0;
    let mut factored = false;
    let mut inner_sub = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if let (Some(id), Some(value)) = (
                    inst.result_id,
                    inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }),
                ) {
                    constants.insert(id, value);
                }
            }
            Op::IAdd => panic!("addition should not remain after factoring the subtract"),
            Op::ISub => {
                sub_count += 1;
                inner_sub = inst.result_id;
            }
            Op::IMul => {
                scaling_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_sub = inner_sub.is_some_and(|sid| lhs == sid || rhs == sid);
                let const_id = if inner_sub.is_some_and(|sid| lhs == sid) {
                    rhs
                } else {
                    lhs
                };
                let is_const_four = constants
                    .get(&const_id)
                    .copied()
                    .map(|v| v == 4)
                    .unwrap_or(false);
                factored = uses_sub && is_const_four;
            }
            Op::ShiftLeftLogical => {
                scaling_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_sub = inner_sub.is_some_and(|sid| lhs == sid);
                let is_shift_two = constants
                    .get(&rhs)
                    .copied()
                    .map(|v| v == 2)
                    .unwrap_or(false);
                factored = uses_sub && is_shift_two;
            }
            _ => {}
        }
    }

    assert_eq!(sub_count, 1, "inner subtract should remain");
    assert_eq!(
        scaling_count, 1,
        "factoring should leave a single scaling op"
    );
    assert!(
        factored,
        "scaling should reuse the subtract result id when only one multiply commutes the constant"
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mul_output() {
    let (words, _, _) = build_factored_mul_sum_module();
    assert_cpp_cli_matches_rust(&words, "factored mul sum");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_output() {
    let (words, _) = build_factored_const_mul_sum_module();
    assert_cpp_cli_matches_rust(&words, "factored const sum");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_commuted_output() {
    let (words, _) = build_factored_const_mul_sum_commuted_module();
    assert_cpp_cli_matches_rust(&words, "factored const sum with commuted mul");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_mixed_output() {
    let (words, _) = build_factored_const_mul_sum_mixed_module();
    assert_cpp_cli_matches_rust(&words, "factored const sum with mixed mul ordering");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_sub_output() {
    let (words, _) = build_factored_const_mul_sub_module();
    assert_cpp_cli_matches_rust(&words, "factored const sub");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_sub_commuted_output() {
    let (words, _) = build_factored_const_mul_sub_commuted_module();
    assert_cpp_cli_matches_rust(&words, "factored const sub with commuted mul");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_sub_mixed_output() {
    let (words, _) = build_factored_const_mul_sub_mixed_module();
    assert_cpp_cli_matches_rust(&words, "factored const sub with mixed mul ordering");
}

#[test]
fn spirv_opt_cli_factors_symbolic_multiplicand_from_sub() {
    let (words, sub_id, base, lhs, rhs) = build_factored_symbolic_mul_sub_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut sub_seen = false;
    let mut factored = false;
    let mut diff_id = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::ISub => {
                diff_id = inst.result_id;
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                    sub_seen = true;
                }
            }
            Op::IMul => {
                mul_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_base = lhs_id == base || rhs_id == base;
                let diff_operand = if lhs_id == base { rhs_id } else { lhs_id };
                factored = uses_base && diff_id == Some(diff_operand);
            }
            Op::IAdd => panic!("addition should not remain in subtract factoring"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(sub_seen, "inner subtraction should remain");
    assert!(factored, "mul should reuse sub id and use base*diff");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_symbolic_sub_output() {
    let (words, ..) = build_factored_symbolic_mul_sub_module();
    assert_cpp_cli_matches_rust(&words, "factored symbolic subtract");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_symbolic_sub_mixed_output() {
    let (words, ..) = build_factored_symbolic_mul_sub_mixed_module();
    assert_cpp_cli_matches_rust(&words, "factored symbolic subtract with mixed mul order");
}

#[test]
fn spirv_opt_cli_factors_symbolic_multiplicand_from_sub_commuted_mul() {
    let (words, sub_id, base, lhs, rhs) = build_factored_symbolic_mul_sub_commuted_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut sub_seen = false;
    let mut factored = false;
    let mut diff_id = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::ISub => {
                diff_id = inst.result_id;
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                    sub_seen = true;
                }
            }
            Op::IMul => {
                mul_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_base = lhs_id == base || rhs_id == base;
                let diff_operand = if lhs_id == base { rhs_id } else { lhs_id };
                factored = uses_base && diff_id == Some(diff_operand);
            }
            Op::IAdd => panic!("addition should not remain in subtract factoring"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(sub_seen, "inner subtraction should remain");
    assert!(
        factored,
        "mul should reuse sub id and keep base as a multiplicand regardless of order"
    );
}

#[test]
fn spirv_opt_cli_factors_symbolic_multiplicand_from_sub_mixed_mul_order() {
    let (words, sub_id, base, lhs, rhs) = build_factored_symbolic_mul_sub_mixed_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut sub_seen = false;
    let mut factored = false;
    let mut diff_id = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::ISub => {
                diff_id = inst.result_id;
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                    sub_seen = true;
                }
            }
            Op::IMul => {
                mul_count += 1;
                if inst.result_id != Some(sub_id) {
                    continue;
                }
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_base = lhs_id == base || rhs_id == base;
                let diff_operand = if lhs_id == base { rhs_id } else { lhs_id };
                factored = uses_base && diff_id == Some(diff_operand);
            }
            Op::IAdd => panic!("addition should not remain for subtract factoring"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(sub_seen, "inner subtraction should remain");
    assert!(
        factored,
        "mul should reuse sub id and keep base as a multiplicand when only one mul commutes operands"
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_symbolic_sub_commuted_output() {
    let (words, ..) = build_factored_symbolic_mul_sub_commuted_module();
    assert_cpp_cli_matches_rust(&words, "factored symbolic subtract with commuted mul");
}

#[test]
fn spirv_opt_cli_factors_symbolic_multiplicand_from_add() {
    let (words, add_id, base, lhs, rhs) = build_factored_symbolic_mul_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut add_seen = false;
    let mut factored = false;
    let mut sum_id = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IAdd => {
                sum_id = inst.result_id;
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                    add_seen = true;
                }
            }
            Op::IMul => {
                mul_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_base = lhs_id == base || rhs_id == base;
                let sum_operand = if lhs_id == base { rhs_id } else { lhs_id };
                factored = uses_base && sum_id == Some(sum_operand);
            }
            Op::ISub => panic!("subtraction should not remain in addition factoring"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(add_seen, "inner addition should remain");
    assert!(factored, "mul should reuse add id and use base*sum");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_symbolic_add_output() {
    let (words, ..) = build_factored_symbolic_mul_add_module();
    assert_cpp_cli_matches_rust(&words, "factored symbolic addition");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_symbolic_add_mixed_output() {
    let (words, ..) = build_factored_symbolic_mul_add_mixed_module();
    assert_cpp_cli_matches_rust(&words, "factored symbolic addition with mixed mul order");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_difference_output() {
    let (words, ..) = build_factored_mixed_const_difference_mul_module();
    assert_cpp_cli_matches_rust(&words, "factored mixed constant difference");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_difference_commuted_output() {
    let (words, ..) = build_factored_mixed_const_difference_mul_commuted_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored mixed constant difference with commuted mul",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_positive_difference_output() {
    let (words, ..) = build_factored_mixed_const_positive_difference_mul_module();
    assert_cpp_cli_matches_rust(&words, "factored mixed positive constant difference");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_positive_difference_commuted_output() {
    let (words, ..) = build_factored_mixed_const_positive_difference_mul_commuted_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored mixed positive constant difference with commuted mul",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_wrap_negative_difference_output() {
    let (words, ..) = build_factored_mixed_const_wrap_negative_difference_mul_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored mixed wrapped negative constant difference",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_wrap_positive_difference_output() {
    let (words, ..) = build_factored_mixed_const_wrap_positive_difference_mul_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored mixed wrapped positive constant difference",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_wrap_positive_difference_commuted_output(
) {
    let (words, ..) = build_factored_mixed_const_wrap_positive_difference_mul_commuted_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored mixed wrapped positive constant difference with commuted mul",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_equal_difference_output() {
    let (words, ..) = build_factored_const_equal_difference_mul_module();
    assert_cpp_cli_matches_rust(&words, "factored equal constant difference to zero");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_equal_difference_commuted_output() {
    let (words, ..) = build_factored_const_equal_difference_mul_commuted_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored equal constant difference to zero with commuted mul",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_equal_difference_unsigned_output() {
    let (words, ..) = build_factored_const_equal_difference_unsigned_mul_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored equal constant difference to zero for unsigned ints",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_const_difference_unsigned_output() {
    let (words, ..) = build_factored_const_difference_unsigned_mul_module();
    assert_cpp_cli_matches_rust(&words, "factored unsigned constant difference");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_wrap_positive_difference_unsigned_output(
) {
    let (words, ..) = build_factored_mixed_const_wrap_positive_difference_unsigned_mul_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored unsigned wrapped positive constant difference",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_wrap_positive_difference_unsigned_commuted_output(
) {
    let (words, ..) =
        build_factored_mixed_const_wrap_positive_difference_unsigned_mul_commuted_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored unsigned wrapped positive constant difference with commuted mul",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_wrap_negative_difference_unsigned_output(
) {
    let (words, ..) = build_factored_mixed_const_wrap_negative_difference_unsigned_mul_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored unsigned wrapped negative constant difference",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_wrap_negative_difference_unsigned_commuted_output(
) {
    let (words, ..) =
        build_factored_mixed_const_wrap_negative_difference_unsigned_mul_commuted_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored unsigned wrapped negative constant difference with commuted mul",
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_mixed_const_wrap_negative_difference_commuted_output(
) {
    let (words, ..) = build_factored_mixed_const_wrap_negative_difference_mul_commuted_module();
    assert_cpp_cli_matches_rust(
        &words,
        "factored mixed wrapped negative constant difference with commuted mul",
    );
}

#[test]
fn spirv_opt_cli_factors_mixed_constant_difference_into_single_mul() {
    let (words, sub_id, param) = build_factored_mixed_const_difference_mul_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);

    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut factored = false;
    let mut constant = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IMul => {
                mul_count += 1;
                assert_eq!(inst.result_id, Some(sub_id), "mul should replace subtract");
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_param = lhs == param || rhs == param;
                let const_id = if lhs == param { rhs } else { lhs };
                constant = Some(const_id);
                factored = uses_param;
            }
            Op::Constant => {
                if let Some(value) = inst.operands.first().and_then(|op| match op {
                    rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                    _ => None,
                }) {
                    if Some(inst.result_id.unwrap()) == constant {
                        // -3 encoded in two's complement for 32-bit signed.
                        assert_eq!(value, u32::MAX - 2);
                    }
                }
            }
            Op::ISub => panic!("subtract should fold into a single multiply"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(
        factored,
        "mul should reuse the original subtract id with the shared param"
    );
    assert!(
        constant.is_some(),
        "factored multiply should include the constant difference"
    );
}

#[test]
fn spirv_opt_cli_factors_mixed_constant_difference_commuted_into_single_mul() {
    let (words, sub_id, param) = build_factored_mixed_const_difference_mul_commuted_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut factored = false;
    let mut constant = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IMul => {
                mul_count += 1;
                assert_eq!(inst.result_id, Some(sub_id), "mul should replace subtract");
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_param = lhs == param || rhs == param;
                let const_id = if lhs == param { rhs } else { lhs };
                constant = Some(const_id);
                factored = uses_param;
            }
            Op::Constant => {
                if let Some(value) = inst.operands.first().and_then(|op| match op {
                    rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                    _ => None,
                }) {
                    if Some(inst.result_id.unwrap()) == constant {
                        assert_eq!(value, u32::MAX - 2);
                    }
                }
            }
            Op::ISub => panic!("subtract should fold into a single multiply"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(
        factored,
        "mul should reuse the original subtract id with the shared param"
    );
    assert!(
        constant.is_some(),
        "factored multiply should include the constant difference"
    );
}

fn assert_const_difference_factors_to_mul(
    words: &[u32],
    expected_result: u32,
    param: u32,
    constant_value: u32,
) {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut factored = false;
    let mut constant = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IMul => {
                mul_count += 1;
                assert_eq!(
                    inst.result_id,
                    Some(expected_result),
                    "mul should replace subtract"
                );
                let Some(lhs) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_param = lhs == param || rhs == param;
                let const_id = if lhs == param { rhs } else { lhs };
                constant = Some(const_id);
                factored = uses_param;
            }
            Op::Constant => {
                if let Some(value) = inst.operands.first().and_then(|op| match op {
                    rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                    _ => None,
                }) {
                    if Some(inst.result_id.unwrap()) == constant {
                        assert_eq!(value, constant_value);
                    }
                }
            }
            Op::ISub => panic!("subtract should fold into a single multiply"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(
        factored,
        "mul should reuse the original subtract id with the shared param"
    );
    assert!(
        constant.is_some(),
        "factored multiply should include the constant difference"
    );
}

#[test]
fn spirv_opt_cli_factors_mixed_positive_constant_difference_into_single_mul() {
    let (words, sub_id, param) = build_factored_mixed_const_positive_difference_mul_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, 5);
}

#[test]
fn spirv_opt_cli_factors_mixed_positive_constant_difference_commuted_into_single_mul() {
    let (words, sub_id, param) =
        build_factored_mixed_const_positive_difference_mul_commuted_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, 5);
}

#[test]
fn spirv_opt_cli_factors_mixed_wrap_negative_constant_difference_into_single_mul() {
    let (words, sub_id, param) = build_factored_mixed_const_wrap_negative_difference_mul_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, u32::MAX - 6);
}

#[test]
fn spirv_opt_cli_factors_mixed_wrap_negative_constant_difference_commuted_into_single_mul() {
    let (words, sub_id, param) =
        build_factored_mixed_const_wrap_negative_difference_mul_commuted_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, u32::MAX - 6);
}

#[test]
fn spirv_opt_cli_factors_mixed_wrap_positive_constant_difference_into_single_mul() {
    let (words, sub_id, param) = build_factored_mixed_const_wrap_positive_difference_mul_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, 3);
}

#[test]
fn spirv_opt_cli_factors_mixed_wrap_positive_constant_difference_commuted_into_single_mul() {
    let (words, sub_id, param) =
        build_factored_mixed_const_wrap_positive_difference_mul_commuted_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, 3);
}

#[test]
fn spirv_opt_cli_factors_equal_constant_difference_into_zero() {
    let (words, sub_id) = build_factored_const_equal_difference_mul_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut saw_zero = false;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::ISub | Op::IMul => panic!("subtract/multiply should fold away"),
            Op::Constant => {
                if inst.result_id == Some(sub_id) {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        assert_eq!(value, 0);
                        saw_zero = true;
                    }
                }
            }
            _ => {}
        }
    }

    assert!(saw_zero, "sub id should become a zero constant");
}

#[test]
fn spirv_opt_cli_factors_equal_constant_difference_commuted_into_zero() {
    let (words, sub_id) = build_factored_const_equal_difference_mul_commuted_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut saw_zero = false;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::ISub | Op::IMul => panic!("subtract/multiply should fold away"),
            Op::Constant => {
                if inst.result_id == Some(sub_id) {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        assert_eq!(value, 0);
                        saw_zero = true;
                    }
                }
            }
            _ => {}
        }
    }

    assert!(saw_zero, "sub id should become a zero constant");
}

#[test]
fn spirv_opt_cli_factors_equal_constant_difference_unsigned_into_zero() {
    let (words, sub_id) = build_factored_const_equal_difference_unsigned_mul_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut saw_zero = false;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::ISub | Op::IMul => panic!("subtract/multiply should fold away"),
            Op::Constant => {
                if inst.result_id == Some(sub_id) {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        assert_eq!(value, 0);
                        saw_zero = true;
                    }
                }
            }
            _ => {}
        }
    }

    assert!(saw_zero, "sub id should become a zero constant");
}

#[test]
fn spirv_opt_cli_factors_unsigned_constant_difference_into_single_mul() {
    let (words, sub_id, param) = build_factored_const_difference_unsigned_mul_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, 5);
}

#[test]
fn spirv_opt_cli_factors_unsigned_wrap_negative_constant_difference_into_single_mul() {
    let (words, sub_id, param) =
        build_factored_mixed_const_wrap_negative_difference_unsigned_mul_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, u32::MAX - 6);
}

#[test]
fn spirv_opt_cli_factors_unsigned_wrap_negative_constant_difference_commuted_into_single_mul() {
    let (words, sub_id, param) =
        build_factored_mixed_const_wrap_negative_difference_unsigned_mul_commuted_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, u32::MAX - 6);
}

#[test]
fn spirv_opt_cli_factors_unsigned_wrap_positive_constant_difference_into_single_mul() {
    let (words, sub_id, param) =
        build_factored_mixed_const_wrap_positive_difference_unsigned_mul_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, 3);
}

#[test]
fn spirv_opt_cli_factors_unsigned_wrap_positive_constant_difference_commuted_into_single_mul() {
    let (words, sub_id, param) =
        build_factored_mixed_const_wrap_positive_difference_unsigned_mul_commuted_module();
    assert_const_difference_factors_to_mul(&words, sub_id, param, 3);
}

#[test]
fn spirv_opt_cli_factors_symbolic_multiplicand_from_add_commuted_mul() {
    let (words, add_id, base, lhs, rhs) = build_factored_symbolic_mul_add_commuted_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut add_seen = false;
    let mut factored = false;
    let mut sum_id = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IAdd => {
                sum_id = inst.result_id;
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                    add_seen = true;
                }
            }
            Op::IMul => {
                mul_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_base = lhs_id == base || rhs_id == base;
                let sum_operand = if lhs_id == base { rhs_id } else { lhs_id };
                factored = uses_base && sum_id == Some(sum_operand);
            }
            Op::ISub => panic!("subtraction should not remain in addition factoring"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(add_seen, "inner addition should remain");
    assert!(
        factored,
        "mul should reuse add id and keep base as a multiplicand regardless of order"
    );
}

#[test]
fn spirv_opt_cli_factors_symbolic_multiplicand_from_add_mixed_mul_order() {
    let (words, add_id, base, lhs, rhs) = build_factored_symbolic_mul_add_mixed_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut mul_count = 0;
    let mut add_seen = false;
    let mut factored = false;
    let mut sum_id = None;

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IAdd => {
                sum_id = inst.result_id;
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                    add_seen = true;
                }
            }
            Op::IMul => {
                mul_count += 1;
                if inst.result_id != Some(add_id) {
                    continue;
                }
                let Some(lhs_id) = inst.operands.first().and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                    continue;
                };
                let uses_base = lhs_id == base || rhs_id == base;
                let sum_operand = if lhs_id == base { rhs_id } else { lhs_id };
                factored = uses_base && sum_id == Some(sum_operand);
            }
            Op::ISub => panic!("subtraction should not remain in addition factoring"),
            _ => {}
        }
    }

    assert_eq!(mul_count, 1, "factoring should leave one multiply");
    assert!(add_seen, "inner addition should remain");
    assert!(
        factored,
        "mul should reuse add id and keep base as a multiplicand when only one mul commutes operands"
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_factored_symbolic_add_commuted_output() {
    let (words, ..) = build_factored_symbolic_mul_add_commuted_module();
    assert_cpp_cli_matches_rust(&words, "factored symbolic addition with commuted mul");
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn build_add_negate_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c5 = b.constant_bit32(int, 5);
    let neg = b.s_negate(int, None, c5).expect("neg");
    let add = b.i_add(int, None, c5, neg).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_mul_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let c1 = b.constant_bit32(int, 1);
    let mul = b.i_mul(int, None, c4, c1).expect("mul id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_udiv_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c9 = b.constant_bit32(int, 9);
    let c1 = b.constant_bit32(int, 1);
    let div = b.u_div(int, None, c9, c1).expect("div id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), div)
}

fn build_urem_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c9 = b.constant_bit32(int, 9);
    let c1 = b.constant_bit32(int, 1);
    let rem = b.u_mod(int, None, c9, c1).expect("rem id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), rem)
}

fn build_sdiv_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c9 = b.constant_bit32(int, 9);
    let c1 = b.constant_bit32(int, 1);
    let div = b.s_div(int, None, c9, c1).expect("div id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), div)
}

fn build_srem_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c9 = b.constant_bit32(int, 9);
    let c1 = b.constant_bit32(int, 1);
    let rem = b.s_rem(int, None, c9, c1).expect("rem id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), rem)
}

fn build_rotate_fold_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let value = b.constant_bit32(int, 0x12);
    let shift = b.constant_bit32(int, 3);
    let left = b.shift_left_logical(int, None, value, shift).expect("shl");
    let right_amount = b.constant_bit32(int, 29);
    let right = b
        .shift_right_logical(int, None, value, right_amount)
        .expect("shr");
    let or = b
        .bitwise_or(int, None, left, right)
        .expect("rotate pattern");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), or)
}

fn build_rotate_fold_commuted_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let value = b.constant_bit32(int, 0x12);
    let shift = b.constant_bit32(int, 3);
    let left = b.shift_left_logical(int, None, value, shift).expect("shl");
    let right_amount = b.constant_bit32(int, 29);
    let right = b
        .shift_right_logical(int, None, value, right_amount)
        .expect("shr");
    let or = b
        .bitwise_or(int, None, right, left)
        .expect("rotate pattern commuted");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), or)
}

fn build_rotate_fold_u64_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Int64);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(64, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let value = b.constant_bit64(int, 0x12);
    let shift = b.constant_bit64(int, 3);
    let left = b.shift_left_logical(int, None, value, shift).expect("shl");
    let right_amount = b.constant_bit64(int, 61);
    let right = b
        .shift_right_logical(int, None, value, right_amount)
        .expect("shr");
    let or = b
        .bitwise_or(int, None, left, right)
        .expect("rotate pattern 64-bit");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), or)
}

fn build_rotate_fold_u64_commuted_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.capability(rspirv::spirv::Capability::Int64);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(64, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let value = b.constant_bit64(int, 0x12);
    let shift = b.constant_bit64(int, 3);
    let left = b.shift_left_logical(int, None, value, shift).expect("shl");
    let right_amount = b.constant_bit64(int, 61);
    let right = b
        .shift_right_logical(int, None, value, right_amount)
        .expect("shr");
    let or = b
        .bitwise_or(int, None, right, left)
        .expect("rotate pattern commuted 64-bit");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), or)
}

fn build_bitand_all_ones_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let value = b.constant_bit32(int, 0x1234_5678);
    let ones = b.constant_bit32(int, u32::MAX);
    let band = b.bitwise_and(int, None, value, ones).expect("bitwise and");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), band)
}

fn build_bitor_all_ones_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let value = b.constant_bit32(int, 0xBEEF_CAFE);
    let ones = b.constant_bit32(int, u32::MAX);
    let bor = b.bitwise_or(int, None, value, ones).expect("bitwise or");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), bor)
}

fn build_bitxor_all_ones_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let lhs = b.constant_bit32(int, 5);
    let rhs = b.constant_bit32(int, 6);
    let value = b.i_add(int, None, lhs, rhs).expect("value");
    let ones = b.constant_bit32(int, u32::MAX);
    let bxor = b.bitwise_xor(int, None, value, ones).expect("bitwise xor");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), bxor)
}

fn build_bitand_zero_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let value = b.constant_bit32(int, 0xDEAD_BEEF);
    let zero = b.constant_bit32(int, 0);
    let band = b.bitwise_and(int, None, value, zero).expect("bitwise and");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), band)
}

fn build_bitor_zero_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let lhs = b.constant_bit32(int, 2);
    let rhs = b.constant_bit32(int, 3);
    let value = b.i_add(int, None, lhs, rhs).expect("value");
    let zero = b.constant_bit32(int, 0);
    let bor = b.bitwise_or(int, None, value, zero).expect("bitwise or");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), bor)
}

fn build_bitxor_zero_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let lhs = b.constant_bit32(int, 7);
    let rhs = b.constant_bit32(int, 9);
    let value = b.i_add(int, None, lhs, rhs).expect("value");
    let zero = b.constant_bit32(int, 0);
    let bxor = b.bitwise_xor(int, None, value, zero).expect("bitwise xor");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), bxor)
}

fn build_bitand_self_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let lhs = b.constant_bit32(int, 4);
    let rhs = b.constant_bit32(int, 6);
    let value = b.i_add(int, None, lhs, rhs).expect("value");
    let band = b.bitwise_and(int, None, value, value).expect("bitwise and");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), band)
}

fn build_bitor_self_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let lhs = b.constant_bit32(int, 7);
    let rhs = b.constant_bit32(int, 8);
    let value = b.i_add(int, None, lhs, rhs).expect("value");
    let bor = b.bitwise_or(int, None, value, value).expect("bitwise or");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), bor)
}

fn build_bitxor_self_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let lhs = b.constant_bit32(int, 5);
    let rhs = b.constant_bit32(int, 9);
    let value = b.i_add(int, None, lhs, rhs).expect("value");
    let bxor = b.bitwise_xor(int, None, value, value).expect("bitwise xor");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), bxor)
}

fn build_bitand_complement_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let lhs = b.constant_bit32(int, 2);
    let rhs = b.constant_bit32(int, 3);
    let value = b.i_add(int, None, lhs, rhs).expect("value");
    let not_value = b.not(int, None, value).expect("not");
    let band = b.bitwise_and(int, None, value, not_value).expect("band");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), band)
}

fn build_bitor_complement_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let lhs = b.constant_bit32(int, 11);
    let rhs = b.constant_bit32(int, 5);
    let value = b.i_add(int, None, lhs, rhs).expect("value");
    let not_value = b.not(int, None, value).expect("not");
    let bor = b.bitwise_or(int, None, value, not_value).expect("bor");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), bor)
}

fn build_shift_zero_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let zero = b.constant_bit32(int, 0);
    let shl = b.shift_left_logical(int, None, c4, zero).expect("shift id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), shl)
}

fn build_mul_neg_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let c_neg_one = b.constant_bit32(int, u32::MAX);
    let mul = b.i_mul(int, None, c6, c_neg_one).expect("mul id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_mul_zero_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let c0 = b.constant_bit32(int, 0);
    let mul = b.i_mul(int, None, c4, c0).expect("mul id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_srem_divisible_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("function parameter");
    let _ = b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let c3 = b.constant_bit32(int, 3);
    let mul = b.i_mul(int, None, param, c6).expect("mul");
    let rem = b.s_rem(int, None, mul, c3).expect("srem");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), rem)
}

fn build_srem_non_divisible_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("function parameter");
    let _ = b.begin_block(None).expect("block");
    let c5 = b.constant_bit32(int, 5);
    let c3 = b.constant_bit32(int, 3);
    let mul = b.i_mul(int, None, param, c5).expect("mul");
    let rem = b.s_rem(int, None, mul, c3).expect("srem");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), rem)
}

fn build_div_rem_neg_shift_module() -> (Vec<u32>, u32, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c20 = b.constant_bit32(int, 20);
    let c4 = b.constant_bit32(int, 4);
    let c5 = b.constant_bit32(int, 5);
    let div = b.u_div(int, None, c20, c4).expect("div");
    let rem = b.u_mod(int, None, c20, c4).expect("rem");
    let neg = b.s_negate(int, None, c5).expect("neg");
    let shl = b.shift_left_logical(int, None, c5, c4).expect("shl");
    let shr = b.shift_right_logical(int, None, c20, c4).expect("shr");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), div, rem, neg, shl, shr)
}

fn build_affine_gcd_add_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let c12 = b.constant_bit32(int, 12);
    let x = b.constant_bit32(int, 4);
    let mul = b.i_mul(int, None, c6, x).expect("mul");
    let add = b.i_add(int, None, mul, c12).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_mul_pow2_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("param");
    let _ = b.begin_block(None).expect("block");
    let c8 = b.constant_bit32(int, 8);
    let mul = b.i_mul(int, None, param, c8).expect("mul");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_div_pow2_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("param");
    let _ = b.begin_block(None).expect("block");
    let c8 = b.constant_bit32(int, 8);
    let div = b.u_div(int, None, param, c8).expect("div");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), div)
}
