use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::io::Write;

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
