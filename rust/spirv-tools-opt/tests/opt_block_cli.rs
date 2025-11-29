use std::process::Command;

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Loader};
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::sync::Mutex;
use tempfile::tempdir;

fn build_sample_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let c4 = b.constant_bit32(int, 4);
    let c5 = b.constant_bit32(int, 5);
    let c2 = b.constant_bit32(int, 2);
    let add = b.i_add(int, None, c4, c5).expect("add");
    let sub = b.i_sub(int, None, add, c2).expect("sub");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), sub)
}

#[test]
fn cli_opt_block_folds_arithmetic() {
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, sub_id) = build_sample_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let output = dir.path().join("output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .status()
        .expect("run opt_block");
    assert!(status.success(), "opt_block should exit successfully");

    let optimized_bytes = std::fs::read(&output).expect("read output");
    let optimized_words = bytes_to_words(&optimized_bytes);
    let mut loader = Loader::new();
    rspirv::binary::parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let has_const_three = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
    });
    let has_sub = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::ISub);
    assert!(
        has_const_three,
        "folded value should be written as constant 7"
    );
    assert!(!has_sub, "subtraction should be folded away");
}

#[test]
fn cli_opt_block_respects_disable_env() {
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
    let (words, sub_id) = build_sample_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let output = dir.path().join("output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .status()
        .expect("run opt_block");
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    assert!(status.success(), "opt_block should exit successfully");

    let optimized_bytes = std::fs::read(&output).expect("read output");
    let optimized_words = bytes_to_words(&optimized_bytes);
    assert_eq!(optimized_words, words, "env disable should passthrough");

    let mut loader = Loader::new();
    rspirv::binary::parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let saw_sub = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::ISub && inst.result_id == Some(sub_id));
    assert!(saw_sub, "sub should remain when env disables optimizer");
}

#[test]
fn cli_opt_block_force_rust_overrides_disable_env() {
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
    let (words, sub_id) = build_sample_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let output = dir.path().join("output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let status = Command::new(exe)
        .arg("--force-rust")
        .arg(&input)
        .arg(&output)
        .status()
        .expect("run opt_block");
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    assert!(status.success(), "opt_block should exit successfully");

    let optimized_bytes = std::fs::read(&output).expect("read output");
    let optimized_words = bytes_to_words(&optimized_bytes);
    let mut loader = Loader::new();
    rspirv::binary::parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let has_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
    });
    let has_sub = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::ISub);
    assert!(has_const, "force should still fold subtraction");
    assert!(!has_sub, "subtraction should be folded when forced");
}

#[test]
fn cli_opt_block_passthrough_flag() {
    let (words, sub_id) = build_sample_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let output = dir.path().join("output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let status = Command::new(exe)
        .arg("--passthrough")
        .arg(&input)
        .arg(&output)
        .status()
        .expect("run opt_block");
    assert!(status.success(), "opt_block should exit successfully");

    let optimized_bytes = std::fs::read(&output).expect("read output");
    let optimized_words = bytes_to_words(&optimized_bytes);
    assert_eq!(
        optimized_words, words,
        "passthrough should skip optimization"
    );

    let mut loader = Loader::new();
    rspirv::binary::parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let saw_sub = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::ISub && inst.result_id == Some(sub_id));
    assert!(saw_sub, "sub should remain when passthrough flag is set");
}

#[test]
fn cli_opt_block_rejects_unaligned_input() {
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let output = dir.path().join("output.spv");
    // Write three bytes to force a non-multiple-of-4 error path.
    std::fs::write(&input, [0u8, 1, 2]).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .status()
        .expect("run opt_block");
    assert!(
        !status.success(),
        "unaligned input should cause opt_block to fail"
    );
    assert!(
        !output.exists(),
        "output should not be produced when input is invalid"
    );
}

static ENV_GUARD: Mutex<()> = Mutex::new(());

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            u32::from_le_bytes(arr)
        })
        .collect()
}
