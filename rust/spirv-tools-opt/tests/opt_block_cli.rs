use std::process::Command;

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Loader};
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::sync::Mutex;
use tempfile::tempdir;

fn cpp_opt_bin() -> Option<String> {
    if let Ok(path) = std::env::var("SPIRV_CPP_OPT") {
        if !path.is_empty() {
            return Some(path);
        }
    }
    let from_path = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join("spirv-opt");
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    });
    if from_path.is_none() {
        eprintln!(
            "SPIRV_CPP_OPT not set and spirv-opt not found on PATH; skipping C++ parity check"
        );
    }
    from_path.map(|p| p.to_string_lossy().into_owned())
}

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

fn build_mul_identity_module() -> (Vec<u32>, u32) {
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
    let c0 = b.constant_bit32(int, 0);
    let c1 = b.constant_bit32(int, 1);
    let left = b.i_mul(int, None, c4, c1).expect("mul by one");
    let right = b.i_mul(int, None, c5, c0).expect("mul by zero");
    let sum = b.i_add(int, None, left, right).expect("add folded terms");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), sum)
}

fn build_div_rem_identity_module() -> (Vec<u32>, (u32, u32)) {
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
    let c42 = b.constant_bit32(int, 42);
    let c1 = b.constant_bit32(int, 1);
    let div = b.u_div(int, None, c42, c1).expect("div by one");
    let rem = b.u_mod(int, None, c42, c1).expect("rem by one");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), (div, rem))
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
fn cli_opt_block_force_env_enables_optimizer() {
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    std::env::set_var("SPIRV_TOOLS_FORCE_RUST_OPT", "1");
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
    std::env::remove_var("SPIRV_TOOLS_FORCE_RUST_OPT");
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
    assert!(has_const, "force env should fold subtraction");
    assert!(
        !has_sub,
        "subtraction should be folded when force env is set"
    );
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

#[test]
fn cli_opt_block_matches_cpp_mul_identities() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, sum_id) = build_mul_identity_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let rust_output = dir.path().join("rust_output.spv");
    let cpp_output = dir.path().join("cpp_output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let rust_status = Command::new(exe)
        .arg(&input)
        .arg(&rust_output)
        .status()
        .expect("run opt_block");
    assert!(
        rust_status.success(),
        "opt_block should succeed for identities"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for identity folding"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    let rust_const = find_const_value(&rust_words, sum_id);
    let cpp_const = find_const_value(&cpp_words, sum_id);
    assert_eq!(
        rust_const, cpp_const,
        "Rust CLI and C++ spirv-opt should fold mul identities the same way"
    );
    assert_eq!(
        rust_const,
        Some(4),
        "mul-by-one/zero identities should fold to constant 4"
    );
    assert!(
        !has_op(&rust_words, Op::IMul) && !has_op(&rust_words, Op::IAdd),
        "Rust output should remove mul/add after folding"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul) && !has_op(&cpp_words, Op::IAdd),
        "C++ output should remove mul/add after folding"
    );
}

#[test]
fn cli_opt_block_matches_cpp_div_rem_identities() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (div_id, rem_id)) = build_div_rem_identity_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let rust_output = dir.path().join("rust_output.spv");
    let cpp_output = dir.path().join("cpp_output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let rust_status = Command::new(exe)
        .arg(&input)
        .arg(&rust_output)
        .status()
        .expect("run opt_block");
    assert!(
        rust_status.success(),
        "opt_block should succeed for div/rem identities"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for div/rem identity folding"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    let rust_div = find_const_value(&rust_words, div_id);
    let cpp_div = find_const_value(&cpp_words, div_id);
    let rust_rem = find_const_value(&rust_words, rem_id);
    let cpp_rem = find_const_value(&cpp_words, rem_id);

    assert_eq!(
        rust_div, cpp_div,
        "Rust CLI and C++ spirv-opt should fold div-by-one the same way"
    );
    assert_eq!(
        rust_div,
        Some(42),
        "div-by-one should fold to the original value"
    );
    assert_eq!(
        rust_rem, cpp_rem,
        "Rust CLI and C++ spirv-opt should fold rem-by-one the same way"
    );
    assert_eq!(rust_rem, Some(0), "rem-by-one should fold to zero");
    assert!(
        !has_op(&rust_words, Op::UDiv) && !has_op(&rust_words, Op::UMod),
        "Rust output should remove div/rem after folding"
    );
    assert!(
        !has_op(&cpp_words, Op::UDiv) && !has_op(&cpp_words, Op::UMod),
        "C++ output should remove div/rem after folding"
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

fn find_const_value(words: &[u32], target_id: u32) -> Option<u32> {
    let mut loader = Loader::new();
    rspirv::binary::parse_words(words, &mut loader).expect("parse module");
    let module = loader.module();
    let result = module.all_inst_iter().find_map(|inst| {
        if inst.class.opcode == Op::Constant && inst.result_id == Some(target_id) {
            match inst.operands.as_slice() {
                [rspirv::dr::Operand::LiteralBit32(v)] => Some(*v),
                _ => None,
            }
        } else {
            None
        }
    });
    result
}

fn has_op(words: &[u32], op: Op) -> bool {
    let mut loader = Loader::new();
    rspirv::binary::parse_words(words, &mut loader).expect("parse module");
    let module = loader.module();
    let present = module.all_inst_iter().any(|inst| inst.class.opcode == op);
    present
}
