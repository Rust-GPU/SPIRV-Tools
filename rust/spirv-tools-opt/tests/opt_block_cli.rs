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

fn build_udiv_rem_identity_module_u64() -> (Vec<u32>, (u32, u32)) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(64, 0);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let c42 = b.constant_bit64(int, 42);
    let c1 = b.constant_bit64(int, 1);
    let div = b.u_div(int, None, c42, c1).expect("udiv by one");
    let rem = b.u_mod(int, None, c42, c1).expect("umod by one");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), (div, rem))
}

fn build_signed_div_rem_identity_module() -> (Vec<u32>, (u32, u32)) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let c42 = b.constant_bit32(int, 42);
    let c1 = b.constant_bit32(int, 1);
    let div = b.s_div(int, None, c42, c1).expect("sdiv by one");
    let rem = b.s_rem(int, None, c42, c1).expect("srem by one");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), (div, rem))
}

fn build_mul_pow2_module() -> (Vec<u32>, (u32, u32)) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit32(int, 8);
    let c3 = b.constant_bit32(int, 3);
    let mul = b.i_mul(int, None, param, c8).expect("mul pow2");
    b.ret_value(mul).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), (mul, c3))
}

fn build_mul_neg_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c_neg_one = b.constant_bit32(int, 0xFFFF_FFFF);
    let mul = b.i_mul(int, None, param, c_neg_one).expect("mul -1");
    b.ret_value(mul).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), mul)
}

fn build_udiv_pow2_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit32(int, 8);
    let div = b.u_div(int, None, param, c8).expect("udiv pow2");
    b.ret_value(div).unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn build_umod_pow2_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit32(int, 8);
    let rem = b.u_mod(int, None, param, c8).expect("umod pow2");
    b.ret_value(rem).unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn build_udiv_pow2_module_u64() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(64, 0);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit64(int, 8);
    let div = b.u_div(int, None, param, c8).expect("udiv pow2 u64");
    b.ret_value(div).unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn build_umod_pow2_module_u64() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(64, 0);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit64(int, 8);
    let rem = b.u_mod(int, None, param, c8).expect("umod pow2 u64");
    b.ret_value(rem).unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn build_sdiv_pow2_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit32(int, 8);
    let div = b.s_div(int, None, param, c8).expect("sdiv pow2");
    b.ret_value(div).unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn build_srem_pow2_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit32(int, 8);
    let rem = b.s_rem(int, None, param, c8).expect("srem pow2");
    b.ret_value(rem).unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn build_sdiv_pow2_module_u64() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(64, 1);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit64(int, 8);
    let div = b.s_div(int, None, param, c8).expect("sdiv pow2 u64");
    b.ret_value(div).unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn build_srem_pow2_module_u64() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(64, 1);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c8 = b.constant_bit64(int, 8);
    let rem = b.s_rem(int, None, param, c8).expect("srem pow2 u64");
    b.ret_value(rem).unwrap();
    b.end_function().unwrap();
    b.module().assemble()
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

#[test]
fn cli_opt_block_matches_cpp_div_rem_identities_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (div_id, rem_id)) = build_udiv_rem_identity_module_u64();
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
        "opt_block should succeed for u64 div/rem identities"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for u64 div/rem identity folding"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    let rust_div = find_const_value(&rust_words, div_id);
    let cpp_div = find_const_value(&cpp_words, div_id);
    let rust_rem = find_const_value(&rust_words, rem_id);
    let cpp_rem = find_const_value(&cpp_words, rem_id);

    assert_eq!(
        rust_div, cpp_div,
        "Rust CLI and C++ spirv-opt should fold u64 div-by-one the same way"
    );
    assert_eq!(
        rust_div,
        Some(42),
        "u64 div-by-one should fold to the original value"
    );
    assert_eq!(
        rust_rem, cpp_rem,
        "Rust CLI and C++ spirv-opt should fold u64 rem-by-one the same way"
    );
    assert_eq!(rust_rem, Some(0), "u64 rem-by-one should fold to zero");
    assert!(
        !has_op(&rust_words, Op::UDiv) && !has_op(&rust_words, Op::UMod),
        "Rust output should remove div/rem after folding"
    );
    assert!(
        !has_op(&cpp_words, Op::UDiv) && !has_op(&cpp_words, Op::UMod),
        "C++ output should remove div/rem after folding"
    );
}

#[test]
fn cli_opt_block_matches_cpp_signed_div_rem_identities() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (div_id, rem_id)) = build_signed_div_rem_identity_module();
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
        "opt_block should succeed for signed div/rem identities"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for signed div/rem identity folding"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    let rust_div = find_const_value(&rust_words, div_id);
    let cpp_div = find_const_value(&cpp_words, div_id);
    let rust_rem = find_const_value(&rust_words, rem_id);
    let cpp_rem = find_const_value(&cpp_words, rem_id);

    assert_eq!(
        rust_div, cpp_div,
        "Rust CLI and C++ spirv-opt should fold signed div-by-one the same way"
    );
    assert_eq!(
        rust_div,
        Some(42),
        "signed div-by-one should fold to the original value"
    );
    assert_eq!(
        rust_rem, cpp_rem,
        "Rust CLI and C++ spirv-opt should fold signed rem-by-one the same way"
    );
    assert_eq!(rust_rem, Some(0), "signed rem-by-one should fold to zero");
    assert!(
        !has_op(&rust_words, Op::SDiv) && !has_op(&rust_words, Op::SRem),
        "Rust output should remove sdiv/srem after folding"
    );
    assert!(
        !has_op(&cpp_words, Op::SDiv) && !has_op(&cpp_words, Op::SRem),
        "C++ output should remove sdiv/srem after folding"
    );
}

#[test]
fn cli_opt_block_matches_cpp_mul_pow2_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (_mul_id, shift_const_id)) = build_mul_pow2_module();
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
        "opt_block should succeed for pow2 rewrite"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for pow2 rewrite"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::IMul),
        "Rust output should remove mul after rewrite"
    );
    assert!(
        has_op(&rust_words, Op::ShiftLeftLogical),
        "Rust output should include shift after rewrite"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul),
        "C++ output should remove mul after rewrite"
    );
    assert!(
        has_op(&cpp_words, Op::ShiftLeftLogical),
        "C++ output should include shift after rewrite"
    );

    let rust_shift = find_const_value(&rust_words, shift_const_id);
    let cpp_shift = find_const_value(&cpp_words, shift_const_id);
    assert_eq!(
        rust_shift, cpp_shift,
        "Rust CLI and C++ spirv-opt should agree on shift amount"
    );
    assert_eq!(rust_shift, Some(3), "pow2 rewrite should shift by 3");
}

#[test]
fn cli_opt_block_matches_cpp_mul_neg_one_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, mul_id) = build_mul_neg_one_module();
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
        "opt_block should succeed for mul-by-neg-one rewrite"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for mul-by-neg-one rewrite"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::IMul),
        "Rust output should remove mul after rewrite"
    );
    assert!(
        has_op(&rust_words, Op::SNegate),
        "Rust output should include negate after rewrite"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul),
        "C++ output should remove mul after rewrite"
    );
    assert!(
        has_op(&cpp_words, Op::SNegate),
        "C++ output should include negate after rewrite"
    );

    // Ensure the result id still maps to the rewritten value by confirming the instruction exists.
    assert!(
        module_has_result(&rust_words, mul_id),
        "Rust output should keep the result id alive after rewrite"
    );
    assert!(
        module_has_result(&cpp_words, mul_id),
        "C++ output should keep the result id alive after rewrite"
    );
}

#[test]
fn cli_opt_block_matches_cpp_udiv_pow2_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_udiv_pow2_module();
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
        "opt_block should succeed for udiv pow2 rewrite"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for udiv pow2 rewrite"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::UDiv),
        "Rust output should remove udiv after rewrite"
    );
    assert!(
        has_op(&rust_words, Op::ShiftRightLogical),
        "Rust output should include logical shift after rewrite"
    );
    assert!(
        has_const_literal(&rust_words, 3),
        "Rust output should include shift amount 3"
    );
    assert!(
        !has_op(&cpp_words, Op::UDiv),
        "C++ output should remove udiv after rewrite"
    );
    assert!(
        has_op(&cpp_words, Op::ShiftRightLogical),
        "C++ output should include logical shift after rewrite"
    );
    assert!(
        has_const_literal(&cpp_words, 3),
        "C++ output should include shift amount 3"
    );
}

#[test]
fn cli_opt_block_matches_cpp_umod_pow2_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_umod_pow2_module();
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
        "opt_block should succeed for umod pow2 rewrite"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for umod pow2 rewrite"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::UMod),
        "Rust output should remove umod after rewrite"
    );
    assert!(
        has_op(&rust_words, Op::BitwiseAnd),
        "Rust output should include mask after rewrite"
    );
    assert!(
        has_const_literal(&rust_words, 7),
        "Rust output should include mask literal 7"
    );
    assert!(
        !has_op(&cpp_words, Op::UMod),
        "C++ output should remove umod after rewrite"
    );
    assert!(
        has_op(&cpp_words, Op::BitwiseAnd),
        "C++ output should include mask after rewrite"
    );
    assert!(
        has_const_literal(&cpp_words, 7),
        "C++ output should include mask literal 7"
    );
}

#[test]
fn cli_opt_block_matches_cpp_udiv_pow2_rewrite_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_udiv_pow2_module_u64();
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
        "opt_block should succeed for udiv pow2 rewrite u64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for udiv pow2 rewrite u64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::UDiv),
        "Rust output should remove udiv after rewrite"
    );
    assert!(
        has_op(&rust_words, Op::ShiftRightLogical),
        "Rust output should include logical shift after rewrite"
    );
    assert!(
        has_const_literal(&rust_words, 3),
        "Rust output should include shift amount 3"
    );
    assert!(
        !has_op(&cpp_words, Op::UDiv),
        "C++ output should remove udiv after rewrite"
    );
    assert!(
        has_op(&cpp_words, Op::ShiftRightLogical),
        "C++ output should include logical shift after rewrite"
    );
    assert!(
        has_const_literal(&cpp_words, 3),
        "C++ output should include shift amount 3"
    );
}

#[test]
fn cli_opt_block_matches_cpp_umod_pow2_rewrite_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_umod_pow2_module_u64();
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
        "opt_block should succeed for umod pow2 rewrite u64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for umod pow2 rewrite u64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::UMod),
        "Rust output should remove umod after rewrite"
    );
    assert!(
        has_op(&rust_words, Op::BitwiseAnd),
        "Rust output should include mask after rewrite"
    );
    assert!(
        has_const_literal(&rust_words, 7),
        "Rust output should include mask literal 7"
    );
    assert!(
        !has_op(&cpp_words, Op::UMod),
        "C++ output should remove umod after rewrite"
    );
    assert!(
        has_op(&cpp_words, Op::BitwiseAnd),
        "C++ output should include mask after rewrite"
    );
    assert!(
        has_const_literal(&cpp_words, 7),
        "C++ output should include mask literal 7"
    );
}

#[test]
fn cli_opt_block_matches_cpp_sdiv_pow2_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_sdiv_pow2_module();
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
        "opt_block should succeed for sdiv pow2 rewrite"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for sdiv pow2 rewrite"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::SDiv),
        "Rust output should remove sdiv after rewrite"
    );
    assert!(
        has_op(&rust_words, Op::ShiftRightArithmetic),
        "Rust output should include arithmetic shift after rewrite"
    );
    assert!(
        has_const_literal(&rust_words, 3),
        "Rust output should include shift amount 3"
    );
    assert!(
        has_const_literal(&rust_words, 7),
        "Rust output should include bias mask 7"
    );
    assert!(
        !has_op(&cpp_words, Op::SDiv),
        "C++ output should remove sdiv after rewrite"
    );
    assert!(
        has_op(&cpp_words, Op::ShiftRightArithmetic),
        "C++ output should include arithmetic shift after rewrite"
    );
    assert!(
        has_const_literal(&cpp_words, 3),
        "C++ output should include shift amount 3"
    );
    assert!(
        has_const_literal(&cpp_words, 7),
        "C++ output should include bias mask 7"
    );
}

#[test]
fn cli_opt_block_matches_cpp_srem_pow2_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_srem_pow2_module();
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
        "opt_block should succeed for srem pow2 rewrite"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for srem pow2 rewrite"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::SRem),
        "Rust output should remove srem after rewrite"
    );
    assert!(
        !has_op(&cpp_words, Op::SRem),
        "C++ output should remove srem after rewrite"
    );

    let rust_consts = const_literals(&rust_words);
    let cpp_consts = const_literals(&cpp_words);
    assert_eq!(
        rust_consts, cpp_consts,
        "Rust CLI and C++ spirv-opt should agree on rewrite constants"
    );
    assert!(
        rust_consts.contains(&7),
        "rewrite should use mask for pow2 remainder"
    );
}

#[test]
fn cli_opt_block_matches_cpp_sdiv_pow2_rewrite_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_sdiv_pow2_module_u64();
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
        "opt_block should succeed for sdiv pow2 rewrite u64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for sdiv pow2 rewrite u64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::SDiv),
        "Rust output should remove sdiv after rewrite"
    );
    assert!(
        has_op(&rust_words, Op::ShiftRightArithmetic),
        "Rust output should include arithmetic shift after rewrite"
    );
    assert!(
        has_const_literal(&rust_words, 3),
        "Rust output should include shift amount 3"
    );
    assert!(
        has_const_literal(&rust_words, 7),
        "Rust output should include bias mask 7"
    );
    assert!(
        !has_op(&cpp_words, Op::SDiv),
        "C++ output should remove sdiv after rewrite"
    );
    assert!(
        has_op(&cpp_words, Op::ShiftRightArithmetic),
        "C++ output should include arithmetic shift after rewrite"
    );
    assert!(
        has_const_literal(&cpp_words, 3),
        "C++ output should include shift amount 3"
    );
    assert!(
        has_const_literal(&cpp_words, 7),
        "C++ output should include bias mask 7"
    );
}

#[test]
fn cli_opt_block_matches_cpp_srem_pow2_rewrite_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_srem_pow2_module_u64();
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
        "opt_block should succeed for srem pow2 rewrite u64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for srem pow2 rewrite u64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::SRem),
        "Rust output should remove srem after rewrite"
    );
    assert!(
        !has_op(&cpp_words, Op::SRem),
        "C++ output should remove srem after rewrite"
    );

    let rust_consts = const_literals(&rust_words);
    let cpp_consts = const_literals(&cpp_words);
    assert_eq!(
        rust_consts, cpp_consts,
        "Rust CLI and C++ spirv-opt should agree on rewrite constants"
    );
    assert!(
        rust_consts.contains(&7),
        "rewrite should use mask for pow2 remainder"
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
                [rspirv::dr::Operand::LiteralBit64(v)] if (*v >> 32) == 0 => Some(*v as u32),
                [rspirv::dr::Operand::LiteralBit32(lo), rspirv::dr::Operand::LiteralBit32(hi)]
                    if *hi == 0 =>
                {
                    Some(*lo)
                }
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

fn has_const_literal(words: &[u32], value: u32) -> bool {
    let mut loader = Loader::new();
    rspirv::binary::parse_words(words, &mut loader).expect("parse module");
    let module = loader.module();
    let present = module.all_inst_iter().any(|inst| {
        if inst.class.opcode == Op::Constant {
            match inst.operands.as_slice() {
                [rspirv::dr::Operand::LiteralBit32(v)] => *v == value,
                [rspirv::dr::Operand::LiteralBit64(v)] => (*v as u32) == value && (*v >> 32) == 0,
                [rspirv::dr::Operand::LiteralBit32(lo), rspirv::dr::Operand::LiteralBit32(hi)] => {
                    *lo == value && *hi == 0
                }
                _ => false,
            }
        } else {
            false
        }
    });
    present
}

fn const_literals(words: &[u32]) -> std::collections::BTreeSet<u32> {
    let mut loader = Loader::new();
    rspirv::binary::parse_words(words, &mut loader).expect("parse module");
    let module = loader.module();
    module
        .all_inst_iter()
        .filter_map(|inst| {
            if inst.class.opcode == Op::Constant {
                match inst.operands.as_slice() {
                    [rspirv::dr::Operand::LiteralBit32(v)] => Some(*v),
                    [rspirv::dr::Operand::LiteralBit64(v)] if (*v >> 32) == 0 => Some(*v as u32),
                    [rspirv::dr::Operand::LiteralBit32(lo), rspirv::dr::Operand::LiteralBit32(hi)]
                        if *hi == 0 =>
                    {
                        Some(*lo)
                    }
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect()
}

fn module_has_result(words: &[u32], result_id: u32) -> bool {
    let mut loader = Loader::new();
    rspirv::binary::parse_words(words, &mut loader).expect("parse module");
    let module = loader.module();
    let present = module
        .all_inst_iter()
        .any(|inst| inst.result_id == Some(result_id));
    present
}
