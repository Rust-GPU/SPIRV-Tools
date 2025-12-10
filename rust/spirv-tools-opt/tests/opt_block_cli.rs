use std::collections::HashMap;
use std::process::Command;

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Loader};
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::sync::{Mutex, MutexGuard};
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

fn build_mul_identity_module_s32() -> (Vec<u32>, u32) {
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
    let c4 = b.constant_bit32(int, 4);
    let c5 = b.constant_bit32(int, 5);
    let c0 = b.constant_bit32(int, 0);
    let c1 = b.constant_bit32(int, 1);
    let left = b.i_mul(int, None, c4, c1).expect("mul by one s32");
    let right = b.i_mul(int, None, c5, c0).expect("mul by zero s32");
    let sum = b
        .i_add(int, None, left, right)
        .expect("add folded terms s32");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), sum)
}

#[test]
fn cli_opt_block_disable_global_flag_round_trips() {
    let (words, _) = build_sample_module();
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.spv");
    let output_default = dir.path().join("out_default.spv");
    let output_disabled = dir.path().join("out_disabled.spv");
    std::fs::write(&input, words_to_bytes(&words)).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_opt_block"));
    cmd.arg(&input).arg(&output_default);
    let status = cmd.status().unwrap();
    assert!(status.success());

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_opt_block"));
    cmd.arg(&input)
        .arg(&output_disabled)
        .arg("--disable-global");
    let status = cmd.status().unwrap();
    assert!(status.success());

    let a = std::fs::read(&output_default).unwrap();
    let b = std::fs::read(&output_disabled).unwrap();
    assert_eq!(a, b, "disable-global flag should not change output");
}

#[test]
fn cli_opt_block_disable_global_env_round_trips() {
    let (words, _) = build_sample_module();
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.spv");
    let output_default = dir.path().join("out_env_enabled.spv");
    let output_env = dir.path().join("out_env_disabled.spv");
    std::fs::write(&input, words_to_bytes(&words)).unwrap();

    // Baseline with global opt enabled (default).
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_opt_block"));
    cmd.arg(&input).arg(&output_default);
    let status = cmd.status().unwrap();
    assert!(status.success());

    // With global opt disabled via env.
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_opt_block"));
    cmd.arg(&input)
        .arg(&output_env)
        .env("SPIRV_TOOLS_DISABLE_GLOBAL_OPT", "1");
    let status = cmd.status().unwrap();
    assert!(status.success());

    let baseline = std::fs::read(&output_default).unwrap();
    let optimized = std::fs::read(&output_env).unwrap();
    assert_eq!(
        baseline, optimized,
        "env flag should disable only the global path"
    );
}

#[test]
fn cli_opt_block_force_global_env_matches_force_flag() {
    let (words, _) = build_sample_module();
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.spv");
    let output_flag = dir.path().join("out_force_flag.spv");
    let output_env = dir.path().join("out_force_env.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_opt_block"));
    cmd.arg(&input).arg(&output_flag).arg("--force-global");
    let status = cmd.status().unwrap();
    assert!(status.success());

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_opt_block"));
    cmd.arg(&input)
        .arg(&output_env)
        .env("SPIRV_TOOLS_FORCE_GLOBAL_OPT", "1");
    let status = cmd.status().unwrap();
    assert!(status.success());

    let a = std::fs::read(&output_flag).unwrap();
    let b = std::fs::read(&output_env).unwrap();
    assert_eq!(a, b, "force-global flag and env should align");
}

#[test]
fn cli_opt_block_disable_overrides_force() {
    let (words, _) = build_sample_module();
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.spv");
    let output_default = dir.path().join("out_default.spv");
    let output_conflict = dir.path().join("out_conflict.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    // Baseline run.
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_opt_block"));
    cmd.arg(&input).arg(&output_default);
    let status = cmd.status().unwrap();
    assert!(status.success());

    // Force + disable: disable should win.
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_opt_block"));
    cmd.arg(&input)
        .arg(&output_conflict)
        .arg("--force-global")
        .env("SPIRV_TOOLS_DISABLE_GLOBAL_OPT", "1");
    let status = cmd.status().unwrap();
    assert!(status.success());

    let a = std::fs::read(&output_default).unwrap();
    let b = std::fs::read(&output_conflict).unwrap();
    assert_eq!(a, b, "disable-global env should override force-global flag");
}

fn build_mul_identity_module_s64() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(64, 1);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let c4 = b.constant_bit64(int, 4);
    let c5 = b.constant_bit64(int, 5);
    let c0 = b.constant_bit64(int, 0);
    let c1 = b.constant_bit64(int, 1);
    let left = b.i_mul(int, None, c4, c1).expect("mul by one s64");
    let right = b.i_mul(int, None, c5, c0).expect("mul by zero s64");
    let sum = b
        .i_add(int, None, left, right)
        .expect("add folded terms s64");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), sum)
}

fn build_mul_identity_module_u64() -> (Vec<u32>, u32) {
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
    let c4 = b.constant_bit64(int, 4);
    let c5 = b.constant_bit64(int, 5);
    let c0 = b.constant_bit64(int, 0);
    let c1 = b.constant_bit64(int, 1);
    let left = b.i_mul(int, None, c4, c1).expect("mul by one u64");
    let right = b.i_mul(int, None, c5, c0).expect("mul by zero u64");
    let sum = b
        .i_add(int, None, left, right)
        .expect("add folded terms u64");
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

fn build_band_complement_u64_module() -> (Vec<u32>, (u32, u32)) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(64, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).expect("param");
    let _ = b.begin_block(None).unwrap();
    let not_param = b.not(int, None, param).expect("not param");
    let band = b
        .bitwise_and(int, None, param, not_param)
        .expect("bitwise and");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), (band, int))
}

fn build_band_complement_u32_module() -> (Vec<u32>, (u32, u32)) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).expect("param");
    let _ = b.begin_block(None).unwrap();
    let not_param = b.not(int, None, param).expect("not param");
    let band = b
        .bitwise_and(int, None, param, not_param)
        .expect("bitwise and");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), (band, int))
}

fn build_band_all_ones_u32_module() -> (Vec<u32>, u32, u32) {
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
    let x = b.constant_bit32(int, 0x1234_5678);
    let mask = b.constant_bit32(int, u32::MAX);
    let band = b.bitwise_and(int, None, x, mask).expect("band");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), band, int)
}

fn build_band_all_ones_u64_module() -> (Vec<u32>, u32, u32) {
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
    let x = b.constant_bit64(int, 0x1234_5678_9ABC_DEF0);
    let mask = b.constant_bit64(int, u64::MAX);
    let band = b.bitwise_and(int, None, x, mask).expect("band");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), band, int)
}

fn build_bor_zero_u32_module() -> (Vec<u32>, u32, u32) {
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
    let x = b.constant_bit32(int, 0x89AB_CDEF);
    let zero = b.constant_bit32(int, 0);
    let bor = b.bitwise_or(int, None, x, zero).expect("bor");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), bor, int)
}

fn build_bor_zero_u64_module() -> (Vec<u32>, u32, u32) {
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
    let x = b.constant_bit64(int, 0x1234_5678_9ABC_DEF0);
    let zero = b.constant_bit64(int, 0);
    let bor = b.bitwise_or(int, None, x, zero).expect("bor");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), bor, int)
}

fn build_bxor_self_u32_module() -> (Vec<u32>, u32, u32) {
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
    let x = b.constant_bit32(int, 0xCAFEBABE);
    let bxor = b.bitwise_xor(int, None, x, x).expect("bxor self");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), bxor, int)
}

fn build_bxor_self_u64_module() -> (Vec<u32>, u32, u32) {
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
    let x = b.constant_bit64(int, 0x1234_5678_9ABC_DEF0);
    let bxor = b.bitwise_xor(int, None, x, x).expect("bxor self");
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), bxor, int)
}

fn build_band_absorb_or_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).expect("param x");
    let y = b.constant_bit32(int, 0xFFFF0000);
    let _ = b.begin_block(None).unwrap();
    let bor = b.bitwise_or(int, None, x, y).expect("bor");
    let _band = b.bitwise_and(int, None, x, bor).expect("band");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn build_bor_absorb_and_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).expect("param x");
    let y = b.constant_bit32(int, 0x0FFF0FFF);
    let _ = b.begin_block(None).unwrap();
    let band = b.bitwise_and(int, None, x, y).expect("band");
    let _bor = b.bitwise_or(int, None, x, band).expect("bor");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.module().assemble()
}

fn arith_signature(words: &[u32]) -> Vec<(Op, Vec<String>)> {
    let mut loader = Loader::new();
    rspirv::binary::parse_words(words, &mut loader).expect("parse module");
    let module = loader.module();
    let is_arith = |opcode: Op| {
        matches!(
            opcode,
            Op::Constant
                | Op::IAdd
                | Op::IMul
                | Op::ISub
                | Op::BitwiseOr
                | Op::BitwiseXor
                | Op::BitwiseAnd
                | Op::Not
                | Op::SNegate
                | Op::SDiv
                | Op::UDiv
                | Op::SRem
                | Op::UMod
                | Op::ShiftLeftLogical
                | Op::ShiftRightLogical
                | Op::ShiftRightArithmetic
        )
    };
    let mut sig: Vec<_> = module
        .types_global_values
        .iter()
        .chain(
            module
                .functions
                .iter()
                .flat_map(|func| func.blocks.iter().flat_map(|blk| blk.instructions.iter())),
        )
        .filter(|inst| is_arith(inst.class.opcode))
        .map(|inst| {
            (
                inst.class.opcode,
                inst.operands
                    .iter()
                    .map(|op| format!("{op:?}"))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    sig.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    sig
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

fn build_signed_div_rem_identity_module_u64() -> (Vec<u32>, (u32, u32)) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(64, 1);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let c42 = b.constant_bit64(int, 42);
    let c1 = b.constant_bit64(int, 1);
    let div = b.s_div(int, None, c42, c1).expect("sdiv by one s64");
    let rem = b.s_rem(int, None, c42, c1).expect("srem by one s64");
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

fn build_mul_pow2_module_s32() -> (Vec<u32>, (u32, u32)) {
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
    let c3 = b.constant_bit32(int, 3);
    let mul = b.i_mul(int, None, param, c8).expect("mul pow2 s32");
    b.ret_value(mul).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), (mul, c3))
}

fn build_mul_pow2_module_s64() -> (Vec<u32>, (u32, u32)) {
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
    let c3 = b.constant_bit64(int, 3);
    let mul = b.i_mul(int, None, param, c8).expect("mul pow2 s64");
    b.ret_value(mul).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), (mul, c3))
}

fn build_mul_pow2_module_u64() -> (Vec<u32>, (u32, u32)) {
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
    let c3 = b.constant_bit64(int, 3);
    let mul = b.i_mul(int, None, param, c8).expect("mul pow2 u64");
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

fn build_mul_neg_one_module_s64() -> (Vec<u32>, u32) {
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
    let c_neg_one = b.constant_bit64(int, u64::MAX);
    let mul = b.i_mul(int, None, param, c_neg_one).expect("mul -1 s64");
    b.ret_value(mul).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), mul)
}

fn build_mul_neg_one_module_u64() -> (Vec<u32>, u32) {
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
    let c_neg_one = b.constant_bit64(int, u64::MAX);
    let mul = b.i_mul(int, None, param, c_neg_one).expect("mul -1 u64");
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

fn build_dead_arith_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(int, vec![]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let c4 = b.constant_bit32(int, 4);
    let c5 = b.constant_bit32(int, 5);
    let dead_add = b.i_add(int, None, c4, c5).expect("dead add");
    let live_add = b.i_add(int, None, c4, c4).expect("live add");
    b.ret_value(live_add).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), dead_add, c5)
}

fn build_two_block_arith_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(int, vec![]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();

    let _ = b.begin_block(None).unwrap();
    let c4 = b.constant_bit32(int, 4);
    let c5 = b.constant_bit32(int, 5);
    let add = b.i_add(int, None, c4, c5).expect("add in block1");
    let block2 = b.id();
    b.branch(block2).unwrap();

    b.begin_block(Some(block2)).unwrap();
    let c2 = b.constant_bit32(int, 2);
    let sub = b.i_sub(int, None, add, c2).expect("sub in block2");
    b.ret_value(sub).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), sub)
}

fn build_two_block_affine_cancel_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(int, vec![int, int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).unwrap();
    let y = b.function_parameter(int).unwrap();

    let _ = b.begin_block(None).unwrap();
    let add = b.i_add(int, None, x, y).expect("add");
    let block1 = b.id();
    b.branch(block1).unwrap();

    b.begin_block(Some(block1)).unwrap();
    let sub = b.i_sub(int, None, add, y).expect("sub");
    b.ret_value(sub).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), sub, x)
}

fn build_loop_invariant_mul_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();

    let _ = b.begin_block(None).unwrap();
    let header = b.id();
    b.branch(header).unwrap();

    b.begin_block(Some(header)).unwrap();
    let body = b.id();
    b.branch(body).unwrap();

    b.begin_block(Some(body)).unwrap();
    let mul = b.i_mul(int, None, param, param).expect("mul");
    b.branch(header).unwrap();

    let exit = b.id();
    b.begin_block(Some(exit)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), mul)
}

fn build_cse_across_blocks_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(int, vec![int, int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).unwrap();
    let y = b.function_parameter(int).unwrap();

    let entry = b.begin_block(None).unwrap();
    let add0 = b.i_add(int, None, x, y).expect("first add");
    let next = b.id();
    b.branch(next).unwrap();

    b.begin_block(Some(next)).unwrap();
    let add1 = b.i_add(int, None, x, y).expect("second add");
    b.ret_value(add1).unwrap();
    b.end_function().unwrap();
    let _ = entry;
    (b.module().assemble(), add0, add1)
}

fn build_copy_chain_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).unwrap();

    let _ = b.begin_block(None).unwrap();
    let copy1 = b.copy_object(int, None, x).expect("copy1");
    let copy2 = b.copy_object(int, None, copy1).expect("copy2");
    b.ret_value(copy2).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), copy1, copy2)
}

fn build_cross_block_factor_module() -> (Vec<u32>, u32, u32) {
    // block0: m2 = x * 2
    // block1: m3 = x * 3; add = m2 + m3
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(int, vec![int]);
    let _func = b
        .begin_function(int, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).unwrap();

    let _ = b.begin_block(None).unwrap();
    let c2 = b.constant_bit32(int, 2);
    let m2 = b.i_mul(int, None, x, c2).expect("mul by 2");
    let block1 = b.id();
    b.branch(block1).unwrap();

    b.begin_block(Some(block1)).unwrap();
    let c3 = b.constant_bit32(int, 3);
    let m3 = b.i_mul(int, None, x, c3).expect("mul by 3");
    let add = b.i_add(int, None, m2, m3).expect("add muls");
    b.ret_value(add).unwrap();
    b.end_function().unwrap();
    (b.module().assemble(), add, x)
}

#[test]
fn cli_opt_block_folds_arithmetic() {
    let _guard = env_guard();
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
    let _guard = env_guard();
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
    let _guard = env_guard();
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
    let _guard = env_guard();
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
fn cli_opt_block_folds_complement_with_width_awareness() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (band_id, int_ty)) = build_band_complement_u64_module();
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
    assert!(status.success(), "opt_block should fold complement");

    let optimized_bytes = std::fs::read(&output).expect("read output");
    let optimized_words = bytes_to_words(&optimized_bytes);
    let mut loader = Loader::new();
    rspirv::binary::parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(band_id)
            && inst.result_type == Some(int_ty)
    });
    assert!(folded.is_some(), "band result should fold to a constant");
    assert_eq!(
        folded.unwrap().operands,
        vec![rspirv::dr::Operand::LiteralBit64(0)],
        "folded constant should use 64-bit encoding"
    );
    let has_band = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::BitwiseAnd && inst.result_id == Some(band_id));
    assert!(!has_band, "bitwise and should be removed after folding");
}

#[test]
fn cli_opt_block_matches_cpp_band_complement_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, _) = build_band_complement_u64_module();
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
    assert!(rust_status.success(), "opt_block should succeed");

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(cpp_status.success(), "spirv-opt should succeed");

    let rust_sig = arith_signature(&bytes_to_words(&std::fs::read(&rust_output).unwrap()));
    let cpp_sig = arith_signature(&bytes_to_words(&std::fs::read(&cpp_output).unwrap()));
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ mismatch for band complement u64 fold"
    );
}

#[test]
fn cli_opt_block_matches_cpp_band_complement_u32() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (band_id, int_ty)) = build_band_complement_u32_module();
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
    assert!(rust_status.success(), "opt_block should succeed");

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(cpp_status.success(), "spirv-opt should succeed");

    let rust_sig = arith_signature(&bytes_to_words(&std::fs::read(&rust_output).unwrap()));
    let cpp_sig = arith_signature(&bytes_to_words(&std::fs::read(&cpp_output).unwrap()));
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ mismatch for band complement u32 fold"
    );

    let mut loader = Loader::new();
    rspirv::binary::parse_words(
        &bytes_to_words(&std::fs::read(&rust_output).unwrap()),
        &mut loader,
    )
    .expect("parse optimized");
    let module = loader.module();
    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(band_id)
            && inst.result_type == Some(int_ty)
    });
    assert!(
        folded.is_some(),
        "Rust output should fold complement to constant"
    );
}

#[test]
fn cli_opt_block_matches_cpp_band_all_ones_u32() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, band_id, int_ty) = build_band_all_ones_u32_module();
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
    assert!(rust_status.success(), "opt_block should succeed");

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(cpp_status.success(), "spirv-opt should succeed");

    let rust_sig = arith_signature(&bytes_to_words(&std::fs::read(&rust_output).unwrap()));
    let cpp_sig = arith_signature(&bytes_to_words(&std::fs::read(&cpp_output).unwrap()));
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ mismatch for band all-ones u32 fold"
    );

    let mut loader = Loader::new();
    rspirv::binary::parse_words(
        &bytes_to_words(&std::fs::read(&rust_output).unwrap()),
        &mut loader,
    )
    .expect("parse optimized");
    let module = loader.module();
    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(band_id)
            && inst.result_type == Some(int_ty)
    });
    assert!(
        folded.is_some(),
        "Rust output should fold band with all ones to the input"
    );
}

#[test]
fn cli_opt_block_matches_cpp_band_all_ones_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, band_id, int_ty) = build_band_all_ones_u64_module();
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
    assert!(rust_status.success(), "opt_block should succeed");

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(cpp_status.success(), "spirv-opt should succeed");

    let rust_sig = arith_signature(&bytes_to_words(&std::fs::read(&rust_output).unwrap()));
    let cpp_sig = arith_signature(&bytes_to_words(&std::fs::read(&cpp_output).unwrap()));
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ mismatch for band all-ones u64 fold"
    );

    let mut loader = Loader::new();
    rspirv::binary::parse_words(
        &bytes_to_words(&std::fs::read(&rust_output).unwrap()),
        &mut loader,
    )
    .expect("parse optimized");
    let module = loader.module();
    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(band_id)
            && inst.result_type == Some(int_ty)
    });
    assert!(
        folded.is_some(),
        "Rust output should fold band with all ones to the input"
    );
}

#[test]
fn cli_opt_block_matches_cpp_bor_zero_u32() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, bor_id, int_ty) = build_bor_zero_u32_module();
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
    assert!(rust_status.success(), "opt_block should succeed");

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(cpp_status.success(), "spirv-opt should succeed");

    let rust_sig = arith_signature(&bytes_to_words(&std::fs::read(&rust_output).unwrap()));
    let cpp_sig = arith_signature(&bytes_to_words(&std::fs::read(&cpp_output).unwrap()));
    assert_eq!(rust_sig, cpp_sig, "Rust vs C++ mismatch for bor zero u32");

    let mut loader = Loader::new();
    rspirv::binary::parse_words(
        &bytes_to_words(&std::fs::read(&rust_output).unwrap()),
        &mut loader,
    )
    .expect("parse optimized");
    let module = loader.module();
    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(bor_id)
            && inst.result_type == Some(int_ty)
    });
    assert!(
        folded.is_some(),
        "Rust output should fold bor zero to operand"
    );
}

#[test]
fn cli_opt_block_matches_cpp_bor_zero_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, bor_id, int_ty) = build_bor_zero_u64_module();
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
    assert!(rust_status.success(), "opt_block should succeed");

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(cpp_status.success(), "spirv-opt should succeed");

    let rust_sig = arith_signature(&bytes_to_words(&std::fs::read(&rust_output).unwrap()));
    let cpp_sig = arith_signature(&bytes_to_words(&std::fs::read(&cpp_output).unwrap()));
    assert_eq!(rust_sig, cpp_sig, "Rust vs C++ mismatch for bor zero u64");

    let mut loader = Loader::new();
    rspirv::binary::parse_words(
        &bytes_to_words(&std::fs::read(&rust_output).unwrap()),
        &mut loader,
    )
    .expect("parse optimized");
    let module = loader.module();
    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(bor_id)
            && inst.result_type == Some(int_ty)
    });
    assert!(
        folded.is_some(),
        "Rust output should fold bor zero to operand"
    );
}

#[test]
fn cli_opt_block_matches_cpp_bxor_self_u32() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, bxor_id, int_ty) = build_bxor_self_u32_module();
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
    assert!(rust_status.success(), "opt_block should succeed");

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(cpp_status.success(), "spirv-opt should succeed");

    let rust_sig = arith_signature(&bytes_to_words(&std::fs::read(&rust_output).unwrap()));
    let cpp_sig = arith_signature(&bytes_to_words(&std::fs::read(&cpp_output).unwrap()));
    assert_eq!(rust_sig, cpp_sig, "Rust vs C++ mismatch for bxor self u32");

    let mut loader = Loader::new();
    rspirv::binary::parse_words(
        &bytes_to_words(&std::fs::read(&rust_output).unwrap()),
        &mut loader,
    )
    .expect("parse optimized");
    let module = loader.module();
    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(bxor_id)
            && inst.result_type == Some(int_ty)
    });
    assert!(
        folded.is_some(),
        "Rust output should fold bxor self to zero"
    );
}

fn has_bitwise_ops(words: &[u32]) -> bool {
    let mut loader = Loader::new();
    rspirv::binary::parse_words(words, &mut loader).expect("parse module");
    loader.module().all_inst_iter().any(|inst| {
        matches!(
            inst.class.opcode,
            Op::BitwiseAnd | Op::BitwiseOr | Op::BitwiseXor | Op::Not
        )
    })
}

fn build_split_y_absorption_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).expect("x");
    let y = b.function_parameter(int).expect("y");
    let _ = b.begin_block(None).unwrap();
    let not_x = b.not(int, None, x).expect("not x");
    let band1 = b.bitwise_and(int, None, x, y).expect("x & y");
    let band2 = b.bitwise_and(int, None, not_x, y).expect("~x & y");
    let _ = b
        .bitwise_or(int, None, band1, band2)
        .expect("(x & y) | (~x & y)");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", &[x, y]);
    b.module().assemble()
}

fn build_split_x_absorption_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).expect("x");
    let y = b.function_parameter(int).expect("y");
    let _ = b.begin_block(None).unwrap();
    let not_y = b.not(int, None, y).expect("not y");
    let band1 = b.bitwise_and(int, None, x, y).expect("x & y");
    let band2 = b.bitwise_and(int, None, x, not_y).expect("x & ~y");
    let _ = b
        .bitwise_or(int, None, band1, band2)
        .expect("(x & y) | (x & ~y)");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", &[x, y]);
    b.module().assemble()
}

#[test]
fn cli_opt_block_absorbs_band_over_bor() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_band_absorb_or_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let rust_output = dir.path().join("rust_output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let rust_status = Command::new(exe)
        .arg(&input)
        .arg(&rust_output)
        .status()
        .expect("run opt_block");
    assert!(rust_status.success(), "opt_block should succeed");

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).unwrap());
    assert!(
        !has_bitwise_ops(&rust_words),
        "bitwise ops should be eliminated after absorption rewrite"
    );
}

#[test]
fn cli_opt_block_absorbs_bor_over_band() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_bor_absorb_and_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let rust_output = dir.path().join("rust_output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let rust_status = Command::new(exe)
        .arg(&input)
        .arg(&rust_output)
        .status()
        .expect("run opt_block");
    assert!(rust_status.success(), "opt_block should succeed");

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).unwrap());
    assert!(
        !has_bitwise_ops(&rust_words),
        "bitwise ops should be eliminated after absorption rewrite"
    );
}

#[test]
fn cli_opt_block_matches_cpp_bxor_self_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, bxor_id, int_ty) = build_bxor_self_u64_module();
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
    assert!(rust_status.success(), "opt_block should succeed");

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(cpp_status.success(), "spirv-opt should succeed");

    let rust_sig = arith_signature(&bytes_to_words(&std::fs::read(&rust_output).unwrap()));
    let cpp_sig = arith_signature(&bytes_to_words(&std::fs::read(&cpp_output).unwrap()));
    assert_eq!(rust_sig, cpp_sig, "Rust vs C++ mismatch for bxor self u64");

    let mut loader = Loader::new();
    rspirv::binary::parse_words(
        &bytes_to_words(&std::fs::read(&rust_output).unwrap()),
        &mut loader,
    )
    .expect("parse optimized");
    let module = loader.module();
    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(bxor_id)
            && inst.result_type == Some(int_ty)
    });
    assert!(
        folded.is_some(),
        "Rust output should fold bxor self to zero"
    );
}

#[test]
fn cli_opt_block_matches_cpp_mul_identities() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
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
fn cli_opt_block_matches_cpp_mul_identities_s32() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, sum_id) = build_mul_identity_module_s32();
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
        "opt_block should succeed for mul identities s32"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for mul identities s32"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    let rust_const = find_const_value(&rust_words, sum_id);
    let cpp_const = find_const_value(&cpp_words, sum_id);
    assert_eq!(
        rust_const, cpp_const,
        "Rust CLI and C++ spirv-opt should fold mul identities the same way (s32)"
    );
    assert_eq!(rust_const, Some(4), "mul identities should fold to 4 (s32)");
    assert_eq!(cpp_const, Some(4), "mul identities should fold to 4 (s32)");
    assert!(
        !has_op(&rust_words, Op::IMul) && !has_op(&rust_words, Op::IAdd),
        "Rust output should remove mul/add after folding (s32)"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul) && !has_op(&cpp_words, Op::IAdd),
        "C++ output should remove mul/add after folding (s32)"
    );
}

#[test]
fn cli_opt_block_matches_cpp_mul_identities_s64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, sum_id) = build_mul_identity_module_s64();
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
        "opt_block should succeed for mul identities s64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for mul identities s64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::IMul) && !has_op(&rust_words, Op::IAdd),
        "Rust output should remove mul/add after folding (s64)"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul) && !has_op(&cpp_words, Op::IAdd),
        "C++ output should remove mul/add after folding (s64)"
    );

    let rust_const = find_const_value(&rust_words, sum_id);
    let cpp_const = find_const_value(&cpp_words, sum_id);
    assert_eq!(
        rust_const, cpp_const,
        "Rust CLI and C++ spirv-opt should fold mul identities the same way (s64)"
    );
    assert_eq!(rust_const, Some(4), "mul identities should fold to 4 (s64)");
    assert_eq!(cpp_const, Some(4), "mul identities should fold to 4 (s64)");
}

#[test]
fn cli_opt_block_matches_cpp_mul_identities_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, sum_id) = build_mul_identity_module_u64();
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
        "opt_block should succeed for mul identities u64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for mul identities u64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::IMul) && !has_op(&rust_words, Op::IAdd),
        "Rust output should remove mul/add after folding (u64)"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul) && !has_op(&cpp_words, Op::IAdd),
        "C++ output should remove mul/add after folding (u64)"
    );

    let rust_const = find_const_value(&rust_words, sum_id);
    let cpp_const = find_const_value(&cpp_words, sum_id);
    assert_eq!(
        rust_const, cpp_const,
        "Rust CLI and C++ spirv-opt should fold mul identities the same way (u64)"
    );
    assert_eq!(rust_const, Some(4), "mul identities should fold to 4 (u64)");
    assert_eq!(cpp_const, Some(4), "mul identities should fold to 4 (u64)");
}

#[test]
fn cli_opt_block_matches_cpp_div_rem_identities() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
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
    let _guard = env_guard();
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
    let _guard = env_guard();
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
fn cli_opt_block_matches_cpp_signed_div_rem_identities_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (div_id, rem_id)) = build_signed_div_rem_identity_module_u64();
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
        "opt_block should succeed for signed div/rem identities u64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for signed div/rem identity folding u64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    let rust_div = find_const_value(&rust_words, div_id);
    let rust_rem = find_const_value(&rust_words, rem_id);
    let cpp_div = find_const_value(&cpp_words, div_id);
    let cpp_rem = find_const_value(&cpp_words, rem_id);

    assert_eq!(
        rust_div, cpp_div,
        "Rust CLI and C++ spirv-opt should fold signed div-by-one the same way (s64)"
    );
    assert_eq!(
        rust_div,
        Some(42),
        "signed div-by-one should fold to the original value (s64)"
    );
    assert_eq!(
        rust_rem, cpp_rem,
        "Rust CLI and C++ spirv-opt should fold signed rem-by-one the same way (s64)"
    );
    assert_eq!(
        rust_rem,
        Some(0),
        "signed rem-by-one should fold to zero (s64)"
    );
    assert!(
        !has_op(&rust_words, Op::SDiv) && !has_op(&rust_words, Op::SRem),
        "Rust output should remove sdiv/srem after folding (s64)"
    );
    assert!(
        !has_op(&cpp_words, Op::SDiv) && !has_op(&cpp_words, Op::SRem),
        "C++ output should remove sdiv/srem after folding (s64)"
    );
}

#[test]
fn cli_opt_block_matches_cpp_mul_pow2_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
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
fn cli_opt_block_matches_cpp_mul_pow2_rewrite_s32() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (_mul_id, shift_const_id)) = build_mul_pow2_module_s32();
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
        "opt_block should succeed for pow2 rewrite s32"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for pow2 rewrite s32"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::IMul),
        "Rust output should remove mul after rewrite (s32)"
    );
    assert!(
        has_op(&rust_words, Op::ShiftLeftLogical),
        "Rust output should include shift after rewrite (s32)"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul),
        "C++ output should remove mul after rewrite (s32)"
    );
    assert!(
        has_op(&cpp_words, Op::ShiftLeftLogical),
        "C++ output should include shift after rewrite (s32)"
    );

    let rust_shift = find_const_value(&rust_words, shift_const_id);
    let cpp_shift = find_const_value(&cpp_words, shift_const_id);
    assert_eq!(
        rust_shift, cpp_shift,
        "Rust CLI and C++ spirv-opt should agree on shift amount (s32)"
    );
    assert_eq!(rust_shift, Some(3), "pow2 rewrite should shift by 3 (s32)");
}

#[test]
fn cli_opt_block_matches_cpp_mul_pow2_rewrite_s64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (_mul_id, shift_const_id)) = build_mul_pow2_module_s64();
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
        "opt_block should succeed for pow2 rewrite s64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for pow2 rewrite s64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::IMul),
        "Rust output should remove mul after rewrite (s64)"
    );
    assert!(
        has_op(&rust_words, Op::ShiftLeftLogical),
        "Rust output should include shift after rewrite (s64)"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul),
        "C++ output should remove mul after rewrite (s64)"
    );
    assert!(
        has_op(&cpp_words, Op::ShiftLeftLogical),
        "C++ output should include shift after rewrite (s64)"
    );

    let rust_shift = find_const_value(&rust_words, shift_const_id);
    let cpp_shift = find_const_value(&cpp_words, shift_const_id);
    assert_eq!(
        rust_shift, cpp_shift,
        "Rust CLI and C++ spirv-opt should agree on shift amount for s64"
    );
    assert_eq!(
        rust_shift,
        Some(3),
        "pow2 rewrite should shift by 3 for signed 64-bit mul"
    );
}

#[test]
fn cli_opt_block_matches_cpp_mul_neg_one_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
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
fn cli_opt_block_matches_cpp_mul_neg_one_rewrite_s64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, mul_id) = build_mul_neg_one_module_s64();
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
        "opt_block should succeed for mul-by-neg-one rewrite s64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for mul-by-neg-one rewrite s64"
    );

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).expect("read rust output"));
    let cpp_words = bytes_to_words(&std::fs::read(&cpp_output).expect("read cpp output"));

    assert!(
        !has_op(&rust_words, Op::IMul),
        "Rust output should remove mul after rewrite (s64)"
    );
    assert!(
        has_op(&rust_words, Op::SNegate),
        "Rust output should include negate after rewrite (s64)"
    );
    assert!(
        !has_op(&cpp_words, Op::IMul),
        "C++ output should remove mul after rewrite (s64)"
    );
    assert!(
        has_op(&cpp_words, Op::SNegate),
        "C++ output should include negate after rewrite (s64)"
    );

    assert!(
        module_has_result(&rust_words, mul_id),
        "Rust output should keep the result id alive after rewrite (s64)"
    );
    assert!(
        module_has_result(&cpp_words, mul_id),
        "C++ output should keep the result id alive after rewrite (s64)"
    );
}

#[test]
fn cli_opt_block_matches_cpp_mul_neg_one_rewrite_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, mul_id) = build_mul_neg_one_module_u64();
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
        "opt_block should succeed for mul-by-neg-one rewrite u64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for mul-by-neg-one rewrite u64"
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
    let _guard = env_guard();
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
fn cli_opt_block_matches_cpp_mul_pow2_rewrite_u64() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, (_mul_id, shift_const_id)) = build_mul_pow2_module_u64();
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
        "opt_block should succeed for pow2 rewrite u64"
    );

    let cpp_status = Command::new(&cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&cpp_output)
        .status()
        .expect("run C++ spirv-opt");
    assert!(
        cpp_status.success(),
        "spirv-opt should succeed for pow2 rewrite u64"
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

    let rust_consts = const_literals(&rust_words);
    let cpp_consts = const_literals(&cpp_words);
    assert_eq!(
        rust_consts, cpp_consts,
        "Rust CLI and C++ spirv-opt should agree on rewrite constants"
    );
    assert!(
        rust_consts.contains(&3),
        "rewrite should encode the shift amount"
    );
    assert_eq!(
        find_const_value(&rust_words, shift_const_id),
        Some(3),
        "shift amount should be 3 in rust output"
    );
    assert_eq!(
        find_const_value(&cpp_words, shift_const_id),
        Some(3),
        "shift amount should be 3 in cpp output"
    );
}

#[test]
fn cli_opt_block_matches_cpp_umod_pow2_rewrite() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let _guard = env_guard();
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
    let _guard = env_guard();
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
    let _guard = env_guard();
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
    let _guard = env_guard();
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
    let _guard = env_guard();
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
    let _guard = env_guard();
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
    let _guard = env_guard();
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

fn env_guard() -> MutexGuard<'static, ()> {
    ENV_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

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

/// Rust-only improvement: absorb split y term (x & y) | (~x & y) into y.
#[test]
fn cli_opt_block_absorbs_split_y_term() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_split_y_absorption_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let rust_output = dir.path().join("rust_output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let rust_status = Command::new(exe)
        .arg(&input)
        .arg(&rust_output)
        .status()
        .expect("run opt_block");
    assert!(rust_status.success(), "opt_block should succeed");

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).unwrap());
    assert!(
        !has_bitwise_ops(&rust_words),
        "Rust optimizer should fold split y term and drop bitwise ops (C++ leaves expanded)"
    );
}

/// Rust-only improvement: absorb split x term (x & y) | (x & ~y) into x.
#[test]
fn cli_opt_block_absorbs_split_x_term() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let words = build_split_x_absorption_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let rust_output = dir.path().join("rust_output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let rust_status = Command::new(exe)
        .arg(&input)
        .arg(&rust_output)
        .status()
        .expect("run opt_block");
    assert!(rust_status.success(), "opt_block should succeed");

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).unwrap());
    assert!(
        !has_bitwise_ops(&rust_words),
        "Rust optimizer should fold split x term and drop bitwise ops (C++ leaves expanded)"
    );
}

#[test]
fn cli_opt_block_prunes_dead_arith_and_consts() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, dead_add_id, dead_const_id) = build_dead_arith_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let rust_output = dir.path().join("rust_output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let rust_status = Command::new(exe)
        .arg(&input)
        .arg(&rust_output)
        .status()
        .expect("run opt_block");
    assert!(rust_status.success(), "opt_block should succeed");

    let rust_words = bytes_to_words(&std::fs::read(&rust_output).unwrap());
    assert!(
        !module_has_result(&rust_words, dead_add_id),
        "dead arithmetic instruction should be removed"
    );
    assert!(
        !has_const_literal(&rust_words, 5),
        "unused constant should be removed after DCE"
    );
    assert!(
        !module_has_result(&rust_words, dead_const_id),
        "dead constant should be removed alongside dead arithmetic"
    );
}

#[test]
fn cli_opt_block_folds_across_blocks() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, sub_id) = build_two_block_arith_module();
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

    let folded = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
    });
    assert!(
        folded.is_some(),
        "folded subtraction across blocks should become constant 7"
    );
    assert!(
        !module
            .all_inst_iter()
            .any(|inst| inst.class.opcode == Op::ISub && inst.result_id == Some(sub_id)),
        "subtraction should be removed across blocks"
    );
}

#[test]
fn cli_opt_block_cancels_affine_across_blocks() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, sub_id, x_id) = build_two_block_affine_cancel_module();
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

    let has_copy = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::CopyObject
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::IdRef(x_id)]
    });
    let has_sub = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::ISub && inst.result_id == Some(sub_id));
    if !has_copy {
        eprintln!(
            "affine module instructions: {:?}",
            module
                .all_inst_iter()
                .map(|inst| (inst.class.opcode, inst.result_id, inst.operands.clone()))
                .collect::<Vec<_>>()
        );
    }
    assert!(has_copy, "affine cancel should reduce to a copy from x");
    assert!(!has_sub, "subtraction should be eliminated across blocks");
}

#[test]
fn cli_opt_block_affine_disable_global_keeps_sub() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, sub_id, _) = build_two_block_affine_cancel_module();
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let output = dir.path().join("output.spv");
    std::fs::write(&input, words_to_bytes(&words)).expect("write input");

    let exe = env!("CARGO_BIN_EXE_opt_block");
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .arg("--disable-global")
        .status()
        .expect("run opt_block");
    assert!(status.success(), "opt_block should exit successfully");

    let optimized_bytes = std::fs::read(&output).expect("read output");
    let optimized_words = bytes_to_words(&optimized_bytes);
    let mut loader = Loader::new();
    rspirv::binary::parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let has_sub = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::ISub && inst.result_id == Some(sub_id));
    assert!(
        has_sub,
        "disabling global optimization should keep the original subtraction"
    );
}

#[test]
fn cli_opt_block_hoists_loop_invariant_mul() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, mul_id) = build_loop_invariant_mul_module();
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
    let func = module
        .functions
        .first()
        .expect("function present after optimization");
    let mut block_of_mul = None;
    for (idx, block) in func.blocks.iter().enumerate() {
        if block
            .instructions
            .iter()
            .any(|inst| inst.result_id == Some(mul_id))
        {
            block_of_mul = Some(idx);
            break;
        }
    }
    if block_of_mul != Some(0) {
        eprintln!(
            "mul placement blocks: {:?}",
            func.blocks
                .iter()
                .map(|blk| {
                    blk.instructions
                        .iter()
                        .map(|inst| (inst.class.opcode, inst.result_id, inst.operands.clone()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(
        block_of_mul,
        Some(0),
        "loop-invariant multiply should be hoisted to the entry block"
    );
}

#[test]
fn cli_opt_block_dedupes_common_expr_across_blocks() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, _first_add, second_add) = build_cse_across_blocks_module();
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

    let has_second_add = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::IAdd && inst.result_id == Some(second_add));
    assert!(
        !has_second_add,
        "duplicate add should be removed; module={:?}",
        module
            .all_inst_iter()
            .map(|inst| (inst.class.opcode, inst.result_id, inst.operands.clone()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn cli_opt_block_collapses_copy_chains() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, _copy1, _copy2) = build_copy_chain_module();
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

    let mut copies: HashMap<u32, u32> = HashMap::new();
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::CopyObject {
            if let (Some(dst), Some(src)) = (
                inst.result_id,
                inst.operands.get(0).and_then(|op| op.id_ref_any()),
            ) {
                copies.insert(dst, src);
            }
        }
    }
    for (dst, src) in &copies {
        assert_ne!(dst, src, "copy chains must not self-loop");
        if let Some(src_src) = copies.get(src) {
            assert_eq!(
                src_src, src,
                "nested copy should have been collapsed to a root operand"
            );
        }
    }
}

#[test]
fn cli_opt_block_factors_across_blocks() {
    let _guard = env_guard();
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    let (words, add_id, x_id) = build_cross_block_factor_module();
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

    let const_five = module.all_inst_iter().find_map(|inst| {
        (inst.class.opcode == Op::Constant
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)])
        .then_some(inst.result_id)
        .flatten()
    });
    let has_mul = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(add_id)
            && inst
                .operands
                .iter()
                .any(|op| matches!(op, rspirv::dr::Operand::IdRef(id) if *id == x_id))
            && const_five.map_or(false, |cid| {
                inst.operands
                    .iter()
                    .any(|op| matches!(op, rspirv::dr::Operand::IdRef(id) if *id == cid))
            })
    });
    let has_add = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::IAdd && inst.result_id == Some(add_id));
    if !has_mul {
        eprintln!(
            "factoring module instructions: {:?}",
            module
                .all_inst_iter()
                .map(|inst| (inst.class.opcode, inst.result_id, inst.operands.clone()))
                .collect::<Vec<_>>()
        );
    }
    assert!(has_mul, "factored add should become a single multiply by 5");
    assert!(!has_add, "add should be eliminated after factoring");
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
