use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::{Builder, Loader};
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::env;
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;

fn cpp_opt_bin() -> Option<String> {
    if let Ok(path) = env::var("SPIRV_CPP_OPT") {
        if !path.is_empty() {
            return Some(path);
        }
    }
    env::var_os("PATH")
        .and_then(|paths| {
            env::split_paths(&paths).find_map(|dir| {
                let candidate = dir.join("spirv-opt");
                if candidate.is_file() {
                    Some(candidate)
                } else {
                    None
                }
            })
        })
        .map(|p: PathBuf| p.to_string_lossy().into_owned())
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

fn arith_signature(words: &[u32]) -> Vec<(Op, Vec<String>)> {
    let mut loader = Loader::new();
    parse_words(words, &mut loader).expect("parse module");
    let module = loader.module();
    let mut sig: Vec<_> = module
        .types_global_values
        .iter()
        .chain(
            module
                .functions
                .iter()
                .flat_map(|func| func.blocks.iter().flat_map(|blk| blk.instructions.iter())),
        )
        .filter(|inst| {
            matches!(
                inst.class.opcode,
                Op::Constant | Op::BitwiseAnd | Op::BitwiseOr | Op::BitwiseXor
            )
        })
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

fn build_and_over_or_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let x = b.constant_bit32(int, 0xABCD1234);
    let y = b.constant_bit32(int, 0x00FF00FF);
    let bor = b.bitwise_or(int, None, x, y).expect("bor");
    let _band = b.bitwise_and(int, None, x, bor).expect("band");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", &[]);
    b.module().assemble()
}

fn build_or_over_and_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let x = b.constant_bit32(int, 0x12345678);
    let y = b.constant_bit32(int, 0x0F0F0F0F);
    let band = b.bitwise_and(int, None, x, y).expect("and");
    let _bor = b.bitwise_or(int, None, x, band).expect("or");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", &[]);
    b.module().assemble()
}

fn build_and_over_or_module_64() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(64, 0);
    let func_ty = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let x = b.constant_bit64(int, 0x1122_3344_5566_7788);
    let y = b.constant_bit64(int, 0x00FF_00FF_00FF_00FF);
    let bor = b.bitwise_or(int, None, x, y).expect("bor");
    let _band = b.bitwise_and(int, None, x, bor).expect("band");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", &[]);
    b.module().assemble()
}

fn build_or_over_and_module_64() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(64, 0);
    let func_ty = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let x = b.constant_bit64(int, 0xA1B2_C3D4_E5F6_0708);
    let y = b.constant_bit64(int, 0x0F0F_0F0F_0F0F_0F0F);
    let band = b.bitwise_and(int, None, x, y).expect("and");
    let _bor = b.bitwise_or(int, None, x, band).expect("or");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", &[]);
    b.module().assemble()
}

fn run_cpp_opt(words: &[u32], cpp_opt: &str) -> Vec<u32> {
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let output = dir.path().join("cpp_out.spv");
    std::fs::write(&input, words_to_bytes(words)).expect("write input");
    let status = Command::new(cpp_opt)
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(status.success(), "C++ spirv-opt failed");
    bytes_to_words(&std::fs::read(&output).expect("read cpp output"))
}

fn run_rust_opt(words: &[u32]) -> Vec<u32> {
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.spv");
    let output = dir.path().join("rust_out.spv");
    std::fs::write(&input, words_to_bytes(words)).expect("write input");
    let exe = env!("CARGO_BIN_EXE_opt_block");
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .status()
        .expect("run opt_block");
    assert!(status.success(), "opt_block should succeed");
    bytes_to_words(&std::fs::read(&output).expect("read rust output"))
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn assert_cli_parity(module_words: &[u32]) {
    let Some(cpp_opt) = cpp_opt_bin() else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not on PATH; skipping CLI parity");
        return;
    };
    let rust_words = run_rust_opt(module_words);
    let cpp_words = run_cpp_opt(module_words, &cpp_opt);
    let rust_sig = arith_signature(&rust_words);
    let cpp_sig = arith_signature(&cpp_words);
    assert_eq!(rust_sig, cpp_sig, "CLI bitwise parity mismatch");
}

#[test]
fn cli_bitwise_and_absorption_parity() {
    assert_cli_parity(&build_and_over_or_module());
}

#[test]
fn cli_bitwise_or_absorption_parity() {
    assert_cli_parity(&build_or_over_and_module());
}

#[test]
fn cli_bitwise_and_absorption_parity_64bit() {
    assert_cli_parity(&build_and_over_or_module_64());
}

#[test]
fn cli_bitwise_or_absorption_parity_64bit() {
    assert_cli_parity(&build_or_over_and_module_64());
}
