use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::{Builder, Loader};
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use spirv_tools_ffi::{
    clear_rust_optimizer_override, optimize_basic_block, set_rust_optimizer_override,
};
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, MutexGuard};
use tempfile::NamedTempFile;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct OptimizerEnvGuard {
    _lock: MutexGuard<'static, ()>,
}

impl OptimizerEnvGuard {
    fn new() -> Self {
        let lock = ENV_LOCK.lock().expect("env mutex poisoned");
        clear_rust_optimizer_override();
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
        set_rust_optimizer_override(true);
        Self { _lock: lock }
    }
}

impl Drop for OptimizerEnvGuard {
    fn drop(&mut self) {
        clear_rust_optimizer_override();
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    }
}

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

fn run_cpp_opt(words: &[u32], cpp_opt: &str) -> Vec<u32> {
    let input = NamedTempFile::new().expect("input temp");
    let output = NamedTempFile::new().expect("output temp");
    std::fs::write(input.path(), words_to_bytes(words)).expect("write input");
    let status = Command::new(cpp_opt)
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(status.success(), "C++ spirv-opt failed");
    bytes_to_words(&std::fs::read(output.path()).expect("read output"))
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn build_and_over_or_module(int_width: u32) -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(int_width, 0);
    let func_ty = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let x = if int_width == 32 {
        b.constant_bit32(int, 0xABCD1234)
    } else {
        b.constant_bit64(int, 0x1122_3344_5566_7788)
    };
    let y = if int_width == 32 {
        b.constant_bit32(int, 0x00FF00FF)
    } else {
        b.constant_bit64(int, 0x00FF_00FF_00FF_00FF)
    };
    let bor = b.bitwise_or(int, None, x, y).expect("bor");
    let _band = b.bitwise_and(int, None, x, bor).expect("band");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
    b.module().assemble()
}

fn build_or_over_and_module(int_width: u32) -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(int_width, 0);
    let func_ty = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let _ = b.begin_block(None).unwrap();
    let x = if int_width == 32 {
        b.constant_bit32(int, 0x12345678)
    } else {
        b.constant_bit64(int, 0xA1B2_C3D4_E5F6_0708)
    };
    let y = if int_width == 32 {
        b.constant_bit32(int, 0x0F0F0F0F)
    } else {
        b.constant_bit64(int, 0x0F0F_0F0F_0F0F_0F0F)
    };
    let band = b.bitwise_and(int, None, x, y).expect("and");
    let _bor = b.bitwise_or(int, None, x, band).expect("or");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
    b.module().assemble()
}

fn build_bor_distribute_over_bxor_module(int_width: u32) -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(int_width, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let x = b.function_parameter(int).expect("x");
    let y = b.function_parameter(int).expect("y");
    let z = if int_width == 32 {
        b.constant_bit32(int, 0xFF00FF00)
    } else {
        b.constant_bit64(int, 0xFF00_FF00_FF00_FF00)
    };
    let _ = b.begin_block(None).unwrap();
    let xor = b.bitwise_xor(int, None, y, z).expect("xor");
    let _ = b.bitwise_or(int, None, x, xor).expect("bor");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
    b.module().assemble()
}

fn assert_ffi_parity(module_words: &[u32]) {
    let Some(cpp_opt) = cpp_opt_bin() else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not on PATH; skipping FFI parity");
        return;
    };
    let guard = OptimizerEnvGuard::new();
    let rust_result = optimize_basic_block(module_words);
    assert!(
        rust_result.success,
        "rust optimize failed: {}",
        rust_result.message
    );
    let rust_words = rust_result.words;
    drop(guard);
    let cpp_words = run_cpp_opt(module_words, &cpp_opt);
    let rust_sig = arith_signature(&rust_words);
    let cpp_sig = arith_signature(&cpp_words);
    assert_eq!(rust_sig, cpp_sig, "FFI bitwise absorption parity mismatch");
}

#[test]
fn ffi_bitwise_and_absorption_parity() {
    assert_ffi_parity(&build_and_over_or_module(32));
}

#[test]
fn ffi_bitwise_or_absorption_parity() {
    assert_ffi_parity(&build_or_over_and_module(32));
}

#[test]
fn ffi_bitwise_and_absorption_parity_64bit() {
    assert_ffi_parity(&build_and_over_or_module(64));
}

#[test]
fn ffi_bitwise_or_absorption_parity_64bit() {
    assert_ffi_parity(&build_or_over_and_module(64));
}

#[test]
fn ffi_bitwise_distributes_over_xor_parity() {
    assert_ffi_parity(&build_bor_distribute_over_bxor_module(32));
}

#[test]
fn ffi_bitwise_distributes_over_xor_parity_64bit() {
    assert_ffi_parity(&build_bor_distribute_over_bxor_module(64));
}
