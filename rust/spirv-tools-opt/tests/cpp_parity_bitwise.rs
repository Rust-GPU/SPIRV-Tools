use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::{Builder, Instruction, Loader};
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::env;
use std::path::PathBuf;
use std::process::Command;
use tempfile::NamedTempFile;

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

fn is_arith_opcode(op: Op) -> bool {
    matches!(
        op,
        Op::Constant | Op::BitwiseAnd | Op::BitwiseOr | Op::BitwiseXor
    )
}

fn arith_signature(insts: &[Instruction]) -> Vec<(Op, Option<u32>, Vec<String>)> {
    let mut sig: Vec<_> = insts
        .iter()
        .map(|inst| {
            (
                inst.class.opcode,
                inst.result_id,
                inst.operands
                    .iter()
                    .map(|op| format!("{op:?}"))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    sig.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    sig
}

fn extract_arith_insts(words: &[u32]) -> Vec<Instruction> {
    let mut loader = Loader::new();
    parse_words(words, &mut loader).expect("parse module");
    let module = loader.module();
    let block = &module.functions[0].blocks[0];
    module
        .types_global_values
        .iter()
        .chain(block.instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect()
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
    let bytes = std::fs::read(output.path()).expect("read output");
    let mut words_out = Vec::new();
    for chunk in bytes.chunks_exact(4) {
        words_out.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    words_out
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
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
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
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
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
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
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
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
    b.module().assemble()
}

fn build_reassociate_bor_const_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c1 = b.constant_bit32(int, 0x0F);
    let c2 = b.constant_bit32(int, 0xF0);
    let inner = b.bitwise_or(int, None, param, c2).expect("bor inner");
    let _outer = b.bitwise_or(int, None, c1, inner).expect("bor outer");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
    b.module().assemble()
}

fn build_reassociate_bxor_const_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .unwrap();
    let param = b.function_parameter(int).unwrap();
    let _ = b.begin_block(None).unwrap();
    let c1 = b.constant_bit32(int, 0x0F);
    let c2 = b.constant_bit32(int, 0xF0);
    let inner = b.bitwise_xor(int, None, param, c2).expect("bxor inner");
    let _outer = b.bitwise_xor(int, None, c1, inner).expect("bxor outer");
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
    b.module().assemble()
}

#[test]
fn rust_and_cpp_bitwise_absorption_parity() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not on PATH; skipping parity");
        return;
    };
    let module_words = build_and_over_or_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let cpp_words = run_cpp_opt(&module_words, &cpp_opt);
    let mut loader = Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse optimized module");
    let cpp_module = loader.module();
    let cpp_arith: Vec<_> = cpp_module
        .types_global_values
        .iter()
        .chain(cpp_module.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(rust_sig, cpp_sig, "bitwise absorption parity mismatch");
}

#[test]
fn rust_and_cpp_bitwise_or_absorption_parity() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not on PATH; skipping parity");
        return;
    };
    let module_words = build_or_over_and_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let cpp_words = run_cpp_opt(&module_words, &cpp_opt);
    let mut loader = Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse optimized module");
    let cpp_module = loader.module();
    let cpp_arith: Vec<_> = cpp_module
        .types_global_values
        .iter()
        .chain(cpp_module.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(rust_sig, cpp_sig, "bitwise OR absorption parity mismatch");
}

#[test]
fn rust_and_cpp_bitwise_reassociate_const_or_parity() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not on PATH; skipping parity");
        return;
    };
    let module_words = build_reassociate_bor_const_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let cpp_words = run_cpp_opt(&module_words, &cpp_opt);
    let mut loader = Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse optimized module");
    let cpp_module = loader.module();
    let cpp_arith: Vec<_> = cpp_module
        .types_global_values
        .iter()
        .chain(cpp_module.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "bitwise const reassociation (or) parity mismatch"
    );
}

#[test]
fn rust_and_cpp_bitwise_reassociate_const_xor_parity() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not on PATH; skipping parity");
        return;
    };
    let module_words = build_reassociate_bxor_const_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let cpp_words = run_cpp_opt(&module_words, &cpp_opt);
    let mut loader = Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse optimized module");
    let cpp_module = loader.module();
    let cpp_arith: Vec<_> = cpp_module
        .types_global_values
        .iter()
        .chain(cpp_module.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "bitwise const reassociation (xor) parity mismatch"
    );
}

#[test]
fn rust_and_cpp_bitwise_absorption_parity_64bit() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not on PATH; skipping parity");
        return;
    };
    let module_words = build_and_over_or_module_64();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let cpp_words = run_cpp_opt(&module_words, &cpp_opt);
    let mut loader = Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse optimized module");
    let cpp_module = loader.module();
    let cpp_arith: Vec<_> = cpp_module
        .types_global_values
        .iter()
        .chain(cpp_module.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "64-bit bitwise absorption parity mismatch"
    );
}

#[test]
fn rust_and_cpp_bitwise_or_absorption_parity_64bit() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not on PATH; skipping parity");
        return;
    };
    let module_words = build_or_over_and_module_64();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let cpp_words = run_cpp_opt(&module_words, &cpp_opt);
    let mut loader = Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse optimized module");
    let cpp_module = loader.module();
    let cpp_arith: Vec<_> = cpp_module
        .types_global_values
        .iter()
        .chain(cpp_module.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "64-bit bitwise OR absorption parity mismatch"
    );
}
