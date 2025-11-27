use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::env;
use std::fs;
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

/// Compare the Rust arithmetic optimizer output against the C++ spirv-opt (when available).
#[test]
fn rust_and_cpp_fold_const_add() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, _) = build_const_add_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_has_const = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::Constant && is_const_five(inst));
    assert!(rust_has_const, "rust optimizer should fold to const 5");

    // Run C++ spirv-opt if available.
    let mut input = NamedTempFile::new().expect("input temp");
    input
        .write_all(&words_to_bytes(&module_words))
        .expect("write input");
    let output = NamedTempFile::new().expect("output temp");

    let status = Command::new(&cpp_opt)
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(status.success(), "C++ spirv-opt failed");

    let cpp_words = bytes_to_words(&fs::read(output.path()).expect("read output"));
    let mut cpp_loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut cpp_loader).expect("parse cpp optimized");
    let cpp_has_const = cpp_loader
        .module()
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::Constant && is_const_five(inst));
    assert!(
        cpp_has_const,
        "C++ spirv-opt should fold const add to literal 5"
    );
}

#[test]
fn rust_and_cpp_fold_mul_by_zero() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let module_words = build_mul_zero_module();
    let rust_insts = extract_mul_zero_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_zero_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    let rust_has_mul = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::IMul);
    assert!(
        rust_zero_const,
        "rust optimizer should fold mul by zero to const 0"
    );
    assert!(
        !rust_has_mul,
        "rust optimizer should remove mul instruction"
    );

    let mut input = NamedTempFile::new().expect("input temp");
    input
        .write_all(&words_to_bytes(&module_words))
        .expect("write input");
    let output = NamedTempFile::new().expect("output temp");
    let status = Command::new(&cpp_opt)
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(status.success(), "C++ spirv-opt failed");
    let cpp_words = bytes_to_words(&fs::read(output.path()).expect("read output"));
    let mut cpp_loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut cpp_loader).expect("parse cpp optimized");
    let module = cpp_loader.module();
    let mut cpp_zero_const = false;
    let mut cpp_has_mul = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IMul {
            cpp_has_mul = true;
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(5)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            cpp_zero_const = true;
        }
    }
    assert!(
        cpp_zero_const,
        "C++ spirv-opt should fold mul by zero to const 0"
    );
    assert!(
        !cpp_has_mul,
        "C++ spirv-opt should remove mul instruction after folding"
    );
}

#[test]
fn rust_and_cpp_fold_add_zero() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, add_id) = build_add_zero_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
    });
    let rust_has_add = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::IAdd);
    assert!(rust_const, "rust optimizer should fold add zero to const 4");
    assert!(
        !rust_has_add,
        "rust optimizer should remove add instruction"
    );

    let mut input = NamedTempFile::new().expect("input temp");
    input
        .write_all(&words_to_bytes(&module_words))
        .expect("write input");
    let output = NamedTempFile::new().expect("output temp");
    let status = Command::new(&cpp_opt)
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(status.success(), "C++ spirv-opt failed");
    let cpp_words = bytes_to_words(&fs::read(output.path()).expect("read output"));
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let mut cpp_const = false;
    let mut cpp_has_add = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IAdd {
            cpp_has_add = true;
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
        {
            cpp_const = true;
        }
    }
    assert!(
        cpp_const,
        "C++ spirv-opt should fold add zero to const 4 with same id"
    );
    assert!(!cpp_has_add, "C++ spirv-opt should remove add instruction");
}

#[test]
fn rust_and_cpp_fold_sub_self_to_zero() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, sub_id) = build_sub_self_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    let rust_has_sub = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::ISub);
    assert!(rust_const, "rust optimizer should fold sub self to const 0");
    assert!(!rust_has_sub, "rust optimizer should remove subtraction");

    let mut input = NamedTempFile::new().expect("input temp");
    input
        .write_all(&words_to_bytes(&module_words))
        .expect("write input");
    let output = NamedTempFile::new().expect("output temp");
    let status = Command::new(&cpp_opt)
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(status.success(), "C++ spirv-opt failed");
    let cpp_words = bytes_to_words(&fs::read(output.path()).expect("read output"));
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let mut cpp_const = false;
    let mut cpp_has_sub = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::ISub {
            cpp_has_sub = true;
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            cpp_const = true;
        }
    }
    assert!(
        cpp_const,
        "C++ spirv-opt should fold sub self to const 0 with same id"
    );
    assert!(!cpp_has_sub, "C++ spirv-opt should remove subtraction");
}

fn build_const_add_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c2 = b.constant_bit32(int, 2);
    let c3 = b.constant_bit32(int, 3);
    let _add = b.i_add(int, None, c2, c3).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), func)
}

fn extract_arith_block(module_words: &[u32]) -> Vec<rspirv::dr::Instruction> {
    let mut loader = rspirv::dr::Loader::new();
    parse_words(module_words, &mut loader).expect("parse module");
    let module = loader.module();
    let block = &module.functions[0].blocks[0];
    module
        .types_global_values
        .iter()
        .chain(block.instructions.iter())
        .filter(|inst| matches!(inst.class.opcode, Op::Constant | Op::IAdd))
        .cloned()
        .collect()
}

fn build_mul_zero_module() -> Vec<u32> {
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
    let c4 = b.constant_bit32(int, 4);
    let c0 = b.constant_bit32(int, 0);
    let _ = b.i_mul(int, None, c4, c0).expect("mul");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn extract_mul_zero_block(module_words: &[u32]) -> Vec<rspirv::dr::Instruction> {
    let mut loader = rspirv::dr::Loader::new();
    parse_words(module_words, &mut loader).expect("parse module");
    let module = loader.module();
    let block = &module.functions[0].blocks[0];
    module
        .types_global_values
        .iter()
        .chain(block.instructions.iter())
        .filter(|inst| matches!(inst.class.opcode, Op::Constant | Op::IMul))
        .cloned()
        .collect()
}

fn build_add_zero_module() -> (Vec<u32>, u32) {
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
    let c4 = b.constant_bit32(int, 4);
    let c0 = b.constant_bit32(int, 0);
    let add = b.i_add(int, None, c4, c0).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_sub_self_module() -> (Vec<u32>, u32) {
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
    let c7 = b.constant_bit32(int, 7);
    let sub = b.i_sub(int, None, c7, c7).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn cpp_opt_bin() -> Option<String> {
    match env::var("SPIRV_CPP_OPT") {
        Ok(path) if !path.is_empty() => Some(path),
        _ => {
            eprintln!("SPIRV_CPP_OPT not set; skipping C++ parity check");
            None
        }
    }
}

fn is_const_five(inst: &rspirv::dr::Instruction) -> bool {
    inst.operands
        .iter()
        .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(5)))
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
