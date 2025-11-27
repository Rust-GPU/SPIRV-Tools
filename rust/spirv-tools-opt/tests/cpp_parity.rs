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
    let cpp_opt = match env::var("SPIRV_CPP_OPT") {
        Ok(path) if !path.is_empty() => path,
        _ => {
            eprintln!("SPIRV_CPP_OPT not set; skipping C++ parity check");
            return;
        }
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
