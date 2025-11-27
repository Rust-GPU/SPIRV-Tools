use std::env;
use std::fs;
use std::io::Write;
use std::process::Command;

use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::{Block, Function, Instruction, Module};
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op, Word};
use tempfile::NamedTempFile;

fn cpp_opt_bin() -> Option<String> {
    match env::var("SPIRV_CPP_OPT") {
        Ok(path) if !path.is_empty() => Some(path),
        _ => {
            eprintln!("SPIRV_CPP_OPT not set; skipping C++ parity check");
            None
        }
    }
}

#[test]
fn rust_and_cpp_fold_rem_by_one() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let module_words = build_rem_one_module();
    let rust_insts = extract_srem_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    assert!(
        rust_optimized.iter().any(|inst| {
            inst.class.opcode == Op::Constant
                && inst.result_id == Some(9)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        }),
        "rust optimizer should fold rem by one to const 0"
    );

    let output_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&output_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_zero_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(9)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        cpp_zero_const,
        "C++ spirv-opt should fold rem by one to const 0"
    );
}

#[test]
fn rust_and_cpp_fold_div_by_one() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let module_words = build_div_one_module();
    let rust_insts = extract_sdiv_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    assert!(
        rust_optimized.iter().any(|inst| {
            inst.class.opcode == Op::Constant
                && inst.result_id == Some(9)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(8)]
        }),
        "rust optimizer should fold div by one to original value"
    );

    let output_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&output_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(9)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(8)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold div by one to original value"
    );
}

#[test]
fn rust_and_cpp_preserve_div_by_zero() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, div_id) = build_div_zero_module();
    let rust_insts = extract_sdiv_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_kept_div = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::SDiv && inst.result_id == Some(div_id));
    assert!(rust_kept_div, "rust optimizer should not fold div by zero");

    let output_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&output_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_kept_div = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::SDiv && inst.result_id == Some(div_id));
    assert!(
        cpp_kept_div,
        "C++ spirv-opt should not fold div by zero and keep the instruction"
    );
}

#[test]
fn rust_and_cpp_preserve_rem_by_zero() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, rem_id) = build_rem_zero_module();
    let rust_insts = extract_srem_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_kept = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::SRem && inst.result_id == Some(rem_id));
    assert!(rust_kept, "rust optimizer should not fold rem by zero");

    let output_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&output_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_kept = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::SRem && inst.result_id == Some(rem_id));
    assert!(
        cpp_kept,
        "C++ spirv-opt should not fold rem by zero and keep the instruction"
    );
}

fn build_rem_one_module() -> Vec<u32> {
    let mut module = Module::default();
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![rspirv::dr::Operand::Capability(Capability::Shader)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            rspirv::dr::Operand::AddressingModel(AddressingModel::Logical),
            rspirv::dr::Operand::MemoryModel(MemoryModel::Simple),
        ],
    ));

    let type_void = Instruction::new(Op::TypeVoid, None, Some(1), vec![]);
    let type_int = Instruction::new(
        Op::TypeInt,
        None,
        Some(2),
        vec![
            rspirv::dr::Operand::LiteralBit32(32),
            rspirv::dr::Operand::LiteralBit32(0),
        ],
    );
    let type_func = Instruction::new(
        Op::TypeFunction,
        None,
        Some(3),
        vec![rspirv::dr::Operand::IdRef(1)],
    );
    let const_five = Instruction::new(
        Op::Constant,
        Some(2),
        Some(4),
        vec![rspirv::dr::Operand::LiteralBit32(5)],
    );
    let const_one = Instruction::new(
        Op::Constant,
        Some(2),
        Some(5),
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    module.types_global_values = vec![type_void, type_int, type_func, const_five, const_one];

    let func = build_function(
        FunctionIds {
            result_type: 1,
            func_type: 3,
            int_type: 2,
            func_id: 6,
            label_id: 7,
            result_id: 9,
        },
        Op::SRem,
        &[rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(5)],
    );
    module.functions.push(func);
    module.assemble()
}

fn build_div_one_module() -> Vec<u32> {
    let mut module = Module::default();
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![rspirv::dr::Operand::Capability(Capability::Shader)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            rspirv::dr::Operand::AddressingModel(AddressingModel::Logical),
            rspirv::dr::Operand::MemoryModel(MemoryModel::Simple),
        ],
    ));
    let type_void = Instruction::new(Op::TypeVoid, None, Some(1), vec![]);
    let type_int = Instruction::new(
        Op::TypeInt,
        None,
        Some(2),
        vec![
            rspirv::dr::Operand::LiteralBit32(32),
            rspirv::dr::Operand::LiteralBit32(0),
        ],
    );
    let type_func = Instruction::new(
        Op::TypeFunction,
        None,
        Some(3),
        vec![rspirv::dr::Operand::IdRef(1)],
    );
    let const_eight = Instruction::new(
        Op::Constant,
        Some(2),
        Some(4),
        vec![rspirv::dr::Operand::LiteralBit32(8)],
    );
    let const_one = Instruction::new(
        Op::Constant,
        Some(2),
        Some(5),
        vec![rspirv::dr::Operand::LiteralBit32(1)],
    );
    module.types_global_values = vec![type_void, type_int, type_func, const_eight, const_one];
    let func = build_function(
        FunctionIds {
            result_type: 1,
            func_type: 3,
            int_type: 2,
            func_id: 6,
            label_id: 7,
            result_id: 9,
        },
        Op::SDiv,
        &[rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(5)],
    );
    module.functions.push(func);
    module.assemble()
}

fn build_div_zero_module() -> (Vec<u32>, Word) {
    let mut module = Module::default();
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![rspirv::dr::Operand::Capability(Capability::Shader)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            rspirv::dr::Operand::AddressingModel(AddressingModel::Logical),
            rspirv::dr::Operand::MemoryModel(MemoryModel::Simple),
        ],
    ));
    let type_void = Instruction::new(Op::TypeVoid, None, Some(1), vec![]);
    let type_int = Instruction::new(
        Op::TypeInt,
        None,
        Some(2),
        vec![
            rspirv::dr::Operand::LiteralBit32(32),
            rspirv::dr::Operand::LiteralBit32(0),
        ],
    );
    let type_func = Instruction::new(
        Op::TypeFunction,
        None,
        Some(3),
        vec![rspirv::dr::Operand::IdRef(1)],
    );
    let const_two = Instruction::new(
        Op::Constant,
        Some(2),
        Some(4),
        vec![rspirv::dr::Operand::LiteralBit32(2)],
    );
    let const_zero = Instruction::new(
        Op::Constant,
        Some(2),
        Some(5),
        vec![rspirv::dr::Operand::LiteralBit32(0)],
    );
    module.types_global_values = vec![type_void, type_int, type_func, const_two, const_zero];
    let func = build_function(
        FunctionIds {
            result_type: 1,
            func_type: 3,
            int_type: 2,
            func_id: 6,
            label_id: 7,
            result_id: 9,
        },
        Op::SDiv,
        &[rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(5)],
    );
    module.functions.push(func);
    (module.assemble(), 9)
}

fn build_rem_zero_module() -> (Vec<u32>, Word) {
    let mut module = Module::default();
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![rspirv::dr::Operand::Capability(Capability::Shader)],
    ));
    module.memory_model = Some(Instruction::new(
        Op::MemoryModel,
        None,
        None,
        vec![
            rspirv::dr::Operand::AddressingModel(AddressingModel::Logical),
            rspirv::dr::Operand::MemoryModel(MemoryModel::Simple),
        ],
    ));
    let type_void = Instruction::new(Op::TypeVoid, None, Some(1), vec![]);
    let type_int = Instruction::new(
        Op::TypeInt,
        None,
        Some(2),
        vec![
            rspirv::dr::Operand::LiteralBit32(32),
            rspirv::dr::Operand::LiteralBit32(0),
        ],
    );
    let type_func = Instruction::new(
        Op::TypeFunction,
        None,
        Some(3),
        vec![rspirv::dr::Operand::IdRef(1)],
    );
    let const_two = Instruction::new(
        Op::Constant,
        Some(2),
        Some(4),
        vec![rspirv::dr::Operand::LiteralBit32(2)],
    );
    let const_zero = Instruction::new(
        Op::Constant,
        Some(2),
        Some(5),
        vec![rspirv::dr::Operand::LiteralBit32(0)],
    );
    module.types_global_values = vec![type_void, type_int, type_func, const_two, const_zero];
    let func = build_function(
        FunctionIds {
            result_type: 1,
            func_type: 3,
            int_type: 2,
            func_id: 6,
            label_id: 7,
            result_id: 9,
        },
        Op::SRem,
        &[rspirv::dr::Operand::IdRef(4), rspirv::dr::Operand::IdRef(5)],
    );
    module.functions.push(func);
    (module.assemble(), 9)
}

fn build_function(ids: FunctionIds, opcode: Op, rem_operands: &[rspirv::dr::Operand]) -> Function {
    let def = Instruction::new(
        Op::Function,
        Some(ids.result_type),
        Some(ids.func_id),
        vec![
            rspirv::dr::Operand::FunctionControl(FunctionControl::NONE),
            rspirv::dr::Operand::IdRef(ids.func_type),
        ],
    );
    let label = Instruction::new(Op::Label, None, Some(ids.label_id), vec![]);
    let arith = Instruction::new(
        opcode,
        Some(ids.int_type),
        Some(ids.result_id),
        rem_operands.to_vec(),
    );
    let ret = Instruction::new(Op::Return, None, None, vec![]);
    let end = Instruction::new(Op::FunctionEnd, None, None, vec![]);
    let block = Block {
        label: Some(label),
        instructions: vec![arith, ret],
    };
    Function {
        def: Some(def),
        parameters: vec![],
        blocks: vec![block],
        end: Some(end),
    }
}

#[derive(Copy, Clone)]
struct FunctionIds {
    result_type: Word,
    func_type: Word,
    int_type: Word,
    func_id: Word,
    label_id: Word,
    result_id: Word,
}

fn extract_srem_block(module_words: &[u32]) -> Vec<rspirv::dr::Instruction> {
    let mut loader = rspirv::dr::Loader::new();
    parse_words(module_words, &mut loader).expect("parse module");
    let module = loader.module();
    let block = &module.functions[0].blocks[0];
    module
        .types_global_values
        .iter()
        .chain(block.instructions.iter())
        .filter(|inst| matches!(inst.class.opcode, Op::Constant | Op::SRem))
        .cloned()
        .collect()
}

fn extract_sdiv_block(module_words: &[u32]) -> Vec<rspirv::dr::Instruction> {
    let mut loader = rspirv::dr::Loader::new();
    parse_words(module_words, &mut loader).expect("parse module");
    let module = loader.module();
    let block = &module.functions[0].blocks[0];
    module
        .types_global_values
        .iter()
        .chain(block.instructions.iter())
        .filter(|inst| matches!(inst.class.opcode, Op::Constant | Op::SDiv))
        .cloned()
        .collect()
}

fn run_cpp_opt(cpp_opt: &str, module_words: &[u32]) -> Vec<u32> {
    let mut input = NamedTempFile::new().expect("input temp");
    input
        .write_all(&words_to_bytes(module_words))
        .expect("write input");
    let output = NamedTempFile::new().expect("output temp");
    let status = Command::new(cpp_opt)
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("-O")
        .status()
        .expect("run spirv-opt");
    assert!(status.success(), "C++ spirv-opt failed");
    bytes_to_words(&fs::read(output.path()).expect("read output"))
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
