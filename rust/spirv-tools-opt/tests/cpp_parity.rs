use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Builder;
use rspirv::dr::Loader;
use rspirv::dr::Module;
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::env;
use std::fs;
use std::io::Write;
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

fn is_arith_opcode(op: Op) -> bool {
    matches!(
        op,
        Op::Constant
            | Op::IAdd
            | Op::IMul
            | Op::ISub
            | Op::SNegate
            | Op::SDiv
            | Op::UDiv
            | Op::UMod
            | Op::BitwiseAnd
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
            | Op::ShiftLeftLogical
    )
}

fn arith_signature(insts: &[rspirv::dr::Instruction]) -> Vec<(Op, Option<u32>, Vec<String>)> {
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

fn run_cpp_opt_module(words: &[u32], cpp_opt: &str) -> Module {
    let mut input = NamedTempFile::new().expect("input temp");
    input
        .write_all(&words_to_bytes(words))
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
    let cpp_words = bytes_to_words(&fs::read(output.path()).expect("read output"));
    let mut loader = Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    loader.module()
}

fn extract_arith_insts(words: &[u32]) -> Vec<rspirv::dr::Instruction> {
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
fn rust_and_cpp_arith_outputs_match_const_add() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let (module_words, _) = build_const_add_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for const add"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_mul_zero() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_mul_zero_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for mul zero"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_div_rem() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_div_rem_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for div/rem"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_mul_pow2() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_mul_pow2_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for mul by pow2"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_mask_pow2() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_mask_pow2_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for mask by pow2"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_shift_chain() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_shift_chain_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for shift chains"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_arith_shift_chain() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_arith_shift_chain_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for arithmetic shift chains"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_neg_chain() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_neg_chain_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for negation chain"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_triple_neg_chain() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_triple_neg_chain_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for triple negation chain"
    );
}

#[test]
fn rust_and_cpp_arith_outputs_match_mask_then_shift() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };
    let module_words = build_mask_then_shift_module();
    let rust_sig = arith_signature(
        &spirv_tools_opt::translate::optimize_arith_block(&extract_arith_insts(&module_words))
            .expect("rust optimize"),
    );
    let optimized_cpp = run_cpp_opt_module(&module_words, &cpp_opt);
    let cpp_arith: Vec<_> = optimized_cpp
        .types_global_values
        .iter()
        .chain(optimized_cpp.functions[0].blocks[0].instructions.iter())
        .filter(|inst| is_arith_opcode(inst.class.opcode))
        .cloned()
        .collect();
    let cpp_sig = arith_signature(&cpp_arith);
    assert_eq!(
        rust_sig, cpp_sig,
        "Rust vs C++ arithmetic output mismatch for mask then shift"
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
fn rust_and_cpp_cancel_add_sub_chain() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_add_sub_cancel_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(42)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (a - b) + b to a constant"
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
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(42)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (a - b) + b to original value"
    );
}

#[test]
fn rust_and_cpp_fold_sub_add_const_chain() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_sub_add_const_chain();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(10)]
    });
    assert!(rust_const, "rust optimizer should fold (10-3)+3 to 10");

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(10)]
    });
    assert!(cpp_const, "C++ spirv-opt should fold (10-3)+3 to 10");
}

#[test]
fn rust_and_cpp_fold_shared_addend_sub_const_chain() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_shared_addend_sub_chain();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(3)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (4+5)-(4+2) to constant 3"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(3)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (4+5)-(4+2) to constant 3"
    );
}

#[test]
fn rust_and_cpp_fold_mirrored_add_sub() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_mirror_add_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
    });
    assert!(rust_const, "rust optimizer should fold 7 + (4 - 7) to 4");

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold mirrored add/sub to original rhs"
    );
}

#[test]
fn rust_and_cpp_cancel_commuted_shared_addends() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_commuted_shared_addends_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (y+x)-(x+y) with constants to zero"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (y+x)-(x+y) with constants to zero"
    );
}

#[test]
fn rust_and_cpp_fold_shared_addends_const_diff() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_shared_addends_const_diff_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let expected = u32::MAX; // 3 - 4 wraps.
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(expected)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (9+3)-(9+4) to wrapped constant -1"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(expected)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (9+3)-(9+4) to wrapped constant -1"
    );
}

#[test]
fn rust_and_cpp_fold_shared_addends_const_diff_positive() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_shared_addends_const_diff_positive_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(1)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (9+4)-(9+3) to constant 1"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(1)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (9+4)-(9+3) to constant 1"
    );
}

#[test]
fn rust_and_cpp_cancel_shared_symbolic_addends_const_diff() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_shared_symbolic_addends_const_diff_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(3)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (x+5)-(x+2) to constant 3"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(3)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (x+5)-(x+2) to constant 3"
    );
}

#[test]
fn rust_and_cpp_cancel_shared_symbolic_addends_exact_match() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_shared_symbolic_addends_match_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (x+7)-(x+7) to constant zero"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (x+7)-(x+7) to constant zero"
    );
}

#[test]
fn rust_and_cpp_factor_symbolic_difference() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, param_id) = build_symbolic_factor_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_three_ids: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(3)]
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(lhs), rspirv::dr::Operand::IdRef(rhs)]
                    if (lhs == &param_id && rust_three_ids.contains(rhs))
                        || (rhs == &param_id && rust_three_ids.contains(lhs))
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should factor x*(5-2) into 3*x"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_three_ids: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(3)]
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(lhs), rspirv::dr::Operand::IdRef(rhs)]
                    if (lhs == &param_id && cpp_three_ids.contains(rhs))
                        || (rhs == &param_id && cpp_three_ids.contains(lhs))
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should factor x*(5-2) into 3*x"
    );
}

#[test]
fn rust_and_cpp_factor_symbolic_commuted_addends() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, x_id, y_id, z_id) = build_symbolic_factor_commuted_add_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_adds: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| inst.class.opcode == Op::IAdd)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if (a == &y_id && b == &z_id) || (a == &z_id && b == &y_id) {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && rust_adds.contains(b))
                        || (b == &x_id && rust_adds.contains(a))
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should factor (y*x)+(x*z) into x*(y+z)"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_adds: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::IAdd)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if (a == &y_id && b == &z_id) || (a == &z_id && b == &y_id) {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && cpp_adds.contains(b))
                        || (b == &x_id && cpp_adds.contains(a))
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should factor (y*x)+(x*z) into x*(y+z)"
    );
}

#[test]
fn rust_and_cpp_factor_symbolic_commuted_subtraction() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, x_id, y_id, z_id) = build_symbolic_factor_commuted_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_subs: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| inst.class.opcode == Op::ISub)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if a == &y_id && b == &z_id {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && rust_subs.contains(b))
                        || (b == &x_id && rust_subs.contains(a))
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should factor (y*x)-(z*x) into x*(y-z)"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_subs: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::ISub)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if a == &y_id && b == &z_id {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && cpp_subs.contains(b))
                        || (b == &x_id && cpp_subs.contains(a))
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should factor (y*x)-(z*x) into x*(y-z)"
    );
}

#[test]
fn rust_and_cpp_factor_constant_commuted_add() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, add_id) = build_const_factor_commuted_add_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_adds: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| inst.class.opcode == Op::IAdd)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if (a == &add_id || b == &add_id) && inst.result_id.is_some() {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if rust_adds.contains(a)
                        && matches!(b, id if {
                            rust_optimized.iter().any(|c| {
                                c.class.opcode == Op::Constant
                                    && c.result_id == Some(*id)
                                    && c.operands
                                        == vec![rspirv::dr::Operand::LiteralBit32(4)]
                            })
                        })
                        || rust_adds.contains(b)
                            && matches!(a, id if {
                                rust_optimized.iter().any(|c| {
                                    c.class.opcode == Op::Constant
                                        && c.result_id == Some(*id)
                                        && c.operands
                                            == vec![rspirv::dr::Operand::LiteralBit32(4)]
                                })
                            })
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should factor (4*x)+(4*y) into 4*(x+y)"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_adds: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::IAdd)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if (a == &add_id || b == &add_id) && inst.result_id.is_some() {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if cpp_adds.contains(a)
                        && matches!(b, id if {
                            module.all_inst_iter().any(|c| {
                                c.class.opcode == Op::Constant
                                    && c.result_id == Some(*id)
                                    && c.operands
                                        == vec![rspirv::dr::Operand::LiteralBit32(4)]
                            })
                        })
                        || cpp_adds.contains(b)
                            && matches!(a, id if {
                                module.all_inst_iter().any(|c| {
                                    c.class.opcode == Op::Constant
                                        && c.result_id == Some(*id)
                                        && c.operands
                                            == vec![rspirv::dr::Operand::LiteralBit32(4)]
                                })
                            })
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should factor (4*x)+(4*y) into 4*(x+y)"
    );
}

#[test]
fn rust_and_cpp_factor_constant_commuted_sub() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, sub_id) = build_const_factor_commuted_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_subs: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| inst.class.opcode == Op::ISub)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if a == &sub_id && b == &sub_id {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if rust_subs.contains(a)
                        && matches!(b, id if {
                            rust_optimized.iter().any(|c| {
                                c.class.opcode == Op::Constant
                                    && c.result_id == Some(*id)
                                    && c.operands
                                        == vec![rspirv::dr::Operand::LiteralBit32(6)]
                            })
                        })
                        || rust_subs.contains(b)
                            && matches!(a, id if {
                                rust_optimized.iter().any(|c| {
                                    c.class.opcode == Op::Constant
                                        && c.result_id == Some(*id)
                                        && c.operands
                                            == vec![rspirv::dr::Operand::LiteralBit32(6)]
                                })
                            })
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should factor (6*x)-(6*y) into 6*(x-y)"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_subs: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::ISub)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if a == &sub_id && b == &sub_id {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if cpp_subs.contains(a)
                        && matches!(b, id if {
                            module.all_inst_iter().any(|c| {
                                c.class.opcode == Op::Constant
                                    && c.result_id == Some(*id)
                                    && c.operands
                                        == vec![rspirv::dr::Operand::LiteralBit32(6)]
                            })
                        })
                        || cpp_subs.contains(b)
                            && matches!(a, id if {
                                module.all_inst_iter().any(|c| {
                                    c.class.opcode == Op::Constant
                                        && c.result_id == Some(*id)
                                        && c.operands
                                            == vec![rspirv::dr::Operand::LiteralBit32(6)]
                                })
                            })
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should factor (6*x)-(6*y) into 6*(x-y)"
    );
}

#[test]
fn rust_and_cpp_factor_symbolic_const_add() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, x_id, y_id, const_id) = build_symbolic_const_factor_add_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_adds: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| inst.class.opcode == Op::IAdd)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if (a == &y_id && b == &const_id) || (a == &const_id && b == &y_id) {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && rust_adds.contains(b))
                        || (b == &x_id && rust_adds.contains(a))
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should factor (x*y)+(x*3) into x*(y+3)"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_adds: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::IAdd)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if (a == &y_id && b == &const_id) || (a == &const_id && b == &y_id) {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && cpp_adds.contains(b))
                        || (b == &x_id && cpp_adds.contains(a))
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should factor (x*y)+(x*3) into x*(y+3)"
    );
}

#[test]
fn rust_and_cpp_factor_symbolic_const_sub() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, x_id, y_id, const_id) = build_symbolic_const_factor_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_subs: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| inst.class.opcode == Op::ISub)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if a == &y_id && b == &const_id {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && rust_subs.contains(b))
                        || (b == &x_id && rust_subs.contains(a))
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should factor (x*y)-(x*3) into x*(y-3)"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_subs: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::ISub)
        .filter_map(|inst| {
            if let [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)] =
                inst.operands.as_slice()
            {
                if a == &y_id && b == &const_id {
                    return inst.result_id;
                }
            }
            None
        })
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && cpp_subs.contains(b))
                        || (b == &x_id && cpp_subs.contains(a))
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should factor (x*y)-(x*3) into x*(y-3)"
    );
}

#[test]
fn rust_and_cpp_factor_mixed_constants_add() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, x_id) = build_mixed_const_factor_add_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const5_ids: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && rust_const5_ids.contains(b))
                        || (b == &x_id && rust_const5_ids.contains(a))
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should fold (2*x)+(x*3) to 5*x"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const5_ids: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && cpp_const5_ids.contains(b))
                        || (b == &x_id && cpp_const5_ids.contains(a))
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should fold (2*x)+(x*3) to 5*x"
    );
}

#[test]
fn rust_and_cpp_factor_mixed_constants_sub() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, x_id) = build_mixed_const_factor_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let neg_one = u32::MAX;
    let rust_const_neg1_ids: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(neg_one)]
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && rust_const_neg1_ids.contains(b))
                        || (b == &x_id && rust_const_neg1_ids.contains(a))
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should fold (2*x)-(x*3) to wrapped -1 * x"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const_neg1_ids: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(neg_one)]
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && cpp_const_neg1_ids.contains(b))
                        || (b == &x_id && cpp_const_neg1_ids.contains(a))
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should fold (2*x)-(x*3) to wrapped -1 * x"
    );
}

#[test]
fn rust_and_cpp_factor_mixed_constants_sub_positive() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id, x_id) = build_mixed_const_factor_sub_pos_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const1_ids: std::collections::HashSet<u32> = rust_optimized
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(1)]
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    let rust_mul_matches = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && rust_const1_ids.contains(b))
                        || (b == &x_id && rust_const1_ids.contains(a))
            )
    });
    assert!(
        rust_mul_matches,
        "rust optimizer should fold (3*x)-(2*x) to 1*x"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const1_ids: std::collections::HashSet<u32> = module
        .all_inst_iter()
        .filter(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(1)]
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    let cpp_mul_matches = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::IMul
            && inst.result_id == Some(result_id)
            && matches!(
                inst.operands.as_slice(),
                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)]
                    if (a == &x_id && cpp_const1_ids.contains(b))
                        || (b == &x_id && cpp_const1_ids.contains(a))
            )
    });
    assert!(
        cpp_mul_matches,
        "C++ spirv-opt should fold (3*x)-(2*x) to 1*x"
    );
}

#[test]
fn rust_and_cpp_fold_zero_factor_add() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_zero_factor_add_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const_zero = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        rust_const_zero,
        "rust optimizer should fold (0*x)+(0*y) to zero"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const_zero = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        cpp_const_zero,
        "C++ spirv-opt should fold (0*x)+(0*y) to zero"
    );
}

#[test]
fn rust_and_cpp_fold_zero_factor_sub() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_zero_factor_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const_zero = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        rust_const_zero,
        "rust optimizer should fold (0*x)-(0*y) to zero"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const_zero = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    assert!(
        cpp_const_zero,
        "C++ spirv-opt should fold (0*x)-(0*y) to zero"
    );
}

#[test]
fn rust_and_cpp_fold_const_factor_sub_chain() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_const_factor_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(12)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (6*x)-(2*x) with x=3 to const 12"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(12)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (6*x)-(2*x) with x=3 to const 12"
    );
}

#[test]
fn rust_and_cpp_distribute_const_mul_over_add() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, mul_id) = build_dist_const_mul_add_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(32)]
    });
    assert!(
        rust_const,
        "rust optimizer should distribute const mul over add and fold to const 32"
    );
    assert!(
        !rust_optimized
            .iter()
            .any(|inst| inst.class.opcode == Op::IMul),
        "rust optimizer should remove mul after distribution"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(32)]
    });
    let cpp_has_mul = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::IMul);
    assert!(
        cpp_const,
        "C++ spirv-opt should distribute const mul over add and fold to const 32"
    );
    assert!(
        !cpp_has_mul,
        "C++ spirv-opt should remove mul after distribution"
    );
}

#[test]
fn rust_and_cpp_fold_affine_gcd_add() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, add_id) = build_affine_gcd_add_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(36)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold affine gcd add with const operand to const 36"
    );
    assert!(
        !rust_optimized
            .iter()
            .any(|inst| matches!(inst.class.opcode, Op::IMul | Op::IAdd)),
        "rust optimizer should remove mul/add after folding"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(36)]
    });
    let cpp_has_ops = module.all_inst_iter().any(|inst| {
        matches!(inst.class.opcode, Op::IMul | Op::IAdd) && inst.result_id == Some(add_id)
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold affine gcd add with const operand to const 36"
    );
    assert!(
        !cpp_has_ops,
        "C++ spirv-opt should remove mul/add after folding"
    );
}

#[test]
fn rust_and_cpp_fold_affine_gcd_sub() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, sub_id) = build_affine_gcd_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(8)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold affine gcd sub with const operand to const 8"
    );
    assert!(
        !rust_optimized
            .iter()
            .any(|inst| matches!(inst.class.opcode, Op::IMul | Op::ISub)),
        "rust optimizer should remove mul/sub after folding"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(8)]
    });
    let cpp_has_ops = module.all_inst_iter().any(|inst| {
        matches!(inst.class.opcode, Op::IMul | Op::ISub) && inst.result_id == Some(sub_id)
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold affine gcd sub with const operand to const 8"
    );
    assert!(
        !cpp_has_ops,
        "C++ spirv-opt should remove mul/sub after folding"
    );
}

#[test]
fn rust_and_cpp_distribute_const_mul_over_sub() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, mul_id) = build_dist_const_mul_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(24)]
    });
    assert!(
        rust_const,
        "rust optimizer should distribute const mul over sub and fold to const 24"
    );
    assert!(
        !rust_optimized
            .iter()
            .any(|inst| inst.class.opcode == Op::IMul),
        "rust optimizer should remove mul after distribution"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(24)]
    });
    let cpp_has_mul = module
        .all_inst_iter()
        .any(|inst| inst.class.opcode == Op::IMul);
    assert!(
        cpp_const,
        "C++ spirv-opt should distribute const mul over sub and fold to const 24"
    );
    assert!(
        !cpp_has_mul,
        "C++ spirv-opt should remove mul after distribution"
    );
}

#[test]
fn rust_and_cpp_factor_commuted_multiplicands() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_commuted_factor_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(25)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (x*2)+(x*3) with x=5 to 25"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(25)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (x*2)+(x*3) with x=5 to 25"
    );
}

#[test]
fn rust_and_cpp_factor_commuted_multiplicands_sub() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_commuted_factor_sub_module();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(15)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (y*x)-(x*z) with y=4,z=1,x=5 to 15"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(15)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (y*x)-(x*z) with y=4,z=1,x=5 to 15"
    );
}

#[test]
fn rust_and_cpp_fold_sub_add_chain_with_shared_mid() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, result_id) = build_sub_add_chain();
    let rust_insts = extract_simple_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(9)]
    });
    assert!(
        rust_const,
        "rust optimizer should fold (10-3)+(3-1) to constant 9"
    );

    let cpp_words = run_cpp_opt(&cpp_opt, &module_words);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let cpp_const = module.all_inst_iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(result_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(9)]
    });
    assert!(
        cpp_const,
        "C++ spirv-opt should fold (10-3)+(3-1) to constant 9"
    );
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

#[test]
fn rust_and_cpp_fold_zero_minus_value() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, sub_id) = build_zero_minus_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(sub_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(9))]
    });
    let rust_has_sub = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::ISub);
    assert!(
        rust_const,
        "rust optimizer should fold zero minus value to constant -operand"
    );
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
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(9))]
        {
            cpp_const = true;
        }
    }
    assert!(
        cpp_const,
        "C++ spirv-opt should fold zero-minus-value to constant with same id"
    );
    assert!(!cpp_has_sub, "C++ spirv-opt should remove subtraction");
}

#[test]
fn rust_and_cpp_fold_add_negate_to_zero() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, add_id) = build_add_negate_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
    });
    let rust_has_add = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::IAdd);
    assert!(rust_const, "rust optimizer should fold add+neg to zero");
    assert!(!rust_has_add, "rust optimizer should remove addition");

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
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            cpp_const = true;
        }
    }
    assert!(
        cpp_const,
        "C++ spirv-opt should fold add+neg to zero with same id"
    );
    assert!(!cpp_has_add, "C++ spirv-opt should remove addition");
}

#[test]
fn rust_and_cpp_fold_double_negation() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, neg_id) = build_double_neg_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(neg_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
    });
    let rust_has_neg = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::SNegate);
    assert!(
        rust_const,
        "rust optimizer should fold double negation to original value"
    );
    assert!(!rust_has_neg, "rust optimizer should remove negates");

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
    let mut cpp_has_neg = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::SNegate {
            cpp_has_neg = true;
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(neg_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
        {
            cpp_const = true;
        }
    }
    assert!(
        cpp_const,
        "C++ spirv-opt should fold double negation to original value"
    );
    assert!(!cpp_has_neg, "C++ spirv-opt should remove negates");
}

#[test]
fn rust_and_cpp_simplify_add_commutativity() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, add_id) = build_commutative_add_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
    });
    assert!(rust_const, "rust optimizer should fold commutative add");

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
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
        {
            cpp_const = true;
        }
    }
    assert!(cpp_const, "C++ spirv-opt should fold commutative add");
    assert!(!cpp_has_add, "C++ spirv-opt should remove add");
}

#[test]
fn rust_and_cpp_fold_mul_by_one() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, mul_id) = build_mul_one_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
    });
    let rust_has_mul = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::IMul);
    assert!(
        rust_const,
        "rust optimizer should fold mul by one to original value"
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
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let mut cpp_const = false;
    let mut cpp_has_mul = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IMul {
            cpp_has_mul = true;
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
        {
            cpp_const = true;
        }
    }
    assert!(
        cpp_const,
        "C++ spirv-opt should fold mul by one to original value with same id"
    );
    assert!(!cpp_has_mul, "C++ spirv-opt should remove mul instruction");
}

#[test]
fn rust_and_cpp_fold_mul_by_neg_one() {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let (module_words, mul_id) = build_mul_neg_one_module();
    let rust_insts = extract_arith_block(&module_words);
    let rust_optimized =
        spirv_tools_opt::translate::optimize_arith_block(&rust_insts).expect("rust optimizer");
    let rust_const = rust_optimized.iter().any(|inst| {
        inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(6))]
    });
    let rust_negate = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::SNegate && inst.result_id == Some(mul_id));
    let rust_has_mul = rust_optimized
        .iter()
        .any(|inst| inst.class.opcode == Op::IMul);
    assert!(
        rust_const || rust_negate,
        "rust optimizer should turn mul by -1 into negate or folded const"
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
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&cpp_words, &mut loader).expect("parse cpp optimized");
    let module = loader.module();
    let mut cpp_const = false;
    let mut cpp_negate = false;
    let mut cpp_has_mul = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IMul {
            cpp_has_mul = true;
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(6))]
        {
            cpp_const = true;
        }
        if inst.class.opcode == Op::SNegate && inst.result_id == Some(mul_id) {
            cpp_negate = true;
        }
    }
    assert!(
        cpp_const || cpp_negate,
        "C++ spirv-opt should fold mul by -1"
    );
    assert!(!cpp_has_mul, "C++ spirv-opt should remove mul instruction");
}

fn build_const_add_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c2 = b.constant_bit32(int, 2);
    let c3 = b.constant_bit32(int, 3);
    let add = b.i_add(int, None, c2, c3).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
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

fn extract_simple_block(module_words: &[u32]) -> Vec<rspirv::dr::Instruction> {
    let mut loader = rspirv::dr::Loader::new();
    parse_words(module_words, &mut loader).expect("parse module");
    let module = loader.module();
    let block = &module.functions[0].blocks[0];
    module
        .types_global_values
        .iter()
        .chain(block.instructions.iter())
        .filter(|inst| {
            matches!(
                inst.class.opcode,
                Op::Constant | Op::IAdd | Op::ISub | Op::IMul
            )
        })
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

fn build_div_rem_module() -> Vec<u32> {
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
    let c9 = b.constant_bit32(int, 9);
    let c3 = b.constant_bit32(int, 3);
    let div = b.u_div(int, None, c9, c3).expect("div");
    let _ = b.u_mod(int, None, div, c3).expect("rem");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn build_mul_pow2_module() -> Vec<u32> {
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
    let x = b.variable(int, None, rspirv::spirv::StorageClass::Function, None);
    let c4 = b.constant_bit32(int, 4);
    let mul = b.i_mul(int, None, x, c4).expect("mul");
    let _ = b.i_add(int, None, mul, c4).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn build_mask_pow2_module() -> Vec<u32> {
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
    let x = b.constant_bit32(int, 13);
    let mask = b.constant_bit32(int, 7);
    let _ = b.bitwise_and(int, None, x, mask).expect("band");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn build_shift_chain_module() -> Vec<u32> {
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
    let x = b.constant_bit32(int, 128);
    let one = b.constant_bit32(int, 1);
    let two = b.constant_bit32(int, 2);
    let shr1 = b.shift_right_logical(int, None, x, one).expect("shr1");
    let _ = b.shift_right_logical(int, None, shr1, two).expect("shr2");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn build_arith_shift_chain_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 1); // signed
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.constant_bit32(int, 0xFFFFFF80); // -128
    let one = b.constant_bit32(int, 1);
    let two = b.constant_bit32(int, 2);
    let shr1 = b.shift_right_arithmetic(int, None, x, one).expect("shr1");
    let _ = b
        .shift_right_arithmetic(int, None, shr1, two)
        .expect("shr2");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn build_neg_chain_module() -> Vec<u32> {
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
    let neg1 = b.s_negate(int, None, c7).expect("neg");
    let neg2 = b.s_negate(int, None, neg1).expect("double neg");
    let _ = b.i_add(int, None, neg2, c7).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn build_triple_neg_chain_module() -> Vec<u32> {
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
    let c9 = b.constant_bit32(int, 9);
    let n1 = b.s_negate(int, None, c9).expect("neg1");
    let n2 = b.s_negate(int, None, n1).expect("neg2");
    let _ = b.s_negate(int, None, n2).expect("neg3");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn build_mask_then_shift_module() -> Vec<u32> {
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
    let x = b.constant_bit32(int, 29);
    let mask = b.constant_bit32(int, 7);
    let band = b.bitwise_and(int, None, x, mask).expect("band");
    let shift = b.constant_bit32(int, 1);
    let _ = b.shift_right_logical(int, None, band, shift).expect("shr");
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

fn build_zero_minus_module() -> (Vec<u32>, u32) {
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
    let c0 = b.constant_bit32(int, 0);
    let c9 = b.constant_bit32(int, 9);
    let sub = b.i_sub(int, None, c0, c9).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_sub_add_const_chain() -> (Vec<u32>, u32) {
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
    let c10 = b.constant_bit32(int, 10);
    let c3 = b.constant_bit32(int, 3);
    let sub = b.i_sub(int, None, c10, c3).expect("sub");
    let add = b.i_add(int, None, sub, c3).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_shared_addend_sub_chain() -> (Vec<u32>, u32) {
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
    let c5 = b.constant_bit32(int, 5);
    let c2 = b.constant_bit32(int, 2);
    let add_lhs = b.i_add(int, None, c4, c5).expect("add lhs");
    let add_rhs = b.i_add(int, None, c4, c2).expect("add rhs");
    let sub = b.i_sub(int, None, add_lhs, add_rhs).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_sub_add_chain() -> (Vec<u32>, u32) {
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
    let c10 = b.constant_bit32(int, 10);
    let c3 = b.constant_bit32(int, 3);
    let c1 = b.constant_bit32(int, 1);
    let sub1 = b.i_sub(int, None, c10, c3).expect("sub1");
    let sub2 = b.i_sub(int, None, c3, c1).expect("sub2");
    let add = b.i_add(int, None, sub1, sub2).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_commuted_factor_module() -> (Vec<u32>, u32) {
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
    let c2 = b.constant_bit32(int, 2);
    let c3 = b.constant_bit32(int, 3);
    let x = b.i_add(int, None, c2, c3).expect("x");
    let mul1 = b.i_mul(int, None, c2, x).expect("mul1");
    let mul2 = b.i_mul(int, None, x, c3).expect("mul2");
    let add = b.i_add(int, None, mul1, mul2).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_commuted_factor_sub_module() -> (Vec<u32>, u32) {
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
    let c1 = b.constant_bit32(int, 1);
    let c5 = b.constant_bit32(int, 5);
    let mul1 = b.i_mul(int, None, c4, c5).expect("mul1");
    let mul2 = b.i_mul(int, None, c5, c1).expect("mul2");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_mirror_add_sub_module() -> (Vec<u32>, u32) {
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
    let c4 = b.constant_bit32(int, 4);
    let sub = b.i_sub(int, None, c4, c7).expect("sub");
    let add = b.i_add(int, None, c7, sub).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_const_factor_sub_module() -> (Vec<u32>, u32) {
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
    let c6 = b.constant_bit32(int, 6);
    let c2 = b.constant_bit32(int, 2);
    let x = b.constant_bit32(int, 3);
    let mul1 = b.i_mul(int, None, c6, x).expect("mul1");
    let mul2 = b.i_mul(int, None, c2, x).expect("mul2");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_dist_const_mul_add_module() -> (Vec<u32>, u32) {
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
    let c5 = b.constant_bit32(int, 5);
    let c3 = b.constant_bit32(int, 3);
    let add = b.i_add(int, None, c5, c3).expect("add");
    let mul = b.i_mul(int, None, c4, add).expect("mul");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_dist_const_mul_sub_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c3 = b.constant_bit32(int, 3);
    let c10 = b.constant_bit32(int, 10);
    let c2 = b.constant_bit32(int, 2);
    let sub = b.i_sub(int, None, c10, c2).expect("sub");
    let mul = b.i_mul(int, None, c3, sub).expect("mul");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_affine_gcd_add_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let c12 = b.constant_bit32(int, 12);
    let x = b.constant_bit32(int, 4);
    let mul = b.i_mul(int, None, c6, x).expect("mul");
    let add = b.i_add(int, None, mul, c12).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_affine_gcd_sub_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c14 = b.constant_bit32(int, 14);
    let c21 = b.constant_bit32(int, 21);
    let x = b.constant_bit32(int, 2);
    let mul = b.i_mul(int, None, c14, x).expect("mul");
    let sub = b.i_sub(int, None, mul, c21).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_commuted_shared_addends_module() -> (Vec<u32>, u32) {
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
    let x = b.constant_bit32(int, 5);
    let y = b.constant_bit32(int, 8);
    let add1 = b.i_add(int, None, y, x).expect("add1");
    let add2 = b.i_add(int, None, x, y).expect("add2");
    let sub = b.i_sub(int, None, add1, add2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_shared_addends_const_diff_module() -> (Vec<u32>, u32) {
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
    let a = b.constant_bit32(int, 9);
    let b_const = b.constant_bit32(int, 3);
    let c_const = b.constant_bit32(int, 4);
    let add1 = b.i_add(int, None, a, b_const).expect("add1");
    let add2 = b.i_add(int, None, a, c_const).expect("add2");
    let sub = b.i_sub(int, None, add1, add2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_shared_addends_const_diff_positive_module() -> (Vec<u32>, u32) {
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
    let a = b.constant_bit32(int, 9);
    let b_const = b.constant_bit32(int, 4);
    let c_const = b.constant_bit32(int, 3);
    let add1 = b.i_add(int, None, a, b_const).expect("add1");
    let add2 = b.i_add(int, None, a, c_const).expect("add2");
    let sub = b.i_sub(int, None, add1, add2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_shared_symbolic_addends_const_diff_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let c5 = b.constant_bit32(int, 5);
    let c2 = b.constant_bit32(int, 2);
    let add1 = b.i_add(int, None, x, c5).expect("add1");
    let add2 = b.i_add(int, None, x, c2).expect("add2");
    let sub = b.i_sub(int, None, add1, add2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_shared_symbolic_addends_match_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let c7 = b.constant_bit32(int, 7);
    let add1 = b.i_add(int, None, x, c7).expect("add1");
    let add2 = b.i_add(int, None, x, c7).expect("add2");
    let sub = b.i_sub(int, None, add1, add2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_symbolic_factor_sub_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let c5 = b.constant_bit32(int, 5);
    let c2 = b.constant_bit32(int, 2);
    let mul1 = b.i_mul(int, None, x, c5).expect("mul1");
    let mul2 = b.i_mul(int, None, x, c2).expect("mul2");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_symbolic_factor_commuted_add_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int, int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let y = b.function_parameter(int).expect("param y");
    let z = b.function_parameter(int).expect("param z");
    let mul1 = b.i_mul(int, None, y, x).expect("mul1");
    let mul2 = b.i_mul(int, None, x, z).expect("mul2");
    let add = b.i_add(int, None, mul1, mul2).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add, x, y, z)
}

fn build_symbolic_factor_commuted_sub_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int, int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let y = b.function_parameter(int).expect("param y");
    let z = b.function_parameter(int).expect("param z");
    let mul1 = b.i_mul(int, None, y, x).expect("mul1");
    let mul2 = b.i_mul(int, None, z, x).expect("mul2");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x, y, z)
}

fn build_const_factor_commuted_add_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let y = b.function_parameter(int).expect("param y");
    let c4 = b.constant_bit32(int, 4);
    let mul1 = b.i_mul(int, None, c4, x).expect("mul1");
    let mul2 = b.i_mul(int, None, y, c4).expect("mul2");
    let add_inner = b.i_add(int, None, x, y).expect("add_inner");
    let add = b.i_add(int, None, mul1, mul2).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    // We expect factoring to leave an add of x+y and a mul by constant.
    (b.module().assemble(), add, add_inner)
}

fn build_const_factor_commuted_sub_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let y = b.function_parameter(int).expect("param y");
    let c6 = b.constant_bit32(int, 6);
    let mul1 = b.i_mul(int, None, c6, x).expect("mul1");
    let mul2 = b.i_mul(int, None, c6, y).expect("mul2");
    let sub_inner = b.i_sub(int, None, x, y).expect("sub_inner");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, sub_inner)
}

fn build_symbolic_const_factor_add_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let y = b.function_parameter(int).expect("param y");
    let c3 = b.constant_bit32(int, 3);
    let mul1 = b.i_mul(int, None, x, y).expect("mul1");
    let mul2 = b.i_mul(int, None, x, c3).expect("mul2");
    let add = b.i_add(int, None, mul1, mul2).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add, x, y, c3)
}

fn build_symbolic_const_factor_sub_module() -> (Vec<u32>, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let y = b.function_parameter(int).expect("param y");
    let c3 = b.constant_bit32(int, 3);
    let mul1 = b.i_mul(int, None, x, y).expect("mul1");
    let mul2 = b.i_mul(int, None, x, c3).expect("mul2");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x, y, c3)
}

fn build_mixed_const_factor_add_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let c2 = b.constant_bit32(int, 2);
    let c3 = b.constant_bit32(int, 3);
    let mul1 = b.i_mul(int, None, c2, x).expect("mul1");
    let mul2 = b.i_mul(int, None, x, c3).expect("mul2");
    let add = b.i_add(int, None, mul1, mul2).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add, x)
}

fn build_mixed_const_factor_sub_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let c2 = b.constant_bit32(int, 2);
    let c3 = b.constant_bit32(int, 3);
    let mul1 = b.i_mul(int, None, c2, x).expect("mul1");
    let mul2 = b.i_mul(int, None, x, c3).expect("mul2");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_mixed_const_factor_sub_pos_module() -> (Vec<u32>, u32, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let c3 = b.constant_bit32(int, 3);
    let c2 = b.constant_bit32(int, 2);
    let mul1 = b.i_mul(int, None, c3, x).expect("mul1");
    let mul2 = b.i_mul(int, None, c2, x).expect("mul2");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub, x)
}

fn build_zero_factor_add_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let y = b.function_parameter(int).expect("param y");
    let c0 = b.constant_bit32(int, 0);
    let mul1 = b.i_mul(int, None, c0, x).expect("mul1");
    let mul2 = b.i_mul(int, None, y, c0).expect("mul2");
    let add = b.i_add(int, None, mul1, mul2).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_zero_factor_sub_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int, int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let x = b.function_parameter(int).expect("param x");
    let y = b.function_parameter(int).expect("param y");
    let c0 = b.constant_bit32(int, 0);
    let mul1 = b.i_mul(int, None, c0, x).expect("mul1");
    let mul2 = b.i_mul(int, None, c0, y).expect("mul2");
    let sub = b.i_sub(int, None, mul1, mul2).expect("sub");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), sub)
}

fn build_add_negate_module() -> (Vec<u32>, u32) {
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
    let c5 = b.constant_bit32(int, 5);
    let neg = b.s_negate(int, None, c5).expect("neg");
    let add = b.i_add(int, None, c5, neg).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_add_sub_cancel_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let ca = b.constant_bit32(int, 42);
    let cb = b.constant_bit32(int, 5);
    let sub = b.i_sub(int, None, ca, cb).expect("sub");
    let add = b.i_add(int, None, sub, cb).expect("add");
    let _ = (ca, cb);
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_double_neg_module() -> (Vec<u32>, u32) {
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
    let neg1 = b.s_negate(int, None, c7).expect("neg1");
    let neg2 = b.s_negate(int, None, neg1).expect("neg2");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), neg2)
}

fn build_commutative_add_module() -> (Vec<u32>, u32) {
    // Build %add = 3 + 2 (commuted).
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
    let c3 = b.constant_bit32(int, 3);
    let c2 = b.constant_bit32(int, 2);
    let add = b.i_add(int, None, c3, c2).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_mul_one_module() -> (Vec<u32>, u32) {
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
    let c1 = b.constant_bit32(int, 1);
    let mul = b.i_mul(int, None, c7, c1).expect("mul");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_mul_neg_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _func = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let c_neg_one = b.constant_bit32(int, u32::MAX);
    let mul = b.i_mul(int, None, c6, c_neg_one).expect("mul");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
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
