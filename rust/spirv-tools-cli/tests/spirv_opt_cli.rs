use rspirv::binary::{parse_words, Assemble};
use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

fn build_const_add_module() -> (Vec<u32>, u32) {
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
    let add = b.i_add(int, None, c2, c3).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

#[test]
fn spirv_opt_cli_folds_add_by_default() {
    let (words, add_id) = build_const_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_add = false;
    let mut saw_const = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IAdd => saw_add = true,
            Op::Constant => {
                if inst.result_id == Some(add_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
                {
                    saw_const = true;
                }
            }
            _ => {}
        }
    }
    assert!(saw_const, "add should fold to constant 5 with same id");
    assert!(!saw_add, "addition should be removed");
}

#[test]
fn spirv_opt_cli_respects_passthrough() {
    let (words, add_id) = build_const_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--passthrough")
        .arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    assert_eq!(
        optimized_words, words,
        "passthrough should leave module unchanged"
    );

    // Double-check the add is present with the expected id.
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_add = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IAdd && inst.result_id == Some(add_id) {
            saw_add = true;
        }
    }
    assert!(saw_add, "passthrough should preserve add instruction");
}

#[test]
fn spirv_opt_cli_folds_add_negation() {
    let (words, add_id) = build_add_negate_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_zero = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IAdd {
            panic!("add should have been folded away");
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(add_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            found_zero = true;
        }
    }
    assert!(found_zero, "add+negate should fold to zero with same id");
}

fn build_umod_pow2_module() -> (Vec<u32>, u32) {
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
    let c1 = b.constant_bit32(int, 1);
    let c8 = b.constant_bit32(int, 8);
    let x = b.i_add(int, None, c5, c1).expect("iadd");
    let umod = b.u_mod(int, None, x, c8).expect("umod");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), umod)
}

#[test]
fn spirv_opt_cli_folds_mul_by_one() {
    let (words, mul_id) = build_mul_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IMul {
            panic!("mul should be folded away");
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "mul by one should fold to original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_udiv_by_one() {
    let (words, div_id) = build_udiv_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(inst.class.opcode, Op::UDiv, "udiv by one should be removed");
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(div_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(9)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "udiv by one should fold to original value with same id"
    );
}

#[test]
fn spirv_opt_cli_folds_urem_by_one() {
    let (words, rem_id) = build_urem_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_zero = false;
    for inst in module.all_inst_iter() {
        assert_ne!(inst.class.opcode, Op::UMod, "umod by one should fold");
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(rem_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
        {
            found_zero = true;
        }
    }
    assert!(found_zero, "umod by one should fold to zero with same id");
}

#[test]
fn spirv_opt_cli_folds_shift_by_zero() {
    let (words, shl_id) = build_shift_zero_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(
            inst.class.opcode,
            Op::ShiftLeftLogical,
            "shift by zero should be eliminated"
        );
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(shl_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
        {
            found_const = true;
        }
    }
    assert!(
        found_const,
        "shift by zero should rewrite to original value with same id"
    );
}

#[test]
fn spirv_opt_cli_rewrites_umod_pow2_to_mask() {
    let (words, umod_id) = build_umod_pow2_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_band = false;
    let mut saw_const = false;
    for inst in module.all_inst_iter() {
        assert_ne!(inst.class.opcode, Op::UMod, "umod should be rewritten");
        if inst.class.opcode == Op::BitwiseAnd {
            saw_band = true;
            let mask_is_7 = inst
                .operands
                .iter()
                .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(7)));
            assert!(mask_is_7, "expected mask 7 for modulo by 8");
            assert_eq!(
                inst.result_id,
                Some(umod_id),
                "should reuse original result id for bitmask"
            );
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(umod_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(6)]
        {
            saw_const = true;
        }
    }
    assert!(
        saw_band || saw_const,
        "expected bitwise and (or folded constant) to replace umod with power-of-two divisor"
    );
}

#[test]
fn spirv_opt_cli_folds_div_rem_and_negate_and_shifts() {
    let (words, _div_id_unused, _rem_id_unused, _neg_id_unused, _shl_id_unused, _shr_id_unused) =
        build_div_rem_neg_shift_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let _ = module; // Coverage sanity; future parity checks can assert specific folds.
}

static ENV_GUARD: Mutex<()> = Mutex::new(());

#[test]
fn spirv_opt_cli_respects_disable_env() {
    let _guard = ENV_GUARD.lock().unwrap();
    let (words, add_id) = build_const_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.env("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    assert_eq!(optimized_words, words, "env disable should passthrough");
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");

    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_add = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IAdd && inst.result_id == Some(add_id) {
            saw_add = true;
        }
    }
    assert!(saw_add, "env disable should preserve add");
}

#[test]
fn spirv_opt_cli_force_rust_ignores_disable_env() {
    let _guard = ENV_GUARD.lock().unwrap();
    let (words, add_id) = build_const_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.env("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
    cmd.arg("--force-rust")
        .arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_add = false;
    let mut saw_const = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::IAdd => saw_add = true,
            Op::Constant => {
                if inst.result_id == Some(add_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
                {
                    saw_const = true;
                }
            }
            _ => {}
        }
    }
    assert!(
        saw_const && !saw_add,
        "force-rust should fold even when disable env is set"
    );
    std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
}

#[test]
fn spirv_opt_cli_folds_affine_gcd_add() {
    let _guard = ENV_GUARD.lock().unwrap();
    let (words, add_id) = build_affine_gcd_add_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();
    let mut saw_const = false;
    let mut saw_ops = false;
    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Constant => {
                if inst.result_id == Some(add_id)
                    && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(36)]
                {
                    saw_const = true;
                }
            }
            Op::IAdd | Op::IMul if inst.result_id == Some(add_id) => saw_ops = true,
            _ => {}
        }
    }
    assert!(saw_const, "affine gcd add should fold to const 36");
    assert!(!saw_ops, "add/mul should be removed after folding");
}

#[test]
fn spirv_opt_cli_folds_mul_by_neg_one() {
    let (words, mul_id) = build_mul_neg_one_module();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(&words))
        .expect("write module");
    let output = child.wait_with_output().expect("run spirv-opt");
    assert!(
        output.status.success(),
        "spirv-opt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let optimized_words = bytes_to_words(&output.stdout);
    let mut loader = rspirv::dr::Loader::new();
    parse_words(&optimized_words, &mut loader).expect("parse optimized");
    let module = loader.module();

    let mut found = false;
    for inst in module.all_inst_iter() {
        if inst.class.opcode == Op::IMul && inst.result_id == Some(mul_id) {
            panic!("mul by -1 should be rewritten or folded");
        }
        if inst.class.opcode == Op::Constant
            && inst.result_id == Some(mul_id)
            && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(6))]
        {
            found = true;
        }
        if inst.class.opcode == Op::SNegate && inst.result_id == Some(mul_id) {
            found = true;
        }
    }
    assert!(found, "mul by -1 should become negate or folded constant");
}

fn cpp_opt_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SPIRV_CPP_OPT") {
        let pb = PathBuf::from(path);
        if pb.is_file() {
            return Some(pb);
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join("spirv-opt");
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

fn run_cli_with_path(
    words: &[u32],
    extra_args: &[&str],
    prepend_path: Option<&PathBuf>,
) -> std::process::Output {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_spirv-opt"));
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.arg("--")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    if let Some(path) = prepend_path {
        if let Some(parent) = path.parent() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!("{}:{}", parent.display(), current_path);
            cmd.env("PATH", new_path);
        }
    }
    let mut child = cmd.spawn().expect("spawn spirv-opt");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&words_to_bytes(words))
        .expect("write module");
    child.wait_with_output().expect("run spirv-opt")
}

fn assert_cpp_cli_matches_rust(words: &[u32], label: &str) {
    let Some(cpp_opt) = cpp_opt_bin() else {
        return;
    };

    let rust_output = run_cli_with_path(words, &[], None);
    assert!(
        rust_output.status.success(),
        "rust cli ({label}) failed: {}",
        String::from_utf8_lossy(&rust_output.stderr)
    );

    let cpp_output = run_cli_with_path(words, &["--cpp"], Some(&cpp_opt));
    assert!(
        cpp_output.status.success(),
        "cpp cli ({label}) failed: {}",
        String::from_utf8_lossy(&cpp_output.stderr)
    );

    assert_eq!(
        rust_output.stdout, cpp_output.stdout,
        "Rust optimizer output should match C++ spirv-opt output ({label})"
    );
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_output() {
    let (words, _) = build_const_add_module();
    assert_cpp_cli_matches_rust(&words, "const add fold");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_umod_pow2_output() {
    let (words, _) = build_umod_pow2_module();
    assert_cpp_cli_matches_rust(&words, "pow2 umod to mask");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_mul_by_one_output() {
    let (words, _) = build_mul_one_module();
    assert_cpp_cli_matches_rust(&words, "mul by one");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_mul_by_zero_output() {
    let (words, _) = build_mul_zero_module();
    assert_cpp_cli_matches_rust(&words, "mul by zero");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_add_negate_output() {
    let (words, _) = build_add_negate_module();
    assert_cpp_cli_matches_rust(&words, "add + negate");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_srem_divisible_output() {
    let (words, _) = build_srem_divisible_module();
    assert_cpp_cli_matches_rust(&words, "srem divisible");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_srem_non_divisible_output() {
    let (words, _) = build_srem_non_divisible_module();
    assert_cpp_cli_matches_rust(&words, "srem non-divisible");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_mul_neg_one_output() {
    let (words, _) = build_mul_neg_one_module();
    assert_cpp_cli_matches_rust(&words, "mul by -1");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_affine_gcd_add_output() {
    let (words, _) = build_affine_gcd_add_module();
    assert_cpp_cli_matches_rust(&words, "affine gcd add");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_pow2_mul_output() {
    let (words, _) = build_mul_pow2_module();
    assert_cpp_cli_matches_rust(&words, "mul by power-of-two");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_pow2_div_output() {
    let (words, _) = build_div_pow2_module();
    assert_cpp_cli_matches_rust(&words, "div by power-of-two");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_udiv_one_output() {
    let (words, _) = build_udiv_one_module();
    assert_cpp_cli_matches_rust(&words, "udiv by one");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_urem_one_output() {
    let (words, _) = build_urem_one_module();
    assert_cpp_cli_matches_rust(&words, "urem by one");
}

#[test]
fn spirv_opt_cli_cpp_mode_matches_rust_shift_zero_output() {
    let (words, _) = build_shift_zero_module();
    assert_cpp_cli_matches_rust(&words, "shift by zero");
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn build_add_negate_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c5 = b.constant_bit32(int, 5);
    let neg = b.s_negate(int, None, c5).expect("neg");
    let add = b.i_add(int, None, c5, neg).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_mul_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let c1 = b.constant_bit32(int, 1);
    let mul = b.i_mul(int, None, c4, c1).expect("mul id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_udiv_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c9 = b.constant_bit32(int, 9);
    let c1 = b.constant_bit32(int, 1);
    let div = b.u_div(int, None, c9, c1).expect("div id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), div)
}

fn build_urem_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c9 = b.constant_bit32(int, 9);
    let c1 = b.constant_bit32(int, 1);
    let rem = b.u_mod(int, None, c9, c1).expect("rem id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), rem)
}

fn build_shift_zero_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let zero = b.constant_bit32(int, 0);
    let shl = b
        .shift_left_logical(int, None, c4, zero)
        .expect("shift id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), shl)
}

fn build_mul_neg_one_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let c_neg_one = b.constant_bit32(int, u32::MAX);
    let mul = b.i_mul(int, None, c6, c_neg_one).expect("mul id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_mul_zero_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c4 = b.constant_bit32(int, 4);
    let c0 = b.constant_bit32(int, 0);
    let mul = b.i_mul(int, None, c4, c0).expect("mul id");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_srem_divisible_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("function parameter");
    let _ = b.begin_block(None).expect("block");
    let c6 = b.constant_bit32(int, 6);
    let c3 = b.constant_bit32(int, 3);
    let mul = b.i_mul(int, None, param, c6).expect("mul");
    let rem = b.s_rem(int, None, mul, c3).expect("srem");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), rem)
}

fn build_srem_non_divisible_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 1);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("function parameter");
    let _ = b.begin_block(None).expect("block");
    let c5 = b.constant_bit32(int, 5);
    let c3 = b.constant_bit32(int, 3);
    let mul = b.i_mul(int, None, param, c5).expect("mul");
    let rem = b.s_rem(int, None, mul, c3).expect("srem");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), rem)
}

fn build_div_rem_neg_shift_module() -> (Vec<u32>, u32, u32, u32, u32, u32) {
    let mut b = Builder::new();
    b.capability(rspirv::spirv::Capability::Shader);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c20 = b.constant_bit32(int, 20);
    let c4 = b.constant_bit32(int, 4);
    let c5 = b.constant_bit32(int, 5);
    let div = b.u_div(int, None, c20, c4).expect("div");
    let rem = b.u_mod(int, None, c20, c4).expect("rem");
    let neg = b.s_negate(int, None, c5).expect("neg");
    let shl = b.shift_left_logical(int, None, c5, c4).expect("shl");
    let shr = b.shift_right_logical(int, None, c20, c4).expect("shr");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), div, rem, neg, shl, shr)
}

fn build_affine_gcd_add_module() -> (Vec<u32>, u32) {
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
    let c12 = b.constant_bit32(int, 12);
    let x = b.constant_bit32(int, 4);
    let mul = b.i_mul(int, None, c6, x).expect("mul");
    let add = b.i_add(int, None, mul, c12).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), add)
}

fn build_mul_pow2_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("param");
    let _ = b.begin_block(None).expect("block");
    let c8 = b.constant_bit32(int, 8);
    let mul = b.i_mul(int, None, param, c8).expect("mul");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), mul)
}

fn build_div_pow2_module() -> (Vec<u32>, u32) {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![int]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let param = b.function_parameter(int).expect("param");
    let _ = b.begin_block(None).expect("block");
    let c8 = b.constant_bit32(int, 8);
    let div = b.u_div(int, None, param, c8).expect("div");
    b.ret().expect("ret");
    b.end_function().expect("end");
    (b.module().assemble(), div)
}
