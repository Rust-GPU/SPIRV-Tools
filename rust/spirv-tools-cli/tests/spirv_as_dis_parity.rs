use std::fs;
use std::path::PathBuf;
use std::process::Command;

use spirv_tools_core::assembly::assemble_text;
use tempfile::tempdir;

/// Try to locate a C++ tool (spirv-as/spirv-dis) via env or PATH.
fn find_cpp_tool(env_var: &str, binary: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_var) {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
        eprintln!("{env_var} is set but not a file: {candidate:?}");
    }
    which::which(binary).ok()
}

fn simple_module() -> &'static str {
    r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#
}

#[test]
fn spirv_as_matches_cpp_binary_output() {
    let Some(cpp_as) = find_cpp_tool("SPIRV_CPP_AS", "spirv-as") else {
        eprintln!("SPIRV_CPP_AS not set and spirv-as not found on PATH; skipping parity");
        return;
    };

    let dir = tempdir().expect("temp dir");
    let asm_path = dir.path().join("module.spvasm");
    fs::write(&asm_path, simple_module()).expect("write asm");
    let rust_bin = dir.path().join("rust.spv");
    let cpp_bin = dir.path().join("cpp.spv");

    let rust_status = Command::new(env!("CARGO_BIN_EXE_spirv-as"))
        .arg(&asm_path)
        .arg("-o")
        .arg(&rust_bin)
        .status()
        .expect("run rust spirv-as");
    assert!(rust_status.success(), "rust spirv-as failed: {rust_status:?}");

    let cpp_status = Command::new(&cpp_as)
        .arg(&asm_path)
        .arg("-o")
        .arg(&cpp_bin)
        .status()
        .expect("run cpp spirv-as");
    assert!(cpp_status.success(), "cpp spirv-as failed: {cpp_status:?}");

    let rust_bytes = fs::read(&rust_bin).expect("read rust binary");
    let cpp_bytes = fs::read(&cpp_bin).expect("read cpp binary");
    assert_eq!(
        rust_bytes, cpp_bytes,
        "Rust spirv-as output differed from C++ output"
    );
}

fn module_with_numeric_ids() -> &'static str {
    r#"
; Expect numeric ids to be preserved when requested
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %1 "main"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%1 = OpFunction %2 None %3
%4 = OpLabel
OpReturn
OpFunctionEnd
"#
}

#[test]
fn spirv_as_preserves_numeric_ids_like_cpp() {
    let Some(cpp_as) = find_cpp_tool("SPIRV_CPP_AS", "spirv-as") else {
        eprintln!("SPIRV_CPP_AS not set and spirv-as not found on PATH; skipping parity");
        return;
    };

    let dir = tempdir().expect("temp dir");
    let asm_path = dir.path().join("module_ids.spvasm");
    fs::write(&asm_path, module_with_numeric_ids()).expect("write asm");
    let rust_bin = dir.path().join("rust_ids.spv");
    let cpp_bin = dir.path().join("cpp_ids.spv");

    let rust_status = Command::new(env!("CARGO_BIN_EXE_spirv-as"))
        .arg(&asm_path)
        .arg("--preserve-numeric-ids")
        .arg("-o")
        .arg(&rust_bin)
        .status()
        .expect("run rust spirv-as");
    assert!(rust_status.success(), "rust spirv-as failed: {rust_status:?}");

    let cpp_status = Command::new(&cpp_as)
        .arg(&asm_path)
        .arg("--preserve-numeric-ids")
        .arg("-o")
        .arg(&cpp_bin)
        .status()
        .expect("run cpp spirv-as");
    assert!(cpp_status.success(), "cpp spirv-as failed: {cpp_status:?}");

    let rust_bytes = fs::read(&rust_bin).expect("read rust binary");
    let cpp_bytes = fs::read(&cpp_bin).expect("read cpp binary");
    assert_eq!(
        rust_bytes, cpp_bytes,
        "Rust spirv-as preserve-numeric-ids output differed from C++ output"
    );
}

fn normalize_text(text: &str) -> Vec<u32> {
    assemble_text(text).expect("assemble disassembled text")
}

#[test]
fn spirv_disassembles_equivalently_to_cpp() {
    let Some(cpp_as) = find_cpp_tool("SPIRV_CPP_AS", "spirv-as") else {
        eprintln!("SPIRV_CPP_AS not set and spirv-as not found on PATH; skipping parity");
        return;
    };
    let Some(cpp_dis) = find_cpp_tool("SPIRV_CPP_DIS", "spirv-dis") else {
        eprintln!("SPIRV_CPP_DIS not set and spirv-dis not found on PATH; skipping parity");
        return;
    };

    let dir = tempdir().expect("temp dir");
    let asm_path = dir.path().join("module.spvasm");
    let cpp_bin = dir.path().join("cpp.spv");
    fs::write(&asm_path, simple_module()).expect("write asm");

    let cpp_status = Command::new(&cpp_as)
        .arg(&asm_path)
        .arg("-o")
        .arg(&cpp_bin)
        .status()
        .expect("run cpp spirv-as");
    assert!(cpp_status.success(), "cpp spirv-as failed: {cpp_status:?}");

    let rust_out = Command::new(env!("CARGO_BIN_EXE_spirv-dis"))
        .arg(&cpp_bin)
        .output()
        .expect("run rust spirv-dis");
    assert!(
        rust_out.status.success(),
        "rust spirv-dis failed: {rust_out:?}"
    );
    let rust_text = String::from_utf8_lossy(&rust_out.stdout);

    let cpp_out = Command::new(&cpp_dis)
        .arg(&cpp_bin)
        .output()
        .expect("run cpp spirv-dis");
    assert!(
        cpp_out.status.success(),
        "cpp spirv-dis failed: {cpp_out:?}"
    );
    let cpp_text = String::from_utf8_lossy(&cpp_out.stdout);

    let rust_words = normalize_text(&rust_text);
    let cpp_words = normalize_text(&cpp_text);
    assert_eq!(
        rust_words, cpp_words,
        "Rust spirv-dis output reassembles differently than C++ output"
    );
}
