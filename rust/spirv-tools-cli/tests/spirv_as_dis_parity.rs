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

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn disassembly_corpus() -> Vec<&'static str> {
    vec![
        simple_module(),
        r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 2 3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#,
        r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%ptr_out = OpTypePointer Output %float
%out = OpVariable %ptr_out Output
%const = OpConstant %float 1.0
%main = OpFunction %void None %fn
%entry = OpLabel
OpStore %out %const
OpReturn
OpFunctionEnd
"#,
        r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %vmain "vmain"
OpEntryPoint Fragment %fmain "fmain"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%vmain = OpFunction %void None %fn
%ventry = OpLabel
OpReturn
OpFunctionEnd
%fmain = OpFunction %void None %fn
%fentry = OpLabel
OpReturn
OpFunctionEnd
"#,
    ]
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

fn assembly_corpus() -> Vec<&'static str> {
    disassembly_corpus()
}

#[test]
fn spirv_as_matches_cpp_binary_output() {
    let Some(cpp_as) = find_cpp_tool("SPIRV_CPP_AS", "spirv-as") else {
        eprintln!("SPIRV_CPP_AS not set and spirv-as not found on PATH; skipping parity");
        return;
    };

    for module in assembly_corpus() {
        let dir = tempdir().expect("temp dir");
        let asm_path = dir.path().join("module.spvasm");
        fs::write(&asm_path, module).expect("write asm");
        let rust_bin = dir.path().join("rust.spv");
        let cpp_bin = dir.path().join("cpp.spv");

        let rust_status = Command::new(env!("CARGO_BIN_EXE_spirv-as"))
            .arg(&asm_path)
            .arg("-o")
            .arg(&rust_bin)
            .status()
            .expect("run rust spirv-as");
        assert!(
            rust_status.success(),
            "rust spirv-as failed: {rust_status:?} for module:\n{module}"
        );

        let cpp_status = Command::new(&cpp_as)
            .arg(&asm_path)
            .arg("-o")
            .arg(&cpp_bin)
            .status()
            .expect("run cpp spirv-as");
        assert!(
            cpp_status.success(),
            "cpp spirv-as failed: {cpp_status:?} for module:\n{module}"
        );

        let rust_bytes = fs::read(&rust_bin).expect("read rust binary");
        let cpp_bytes = fs::read(&cpp_bin).expect("read cpp binary");
        assert_eq!(
            rust_bytes, cpp_bytes,
            "Rust spirv-as output differed from C++ output for module:\n{module}"
        );
    }
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
    assert!(
        rust_status.success(),
        "rust spirv-as failed: {rust_status:?}"
    );

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

#[test]
fn spirv_as_reports_errors_like_cpp() {
    let Some(cpp_as) = find_cpp_tool("SPIRV_CPP_AS", "spirv-as") else {
        eprintln!("SPIRV_CPP_AS not set and spirv-as not found on PATH; skipping parity");
        return;
    };

    let dir = tempdir().expect("temp dir");
    let bad_path = dir.path().join("invalid.spvasm");
    fs::write(
        &bad_path,
        "%void = OpTypeVoid\nOpEntryPoint Vertex %main \"main\"",
    )
    .expect("write asm");

    let rust = Command::new(env!("CARGO_BIN_EXE_spirv-as"))
        .arg(&bad_path)
        .output()
        .expect("run rust spirv-as");
    let cpp = Command::new(&cpp_as)
        .arg(&bad_path)
        .output()
        .expect("run cpp spirv-as");

    assert!(
        !rust.status.success(),
        "expected rust spirv-as to fail on invalid module"
    );
    assert!(
        !cpp.status.success(),
        "expected cpp spirv-as to fail on invalid module"
    );
    assert_eq!(
        rust.status.code(),
        cpp.status.code(),
        "error exit codes should match between rust and cpp spirv-as"
    );
}

#[test]
fn spirv_dis_reports_errors_like_cpp() {
    let Some(cpp_dis) = find_cpp_tool("SPIRV_CPP_DIS", "spirv-dis") else {
        eprintln!("SPIRV_CPP_DIS not set and spirv-dis not found on PATH; skipping parity");
        return;
    };

    let dir = tempdir().expect("temp dir");
    let bad_path = dir.path().join("invalid.spv");
    // Write an invalid SPIR-V binary (wrong magic).
    fs::write(&bad_path, &[0u8, 1, 2, 3]).expect("write invalid spv");

    let rust = Command::new(env!("CARGO_BIN_EXE_spirv-dis"))
        .arg(&bad_path)
        .output()
        .expect("run rust spirv-dis");
    let cpp = Command::new(&cpp_dis)
        .arg(&bad_path)
        .output()
        .expect("run cpp spirv-dis");

    assert!(
        !rust.status.success(),
        "expected rust spirv-dis to fail on invalid binary"
    );
    assert!(
        !cpp.status.success(),
        "expected cpp spirv-dis to fail on invalid binary"
    );
    assert_eq!(
        rust.status.code(),
        cpp.status.code(),
        "error exit codes should match between rust and cpp spirv-dis"
    );
}

#[test]
fn spirv_dis_outputs_match_cpp() {
    let Some(cpp_dis) = find_cpp_tool("SPIRV_CPP_DIS", "spirv-dis") else {
        eprintln!("SPIRV_CPP_DIS not set and spirv-dis not found on PATH; skipping parity");
        return;
    };

    for module in disassembly_corpus() {
        let dir = tempdir().expect("temp dir");
        let bin_path = dir.path().join("module.spv");
        let binary = assemble_text(module).expect("assemble module");
        fs::write(&bin_path, words_to_bytes(&binary)).expect("write binary");

        let rust = Command::new(env!("CARGO_BIN_EXE_spirv-dis"))
            .arg(&bin_path)
            .output()
            .expect("run rust spirv-dis");
        assert!(
            rust.status.success(),
            "rust spirv-dis failed: {:?}",
            rust.status
        );

        let cpp = Command::new(&cpp_dis)
            .arg(&bin_path)
            .output()
            .expect("run cpp spirv-dis");
        assert!(
            cpp.status.success(),
            "cpp spirv-dis failed: {:?}",
            cpp.status
        );

        let rust_text = String::from_utf8_lossy(&rust.stdout).trim().to_owned();
        let cpp_text = String::from_utf8_lossy(&cpp.stdout).trim().to_owned();
        assert_eq!(
            rust_text, cpp_text,
            "Rust spirv-dis output differed from C++ output for module:\n{module}"
        );
    }
}

#[test]
fn spirv_text_round_trip_matches_cpp() {
    let Some(cpp_as) = find_cpp_tool("SPIRV_CPP_AS", "spirv-as") else {
        eprintln!("SPIRV_CPP_AS not set and spirv-as not found on PATH; skipping parity");
        return;
    };
    let Some(cpp_dis) = find_cpp_tool("SPIRV_CPP_DIS", "spirv-dis") else {
        eprintln!("SPIRV_CPP_DIS not set and spirv-dis not found on PATH; skipping parity");
        return;
    };

    for module in disassembly_corpus() {
        let dir = tempdir().expect("temp dir");
        let asm_path = dir.path().join("module.spvasm");
        fs::write(&asm_path, module).expect("write asm");
        let rust_bin = dir.path().join("rust.spv");
        let cpp_bin = dir.path().join("cpp.spv");

        let rust_status = Command::new(env!("CARGO_BIN_EXE_spirv-as"))
            .arg(&asm_path)
            .arg("-o")
            .arg(&rust_bin)
            .status()
            .expect("run rust spirv-as");
        assert!(
            rust_status.success(),
            "rust spirv-as failed: {rust_status:?} for module:\n{module}"
        );

        let cpp_status = Command::new(&cpp_as)
            .arg(&asm_path)
            .arg("-o")
            .arg(&cpp_bin)
            .status()
            .expect("run cpp spirv-as");
        assert!(
            cpp_status.success(),
            "cpp spirv-as failed: {cpp_status:?} for module:\n{module}"
        );

        let rust_dis = Command::new(env!("CARGO_BIN_EXE_spirv-dis"))
            .arg(&rust_bin)
            .output()
            .expect("run rust spirv-dis");
        assert!(
            rust_dis.status.success(),
            "rust spirv-dis failed: {:?} for module:\n{module}",
            rust_dis.status
        );

        let cpp_dis_out = Command::new(&cpp_dis)
            .arg(&cpp_bin)
            .output()
            .expect("run cpp spirv-dis");
        assert!(
            cpp_dis_out.status.success(),
            "cpp spirv-dis failed: {:?} for module:\n{module}",
            cpp_dis_out.status
        );

        let rust_text = String::from_utf8_lossy(&rust_dis.stdout).trim().to_owned();
        let cpp_text = String::from_utf8_lossy(&cpp_dis_out.stdout).trim().to_owned();
        assert_eq!(
            rust_text, cpp_text,
            "Rust round-trip text differed from C++ for module:\n{module}"
        );
    }
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

fn help_has_binary_name(output: &str, binary: &str) -> bool {
    output.contains(binary)
}

fn help_and_version_parity(rust_bin: &str, cpp_bin: &PathBuf, name: &str) {
    let rust_help = Command::new(rust_bin)
        .arg("--help")
        .output()
        .expect("run rust --help");
    let cpp_help = Command::new(cpp_bin)
        .arg("--help")
        .output()
        .expect("run cpp --help");

    assert!(rust_help.status.success());
    assert!(cpp_help.status.success());
    let rust_help_out = String::from_utf8_lossy(&rust_help.stdout);
    let cpp_help_out = String::from_utf8_lossy(&cpp_help.stdout);
    assert!(
        help_has_binary_name(&rust_help_out, name),
        "rust --help did not mention binary name"
    );
    assert!(
        help_has_binary_name(&cpp_help_out, name),
        "cpp --help did not mention binary name"
    );

    let rust_version = Command::new(rust_bin)
        .arg("--version")
        .output()
        .expect("run rust --version");
    let cpp_version = Command::new(cpp_bin)
        .arg("--version")
        .output()
        .expect("run cpp --version");

    assert!(rust_version.status.success());
    assert!(cpp_version.status.success());
    let rust_version_out = String::from_utf8_lossy(&rust_version.stdout);
    let cpp_version_out = String::from_utf8_lossy(&cpp_version.stdout);
    assert!(
        rust_version_out.contains("SPIRV-Tools"),
        "rust version missing SPIRV-Tools tag: {rust_version_out}"
    );
    assert!(
        cpp_version_out.contains("SPIRV-Tools"),
        "cpp version missing SPIRV-Tools tag: {cpp_version_out}"
    );
}

#[test]
fn spirv_as_help_and_version_match_cpp() {
    let Some(cpp_as) = find_cpp_tool("SPIRV_CPP_AS", "spirv-as") else {
        eprintln!("SPIRV_CPP_AS not set and spirv-as not found on PATH; skipping parity");
        return;
    };

    help_and_version_parity(env!("CARGO_BIN_EXE_spirv-as"), &cpp_as, "spirv-as");
}

#[test]
fn spirv_dis_help_and_version_match_cpp() {
    let Some(cpp_dis) = find_cpp_tool("SPIRV_CPP_DIS", "spirv-dis") else {
        eprintln!("SPIRV_CPP_DIS not set and spirv-dis not found on PATH; skipping parity");
        return;
    };

    help_and_version_parity(env!("CARGO_BIN_EXE_spirv-dis"), &cpp_dis, "spirv-dis");
}

#[test]
fn spirv_val_help_and_version_match_cpp() {
    let Some(cpp_val) = find_cpp_tool("SPIRV_CPP_VAL", "spirv-val") else {
        eprintln!("SPIRV_CPP_VAL not set and spirv-val not found on PATH; skipping parity");
        return;
    };

    help_and_version_parity(env!("CARGO_BIN_EXE_spirv-val"), &cpp_val, "spirv-val");
}

#[test]
fn spirv_opt_help_and_version_match_cpp() {
    let Some(cpp_opt) = find_cpp_tool("SPIRV_CPP_OPT", "spirv-opt") else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not found on PATH; skipping parity");
        return;
    };

    help_and_version_parity(env!("CARGO_BIN_EXE_spirv-opt"), &cpp_opt, "spirv-opt");
}

#[test]
fn spirv_val_reports_errors_like_cpp() {
    let Some(cpp_val) = find_cpp_tool("SPIRV_CPP_VAL", "spirv-val") else {
        eprintln!("SPIRV_CPP_VAL not set and spirv-val not found on PATH; skipping parity");
        return;
    };

    let dir = tempdir().expect("temp dir");
    let bad_path = dir.path().join("invalid.spv");
    fs::write(&bad_path, &[0u8, 1, 2, 3]).expect("write invalid spv");

    let rust = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .arg(&bad_path)
        .output()
        .expect("run rust spirv-val");
    let cpp = Command::new(&cpp_val)
        .arg(&bad_path)
        .output()
        .expect("run cpp spirv-val");

    assert!(
        !rust.status.success(),
        "expected rust spirv-val to fail on invalid binary"
    );
    assert!(
        !cpp.status.success(),
        "expected cpp spirv-val to fail on invalid binary"
    );
    assert_eq!(
        rust.status.code(),
        cpp.status.code(),
        "error exit codes should match between rust and cpp spirv-val"
    );
}

#[test]
fn spirv_opt_reports_errors_like_cpp() {
    let Some(cpp_opt) = find_cpp_tool("SPIRV_CPP_OPT", "spirv-opt") else {
        eprintln!("SPIRV_CPP_OPT not set and spirv-opt not found on PATH; skipping parity");
        return;
    };

    let dir = tempdir().expect("temp dir");
    let bad_path = dir.path().join("invalid.spv");
    fs::write(&bad_path, &[0u8, 1, 2, 3]).expect("write invalid spv");

    let rust = Command::new(env!("CARGO_BIN_EXE_spirv-opt"))
        .arg(&bad_path)
        .output()
        .expect("run rust spirv-opt");
    let cpp = Command::new(&cpp_opt)
        .arg(&bad_path)
        .output()
        .expect("run cpp spirv-opt");

    assert!(
        !rust.status.success(),
        "expected rust spirv-opt to fail on invalid binary"
    );
    assert!(
        !cpp.status.success(),
        "expected cpp spirv-opt to fail on invalid binary"
    );
    assert_eq!(
        rust.status.code(),
        cpp.status.code(),
        "error exit codes should match between rust and cpp spirv-opt"
    );
}
