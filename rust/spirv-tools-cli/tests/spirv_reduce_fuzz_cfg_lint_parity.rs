use std::fs;
use std::path::PathBuf;
use std::process::Command;

use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv;
use spirv_tools_cli::assembly::words_to_bytes;
use tempfile::tempdir;

/// Try to locate a C++ tool (spirv-reduce/fuzz/cfg/lint) via env or PATH.
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

fn help_has_binary_name(output: &str, binary: &str) -> bool {
    output.contains(binary)
}

fn rust_bin(name: &str) -> PathBuf {
    let key = format!("CARGO_BIN_EXE_{name}");
    std::env::var(&key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("missing test binary path for {name} (expected {key})"))
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

fn write_invalid_spirv() -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("temp dir");
    let bad_path = dir.path().join("invalid.spv");
    fs::write(&bad_path, &[0u8, 1, 2, 3]).expect("write invalid spv");
    (dir, bad_path)
}

fn write_source_module(path: &PathBuf) {
    let mut b = Builder::new();
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

    let file_id = b.string("main.hlsl");
    b.source(
        spirv::SourceLanguage::GLSL,
        450,
        Some(file_id),
        Some("void main() {}"),
    );
    b.source_continued(" // tail");

    let void = b.type_void();
    let fn_ty = b.type_function(void, vec![void]);
    let func = b
        .begin_function(void, None, spirv::FunctionControl::NONE, fn_ty)
        .expect("begin function");
    b.begin_block(None).expect("entry block");
    b.ret().expect("ret");
    b.end_function().expect("end fn");
    b.entry_point(spirv::ExecutionModel::Fragment, func, "main", &[]);

    let words = b.module().assemble();
    fs::write(path, &words_to_bytes(&words)).expect("write module");
}

fn simple_module_lines() -> [&'static str; 9] {
    [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Vertex %main \"main\"",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
}

fn simple_module_text() -> String {
    simple_module_lines().join("\n")
}

fn assemble_with_rust_as(path: &PathBuf) {
    let asm = simple_module_text();
    let dir = path.parent().expect("parent dir");
    let asm_path = dir.join("module.spvasm");
    fs::write(&asm_path, asm).expect("write asm");
    let status = Command::new(rust_bin("spirv-as"))
        .arg(&asm_path)
        .arg("-o")
        .arg(path)
        .status()
        .expect("run rust spirv-as");
    assert!(status.success(), "spirv-as failed: {status:?}");
}

#[test]
fn spirv_reduce_matches_cpp_output() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_REDUCE", "spirv-reduce") else {
        eprintln!("SPIRV_CPP_REDUCE not set and spirv-reduce not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    assemble_with_rust_as(&bin_path);

    let rust_out = dir.path().join("rust-out.spv");
    let cpp_out = dir.path().join("cpp-out.spv");

    let rust = Command::new(rust_bin("spirv-reduce"))
        .arg(&bin_path)
        .arg("-o")
        .arg(&rust_out)
        .output()
        .expect("run rust spirv-reduce");
    let cpp = Command::new(&cpp_tool)
        .arg(&bin_path)
        .arg("-o")
        .arg(&cpp_out)
        .output()
        .expect("run cpp spirv-reduce");

    assert!(
        rust.status.success(),
        "rust spirv-reduce failed: {:?}",
        rust.status
    );
    assert!(
        cpp.status.success(),
        "cpp spirv-reduce failed: {:?}",
        cpp.status
    );

    let rust_bytes = fs::read(&rust_out).expect("read rust reduce output");
    let cpp_bytes = fs::read(&cpp_out).expect("read cpp reduce output");
    assert_eq!(rust_bytes, cpp_bytes, "spirv-reduce outputs differed");
}

#[test]
fn spirv_fuzz_matches_cpp_output() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_FUZZ", "spirv-fuzz") else {
        eprintln!("SPIRV_CPP_FUZZ not set and spirv-fuzz not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    assemble_with_rust_as(&bin_path);

    let rust_out = dir.path().join("rust-out.spv");
    let cpp_out = dir.path().join("cpp-out.spv");

    let rust = Command::new(rust_bin("spirv-fuzz"))
        .arg(&bin_path)
        .arg("-o")
        .arg(&rust_out)
        .output()
        .expect("run rust spirv-fuzz");
    let cpp = Command::new(&cpp_tool)
        .arg(&bin_path)
        .arg("-o")
        .arg(&cpp_out)
        .output()
        .expect("run cpp spirv-fuzz");

    assert!(
        rust.status.success(),
        "rust spirv-fuzz failed: {:?}",
        rust.status
    );
    assert!(
        cpp.status.success(),
        "cpp spirv-fuzz failed: {:?}",
        cpp.status
    );

    let rust_bytes = fs::read(&rust_out).expect("read rust fuzz output");
    let cpp_bytes = fs::read(&cpp_out).expect("read cpp fuzz output");
    assert_eq!(rust_bytes, cpp_bytes, "spirv-fuzz outputs differed");
}

fn assert_error_parity(rust_cmd: &mut Command, cpp_cmd: &mut Command, label: &str) {
    let rust = rust_cmd.output().expect("run rust tool");
    let cpp = cpp_cmd.output().expect("run cpp tool");

    assert!(
        !rust.status.success(),
        "expected rust {label} to fail on invalid binary"
    );
    assert!(
        !cpp.status.success(),
        "expected cpp {label} to fail on invalid binary"
    );
    assert_eq!(
        rust.status.code(),
        cpp.status.code(),
        "error exit codes should match between rust and cpp {label}"
    );
}

#[test]
fn spirv_reduce_help_and_version_match_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_REDUCE", "spirv-reduce") else {
        eprintln!("SPIRV_CPP_REDUCE not set and spirv-reduce not found on PATH; skipping parity");
        return;
    };
    let rust = rust_bin("spirv-reduce");
    help_and_version_parity(rust.to_str().expect("utf8 path"), &cpp_tool, "spirv-reduce");
}

#[test]
fn spirv_fuzz_help_and_version_match_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_FUZZ", "spirv-fuzz") else {
        eprintln!("SPIRV_CPP_FUZZ not set and spirv-fuzz not found on PATH; skipping parity");
        return;
    };
    let rust = rust_bin("spirv-fuzz");
    help_and_version_parity(rust.to_str().expect("utf8 path"), &cpp_tool, "spirv-fuzz");
}

#[test]
fn spirv_cfg_help_and_version_match_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_CFG", "spirv-cfg") else {
        eprintln!("SPIRV_CPP_CFG not set and spirv-cfg not found on PATH; skipping parity");
        return;
    };
    let rust = rust_bin("spirv-cfg");
    help_and_version_parity(rust.to_str().expect("utf8 path"), &cpp_tool, "spirv-cfg");
}

#[test]
fn spirv_lint_help_and_version_match_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_LINT", "spirv-lint") else {
        eprintln!("SPIRV_CPP_LINT not set and spirv-lint not found on PATH; skipping parity");
        return;
    };
    let rust = rust_bin("spirv-lint");
    help_and_version_parity(rust.to_str().expect("utf8 path"), &cpp_tool, "spirv-lint");
}

#[test]
fn spirv_objdump_help_and_version_match_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_OBJDUMP", "spirv-objdump") else {
        eprintln!("SPIRV_CPP_OBJDUMP not set and spirv-objdump not found on PATH; skipping parity");
        return;
    };
    let rust = rust_bin("spirv-objdump");
    help_and_version_parity(
        rust.to_str().expect("utf8 path"),
        &cpp_tool,
        "spirv-objdump",
    );
}

#[test]
fn spirv_size_help_and_version_match_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_SIZE", "spirv-size") else {
        eprintln!("SPIRV_CPP_SIZE not set and spirv-size not found on PATH; skipping parity");
        return;
    };
    let rust = rust_bin("spirv-size");
    help_and_version_parity(rust.to_str().expect("utf8 path"), &cpp_tool, "spirv-size");
}

#[test]
fn spirv_objdump_reports_errors_like_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_OBJDUMP", "spirv-objdump") else {
        eprintln!("SPIRV_CPP_OBJDUMP not set and spirv-objdump not found on PATH; skipping parity");
        return;
    };
    let rust_bin = rust_bin("spirv-objdump");
    let (_dir, bad_path) = write_invalid_spirv();

    assert_error_parity(
        Command::new(&rust_bin).arg(&bad_path),
        Command::new(&cpp_tool).arg(&bad_path),
        "spirv-objdump",
    );
}

#[test]
fn spirv_size_reports_errors_like_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_SIZE", "spirv-size") else {
        eprintln!("SPIRV_CPP_SIZE not set and spirv-size not found on PATH; skipping parity");
        return;
    };
    let rust_bin = rust_bin("spirv-size");
    let (_dir, bad_path) = write_invalid_spirv();

    assert_error_parity(
        Command::new(&rust_bin).arg(&bad_path),
        Command::new(&cpp_tool).arg(&bad_path),
        "spirv-size",
    );
}

#[test]
fn spirv_reduce_reports_errors_like_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_REDUCE", "spirv-reduce") else {
        eprintln!("SPIRV_CPP_REDUCE not set and spirv-reduce not found on PATH; skipping parity");
        return;
    };
    let rust_bin = rust_bin("spirv-reduce");
    let (dir, bad_path) = write_invalid_spirv();
    let rust_out = dir.path().join("rust-out.spv");
    let cpp_out = dir.path().join("cpp-out.spv");

    assert_error_parity(
        Command::new(&rust_bin)
            .arg(&bad_path)
            .arg("-o")
            .arg(&rust_out),
        Command::new(&cpp_tool)
            .arg(&bad_path)
            .arg("-o")
            .arg(&cpp_out),
        "spirv-reduce",
    );
}

#[test]
fn spirv_fuzz_reports_errors_like_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_FUZZ", "spirv-fuzz") else {
        eprintln!("SPIRV_CPP_FUZZ not set and spirv-fuzz not found on PATH; skipping parity");
        return;
    };
    let rust_bin = rust_bin("spirv-fuzz");
    let (dir, bad_path) = write_invalid_spirv();
    let rust_out = dir.path().join("rust-out.spv");
    let cpp_out = dir.path().join("cpp-out.spv");

    assert_error_parity(
        Command::new(&rust_bin)
            .arg(&bad_path)
            .arg("-o")
            .arg(&rust_out),
        Command::new(&cpp_tool)
            .arg(&bad_path)
            .arg("-o")
            .arg(&cpp_out),
        "spirv-fuzz",
    );
}

#[test]
fn spirv_cfg_reports_errors_like_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_CFG", "spirv-cfg") else {
        eprintln!("SPIRV_CPP_CFG not set and spirv-cfg not found on PATH; skipping parity");
        return;
    };
    let rust_bin = rust_bin("spirv-cfg");
    let (_dir, bad_path) = write_invalid_spirv();

    assert_error_parity(
        Command::new(&rust_bin).arg(&bad_path),
        Command::new(&cpp_tool).arg(&bad_path),
        "spirv-cfg",
    );
}

#[test]
fn spirv_cfg_matches_cpp_output() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_CFG", "spirv-cfg") else {
        eprintln!("SPIRV_CPP_CFG not set and spirv-cfg not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    assemble_with_rust_as(&bin_path);

    let rust = Command::new(rust_bin("spirv-cfg"))
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-cfg");
    let cpp = Command::new(&cpp_tool)
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-cfg");

    assert!(
        rust.status.success(),
        "rust spirv-cfg failed: {:?}",
        rust.status
    );
    assert!(
        cpp.status.success(),
        "cpp spirv-cfg failed: {:?}",
        cpp.status
    );

    let rust_out = String::from_utf8_lossy(&rust.stdout);
    let cpp_out = String::from_utf8_lossy(&cpp.stdout);
    assert_eq!(rust_out, cpp_out, "spirv-cfg outputs differed");
}

#[test]
fn spirv_lint_reports_errors_like_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_LINT", "spirv-lint") else {
        eprintln!("SPIRV_CPP_LINT not set and spirv-lint not found on PATH; skipping parity");
        return;
    };
    let rust_bin = rust_bin("spirv-lint");
    let (_dir, bad_path) = write_invalid_spirv();

    assert_error_parity(
        Command::new(&rust_bin).arg(&bad_path),
        Command::new(&cpp_tool).arg(&bad_path),
        "spirv-lint",
    );
}

#[test]
fn spirv_lint_matches_cpp_output() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_LINT", "spirv-lint") else {
        eprintln!("SPIRV_CPP_LINT not set and spirv-lint not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    assemble_with_rust_as(&bin_path);

    let rust = Command::new(rust_bin("spirv-lint"))
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-lint");
    let cpp = Command::new(&cpp_tool)
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-lint");

    assert!(
        rust.status.success(),
        "rust spirv-lint failed: {:?}",
        rust.status
    );
    assert!(
        cpp.status.success(),
        "cpp spirv-lint failed: {:?}",
        cpp.status
    );

    let rust_out = String::from_utf8_lossy(&rust.stdout);
    let cpp_out = String::from_utf8_lossy(&cpp.stdout);
    assert_eq!(rust_out, cpp_out, "spirv-lint outputs differed");
}

#[test]
fn spirv_objdump_matches_cpp_output() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_OBJDUMP", "spirv-objdump") else {
        eprintln!("SPIRV_CPP_OBJDUMP not set and spirv-objdump not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    assemble_with_rust_as(&bin_path);

    let rust = Command::new(rust_bin("spirv-objdump"))
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-objdump");
    let cpp = Command::new(&cpp_tool)
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-objdump");

    assert!(rust.status.success(), "rust spirv-objdump failed");
    assert!(cpp.status.success(), "cpp spirv-objdump failed");
    let rust_stdout = String::from_utf8_lossy(&rust.stdout);
    let cpp_stdout = String::from_utf8_lossy(&cpp.stdout);
    assert_eq!(
        rust_stdout, cpp_stdout,
        "spirv-objdump outputs differed between rust and cpp"
    );
}

#[test]
fn spirv_size_matches_cpp_output() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_SIZE", "spirv-size") else {
        eprintln!("SPIRV_CPP_SIZE not set and spirv-size not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    assemble_with_rust_as(&bin_path);

    let rust = Command::new(rust_bin("spirv-size"))
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-size");
    let cpp = Command::new(&cpp_tool)
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-size");

    assert!(rust.status.success(), "rust spirv-size failed");
    assert!(cpp.status.success(), "cpp spirv-size failed");
    let rust_stdout = String::from_utf8_lossy(&rust.stdout);
    let cpp_stdout = String::from_utf8_lossy(&cpp.stdout);
    assert_eq!(
        rust_stdout, cpp_stdout,
        "spirv-size outputs differed between rust and cpp"
    );
}

#[test]
fn spirv_objdump_source_list_matches_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_OBJDUMP", "spirv-objdump") else {
        eprintln!("SPIRV_CPP_OBJDUMP not set and spirv-objdump not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    write_source_module(&bin_path);

    let rust = Command::new(rust_bin("spirv-objdump"))
        .arg("--source")
        .arg("--list")
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-objdump --source --list");
    let cpp = Command::new(&cpp_tool)
        .arg("--source")
        .arg("--list")
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-objdump --source --list");

    assert!(rust.status.success(), "rust spirv-objdump failed");
    assert!(cpp.status.success(), "cpp spirv-objdump failed");
    let rust_stdout = String::from_utf8_lossy(&rust.stdout);
    let cpp_stdout = String::from_utf8_lossy(&cpp.stdout);
    assert_eq!(
        rust_stdout, cpp_stdout,
        "spirv-objdump --source --list outputs differed"
    );
}

#[test]
fn spirv_objdump_source_stdout_matches_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_OBJDUMP", "spirv-objdump") else {
        eprintln!("SPIRV_CPP_OBJDUMP not set and spirv-objdump not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    write_source_module(&bin_path);

    let rust = Command::new(rust_bin("spirv-objdump"))
        .arg("--source")
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-objdump --source");
    let cpp = Command::new(&cpp_tool)
        .arg("--source")
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-objdump --source");

    assert!(rust.status.success(), "rust spirv-objdump failed");
    assert!(cpp.status.success(), "cpp spirv-objdump failed");
    let rust_stdout = String::from_utf8_lossy(&rust.stdout);
    let cpp_stdout = String::from_utf8_lossy(&cpp.stdout);
    assert_eq!(
        rust_stdout, cpp_stdout,
        "spirv-objdump --source outputs differed"
    );
}

#[test]
fn spirv_objdump_source_outdir_matches_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_OBJDUMP", "spirv-objdump") else {
        eprintln!("SPIRV_CPP_OBJDUMP not set and spirv-objdump not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let outdir = dir.path().join("out");
    let bin_path = dir.path().join("module.spv");
    write_source_module(&bin_path);

    let rust = Command::new(rust_bin("spirv-objdump"))
        .arg("--source")
        .arg("--outdir")
        .arg(&outdir)
        .arg("--force")
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-objdump --source --outdir");
    let cpp = Command::new(&cpp_tool)
        .arg("--source")
        .arg("--outdir")
        .arg(&outdir)
        .arg("--force")
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-objdump --source --outdir");

    assert!(rust.status.success(), "rust spirv-objdump failed");
    assert!(cpp.status.success(), "cpp spirv-objdump failed");

    let rust_stdout = String::from_utf8_lossy(&rust.stdout);
    let cpp_stdout = String::from_utf8_lossy(&cpp.stdout);
    assert_eq!(
        rust_stdout, cpp_stdout,
        "spirv-objdump --source --outdir outputs differed"
    );

    let exported = outdir.join("main.hlsl");
    let contents = fs::read_to_string(&exported).expect("read exported source");
    assert!(
        contents.contains("void main() {}"),
        "exported source missing body: {contents}"
    );
    assert!(
        contents.contains("tail"),
        "exported source missing continued text: {contents}"
    );
}

#[test]
fn spirv_objdump_source_ignores_empty_like_cpp() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_OBJDUMP", "spirv-objdump") else {
        eprintln!("SPIRV_CPP_OBJDUMP not set and spirv-objdump not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");

    // Build a module with a source filename but no literal payload.
    let mut b = Builder::new();
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let file_id = b.string("empty.hlsl");
    b.source(
        spirv::SourceLanguage::GLSL,
        450,
        Some(file_id),
        None::<String>,
    );
    let void = b.type_void();
    let fn_ty = b.type_function(void, vec![void]);
    let func = b
        .begin_function(void, None, spirv::FunctionControl::NONE, fn_ty)
        .expect("begin function");
    b.begin_block(None).expect("entry block");
    b.ret().expect("ret");
    b.end_function().expect("end fn");
    b.entry_point(spirv::ExecutionModel::Fragment, func, "main", &[]);
    let words = b.module().assemble();
    fs::write(&bin_path, &words_to_bytes(&words)).expect("write module");

    let rust = Command::new(rust_bin("spirv-objdump"))
        .arg("--source")
        .arg("--list")
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-objdump --source --list");
    let cpp = Command::new(&cpp_tool)
        .arg("--source")
        .arg("--list")
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-objdump --source --list");

    assert!(rust.status.success(), "rust spirv-objdump failed");
    assert!(cpp.status.success(), "cpp spirv-objdump failed");
    let rust_stdout = String::from_utf8_lossy(&rust.stdout);
    let cpp_stdout = String::from_utf8_lossy(&cpp.stdout);
    assert_eq!(
        rust_stdout, cpp_stdout,
        "spirv-objdump --source --list for empty source differed"
    );
}

#[test]
fn spirv_objdump_compiler_cmd_matches_cpp_failure() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_OBJDUMP", "spirv-objdump") else {
        eprintln!("SPIRV_CPP_OBJDUMP not set and spirv-objdump not found on PATH; skipping parity");
        return;
    };
    let dir = tempdir().expect("temp dir");
    let bin_path = dir.path().join("module.spv");
    write_source_module(&bin_path);

    let rust = Command::new(rust_bin("spirv-objdump"))
        .arg("--compiler-cmd")
        .arg(&bin_path)
        .output()
        .expect("run rust spirv-objdump --compiler-cmd");
    let cpp = Command::new(&cpp_tool)
        .arg("--compiler-cmd")
        .arg(&bin_path)
        .output()
        .expect("run cpp spirv-objdump --compiler-cmd");

    assert!(
        !rust.status.success(),
        "rust spirv-objdump should fail for compiler-cmd"
    );
    assert!(
        !cpp.status.success(),
        "cpp spirv-objdump should fail for compiler-cmd"
    );
    assert_eq!(
        rust.status.code(),
        cpp.status.code(),
        "compiler-cmd exit codes should match"
    );
    let rust_err = String::from_utf8_lossy(&rust.stderr);
    let cpp_err = String::from_utf8_lossy(&cpp.stderr);
    assert!(
        rust_err.to_lowercase().contains("unimplemented"),
        "expected rust stderr to mention unimplemented: {rust_err}"
    );
    assert!(
        cpp_err.to_lowercase().contains("unimplemented"),
        "expected cpp stderr to mention unimplemented: {cpp_err}"
    );
}
