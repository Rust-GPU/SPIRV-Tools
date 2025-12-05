use std::fs;
use std::path::PathBuf;
use std::process::Command;

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
