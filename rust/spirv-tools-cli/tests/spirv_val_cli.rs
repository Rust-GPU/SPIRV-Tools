use std::io::Write;
use std::process::Command;

use spirv_tools_cli::assembly::words_to_bytes;
use spirv_tools_core::assembly::assemble_text;
use tempfile::NamedTempFile;

fn write_binary(text: &str) -> NamedTempFile {
    let binary = assemble_text(text).expect("assemble text");
    let bytes = words_to_bytes(&binary);
    let mut file = NamedTempFile::new().expect("temp file");
    file.write_all(&bytes).expect("write binary");
    file
}

#[test]
fn spirv_val_cli_succeeds_on_valid_module() {
    let file = write_binary(
        r#"
OpCapability Shader
OpMemoryModel Logical Simple
OpEntryPoint Vertex %main "main"
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#,
    );

    let status = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .arg(file.path())
        .status()
        .expect("run spirv-val");
    assert!(status.success(), "expected success, got {status:?}");
}

#[test]
fn spirv_val_cli_reports_failure() {
    let file = write_binary("%void = OpTypeVoid");
    let status = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .arg(file.path())
        .status()
        .expect("run spirv-val");
    assert!(
        !status.success(),
        "expected validation failure to return non-zero, got {status:?}"
    );
}

fn spirv_val_supports_force_rust() -> bool {
    let help = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .arg("--help")
        .output()
        .expect("run spirv-val --help");
    let stdout = String::from_utf8_lossy(&help.stdout);
    stdout.contains("--force-rust-validator")
}

fn spirv_val_supports_prefer_flags() -> bool {
    let help = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .arg("--help")
        .output()
        .expect("run spirv-val --help");
    let stdout = String::from_utf8_lossy(&help.stdout);
    stdout.contains("--prefer-rust-validator") && stdout.contains("--prefer-cpp-validator")
}

fn reorder_extension_to_end(mut words: Vec<u32>) -> Vec<u32> {
    let mut idx = 5; // skip header
    let mut ext_slice: Option<(usize, usize)> = None;
    while idx < words.len() {
        let wc = (words[idx] >> 16) as usize;
        let opcode = (words[idx] & 0xffff) as u16;
        if opcode == rspirv::spirv::Op::Extension as u16 {
            ext_slice = Some((idx, wc));
            break;
        }
        idx += wc;
    }
    if let Some((start, len)) = ext_slice {
        let extension: Vec<u32> = words.drain(start..start + len).collect();
        words.extend(extension);
    }
    words
}

#[test]
fn spirv_val_cli_rust_validator_reports_layout_error() {
    if !spirv_val_supports_force_rust() {
        eprintln!("spirv-val binary does not support --force-rust-validator; skipping");
        return;
    }

    let text = r#"
OpCapability Shader
OpExtension "SPV_KHR_shader_clock"
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
    let words = assemble_text(text).expect("assemble text");
    let misordered = reorder_extension_to_end(words);
    let bytes = words_to_bytes(&misordered);
    let mut file = NamedTempFile::new().expect("temp file");
    file.write_all(&bytes).expect("write binary");

    let output = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .arg("--force-rust-validator")
        .arg(file.path())
        .output()
        .expect("run spirv-val");
    assert!(
        !output.status.success(),
        "expected layout failure from rust validator, got {output:?}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("out of order") || stderr.contains("LayoutOutOfOrder"),
        "expected layout error message, got: {}",
        stderr
    );
}

#[test]
fn spirv_val_cli_accepts_prefer_rust_validator_flag() {
    if !spirv_val_supports_prefer_flags() {
        eprintln!("spirv-val binary does not support prefer flags; skipping");
        return;
    }

    let file = write_binary(
        r#"
OpCapability Shader
OpMemoryModel Logical Simple
OpEntryPoint Vertex %main "main"
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#,
    );

    let status = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .arg("--prefer-rust-validator")
        .arg(file.path())
        .status()
        .expect("run spirv-val");
    assert!(status.success(), "expected success, got {status:?}");
}

#[test]
fn spirv_val_cli_accepts_prefer_cpp_validator_flag() {
    if !spirv_val_supports_prefer_flags() {
        eprintln!("spirv-val binary does not support prefer flags; skipping");
        return;
    }

    let file = write_binary(
        r#"
OpCapability Shader
OpMemoryModel Logical Simple
OpEntryPoint Vertex %main "main"
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#,
    );

    let status = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .arg("--prefer-cpp-validator")
        .arg(file.path())
        .status()
        .expect("run spirv-val");
    assert!(status.success(), "expected success, got {status:?}");
}

fn simple_module() -> NamedTempFile {
    write_binary(
        r#"
OpCapability Shader
OpMemoryModel Logical Simple
OpEntryPoint Vertex %main "main"
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#,
    )
}

#[test]
fn spirv_val_cli_honors_env_prefer_rust() {
    let file = simple_module();
    let status = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .env("SPIRV_TOOLS_PREFER_RUST_VALIDATOR", "1")
        .arg(file.path())
        .status()
        .expect("run spirv-val");
    assert!(status.success(), "expected success, got {status:?}");
}

#[test]
fn spirv_val_cli_honors_env_prefer_cpp() {
    let file = simple_module();
    let status = Command::new(env!("CARGO_BIN_EXE_spirv-val"))
        .env("SPIRV_TOOLS_PREFER_CPP_VALIDATOR", "1")
        .arg(file.path())
        .status()
        .expect("run spirv-val");
    assert!(status.success(), "expected success, got {status:?}");
}
