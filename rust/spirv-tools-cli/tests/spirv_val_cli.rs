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
