use std::io::Write;
use std::process::Command;

use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv;
use spirv_tools_cli::assembly::words_to_bytes;
use tempfile::NamedTempFile;

fn build_module_with_compiler_cmd(cmd: &str) -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    b.module_processed(cmd);

    let void = b.type_void();
    let fn_ty = b.type_function(void, vec![void]);
    let func_id = b
        .begin_function(void, None, spirv::FunctionControl::NONE, fn_ty)
        .expect("begin function");
    b.begin_block(None).expect("begin block");
    b.ret().expect("ret");
    b.end_function().expect("end function");
    b.entry_point(spirv::ExecutionModel::Fragment, func_id, "main", []);
    b.module().assemble()
}

#[test]
fn compiler_cmd_disabled_without_env() {
    let words = build_module_with_compiler_cmd("glslc -O shader.glsl");
    let mut file = NamedTempFile::new().expect("temp file");
    file.write_all(&words_to_bytes(&words))
        .expect("write module");

    let output = Command::new(env!("CARGO_BIN_EXE_spirv-objdump"))
        .arg("--compiler-cmd")
        .arg(file.path())
        .output()
        .expect("run spirv-objdump");
    assert!(
        !output.status.success(),
        "expected compiler-cmd to be disabled by default"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("unimplemented"),
        "expected unimplemented error, got: {stderr}"
    );
}

#[test]
fn compiler_cmd_enabled_with_env() {
    let words = build_module_with_compiler_cmd("glslc -O shader.glsl");
    let mut file = NamedTempFile::new().expect("temp file");
    file.write_all(&words_to_bytes(&words))
        .expect("write module");

    let output = Command::new(env!("CARGO_BIN_EXE_spirv-objdump"))
        .arg("--compiler-cmd")
        .arg(file.path())
        .env("SPIRV_TOOLS_ENABLE_COMPILER_CMD", "1")
        .output()
        .expect("run spirv-objdump");
    assert!(
        output.status.success(),
        "expected compiler-cmd to succeed when enabled: {:?}",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("glslc") && stdout.contains("-O"),
        "unexpected compiler command output: {stdout}"
    );
}
