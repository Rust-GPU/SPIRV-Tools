use std::fs;
use std::path::PathBuf;
use std::process::Command;

use tempfile::tempdir;

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

fn rust_bin(name: &str) -> PathBuf {
    let key = format!("CARGO_BIN_EXE_{name}");
    std::env::var(&key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("missing test binary path for {name} (expected {key})"))
}

fn corpus() -> [&'static str; 3] {
    [
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
    ]
}

#[test]
fn spirv_fuzz_cli_matches_cpp_output_when_available() {
    let Some(cpp_tool) = find_cpp_tool("SPIRV_CPP_FUZZ", "spirv-fuzz") else {
        eprintln!("SPIRV_CPP_FUZZ not set and spirv-fuzz not found on PATH; skipping parity");
        return;
    };

    let rust_fuzz = rust_bin("spirv-fuzz");
    let rust_as = rust_bin("spirv-as");

    let dir = tempdir().expect("tempdir");

    for (idx, text) in corpus().iter().enumerate() {
        let asm_path = dir.path().join(format!("module_{idx}.spvasm"));
        let input_spv = dir.path().join(format!("module_{idx}.spv"));
        fs::write(&asm_path, text).expect("write asm");

        let status = Command::new(&rust_as)
            .arg(&asm_path)
            .arg("-o")
            .arg(&input_spv)
            .status()
            .expect("run spirv-as");
        assert!(status.success(), "rust spirv-as failed: {status:?}");

        let rust_out = dir.path().join(format!("rust-out-{idx}.spv"));
        let cpp_out = dir.path().join(format!("cpp-out-{idx}.spv"));

        let rust_status = Command::new(&rust_fuzz)
            .arg(&input_spv)
            .arg("-o")
            .arg(&rust_out)
            .status()
            .expect("run rust spirv-fuzz");
        let cpp_status = Command::new(&cpp_tool)
            .arg(&input_spv)
            .arg("-o")
            .arg(&cpp_out)
            .status()
            .expect("run cpp spirv-fuzz");

        assert!(
            rust_status.success(),
            "rust spirv-fuzz failed on corpus {idx}: {rust_status:?}"
        );
        assert!(
            cpp_status.success(),
            "cpp spirv-fuzz failed on corpus {idx}: {cpp_status:?}"
        );

        let rust_bytes = fs::read(&rust_out).expect("read rust fuzz output");
        let cpp_bytes = fs::read(&cpp_out).expect("read cpp fuzz output");
        assert_eq!(rust_bytes, cpp_bytes, "spirv-fuzz outputs differed on corpus {idx}");
    }
}
