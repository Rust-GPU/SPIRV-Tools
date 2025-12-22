use std::fs;
use std::process::Command;

use tempfile::tempdir;

/// Ensure the fuzz CLI reports errors on empty input; skip if the binary isn't built.
#[test]
fn spirv_fuzz_cli_rejects_empty_input() {
    let rust_bin = match std::env::var("CARGO_BIN_EXE_spirv-fuzz") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("spirv-fuzz binary not built in this profile; skipping");
            return;
        }
    };

    let tmp = tempdir().expect("tempdir");
    let empty = tmp.path().join("empty.spv");
    fs::write(&empty, []).expect("write empty file");

    let status = Command::new(&rust_bin)
        .arg(&empty)
        .status()
        .expect("run spirv-fuzz");

    assert!(
        !status.success(),
        "spirv-fuzz should reject empty input binaries"
    );
}
