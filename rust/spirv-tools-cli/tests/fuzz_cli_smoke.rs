use std::process::Command;

use tempfile::tempdir;

/// Run the Rust fuzz binary on a tiny module; skip cleanly if the binary is absent.
#[test]
fn spirv_fuzz_cli_smoke() {
    let rust_bin = match std::env::var("CARGO_BIN_EXE_spirv-fuzz") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("spirv-fuzz binary not built in this profile; skipping");
            return;
        }
    };

    let tmp = tempdir().expect("tempdir");
    let input = tmp.path().join("input.spv");
    // Minimal valid header-only module.
    std::fs::write(
        &input,
        [
            0x03, 0x02, 0x23, 0x07, // magic number little-endian bytes
            0x00, 0x00, 0x01, 0x00, // version
            0x00, 0x00, 0x00, 0x00, // generator
            0x01, 0x00, 0x00, 0x00, // bound
            0x00, 0x00, 0x00, 0x00, // reserved
        ],
    )
    .expect("write input");

    let status = Command::new(&rust_bin)
        .arg(&input)
        .arg("--set-env=Universal1_6")
        .status()
        .expect("run spirv-fuzz");

    assert!(
        status.success(),
        "spirv-fuzz CLI should succeed on minimal module"
    );
}
