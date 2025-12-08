use std::process::Command;

/// Optional hyperfine smoke for spirv-fuzz; skip if hyperfine or binary missing.
#[test]
fn spirv_fuzz_hyperfine_smoke() {
    let rust_bin = match std::env::var("CARGO_BIN_EXE_spirv-fuzz") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("spirv-fuzz binary not built in this profile; skipping hyperfine");
            return;
        }
    };

    // Check if hyperfine exists.
    if Command::new("hyperfine").arg("--version").output().is_err() {
        eprintln!("hyperfine not installed; skipping benchmark smoke");
        return;
    }

    let status = Command::new("hyperfine")
        .arg("--runs=1")
        .arg("--warmup=0")
        .arg(format!("{} --help", rust_bin))
        .status();

    if let Ok(status) = status {
        if !status.success() {
            panic!("hyperfine run failed with status {:?}", status);
        }
    } else {
        eprintln!("hyperfine invocation failed; skipping");
    }
}
