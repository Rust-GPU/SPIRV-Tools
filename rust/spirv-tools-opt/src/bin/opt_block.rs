//! SPIR-V basic block optimizer binary.
//!
//! This binary optimizes SPIR-V modules using the egglog-based e-graph optimizer.

use clap::Parser;
use spirv_tools_opt::translate::optimize_words;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Optimize a SPIR-V module using the Rust e-graph optimizer."
)]
struct Args {
    /// Input SPIR-V binary; required.
    input: PathBuf,
    /// Optional output path; writes to stdout when omitted.
    output: Option<PathBuf>,
    /// Force the Rust optimizer even if SPIRV_TOOLS_DISABLE_RUST_OPT is set.
    #[arg(long, default_value_t = false)]
    force_rust: bool,
    /// Skip optimization and emit the input unchanged.
    #[arg(long, default_value_t = false)]
    passthrough: bool,
    /// Disable the global (multi-block) optimizer path even when available.
    /// NOTE: This flag is accepted for backwards compatibility but currently has no effect.
    #[arg(long, default_value_t = false)]
    disable_global: bool,
    /// Force the global (multi-block) optimizer path when possible.
    /// NOTE: This flag is accepted for backwards compatibility but currently has no effect.
    #[arg(long, default_value_t = false)]
    force_global: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let input_bytes = fs::read(&args.input)?;
    let words = bytes_to_words(&input_bytes)?;

    let optimized = optimize_module_cli(
        &words,
        args.force_rust,
        args.passthrough,
        args.disable_global,
        args.force_global,
    )?;
    let output_bytes = words_to_bytes(&optimized);

    if let Some(path) = args.output {
        fs::write(path, output_bytes)?;
    } else {
        std::io::stdout().write_all(&output_bytes)?;
    }
    Ok(())
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    if bytes.len() % 4 != 0 {
        return Err("input length is not a multiple of 4 bytes".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            u32::from_le_bytes(arr)
        })
        .collect())
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn optimize_module_cli(
    words: &[u32],
    force_rust: bool,
    passthrough: bool,
    _disable_global: bool,
    _force_global: bool,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // Handle passthrough mode
    if passthrough {
        return Ok(words.to_vec());
    }

    // Check environment variables for disable/force
    let force_env = env::var_os("SPIRV_TOOLS_FORCE_RUST_OPT").is_some();
    let disable_env = matches!(env::var("SPIRV_TOOLS_DISABLE_RUST_OPT"), Ok(v) if v == "1");

    // If disabled and not forced, passthrough
    if disable_env && !force_rust && !force_env {
        return Ok(words.to_vec());
    }

    // Run the optimizer
    match optimize_words(words) {
        Ok(optimized) => Ok(optimized),
        Err(e) => Err(Box::new(e) as Box<dyn std::error::Error>),
    }
}
