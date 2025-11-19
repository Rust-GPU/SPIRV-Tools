use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use spirv_tools_cli::disassemble::InputSource;
use spirv_tools_cli::validation::{run_validate, ValidateConfig};

/// Validate a SPIR-V binary module.
#[derive(Debug, Parser)]
#[command(name = "spirv-val", about = "Validate a SPIR-V binary module")]
struct Args {
    /// Input SPIR-V binary. Use '-' or omit to read from stdin.
    #[arg(value_name = "INPUT", default_value = "-")]
    input: String,

    /// Target environment (e.g., 'spv1.6', 'vulkan1.3').
    #[arg(long = "target-env", value_name = "ENV")]
    target_env: Option<String>,
}

fn main() {
    let args = Args::parse();
    let input = if args.input == "-" {
        InputSource::Stdin
    } else {
        InputSource::Path(PathBuf::from(&args.input))
    };

    let config = ValidateConfig {
        input,
        target_env: args.target_env,
    };

    if let Err(err) = run_validate(&config) {
        eprintln!("spirv-val: {err}");
        exit(1);
    }
}
