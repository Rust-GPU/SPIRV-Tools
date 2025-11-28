use clap::{ArgAction, Parser};
use std::path::PathBuf;

use spirv_tools_cli::disassemble::InputSource;
use spirv_tools_cli::optimizer::{run_optimize, write_output, OptimizeCliError, OptimizeConfig};

/// SPIR-V optimizer (Rust arithmetic pass).
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input SPIR-V binary; reads stdin when omitted.
    input: Option<PathBuf>,
    /// Output path for optimized SPIR-V; writes to stdout when omitted.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Passthrough mode: skip Rust optimizer and emit the input unchanged.
    #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
    passthrough: bool,

    /// Use the C++ spirv-opt binary instead of the Rust optimizer.
    #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
    cpp: bool,
}

fn main() {
    let args = Args::parse();
    let input = match args.input {
        Some(path) => InputSource::Path(path),
        None => InputSource::Stdin,
    };
    let config = OptimizeConfig {
        input,
        output: args.output.clone(),
        rust_arith_pass: !(args.passthrough || args.cpp),
        cpp_opt_path: args.cpp.then(|| std::ffi::OsString::from("spirv-opt")),
    };
    if let Err(err) = run_and_write(&config) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run_and_write(config: &OptimizeConfig) -> Result<(), OptimizeCliError> {
    let optimized = run_optimize(config)?;
    write_output(&optimized, &config.output)
}
