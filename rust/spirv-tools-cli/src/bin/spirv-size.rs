use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use spirv_tools_cli::disassemble::InputSource;
use spirv_tools_cli::size::{run_size, SizeCliError, SizeConfig};

/// Report size statistics for a SPIR-V binary module.
#[derive(Debug, Parser)]
#[command(name = "spirv-size", about = "Report SPIR-V binary size statistics")]
struct Args {
    /// Input SPIR-V binary. Use '-' or omit to read from stdin.
    #[arg(value_name = "INPUT", default_value = "-")]
    input: String,
}

fn main() {
    let args = Args::parse();
    let input = if args.input == "-" {
        InputSource::Stdin
    } else {
        InputSource::Path(PathBuf::from(&args.input))
    };

    let config = SizeConfig { input };
    match run_size(&config) {
        Ok(stats) => print!("{stats}"),
        Err(err) => report_error(err),
    }
}

fn report_error(err: SizeCliError) -> ! {
    eprintln!("spirv-size: {err}");
    exit(1);
}
