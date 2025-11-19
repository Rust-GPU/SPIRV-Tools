use std::io::{self, Write};
use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use spirv_tools_cli::assembly::{run_assemble, words_to_bytes, AssembleCliError, AssembleConfig};
use spirv_tools_cli::disassemble::InputSource;

/// Assemble a SPIR-V source module into binary form.
#[derive(Debug, Parser)]
#[command(name = "spirv-as", about = "Assemble SPIR-V text into binary")]
struct Args {
    /// Input SPIR-V assembly. Use '-' or omit to read from stdin.
    #[arg(value_name = "INPUT", default_value = "-")]
    input: String,

    /// Output path for the assembled binary. Defaults to stdout.
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    output: Option<PathBuf>,

    /// Preserve numeric IDs from the source, mirroring the C++ --preserve-numeric-ids flag.
    #[arg(long = "preserve-numeric-ids")]
    preserve_numeric_ids: bool,

    /// Target environment (e.g., "universal1.6", "vulkan1.2"). Defaults to universal1.6.
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

    let config = AssembleConfig {
        input,
        target_env: args.target_env,
        preserve_numeric_ids: args.preserve_numeric_ids,
    };

    match run_assemble(&config) {
        Ok(words) => {
            let bytes = words_to_bytes(&words);
            if let Some(path) = args.output {
                if let Err(err) = std::fs::write(&path, &bytes) {
                    report_error(AssembleCliError::Input(err));
                }
            } else {
                let mut stdout = io::stdout();
                if let Err(err) = stdout.write_all(&bytes) {
                    report_error(AssembleCliError::Input(err));
                }
            }
        }
        Err(err) => report_error(err),
    }
}

fn report_error(err: AssembleCliError) -> ! {
    eprintln!("spirv-as: {err}");
    exit(1);
}
