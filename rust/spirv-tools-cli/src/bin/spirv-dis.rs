use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use spirv_tools_cli::disassemble::{
    run_disassemble, DisassembleCliError, DisassembleConfig, InputSource,
};

/// Disassembles a SPIR-V binary module into text.
#[derive(Debug, Parser)]
#[command(name = "spirv-dis", about = "Disassemble a SPIR-V binary module")]
struct Args {
    /// Input SPIR-V binary. Use '-' or omit to read from stdin.
    #[arg(value_name = "INPUT", default_value = "-")]
    input: String,

    /// Output file. Defaults to stdout when omitted.
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    output: Option<PathBuf>,

    /// Suppress the textual header in the output (equivalent to --no-header in the C++ tool).
    #[arg(long = "no-header")]
    no_header: bool,

    /// Emit byte offsets alongside each instruction.
    #[arg(long = "offsets")]
    offsets: bool,
}

fn main() {
    let args = Args::parse();
    let input = if args.input == "-" {
        InputSource::Stdin
    } else {
        InputSource::Path(PathBuf::from(&args.input))
    };

    let config = DisassembleConfig {
        input,
        suppress_header: args.no_header,
        show_byte_offsets: args.offsets,
    };

    match run_disassemble(&config) {
        Ok(text) => {
            if let Some(path) = args.output {
                if let Err(err) = std::fs::write(&path, text) {
                    report_error(DisassembleCliError::Input(err));
                }
            } else {
                print!("{}", text);
            }
        }
        Err(err) => report_error(err),
    }
}

fn report_error(err: DisassembleCliError) -> ! {
    eprintln!("spirv-dis: {err}");
    exit(1);
}
