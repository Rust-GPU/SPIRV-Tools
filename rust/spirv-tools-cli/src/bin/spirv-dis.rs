use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use spirv_tools_cli::disassemble::{
    run_disassemble, DisassembleCliError, DisassembleConfig, InputSource,
};
use std::io::IsTerminal;

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

    /// Disable indentation in the output (equivalent to --no-indent in the C++ tool).
    #[arg(long = "no-indent")]
    no_indent: bool,

    /// Emit raw numeric ids instead of friendly names.
    #[arg(long = "raw-id")]
    raw_id: bool,

    /// Align blocks using structured-nesting indentation.
    #[arg(long = "nested-indent")]
    nested_indent: bool,

    /// Reorder blocks to follow structured control flow.
    #[arg(long = "reorder-blocks")]
    reorder_blocks: bool,

    /// Emit decoration comments in the output.
    #[arg(long = "comment")]
    comment: bool,

    /// Format literal numbers using hexadecimal notation.
    #[arg(long = "hex")]
    hex_literals: bool,

    /// Force colorized output even when stdout is redirected.
    #[arg(long = "color")]
    color: bool,

    /// Disable color output even if stdout is a terminal.
    #[arg(long = "no-color")]
    no_color: bool,
}

fn main() {
    let args = Args::parse();
    let input = if args.input == "-" {
        InputSource::Stdin
    } else {
        InputSource::Path(PathBuf::from(&args.input))
    };

    let stdout_is_tty = std::io::stdout().is_terminal();
    let colorize = if args.color {
        true
    } else if args.no_color {
        false
    } else {
        stdout_is_tty && args.output.is_none()
    };

    let config = DisassembleConfig {
        input,
        suppress_header: args.no_header,
        show_byte_offsets: args.offsets,
        indent: !args.no_indent,
        friendly_names: !args.raw_id,
        nested_indent: args.nested_indent,
        reorder_blocks: args.reorder_blocks,
        comments: args.comment,
        colorize,
        hex_literals: args.hex_literals,
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
