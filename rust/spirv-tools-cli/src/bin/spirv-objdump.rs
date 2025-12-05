use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use spirv_tools_cli::disassemble::InputSource;
use spirv_tools_cli::objdump::{
    run_objdump, DisassembleOptions, ObjdumpCliError, ObjdumpConfig, ObjdumpMode,
    ObjdumpSourceOptions,
};
use std::io::IsTerminal;

/// Dump information from a SPIR-V binary (disassembly, source, metadata).
#[derive(Debug, Parser)]
#[command(
    name = "spirv-objdump",
    about = "Dump information from a SPIR-V binary"
)]
struct Args {
    /// Input SPIR-V binary. Use '-' to read from stdin.
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Extract embedded source files from debug info instead of disassembling.
    #[arg(long, conflicts_with_all = ["entrypoint", "compiler_cmd"])]
    source: bool,

    /// Only list embedded source file names (implies --source).
    #[arg(long)]
    list: bool,

    /// Write extracted sources to a directory instead of stdout (implies --source).
    #[arg(long, value_name = "DIR")]
    outdir: Option<PathBuf>,

    /// Overwrite existing files when extracting sources.
    #[arg(long)]
    force: bool,

    /// Extract the recorded entrypoint name instead of disassembling.
    #[arg(long, conflicts_with_all = ["source", "compiler_cmd"])]
    entrypoint: bool,

    /// Extract the recorded compiler command instead of disassembling.
    #[arg(long, conflicts_with_all = ["source", "entrypoint"])]
    compiler_cmd: bool,

    /// Suppress the textual header in the disassembly.
    #[arg(long = "no-header")]
    no_header: bool,

    /// Emit byte offsets alongside each instruction.
    #[arg(long = "offsets")]
    offsets: bool,

    /// Disable indentation in the output.
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
    let input = if args.input == PathBuf::from("-") {
        InputSource::Stdin
    } else {
        InputSource::Path(args.input.clone())
    };

    let mode = select_mode(&args);
    let config = ObjdumpConfig { input, mode };

    match run_objdump(&config) {
        Ok(text) => {
            if !text.is_empty() {
                print!("{text}");
            }
        }
        Err(err) => report_error(err),
    }
}

fn select_mode(args: &Args) -> ObjdumpMode {
    let colorize = if args.color {
        true
    } else if args.no_color {
        false
    } else {
        std::io::stdout().is_terminal()
    };

    if args.source || args.list || args.outdir.is_some() {
        return ObjdumpMode::Source(ObjdumpSourceOptions {
            list_only: args.list,
            output_dir: args.outdir.clone(),
            overwrite: args.force,
        });
    }
    if args.entrypoint {
        return ObjdumpMode::EntrypointOnly;
    }
    if args.compiler_cmd {
        return ObjdumpMode::CompilerCommand;
    }

    ObjdumpMode::Disassemble(DisassembleOptions {
        suppress_header: args.no_header,
        show_byte_offsets: args.offsets,
        indent: !args.no_indent,
        friendly_names: !args.raw_id,
        nested_indent: args.nested_indent,
        reorder_blocks: args.reorder_blocks,
        comments: args.comment,
        colorize,
        hex_literals: args.hex_literals,
    })
}

fn report_error(err: ObjdumpCliError) -> ! {
    eprintln!("spirv-objdump: {err}");
    exit(1);
}
