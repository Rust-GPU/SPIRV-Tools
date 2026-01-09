//! SPIR-V assembler support.

use crate::binary::Binary;
use crate::error::{Diagnostic, Error, SpirvResult, TargetEnv};
use rspirv::binary::Disassemble;

/// Options for SPIR-V assembly.
#[derive(Copy, Clone, Default)]
pub struct AssemblerOptions {
    /// Numeric IDs in the binary will have the same values as in the source.
    /// Non-numeric IDs are allocated by filling in the gaps, starting with 1
    /// and going up.
    pub preserve_numeric_ids: bool,
}

/// Options for SPIR-V disassembly.
#[derive(Copy, Clone)]
pub struct DisassembleOptions {
    /// Print to stdout.
    pub print: bool,
    /// Add color codes to output.
    pub color: bool,
    /// Indent assembly.
    pub indent: bool,
    /// Show byte offset.
    pub show_byte_offset: bool,
    /// Do not output the module header as leading comments in the assembly.
    pub no_header: bool,
    /// Use friendly names where possible.
    pub use_friendly_names: bool,
    /// Add some comments to the generated assembly.
    pub comment: bool,
}

impl Default for DisassembleOptions {
    fn default() -> Self {
        Self {
            print: false,
            color: false,
            indent: true,
            show_byte_offset: false,
            no_header: false,
            use_friendly_names: true,
            comment: true,
        }
    }
}

/// Trait for SPIR-V assemblers.
pub trait Assembler: Default {
    fn with_env(target_env: TargetEnv) -> Self;
    fn assemble(&self, text: &str, options: AssemblerOptions) -> Result<Binary, Error>;
    fn disassemble(
        &self,
        binary: impl AsRef<[u32]>,
        options: DisassembleOptions,
    ) -> Result<Option<String>, Error>;
}

/// Create an assembler for the given target environment.
pub fn create(te: Option<TargetEnv>) -> impl Assembler {
    let target_env = te.unwrap_or_default();
    RustAssembler::with_env(target_env)
}

/// A pure Rust implementation of the SPIR-V assembler.
#[allow(dead_code)]
pub struct RustAssembler {
    target_env: TargetEnv,
}

impl Default for RustAssembler {
    fn default() -> Self {
        Self {
            target_env: TargetEnv::default(),
        }
    }
}

impl Assembler for RustAssembler {
    fn with_env(target_env: TargetEnv) -> Self {
        Self { target_env }
    }

    fn assemble(&self, _text: &str, _options: AssemblerOptions) -> Result<Binary, Error> {
        // Note: rspirv doesn't have a text assembler, so this is not implemented.
        // The spirv-tools crate on crates.io uses the C++ tools for this.
        // For rust-gpu integration, assembly is done through rspirv's module builder.
        Err(Error {
            inner: SpirvResult::Unsupported,
            diagnostic: Some(Diagnostic::from(
                "Text assembly is not supported in the pure Rust implementation. \
                 Use rspirv's module builder API instead."
                    .to_string(),
            )),
        })
    }

    fn disassemble(
        &self,
        binary: impl AsRef<[u32]>,
        options: DisassembleOptions,
    ) -> Result<Option<String>, Error> {
        let words = binary.as_ref();

        // Use rspirv's disassembler
        match rspirv::dr::load_words(words) {
            Ok(module) => {
                let disassembly = module.disassemble();
                if options.print {
                    println!("{}", disassembly);
                }
                Ok(Some(disassembly))
            }
            Err(e) => Err(Error {
                inner: SpirvResult::InvalidBinary,
                diagnostic: Some(Diagnostic::from(format!(
                    "Failed to disassemble SPIR-V: {}",
                    e
                ))),
            }),
        }
    }
}
