//! SPIR-V assembler support.

use crate::binary::Binary;
use crate::error::{Diagnostic, Error, SpirvResult, TargetEnv};

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

/// Convert spirv-tools TargetEnv to spirv-tools-core TargetEnv
fn to_core_target_env(env: TargetEnv) -> spirv_tools_core::TargetEnv {
    use spirv_tools_core::TargetEnv as CoreEnv;
    match env {
        TargetEnv::Universal_1_0 => CoreEnv::Universal1_0,
        TargetEnv::Universal_1_1 => CoreEnv::Universal1_1,
        TargetEnv::Universal_1_2 => CoreEnv::Universal1_2,
        TargetEnv::Universal_1_3 => CoreEnv::Universal1_3,
        TargetEnv::Universal_1_4 => CoreEnv::Universal1_4,
        TargetEnv::Universal_1_5 => CoreEnv::Universal1_5,
        TargetEnv::Universal_1_6 => CoreEnv::Universal1_6,
        TargetEnv::Vulkan_1_0 => CoreEnv::Vulkan1_0,
        TargetEnv::Vulkan_1_1 => CoreEnv::Vulkan1_1,
        TargetEnv::Vulkan_1_1_Spirv_1_4 => CoreEnv::Vulkan1_1Spirv1_4,
        TargetEnv::Vulkan_1_2 => CoreEnv::Vulkan1_2,
        TargetEnv::Vulkan_1_3 => CoreEnv::Vulkan1_3,
        TargetEnv::Vulkan_1_4 => CoreEnv::Vulkan1_4,
        TargetEnv::OpenGL_4_0 => CoreEnv::OpenGl4_0,
        TargetEnv::OpenGL_4_1 => CoreEnv::OpenGl4_1,
        TargetEnv::OpenGL_4_2 => CoreEnv::OpenGl4_2,
        TargetEnv::OpenGL_4_3 => CoreEnv::OpenGl4_3,
        TargetEnv::OpenGL_4_5 => CoreEnv::OpenGl4_5,
        TargetEnv::OpenCL_1_2 => CoreEnv::OpenCl1_2,
        TargetEnv::OpenCL_2_0 => CoreEnv::OpenCl2_0,
        TargetEnv::OpenCL_2_1 => CoreEnv::OpenCl2_1,
        TargetEnv::OpenCL_2_2 => CoreEnv::OpenCl2_2,
        TargetEnv::OpenCLEmbedded_1_2 => CoreEnv::OpenClEmbedded1_2,
        TargetEnv::OpenCLEmbedded_2_0 => CoreEnv::OpenClEmbedded2_0,
        TargetEnv::OpenCLEmbedded_2_1 => CoreEnv::OpenClEmbedded2_1,
        TargetEnv::OpenCLEmbedded_2_2 => CoreEnv::OpenClEmbedded2_2,
        TargetEnv::WebGPU_0_DEPRECATED => CoreEnv::WebGpu0,
    }
}

/// Convert assembler options to spirv-tools-core TextToBinaryOptions
fn to_core_assembly_options(options: AssemblerOptions) -> spirv_tools_core::TextToBinaryOptions {
    let mut core_options = spirv_tools_core::TextToBinaryOptions::NONE;
    if options.preserve_numeric_ids {
        core_options |= spirv_tools_core::TextToBinaryOptions::PRESERVE_NUMERIC_IDS;
    }
    core_options
}

/// Convert disassemble options to spirv-tools-core BinaryToTextOptions
fn to_core_disassemble_options(options: DisassembleOptions) -> spirv_tools_core::BinaryToTextOptions {
    let mut core_options = spirv_tools_core::BinaryToTextOptions::empty();
    if options.print {
        core_options |= spirv_tools_core::BinaryToTextOptions::PRINT;
    }
    if options.color {
        core_options |= spirv_tools_core::BinaryToTextOptions::COLOR;
    }
    if options.indent {
        core_options |= spirv_tools_core::BinaryToTextOptions::INDENT;
    }
    if options.show_byte_offset {
        core_options |= spirv_tools_core::BinaryToTextOptions::SHOW_BYTE_OFFSET;
    }
    if options.no_header {
        core_options |= spirv_tools_core::BinaryToTextOptions::NO_HEADER;
    }
    if options.use_friendly_names {
        core_options |= spirv_tools_core::BinaryToTextOptions::FRIENDLY_NAMES;
    }
    if options.comment {
        core_options |= spirv_tools_core::BinaryToTextOptions::COMMENT;
    }
    core_options
}

impl Assembler for RustAssembler {
    fn with_env(target_env: TargetEnv) -> Self {
        Self { target_env }
    }

    fn assemble(&self, text: &str, options: AssemblerOptions) -> Result<Binary, Error> {
        let core_env = to_core_target_env(self.target_env);
        let core_options = to_core_assembly_options(options);

        match spirv_tools_core::assemble_text_with_options(text, core_env, core_options) {
            Ok(words) => Ok(Binary::OwnedU32(words)),
            Err(e) => {
                let diagnostics = e.diagnostics();
                let message = if let Some(first) = diagnostics.first() {
                    first.message().to_string()
                } else {
                    "assembly failed".to_string()
                };
                let (line, column) = diagnostics
                    .first()
                    .map(|d| (d.position().line() as usize, d.position().column() as usize))
                    .unwrap_or((0, 0));

                Err(Error {
                    inner: SpirvResult::InvalidText,
                    diagnostic: Some(Diagnostic {
                        line,
                        column,
                        index: 0,
                        message,
                        notes: String::new(),
                        is_text: true,
                    }),
                })
            }
        }
    }

    fn disassemble(
        &self,
        binary: impl AsRef<[u32]>,
        options: DisassembleOptions,
    ) -> Result<Option<String>, Error> {
        let words = binary.as_ref();
        let core_options = to_core_disassemble_options(options);

        match spirv_tools_core::disassemble_binary(words, core_options) {
            Ok(text) => {
                if text.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(text))
                }
            }
            Err(e) => {
                let diagnostics = e.diagnostics();
                let message = if let Some(first) = diagnostics.first() {
                    first.message().to_string()
                } else {
                    format!("{}", e)
                };

                Err(Error {
                    inner: SpirvResult::InvalidBinary,
                    diagnostic: Some(Diagnostic::from(message)),
                })
            }
        }
    }
}
