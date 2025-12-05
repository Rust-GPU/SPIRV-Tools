use std::path::PathBuf;

use crate::disassemble::{run_disassemble, DisassembleCliError, DisassembleConfig, InputSource};
use thiserror::Error;

/// Configuration for the objdump tool.
#[derive(Clone, Debug)]
pub struct ObjdumpConfig {
    /// Input source to consume.
    pub input: InputSource,
    /// Which operation to perform.
    pub mode: ObjdumpMode,
}

impl Default for ObjdumpConfig {
    fn default() -> Self {
        Self {
            input: InputSource::default(),
            mode: ObjdumpMode::Disassemble(DisassembleOptions::default()),
        }
    }
}

/// The supported objdump tasks.
#[derive(Clone, Debug)]
pub enum ObjdumpMode {
    /// Disassemble the binary, matching the default C++ behavior.
    Disassemble(DisassembleOptions),
    /// Extract embedded source code from debug information.
    Source(ObjdumpSourceOptions),
    /// Emit only the entry point name.
    EntrypointOnly,
    /// Emit the original compiler command recorded in the binary.
    CompilerCommand,
}

/// Options for disassembling via the objdump entry point.
#[derive(Clone, Debug)]
pub struct DisassembleOptions {
    /// Omit the leading header comment.
    pub suppress_header: bool,
    /// Show byte offsets for each instruction.
    pub show_byte_offsets: bool,
    /// Use indentation when rendering operands.
    pub indent: bool,
    /// Emit friendly names instead of raw ids when available.
    pub friendly_names: bool,
    /// Emit structured-nesting indentation.
    pub nested_indent: bool,
    /// Reorder blocks to follow structured control flow.
    pub reorder_blocks: bool,
    /// Include decoration comments alongside instructions.
    pub comments: bool,
    /// Emit ANSI color escapes in the disassembly.
    pub colorize: bool,
    /// Format literal numbers in hexadecimal.
    pub hex_literals: bool,
}

impl Default for DisassembleOptions {
    fn default() -> Self {
        let defaults = DisassembleConfig::default();
        Self {
            suppress_header: defaults.suppress_header,
            show_byte_offsets: defaults.show_byte_offsets,
            indent: defaults.indent,
            friendly_names: defaults.friendly_names,
            nested_indent: defaults.nested_indent,
            reorder_blocks: defaults.reorder_blocks,
            comments: defaults.comments,
            colorize: defaults.colorize,
            hex_literals: defaults.hex_literals,
        }
    }
}

impl DisassembleOptions {
    fn to_disassemble_config(&self, input: InputSource) -> DisassembleConfig {
        DisassembleConfig {
            input,
            suppress_header: self.suppress_header,
            show_byte_offsets: self.show_byte_offsets,
            indent: self.indent,
            friendly_names: self.friendly_names,
            nested_indent: self.nested_indent,
            reorder_blocks: self.reorder_blocks,
            comments: self.comments,
            colorize: self.colorize,
            hex_literals: self.hex_literals,
        }
    }
}

/// Source extraction output settings.
#[derive(Clone, Debug)]
pub struct ObjdumpSourceOptions {
    /// When true, only list the discovered source file names.
    pub list_only: bool,
    /// Where to write extracted sources; `None` sends them to stdout.
    pub output_dir: Option<PathBuf>,
    /// Overwrite any existing files in the output directory.
    pub overwrite: bool,
}

impl Default for ObjdumpSourceOptions {
    fn default() -> Self {
        Self {
            list_only: false,
            output_dir: None,
            overwrite: false,
        }
    }
}

/// Errors surfaced by the objdump CLI entry points.
#[derive(Debug, Error)]
pub enum ObjdumpCliError {
    /// Disassembly failed.
    #[error(transparent)]
    Disassemble(#[from] DisassembleCliError),
    /// Source/entrypoint/command extraction is not implemented yet.
    #[error("unimplemented objdump mode: {0}")]
    Unimplemented(&'static str),
}

/// Run the objdump tool with the provided configuration.
pub fn run_objdump(config: &ObjdumpConfig) -> Result<String, ObjdumpCliError> {
    match &config.mode {
        ObjdumpMode::Disassemble(options) => {
            let disassemble_config = options.to_disassemble_config(config.input.clone());
            run_disassemble(&disassemble_config).map_err(Into::into)
        }
        ObjdumpMode::Source(_) => Err(ObjdumpCliError::Unimplemented("source extraction")),
        ObjdumpMode::EntrypointOnly => Err(ObjdumpCliError::Unimplemented("entrypoint listing")),
        ObjdumpMode::CompilerCommand => Err(ObjdumpCliError::Unimplemented("compiler command")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::words_to_bytes;
    use spirv_tools_core::assembly::assemble_text;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn disassembles_simple_module() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Fragment %main \"main\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let words = assemble_text(&text).expect("assemble module");
        let mut temp = NamedTempFile::new().expect("temp file");
        temp.write_all(&words_to_bytes(&words))
            .expect("write module");

        let config = ObjdumpConfig {
            input: InputSource::Path(temp.path().to_path_buf()),
            mode: ObjdumpMode::Disassemble(DisassembleOptions::default()),
        };
        let output = run_objdump(&config).expect("disassemble");
        assert!(
            output.contains("OpEntryPoint") && output.contains("OpFunctionEnd"),
            "expected disassembly output, got: {output}"
        );
    }

    #[test]
    fn entrypoint_mode_is_unimplemented() {
        let config = ObjdumpConfig {
            input: InputSource::Stdin,
            mode: ObjdumpMode::EntrypointOnly,
        };
        let err = run_objdump(&config).expect_err("expected unimplemented error");
        assert!(matches!(err, ObjdumpCliError::Unimplemented(_)));
    }
}
