use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

use crate::disassemble::{run_disassemble, DisassembleCliError, DisassembleConfig, InputSource};
use rspirv::binary::ParseState;
use rspirv::dr::load_words;
use rspirv::spirv::Op;
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
    /// Emit only the entry point names.
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
    /// Failed to read the input stream.
    #[error("failed to read SPIR-V module: {0}")]
    Input(#[from] std::io::Error),
    /// Input length is not a multiple of four bytes.
    #[error("input size must be a multiple of 4 bytes")]
    MisalignedInput,
    /// Failed to parse the module.
    #[error("failed to decode SPIR-V module: {0}")]
    Decode(#[from] ParseState),
    /// The source payload was malformed.
    #[error("malformed debug source section: {0}")]
    InvalidSource(&'static str),
    /// A source file name collided with another extracted file.
    #[error("duplicate source file name {0}")]
    SourceNameConflict(String),
    /// Attempted to overwrite an existing file without --force.
    #[error("refusing to overwrite existing file {0} (use --force)")]
    OverwriteDenied(String),
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
        ObjdumpMode::Source(options) => run_source_dump(config, options),
        ObjdumpMode::EntrypointOnly => run_entrypoint_dump(config),
        ObjdumpMode::CompilerCommand => Err(ObjdumpCliError::Unimplemented("compiler command")),
    }
}

fn run_source_dump(
    config: &ObjdumpConfig,
    options: &ObjdumpSourceOptions,
) -> Result<String, ObjdumpCliError> {
    let words = read_words(&config.input)?;
    let sources = extract_sources(&words)?;
    if options.list_only {
        let mut names: Vec<_> = sources.iter().map(|s| s.name.clone()).collect();
        names.sort();
        let listing = names.join("\n");
        return Ok(if listing.is_empty() {
            String::new()
        } else {
            format!("{listing}\n")
        });
    }

    if let Some(dir) = &options.output_dir {
        fs::create_dir_all(dir)?;
        let mut exported = String::new();
        for src in &sources {
            let out_path = dir.join(&src.name);
            if out_path.exists() && !options.overwrite {
                return Err(ObjdumpCliError::OverwriteDenied(
                    out_path.to_string_lossy().into_owned(),
                ));
            }
            if src.contents.is_empty() {
                exported.push_str(&format!(
                    "Ignoring source for {}: no code source in debug infos.\n",
                    src.name
                ));
                continue;
            }
            fs::write(&out_path, &src.contents)?;
            exported.push_str(&format!("Exporting {}\n", out_path.display()));
        }
        return Ok(exported);
    }

    // Default: emit to stdout with filename markers.
    let mut out = String::new();
    for src in &sources {
        if src.contents.is_empty() {
            out.push_str(&format!(
                "Ignoring source for {}: no code source in debug infos.\n",
                src.name
            ));
            continue;
        }
        out.push_str(&src.name);
        out.push_str(":\n");
        out.push_str(&src.contents);
        out.push_str("\n\n");
    }
    Ok(out)
}

fn run_entrypoint_dump(config: &ObjdumpConfig) -> Result<String, ObjdumpCliError> {
    let words = read_words(&config.input)?;
    let module = load_words(&words)?;
    let mut names = Vec::new();
    for entry in &module.entry_points {
        // OpEntryPoint: ExecutionModel | EntryPoint <id> | Name | ...interfaces
        if let Some(name_operand) = entry.operands.get(2) {
            if let rspirv::dr::Operand::LiteralString(name) = name_operand {
                names.push(name.clone());
            }
        }
    }
    names.sort();
    let listing = names.join("\n");
    Ok(if listing.is_empty() {
        String::new()
    } else {
        format!("{listing}\n")
    })
}

fn read_words(input: &InputSource) -> Result<Vec<u32>, ObjdumpCliError> {
    let bytes = match input {
        InputSource::Stdin => {
            let mut buffer = Vec::new();
            std::io::stdin().read_to_end(&mut buffer)?;
            buffer
        }
        InputSource::Path(path) => fs::read(path)?,
    };
    if !bytes.len().is_multiple_of(4) {
        return Err(ObjdumpCliError::MisalignedInput);
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(words)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceFile {
    name: String,
    contents: String,
}

fn extract_sources(words: &[u32]) -> Result<Vec<SourceFile>, ObjdumpCliError> {
    let module = load_words(words)?;
    let mut strings = HashMap::new();
    let mut sources: Vec<(Option<u32>, String)> = Vec::new();
    for inst in &module.debug_string_source {
        match inst.class.opcode {
            Op::String => {
                if let Some(rspirv::dr::Operand::LiteralString(value)) = inst.operands.get(0) {
                    if let Some(id) = inst.result_id {
                        strings.entry(id).or_insert_with(|| value.clone());
                    }
                } else {
                    return Err(ObjdumpCliError::InvalidSource("OpString missing literal"));
                }
            }
            Op::Source => {
                // Operands: SourceLanguage, Version, optional File, optional Source
                let mut file_id = None;
                let mut source = String::new();
                if let Some(file_operand) = inst.operands.get(2) {
                    if let rspirv::dr::Operand::IdRef(id) = file_operand {
                        file_id = Some(*id);
                    }
                }
                if let Some(rspirv::dr::Operand::LiteralString(content)) = inst.operands.get(3) {
                    source.push_str(content);
                }
                sources.push((file_id, source));
            }
            Op::SourceContinued => {
                if let Some((_, code)) = sources.last_mut() {
                    if let Some(rspirv::dr::Operand::LiteralString(fragment)) = inst.operands.get(0)
                    {
                        code.push_str(fragment);
                    } else {
                        return Err(ObjdumpCliError::InvalidSource(
                            "OpSourceContinued missing literal",
                        ));
                    }
                } else {
                    return Err(ObjdumpCliError::InvalidSource(
                        "OpSourceContinued without preceding OpSource",
                    ));
                }
            }
            _ => {}
        }
    }

    let mut extracted = Vec::new();
    let mut unnamed_counter = 0usize;
    for (maybe_file, contents) in sources {
        let name = maybe_file
            .and_then(|id| strings.get(&id).cloned())
            .unwrap_or_else(|| {
                let name = format!("unnamed-{unnamed_counter}.hlsl");
                unnamed_counter += 1;
                name
            });
        if extracted.iter().any(|s: &SourceFile| s.name == name) {
            return Err(ObjdumpCliError::SourceNameConflict(name));
        }
        extracted.push(SourceFile { name, contents });
    }
    Ok(extracted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::words_to_bytes;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv;
    use spirv_tools_core::assembly::assemble_text;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

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
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&words_to_bytes(&words))
            .expect("write module");

        let listing = run_objdump(&ObjdumpConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            mode: ObjdumpMode::EntrypointOnly,
        })
        .expect("list entrypoints");

        assert_eq!(listing, "main\n");
    }

    #[test]
    fn extracts_sources_and_lists_names() {
        let words = build_module_with_source("main.hlsl", "void main() {}", Some(" // tail"));
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&words_to_bytes(&words))
            .expect("write module");

        let output = run_objdump(&ObjdumpConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            mode: ObjdumpMode::Source(ObjdumpSourceOptions {
                list_only: true,
                output_dir: None,
                overwrite: false,
            }),
        })
        .expect("extract sources");

        assert_eq!(output, "main.hlsl\n");
    }

    #[test]
    fn extracts_sources_and_writes_files() {
        let words = build_module_with_source("main.hlsl", "void main() {}", None);
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&words_to_bytes(&words))
            .expect("write module");

        let outdir = tempdir().expect("outdir");
        let config = ObjdumpConfig {
            input: InputSource::Path(file.path().to_path_buf()),
            mode: ObjdumpMode::Source(ObjdumpSourceOptions {
                list_only: false,
                output_dir: Some(outdir.path().to_path_buf()),
                overwrite: false,
            }),
        };
        run_objdump(&config).expect("write sources");

        let written = outdir.path().join("main.hlsl");
        let contents = fs::read_to_string(&written).expect("read written file");
        assert!(
            contents.contains("void main() {}"),
            "unexpected contents: {contents}"
        );

        // Attempting again without --force should error.
        let err = run_objdump(&config).expect_err("expected overwrite denial");
        assert!(matches!(err, ObjdumpCliError::OverwriteDenied(_)));
    }

    fn build_module_with_source(
        filename: &str,
        contents: &str,
        continued: Option<&str>,
    ) -> Vec<u32> {
        let mut builder = Builder::new();
        builder.capability(spirv::Capability::Shader);
        builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

        let file_id = builder.string(filename);
        builder.source(
            spirv::SourceLanguage::GLSL,
            450,
            Some(file_id),
            Some(contents),
        );
        if let Some(tail) = continued {
            builder.source_continued(tail);
        }

        let void = builder.type_void();
        let fn_ty = builder.type_function(void, vec![void]);
        let func_id = builder
            .begin_function(void, None, spirv::FunctionControl::NONE, fn_ty)
            .expect("begin function");
        builder.begin_block(None).expect("begin block");
        builder.ret().expect("return");
        builder.end_function().expect("end function");
        builder.entry_point(spirv::ExecutionModel::Fragment, func_id, "main", &[]);

        builder.module().assemble()
    }
}
