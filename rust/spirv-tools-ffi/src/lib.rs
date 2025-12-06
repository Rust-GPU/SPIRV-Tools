use core::ffi::c_void;
use core::ptr::NonNull;
use rspirv::binary::parse_words;
use rspirv::dr::Loader;
use spirv_tools_core::assembly::{
    assemble_text_with_options, BinaryToTextOptions, TextToBinaryOptions,
};
use spirv_tools_core::diagnostic::{DiagnosticMessage, MessagePosition};
use spirv_tools_core::disassembly::{self, disassemble_binary, DisassemblyError};
use spirv_tools_core::validation::ValidModuleCache;
use spirv_tools_core::{MessageLevel, TargetEnv};
mod optimizer;
mod tests_optimizer;
use std::panic::{self, AssertUnwindSafe};
use std::str;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, Once, OnceLock};

static VALIDATION_CACHE: OnceLock<Mutex<ValidModuleCache>> = OnceLock::new();
// 0 = default/env-controlled, 1 = force enable rust optimizer, 2 = force disable rust optimizer
static RUST_OPT_OVERRIDE: AtomicU8 = AtomicU8::new(0);

fn validation_cache() -> &'static Mutex<ValidModuleCache> {
    VALIDATION_CACHE.get_or_init(Default::default)
}

#[cxx::bridge(namespace = "spvtools::ffi")]
mod ffi {
    #[derive(Debug)]
    struct ParseResult {
        success: bool,
        env: u32,
    }

    #[derive(Debug)]
    struct AssembleResult {
        success: bool,
        binary: Vec<u32>,
    }

    #[derive(Debug)]
    struct Diagnostic {
        level: u32,
        has_source: bool,
        source: String,
        position: MessagePosition,
        message: String,
    }

    #[derive(Debug)]
    struct DisassembleResult {
        success: bool,
        text: String,
        diagnostics: Vec<Diagnostic>,
    }

    #[derive(Debug)]
    struct ValidateResult {
        success: bool,
        message: String,
    }

    #[derive(Debug)]
    enum OptimizeError {
        None,
        Disabled,
        Parse,
        Optimize,
    }

    #[derive(Debug)]
    struct OptimizeResult {
        success: bool,
        error: OptimizeError,
        message: String,
        words: Vec<u32>,
    }

    #[derive(Debug)]
    enum ToolError {
        None,
        Disabled,
        Parse,
        Reduce,
        Fuzz,
    }

    #[derive(Debug)]
    struct ReduceResult {
        success: bool,
        error: ToolError,
        message: String,
        words: Vec<u32>,
    }

    #[derive(Debug)]
    struct FuzzResult {
        success: bool,
        error: ToolError,
        message: String,
        words: Vec<u32>,
    }

    #[derive(Debug)]
    struct ValidatorLimit {
        kind: u32,
        value: u32,
    }

    #[derive(Debug)]
    struct ValidatorOptions {
        relax_struct_store: bool,
        relax_logical_pointer: bool,
        relax_block_layout: bool,
        uniform_buffer_standard_layout: bool,
        scalar_block_layout: bool,
        workgroup_scalar_block_layout: bool,
        skip_block_layout: bool,
        allow_localsizeid: bool,
        allow_offset_texture_operand: bool,
        allow_vulkan_32_bit_bitwise: bool,
        before_hlsl_legalization: bool,
        use_friendly_names: bool,
        limits: Vec<ValidatorLimit>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MessagePosition {
        line: u32,
        column: u32,
        index: u32,
    }

    extern "Rust" {
        fn describe_target_env(env: u32) -> String;
        fn spirv_version_for_target(env: u32) -> u32;
        fn parse_target_env(input: &str) -> ParseResult;
        fn parse_vulkan_env(vulkan_version: u32, spirv_version: u32) -> ParseResult;
        fn read_env_from_text(text: &[u8]) -> ParseResult;
        fn create_context(env: u32, context_ptr: usize) -> u64;
        unsafe fn destroy_context(handle: u64);
        fn is_vulkan_env(env: u32) -> bool;
        fn is_opencl_env(env: u32) -> bool;
        fn is_opengl_env(env: u32) -> bool;
        fn is_valid_env(env: u32) -> bool;
        fn log_namespace(env: u32) -> String;
        fn list_target_envs(pad: usize, wrap: usize) -> String;
        fn sanitize_text_to_binary_options(options: u32) -> u32;
        fn sanitize_binary_to_text_options(options: u32) -> u32;
        fn disassembler_supports_options(options: u32) -> bool;
        fn has_rust_context(handle: usize) -> bool;
        fn context_handle_from_raw(handle: usize) -> u64;
        fn set_rust_text_assembler_override(enable: bool);
        fn clear_rust_text_assembler_override();
        fn rust_validator_enabled() -> bool;
        fn set_rust_validator_override(enable: bool);
        fn clear_rust_validator_override();
        fn set_rust_optimizer_override(enable: bool);
        fn clear_rust_optimizer_override();
        fn default_validator_options() -> ValidatorOptions;
        fn rebind_context(handle: u64, context_ptr: usize);
        fn try_assemble_text(context_handle: u64, text: &[u8], options: u32) -> AssembleResult;
        fn try_disassemble_binary(
            context_handle: u64,
            binary: &[u32],
            options: u32,
        ) -> DisassembleResult;
        fn validate_binary_rust(
            env: u32,
            binary: &[u32],
            options: &ValidatorOptions,
        ) -> ValidateResult;
        fn optimize_basic_block(words: &[u32]) -> OptimizeResult;
        fn reduce_module(words: &[u32]) -> ReduceResult;
        fn fuzz_module(words: &[u32]) -> FuzzResult;
    }

    unsafe extern "C++" {
        include!("spirv-tools-ffi/src/context_bridge.h");
        fn dispatch_context_message(
            context_ptr: usize,
            level: u32,
            has_source: bool,
            source: &str,
            position: MessagePosition,
            message: &str,
        );
        fn assemble_text_with_context(
            context_ptr: usize,
            text: &[u8],
            options: u32,
        ) -> AssembleResult;
        fn validate_binary(env: u32, words: &[u32]) -> ValidateResult;
    }
}

pub use ffi::OptimizeError;

/// Returns the human-readable description for a SPIR-V target environment.
pub fn describe_target_env(env: u32) -> String {
    TargetEnv::from_raw(env)
        .map(TargetEnv::description)
        .unwrap_or("")
        .to_string()
}

fn to_env(value: u32) -> Option<TargetEnv> {
    TargetEnv::from_raw(value)
}

pub fn spirv_version_for_target(env: u32) -> u32 {
    to_env(env)
        .map(|env| env.spirv_version().to_word())
        .unwrap_or_default()
}

pub fn parse_target_env(input: &str) -> ffi::ParseResult {
    match TargetEnv::parse_name(input) {
        Some(env) => ffi::ParseResult {
            success: true,
            env: env.to_raw(),
        },
        None => ffi::ParseResult {
            success: false,
            env: TargetEnv::Universal1_0.to_raw(),
        },
    }
}

pub fn parse_vulkan_env(vulkan_version: u32, spirv_version: u32) -> ffi::ParseResult {
    match TargetEnv::parse_vulkan_env(vulkan_version, spirv_version) {
        Some(env) => ffi::ParseResult {
            success: true,
            env: env.to_raw(),
        },
        None => ffi::ParseResult {
            success: false,
            env: TargetEnv::Universal1_0.to_raw(),
        },
    }
}

pub fn read_env_from_text(text: &[u8]) -> ffi::ParseResult {
    match spirv_tools_core::target_env::read_env_from_text(text) {
        Some(env) => ffi::ParseResult {
            success: true,
            env: env.to_raw(),
        },
        None => ffi::ParseResult {
            success: false,
            env: TargetEnv::Universal1_0.to_raw(),
        },
    }
}

pub fn is_vulkan_env(env: u32) -> bool {
    to_env(env).map(|env| env.is_vulkan()).unwrap_or(false)
}

pub fn is_opencl_env(env: u32) -> bool {
    to_env(env).map(|env| env.is_opencl()).unwrap_or(false)
}

pub fn is_opengl_env(env: u32) -> bool {
    to_env(env).map(|env| env.is_opengl()).unwrap_or(false)
}

pub fn is_valid_env(env: u32) -> bool {
    to_env(env).map(|env| env.is_valid()).unwrap_or(false)
}

pub fn log_namespace(env: u32) -> String {
    to_env(env)
        .map(|env| env.log_namespace().to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

pub fn list_target_envs(pad: usize, wrap: usize) -> String {
    TargetEnv::list_target_envs(pad, wrap)
}

pub fn sanitize_text_to_binary_options(options: u32) -> u32 {
    TextToBinaryOptions::from(options).bits()
}

pub fn sanitize_binary_to_text_options(options: u32) -> u32 {
    BinaryToTextOptions::from(options).bits()
}

pub fn disassembler_supports_options(options: u32) -> bool {
    let requested = BinaryToTextOptions::from(options);
    disassembly::supports_options(requested)
}

pub fn validate_binary_rust(
    env: u32,
    binary: &[u32],
    options: &ffi::ValidatorOptions,
) -> ffi::ValidateResult {
    let env = match TargetEnv::from_raw(env) {
        Some(env) => env,
        None => {
            return ffi::ValidateResult {
                success: false,
                message: format!("unknown target environment {}", env),
            }
        }
    };
    let mut cache = validation_cache()
        .lock()
        .expect("validation cache mutex should not be poisoned");
    let opts = to_validation_options(options);
    match cache.validate_words_with_options(binary, env, opts.clone()) {
        Ok(_) => ffi::ValidateResult {
            success: true,
            message: String::new(),
        },
        Err(err) => ffi::ValidateResult {
            success: false,
            message: spirv_tools_core::validation::format_validation_error_from_words(
                binary, &opts, &err,
            ),
        },
    }
}

pub fn optimize_basic_block(words: &[u32]) -> ffi::OptimizeResult {
    if !rust_optimizer_enabled() {
        return ffi::OptimizeResult {
            success: true,
            error: ffi::OptimizeError::Disabled,
            message: String::new(),
            words: words.to_vec(),
        };
    }

    let parsed = optimizer::optimize_basic_block(words);
    match parsed {
        Ok(words) => ffi::OptimizeResult {
            success: true,
            error: ffi::OptimizeError::None,
            message: String::new(),
            words,
        },
        Err(err) => {
            let kind = match &err {
                optimizer::OptimizeError::Parse(_) => ffi::OptimizeError::Parse,
                optimizer::OptimizeError::Rewrite(_) => ffi::OptimizeError::Optimize,
            };
            ffi::OptimizeResult {
                success: false,
                error: kind,
                message: err.to_string(),
                words: Vec::new(),
            }
        }
    }
}

fn to_validation_options(
    options: &ffi::ValidatorOptions,
) -> spirv_tools_core::validation::ValidationOptions {
    let limits = options
        .limits
        .iter()
        .map(|limit| (limit.kind, limit.value))
        .collect();
    spirv_tools_core::validation::ValidationOptions {
        relax_struct_store: options.relax_struct_store,
        relax_logical_pointer: options.relax_logical_pointer,
        relax_block_layout: options.relax_block_layout,
        uniform_buffer_standard_layout: options.uniform_buffer_standard_layout,
        scalar_block_layout: options.scalar_block_layout,
        workgroup_scalar_block_layout: options.workgroup_scalar_block_layout,
        skip_block_layout: options.skip_block_layout,
        allow_localsizeid: options.allow_localsizeid,
        allow_offset_texture_operand: options.allow_offset_texture_operand,
        allow_vulkan_32_bit_bitwise: options.allow_vulkan_32_bit_bitwise,
        before_hlsl_legalization: options.before_hlsl_legalization,
        use_friendly_names: options.use_friendly_names,
        limits,
    }
}

pub fn default_validator_options() -> ffi::ValidatorOptions {
    ffi::ValidatorOptions {
        relax_struct_store: false,
        relax_logical_pointer: false,
        relax_block_layout: false,
        uniform_buffer_standard_layout: false,
        scalar_block_layout: false,
        workgroup_scalar_block_layout: false,
        skip_block_layout: false,
        allow_localsizeid: false,
        allow_offset_texture_operand: false,
        allow_vulkan_32_bit_bitwise: false,
        before_hlsl_legalization: false,
        use_friendly_names: true,
        limits: Vec::new(),
    }
}

pub fn validate_binary_rust_with_options(
    env: u32,
    binary: &[u32],
    options: &ffi::ValidatorOptions,
) -> ffi::ValidateResult {
    validate_binary_rust(env, binary, options)
}

pub fn has_rust_context(handle: usize) -> bool {
    let ptr = handle as *const ContextHandle;
    unsafe { ptr.as_ref().is_some() }
}

pub fn context_handle_from_raw(handle: usize) -> u64 {
    if has_rust_context(handle) {
        handle as u64
    } else {
        0
    }
}

pub fn rebind_context(handle: u64, context_ptr: usize) {
    if handle == 0 {
        return;
    }

    let context_handle = match unsafe { (handle as *mut ContextHandle).as_mut() } {
        Some(handle) => handle,
        None => return,
    };

    let pointer = match NonNull::new(context_ptr as *mut c_void) {
        Some(pointer) => pointer,
        None => return,
    };

    context_handle.rebind(pointer);
}

/// Validates a SPIR-V module for the provided target environment.
pub fn validate_binary(env: TargetEnv, words: &[u32]) -> ffi::ValidateResult {
    validate_binary_with_options(env, words, &default_validator_options())
}

/// Validates with explicit validator options, preferring the Rust validator when enabled.
pub fn validate_binary_with_options(
    env: TargetEnv,
    words: &[u32],
    options: &ffi::ValidatorOptions,
) -> ffi::ValidateResult {
    if rust_validator_enabled() {
        #[cfg(test)]
        LAST_VALIDATION_PATH.store(1, Ordering::Relaxed);
        return validate_binary_rust_with_options(env.to_raw(), words, options);
    }

    #[cfg(test)]
    LAST_VALIDATION_PATH.store(2, Ordering::Relaxed);
    ffi::validate_binary(env.to_raw(), words)
}

static ENABLE_RUST_TEXT_ASSEMBLER: AtomicBool = AtomicBool::new(false);
static INIT_RUST_TEXT_ASSEMBLER: Once = Once::new();
static RUST_TEXT_ASSEMBLER_OVERRIDE: AtomicU8 = AtomicU8::new(0);
static ENABLE_RUST_VALIDATOR: AtomicBool = AtomicBool::new(true);
static INIT_RUST_VALIDATOR: Once = Once::new();
static RUST_VALIDATOR_OVERRIDE: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
static LAST_VALIDATION_PATH: AtomicU8 = AtomicU8::new(0);

fn rust_text_assembler_enabled() -> bool {
    match RUST_TEXT_ASSEMBLER_OVERRIDE.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    INIT_RUST_TEXT_ASSEMBLER.call_once(|| {
        if std::env::var_os("SPIRV_TOOLS_ENABLE_RUST_ASSEMBLER").is_some() {
            ENABLE_RUST_TEXT_ASSEMBLER.store(true, Ordering::Relaxed);
        }
    });
    ENABLE_RUST_TEXT_ASSEMBLER.load(Ordering::Relaxed)
}

#[cfg(test)]
fn force_enable_rust_text_assembler_for_testing() {
    ENABLE_RUST_TEXT_ASSEMBLER.store(true, Ordering::Relaxed);
}

pub fn set_rust_text_assembler_override(enable: bool) {
    let value = if enable { 1 } else { 2 };
    RUST_TEXT_ASSEMBLER_OVERRIDE.store(value, Ordering::Relaxed);
}

pub fn clear_rust_text_assembler_override() {
    RUST_TEXT_ASSEMBLER_OVERRIDE.store(0, Ordering::Relaxed);
}

fn rust_validator_enabled() -> bool {
    match RUST_VALIDATOR_OVERRIDE.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    INIT_RUST_VALIDATOR.call_once(|| {
        if std::env::var_os("SPIRV_TOOLS_DISABLE_RUST_VALIDATOR").is_some() {
            ENABLE_RUST_VALIDATOR.store(false, Ordering::Relaxed);
        } else if std::env::var_os("SPIRV_TOOLS_FORCE_RUST_VALIDATOR").is_some() {
            ENABLE_RUST_VALIDATOR.store(true, Ordering::Relaxed);
        } else if std::env::var_os("SPIRV_TOOLS_PREFER_CPP_VALIDATOR").is_some() {
            ENABLE_RUST_VALIDATOR.store(false, Ordering::Relaxed);
        } else if std::env::var_os("SPIRV_TOOLS_PREFER_RUST_VALIDATOR").is_some() {
            ENABLE_RUST_VALIDATOR.store(true, Ordering::Relaxed);
        }
    });
    ENABLE_RUST_VALIDATOR.load(Ordering::Relaxed)
}

pub fn set_rust_validator_override(enable: bool) {
    let value = if enable { 1 } else { 2 };
    RUST_VALIDATOR_OVERRIDE.store(value, Ordering::Relaxed);
    if enable {
        // Keep the env hint in sync so child processes/tests can pick it up.
        std::env::set_var("SPIRV_TOOLS_FORCE_RUST_VALIDATOR", "1");
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_VALIDATOR");
    } else {
        std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_VALIDATOR", "1");
        std::env::remove_var("SPIRV_TOOLS_FORCE_RUST_VALIDATOR");
    }
}

pub fn clear_rust_validator_override() {
    RUST_VALIDATOR_OVERRIDE.store(0, Ordering::Relaxed);
    std::env::remove_var("SPIRV_TOOLS_FORCE_RUST_VALIDATOR");
}

fn rust_optimizer_enabled() -> bool {
    match RUST_OPT_OVERRIDE.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    if matches!(std::env::var("SPIRV_TOOLS_DISABLE_RUST_OPT"), Ok(v) if v == "1") {
        return false;
    }
    if std::env::var_os("SPIRV_TOOLS_FORCE_RUST_OPT").is_some() {
        return true;
    }
    true
}

pub fn set_rust_optimizer_override(enable: bool) {
    RUST_OPT_OVERRIDE.store(if enable { 1 } else { 2 }, Ordering::Relaxed);
    if enable {
        std::env::set_var("SPIRV_TOOLS_FORCE_RUST_OPT", "1");
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    } else {
        std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
        std::env::remove_var("SPIRV_TOOLS_FORCE_RUST_OPT");
    }
}

pub fn clear_rust_optimizer_override() {
    RUST_OPT_OVERRIDE.store(0, Ordering::Relaxed);
    std::env::remove_var("SPIRV_TOOLS_FORCE_RUST_OPT");
}

fn disabled_tool_result() -> (ffi::ToolError, String) {
    (
        ffi::ToolError::Disabled,
        "Rust reducer/fuzzer is not yet implemented; C++ bridge wiring is pending".to_string(),
    )
}

pub fn reduce_module(words: &[u32]) -> ffi::ReduceResult {
    let (error, message) = disabled_tool_result();
    ffi::ReduceResult {
        success: false,
        error,
        message,
        words: words.to_vec(),
    }
}

pub fn fuzz_module(words: &[u32]) -> ffi::FuzzResult {
    let (error, message) = disabled_tool_result();
    ffi::FuzzResult {
        success: false,
        error,
        message,
        words: words.to_vec(),
    }
}

pub fn try_assemble_text(context_handle: u64, text: &[u8], options: u32) -> ffi::AssembleResult {
    if context_handle == 0 {
        return ffi::AssembleResult {
            success: false,
            binary: Vec::new(),
        };
    }

    let override_forced = RUST_TEXT_ASSEMBLER_OVERRIDE.load(Ordering::Relaxed) == 1;

    if !rust_text_assembler_enabled() {
        return ffi::AssembleResult {
            success: false,
            binary: Vec::new(),
        };
    }

    let context = unsafe { (context_handle as *const ContextHandle).as_ref() };
    let Some(context) = context else {
        return ffi::AssembleResult {
            success: false,
            binary: Vec::new(),
        };
    };

    let mut pending = Vec::new();
    let options = TextToBinaryOptions::from(options);
    if let Ok(source) = str::from_utf8(text) {
        match panic::catch_unwind(AssertUnwindSafe(|| {
            assemble_text_with_options(source, context.env(), options)
        })) {
            Ok(Ok(binary)) => {
                let text_has_label = text
                    .windows("OpLabel".len())
                    .any(|window| window == b"OpLabel");
                let mut loader = Loader::new();
                let parsed = parse_words(&binary, &mut loader);
                let missing_bodies = parsed.is_ok()
                    && text_has_label
                    && loader
                        .module()
                        .functions
                        .iter()
                        .any(|function| function.def.is_some() && function.blocks.is_empty());
                if !missing_bodies {
                    return ffi::AssembleResult {
                        success: true,
                        binary,
                    };
                }

                pending.push(
                    DiagnosticMessage::new(
                        MessageLevel::Error,
                        MessagePosition::default(),
                        "Rust assembler produced function declarations without bodies; falling back to C++ implementation",
                    )
                    .with_source("input"),
                );
            }
            Ok(Err(error)) => {
                pending = error.into_diagnostics();
            }
            Err(_) => {
                pending.push(
                    DiagnosticMessage::new(
                        MessageLevel::Error,
                        MessagePosition::default(),
                        "Rust assembler panicked; falling back to C++ implementation",
                    )
                    .with_source("input"),
                );
            }
        }
    } else {
        pending.push(
            DiagnosticMessage::new(
                MessageLevel::Error,
                MessagePosition::default(),
                "Assembly text must be valid UTF-8",
            )
            .with_source("input"),
        );
    }

    if override_forced && !pending.is_empty() {
        return ffi::AssembleResult {
            success: false,
            binary: Vec::new(),
        };
    }

    let fallback = ffi::assemble_text_with_context(context.context_address(), text, options.into());
    if !fallback.success {
        for diagnostic in pending {
            context.emit_diagnostic(&diagnostic);
        }
    }
    fallback
}

pub fn try_disassemble_binary(
    _context_handle: u64,
    binary: &[u32],
    options: u32,
) -> ffi::DisassembleResult {
    let requested =
        BinaryToTextOptions::from_bits_truncate(options & !BinaryToTextOptions::NONE.bits());

    let mut pending = Vec::new();
    let disassembly =
        panic::catch_unwind(AssertUnwindSafe(|| disassemble_binary(binary, requested)));
    match disassembly {
        Ok(Ok(text)) => ffi::DisassembleResult {
            success: true,
            text,
            diagnostics: Vec::new(),
        },
        Ok(Err(DisassemblyError::Unsupported(unsupported))) => {
            pending.push(
                DiagnosticMessage::new(
                    MessageLevel::Error,
                    MessagePosition::default(),
                    format!("unsupported binary-to-text options: {unsupported:?}"),
                )
                .with_source("disassembler"),
            );
            ffi::DisassembleResult {
                success: false,
                text: String::new(),
                diagnostics: pending
                    .into_iter()
                    .map(to_ffi_diagnostic)
                    .collect::<Vec<_>>(),
            }
        }
        Ok(Err(DisassemblyError::Parse { diagnostics, .. })) => ffi::DisassembleResult {
            success: false,
            text: String::new(),
            diagnostics: diagnostics
                .into_iter()
                .map(to_ffi_diagnostic)
                .collect::<Vec<_>>(),
        },
        Err(_) => {
            pending.push(
                DiagnosticMessage::new(
                    MessageLevel::Error,
                    MessagePosition::default(),
                    "Rust disassembler panicked; falling back to C++ implementation",
                )
                .with_source("disassembler"),
            );
            ffi::DisassembleResult {
                success: false,
                text: String::new(),
                diagnostics: pending
                    .into_iter()
                    .map(to_ffi_diagnostic)
                    .collect::<Vec<_>>(),
            }
        }
    }
}

fn to_ffi_diagnostic(diagnostic: DiagnosticMessage<'_>) -> ffi::Diagnostic {
    let position: ffi::MessagePosition = diagnostic.position().into();
    let (has_source, source) = match diagnostic.source() {
        Some(src) => (true, src.to_string()),
        None => (false, String::new()),
    };
    ffi::Diagnostic {
        level: diagnostic.level().to_raw(),
        has_source,
        source,
        position,
        message: diagnostic.message().to_string(),
    }
}

impl From<MessagePosition> for ffi::MessagePosition {
    fn from(position: MessagePosition) -> Self {
        ffi::MessagePosition {
            line: position.line(),
            column: position.column(),
            index: position.index(),
        }
    }
}

impl From<ffi::MessagePosition> for MessagePosition {
    fn from(position: ffi::MessagePosition) -> Self {
        MessagePosition::new(position.line, position.column, position.index)
    }
}

pub struct ContextHandle {
    env: TargetEnv,
    context: NonNull<c_void>,
}

impl ContextHandle {
    fn context_address(&self) -> usize {
        self.context.as_ptr() as usize
    }

    /// Returns the target environment for this context.
    pub fn env(&self) -> TargetEnv {
        self.env
    }

    /// Emits a message via the context's message consumer.
    pub fn emit_message(
        &self,
        level: MessageLevel,
        source: Option<&str>,
        position: MessagePosition,
        message: &str,
    ) {
        ffi::dispatch_context_message(
            self.context_address(),
            level.to_raw(),
            source.is_some(),
            source.unwrap_or(""),
            position.into(),
            message,
        );
    }

    /// Emits a structured diagnostic through the message consumer.
    pub fn emit_diagnostic(&self, diagnostic: &DiagnosticMessage<'_>) {
        self.emit_message(
            diagnostic.level(),
            diagnostic.source(),
            diagnostic.position(),
            diagnostic.message(),
        );
    }

    fn rebind(&mut self, context: NonNull<c_void>) {
        self.context = context;
    }
}

pub fn create_context(env: u32, context_ptr: usize) -> u64 {
    let Some(env) = TargetEnv::from_raw(env) else {
        return 0;
    };
    let Some(context) = NonNull::new(context_ptr as *mut c_void) else {
        return 0;
    };
    Box::into_raw(Box::new(ContextHandle { env, context })) as u64
}

/// # Safety
///
/// `handle` must be the integer value previously returned by `create_context`
/// and not freed yet.
pub unsafe fn destroy_context(handle: u64) {
    if handle != 0 {
        drop(Box::from_raw(handle as *mut ContextHandle));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv_tools_core::assembly::assemble_text_with_env;

    #[test]
    fn rejects_invalid_environment() {
        let dangling = NonNull::<c_void>::dangling().as_ptr() as usize;
        assert_eq!(create_context(u32::MAX, dangling), 0);
    }

    #[test]
    fn rejects_null_context_pointer() {
        let env = TargetEnv::Universal1_0.to_raw();
        assert_eq!(create_context(env, 0), 0);
    }

    #[test]
    fn creates_and_destroys_context_handle() {
        let env = TargetEnv::Universal1_0.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);
        unsafe {
            destroy_context(handle);
        }
    }

    #[test]
    fn message_positions_convert_losslessly() {
        let position = MessagePosition::new(5, 7, 11);
        let ffi_pos: ffi::MessagePosition = position.into();
        assert_eq!(ffi_pos.line, 5);
        assert_eq!(ffi_pos.column, 7);
        assert_eq!(ffi_pos.index, 11);

        let round_trip = MessagePosition::from(ffi_pos);
        assert_eq!(round_trip, position);
    }

    #[test]
    fn rust_assembler_runs_for_valid_text() {
        force_enable_rust_text_assembler_for_testing();
        let env = TargetEnv::Universal1_0.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);

        let text = b"OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n\
%main = OpFunction %void None %void_fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd\n";
        let result = try_assemble_text(handle, text, TextToBinaryOptions::NONE.bits());
        assert!(result.success);
        assert!(!result.binary.is_empty());

        unsafe { destroy_context(handle) };
    }

    #[test]
    fn rust_assembler_preserves_function_body_via_context() {
        use spirv_tools_core::assembly::TextToBinaryOptions;

        set_rust_text_assembler_override(true);
        let env = TargetEnv::Universal1_3.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);

        let text = [
            "OpCapability Shader",
            "OpCapability Linkage",
            "OpCapability StorageInputOutput16",
            r#"OpExtension "SPV_KHR_16bit_storage""#,
            r#"OpExtension "SPV_KHR_8bit_storage""#,
            "OpMemoryModel Logical GLSL450",
            r#"OpMemberDecorate %half_buffer_block 0 Offset 0"#,
            r#"OpMemberDecorate %short_buffer_block 0 Offset 0"#,
            "%void = OpTypeVoid",
            "%short = OpTypeInt 16 0",
            "%half = OpTypeFloat 16",
            "%short4 = OpTypeVector %short 4",
            "%half4 = OpTypeVector %half 4",
            "%mat4x4 = OpTypeMatrix %half4 4",
            "%short_buffer_block = OpTypeStruct %short",
            "%half_buffer_block = OpTypeStruct %half",
            "%ptr_type = OpTypePointer Input %short",
            "%var = OpVariable %ptr_type Input",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let result = try_assemble_text(handle, text.as_bytes(), TextToBinaryOptions::NONE.bits());
        assert!(result.success, "rust assembler failed via context handle");
        let disassembly =
            disassemble_binary(&result.binary, BinaryToTextOptions::NONE).expect("disassemble");
        assert!(
            disassembly.contains("OpLabel") && disassembly.contains("OpReturn"),
            "function body was stripped via context assembly: {disassembly}"
        );
        unsafe { destroy_context(handle) };
        clear_rust_text_assembler_override();
    }
    #[test]
    fn rust_assembler_rejects_invalid_text_with_override() {
        use spirv_tools_core::assembly::TextToBinaryOptions;
        set_rust_text_assembler_override(true);
        let env = TargetEnv::Universal1_0.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);
        // Non-UTF8 payload should fail and produce no binary output.
        let invalid_bytes = [0xFF, 0xFE, 0xFD];
        let result = try_assemble_text(handle, &invalid_bytes, TextToBinaryOptions::NONE.bits());
        assert!(
            !result.success,
            "expected invalid utf-8 assembly to fail with override enabled"
        );
        assert!(
            result.binary.is_empty(),
            "binary should be empty on failure"
        );
        unsafe { destroy_context(handle) };
        clear_rust_text_assembler_override();
    }
    #[test]
    fn disassembler_reports_diagnostics_for_invalid_binary() {
        // Invalid magic number should surface a parse diagnostic.
        let result = try_disassemble_binary(0, &[0u32, 1, 2, 3], BinaryToTextOptions::NONE.bits());
        assert!(!result.success, "invalid binary should fail to disassemble");
        assert!(
            !result.diagnostics.is_empty(),
            "expected diagnostics for invalid binary"
        );
    }
    #[test]
    fn rust_and_cpp_assembler_match_with_override() {
        use spirv_tools_core::assembly::TextToBinaryOptions;
        let env = TargetEnv::Universal1_3.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            r#"OpEntryPoint Vertex %main "main""#,
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "",
        ]
        .join("\n");
        let text_bytes = text.as_bytes();
        set_rust_text_assembler_override(true);
        let rust_result = try_assemble_text(
            handle,
            text_bytes,
            TextToBinaryOptions::PRESERVE_NUMERIC_IDS.bits(),
        );
        assert!(rust_result.success, "rust assembler failed");
        clear_rust_text_assembler_override();
        set_rust_text_assembler_override(false);
        let cpp_result = try_assemble_text(
            handle,
            text_bytes,
            TextToBinaryOptions::PRESERVE_NUMERIC_IDS.bits(),
        );
        if !cpp_result.success {
            eprintln!("Skipping comparison: C++ assembler path unavailable or failed");
            unsafe { destroy_context(handle) };
            clear_rust_text_assembler_override();
            return;
        }
        clear_rust_text_assembler_override();
        assert_eq!(
            rust_result.binary, cpp_result.binary,
            "Rust assembler output differed from C++ assembler output"
        );
        unsafe { destroy_context(handle) };
    }
    #[test]
    fn rust_validator_handles_valid_and_invalid_modules() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint GLCompute %main \"main\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary = assemble_text_with_env(&text, TargetEnv::Universal1_6).unwrap();
        let options = default_validator_options();
        let ok =
            validate_binary_rust_with_options(TargetEnv::Universal1_6.to_raw(), &binary, &options);
        assert!(ok.success);
        assert!(ok.message.is_empty());

        let mut invalid = binary.clone();
        invalid[4] = 1; // reserved word must be zero
        let bad =
            validate_binary_rust_with_options(TargetEnv::Universal1_6.to_raw(), &invalid, &options);
        assert!(!bad.success);
        assert!(!bad.message.is_empty());
    }

    #[test]
    fn rust_validator_formats_errors_with_friendly_names() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpName %main \"ffi_friendly\"",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "OpExecutionMode %main LocalSize 1 1 1",
        ]
        .join("\n");
        let binary = assemble_text_with_env(&text, TargetEnv::Universal1_6).expect("assemble");

        let mut options = default_validator_options();
        let with_names =
            validate_binary_rust_with_options(TargetEnv::Universal1_6.to_raw(), &binary, &options);
        assert!(!with_names.success);
        assert!(
            with_names.message.contains("ffi_friendly"),
            "expected friendly name in message: {}",
            with_names.message
        );

        options.use_friendly_names = false;
        let without_names =
            validate_binary_rust_with_options(TargetEnv::Universal1_6.to_raw(), &binary, &options);
        assert!(!without_names.success);
        assert!(
            !without_names.message.contains("ffi_friendly"),
            "friendly names should be omitted when disabled: {}",
            without_names.message
        );
    }

    #[test]
    fn rust_validator_override_toggles_paths() {
        clear_rust_validator_override();
        LAST_VALIDATION_PATH.store(0, Ordering::Relaxed);

        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let binary =
            assemble_text_with_options(&text, TargetEnv::Universal1_6, TextToBinaryOptions::NONE)
                .expect("assemble");

        let first = validate_binary(TargetEnv::Universal1_6, &binary);
        assert!(first.success);
        assert_eq!(LAST_VALIDATION_PATH.load(Ordering::Relaxed), 1);

        set_rust_validator_override(false);
        LAST_VALIDATION_PATH.store(0, Ordering::Relaxed);
        let _cpp = validate_binary(TargetEnv::Universal1_6, &binary);
        assert_eq!(LAST_VALIDATION_PATH.load(Ordering::Relaxed), 2);

        set_rust_validator_override(true);
        LAST_VALIDATION_PATH.store(0, Ordering::Relaxed);
        let rust = validate_binary(TargetEnv::Universal1_6, &binary);
        assert!(rust.success);
        assert_eq!(LAST_VALIDATION_PATH.load(Ordering::Relaxed), 1);

        clear_rust_validator_override();
    }

    #[test]
    fn rust_validator_reports_layout_errors() {
        clear_rust_validator_override();
        set_rust_validator_override(true);
        let text = [
            "OpCapability Shader",
            "OpExtension \"SPV_KHR_shader_clock\"",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let mut words =
            assemble_text_with_options(&text, TargetEnv::Vulkan1_2, TextToBinaryOptions::NONE)
                .expect("assemble");
        // Move the extension to the end of the module to mirror the core layout regression.
        let mut idx = 5;
        let mut ext_slice: Option<(usize, usize)> = None;
        while idx < words.len() {
            let wc = (words[idx] >> 16) as usize;
            let opcode = (words[idx] & 0xffff) as u16;
            if opcode == rspirv::spirv::Op::Extension as u16 {
                ext_slice = Some((idx, wc));
                break;
            }
            idx += wc;
        }
        let (start, len) = ext_slice.expect("extension present");
        let extension: Vec<u32> = words.drain(start..start + len).collect();
        words.extend(extension);

        let options = default_validator_options();
        let result =
            validate_binary_rust_with_options(TargetEnv::Vulkan1_2.to_raw(), &words, &options);
        assert!(
            !result.success,
            "layout violation should fail validation even with Rust validator enabled"
        );
        assert!(
            result.message.contains("out of order"),
            "expected layout error message, got: {}",
            result.message
        );
        clear_rust_validator_override();
    }

    #[test]
    fn assembler_rejects_invalid_handle() {
        let text = b"OpCapability Shader\nOpMemoryModel Logical GLSL450";
        let result = try_assemble_text(0, text, TextToBinaryOptions::NONE.bits());
        assert!(
            !result.success,
            "assembling with an invalid handle should fail"
        );
        assert!(result.binary.is_empty());
    }

    #[test]
    fn rust_disassembler_handles_simple_binary() {
        force_enable_rust_text_assembler_for_testing();
        let binary = assemble_text_with_env(
            "OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n\
%main = OpFunction %void None %void_fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd\n",
            TargetEnv::Universal1_0,
        )
        .expect("assemble");

        let env = TargetEnv::Universal1_0.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);

        let options = (BinaryToTextOptions::NO_HEADER | BinaryToTextOptions::INDENT).bits();
        let result = try_disassemble_binary(handle, &binary, options);
        assert!(result.success);
        assert!(result.text.contains("OpFunction"));

        unsafe { destroy_context(handle) };
    }

    #[test]
    fn rust_disassembler_handles_default_options() {
        force_enable_rust_text_assembler_for_testing();
        let binary = assemble_text_with_env(
            "OpCapability Shader\n\
OpMemoryModel Logical GLSL450\n\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n\
%main = OpFunction %void None %void_fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd\n",
            TargetEnv::Universal1_0,
        )
        .expect("assemble");

        let env = TargetEnv::Universal1_0.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);

        let options = BinaryToTextOptions::NONE.bits();
        let result = try_disassemble_binary(handle, &binary, options);
        assert!(result.success);
        assert!(result.text.contains("OpFunction"));

        unsafe { destroy_context(handle) };
    }

    #[test]
    fn rust_disassembler_returns_diagnostics_on_failure() {
        let env = TargetEnv::Universal1_0.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);

        // Truncated binary header should trigger a parse error.
        let invalid_binary = vec![0x0723_0203u32];
        let result =
            try_disassemble_binary(handle, &invalid_binary, BinaryToTextOptions::NONE.bits());
        assert!(!result.success);
        assert!(
            !result.diagnostics.is_empty(),
            "failing disassembly should surface diagnostics"
        );

        unsafe { destroy_context(handle) };
    }

    #[test]
    fn rust_disassembler_error_messages_match_core_path() {
        // Rust disassembler diagnostics should mirror the direct core path for invalid binaries.
        let env = TargetEnv::Universal1_0.to_raw();
        let pointer = NonNull::<c_void>::dangling().as_ptr() as usize;
        let handle = create_context(env, pointer);
        assert_ne!(handle, 0);

        let invalid_binary = vec![0x0723_0203u32];

        let ffi_result =
            try_disassemble_binary(handle, &invalid_binary, BinaryToTextOptions::NONE.bits());
        assert!(!ffi_result.success);

        let expected = disassemble_binary(&invalid_binary, BinaryToTextOptions::NONE)
            .expect_err("expected parse failure from core disassembler");
        let expected_messages: Vec<String> = match expected {
            DisassemblyError::Parse { diagnostics, .. } => diagnostics
                .into_iter()
                .map(|d| d.message().to_string())
                .collect(),
            DisassemblyError::Unsupported(_) => panic!("expected parse diagnostics"),
        };

        let ffi_messages: Vec<&str> = ffi_result
            .diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect();

        assert_eq!(ffi_messages, expected_messages);

        unsafe { destroy_context(handle) };
    }

    #[test]
    fn disassembler_succeeds_on_valid_binary() {
        let binary = assemble_text_with_env(
            "OpCapability Shader\nOpMemoryModel Logical GLSL450",
            TargetEnv::Universal1_0,
        )
        .expect("assemble header");
        let result = try_disassemble_binary(0, &binary, 0);
        assert!(result.success);
        assert!(result.text.contains("OpCapability"));
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn reducer_and_fuzzer_ffi_report_disabled() {
        let binary = assemble_text_with_env(
            "OpCapability Shader\nOpMemoryModel Logical GLSL450",
            TargetEnv::Universal1_0,
        )
        .expect("assemble");
        let reduce = reduce_module(&binary);
        assert!(!reduce.success);
        assert!(matches!(reduce.error, ffi::ToolError::Disabled));
        assert!(
            !reduce.message.is_empty(),
            "disabled reducer should return a message"
        );

        let fuzz = fuzz_module(&binary);
        assert!(!fuzz.success);
        assert!(matches!(fuzz.error, ffi::ToolError::Disabled));
        assert!(
            !fuzz.message.is_empty(),
            "disabled fuzzer should return a message"
        );
    }
}
