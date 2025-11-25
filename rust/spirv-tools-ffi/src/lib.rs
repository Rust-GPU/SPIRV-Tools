use core::ffi::c_void;
use core::ptr::NonNull;
use spirv_tools_core::assembly::{
    assemble_text_with_options, BinaryToTextOptions, TextToBinaryOptions,
};
use spirv_tools_core::diagnostic::{DiagnosticMessage, MessagePosition};
use spirv_tools_core::disassembly::{self, disassemble_binary, DisassemblyError};
use spirv_tools_core::validation::MaybeValidModule;
use spirv_tools_core::{MessageLevel, TargetEnv};
use std::panic::{self, AssertUnwindSafe};
use std::str;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Once;

#[cfg(test)]
use spirv_tools_core::assembly::assemble_text_with_env;

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
    struct DisassembleResult {
        success: bool,
        text: String,
    }

    #[derive(Debug)]
    struct ValidateResult {
        success: bool,
        message: String,
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
        fn rebind_context(handle: u64, context_ptr: usize);
        fn try_assemble_text(context_handle: u64, text: &[u8], options: u32) -> AssembleResult;
        fn try_disassemble_binary(
            context_handle: u64,
            binary: &[u32],
            options: u32,
        ) -> DisassembleResult;
        fn validate_binary_rust(env: u32, binary: &[u32]) -> ValidateResult;
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

pub fn validate_binary_rust(env: u32, binary: &[u32]) -> ffi::ValidateResult {
    let env = match TargetEnv::from_raw(env) {
        Some(env) => env,
        None => {
            return ffi::ValidateResult {
                success: false,
                message: format!("unknown target environment {}", env),
            }
        }
    };
    match MaybeValidModule::Binary(binary).validate(env) {
        Ok(_) => ffi::ValidateResult {
            success: true,
            message: String::new(),
        },
        Err(err) => ffi::ValidateResult {
            success: false,
            message: err.to_string(),
        },
    }
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
    ffi::validate_binary(env.to_raw(), words)
}

static ENABLE_RUST_TEXT_ASSEMBLER: AtomicBool = AtomicBool::new(false);
static INIT_RUST_TEXT_ASSEMBLER: Once = Once::new();
static RUST_TEXT_ASSEMBLER_OVERRIDE: AtomicU8 = AtomicU8::new(0);

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

pub fn try_assemble_text(context_handle: u64, text: &[u8], options: u32) -> ffi::AssembleResult {
    if context_handle == 0 {
        return ffi::AssembleResult {
            success: false,
            binary: Vec::new(),
        };
    }

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
                return ffi::AssembleResult {
                    success: true,
                    binary,
                };
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

    let fallback = ffi::assemble_text_with_context(context.context_address(), text, options.into());
    if !fallback.success {
        for diagnostic in pending {
            context.emit_diagnostic(&diagnostic);
        }
    }
    fallback
}

pub fn try_disassemble_binary(
    context_handle: u64,
    binary: &[u32],
    options: u32,
) -> ffi::DisassembleResult {
    let requested =
        BinaryToTextOptions::from_bits_truncate(options & !BinaryToTextOptions::NONE.bits());

    let context = unsafe { (context_handle as *const ContextHandle).as_ref() };

    match disassemble_binary(binary, requested) {
        Ok(text) => ffi::DisassembleResult {
            success: true,
            text,
        },
        Err(DisassemblyError::Unsupported(unsupported)) => {
            if let Some(context) = context {
                let diagnostic = DiagnosticMessage::new(
                    MessageLevel::Error,
                    MessagePosition::default(),
                    format!("unsupported binary-to-text options: {unsupported:?}"),
                )
                .with_source("disassembler");
                context.emit_diagnostic(&diagnostic);
            }
            ffi::DisassembleResult {
                success: false,
                text: String::new(),
            }
        }
        Err(DisassemblyError::Parse { diagnostics, .. }) => {
            if let Some(context) = context {
                for diagnostic in &diagnostics {
                    context.emit_diagnostic(diagnostic);
                }
            }
            ffi::DisassembleResult {
                success: false,
                text: String::new(),
            }
        }
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
    fn rust_validator_handles_valid_and_invalid_modules() {
        let text = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;
        let binary = assemble_text_with_env(text, TargetEnv::Universal1_6).unwrap();
        let ok = validate_binary_rust(TargetEnv::Universal1_6.to_raw(), &binary);
        assert!(ok.success);
        assert!(ok.message.is_empty());

        let mut invalid = binary.clone();
        invalid[4] = 1; // reserved word must be zero
        let bad = validate_binary_rust(TargetEnv::Universal1_6.to_raw(), &invalid);
        assert!(!bad.success);
        assert!(!bad.message.is_empty());
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
}
