use core::ffi::c_void;
use core::ptr::NonNull;
use spirv_tools_core::assembly::{
    assemble_text_with_env, BinaryToTextOptions, TextToBinaryOptions,
};
use spirv_tools_core::diagnostic::{DiagnosticMessage, MessagePosition};
use spirv_tools_core::disassembly::{self, disassemble_binary, DisassemblyError};
use spirv_tools_core::{MessageLevel, TargetEnv};
use std::panic::{self, AssertUnwindSafe};
use std::str;

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
        fn try_assemble_text(context_handle: u64, text: &[u8], options: u32) -> AssembleResult;
        fn try_disassemble_binary(
            context_handle: u64,
            binary: &[u32],
            options: u32,
        ) -> DisassembleResult;
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

/// Validates a SPIR-V module for the provided target environment.
pub fn validate_binary(env: TargetEnv, words: &[u32]) -> ffi::ValidateResult {
    ffi::validate_binary(env.to_raw(), words)
}

const ENABLE_RUST_TEXT_ASSEMBLER: bool = false;

pub fn try_assemble_text(context_handle: u64, text: &[u8], options: u32) -> ffi::AssembleResult {
    if context_handle == 0 {
        return ffi::AssembleResult {
            success: false,
            binary: Vec::new(),
        };
    }

    if !ENABLE_RUST_TEXT_ASSEMBLER {
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

    if let Ok(source) = str::from_utf8(text) {
        match panic::catch_unwind(AssertUnwindSafe(|| {
            assemble_text_with_env(source, context.env())
        })) {
            Ok(Ok(binary)) => {
                return ffi::AssembleResult {
                    success: true,
                    binary,
                };
            }
            Ok(Err(error)) => {
                for diagnostic in error.into_diagnostics() {
                    context.emit_diagnostic(&diagnostic);
                }
            }
            Err(_) => {
                context.emit_message(
                    MessageLevel::Error,
                    Some("assembler"),
                    MessagePosition::default(),
                    "Rust assembler panicked; falling back to C++ implementation",
                );
            }
        }
    } else {
        context.emit_message(
            MessageLevel::Error,
            Some("assembler"),
            MessagePosition::default(),
            "Assembly text must be valid UTF-8",
        );
    }

    ffi::assemble_text_with_context(context.context_address(), text, options)
}

pub fn try_disassemble_binary(
    context_handle: u64,
    binary: &[u32],
    options: u32,
) -> ffi::DisassembleResult {
    let requested =
        BinaryToTextOptions::from_bits_truncate(options & !BinaryToTextOptions::NONE.bits());

    if context_handle == 0 {
        return ffi::DisassembleResult {
            success: false,
            text: String::new(),
        };
    }

    let context = unsafe { (context_handle as *const ContextHandle).as_ref() };
    let Some(context) = context else {
        return ffi::DisassembleResult {
            success: false,
            text: String::new(),
        };
    };

    match disassemble_binary(binary, requested) {
        Ok(text) => ffi::DisassembleResult {
            success: true,
            text,
        },
        Err(DisassemblyError::Unsupported(_)) => ffi::DisassembleResult {
            success: false,
            text: String::new(),
        },
        Err(error) => {
            context.emit_message(
                MessageLevel::Error,
                Some("disassembler"),
                MessagePosition::default(),
                &error.to_string(),
            );
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
}
