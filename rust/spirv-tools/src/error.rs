//! Error types and message handling for SPIR-V tools.

use std::fmt;
use std::str::FromStr;

/// The target environment for SPIR-V validation and optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum TargetEnv {
    // Universal SPIR-V versions
    Universal_1_0,
    Universal_1_1,
    Universal_1_2,
    Universal_1_3,
    Universal_1_4,
    Universal_1_5,
    Universal_1_6,

    // Vulkan versions
    #[default]
    Vulkan_1_0,
    Vulkan_1_1,
    Vulkan_1_1_Spirv_1_4,
    Vulkan_1_2,
    Vulkan_1_3,
    Vulkan_1_4,

    // OpenGL versions
    OpenGL_4_0,
    OpenGL_4_1,
    OpenGL_4_2,
    OpenGL_4_3,
    OpenGL_4_5,

    // OpenCL versions
    OpenCL_1_2,
    OpenCL_2_0,
    OpenCL_2_1,
    OpenCL_2_2,
    OpenCLEmbedded_1_2,
    OpenCLEmbedded_2_0,
    OpenCLEmbedded_2_1,
    OpenCLEmbedded_2_2,

    // WebGPU (deprecated)
    WebGPU_0_DEPRECATED,
}

impl TargetEnv {
    /// Returns the SPIR-V version for this target environment.
    pub fn spirv_version(&self) -> (u8, u8) {
        match self {
            Self::Universal_1_0
            | Self::Vulkan_1_0
            | Self::OpenGL_4_0
            | Self::OpenGL_4_1
            | Self::OpenGL_4_2
            | Self::OpenGL_4_3 => (1, 0),

            Self::Universal_1_1 | Self::OpenGL_4_5 => (1, 1),

            Self::Universal_1_2 | Self::OpenCL_1_2 | Self::OpenCLEmbedded_1_2 => (1, 2),

            Self::Universal_1_3
            | Self::Vulkan_1_1
            | Self::OpenCL_2_0
            | Self::OpenCL_2_1
            | Self::OpenCLEmbedded_2_0
            | Self::OpenCLEmbedded_2_1 => (1, 3),

            Self::Universal_1_4
            | Self::Vulkan_1_1_Spirv_1_4
            | Self::OpenCL_2_2
            | Self::OpenCLEmbedded_2_2 => (1, 4),

            Self::Universal_1_5 | Self::Vulkan_1_2 | Self::WebGPU_0_DEPRECATED => (1, 5),

            Self::Universal_1_6 | Self::Vulkan_1_3 | Self::Vulkan_1_4 => (1, 6),
        }
    }
}

impl fmt::Display for TargetEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Universal_1_0 => "spv1.0",
            Self::Universal_1_1 => "spv1.1",
            Self::Universal_1_2 => "spv1.2",
            Self::Universal_1_3 => "spv1.3",
            Self::Universal_1_4 => "spv1.4",
            Self::Universal_1_5 => "spv1.5",
            Self::Universal_1_6 => "spv1.6",
            Self::Vulkan_1_0 => "vulkan1.0",
            Self::Vulkan_1_1 => "vulkan1.1",
            Self::Vulkan_1_1_Spirv_1_4 => "vulkan1.1spv1.4",
            Self::Vulkan_1_2 => "vulkan1.2",
            Self::Vulkan_1_3 => "vulkan1.3",
            Self::Vulkan_1_4 => "vulkan1.4",
            Self::OpenGL_4_0 => "opengl4.0",
            Self::OpenGL_4_1 => "opengl4.1",
            Self::OpenGL_4_2 => "opengl4.2",
            Self::OpenGL_4_3 => "opengl4.3",
            Self::OpenGL_4_5 => "opengl4.5",
            Self::OpenCL_1_2 => "opencl1.2",
            Self::OpenCL_2_0 => "opencl2.0",
            Self::OpenCL_2_1 => "opencl2.1",
            Self::OpenCL_2_2 => "opencl2.2",
            Self::OpenCLEmbedded_1_2 => "opencl1.2embedded",
            Self::OpenCLEmbedded_2_0 => "opencl2.0embedded",
            Self::OpenCLEmbedded_2_1 => "opencl2.1embedded",
            Self::OpenCLEmbedded_2_2 => "opencl2.2embedded",
            Self::WebGPU_0_DEPRECATED => "webgpu0",
        };
        f.write_str(s)
    }
}

impl FromStr for TargetEnv {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "spv1.0" | "spirv1.0" => Ok(Self::Universal_1_0),
            "spv1.1" | "spirv1.1" => Ok(Self::Universal_1_1),
            "spv1.2" | "spirv1.2" => Ok(Self::Universal_1_2),
            "spv1.3" | "spirv1.3" => Ok(Self::Universal_1_3),
            "spv1.4" | "spirv1.4" => Ok(Self::Universal_1_4),
            "spv1.5" | "spirv1.5" => Ok(Self::Universal_1_5),
            "spv1.6" | "spirv1.6" => Ok(Self::Universal_1_6),
            "vulkan1.0" => Ok(Self::Vulkan_1_0),
            "vulkan1.1" => Ok(Self::Vulkan_1_1),
            "vulkan1.1spv1.4" => Ok(Self::Vulkan_1_1_Spirv_1_4),
            "vulkan1.2" => Ok(Self::Vulkan_1_2),
            "vulkan1.3" => Ok(Self::Vulkan_1_3),
            "vulkan1.4" => Ok(Self::Vulkan_1_4),
            "opengl4.0" => Ok(Self::OpenGL_4_0),
            "opengl4.1" => Ok(Self::OpenGL_4_1),
            "opengl4.2" => Ok(Self::OpenGL_4_2),
            "opengl4.3" => Ok(Self::OpenGL_4_3),
            "opengl4.5" => Ok(Self::OpenGL_4_5),
            "opencl1.2" => Ok(Self::OpenCL_1_2),
            "opencl2.0" => Ok(Self::OpenCL_2_0),
            "opencl2.1" => Ok(Self::OpenCL_2_1),
            "opencl2.2" => Ok(Self::OpenCL_2_2),
            "opencl1.2embedded" => Ok(Self::OpenCLEmbedded_1_2),
            "opencl2.0embedded" => Ok(Self::OpenCLEmbedded_2_0),
            "opencl2.1embedded" => Ok(Self::OpenCLEmbedded_2_1),
            "opencl2.2embedded" => Ok(Self::OpenCLEmbedded_2_2),
            "webgpu0" => Ok(Self::WebGPU_0_DEPRECATED),
            _ => Err(()),
        }
    }
}

/// Result type returned by SPIR-V operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpirvResult {
    Success,
    InvalidBinary,
    InvalidText,
    InvalidId,
    InvalidPointer,
    InvalidTable,
    InvalidValue,
    InvalidDiagnostic,
    InvalidLookup,
    InvalidTarget,
    InvalidCapability,
    Unsupported,
    EndOfStream,
    OutOfMemory,
    InternalError,
    MissingExtension,
    RequestedCapability,
}

impl std::error::Error for SpirvResult {}

impl fmt::Display for SpirvResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::Success => "success",
            Self::InvalidBinary => "invalid binary",
            Self::InvalidText => "invalid text",
            Self::InvalidId => "invalid id",
            Self::InvalidPointer => "invalid pointer",
            Self::InvalidTable => "invalid table",
            Self::InvalidValue => "invalid value",
            Self::InvalidDiagnostic => "invalid diagnostic",
            Self::InvalidLookup => "invalid lookup",
            Self::InvalidTarget => "invalid target",
            Self::InvalidCapability => "invalid capability",
            Self::Unsupported => "unsupported",
            Self::EndOfStream => "end of stream",
            Self::OutOfMemory => "out of memory",
            Self::InternalError => "internal error",
            Self::MissingExtension => "missing extension",
            Self::RequestedCapability => "requested capability",
        };
        f.write_str(msg)
    }
}

/// Diagnostic information from SPIR-V operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub line: usize,
    pub column: usize,
    pub index: usize,
    pub message: String,
    pub notes: String,
    pub is_text: bool,
}

impl From<String> for Diagnostic {
    fn from(message: String) -> Self {
        Self {
            line: 0,
            column: 0,
            index: 0,
            is_text: false,
            message,
            notes: String::new(),
        }
    }
}

impl From<Message> for Diagnostic {
    fn from(msg: Message) -> Self {
        Self {
            line: msg.line,
            column: msg.column,
            index: msg.index,
            message: msg.message,
            notes: msg.notes,
            is_text: false,
        }
    }
}

/// The main error type for SPIR-V operations.
#[derive(Debug, PartialEq)]
pub struct Error {
    pub inner: SpirvResult,
    pub diagnostic: Option<Diagnostic>,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.diagnostic {
            Some(diag) => {
                write!(f, "error:{}:{} - {}", diag.line, diag.column, diag.message)?;
                if !diag.notes.is_empty() {
                    write!(f, "\n{}", diag.notes)?;
                }
                Ok(())
            }
            None => f.write_str("an unknown error occurred"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.inner)
    }
}

/// Severity level for messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageLevel {
    Fatal,
    InternalError,
    Error,
    Warning,
    Info,
    Debug,
}

/// A message from SPIR-V operations (validation, optimization, etc.).
#[derive(Debug)]
pub struct Message {
    pub level: MessageLevel,
    pub source: Option<String>,
    pub line: usize,
    pub column: usize,
    pub index: usize,
    pub message: String,
    pub notes: String,
}

impl Message {
    /// Create a fatal error message.
    pub fn fatal(message: String) -> Self {
        Self {
            level: MessageLevel::Fatal,
            source: None,
            line: 0,
            column: 0,
            index: 0,
            message,
            notes: String::new(),
        }
    }

    /// Create an error message.
    pub fn error(message: String) -> Self {
        Self {
            level: MessageLevel::Error,
            source: None,
            line: 0,
            column: 0,
            index: 0,
            message,
            notes: String::new(),
        }
    }
}

/// Trait for callbacks that receive messages.
pub trait MessageCallback {
    fn on_message(&mut self, msg: Message);
}

impl<F> MessageCallback for F
where
    F: FnMut(Message),
{
    fn on_message(&mut self, msg: Message) {
        self(msg);
    }
}
