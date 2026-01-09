//! SPIR-V validation support.

use crate::error::{Diagnostic, Error, SpirvResult, TargetEnv};
use spirv_tools_core::{
    target_env::TargetEnv as CoreTargetEnv,
    validation::{validate_module_with_options, ValidationError, ValidationOptions as CoreOptions},
};

/// Limits that can be configured for validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatorLimits {
    StructDepth,
    StructMemberCount,
    StructLocalSize,
    PushConstantSize,
    WorkgroupSize,
    FunctionParameterCount,
    FunctionControlFlowNestingDepth,
    IdBound,
    SwitchBranchCount,
}

/// Options for the SPIR-V validator.
#[derive(Default, Clone)]
pub struct ValidatorOptions {
    /// Record whether or not the validator should relax the rules on types for
    /// stores to structs.
    pub relax_struct_store: bool,
    /// Records whether or not the validator should relax the rules on pointer usage
    /// in logical addressing mode.
    pub relax_logical_pointer: bool,
    /// Records whether or not the validator should relax the rules because it is
    /// expected that the optimizations will make the code legal.
    pub before_legalization: bool,
    /// Records whether the validator should use "relaxed" block layout rules.
    pub relax_block_layout: Option<bool>,
    /// Records whether the validator should use standard block layout rules for
    /// uniform blocks.
    pub uniform_buffer_standard_layout: bool,
    /// Records whether the validator should use "scalar" block layout rules.
    pub scalar_block_layout: bool,
    /// Records whether or not the validator should skip validating standard
    /// uniform/storage block layout.
    pub skip_block_layout: bool,
    /// Applies a maximum to one or more Universal limits.
    pub max_limits: Vec<(ValidatorLimits, u32)>,
}

/// Trait for SPIR-V validators.
pub trait Validator: Default {
    fn with_env(target_env: TargetEnv) -> Self;
    fn validate(
        &self,
        binary: impl AsRef<[u32]>,
        options: Option<ValidatorOptions>,
    ) -> Result<(), Error>;
}

/// Create a validator for the given target environment.
pub fn create(te: Option<TargetEnv>) -> impl Validator {
    let target_env = te.unwrap_or_default();
    RustValidator::with_env(target_env)
}

/// A pure Rust implementation of the SPIR-V validator.
pub struct RustValidator {
    target_env: TargetEnv,
}

impl Default for RustValidator {
    fn default() -> Self {
        Self {
            target_env: TargetEnv::default(),
        }
    }
}

impl Validator for RustValidator {
    fn with_env(target_env: TargetEnv) -> Self {
        Self { target_env }
    }

    fn validate(
        &self,
        binary: impl AsRef<[u32]>,
        options: Option<ValidatorOptions>,
    ) -> Result<(), Error> {
        let words = binary.as_ref();
        let core_env = target_env_to_core(self.target_env);

        // Build core validation options
        let mut core_opts = CoreOptions::default();
        if let Some(opts) = options {
            core_opts.relax_struct_store = opts.relax_struct_store;
            core_opts.relax_logical_pointer = opts.relax_logical_pointer;
            core_opts.before_hlsl_legalization = opts.before_legalization;
            core_opts.relax_block_layout = opts.relax_block_layout.unwrap_or(false);
            core_opts.uniform_buffer_standard_layout = opts.uniform_buffer_standard_layout;
            core_opts.scalar_block_layout = opts.scalar_block_layout;
            core_opts.skip_block_layout = opts.skip_block_layout;
        }

        // Run validation
        match validate_module_with_options(words, core_env, core_opts) {
            Ok(()) => Ok(()),
            Err(e) => Err(validation_error_to_error(e)),
        }
    }
}

fn target_env_to_core(env: TargetEnv) -> CoreTargetEnv {
    match env {
        TargetEnv::Universal_1_0 => CoreTargetEnv::Universal1_0,
        TargetEnv::Universal_1_1 => CoreTargetEnv::Universal1_1,
        TargetEnv::Universal_1_2 => CoreTargetEnv::Universal1_2,
        TargetEnv::Universal_1_3 => CoreTargetEnv::Universal1_3,
        TargetEnv::Universal_1_4 => CoreTargetEnv::Universal1_4,
        TargetEnv::Universal_1_5 => CoreTargetEnv::Universal1_5,
        TargetEnv::Universal_1_6 => CoreTargetEnv::Universal1_6,
        TargetEnv::Vulkan_1_0 => CoreTargetEnv::Vulkan1_0,
        TargetEnv::Vulkan_1_1 => CoreTargetEnv::Vulkan1_1,
        TargetEnv::Vulkan_1_1_Spirv_1_4 => CoreTargetEnv::Vulkan1_1Spirv1_4,
        TargetEnv::Vulkan_1_2 => CoreTargetEnv::Vulkan1_2,
        TargetEnv::Vulkan_1_3 => CoreTargetEnv::Vulkan1_3,
        TargetEnv::Vulkan_1_4 => CoreTargetEnv::Vulkan1_4,
        TargetEnv::OpenGL_4_0 => CoreTargetEnv::OpenGl4_0,
        TargetEnv::OpenGL_4_1 => CoreTargetEnv::OpenGl4_1,
        TargetEnv::OpenGL_4_2 => CoreTargetEnv::OpenGl4_2,
        TargetEnv::OpenGL_4_3 => CoreTargetEnv::OpenGl4_3,
        TargetEnv::OpenGL_4_5 => CoreTargetEnv::OpenGl4_5,
        TargetEnv::OpenCL_1_2 => CoreTargetEnv::OpenCl1_2,
        TargetEnv::OpenCL_2_0 => CoreTargetEnv::OpenCl2_0,
        TargetEnv::OpenCL_2_1 => CoreTargetEnv::OpenCl2_1,
        TargetEnv::OpenCL_2_2 => CoreTargetEnv::OpenCl2_2,
        TargetEnv::OpenCLEmbedded_1_2 => CoreTargetEnv::OpenClEmbedded1_2,
        TargetEnv::OpenCLEmbedded_2_0 => CoreTargetEnv::OpenClEmbedded2_0,
        TargetEnv::OpenCLEmbedded_2_1 => CoreTargetEnv::OpenClEmbedded2_1,
        TargetEnv::OpenCLEmbedded_2_2 => CoreTargetEnv::OpenClEmbedded2_2,
        TargetEnv::WebGPU_0_DEPRECATED => CoreTargetEnv::WebGpu0,
    }
}

fn validation_error_to_error(e: ValidationError) -> Error {
    let message = e.to_string();
    Error {
        inner: SpirvResult::InvalidBinary,
        diagnostic: Some(Diagnostic {
            line: 0,
            column: 0,
            index: 0,
            message,
            notes: String::new(),
            is_text: false,
        }),
    }
}
