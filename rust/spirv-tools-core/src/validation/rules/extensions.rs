//! Extension validation rules.
//!
//! This module validates SPIR-V extension declarations and their requirements,
//! including environment restrictions, SPIR-V version requirements, and
//! extension promotion tracking.
//!
//! # Extension Validation
//!
//! SPIR-V extensions provide additional functionality beyond core SPIR-V. Key rules:
//!
//! - Extensions must be allowed by the target environment
//! - Some extensions require minimum SPIR-V versions
//! - Some extensions have been promoted to core in later SPIR-V versions
//! - Extensions can have aliases (e.g., KHR superseding NV)
//!
//! # Adding New Extension Rules
//!
//! To add support for a new extension:
//!
//! 1. If it requires a specific SPIR-V version, add to `required_spirv_version_for_extension()`
//! 2. If it was promoted to core, add to `extension_promoted_to_core_version()`
//! 3. If it has aliases/supersession, add to `extension_aliases()`

use std::collections::HashSet;

use rspirv::dr::{Instruction, Module, Operand};

use crate::target_env::TargetEnv;
use crate::validation::error::ValidationError;
use crate::validation::types::ExtensionName;
use crate::version::SpirvVersion;

// ============================================================================
// Extension set type
// ============================================================================

/// A set of declared extensions for a module.
#[derive(Debug, Default)]
pub struct ExtensionSet {
    /// The extension names that have been declared.
    pub values: HashSet<ExtensionName>,
}

impl ExtensionSet {
    /// Inserts an extension without checking for environment allowance.
    fn insert_unchecked(&mut self, extension: ExtensionName) -> Result<(), ValidationError> {
        if !self.values.insert(extension.clone()) {
            return Err(ValidationError::DuplicateExtension { extension });
        }
        Ok(())
    }

    /// Inserts an extension into the set, validating it against the environment.
    pub fn insert(&mut self, extension: ExtensionName, env: TargetEnv) -> Result<(), ValidationError> {
        self.insert_unchecked(extension.clone())?;
        if !env.is_extension_allowed(&extension) {
            return Err(ValidationError::DisallowedExtension { extension, env });
        }
        Ok(())
    }
}

// ============================================================================
// Extension extraction
// ============================================================================

/// Extracts the extension name from an OpExtension instruction.
pub fn extension_operand(inst: &Instruction) -> Option<ExtensionName> {
    inst.operands.iter().find_map(|operand| {
        if let Operand::LiteralString(extension) = operand {
            Some(ExtensionName::from(extension.as_str()))
        } else {
            None
        }
    })
}

/// Returns an iterator over all extension instructions in a module.
pub fn module_extension_instructions(
    module: &Module,
) -> impl Iterator<Item = &Instruction> {
    let top_level = module.extensions.iter();
    let function_bodies = module.functions.iter().flat_map(|function| {
        function.parameters.iter().chain(
            function
                .blocks
                .iter()
                .flat_map(|block| block.instructions.iter()),
        )
    });
    top_level.chain(function_bodies)
}

// ============================================================================
// Main validation functions
// ============================================================================

/// Validates that all extensions are allowed by the environment.
pub fn validate_extension_allowlist(module: &Module, env: TargetEnv) -> Result<(), ValidationError> {
    for inst in module_extension_instructions(module) {
        if let Some(extension) = extension_operand(inst) {
            if !env.is_extension_allowed(&extension) {
                return Err(ValidationError::DisallowedExtension { extension, env });
            }
        }
    }
    Ok(())
}

/// Validates all declared extensions and returns the extension set.
///
/// Checks that each extension:
/// - Is allowed by the target environment
/// - Meets SPIR-V version requirements
pub fn validate_extensions(
    module: &Module,
    env: TargetEnv,
    target_version: SpirvVersion,
) -> Result<ExtensionSet, ValidationError> {
    let mut extensions = ExtensionSet::default();
    for inst in &module.extensions {
        if let Some(extension) = extension_operand(inst) {
            let required_check = extension.clone();
            extensions.insert(extension, env)?;
            if let Some(required_version) = required_spirv_version_for_extension(&required_check) {
                if target_version < required_version {
                    return Err(ValidationError::ExtensionRequiresSpirvVersion {
                        extension: required_check,
                        required_version,
                        target_version,
                    });
                }
            }
        }
    }
    Ok(extensions)
}

// ============================================================================
// Extension helpers
// ============================================================================

/// Checks if an extension is declared in the set.
pub fn has_extension(extensions: &ExtensionSet, required_extension: &str) -> bool {
    extensions
        .values
        .iter()
        .any(|ext| ext.as_str() == required_extension)
}

/// Checks if an extension is allowed in the target environment.
pub fn extension_allowed_in_env(extension: &str, env: TargetEnv) -> bool {
    env.is_extension_allowed(&ExtensionName::from(extension))
}

// ============================================================================
// Extension satisfaction
// ============================================================================

/// Returns extensions that can satisfy the requirement for the given extension.
///
/// For example, SPV_KHR_ray_tracing can satisfy requirements for SPV_NV_ray_tracing.
pub fn extension_aliases(extension: &str) -> &'static [&'static str] {
    let normalized = extension.to_ascii_lowercase();
    match normalized.as_str() {
        // SPV_KHR_ray_tracing supersedes SPV_NV_ray_tracing
        "spv_nv_ray_tracing" => &["SPV_KHR_ray_tracing"],
        // SPV_EXT_mesh_shader supersedes SPV_NV_mesh_shader
        "spv_nv_mesh_shader" => &["SPV_EXT_mesh_shader"],
        // SPV_KHR_physical_storage_buffer supersedes SPV_EXT_physical_storage_buffer
        "spv_ext_physical_storage_buffer" => &["SPV_KHR_physical_storage_buffer"],
        _ => &[],
    }
}

/// Check if an extension requirement is satisfied by the declared extensions,
/// considering extension aliasing (e.g., KHR extensions satisfying NV requirements).
pub fn extension_satisfied(
    required_ext: &str,
    extensions: &ExtensionSet,
    target_version: SpirvVersion,
) -> bool {
    // First check if extension was promoted to core
    if let Some(promoted_version) = extension_promoted_to_core_version(required_ext) {
        if target_version >= promoted_version {
            return true;
        }
    }

    // Direct match (case-insensitive)
    let required_lower = required_ext.to_ascii_lowercase();
    if extensions
        .values
        .iter()
        .any(|ext| ext.as_str().to_ascii_lowercase() == required_lower)
    {
        return true;
    }

    // Check if any alias satisfies the requirement
    for &alias in extension_aliases(required_ext) {
        // Check if the alias is declared
        let alias_lower = alias.to_ascii_lowercase();
        if extensions
            .values
            .iter()
            .any(|ext| ext.as_str().to_ascii_lowercase() == alias_lower)
        {
            return true;
        }
        // Also check if the alias was promoted to core
        if let Some(promoted_version) = extension_promoted_to_core_version(alias) {
            if target_version >= promoted_version {
                return true;
            }
        }
    }

    false
}

// ============================================================================
// Extension metadata
// ============================================================================

/// Returns the minimum SPIR-V version required for an extension.
pub fn required_spirv_version_for_extension(extension: &ExtensionName) -> Option<SpirvVersion> {
    let normalized = extension.as_str().to_ascii_lowercase();
    match normalized.as_str() {
        "spv_khr_vulkan_memory_model" | "spv_qcom_cooperative_matrix_conversion" => {
            Some(SpirvVersion::new(1, 3))
        }
        "spv_khr_workgroup_memory_explicit_layout" => Some(SpirvVersion::new(1, 4)),
        "spv_khr_physical_storage_buffer" => Some(SpirvVersion::new(1, 4)),
        "spv_khr_ray_tracing" | "spv_khr_ray_query" | "spv_khr_ray_tracing_position_fetch" => {
            Some(SpirvVersion::new(1, 4))
        }
        "spv_ext_mesh_shader"
        | "spv_nv_shader_invocation_reorder"
        | "spv_nv_cluster_acceleration_structure"
        | "spv_nv_linear_swept_spheres"
        | "spv_ext_shader_invocation_reorder"
        | "spv_qcom_image_processing"
        | "spv_qcom_image_processing2" => Some(SpirvVersion::new(1, 4)),
        "spv_qcom_tile_shading" => Some(SpirvVersion::new(1, 6)),
        "spv_ext_fragment_shader_interlock" => Some(SpirvVersion::new(1, 4)),
        "spv_khr_fragment_shading_rate" | "spv_ext_fragment_invocation_density" => {
            Some(SpirvVersion::new(1, 5))
        }
        "spv_khr_storage_buffer_storage_class" | "spv_khr_variable_pointers" => {
            Some(SpirvVersion::new(1, 3))
        }
        "spv_khr_shader_clock" | "spv_khr_device_group" => Some(SpirvVersion::new(1, 3)),
        "spv_khr_maximal_reconvergence" => Some(SpirvVersion::new(1, 6)),
        "spv_ext_descriptor_indexing" => Some(SpirvVersion::new(1, 5)),
        _ => None,
    }
}

/// Returns the SPIR-V version at which an extension was promoted to core.
///
/// If the extension was promoted and the module's SPIR-V version is at least
/// this version, the extension is not required to be explicitly declared.
pub fn extension_promoted_to_core_version(extension: &str) -> Option<SpirvVersion> {
    let normalized = extension.to_ascii_lowercase();
    match normalized.as_str() {
        // VulkanMemoryModel was promoted to core in SPIR-V 1.5
        "spv_khr_vulkan_memory_model" => Some(SpirvVersion::new(1, 5)),
        // StorageBuffer storage class was promoted in SPIR-V 1.3
        "spv_khr_storage_buffer_storage_class" => Some(SpirvVersion::new(1, 3)),
        // Variable pointers was promoted in SPIR-V 1.3
        "spv_khr_variable_pointers" => Some(SpirvVersion::new(1, 3)),
        // Physical storage buffer addresses was promoted in SPIR-V 1.5
        "spv_khr_physical_storage_buffer" | "spv_ext_physical_storage_buffer" => {
            Some(SpirvVersion::new(1, 5))
        }
        // 16-bit storage was promoted in SPIR-V 1.3
        "spv_khr_16bit_storage" => Some(SpirvVersion::new(1, 3)),
        // 8-bit storage was promoted in SPIR-V 1.5
        "spv_khr_8bit_storage" => Some(SpirvVersion::new(1, 5)),
        // Shader atomic int64 was promoted in SPIR-V 1.6
        "spv_khr_shader_atomic_int64" => Some(SpirvVersion::new(1, 6)),
        // Terminate invocation was promoted in SPIR-V 1.6
        "spv_khr_terminate_invocation" => Some(SpirvVersion::new(1, 6)),
        // Float controls was promoted in SPIR-V 1.4
        "spv_khr_float_controls" => Some(SpirvVersion::new(1, 4)),
        // Subgroup vote was promoted in SPIR-V 1.3
        "spv_khr_subgroup_vote" => Some(SpirvVersion::new(1, 3)),
        // Shader ballot was promoted in SPIR-V 1.3
        "spv_khr_shader_ballot" => Some(SpirvVersion::new(1, 3)),
        // Multiview was promoted in SPIR-V 1.3
        "spv_khr_multiview" => Some(SpirvVersion::new(1, 3)),
        // Device group was promoted in SPIR-V 1.3
        "spv_khr_device_group" => Some(SpirvVersion::new(1, 3)),
        // Shader draw parameters was promoted in SPIR-V 1.3
        "spv_khr_shader_draw_parameters" => Some(SpirvVersion::new(1, 3)),
        // Post-depth coverage was promoted in SPIR-V 1.3
        "spv_khr_post_depth_coverage" => Some(SpirvVersion::new(1, 3)),
        // Demote to helper invocation was promoted in SPIR-V 1.6
        "spv_ext_demote_to_helper_invocation" => Some(SpirvVersion::new(1, 6)),
        // Non-semantic info was promoted in SPIR-V 1.6
        "spv_khr_non_semantic_info" => Some(SpirvVersion::new(1, 6)),
        // Integer dot product was promoted in SPIR-V 1.6
        "spv_khr_integer_dot_product" => Some(SpirvVersion::new(1, 6)),
        // Workgroup memory explicit layout was promoted in SPIR-V 1.4
        "spv_khr_workgroup_memory_explicit_layout" => Some(SpirvVersion::new(1, 4)),
        _ => None,
    }
}
