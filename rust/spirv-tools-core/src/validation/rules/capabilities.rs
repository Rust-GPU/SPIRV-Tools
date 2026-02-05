//! Capability validation rules.
//!
//! This module validates SPIR-V capability declarations and their requirements,
//! including environment restrictions, extension dependencies, and SPIR-V version
//! requirements.
//!
//! # Capability Validation
//!
//! SPIR-V capabilities control which features are available in a module. Key rules:
//!
//! - Capabilities must be allowed by the target environment
//! - Some capabilities require specific extensions
//! - Some capabilities require minimum SPIR-V versions
//! - Capabilities can have dependencies on other capabilities
//!
//! # Adding New Capability Rules
//!
//! To add support for a new capability:
//!
//! 1. If it requires an extension, add to `required_extension_for_capability()`
//! 2. If it requires a specific SPIR-V version, add to `manual_required_spirv_version_for_capability()`
//! 3. If it depends on other capabilities, add to `required_capabilities_for_capability()`
//! 4. If it has aliases/supersession, add to `capability_aliases()`

use std::collections::HashSet;

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::Capability;

use super::extensions::{
    extension_allowed_in_env, extension_satisfied, has_extension, ExtensionSet,
};
use crate::target_env::TargetEnv;
use crate::validation::error::ValidationError;
use crate::validation::types::ExtensionName;
use crate::version::SpirvVersion;

// ============================================================================
// Capability extraction
// ============================================================================

/// Extracts the Capability operand from an OpCapability instruction.
pub fn capability_operand(inst: &Instruction) -> Option<Capability> {
    inst.operands.iter().find_map(|operand| {
        if let Operand::Capability(cap) = operand {
            Some(*cap)
        } else {
            None
        }
    })
}

// ============================================================================
// Main validation function
// ============================================================================

/// Validates all declared capabilities in a module.
///
/// Checks that each capability:
/// - Is allowed by the target environment
/// - Has required extensions declared (if any)
/// - Meets SPIR-V version requirements
/// - Has required dependent capabilities declared
pub fn validate_capabilities(
    module: &Module,
    env: TargetEnv,
    target_version: SpirvVersion,
    extensions: &ExtensionSet,
) -> Result<HashSet<Capability>, ValidationError> {
    use crate::validation::capability_info::capability_info_from_grammar;
    use crate::validation::merge_versions;

    let declared: HashSet<_> = module
        .capabilities
        .iter()
        .filter_map(capability_operand)
        .collect();

    for inst in &module.capabilities {
        if let Some(capability) = capability_operand(inst) {
            let grammar_requirements = capability_info_from_grammar(capability);
            let allowed_by_env = env.is_capability_allowed(capability);

            if env.is_opencl()
                && matches!(
                    capability,
                    Capability::LiteralSampler
                        | Capability::Sampled1D
                        | Capability::Image1D
                        | Capability::SampledBuffer
                        | Capability::ImageBuffer
                        | Capability::ImageReadWrite
                )
                && !declared.contains(&Capability::ImageBasic)
            {
                return Err(ValidationError::MissingRequiredCapability {
                    required_capability: Capability::ImageBasic,
                    capability,
                });
            }

            // ImageReadWrite requires OpenCL 2.0+ (not just ImageBasic)
            if env.is_opencl_1_2() && capability == Capability::ImageReadWrite {
                return Err(ValidationError::DisallowedCapability {
                    capability,
                    env,
                });
            }

            let grammar_version = grammar_requirements.required_version;
            let required_version = merge_versions(
                grammar_version,
                manual_required_spirv_version_for_capability(capability),
            );

            let manual_required_extension = required_extension_for_capability(capability);
            let always_require_extension = manual_required_extension
                .map(extension_always_required)
                .unwrap_or(false);
            let version_allows_core = required_version
                .map(|required| target_version >= required)
                .unwrap_or(false);

            let grammar_requires_extension = !grammar_requirements.required_extensions.is_empty()
                && (grammar_version.is_none_or(|required| target_version < required)
                    || always_require_extension);
            let manual_requires_extension = manual_required_extension.is_some()
                && (always_require_extension || !version_allows_core);

            if grammar_requires_extension {
                // Grammar extension lists are alternatives (ANY suffices).
                // Only error if NONE of the alternative extensions is allowed.
                let any_ext_allowed = grammar_requirements
                    .required_extensions
                    .iter()
                    .any(|&ext| extension_allowed_in_env(ext, env));
                if !any_ext_allowed {
                    if let Some(&ext) = grammar_requirements.required_extensions.first() {
                        return Err(ValidationError::DisallowedExtension {
                            extension: ExtensionName::from(ext),
                            env,
                        });
                    }
                }
            }
            if manual_requires_extension {
                if let Some(required_ext) = manual_required_extension {
                    if !extension_allowed_in_env(required_ext, env) {
                        return Err(ValidationError::DisallowedExtension {
                            extension: ExtensionName::from(required_ext),
                            env,
                        });
                    }
                }
            }

            if let Some(required_version) = required_version {
                if target_version < required_version {
                    // Check if an enabling extension is available and declared
                    let has_grammar_extension =
                        !grammar_requirements.required_extensions.is_empty()
                            && grammar_requirements
                                .required_extensions
                                .iter()
                                .any(|&ext| extension_satisfied(ext, extensions, target_version));
                    let has_manual_extension = manual_required_extension
                        .map(|ext| extension_satisfied(ext, extensions, target_version))
                        .unwrap_or(false);

                    // If an enabling extension is declared, skip the version check
                    if !has_grammar_extension && !has_manual_extension {
                        let has_required_extension =
                            !grammar_requirements.required_extensions.is_empty()
                                || manual_required_extension.is_some();
                        if !allowed_by_env && !has_required_extension {
                            return Err(ValidationError::DisallowedCapability { capability, env });
                        }
                        return Err(ValidationError::CapabilityRequiresSpirvVersion {
                            capability,
                            required_version,
                            target_version,
                        });
                    }
                }
            }

            if !grammar_requirements.required_extensions.is_empty() && grammar_requires_extension {
                // Grammar extension lists are alternatives (ANY suffices).
                // The capability is valid if at least one listed extension is
                // both allowed in the environment and declared in the module.
                let any_ext_satisfied =
                    grammar_requirements.required_extensions.iter().any(|&ext| {
                        extension_allowed_in_env(ext, env) && has_extension(extensions, ext)
                    });
                if !any_ext_satisfied {
                    let any_allowed = grammar_requirements
                        .required_extensions
                        .iter()
                        .any(|&ext| extension_allowed_in_env(ext, env));
                    if !any_allowed {
                        return Err(ValidationError::DisallowedExtension {
                            extension: ExtensionName::from(
                                grammar_requirements.required_extensions[0],
                            ),
                            env,
                        });
                    }
                    // At least one is allowed but none is declared
                    let first_allowed = grammar_requirements
                        .required_extensions
                        .iter()
                        .find(|&&ext| extension_allowed_in_env(ext, env))
                        .unwrap();
                    return Err(ValidationError::DisallowedCapabilityMissingExtension {
                        capability,
                        required_extension: first_allowed.to_string(),
                    });
                }
            }

            if let Some(required_ext) = manual_required_extension {
                if manual_requires_extension {
                    if !extension_allowed_in_env(required_ext, env) {
                        return Err(ValidationError::DisallowedExtension {
                            extension: ExtensionName::from(required_ext),
                            env,
                        });
                    }
                    if !has_extension(extensions, required_ext) {
                        return Err(ValidationError::DisallowedCapabilityMissingExtension {
                            capability,
                            required_extension: required_ext.to_string(),
                        });
                    }
                }
            }

            let allowed_by_extension = capability_allowed_by_extension(env, capability, extensions);
            let allowed_by_capability =
                capability_enabled_by_capability(env, capability, &declared);

            // If the capability is not allowed by any means (env allowlist,
            // extension, or implied by another capability), reject it
            // immediately. This matches C++ spirv-val which checks env/ext
            // allowlists first.
            if !(allowed_by_env || allowed_by_extension || allowed_by_capability) {
                return Err(ValidationError::DisallowedCapability { capability, env });
            }

            for required_cap in grammar_requirements
                .required_capabilities
                .iter()
                .chain(required_capabilities_for_capability(capability).iter())
            {
                if is_soft_dependency(capability, *required_cap) {
                    continue;
                }
                if !declared.contains(required_cap) {
                    return Err(ValidationError::MissingRequiredCapability {
                        required_capability: *required_cap,
                        capability,
                    });
                }
            }
        }
    }

    Ok(declared)
}

// ============================================================================
// Capability enablement helpers
// ============================================================================

/// Checks if a capability is allowed by an extension.
pub fn capability_allowed_by_extension(
    env: TargetEnv,
    capability: Capability,
    extensions: &ExtensionSet,
) -> bool {
    use crate::validation::capability_info::capability_info_from_grammar;

    let grammar_requirements = capability_info_from_grammar(capability);
    grammar_requirements
        .required_extensions
        .iter()
        .any(|required_ext| {
            extension_allowed_in_env(required_ext, env) && has_extension(extensions, required_ext)
        })
        || required_extension_for_capability(capability)
            .map(|required_ext| {
                extension_allowed_in_env(required_ext, env)
                    && has_extension(extensions, required_ext)
            })
            .unwrap_or(false)
}

/// Checks if a capability is enabled by another capability (OpenCL-specific).
pub fn capability_enabled_by_capability(
    env: TargetEnv,
    capability: Capability,
    declared: &HashSet<Capability>,
) -> bool {
    if !env.is_opencl() {
        return false;
    }
    if !declared.contains(&Capability::ImageBasic) {
        return false;
    }
    matches!(
        capability,
        Capability::LiteralSampler
            | Capability::Sampled1D
            | Capability::Image1D
            | Capability::SampledBuffer
            | Capability::ImageBuffer
            | Capability::ImageReadWrite
    )
}

/// Returns true if the dependency is "soft" (implied rather than required).
pub fn is_soft_dependency(capability: Capability, required_capability: Capability) -> bool {
    matches!(
        (capability, required_capability),
        (Capability::Shader, Capability::Matrix)
    )
}

// ============================================================================
// Capability satisfaction
// ============================================================================

/// Returns capabilities that can satisfy the requirement for the given capability.
///
/// For example, RayTracingKHR can satisfy requirements for RayTracingNV since
/// KHR capabilities supersede their NV/EXT predecessors.
/// Also handles alternative capabilities from the grammar (e.g., GroupNonUniformArithmetic
/// can satisfy a Kernel requirement for GroupOperation operands).
pub fn capability_aliases(capability: Capability) -> &'static [Capability] {
    match capability {
        // RayTracingKHR supersedes RayTracingNV
        Capability::RayTracingNV => &[Capability::RayTracingKHR],
        // MeshShadingEXT supersedes MeshShadingNV
        Capability::MeshShadingNV => &[Capability::MeshShadingEXT],
        // GroupNonUniform capabilities - GroupNonUniformClustered implies others in hierarchy
        Capability::GroupNonUniformArithmetic => &[Capability::GroupNonUniformClustered],
        // Kernel capability can be satisfied by various GroupNonUniform capabilities
        // when used for GroupOperation operands (the grammar lists them as alternatives)
        Capability::Kernel => &[
            Capability::GroupNonUniformArithmetic,
            Capability::GroupNonUniformBallot,
            Capability::GroupNonUniformClustered,
        ],
        // Matrix capability is implied by Shader (Shader implies Matrix per SPIR-V spec)
        Capability::Matrix => &[Capability::Shader],
        _ => &[],
    }
}

/// Check if a capability requirement is satisfied by the declared capabilities,
/// considering capability aliases (e.g., KHR capabilities satisfying NV requirements).
pub fn capability_satisfied(required_cap: Capability, capabilities: &HashSet<Capability>) -> bool {
    // Direct match
    if capabilities.contains(&required_cap) {
        return true;
    }
    // Check if any alias satisfies the requirement
    for &alias in capability_aliases(required_cap) {
        if capabilities.contains(&alias) {
            return true;
        }
    }
    false
}

// ============================================================================
// Capability metadata
// ============================================================================

/// Returns the extension required for a capability (manual overrides).
pub fn required_extension_for_capability(capability: Capability) -> Option<&'static str> {
    match capability {
        Capability::CooperativeMatrixKHR => Some("SPV_KHR_cooperative_matrix"),
        Capability::BindlessTextureNV => Some("SPV_NV_bindless_texture"),
        Capability::RayTracingNV => Some("SPV_NV_ray_tracing"),
        Capability::RayTracingKHR => Some("SPV_KHR_ray_tracing"),
        Capability::RayQueryKHR => Some("SPV_KHR_ray_query"),
        Capability::RayTracingPositionFetchKHR => Some("SPV_KHR_ray_tracing_position_fetch"),
        // DemoteToHelperInvocation was added in SPIR-V 1.6, but available via extension before
        // (DemoteToHelperInvocationEXT is a const alias to DemoteToHelperInvocation with same value)
        Capability::DemoteToHelperInvocation => Some("SPV_EXT_demote_to_helper_invocation"),
        Capability::RayTracingMotionBlurNV => Some("SPV_NV_ray_tracing_motion_blur"),
        Capability::CooperativeMatrixNV => Some("SPV_NV_cooperative_matrix"),
        Capability::MeshShadingNV => Some("SPV_NV_mesh_shader"),
        Capability::MeshShadingEXT => Some("SPV_EXT_mesh_shader"),
        Capability::FragmentShadingRateKHR => Some("SPV_KHR_fragment_shading_rate"),
        Capability::FragmentDensityEXT => Some("SPV_EXT_fragment_invocation_density"),
        Capability::FragmentShaderSampleInterlockEXT
        | Capability::FragmentShaderShadingRateInterlockEXT
        | Capability::FragmentShaderPixelInterlockEXT => Some("SPV_EXT_fragment_shader_interlock"),
        Capability::ImageFootprintNV => Some("SPV_NV_shader_image_footprint"),
        Capability::RayTracingLinearSweptSpheresGeometryNV => Some("SPV_NV_linear_swept_spheres"),
        Capability::RayTracingDisplacementMicromapNV => Some("SPV_NV_displacement_micromap"),
        Capability::RayTracingOpacityMicromapEXT => Some("SPV_EXT_opacity_micromap"),
        Capability::AtomicFloat32MinMaxEXT
        | Capability::AtomicFloat64MinMaxEXT
        | Capability::AtomicFloat16MinMaxEXT => Some("SPV_EXT_shader_atomic_float_min_max"),
        Capability::AtomicFloat16AddEXT
        | Capability::AtomicFloat32AddEXT
        | Capability::AtomicFloat64AddEXT => Some("SPV_EXT_shader_atomic_float_add"),
        Capability::AtomicFloat16VectorNV => Some("SPV_NV_shader_atomic_float"),
        Capability::ShaderSMBuiltinsNV => Some("SPV_NV_shader_sm_builtins"),
        Capability::ShaderClockKHR => Some("SPV_KHR_shader_clock"),
        Capability::TileShadingQCOM => Some("SPV_QCOM_tile_shading"),
        Capability::SpecConditionalINTEL | Capability::FunctionVariantsINTEL => {
            Some("SPV_INTEL_function_variants")
        }
        Capability::CoreBuiltinsARM => Some("SPV_ARM_core_builtins"),
        _ => None,
    }
}

/// Returns true if an extension is always required (never promoted to core).
pub fn extension_always_required(extension: &str) -> bool {
    extension.starts_with("SPV_NV_")
        || extension.starts_with("SPV_EXT_")
        || extension.starts_with("SPV_AMD_")
        || extension.starts_with("SPV_QCOM_")
        || matches!(
            extension,
            "SPV_KHR_ray_tracing"
                | "SPV_KHR_ray_query"
                | "SPV_KHR_ray_tracing_position_fetch"
                | "SPV_KHR_vulkan_memory_model"
                | "SPV_KHR_shader_clock"
        )
}

/// Returns the manually specified minimum SPIR-V version for a capability.
pub fn manual_required_spirv_version_for_capability(
    capability: Capability,
) -> Option<SpirvVersion> {
    match capability {
        Capability::RayTracingKHR
        | Capability::RayTracingPositionFetchKHR
        | Capability::RayTracingNV
        | Capability::RayTracingMotionBlurNV
        | Capability::RayTracingOpacityMicromapEXT
        | Capability::RayTracingDisplacementMicromapNV
        | Capability::RayTracingSpheresGeometryNV
        | Capability::RayTracingLinearSweptSpheresGeometryNV
        | Capability::RayTracingClusterAccelerationStructureNV
        | Capability::RayQueryKHR
        | Capability::RayTracingProvisionalKHR => Some(SpirvVersion::new(1, 4)),
        Capability::MeshShadingEXT | Capability::MeshShadingNV => Some(SpirvVersion::new(1, 4)),
        Capability::FragmentShadingRateKHR | Capability::FragmentDensityEXT => {
            Some(SpirvVersion::new(1, 5))
        }
        Capability::FragmentShaderSampleInterlockEXT
        | Capability::FragmentShaderShadingRateInterlockEXT
        | Capability::FragmentShaderPixelInterlockEXT => Some(SpirvVersion::new(1, 4)),
        Capability::ShaderClockKHR => Some(SpirvVersion::new(1, 3)),
        Capability::DeviceGroup => Some(SpirvVersion::new(1, 3)),
        Capability::AtomicFloat16AddEXT
        | Capability::AtomicFloat32AddEXT
        | Capability::AtomicFloat64AddEXT
        | Capability::AtomicFloat16MinMaxEXT
        | Capability::AtomicFloat32MinMaxEXT
        | Capability::AtomicFloat64MinMaxEXT
        | Capability::AtomicFloat16VectorNV => Some(SpirvVersion::new(1, 3)),
        Capability::TileShadingQCOM => Some(SpirvVersion::new(1, 6)),
        Capability::PhysicalStorageBufferAddresses => Some(SpirvVersion::new(1, 4)),
        _ => None,
    }
}

/// Returns the capabilities that must be declared for a given capability.
pub fn required_capabilities_for_capability(capability: Capability) -> &'static [Capability] {
    match capability {
        // Shader-based feature capabilities require the Shader capability.
        Capability::Geometry
        | Capability::Tessellation
        | Capability::MeshShadingNV
        | Capability::MeshShadingEXT
        | Capability::RayTracingNV
        | Capability::RayTracingKHR
        | Capability::RayQueryKHR
        | Capability::RayTracingMotionBlurNV
        | Capability::RayTracingOpacityMicromapEXT
        | Capability::RayTracingDisplacementMicromapNV
        | Capability::RayTracingSpheresGeometryNV
        | Capability::RayTracingLinearSweptSpheresGeometryNV
        | Capability::RayTracingClusterAccelerationStructureNV
        | Capability::RayTracingPositionFetchKHR
        | Capability::FragmentShadingRateKHR
        | Capability::FragmentDensityEXT
        | Capability::FragmentShaderSampleInterlockEXT
        | Capability::FragmentShaderShadingRateInterlockEXT
        | Capability::FragmentShaderPixelInterlockEXT
        | Capability::SampleRateShading
        | Capability::ImageFootprintNV
        | Capability::ShaderSMBuiltinsNV
        | Capability::AtomicFloat16AddEXT
        | Capability::AtomicFloat32AddEXT
        | Capability::AtomicFloat64AddEXT
        | Capability::AtomicFloat16MinMaxEXT
        | Capability::AtomicFloat32MinMaxEXT
        | Capability::AtomicFloat64MinMaxEXT
        | Capability::AtomicFloat16VectorNV
        | Capability::TileShadingQCOM => &[Capability::Shader],
        // OpenCL address-related capabilities require Kernel.
        Capability::Addresses
        | Capability::GenericPointer
        | Capability::DeviceEnqueue
        | Capability::Pipes => &[Capability::Kernel],
        Capability::VariablePointers => &[Capability::VariablePointersStorageBuffer],
        Capability::VariablePointersStorageBuffer => &[Capability::Shader],
        Capability::GroupNonUniformVote
        | Capability::GroupNonUniformArithmetic
        | Capability::GroupNonUniformBallot
        | Capability::GroupNonUniformShuffle
        | Capability::GroupNonUniformShuffleRelative
        | Capability::GroupNonUniformClustered
        | Capability::GroupNonUniformQuad => &[Capability::GroupNonUniform],
        Capability::SubgroupDispatch => &[Capability::DeviceEnqueue],
        _ => &[],
    }
}
