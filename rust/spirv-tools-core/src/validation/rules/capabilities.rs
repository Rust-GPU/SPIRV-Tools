//! Capability validation rules.
//!
//! Validates SPIR-V capability declarations against environment restrictions,
//! extension dependencies, and SPIR-V version requirements.
//!
//! # Implicit Declarations
//!
//! Per the spec's Capability table "Implicitly Declares" column, declaring a
//! capability transitively declares its dependencies. For example, declaring
//! `GroupNonUniformArithmetic` implicitly declares `GroupNonUniform`. The spec
//! says: "It is not necessary, but allowed, to explicitly declare an implicitly
//! declared capability."
//!
//! # Adding New Capability Rules
//!
//! 1. If it requires an extension, add to `required_extension()`
//! 2. If it requires a specific SPIR-V version, add to `required_version()`
//! 3. If it has aliases/supersession, add to `aliases()`

use std::collections::HashSet;

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::Capability;

use super::extensions::{
    extension_allowed_in_env, extension_satisfied, has_extension, ExtensionSet,
};
use crate::target_env::TargetEnv;
use crate::validation::capability_info::capability_info_from_grammar;
use crate::validation::error::ValidationError;
use crate::validation::types::ExtensionName;
use crate::version::SpirvVersion;

/// Extracts the Capability operand from an OpCapability instruction.
pub fn capability_operand(inst: &Instruction) -> Option<Capability> {
    inst.operands.iter().find_map(|op| match op {
        Operand::Capability(cap) => Some(*cap),
        _ => None,
    })
}

/// Collects explicit capabilities from a module's OpCapability instructions.
pub fn collect_explicit_capabilities(module: &Module) -> HashSet<Capability> {
    module
        .capabilities
        .iter()
        .filter_map(capability_operand)
        .collect()
}

/// Expands a set of explicitly declared capabilities to include all implicitly
/// declared capabilities (transitive closure per SPIR-V spec).
pub fn expand_implicit_capabilities(explicit: &HashSet<Capability>) -> HashSet<Capability> {
    let mut effective = explicit.clone();
    let mut worklist: Vec<Capability> = explicit.iter().copied().collect();
    while let Some(cap) = worklist.pop() {
        for &dep in capability_info_from_grammar(cap).required_capabilities {
            if effective.insert(dep) {
                worklist.push(dep);
            }
        }
    }
    effective
}

/// Validates all declared capabilities in a module.
///
/// Checks each explicitly declared capability against environment, extension,
/// and version requirements. Returns the effective capability set (explicit +
/// implicitly declared).
pub fn validate_capabilities(
    module: &Module,
    env: TargetEnv,
    target_version: SpirvVersion,
    extensions: &ExtensionSet,
) -> Result<HashSet<Capability>, ValidationError> {
    let explicit = collect_explicit_capabilities(module);
    let declared = expand_implicit_capabilities(&explicit);

    for &capability in &explicit {
        validate_opencl_image_rules(env, capability, &declared)?;
        validate_extension_allowed(env, target_version, capability)?;
        validate_version(env, target_version, capability, extensions)?;
        validate_extension_declared(env, target_version, capability, extensions)?;
        validate_env_allowed(env, capability, extensions, &declared)?;
    }

    Ok(declared)
}

/// OpenCL image capabilities require ImageBasic; ImageReadWrite needs OpenCL 2.0+.
fn validate_opencl_image_rules(
    env: TargetEnv,
    capability: Capability,
    declared: &HashSet<Capability>,
) -> Result<(), ValidationError> {
    if !env.is_opencl() {
        return Ok(());
    }

    // These OpenCL image caps require ImageBasic but don't implicitly declare it.
    if matches!(
        capability,
        Capability::LiteralSampler
            | Capability::Sampled1D
            | Capability::Image1D
            | Capability::SampledBuffer
            | Capability::ImageBuffer
    ) && !declared.contains(&Capability::ImageBasic)
    {
        return Err(ValidationError::MissingRequiredCapability {
            required_capability: Capability::ImageBasic,
            capability,
        });
    }

    // ImageReadWrite requires OpenCL 2.0+ (it implicitly declares ImageBasic via grammar).
    if capability == Capability::ImageReadWrite && env.is_opencl_1_2() {
        return Err(ValidationError::DisallowedCapability { capability, env });
    }

    Ok(())
}

/// Check that any required extension is allowed in the target environment.
fn validate_extension_allowed(
    env: TargetEnv,
    target_version: SpirvVersion,
    capability: Capability,
) -> Result<(), ValidationError> {
    let grammar = capability_info_from_grammar(capability);
    let manual_ext = required_extension(capability);

    if needs_grammar_extension(grammar.required_version, target_version, manual_ext) {
        let any_allowed = grammar
            .required_extensions
            .iter()
            .any(|&ext| extension_allowed_in_env(ext, env));
        if !any_allowed {
            if let Some(&ext) = grammar.required_extensions.first() {
                return Err(ValidationError::DisallowedExtension {
                    extension: ExtensionName::from(ext),
                    env,
                });
            }
        }
    }

    if needs_manual_extension(target_version, capability) {
        if let Some(ext) = manual_ext {
            if !extension_allowed_in_env(ext, env) {
                return Err(ValidationError::DisallowedExtension {
                    extension: ExtensionName::from(ext),
                    env,
                });
            }
        }
    }

    Ok(())
}

/// Check that the SPIR-V version is sufficient, or an enabling extension is declared.
fn validate_version(
    env: TargetEnv,
    target_version: SpirvVersion,
    capability: Capability,
    extensions: &ExtensionSet,
) -> Result<(), ValidationError> {
    use crate::validation::merge_versions;

    let grammar = capability_info_from_grammar(capability);
    let manual_ext = required_extension(capability);
    let min_version = merge_versions(grammar.required_version, required_version(capability));

    let Some(min_version) = min_version else {
        return Ok(());
    };
    if target_version >= min_version {
        return Ok(());
    }

    // Version too low — check if an enabling extension bridges the gap.
    let has_enabling_ext = grammar
        .required_extensions
        .iter()
        .any(|&ext| extension_satisfied(ext, extensions, target_version))
        || manual_ext.is_some_and(|ext| extension_satisfied(ext, extensions, target_version));

    if has_enabling_ext {
        return Ok(());
    }

    let has_any_extension = !grammar.required_extensions.is_empty() || manual_ext.is_some();
    if !env.is_capability_allowed(capability) && !has_any_extension {
        return Err(ValidationError::DisallowedCapability { capability, env });
    }

    Err(ValidationError::CapabilityRequiresSpirvVersion {
        capability,
        required_version: min_version,
        target_version,
    })
}

/// Check that any required extension is actually declared in the module.
fn validate_extension_declared(
    env: TargetEnv,
    target_version: SpirvVersion,
    capability: Capability,
    extensions: &ExtensionSet,
) -> Result<(), ValidationError> {
    let grammar = capability_info_from_grammar(capability);
    let manual_ext = required_extension(capability);

    // Grammar extensions (alternatives — any one suffices).
    if needs_grammar_extension(grammar.required_version, target_version, manual_ext)
        && !grammar.required_extensions.is_empty()
    {
        let any_satisfied = grammar
            .required_extensions
            .iter()
            .any(|&ext| extension_allowed_in_env(ext, env) && has_extension(extensions, ext));

        if !any_satisfied {
            let first_allowed = grammar
                .required_extensions
                .iter()
                .find(|&&ext| extension_allowed_in_env(ext, env));
            return match first_allowed {
                Some(&ext) => Err(ValidationError::DisallowedCapabilityMissingExtension {
                    capability,
                    required_extension: ext.to_string(),
                }),
                None => Err(ValidationError::DisallowedExtension {
                    extension: ExtensionName::from(grammar.required_extensions[0]),
                    env,
                }),
            };
        }
    }

    // Manual extension.
    if let Some(ext) = manual_ext {
        if needs_manual_extension(target_version, capability)
            && extension_allowed_in_env(ext, env)
            && !has_extension(extensions, ext)
        {
            return Err(ValidationError::DisallowedCapabilityMissingExtension {
                capability,
                required_extension: ext.to_string(),
            });
        }
    }

    Ok(())
}

/// Final check: capability must be allowed by env, extension, or OpenCL ImageBasic.
fn validate_env_allowed(
    env: TargetEnv,
    capability: Capability,
    extensions: &ExtensionSet,
    declared: &HashSet<Capability>,
) -> Result<(), ValidationError> {
    let allowed_by_env = env.is_capability_allowed(capability);
    let allowed_by_extension = is_enabled_by_extension(env, capability, extensions);
    let allowed_by_opencl_image = env.is_opencl()
        && declared.contains(&Capability::ImageBasic)
        && is_opencl_image_cap(capability);

    if !(allowed_by_env || allowed_by_extension || allowed_by_opencl_image) {
        return Err(ValidationError::DisallowedCapability { capability, env });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Extension requirement helpers
// ---------------------------------------------------------------------------

/// Whether the grammar-specified extension is currently required.
///
/// Extensions are required when:
/// - There is no core version (version is None — never promoted), OR
/// - The target version is below the core version, OR
/// - The associated extension is vendor-specific (never promoted)
fn needs_grammar_extension(
    grammar_version: Option<SpirvVersion>,
    target_version: SpirvVersion,
    manual_ext: Option<&str>,
) -> bool {
    let always_required = manual_ext.is_some_and(is_vendor_extension);
    grammar_version.is_none_or(|v| target_version < v) || always_required
}

/// Whether the manual extension is currently required.
fn needs_manual_extension(target_version: SpirvVersion, capability: Capability) -> bool {
    let Some(ext) = required_extension(capability) else {
        return false;
    };
    if is_vendor_extension(ext) {
        return true;
    }
    let version_allows_core = required_version(capability).is_some_and(|v| target_version >= v);
    !version_allows_core
}

/// Checks if a capability is enabled by a declared extension.
fn is_enabled_by_extension(
    env: TargetEnv,
    capability: Capability,
    extensions: &ExtensionSet,
) -> bool {
    let grammar = capability_info_from_grammar(capability);
    let from_grammar = grammar
        .required_extensions
        .iter()
        .any(|ext| extension_allowed_in_env(ext, env) && has_extension(extensions, ext));
    let from_manual = required_extension(capability)
        .is_some_and(|ext| extension_allowed_in_env(ext, env) && has_extension(extensions, ext));
    from_grammar || from_manual
}

/// OpenCL image capabilities that are enabled by ImageBasic.
fn is_opencl_image_cap(capability: Capability) -> bool {
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

/// Vendor/EXT extensions are never promoted to core SPIR-V.
fn is_vendor_extension(extension: &str) -> bool {
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

// ---------------------------------------------------------------------------
// Capability satisfaction (used by instruction validation)
// ---------------------------------------------------------------------------

/// Check if a capability requirement is satisfied by the declared capabilities,
/// considering aliases (e.g., KHR capabilities satisfying NV requirements).
pub fn capability_satisfied(required_cap: Capability, capabilities: &HashSet<Capability>) -> bool {
    capabilities.contains(&required_cap)
        || aliases(required_cap)
            .iter()
            .any(|alias| capabilities.contains(alias))
}

/// Capabilities that can satisfy a requirement for the given capability.
fn aliases(capability: Capability) -> &'static [Capability] {
    match capability {
        Capability::RayTracingNV => &[Capability::RayTracingKHR],
        Capability::MeshShadingNV => &[Capability::MeshShadingEXT],
        Capability::GroupNonUniformArithmetic => &[Capability::GroupNonUniformClustered],
        Capability::Kernel => &[
            Capability::GroupNonUniformArithmetic,
            Capability::GroupNonUniformBallot,
            Capability::GroupNonUniformClustered,
        ],
        Capability::Matrix => &[Capability::Shader],
        _ => &[],
    }
}

// ---------------------------------------------------------------------------
// Manual capability metadata (supplements the grammar)
// ---------------------------------------------------------------------------

/// Extension required by a capability (manual overrides not in the grammar).
pub fn required_extension(capability: Capability) -> Option<&'static str> {
    match capability {
        Capability::CooperativeMatrixKHR => Some("SPV_KHR_cooperative_matrix"),
        Capability::BindlessTextureNV => Some("SPV_NV_bindless_texture"),
        Capability::RayTracingNV => Some("SPV_NV_ray_tracing"),
        Capability::RayTracingKHR => Some("SPV_KHR_ray_tracing"),
        Capability::RayQueryKHR => Some("SPV_KHR_ray_query"),
        Capability::RayTracingPositionFetchKHR => Some("SPV_KHR_ray_tracing_position_fetch"),
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

/// Minimum SPIR-V version for a capability (manual overrides not in the grammar).
fn required_version(capability: Capability) -> Option<SpirvVersion> {
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
        | Capability::RayTracingProvisionalKHR
        | Capability::MeshShadingEXT
        | Capability::MeshShadingNV
        | Capability::FragmentShaderSampleInterlockEXT
        | Capability::FragmentShaderShadingRateInterlockEXT
        | Capability::FragmentShaderPixelInterlockEXT
        | Capability::PhysicalStorageBufferAddresses => Some(SpirvVersion::new(1, 4)),
        Capability::FragmentShadingRateKHR | Capability::FragmentDensityEXT => {
            Some(SpirvVersion::new(1, 5))
        }
        Capability::ShaderClockKHR
        | Capability::DeviceGroup
        | Capability::AtomicFloat16AddEXT
        | Capability::AtomicFloat32AddEXT
        | Capability::AtomicFloat64AddEXT
        | Capability::AtomicFloat16MinMaxEXT
        | Capability::AtomicFloat32MinMaxEXT
        | Capability::AtomicFloat64MinMaxEXT
        | Capability::AtomicFloat16VectorNV => Some(SpirvVersion::new(1, 3)),
        Capability::TileShadingQCOM => Some(SpirvVersion::new(1, 6)),
        _ => None,
    }
}
