//! Built-in validation rules.
//!
//! This module validates SPIR-V built-in variable requirements including:
//!
//! - Built-in and location decoration exclusivity
//! - Built-in storage class requirements
//! - Built-in execution model requirements
//! - Built-in type requirements

use std::collections::{HashMap, HashSet};

use rspirv::dr::{Instruction, Module};
use rspirv::spirv::{BuiltIn, Capability, Decoration, ExecutionModel, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::op_ext::BuiltInExt;
use crate::validation::types::ResultId;
use crate::validation::ValidationResult;

// ============================================================================
// Built-in Location Exclusivity Rule
// ============================================================================

/// Validates that built-in decorated variables don't have Location/Component decorations.
pub struct BuiltinLocationExclusivityRule;

impl ValidationRule for BuiltinLocationExclusivityRule {
    fn name(&self) -> &'static str {
        "builtin-location-exclusivity"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;

        let mut built_ins: HashSet<ResultId> = HashSet::new();
        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
            ) = (inst.operands.first(), inst.operands.get(1))
            {
                if *decoration == Decoration::BuiltIn {
                    if let Ok(id) = ResultId::try_from(*target) {
                        built_ins.insert(id);
                    }
                }
            }
        }
        if built_ins.is_empty() {
            return Ok(());
        }

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };
            if *decoration != Decoration::Location && *decoration != Decoration::Component {
                continue;
            }
            let Ok(id) = ResultId::try_from(*target) else {
                continue;
            };
            if built_ins.contains(&id) {
                return Err(ValidationError::LocationConflictsWithBuiltIn.into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// Built-in Storage Class Rule
// ============================================================================

/// Validates that built-in variables use valid storage classes.
pub struct BuiltinStorageClassRule;

impl ValidationRule for BuiltinStorageClassRule {
    fn name(&self) -> &'static str {
        "builtin-storage-classes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;
        let entry_models = ctx.entry_models;
        let capabilities = ctx.declared_capabilities;
        let env = ctx.env;

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };
            if *decoration != Decoration::BuiltIn {
                continue;
            }
            let builtin = inst
                .operands
                .get(2)
                .and_then(|op| match op {
                    rspirv::dr::Operand::BuiltIn(b) => Some(*b),
                    rspirv::dr::Operand::LiteralBit32(raw) => BuiltIn::from_u32(*raw),
                    _ => None,
                })
                .unwrap_or(BuiltIn::Position);

            let Ok(id) = ResultId::try_from(*target) else {
                continue;
            };

            // Look up storage class of the variable.
            let storage_class = module.types_global_values.iter().find_map(|var| {
                if var.class.opcode != Op::Variable {
                    return None;
                }
                if var.result_id != Some(u32::from(id)) {
                    return None;
                }
                match var.operands.first() {
                    Some(rspirv::dr::Operand::StorageClass(sc)) => Some(*sc),
                    _ => None,
                }
            });
            let Some(storage_class) = storage_class else {
                continue;
            };

            if builtin == BuiltIn::WorkgroupSize {
                continue;
            }

            // Environment-specific built-in restrictions
            // VertexId is not allowed in Vulkan (use VertexIndex instead)
            if env.is_vulkan() && builtin == BuiltIn::VertexId {
                return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env }.into());
            }
            // InstanceId is not allowed in Vulkan vertex shaders (use InstanceIndex instead),
            // but IS allowed in ray tracing shaders (Intersection, AnyHit, ClosestHit)
            if env.is_vulkan()
                && builtin == BuiltIn::InstanceId
                && entry_models.contains(&ExecutionModel::Vertex)
                && !entry_models.iter().any(|m| {
                    matches!(
                        m,
                        ExecutionModel::IntersectionKHR
                            | ExecutionModel::AnyHitKHR
                            | ExecutionModel::ClosestHitKHR
                    )
                })
            {
                return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env }.into());
            }
            if !env.is_vulkan()
                && matches!(
                    builtin,
                    BuiltIn::ShadingRateKHR | BuiltIn::PrimitiveShadingRateKHR
                )
            {
                return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env }.into());
            }
            if !env.is_vulkan()
                && matches!(
                    builtin,
                    BuiltIn::PrimitivePointIndicesEXT
                        | BuiltIn::PrimitiveLineIndicesEXT
                        | BuiltIn::PrimitiveTriangleIndicesEXT
                        | BuiltIn::CullPrimitiveEXT
                )
            {
                return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env }.into());
            }

            // Basic storage class check
            let allowed = matches!(storage_class, StorageClass::Input | StorageClass::Output);
            if !allowed {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                }
                .into());
            }

            // Capability requirements
            check_builtin_capability(builtin, capabilities)?;

            // Fragment-only built-ins
            if builtin.is_fragment_only() && !entry_models.contains(&ExecutionModel::Fragment) {
                return Err(ValidationError::BuiltInRequiresFragment { builtin }.into());
            }

            // Barycentric built-ins must be Input
            if builtin.is_barycentric() && storage_class != StorageClass::Input {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                }
                .into());
            }

            if builtin == BuiltIn::ShadingRateKHR && storage_class != StorageClass::Input {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                }
                .into());
            }

            // Mesh output-only built-ins
            if builtin.is_mesh_output() && storage_class != StorageClass::Output {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                }
                .into());
            }

            if builtin == BuiltIn::PrimitiveShadingRateKHR && storage_class != StorageClass::Output
            {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                }
                .into());
            }

            // Compute-only built-ins
            if builtin.is_compute_only()
                && !entry_models.contains(&ExecutionModel::GLCompute)
                && !entry_models.contains(&ExecutionModel::Kernel)
            {
                return Err(ValidationError::BuiltInRequiresExecutionModel {
                    builtin,
                    allowed: vec![ExecutionModel::GLCompute, ExecutionModel::Kernel],
                }
                .into());
            }

            // Kernel-only built-ins
            if builtin.is_kernel_only() && !entry_models.contains(&ExecutionModel::Kernel) {
                return Err(ValidationError::BuiltInRequiresExecutionModel {
                    builtin,
                    allowed: vec![ExecutionModel::Kernel],
                }
                .into());
            }

            // Execution model allowlists
            if let Some(models) = required_execution_models(builtin) {
                if !entry_models.iter().any(|m| models.contains(m)) {
                    return Err(ValidationError::BuiltInRequiresExecutionModel {
                        builtin,
                        allowed: models.to_vec(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn check_builtin_capability(
    builtin: BuiltIn,
    capabilities: &HashSet<Capability>,
) -> ValidationResult {
    let required: Option<&[Capability]> = match builtin {
        BuiltIn::ShadingRateKHR | BuiltIn::PrimitiveShadingRateKHR => {
            Some(&[Capability::FragmentShadingRateKHR])
        }
        BuiltIn::ViewIndex => Some(&[Capability::MultiView]),
        BuiltIn::DeviceIndex => Some(&[Capability::DeviceGroup]),
        BuiltIn::WorkDim
        | BuiltIn::GlobalSize
        | BuiltIn::GlobalOffset
        | BuiltIn::EnqueuedWorkgroupSize
        | BuiltIn::GlobalLinearId
        | BuiltIn::SubgroupMaxSize => Some(&[Capability::Kernel]),
        BuiltIn::NumEnqueuedSubgroups => Some(&[Capability::DeviceEnqueue]),
        BuiltIn::WarpIDNV | BuiltIn::SMIDNV | BuiltIn::SMCountNV | BuiltIn::WarpsPerSMNV => {
            Some(&[Capability::ShaderSMBuiltinsNV])
        }
        BuiltIn::CoreIDARM
        | BuiltIn::CoreCountARM
        | BuiltIn::CoreMaxIDARM
        | BuiltIn::WarpIDARM
        | BuiltIn::WarpMaxIDARM => Some(&[Capability::CoreBuiltinsARM]),
        BuiltIn::SubgroupId
        | BuiltIn::NumSubgroups
        | BuiltIn::SubgroupLocalInvocationId
        | BuiltIn::SubgroupSize => Some(&[Capability::GroupNonUniform]),
        BuiltIn::SubgroupEqMask
        | BuiltIn::SubgroupGeMask
        | BuiltIn::SubgroupGtMask
        | BuiltIn::SubgroupLeMask
        | BuiltIn::SubgroupLtMask => Some(&[
            Capability::GroupNonUniformBallot,
            Capability::SubgroupBallotKHR,
        ]),
        BuiltIn::BaryCoordKHR
        | BuiltIn::BaryCoordNoPerspKHR
        | BuiltIn::BaryCoordSmoothAMD
        | BuiltIn::BaryCoordSmoothCentroidAMD
        | BuiltIn::BaryCoordSmoothSampleAMD
        | BuiltIn::BaryCoordNoPerspAMD
        | BuiltIn::BaryCoordNoPerspCentroidAMD
        | BuiltIn::BaryCoordNoPerspSampleAMD
        | BuiltIn::BaryCoordPullModelAMD => Some(&[Capability::FragmentBarycentricKHR]),
        BuiltIn::CullPrimitiveEXT
        | BuiltIn::PrimitivePointIndicesEXT
        | BuiltIn::PrimitiveLineIndicesEXT
        | BuiltIn::PrimitiveTriangleIndicesEXT => {
            Some(&[Capability::MeshShadingEXT, Capability::MeshShadingNV])
        }
        BuiltIn::LaunchIdKHR
        | BuiltIn::LaunchSizeKHR
        | BuiltIn::RayTminKHR
        | BuiltIn::RayTmaxKHR
        | BuiltIn::WorldRayOriginKHR
        | BuiltIn::WorldRayDirectionKHR
        | BuiltIn::ObjectRayOriginKHR
        | BuiltIn::ObjectRayDirectionKHR
        | BuiltIn::ObjectToWorldKHR
        | BuiltIn::WorldToObjectKHR
        | BuiltIn::InstanceCustomIndexKHR
        | BuiltIn::InstanceId
        | BuiltIn::RayGeometryIndexKHR
        | BuiltIn::IncomingRayFlagsKHR
        | BuiltIn::CullMaskKHR
        | BuiltIn::HitKindKHR
        | BuiltIn::HitTNV => Some(&[Capability::RayTracingKHR, Capability::RayTracingNV]),
        _ => None,
    };

    if let Some(required_caps) = required {
        if !required_caps.iter().any(|cap| capabilities.contains(cap)) {
            return Err(ValidationError::BuiltInRequiresCapability {
                builtin,
                capability: required_caps[0],
            }
            .into());
        }
    }
    Ok(())
}

fn required_execution_models(builtin: BuiltIn) -> Option<&'static [ExecutionModel]> {
    match builtin {
        BuiltIn::TessCoord | BuiltIn::TessLevelInner | BuiltIn::TessLevelOuter => {
            Some(&[ExecutionModel::TessellationEvaluation])
        }
        BuiltIn::PatchVertices => Some(&[ExecutionModel::TessellationControl]),
        BuiltIn::PrimitiveId => Some(&[
            ExecutionModel::Geometry,
            ExecutionModel::TessellationControl,
            ExecutionModel::TessellationEvaluation,
            ExecutionModel::MeshNV,
            ExecutionModel::MeshEXT,
            ExecutionModel::Fragment,
            ExecutionModel::IntersectionKHR,
            ExecutionModel::AnyHitKHR,
            ExecutionModel::ClosestHitKHR,
        ]),
        // HitKindKHR and HitTNV: only AnyHit and ClosestHit
        BuiltIn::HitKindKHR | BuiltIn::HitTNV => {
            Some(&[ExecutionModel::AnyHitKHR, ExecutionModel::ClosestHitKHR])
        }
        // Object space ray tracing built-ins: Intersection, AnyHit, ClosestHit only
        BuiltIn::InstanceCustomIndexKHR
        | BuiltIn::RayGeometryIndexKHR
        | BuiltIn::ObjectRayDirectionKHR
        | BuiltIn::ObjectRayOriginKHR
        | BuiltIn::ObjectToWorldKHR
        | BuiltIn::WorldToObjectKHR => Some(&[
            ExecutionModel::IntersectionKHR,
            ExecutionModel::AnyHitKHR,
            ExecutionModel::ClosestHitKHR,
        ]),
        // InstanceId in ray tracing context (not vertex shader)
        // Note: InstanceId is also valid in vertex shaders, but that's handled separately
        // in the capability check. For RT shaders, it's restricted.
        // The C++ validator handles this through ValidateRayTracingBuiltinsAtReference.
        // World space ray built-ins and ray parameters: Intersection, AnyHit, ClosestHit, Miss
        BuiltIn::IncomingRayFlagsKHR
        | BuiltIn::RayTminKHR
        | BuiltIn::RayTmaxKHR
        | BuiltIn::WorldRayDirectionKHR
        | BuiltIn::WorldRayOriginKHR
        | BuiltIn::CullMaskKHR => Some(&[
            ExecutionModel::IntersectionKHR,
            ExecutionModel::AnyHitKHR,
            ExecutionModel::ClosestHitKHR,
            ExecutionModel::MissKHR,
        ]),
        // LaunchId and LaunchSize: all ray tracing stages
        BuiltIn::LaunchIdKHR | BuiltIn::LaunchSizeKHR => Some(&[
            ExecutionModel::RayGenerationKHR,
            ExecutionModel::IntersectionKHR,
            ExecutionModel::AnyHitKHR,
            ExecutionModel::ClosestHitKHR,
            ExecutionModel::MissKHR,
            ExecutionModel::CallableKHR,
        ]),
        BuiltIn::ShadingRateKHR => Some(&[ExecutionModel::Fragment]),
        BuiltIn::PrimitivePointIndicesEXT
        | BuiltIn::PrimitiveLineIndicesEXT
        | BuiltIn::PrimitiveTriangleIndicesEXT
        | BuiltIn::CullPrimitiveEXT => Some(&[ExecutionModel::MeshEXT, ExecutionModel::MeshNV]),
        BuiltIn::PrimitiveShadingRateKHR => Some(&[
            ExecutionModel::Vertex,
            ExecutionModel::Geometry,
            ExecutionModel::MeshEXT,
            ExecutionModel::MeshNV,
        ]),
        BuiltIn::VertexIndex | BuiltIn::InstanceIndex => Some(&[ExecutionModel::Vertex]),
        // InvocationId is valid in Geometry and TessellationControl
        BuiltIn::InvocationId => Some(&[
            ExecutionModel::Geometry,
            ExecutionModel::TessellationControl,
        ]),
        // Layer and ViewportIndex can be used in multiple stages
        BuiltIn::Layer | BuiltIn::ViewportIndex => Some(&[
            ExecutionModel::Vertex,
            ExecutionModel::Geometry,
            ExecutionModel::TessellationEvaluation,
            ExecutionModel::MeshNV,
            ExecutionModel::MeshEXT,
            ExecutionModel::Fragment,
        ]),
        // ViewIndex requires MultiView capability and specific execution models
        BuiltIn::ViewIndex => Some(&[
            ExecutionModel::Vertex,
            ExecutionModel::Geometry,
            ExecutionModel::TessellationEvaluation,
            ExecutionModel::MeshNV,
            ExecutionModel::MeshEXT,
        ]),
        _ => None,
    }
}

fn resolve_builtin_pointee_type(
    definitions: &HashMap<ResultId, Instruction>,
    var_id: ResultId,
) -> Option<&Instruction> {
    let var_inst = definitions.get(&var_id)?;
    let ptr_type_id = var_inst.result_type?;
    let ptr_type = ResultId::try_from(ptr_type_id)
        .ok()
        .and_then(|id| definitions.get(&id))?;
    if ptr_type.class.opcode != Op::TypePointer {
        return None;
    }
    let pointee_id = match ptr_type.operands.get(1) {
        Some(rspirv::dr::Operand::IdRef(id)) => ResultId::try_from(*id).ok()?,
        _ => return None,
    };
    definitions.get(&pointee_id)
}

fn has_patch_decoration(module: &Module, target: ResultId) -> bool {
    module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && matches!(
                (inst.operands.first(), inst.operands.get(1)),
                (
                    Some(rspirv::dr::Operand::IdRef(t)),
                    Some(rspirv::dr::Operand::Decoration(Decoration::Patch))
                ) if t == &u32::from(target)
            )
    })
}

fn validate_builtin_type(
    builtin: BuiltIn,
    pointee: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<ValidationError> {
    // Type validation for specific built-ins
    // Based on C++ validate_builtins.cpp ValidateSingleBuiltInAtDefinition
    match builtin {
        // === vec4<f32> builtins (Position can be arrayed for mesh/geometry/tessellation shaders) ===
        BuiltIn::Position => {
            if !is_vec4_f32_or_array(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec4<f32> or array<vec4<f32>>",
                });
            }
        }
        BuiltIn::FragCoord => {
            if !is_vec4_f32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec4<f32>",
                });
            }
        }

        // === vec3<f32> builtins ===
        BuiltIn::BaryCoordKHR
        | BuiltIn::BaryCoordNoPerspKHR
        | BuiltIn::BaryCoordSmoothAMD
        | BuiltIn::BaryCoordSmoothCentroidAMD
        | BuiltIn::BaryCoordSmoothSampleAMD
        | BuiltIn::BaryCoordNoPerspAMD
        | BuiltIn::BaryCoordNoPerspCentroidAMD
        | BuiltIn::BaryCoordNoPerspSampleAMD
        | BuiltIn::TessCoord
        | BuiltIn::WorldRayOriginKHR
        | BuiltIn::WorldRayDirectionKHR
        | BuiltIn::ObjectRayOriginKHR
        | BuiltIn::ObjectRayDirectionKHR => {
            if !is_vec3_f32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec3<f32>",
                });
            }
        }

        // === vec2<f32> builtins ===
        BuiltIn::PointCoord | BuiltIn::SamplePosition | BuiltIn::FragSizeEXT => {
            if !is_vec2_f32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec2<f32>",
                });
            }
        }

        // === vec3<i32/u32> builtins ===
        BuiltIn::GlobalInvocationId
        | BuiltIn::LocalInvocationId
        | BuiltIn::NumWorkgroups
        | BuiltIn::WorkgroupId
        | BuiltIn::LaunchIdKHR
        | BuiltIn::LaunchSizeKHR => {
            if !is_vec3_i32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec3<i32>",
                });
            }
        }

        // === vec4<i32/u32> builtins (subgroup masks) ===
        BuiltIn::SubgroupEqMask
        | BuiltIn::SubgroupGeMask
        | BuiltIn::SubgroupGtMask
        | BuiltIn::SubgroupLeMask
        | BuiltIn::SubgroupLtMask => {
            if !is_vec4_i32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec4<i32>",
                });
            }
        }

        // === f32 scalar builtins ===
        BuiltIn::PointSize
        | BuiltIn::FragDepth
        | BuiltIn::RayTminKHR
        | BuiltIn::RayTmaxKHR
        | BuiltIn::HitTNV => {
            if !is_f32(pointee) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "f32",
                });
            }
        }

        // === i32 scalar builtins ===
        BuiltIn::VertexIndex
        | BuiltIn::InstanceIndex
        | BuiltIn::PrimitiveId
        | BuiltIn::InvocationId
        | BuiltIn::Layer
        | BuiltIn::ViewportIndex
        | BuiltIn::PatchVertices
        | BuiltIn::SampleId
        | BuiltIn::SubgroupId
        | BuiltIn::NumSubgroups
        | BuiltIn::SubgroupLocalInvocationId
        | BuiltIn::SubgroupSize
        | BuiltIn::LocalInvocationIndex
        | BuiltIn::ViewIndex
        | BuiltIn::DeviceIndex
        | BuiltIn::BaseInstance
        | BuiltIn::BaseVertex
        | BuiltIn::DrawIndex
        | BuiltIn::ShadingRateKHR
        | BuiltIn::PrimitiveShadingRateKHR
        | BuiltIn::FragInvocationCountEXT
        | BuiltIn::FragStencilRefEXT
        | BuiltIn::HitKindKHR
        | BuiltIn::InstanceCustomIndexKHR
        | BuiltIn::RayGeometryIndexKHR
        | BuiltIn::IncomingRayFlagsKHR
        | BuiltIn::CullMaskKHR
        | BuiltIn::CoreIDARM
        | BuiltIn::CoreCountARM
        | BuiltIn::CoreMaxIDARM
        | BuiltIn::WarpIDARM
        | BuiltIn::WarpMaxIDARM
        | BuiltIn::WarpsPerSMNV
        | BuiltIn::SMCountNV
        | BuiltIn::WarpIDNV
        | BuiltIn::SMIDNV => {
            if !is_i32(pointee) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "i32",
                });
            }
        }

        // === bool scalar builtins ===
        BuiltIn::FrontFacing | BuiltIn::HelperInvocation | BuiltIn::FullyCoveredEXT => {
            if pointee.class.opcode != Op::TypeBool {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "bool",
                });
            }
        }

        // === array<f32> builtins ===
        BuiltIn::ClipDistance | BuiltIn::CullDistance => {
            // Can be array<f32> or optionally arrayed (for per-vertex arrays)
            if !is_f32_array(pointee, definitions) && !is_f32(pointee) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "array<f32>",
                });
            }
        }

        // === array<i32> builtins ===
        BuiltIn::SampleMask => {
            if !is_i32_array(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "array<i32>",
                });
            }
        }

        // === array[4]<f32> builtins ===
        BuiltIn::TessLevelOuter => {
            if !is_f32_array_of_size(pointee, definitions, 4) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "array[4]<f32>",
                });
            }
        }

        // === array[2]<f32> builtins ===
        BuiltIn::TessLevelInner => {
            if !is_f32_array_of_size(pointee, definitions, 2) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "array[2]<f32>",
                });
            }
        }

        // === mat4x3<f32> builtins (ray tracing transforms) ===
        BuiltIn::ObjectToWorldKHR | BuiltIn::WorldToObjectKHR => {
            if !is_mat4x3_f32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "mat4x3<f32>",
                });
            }
        }

        // === Mesh shader built-ins ===

        // CullPrimitiveEXT: must be bool or array<bool>
        BuiltIn::CullPrimitiveEXT => {
            if pointee.class.opcode != Op::TypeBool && !is_bool_array(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "bool or array<bool>",
                });
            }
        }

        // PrimitivePointIndicesEXT: must be array<u32>
        BuiltIn::PrimitivePointIndicesEXT => {
            if !is_u32_array(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "array<u32>",
                });
            }
        }

        // PrimitiveLineIndicesEXT: must be array<uvec2>
        BuiltIn::PrimitiveLineIndicesEXT => {
            if !is_vec2_u32_array(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "array<uvec2>",
                });
            }
        }

        // PrimitiveTriangleIndicesEXT: must be array<uvec3>
        BuiltIn::PrimitiveTriangleIndicesEXT => {
            if !is_vec3_u32_array(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "array<uvec3>",
                });
            }
        }

        _ => {}
    }
    None
}

fn is_vec4_f32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    is_vec_of(ty, definitions, 4, Op::TypeFloat, 32)
}

/// Checks if type is `vec4<f32>` or `array<vec4<f32>>`.
/// Position can be arrayed in mesh/geometry/tessellation shaders.
fn is_vec4_f32_or_array(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if is_vec4_f32(ty, definitions) {
        return true;
    }
    // Check for array<vec4<f32>>
    if ty.class.opcode == Op::TypeArray {
        let elem_type_id = ty.operands.first().and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
            _ => None,
        });
        if let Some(elem_id) = elem_type_id {
            if let Some(elem_inst) = definitions.get(&elem_id) {
                return is_vec4_f32(elem_inst, definitions);
            }
        }
    }
    false
}

fn is_vec3_f32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    is_vec_of(ty, definitions, 3, Op::TypeFloat, 32)
}

fn is_vec3_i32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    is_vec_of(ty, definitions, 3, Op::TypeInt, 32)
}

fn is_vec_of(
    ty: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
    expected_count: u32,
    expected_component_op: Op,
    expected_width: u32,
) -> bool {
    if ty.class.opcode != Op::TypeVector {
        return false;
    }
    let component_count = ty.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(n) => Some(*n),
        _ => None,
    });
    if component_count != Some(expected_count) {
        return false;
    }
    let component_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(component_type_id) = component_type_id else {
        return false;
    };
    let Some(component_type) = definitions.get(&component_type_id) else {
        return false;
    };
    if component_type.class.opcode != expected_component_op {
        return false;
    }
    matches!(
        component_type.operands.first(),
        Some(rspirv::dr::Operand::LiteralBit32(w)) if *w == expected_width
    )
}

fn is_i32(ty: &Instruction) -> bool {
    if ty.class.opcode != Op::TypeInt {
        return false;
    }
    matches!(
        ty.operands.first(),
        Some(rspirv::dr::Operand::LiteralBit32(32))
    )
}

fn is_f32(ty: &Instruction) -> bool {
    if ty.class.opcode != Op::TypeFloat {
        return false;
    }
    matches!(
        ty.operands.first(),
        Some(rspirv::dr::Operand::LiteralBit32(32))
    )
}

fn is_vec2_f32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    is_vec_of(ty, definitions, 2, Op::TypeFloat, 32)
}

fn is_vec4_i32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    is_vec_of(ty, definitions, 4, Op::TypeInt, 32)
}

// Helper functions for mesh shader type validation
#[allow(dead_code)]
fn is_vec2_i32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    is_vec_of(ty, definitions, 2, Op::TypeInt, 32)
}

fn is_f32_array(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if ty.class.opcode != Op::TypeArray {
        return false;
    }
    let element_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(element_type_id) = element_type_id else {
        return false;
    };
    let Some(element_type) = definitions.get(&element_type_id) else {
        return false;
    };
    is_f32(element_type)
}

fn is_i32_array(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if ty.class.opcode != Op::TypeArray {
        return false;
    }
    let element_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(element_type_id) = element_type_id else {
        return false;
    };
    let Some(element_type) = definitions.get(&element_type_id) else {
        return false;
    };
    is_i32(element_type)
}

fn is_f32_array_of_size(
    ty: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
    expected_size: u32,
) -> bool {
    if !is_f32_array(ty, definitions) {
        return false;
    }
    // Check array size - operand 1 is the length constant ID
    let length_id = ty.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(length_id) = length_id else {
        return false;
    };
    let Some(length_inst) = definitions.get(&length_id) else {
        return false;
    };
    // Should be OpConstant with the expected value
    if length_inst.class.opcode != Op::Constant {
        return false;
    }
    matches!(
        length_inst.operands.first(),
        Some(rspirv::dr::Operand::LiteralBit32(n)) if *n == expected_size
    )
}

fn is_mat4x3_f32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    // Matrix is 4 columns of vec3<f32>
    if ty.class.opcode != Op::TypeMatrix {
        return false;
    }
    let column_count = ty.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(n) => Some(*n),
        _ => None,
    });
    if column_count != Some(4) {
        return false;
    }
    let column_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(column_type_id) = column_type_id else {
        return false;
    };
    let Some(column_type) = definitions.get(&column_type_id) else {
        return false;
    };
    is_vec3_f32(column_type, definitions)
}

/// Check if type is array of u32 (for PrimitivePointIndicesEXT).
fn is_u32_array(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if ty.class.opcode != Op::TypeArray {
        return false;
    }
    let element_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(element_type_id) = element_type_id else {
        return false;
    };
    let Some(element_type) = definitions.get(&element_type_id) else {
        return false;
    };
    // Must be u32 (unsigned 32-bit integer)
    if element_type.class.opcode != Op::TypeInt {
        return false;
    }
    let width = element_type.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(w) => Some(*w),
        _ => None,
    });
    let signedness = element_type.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(s) => Some(*s),
        _ => None,
    });
    width == Some(32) && signedness == Some(0)
}

/// Check if type is array of bool (for CullPrimitiveEXT).
fn is_bool_array(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if ty.class.opcode != Op::TypeArray {
        return false;
    }
    let element_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(element_type_id) = element_type_id else {
        return false;
    };
    let Some(element_type) = definitions.get(&element_type_id) else {
        return false;
    };
    element_type.class.opcode == Op::TypeBool
}

/// Check if type is uvec2 (2-component unsigned 32-bit integer vector).
fn is_vec2_u32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    is_vec_of_unsigned(ty, definitions, 2, 32)
}

/// Check if type is uvec3 (3-component unsigned 32-bit integer vector).
fn is_vec3_u32(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    is_vec_of_unsigned(ty, definitions, 3, 32)
}

/// Check if type is an unsigned integer vector with given component count and width.
fn is_vec_of_unsigned(
    ty: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
    expected_count: u32,
    expected_width: u32,
) -> bool {
    if ty.class.opcode != Op::TypeVector {
        return false;
    }
    let component_count = ty.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(n) => Some(*n),
        _ => None,
    });
    if component_count != Some(expected_count) {
        return false;
    }
    let component_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(component_type_id) = component_type_id else {
        return false;
    };
    let Some(component_type) = definitions.get(&component_type_id) else {
        return false;
    };
    if component_type.class.opcode != Op::TypeInt {
        return false;
    }
    let width = component_type.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(w) => Some(*w),
        _ => None,
    });
    let signedness = component_type.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(s) => Some(*s),
        _ => None,
    });
    width == Some(expected_width) && signedness == Some(0)
}

/// Check if type is array of uvec2 (for PrimitiveLineIndicesEXT).
fn is_vec2_u32_array(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if ty.class.opcode != Op::TypeArray {
        return false;
    }
    let element_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(element_type_id) = element_type_id else {
        return false;
    };
    let Some(element_type) = definitions.get(&element_type_id) else {
        return false;
    };
    is_vec2_u32(element_type, definitions)
}

/// Check if type is array of uvec3 (for PrimitiveTriangleIndicesEXT).
fn is_vec3_u32_array(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if ty.class.opcode != Op::TypeArray {
        return false;
    }
    let element_type_id = ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    });
    let Some(element_type_id) = element_type_id else {
        return false;
    };
    let Some(element_type) = definitions.get(&element_type_id) else {
        return false;
    };
    is_vec3_u32(element_type, definitions)
}

// ============================================================================
// Built-in Storage Class Direction Rule
// ============================================================================

/// Validates that built-in variables use the correct Input/Output direction
/// for each execution model.
///
/// For example:
/// - Position must be Output in Vertex, but can be Input in other stages
/// - FragCoord must be Input in Fragment
/// - ClipDistance/CullDistance can't be Input in Vertex/Mesh, can't be Output in Fragment
pub struct BuiltinStorageClassDirectionRule;

impl ValidationRule for BuiltinStorageClassDirectionRule {
    fn name(&self) -> &'static str {
        "builtin-storage-class-direction"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.env.is_vulkan() {
            return Ok(());
        }

        let module = ctx.module;
        let entry_models = ctx.entry_models;

        // Collect built-in decorations with their storage classes
        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };
            if *decoration != Decoration::BuiltIn {
                continue;
            }
            let builtin = inst
                .operands
                .get(2)
                .and_then(|op| match op {
                    rspirv::dr::Operand::BuiltIn(b) => Some(*b),
                    rspirv::dr::Operand::LiteralBit32(raw) => BuiltIn::from_u32(*raw),
                    _ => None,
                })
                .unwrap_or(BuiltIn::Position);

            let Ok(id) = ResultId::try_from(*target) else {
                continue;
            };

            // Look up storage class of the variable
            let storage_class = module.types_global_values.iter().find_map(|var| {
                if var.class.opcode != Op::Variable {
                    return None;
                }
                if var.result_id != Some(u32::from(id)) {
                    return None;
                }
                match var.operands.first() {
                    Some(rspirv::dr::Operand::StorageClass(sc)) => Some(*sc),
                    _ => None,
                }
            });
            let Some(storage_class) = storage_class else {
                continue;
            };

            // Validate direction per built-in per execution model
            for model in entry_models {
                if let Some(error) = validate_builtin_direction(builtin, storage_class, *model) {
                    return Err(error.into());
                }
            }
        }
        Ok(())
    }
}

/// Validates that the storage class direction is correct for the given built-in
/// and execution model combination.
fn validate_builtin_direction(
    builtin: BuiltIn,
    storage_class: StorageClass,
    model: ExecutionModel,
) -> Option<ValidationError> {
    match builtin {
        // Position: Output in Vertex/Mesh, Input not allowed in Vertex/Mesh
        BuiltIn::Position => {
            if storage_class == StorageClass::Input
                && matches!(
                    model,
                    ExecutionModel::Vertex | ExecutionModel::MeshNV | ExecutionModel::MeshEXT
                )
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // PointSize: Same rules as Position
        BuiltIn::PointSize => {
            if storage_class == StorageClass::Input
                && matches!(
                    model,
                    ExecutionModel::Vertex | ExecutionModel::MeshNV | ExecutionModel::MeshEXT
                )
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // ClipDistance/CullDistance: Input not allowed in Vertex/Mesh, Output not allowed in Fragment
        BuiltIn::ClipDistance | BuiltIn::CullDistance => {
            if storage_class == StorageClass::Input
                && matches!(
                    model,
                    ExecutionModel::Vertex | ExecutionModel::MeshNV | ExecutionModel::MeshEXT
                )
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
            if storage_class == StorageClass::Output && model == ExecutionModel::Fragment {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // FragCoord: Must be Input in Fragment
        BuiltIn::FragCoord => {
            if model == ExecutionModel::Fragment && storage_class != StorageClass::Input {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // FragDepth: Must be Output in Fragment
        BuiltIn::FragDepth => {
            if model == ExecutionModel::Fragment && storage_class != StorageClass::Output {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // FrontFacing, HelperInvocation, SampleId, SamplePosition, SampleMask (input):
        // Must be Input in Fragment
        BuiltIn::FrontFacing
        | BuiltIn::HelperInvocation
        | BuiltIn::SampleId
        | BuiltIn::SamplePosition => {
            if model == ExecutionModel::Fragment && storage_class != StorageClass::Input {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // VertexIndex, InstanceIndex: Must be Input in Vertex
        BuiltIn::VertexIndex
        | BuiltIn::InstanceIndex
        | BuiltIn::BaseVertex
        | BuiltIn::BaseInstance
        | BuiltIn::DrawIndex => {
            if model == ExecutionModel::Vertex && storage_class != StorageClass::Input {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // PrimitiveId: Input in Fragment, Output in Geometry/Mesh
        BuiltIn::PrimitiveId => {
            if model == ExecutionModel::Fragment && storage_class != StorageClass::Input {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
            if matches!(
                model,
                ExecutionModel::Geometry | ExecutionModel::MeshNV | ExecutionModel::MeshEXT
            ) && storage_class != StorageClass::Output
            {
                // In Geometry, PrimitiveId can be both Input and Output
                // Only check for mesh shaders where it must be Output
                if matches!(model, ExecutionModel::MeshNV | ExecutionModel::MeshEXT) {
                    return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                        builtin,
                        storage_class,
                        execution_model: model,
                    });
                }
            }
        }

        // Layer, ViewportIndex: Output in Vertex/Geometry/TessEval/Mesh, Input in Fragment
        BuiltIn::Layer | BuiltIn::ViewportIndex => {
            if model == ExecutionModel::Fragment && storage_class != StorageClass::Input {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
            if matches!(
                model,
                ExecutionModel::Vertex
                    | ExecutionModel::Geometry
                    | ExecutionModel::TessellationEvaluation
                    | ExecutionModel::MeshNV
                    | ExecutionModel::MeshEXT
            ) && storage_class != StorageClass::Output
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // InvocationId: Input in Geometry and TessellationControl
        BuiltIn::InvocationId => {
            if matches!(
                model,
                ExecutionModel::Geometry | ExecutionModel::TessellationControl
            ) && storage_class != StorageClass::Input
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // TessCoord: Must be Input in TessellationEvaluation
        BuiltIn::TessCoord => {
            if model == ExecutionModel::TessellationEvaluation
                && storage_class != StorageClass::Input
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // TessLevelOuter, TessLevelInner: Output in TessControl, Input in TessEval
        BuiltIn::TessLevelOuter | BuiltIn::TessLevelInner => {
            if model == ExecutionModel::TessellationControl && storage_class != StorageClass::Output
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
            if model == ExecutionModel::TessellationEvaluation
                && storage_class != StorageClass::Input
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // Compute shader built-ins: Must be Input
        BuiltIn::NumWorkgroups
        | BuiltIn::WorkgroupId
        | BuiltIn::LocalInvocationId
        | BuiltIn::GlobalInvocationId
        | BuiltIn::LocalInvocationIndex => {
            if matches!(model, ExecutionModel::GLCompute | ExecutionModel::Kernel)
                && storage_class != StorageClass::Input
            {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // Ray tracing built-ins: All are Input
        BuiltIn::LaunchIdKHR
        | BuiltIn::LaunchSizeKHR
        | BuiltIn::WorldRayOriginKHR
        | BuiltIn::WorldRayDirectionKHR
        | BuiltIn::ObjectRayOriginKHR
        | BuiltIn::ObjectRayDirectionKHR
        | BuiltIn::RayTminKHR
        | BuiltIn::RayTmaxKHR
        | BuiltIn::InstanceCustomIndexKHR
        | BuiltIn::ObjectToWorldKHR
        | BuiltIn::WorldToObjectKHR
        | BuiltIn::HitKindKHR
        | BuiltIn::HitTNV
        | BuiltIn::IncomingRayFlagsKHR
        | BuiltIn::RayGeometryIndexKHR
        | BuiltIn::CullMaskKHR => {
            if storage_class != StorageClass::Input {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // PrimitiveShadingRateKHR: Must be Output
        BuiltIn::PrimitiveShadingRateKHR => {
            if storage_class != StorageClass::Output {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        // ShadingRateKHR: Must be Input in Fragment
        BuiltIn::ShadingRateKHR => {
            if model == ExecutionModel::Fragment && storage_class != StorageClass::Input {
                return Some(ValidationError::BuiltInWrongStorageClassForExecutionModel {
                    builtin,
                    storage_class,
                    execution_model: model,
                });
            }
        }

        _ => {}
    }
    None
}

// ============================================================================
// Built-in Type Rule
// ============================================================================

/// Validates that built-in variables have the correct types.
///
/// In Vulkan environments, built-ins have specific type requirements:
/// - VertexIndex, InstanceIndex, PrimitiveId, etc. must be 32-bit int scalar
/// - Position, FragCoord must be vec4 of 32-bit float
/// - PointSize, FragDepth must be 32-bit float scalar
/// - FrontFacing, HelperInvocation must be bool
/// - GlobalInvocationId, LocalInvocationId, etc. must be vec3 of 32-bit int
/// - etc.
pub struct BuiltinTypeRule;

impl ValidationRule for BuiltinTypeRule {
    fn name(&self) -> &'static str {
        "builtin-types"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;
        let definitions = ctx.definitions;

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };
            if *decoration != Decoration::BuiltIn {
                continue;
            }
            let builtin = inst
                .operands
                .get(2)
                .and_then(|op| match op {
                    rspirv::dr::Operand::BuiltIn(b) => Some(*b),
                    rspirv::dr::Operand::LiteralBit32(raw) => BuiltIn::from_u32(*raw),
                    _ => None,
                })
                .unwrap_or(BuiltIn::Position);

            let Ok(var_id) = ResultId::try_from(*target) else {
                continue;
            };

            // Get the pointee type for the variable
            if let Some(pointee) = resolve_builtin_pointee_type(definitions, var_id) {
                if let Some(error) = validate_builtin_type(builtin, pointee, definitions) {
                    return Err(error.into());
                }
            }

            // TessLevel built-ins require Patch decoration
            if matches!(builtin, BuiltIn::TessLevelOuter | BuiltIn::TessLevelInner)
                && !has_patch_decoration(module, var_id)
            {
                return Err(ValidationError::BuiltInRequiresPatchDecoration { builtin }.into());
            }
        }
        Ok(())
    }
}

// ============================================================================
// Built-in Execution Mode Rule
// ============================================================================

/// Validates that built-in variables requiring specific execution modes have
/// those modes declared.
///
/// Examples:
/// - FragDepth requires DepthReplacing execution mode
/// - PrimitivePointIndicesEXT requires OutputPoints execution mode
/// - PrimitiveLineIndicesEXT requires OutputLinesEXT execution mode
/// - PrimitiveTriangleIndicesEXT requires OutputTrianglesEXT execution mode
pub struct BuiltinExecutionModeRule;

impl ValidationRule for BuiltinExecutionModeRule {
    fn name(&self) -> &'static str {
        "builtin-execution-modes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.env.is_vulkan() {
            return Ok(());
        }

        let module = ctx.module;

        // Collect execution modes for each entry point
        let mut entry_point_modes: HashMap<u32, HashSet<rspirv::spirv::ExecutionMode>> =
            HashMap::new();
        for mode_inst in &module.execution_modes {
            if mode_inst.class.opcode != Op::ExecutionMode
                && mode_inst.class.opcode != Op::ExecutionModeId
            {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(entry_point_id)) = mode_inst.operands.first()
            else {
                continue;
            };
            let Some(rspirv::dr::Operand::ExecutionMode(mode)) = mode_inst.operands.get(1) else {
                continue;
            };
            entry_point_modes
                .entry(*entry_point_id)
                .or_default()
                .insert(*mode);
        }

        // Check built-in decorations
        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };
            if *decoration != Decoration::BuiltIn {
                continue;
            }
            let builtin = inst
                .operands
                .get(2)
                .and_then(|op| match op {
                    rspirv::dr::Operand::BuiltIn(b) => Some(*b),
                    rspirv::dr::Operand::LiteralBit32(raw) => BuiltIn::from_u32(*raw),
                    _ => None,
                })
                .unwrap_or(BuiltIn::Position);

            // Check if this built-in requires a specific execution mode
            if let Some(required_mode) = required_execution_mode_for_builtin(builtin) {
                // Check all entry points to see if the required mode is declared
                let has_required_mode = entry_point_modes
                    .values()
                    .any(|modes| modes.contains(&required_mode));

                // If no entry points declare this mode, error
                if !has_required_mode && !entry_point_modes.is_empty() {
                    return Err(ValidationError::BuiltInRequiresExecutionModeDeclaration {
                        builtin,
                        required_mode,
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

/// Returns the execution mode required by a built-in, if any.
fn required_execution_mode_for_builtin(builtin: BuiltIn) -> Option<rspirv::spirv::ExecutionMode> {
    match builtin {
        // FragDepth requires DepthReplacing
        BuiltIn::FragDepth => Some(rspirv::spirv::ExecutionMode::DepthReplacing),
        // Mesh shader primitive indices require their corresponding output modes
        BuiltIn::PrimitivePointIndicesEXT => Some(rspirv::spirv::ExecutionMode::OutputPoints),
        BuiltIn::PrimitiveLineIndicesEXT => Some(rspirv::spirv::ExecutionMode::OutputLinesEXT),
        BuiltIn::PrimitiveTriangleIndicesEXT => {
            Some(rspirv::spirv::ExecutionMode::OutputTrianglesEXT)
        }
        _ => None,
    }
}

// ============================================================================
// All builtin rules
// ============================================================================

/// Returns all built-in validation rules.
pub fn all_builtin_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &BuiltinLocationExclusivityRule,
        &BuiltinStorageClassRule,
        &BuiltinStorageClassDirectionRule,
        &BuiltinTypeRule,
        &BuiltinExecutionModeRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::spirv::ExecutionModel;

    #[test]
    fn test_hit_kind_execution_models() {
        // HitKindKHR should only be valid in AnyHit and ClosestHit
        let models = required_execution_models(BuiltIn::HitKindKHR).unwrap();
        assert!(models.contains(&ExecutionModel::AnyHitKHR));
        assert!(models.contains(&ExecutionModel::ClosestHitKHR));
        assert!(!models.contains(&ExecutionModel::RayGenerationKHR));
        assert!(!models.contains(&ExecutionModel::MissKHR));
        assert!(!models.contains(&ExecutionModel::IntersectionKHR));
    }

    #[test]
    fn test_launch_id_execution_models() {
        // LaunchIdKHR should be valid in all RT stages
        let models = required_execution_models(BuiltIn::LaunchIdKHR).unwrap();
        assert!(models.contains(&ExecutionModel::RayGenerationKHR));
        assert!(models.contains(&ExecutionModel::IntersectionKHR));
        assert!(models.contains(&ExecutionModel::AnyHitKHR));
        assert!(models.contains(&ExecutionModel::ClosestHitKHR));
        assert!(models.contains(&ExecutionModel::MissKHR));
        assert!(models.contains(&ExecutionModel::CallableKHR));
    }

    #[test]
    fn test_object_ray_direction_execution_models() {
        // ObjectRayDirectionKHR should only be valid in Intersection, AnyHit, ClosestHit
        let models = required_execution_models(BuiltIn::ObjectRayDirectionKHR).unwrap();
        assert!(models.contains(&ExecutionModel::IntersectionKHR));
        assert!(models.contains(&ExecutionModel::AnyHitKHR));
        assert!(models.contains(&ExecutionModel::ClosestHitKHR));
        assert!(!models.contains(&ExecutionModel::MissKHR));
        assert!(!models.contains(&ExecutionModel::RayGenerationKHR));
    }

    #[test]
    fn test_world_ray_direction_execution_models() {
        // WorldRayDirectionKHR should be valid in Intersection, AnyHit, ClosestHit, Miss
        let models = required_execution_models(BuiltIn::WorldRayDirectionKHR).unwrap();
        assert!(models.contains(&ExecutionModel::IntersectionKHR));
        assert!(models.contains(&ExecutionModel::AnyHitKHR));
        assert!(models.contains(&ExecutionModel::ClosestHitKHR));
        assert!(models.contains(&ExecutionModel::MissKHR));
        assert!(!models.contains(&ExecutionModel::RayGenerationKHR));
    }

    #[test]
    fn test_layer_execution_models() {
        // Layer should be valid in multiple stages
        let models = required_execution_models(BuiltIn::Layer).unwrap();
        assert!(models.contains(&ExecutionModel::Vertex));
        assert!(models.contains(&ExecutionModel::Geometry));
        assert!(models.contains(&ExecutionModel::TessellationEvaluation));
        assert!(models.contains(&ExecutionModel::Fragment));
    }

    #[test]
    fn test_invocation_id_execution_models() {
        // InvocationId should be valid in Geometry and TessellationControl
        let models = required_execution_models(BuiltIn::InvocationId).unwrap();
        assert!(models.contains(&ExecutionModel::Geometry));
        assert!(models.contains(&ExecutionModel::TessellationControl));
        assert!(!models.contains(&ExecutionModel::Vertex));
        assert!(!models.contains(&ExecutionModel::Fragment));
    }
}
