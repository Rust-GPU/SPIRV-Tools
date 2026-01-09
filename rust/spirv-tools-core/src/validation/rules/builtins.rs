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
use crate::validation::types::ResultId;

// ============================================================================
// Built-in Location Exclusivity Rule
// ============================================================================

/// Validates that built-in decorated variables don't have Location/Component decorations.
pub struct BuiltinLocationExclusivityRule;

impl ValidationRule for BuiltinLocationExclusivityRule {
    fn name(&self) -> &'static str {
        "builtin-location-exclusivity"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                return Err(ValidationError::LocationConflictsWithBuiltIn);
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module;
        let definitions = ctx.definitions;
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
            if env.is_vulkan() && (builtin == BuiltIn::VertexId || builtin == BuiltIn::InstanceId) {
                return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env });
            }
            if !env.is_vulkan()
                && matches!(
                    builtin,
                    BuiltIn::ShadingRateKHR | BuiltIn::PrimitiveShadingRateKHR
                )
            {
                return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env });
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
                return Err(ValidationError::BuiltInDisallowedForEnv { builtin, env });
            }

            // Basic storage class check
            let allowed = matches!(storage_class, StorageClass::Input | StorageClass::Output);
            if !allowed {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                });
            }

            // ViewIndex special case
            if env.is_vulkan()
                && builtin == BuiltIn::ViewIndex
                && entry_models.contains(&ExecutionModel::GLCompute)
            {
                return Err(ValidationError::BuiltInRequiresExecutionModel {
                    builtin,
                    allowed: vec![
                        ExecutionModel::Vertex,
                        ExecutionModel::Geometry,
                        ExecutionModel::TessellationEvaluation,
                        ExecutionModel::MeshEXT,
                        ExecutionModel::MeshNV,
                    ],
                });
            }

            // Capability requirements
            if let Err(e) = check_builtin_capability(builtin, capabilities) {
                return Err(e);
            }

            // Fragment-only built-ins
            if is_fragment_only_builtin(builtin)
                && !entry_models.contains(&ExecutionModel::Fragment)
            {
                return Err(ValidationError::BuiltInRequiresFragment { builtin });
            }

            // Barycentric built-ins must be Input
            if is_barycentric_builtin(builtin) && storage_class != StorageClass::Input {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                });
            }

            if builtin == BuiltIn::ShadingRateKHR && storage_class != StorageClass::Input {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                });
            }

            // Mesh output-only built-ins
            if is_mesh_output_builtin(builtin) && storage_class != StorageClass::Output {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                });
            }

            if builtin == BuiltIn::PrimitiveShadingRateKHR && storage_class != StorageClass::Output
            {
                return Err(ValidationError::InvalidBuiltInStorageClass {
                    builtin,
                    storage_class,
                });
            }

            // Compute-only built-ins
            if is_compute_only_builtin(builtin)
                && !entry_models.contains(&ExecutionModel::GLCompute)
                && !entry_models.contains(&ExecutionModel::Kernel)
            {
                return Err(ValidationError::BuiltInRequiresExecutionModel {
                    builtin,
                    allowed: vec![ExecutionModel::GLCompute, ExecutionModel::Kernel],
                });
            }

            // Kernel-only built-ins
            if is_kernel_only_builtin(builtin) && !entry_models.contains(&ExecutionModel::Kernel) {
                return Err(ValidationError::BuiltInRequiresExecutionModel {
                    builtin,
                    allowed: vec![ExecutionModel::Kernel],
                });
            }

            // Type checks for selected built-ins
            if let Ok(var_id) = ResultId::try_from(*target) {
                if let Some(pointee) = resolve_builtin_pointee_type(definitions, var_id) {
                    if let Some(error) = validate_builtin_type(builtin, pointee, definitions) {
                        return Err(error);
                    }
                }

                if matches!(builtin, BuiltIn::TessLevelOuter | BuiltIn::TessLevelInner)
                    && !has_patch_decoration(module, var_id)
                {
                    return Err(ValidationError::BuiltInRequiresPatchDecoration { builtin });
                }
            }

            // Execution model allowlists
            if let Some(models) = required_execution_models(builtin) {
                if !entry_models.iter().any(|m| models.contains(m)) {
                    return Err(ValidationError::BuiltInRequiresExecutionModel {
                        builtin,
                        allowed: models.to_vec(),
                    });
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
) -> Result<(), ValidationError> {
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
        | BuiltIn::SubgroupLtMask => {
            Some(&[Capability::GroupNonUniformBallot, Capability::SubgroupBallotKHR])
        }
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
            });
        }
    }
    Ok(())
}

fn is_fragment_only_builtin(builtin: BuiltIn) -> bool {
    matches!(
        builtin,
        BuiltIn::FragCoord
            | BuiltIn::PointCoord
            | BuiltIn::FrontFacing
            | BuiltIn::SampleId
            | BuiltIn::SamplePosition
            | BuiltIn::SampleMask
            | BuiltIn::FragDepth
            | BuiltIn::HelperInvocation
            | BuiltIn::FragInvocationCountEXT
            | BuiltIn::FragSizeEXT
            | BuiltIn::FragStencilRefEXT
            | BuiltIn::FullyCoveredEXT
            | BuiltIn::BaryCoordKHR
            | BuiltIn::BaryCoordNoPerspKHR
            | BuiltIn::BaryCoordSmoothAMD
            | BuiltIn::BaryCoordSmoothCentroidAMD
            | BuiltIn::BaryCoordSmoothSampleAMD
            | BuiltIn::BaryCoordNoPerspAMD
            | BuiltIn::BaryCoordNoPerspCentroidAMD
            | BuiltIn::BaryCoordNoPerspSampleAMD
            | BuiltIn::BaryCoordPullModelAMD
    )
}

fn is_barycentric_builtin(builtin: BuiltIn) -> bool {
    matches!(
        builtin,
        BuiltIn::BaryCoordKHR
            | BuiltIn::BaryCoordNoPerspKHR
            | BuiltIn::BaryCoordSmoothAMD
            | BuiltIn::BaryCoordSmoothCentroidAMD
            | BuiltIn::BaryCoordSmoothSampleAMD
            | BuiltIn::BaryCoordNoPerspAMD
            | BuiltIn::BaryCoordNoPerspCentroidAMD
            | BuiltIn::BaryCoordNoPerspSampleAMD
            | BuiltIn::BaryCoordPullModelAMD
    )
}

fn is_mesh_output_builtin(builtin: BuiltIn) -> bool {
    matches!(
        builtin,
        BuiltIn::PrimitivePointIndicesEXT
            | BuiltIn::PrimitiveLineIndicesEXT
            | BuiltIn::PrimitiveTriangleIndicesEXT
            | BuiltIn::CullPrimitiveEXT
    )
}

fn is_compute_only_builtin(builtin: BuiltIn) -> bool {
    matches!(
        builtin,
        BuiltIn::GlobalInvocationId
            | BuiltIn::LocalInvocationId
            | BuiltIn::LocalInvocationIndex
            | BuiltIn::NumWorkgroups
            | BuiltIn::WorkgroupId
            | BuiltIn::NumSubgroups
            | BuiltIn::SubgroupId
            | BuiltIn::SubgroupLocalInvocationId
    )
}

fn is_kernel_only_builtin(builtin: BuiltIn) -> bool {
    matches!(
        builtin,
        BuiltIn::WorkDim
            | BuiltIn::GlobalSize
            | BuiltIn::GlobalOffset
            | BuiltIn::EnqueuedWorkgroupSize
            | BuiltIn::GlobalLinearId
            | BuiltIn::SubgroupMaxSize
            | BuiltIn::NumEnqueuedSubgroups
    )
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
            ExecutionModel::RayGenerationKHR,
            ExecutionModel::ClosestHitKHR,
            ExecutionModel::AnyHitKHR,
            ExecutionModel::MissKHR,
            ExecutionModel::IntersectionKHR,
            ExecutionModel::CallableKHR,
        ]),
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
        | BuiltIn::HitTNV => Some(&[
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
        _ => None,
    }
}

fn resolve_builtin_pointee_type<'a>(
    definitions: &'a HashMap<ResultId, Instruction>,
    var_id: ResultId,
) -> Option<&'a Instruction> {
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
    match builtin {
        BuiltIn::Position | BuiltIn::FragCoord => {
            // Must be vec4<f32>
            if !is_vec4_f32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec4<f32>",
                });
            }
        }
        BuiltIn::PointSize | BuiltIn::FragDepth => {
            // Must be f32
            if pointee.class.opcode != Op::TypeFloat {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "f32",
                });
            }
            if let Some(rspirv::dr::Operand::LiteralBit32(width)) = pointee.operands.first() {
                if *width != 32 {
                    return Some(ValidationError::InvalidBuiltInType {
                        builtin,
                        expected: "f32",
                    });
                }
            }
        }
        BuiltIn::VertexIndex | BuiltIn::InstanceIndex | BuiltIn::PrimitiveId => {
            // Must be i32 or u32
            if pointee.class.opcode != Op::TypeInt {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "i32 or u32",
                });
            }
            if let Some(rspirv::dr::Operand::LiteralBit32(width)) = pointee.operands.first() {
                if *width != 32 {
                    return Some(ValidationError::InvalidBuiltInType {
                        builtin,
                        expected: "i32 or u32",
                    });
                }
            }
        }
        BuiltIn::FrontFacing | BuiltIn::HelperInvocation => {
            // Must be bool
            if pointee.class.opcode != Op::TypeBool {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "bool",
                });
            }
        }
        BuiltIn::GlobalInvocationId
        | BuiltIn::LocalInvocationId
        | BuiltIn::NumWorkgroups
        | BuiltIn::WorkgroupId
        | BuiltIn::LaunchIdKHR
        | BuiltIn::LaunchSizeKHR => {
            // Must be vec3<i32>
            if !is_vec3_i32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec3<i32>",
                });
            }
        }
        BuiltIn::BaryCoordKHR | BuiltIn::BaryCoordNoPerspKHR => {
            // Must be vec3<f32>
            if !is_vec3_f32(pointee, definitions) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "vec3<f32>",
                });
            }
        }
        BuiltIn::ShadingRateKHR | BuiltIn::PrimitiveShadingRateKHR => {
            // Must be i32
            if !is_i32(pointee) {
                return Some(ValidationError::InvalidBuiltInType {
                    builtin,
                    expected: "i32",
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

// ============================================================================
// All builtin rules
// ============================================================================

/// Returns all built-in validation rules.
pub fn all_builtin_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&BuiltinLocationExclusivityRule, &BuiltinStorageClassRule]
}
