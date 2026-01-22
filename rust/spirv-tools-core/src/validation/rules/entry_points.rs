//! Entry point validation rules.
//!
//! This module validates SPIR-V entry point requirements including:
//!
//! - Entry point interface storage class restrictions
//! - Ray tracing storage class restrictions
//! - Float encoding restrictions for interface variables

use std::collections::HashSet;

use rspirv::spirv::{Capability, Decoration, ExecutionModel, FPEncoding, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::{build_decoration_lookup, is_vulkan_env};
use crate::validation::types::{Id, ResultId};
use crate::validation::ValidationResult;

// ============================================================================
// Entry Point Interface Storage Classes Rule
// ============================================================================

/// Validates entry point interface variable storage classes.
pub struct EntryPointInterfaceStorageClassesRule;

impl ValidationRule for EntryPointInterfaceStorageClassesRule {
    fn name(&self) -> &'static str {
        "entry-point-interface-storage-classes"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        !is_vulkan_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let decoration_lookup = build_decoration_lookup(ctx.module);

        for ep in &ctx.module.entry_points {
            let mut operands = ep.operands.iter();
            if ep.class.opcode == Op::ConditionalEntryPointINTEL {
                let _ = operands.next();
            }
            // ExecutionModel
            let exec_model = operands
                .next()
                .and_then(|op| match op {
                    rspirv::dr::Operand::ExecutionModel(model) => Some(*model),
                    _ => None,
                })
                .ok_or(ValidationError::InvalidEntryPointOperand)?;
            let entry_point_id = operands
                .next()
                .and_then(|op| match op {
                    rspirv::dr::Operand::IdRef(ep_id) => Some(*ep_id),
                    _ => None,
                })
                .and_then(|raw| Id::try_from(raw).ok())
                .ok_or(ValidationError::InvalidEntryPointOperand)?;
            // Skip the name operand.
            let operands = operands.skip(1);
            let mut seen_push_constant = false;
            let mut seen_ray_payload = false;
            let mut seen_hit_attribute = false;
            let mut seen_callable_data = false;
            let mut seen_interface_ids: HashSet<Id> = HashSet::new();

            for operand in operands {
                let interface_id = match operand {
                    rspirv::dr::Operand::IdRef(id) => *id,
                    _ => continue,
                };
                if let Ok(id) = ResultId::try_from(interface_id) {
                    if let Some(inst) = ctx.definitions.get(&id) {
                        if let Some(rspirv::dr::Operand::StorageClass(storage)) =
                            inst.operands.first()
                        {
                            if !seen_interface_ids.insert(id.into()) {
                                return Err(ValidationError::DuplicateEntryPointInterface {
                                    entry_point: entry_point_id,
                                    interface: id.into(),
                                }
                                .into());
                            }
                            let has_patch = decoration_lookup
                                .get(&id)
                                .is_some_and(|decs| decs.contains(&Decoration::Patch));
                            let storage_allowed = matches!(
                                *storage,
                                StorageClass::Input
                                    | StorageClass::Output
                                    | StorageClass::Uniform
                                    | StorageClass::UniformConstant
                                    | StorageClass::PushConstant
                                    | StorageClass::StorageBuffer
                                    | StorageClass::PhysicalStorageBuffer
                                    | StorageClass::Workgroup
                                    | StorageClass::Private
                                    | StorageClass::IncomingRayPayloadKHR
                                    | StorageClass::RayPayloadKHR
                                    | StorageClass::HitAttributeKHR
                                    | StorageClass::IncomingCallableDataKHR
                                    | StorageClass::CallableDataKHR
                                    | StorageClass::ShaderRecordBufferKHR
                                    | StorageClass::TaskPayloadWorkgroupEXT
                            );
                            if !storage_allowed {
                                return Err(
                                    ValidationError::EntryPointInterfaceStorageClassInvalid {
                                        entry_point: entry_point_id,
                                        interface: id.into_inner(),
                                        storage_class: *storage,
                                    }
                                    .into(),
                                );
                            }
                            if has_patch
                                && !ctx
                                    .declared_capabilities
                                    .contains(&Capability::Tessellation)
                            {
                                return Err(ValidationError::DecorationRequiresCapability {
                                    decoration: Decoration::Patch,
                                    capability: Capability::Tessellation,
                                }
                                .into());
                            }
                            if has_patch
                                && !matches!(
                                    exec_model,
                                    ExecutionModel::TessellationControl
                                        | ExecutionModel::TessellationEvaluation
                                )
                            {
                                return Err(ValidationError::PatchDecorationRequiresTessellation {
                                    execution_model: exec_model,
                                }
                                .into());
                            }
                            if matches!(
                                exec_model,
                                ExecutionModel::RayGenerationKHR
                                    | ExecutionModel::IntersectionKHR
                                    | ExecutionModel::AnyHitKHR
                                    | ExecutionModel::ClosestHitKHR
                                    | ExecutionModel::MissKHR
                                    | ExecutionModel::CallableKHR
                            ) {
                                let allowed = matches!(
                                    *storage,
                                    StorageClass::IncomingRayPayloadKHR
                                        | StorageClass::RayPayloadKHR
                                        | StorageClass::HitAttributeKHR
                                        | StorageClass::IncomingCallableDataKHR
                                        | StorageClass::CallableDataKHR
                                        | StorageClass::PushConstant
                                        | StorageClass::ShaderRecordBufferKHR
                                        | StorageClass::UniformConstant
                                        | StorageClass::Input
                                        | StorageClass::Output
                                );
                                if !allowed {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassInvalid {
                                            entry_point: entry_point_id,
                                            interface: id.into_inner(),
                                            storage_class: *storage,
                                        }
                                        .into(),
                                    );
                                }
                            } else {
                                // Non-ray entry points cannot list ray-specific storage classes.
                                if matches!(
                                    *storage,
                                    StorageClass::IncomingRayPayloadKHR
                                        | StorageClass::RayPayloadKHR
                                        | StorageClass::HitAttributeKHR
                                        | StorageClass::IncomingCallableDataKHR
                                        | StorageClass::CallableDataKHR
                                        | StorageClass::ShaderRecordBufferKHR
                                ) {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassInvalid {
                                            entry_point: entry_point_id,
                                            interface: id.into_inner(),
                                            storage_class: *storage,
                                        }
                                        .into(),
                                    );
                                }
                            }
                            match storage {
                                StorageClass::PushConstant => {
                                    if seen_push_constant {
                                        return Err(
                                            ValidationError::EntryPointInterfaceStorageClassDuplicate {
                                                entry_point: entry_point_id,
                                                storage_class: *storage,
                                            }.into(),
                        );
                                    }
                                    seen_push_constant = true;
                                }
                                StorageClass::IncomingRayPayloadKHR => {
                                    if seen_ray_payload {
                                        return Err(
                                            ValidationError::EntryPointInterfaceStorageClassDuplicate {
                                                entry_point: entry_point_id,
                                                storage_class: *storage,
                                            }.into(),
                        );
                                    }
                                    seen_ray_payload = true;
                                }
                                StorageClass::HitAttributeKHR => {
                                    if seen_hit_attribute {
                                        return Err(
                                            ValidationError::EntryPointInterfaceStorageClassDuplicate {
                                                entry_point: entry_point_id,
                                                storage_class: *storage,
                                            }.into(),
                        );
                                    }
                                    seen_hit_attribute = true;
                                }
                                StorageClass::IncomingCallableDataKHR => {
                                    if seen_callable_data {
                                        return Err(
                                            ValidationError::EntryPointInterfaceStorageClassDuplicate {
                                                entry_point: entry_point_id,
                                                storage_class: *storage,
                                            }.into(),
                        );
                                    }
                                    seen_callable_data = true;
                                }
                                StorageClass::Input => {
                                    let allow_input = matches!(
                                        exec_model,
                                        ExecutionModel::Vertex
                                            | ExecutionModel::TessellationControl
                                            | ExecutionModel::TessellationEvaluation
                                            | ExecutionModel::Geometry
                                            | ExecutionModel::Fragment
                                            | ExecutionModel::MeshEXT
                                            | ExecutionModel::TaskEXT
                                            | ExecutionModel::GLCompute
                                            | ExecutionModel::RayGenerationKHR
                                            | ExecutionModel::IntersectionKHR
                                            | ExecutionModel::AnyHitKHR
                                            | ExecutionModel::ClosestHitKHR
                                            | ExecutionModel::MissKHR
                                            | ExecutionModel::CallableKHR
                                    );
                                    if !allow_input {
                                        return Err(
                                            ValidationError::EntryPointInterfaceStorageClassInvalid {
                                                entry_point: entry_point_id,
                                                interface: id.into_inner(),
                                                storage_class: *storage,
                                            }.into(),
                        );
                                    }
                                    if let Some(pointer_type) =
                                        inst.result_type.and_then(|ty| ResultId::try_from(ty).ok())
                                    {
                                        if let Some(pointer_inst) =
                                            ctx.definitions.get(&pointer_type)
                                        {
                                            if let Some(rspirv::dr::Operand::IdRef(pointee)) =
                                                pointer_inst.operands.get(1)
                                            {
                                                if let Ok(pointee_id) = ResultId::try_from(*pointee)
                                                {
                                                    let mut seen_types = HashSet::new();
                                                    if let Some(encoding) =
                                                        contains_disallowed_fp_encoding(
                                                            ctx.definitions,
                                                            pointee_id,
                                                            &mut seen_types,
                                                        )
                                                    {
                                                        return Err(
                                                            ValidationError::EntryPointInterfaceFloatEncodingInvalid {
                                                                interface: id.into(),
                                                                storage_class: *storage,
                                                                encoding,
                                                            }.into(),
                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                StorageClass::Output => {
                                    let allow_output = matches!(
                                        exec_model,
                                        ExecutionModel::Vertex
                                            | ExecutionModel::TessellationControl
                                            | ExecutionModel::TessellationEvaluation
                                            | ExecutionModel::Geometry
                                            | ExecutionModel::Fragment
                                            | ExecutionModel::MeshEXT
                                            | ExecutionModel::TaskEXT
                                    );
                                    if !allow_output {
                                        return Err(
                                            ValidationError::EntryPointInterfaceStorageClassInvalid {
                                                entry_point: entry_point_id,
                                                interface: id.into_inner(),
                                                storage_class: *storage,
                                            }.into(),
                        );
                                    }
                                    if let Some(pointer_type) =
                                        inst.result_type.and_then(|ty| ResultId::try_from(ty).ok())
                                    {
                                        if let Some(pointer_inst) =
                                            ctx.definitions.get(&pointer_type)
                                        {
                                            if let Some(rspirv::dr::Operand::IdRef(pointee)) =
                                                pointer_inst.operands.get(1)
                                            {
                                                if let Ok(pointee_id) = ResultId::try_from(*pointee)
                                                {
                                                    let mut seen_types = HashSet::new();
                                                    if let Some(encoding) =
                                                        contains_disallowed_fp_encoding(
                                                            ctx.definitions,
                                                            pointee_id,
                                                            &mut seen_types,
                                                        )
                                                    {
                                                        return Err(
                                                            ValidationError::EntryPointInterfaceFloatEncodingInvalid {
                                                                interface: id.into(),
                                                                storage_class: *storage,
                                                                encoding,
                                                            }.into(),
                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                StorageClass::Function => {
                                    return Err(
                                        ValidationError::EntryPointInterfaceStorageClassInvalid {
                                            entry_point: entry_point_id,
                                            interface: id.into_inner(),
                                            storage_class: *storage,
                                        }
                                        .into(),
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn contains_disallowed_fp_encoding(
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
    ty: ResultId,
    seen: &mut HashSet<ResultId>,
) -> Option<FPEncoding> {
    if !seen.insert(ty) {
        return None;
    }
    let inst = definitions.get(&ty)?;
    match inst.class.opcode {
        Op::TypeFloat => inst.operands.iter().find_map(|op| match op {
            rspirv::dr::Operand::FPEncoding(encoding)
                if matches!(
                    encoding,
                    FPEncoding::Float8E4M3EXT | FPEncoding::Float8E5M2EXT | FPEncoding::BFloat16KHR
                ) =>
            {
                Some(*encoding)
            }
            _ => None,
        }),
        Op::TypePointer => inst.operands.get(1).and_then(|op| {
            if let rspirv::dr::Operand::IdRef(pointee) = op {
                ResultId::try_from(*pointee)
                    .ok()
                    .and_then(|id| contains_disallowed_fp_encoding(definitions, id, seen))
            } else {
                None
            }
        }),
        Op::TypeVector | Op::TypeMatrix | Op::TypeArray | Op::TypeRuntimeArray => {
            inst.operands.first().and_then(|op| {
                if let rspirv::dr::Operand::IdRef(element) = op {
                    ResultId::try_from(*element)
                        .ok()
                        .and_then(|id| contains_disallowed_fp_encoding(definitions, id, seen))
                } else {
                    None
                }
            })
        }
        Op::TypeStruct => inst.operands.iter().find_map(|op| {
            if let rspirv::dr::Operand::IdRef(member) = op {
                ResultId::try_from(*member)
                    .ok()
                    .and_then(|id| contains_disallowed_fp_encoding(definitions, id, seen))
            } else {
                None
            }
        }),
        _ => None,
    }
}

// ============================================================================
// All entry point rules
// ============================================================================

/// Returns all entry point validation rules.
pub fn all_entry_point_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&EntryPointInterfaceStorageClassesRule]
}
