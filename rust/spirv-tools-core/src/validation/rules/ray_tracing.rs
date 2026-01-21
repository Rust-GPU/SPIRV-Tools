//! Ray tracing and ray query instruction validation rules.
//!
//! This module validates SPIR-V ray tracing instructions from SPV_KHR_ray_tracing,
//! SPV_NV_ray_tracing_motion_blur, and ray query instructions from SPV_KHR_ray_query:
//!
//! Ray Tracing:
//! - `OpTraceRayKHR`
//! - `OpTraceRayMotionNV` (motion blur extension)
//! - `OpReportIntersectionKHR`
//! - `OpExecuteCallableKHR`
//!
//! Ray Query:
//! - `OpRayQueryInitializeKHR`
//! - `OpRayQueryTerminateKHR`
//! - `OpRayQueryConfirmIntersectionKHR`
//! - `OpRayQueryGenerateIntersectionKHR`
//! - `OpRayQueryProceedKHR`
//! - Various `OpRayQueryGet*KHR` instructions

use rspirv::dr::Operand;
use rspirv::spirv::{Op, StorageClass};

use crate::validation::context::ValidationContext;
use crate::validation::error::ValidationError;
use crate::validation::helpers::{get_type_structure, id_ref, is_constant_opcode};
use crate::validation::types::{Id, ResultId, ScalarKind, TypeId, TypeStructure, VectorSize};
use crate::validation::ValidationResult;

use super::super::context::ValidationRule;

/// Helper to check if a type is a 32-bit int scalar.
fn is_int32_scalar(ty: &TypeStructure) -> bool {
    match ty {
        TypeStructure::Scalar(ScalarKind::SignedInt(w) | ScalarKind::UnsignedInt(w)) => {
            w.get() == 32
        }
        _ => false,
    }
}

/// Helper to check if a type is a 32-bit unsigned int scalar.
fn is_uint32_scalar(ty: &TypeStructure) -> bool {
    match ty {
        TypeStructure::Scalar(ScalarKind::UnsignedInt(w)) => w.get() == 32,
        _ => false,
    }
}

/// Helper to check if a type is a 32-bit float scalar.
fn is_float32_scalar(ty: &TypeStructure) -> bool {
    match ty {
        TypeStructure::Scalar(ScalarKind::Float(w)) => w.get() == 32,
        _ => false,
    }
}

/// Helper to check if a type is a 32-bit float 3-component vector.
fn is_float32_vec3(ty: &TypeStructure) -> bool {
    match ty {
        TypeStructure::Vector { component: ScalarKind::Float(w), size } => {
            w.get() == 32 && *size == VectorSize::VEC3
        }
        _ => false,
    }
}

/// Helper to check if a type is a 32-bit float 2-component vector.
fn is_float32_vec2(ty: &TypeStructure) -> bool {
    match ty {
        TypeStructure::Vector { component: ScalarKind::Float(w), size } => {
            w.get() == 32 && *size == VectorSize::VEC2
        }
        _ => false,
    }
}

/// Helper to get operand type structure.
fn get_operand_type(
    inst: &rspirv::dr::Instruction,
    operand_idx: usize,
    ctx: &ValidationContext<'_>,
) -> Option<TypeStructure> {
    let operand_id = inst.operands.get(operand_idx).and_then(id_ref)?;
    let operand_result_id = ResultId::try_from(operand_id).ok()?;
    let operand_inst = ctx.definitions.get(&operand_result_id)?;
    let operand_type_id = TypeId::try_from(operand_inst.result_type?).ok()?;
    Some(get_type_structure(operand_type_id, ctx.definitions))
}

/// Helper to get the opcode of a type.
fn get_operand_type_opcode(
    inst: &rspirv::dr::Instruction,
    operand_idx: usize,
    ctx: &ValidationContext<'_>,
) -> Option<Op> {
    let operand_id = inst.operands.get(operand_idx).and_then(id_ref)?;
    let operand_result_id = ResultId::try_from(operand_id).ok()?;
    let operand_inst = ctx.definitions.get(&operand_result_id)?;
    let type_id = operand_inst.result_type?;
    let type_result_id = ResultId::try_from(type_id).ok()?;
    let type_inst = ctx.definitions.get(&type_result_id)?;
    Some(type_inst.class.opcode)
}

/// Validates OpTraceRayKHR.
pub struct TraceRayRule;

impl ValidationRule for TraceRayRule {
    fn name(&self) -> &'static str {
        "trace-ray"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::TraceRayKHR {
                        continue;
                    }

                    // Acceleration Structure (operand 0) must be OpTypeAccelerationStructureKHR
                    if let Some(accel_type_op) = get_operand_type_opcode(inst, 0, ctx) {
                        if accel_type_op != Op::TypeAccelerationStructureKHR {
                            return Err(ValidationError::RayTracingExpectedAccelerationStructure {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                            }.into());
                        }
                    }

                    // Ray Flags (operand 1) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 1, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayFlags {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                            }.into());
                        }
                    }

                    // Cull Mask (operand 2) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 2, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidCullMask {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                            }.into());
                        }
                    }

                    // SBT Offset (operand 3) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 3, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidSbtParam {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                                param_name: "SBT Offset",
                            }.into());
                        }
                    }

                    // SBT Stride (operand 4) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 4, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidSbtParam {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                                param_name: "SBT Stride",
                            }.into());
                        }
                    }

                    // Miss Index (operand 5) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 5, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidSbtParam {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                                param_name: "Miss Index",
                            }.into());
                        }
                    }

                    // Ray Origin (operand 6) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 6, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayOrigin {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                            }.into());
                        }
                    }

                    // Ray TMin (operand 7) must be 32-bit float scalar
                    if let Some(ty) = get_operand_type(inst, 7, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                                param_name: "TMin",
                            }.into());
                        }
                    }

                    // Ray Direction (operand 8) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 8, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayDirection {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                            }.into());
                        }
                    }

                    // Ray TMax (operand 9) must be 32-bit float scalar
                    if let Some(ty) = get_operand_type(inst, 9, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayKHR,
                                param_name: "TMax",
                            }.into());
                        }
                    }

                    // Payload (operand 10) must be a variable with RayPayloadKHR or IncomingRayPayloadKHR
                    if let Some(payload_id) = inst.operands.get(10).and_then(id_ref) {
                        if let Ok(payload_result) = ResultId::try_from(payload_id) {
                            if let Some(payload_inst) = ctx.definitions.get(&payload_result) {
                                if payload_inst.class.opcode != Op::Variable {
                                    return Err(ValidationError::RayTracingInvalidPayload {
                                        function: func_id,
                                        block: block_id,
                                        opcode: Op::TraceRayKHR,
                                    }.into());
                                }

                                // Check storage class
                                if let Some(Operand::StorageClass(sc)) =
                                    payload_inst.operands.first()
                                {
                                    if *sc != StorageClass::RayPayloadKHR
                                        && *sc != StorageClass::IncomingRayPayloadKHR
                                    {
                                        return Err(ValidationError::RayTracingInvalidPayload {
                                            function: func_id,
                                            block: block_id,
                                            opcode: Op::TraceRayKHR,
                                        }.into());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpTraceRayMotionNV.
///
/// This is similar to OpTraceRayKHR but with an additional "current time" parameter
/// at operand 10, which shifts the payload to operand 11.
pub struct TraceRayMotionRule;

impl ValidationRule for TraceRayMotionRule {
    fn name(&self) -> &'static str {
        "trace-ray-motion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::TraceRayMotionNV {
                        continue;
                    }

                    // Acceleration Structure (operand 0) must be OpTypeAccelerationStructureKHR
                    if let Some(accel_type_op) = get_operand_type_opcode(inst, 0, ctx) {
                        if accel_type_op != Op::TypeAccelerationStructureKHR {
                            return Err(ValidationError::RayTracingExpectedAccelerationStructure {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                            }.into());
                        }
                    }

                    // Ray Flags (operand 1) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 1, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayFlags {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                            }.into());
                        }
                    }

                    // Cull Mask (operand 2) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 2, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidCullMask {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                            }.into());
                        }
                    }

                    // SBT Offset (operand 3) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 3, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidSbtParam {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                                param_name: "SBT Offset",
                            }.into());
                        }
                    }

                    // SBT Stride (operand 4) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 4, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidSbtParam {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                                param_name: "SBT Stride",
                            }.into());
                        }
                    }

                    // Miss Index (operand 5) must be 32-bit int scalar
                    if let Some(ty) = get_operand_type(inst, 5, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidSbtParam {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                                param_name: "Miss Index",
                            }.into());
                        }
                    }

                    // Ray Origin (operand 6) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 6, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayOrigin {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                            }.into());
                        }
                    }

                    // Ray TMin (operand 7) must be 32-bit float scalar
                    if let Some(ty) = get_operand_type(inst, 7, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                                param_name: "TMin",
                            }.into());
                        }
                    }

                    // Ray Direction (operand 8) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 8, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayDirection {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                            }.into());
                        }
                    }

                    // Ray TMax (operand 9) must be 32-bit float scalar
                    if let Some(ty) = get_operand_type(inst, 9, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                                param_name: "TMax",
                            }.into());
                        }
                    }

                    // Current Time (operand 10) must be 32-bit float scalar
                    if let Some(ty) = get_operand_type(inst, 10, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidCurrentTime {
                                function: func_id,
                                block: block_id,
                                opcode: Op::TraceRayMotionNV,
                            }.into());
                        }
                    }

                    // Payload (operand 11) must be a variable with RayPayloadKHR or IncomingRayPayloadKHR
                    if let Some(payload_id) = inst.operands.get(11).and_then(id_ref) {
                        if let Ok(payload_result) = ResultId::try_from(payload_id) {
                            if let Some(payload_inst) = ctx.definitions.get(&payload_result) {
                                if payload_inst.class.opcode != Op::Variable {
                                    return Err(ValidationError::RayTracingInvalidPayload {
                                        function: func_id,
                                        block: block_id,
                                        opcode: Op::TraceRayMotionNV,
                                    }.into());
                                }

                                // Check storage class
                                if let Some(Operand::StorageClass(sc)) =
                                    payload_inst.operands.first()
                                {
                                    if *sc != StorageClass::RayPayloadKHR
                                        && *sc != StorageClass::IncomingRayPayloadKHR
                                    {
                                        return Err(ValidationError::RayTracingInvalidPayload {
                                            function: func_id,
                                            block: block_id,
                                            opcode: Op::TraceRayMotionNV,
                                        }.into());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpReportIntersectionKHR.
pub struct ReportIntersectionRule;

impl ValidationRule for ReportIntersectionRule {
    fn name(&self) -> &'static str {
        "report-intersection"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::ReportIntersectionKHR {
                        continue;
                    }

                    // Result must be bool scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_bool_scalar() {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::ReportIntersectionKHR,
                                    expected: "bool scalar type",
                                }.into());
                            }
                        }
                    }

                    // Hit (operand 2) must be 32-bit float scalar
                    if let Some(ty) = get_operand_type(inst, 2, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidHit {
                                function: func_id,
                                block: block_id,
                                opcode: Op::ReportIntersectionKHR,
                            }.into());
                        }
                    }

                    // Hit Kind (operand 3) must be 32-bit unsigned int scalar
                    if let Some(ty) = get_operand_type(inst, 3, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidHitKind {
                                function: func_id,
                                block: block_id,
                                opcode: Op::ReportIntersectionKHR,
                            }.into());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpExecuteCallableKHR.
pub struct ExecuteCallableRule;

impl ValidationRule for ExecuteCallableRule {
    fn name(&self) -> &'static str {
        "execute-callable"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::ExecuteCallableKHR {
                        continue;
                    }

                    // SBT Index (operand 0) must be 32-bit unsigned int scalar
                    if let Some(ty) = get_operand_type(inst, 0, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidSbtParam {
                                function: func_id,
                                block: block_id,
                                opcode: Op::ExecuteCallableKHR,
                                param_name: "SBT Index",
                            }.into());
                        }
                    }

                    // Callable Data (operand 1) must be a variable with CallableDataKHR or IncomingCallableDataKHR
                    if let Some(data_id) = inst.operands.get(1).and_then(id_ref) {
                        if let Ok(data_result) = ResultId::try_from(data_id) {
                            if let Some(data_inst) = ctx.definitions.get(&data_result) {
                                if data_inst.class.opcode != Op::Variable {
                                    return Err(ValidationError::RayTracingInvalidCallableData {
                                        function: func_id,
                                        block: block_id,
                                        opcode: Op::ExecuteCallableKHR,
                                    }.into());
                                }

                                // Check storage class
                                if let Some(Operand::StorageClass(sc)) = data_inst.operands.first()
                                {
                                    if *sc != StorageClass::CallableDataKHR
                                        && *sc != StorageClass::IncomingCallableDataKHR
                                    {
                                        return Err(ValidationError::RayTracingInvalidCallableData {
                                            function: func_id,
                                            block: block_id,
                                            opcode: Op::ExecuteCallableKHR,
                                        }.into());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Helper to validate ray query pointer operand.
fn validate_ray_query_pointer(
    inst: &rspirv::dr::Instruction,
    operand_idx: usize,
    ctx: &ValidationContext<'_>,
    func_id: Option<Id>,
    block_id: Option<Id>,
) -> ValidationResult {
    let query_id = inst.operands.get(operand_idx).and_then(id_ref);
    let query_id = match query_id {
        Some(id) => id,
        None => return Ok(()), // Missing operand - other validation will catch this
    };

    let query_result = ResultId::try_from(query_id).ok();
    let query_inst = query_result.and_then(|id| ctx.definitions.get(&id));

    if let Some(query_inst) = query_inst {
        // Must be a variable, function parameter, or access chain
        if !matches!(
            query_inst.class.opcode,
            Op::Variable | Op::FunctionParameter | Op::AccessChain
        ) {
            return Err(ValidationError::RayQueryInvalidPointer {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }.into());
        }

        // Get pointer type
        if let Some(ptr_type_id) = query_inst.result_type {
            if let Ok(type_result_id) = ResultId::try_from(ptr_type_id) {
                if let Some(ptr_type_inst) = ctx.definitions.get(&type_result_id) {
                    if ptr_type_inst.class.opcode != Op::TypePointer {
                        return Err(ValidationError::RayQueryInvalidPointer {
                            function: func_id,
                            block: block_id,
                            opcode: inst.class.opcode,
                        }.into());
                    }

                    // Get pointee type
                    if let Some(pointee_id) = ptr_type_inst.operands.get(1).and_then(id_ref) {
                        if let Ok(pointee_result_id) = ResultId::try_from(pointee_id) {
                            if let Some(pointee_inst) = ctx.definitions.get(&pointee_result_id) {
                                if pointee_inst.class.opcode != Op::TypeRayQueryKHR {
                                    return Err(ValidationError::RayQueryInvalidPointer {
                                        function: func_id,
                                        block: block_id,
                                        opcode: inst.class.opcode,
                                    }.into());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Helper to validate intersection ID operand.
fn validate_intersection_id(
    inst: &rspirv::dr::Instruction,
    operand_idx: usize,
    ctx: &ValidationContext<'_>,
    func_id: Option<Id>,
    block_id: Option<Id>,
) -> ValidationResult {
    let intersect_id = match inst.operands.get(operand_idx).and_then(id_ref) {
        Some(id) => id,
        None => return Ok(()),
    };

    let intersect_result = ResultId::try_from(intersect_id).ok();
    let intersect_inst = intersect_result.and_then(|id| ctx.definitions.get(&id));

    if let Some(intersect_inst) = intersect_inst {
        // Must be a constant
        if !is_constant_opcode(intersect_inst.class.opcode) {
            return Err(ValidationError::RayQueryInvalidIntersectionId {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }.into());
        }

        // Must be 32-bit int
        if let Some(type_id) = intersect_inst.result_type {
            if let Ok(tid) = TypeId::try_from(type_id) {
                let ty = get_type_structure(tid, ctx.definitions);
                if !is_int32_scalar(&ty) {
                    return Err(ValidationError::RayQueryInvalidIntersectionId {
                        function: func_id,
                        block: block_id,
                        opcode: inst.class.opcode,
                    }.into());
                }
            }
        }
    }

    Ok(())
}

/// Validates OpRayQueryInitializeKHR.
pub struct RayQueryInitializeRule;

impl ValidationRule for RayQueryInitializeRule {
    fn name(&self) -> &'static str {
        "ray-query-initialize"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::RayQueryInitializeKHR {
                        continue;
                    }

                    // Ray Query pointer (operand 0)
                    validate_ray_query_pointer(inst, 0, ctx, func_id, block_id)?;

                    // Acceleration Structure (operand 1)
                    if let Some(accel_type_op) = get_operand_type_opcode(inst, 1, ctx) {
                        if accel_type_op != Op::TypeAccelerationStructureKHR {
                            return Err(ValidationError::RayTracingExpectedAccelerationStructure {
                                function: func_id,
                                block: block_id,
                                opcode: Op::RayQueryInitializeKHR,
                            }.into());
                        }
                    }

                    // Ray Flags (operand 2)
                    if let Some(ty) = get_operand_type(inst, 2, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayFlags {
                                function: func_id,
                                block: block_id,
                                opcode: Op::RayQueryInitializeKHR,
                            }.into());
                        }
                    }

                    // Cull Mask (operand 3)
                    if let Some(ty) = get_operand_type(inst, 3, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidCullMask {
                                function: func_id,
                                block: block_id,
                                opcode: Op::RayQueryInitializeKHR,
                            }.into());
                        }
                    }

                    // Ray Origin (operand 4)
                    if let Some(ty) = get_operand_type(inst, 4, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayOrigin {
                                function: func_id,
                                block: block_id,
                                opcode: Op::RayQueryInitializeKHR,
                            }.into());
                        }
                    }

                    // Ray TMin (operand 5)
                    if let Some(ty) = get_operand_type(inst, 5, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: Op::RayQueryInitializeKHR,
                                param_name: "TMin",
                            }.into());
                        }
                    }

                    // Ray Direction (operand 6)
                    if let Some(ty) = get_operand_type(inst, 6, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayDirection {
                                function: func_id,
                                block: block_id,
                                opcode: Op::RayQueryInitializeKHR,
                            }.into());
                        }
                    }

                    // Ray TMax (operand 7)
                    if let Some(ty) = get_operand_type(inst, 7, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: Op::RayQueryInitializeKHR,
                                param_name: "TMax",
                            }.into());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates ray query instructions that only need ray query pointer validation.
pub struct RayQueryPointerOnlyRule;

impl ValidationRule for RayQueryPointerOnlyRule {
    fn name(&self) -> &'static str {
        "ray-query-pointer-only"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // These instructions only need ray query pointer validation
                    if !matches!(
                        opcode,
                        Op::RayQueryTerminateKHR | Op::RayQueryConfirmIntersectionKHR
                    ) {
                        continue;
                    }

                    validate_ray_query_pointer(inst, 0, ctx, func_id, block_id)?;
                }
            }
        }
        Ok(())
    }
}

/// Validates ray query instructions that return bool.
pub struct RayQueryBoolResultRule;

impl ValidationRule for RayQueryBoolResultRule {
    fn name(&self) -> &'static str {
        "ray-query-bool-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::RayQueryProceedKHR
                            | Op::RayQueryGetIntersectionFrontFaceKHR
                            | Op::RayQueryGetIntersectionCandidateAABBOpaqueKHR
                    ) {
                        continue;
                    }

                    // Ray query pointer at operand 2
                    validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be bool scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_bool_scalar() {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "bool scalar type",
                                }.into());
                            }
                        }
                    }

                    // FrontFace has intersection ID at operand 3
                    if opcode == Op::RayQueryGetIntersectionFrontFaceKHR {
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates ray query instructions that return 32-bit float scalar.
pub struct RayQueryFloatResultRule;

impl ValidationRule for RayQueryFloatResultRule {
    fn name(&self) -> &'static str {
        "ray-query-float-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::RayQueryGetIntersectionTKHR | Op::RayQueryGetRayTMinKHR
                    ) {
                        continue;
                    }

                    validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be 32-bit float scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_float32_scalar(&ty) {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "32-bit float scalar type",
                                }.into());
                            }
                        }
                    }

                    // IntersectionT has intersection ID at operand 3
                    if opcode == Op::RayQueryGetIntersectionTKHR {
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates ray query instructions that return 32-bit int scalar.
pub struct RayQueryIntResultRule;

impl ValidationRule for RayQueryIntResultRule {
    fn name(&self) -> &'static str {
        "ray-query-int-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::RayQueryGetIntersectionTypeKHR
                            | Op::RayQueryGetIntersectionInstanceCustomIndexKHR
                            | Op::RayQueryGetIntersectionInstanceIdKHR
                            | Op::RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR
                            | Op::RayQueryGetIntersectionGeometryIndexKHR
                            | Op::RayQueryGetIntersectionPrimitiveIndexKHR
                            | Op::RayQueryGetRayFlagsKHR
                    ) {
                        continue;
                    }

                    validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be 32-bit int scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_int32_scalar(&ty) {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "32-bit int scalar type",
                                }.into());
                            }
                        }
                    }

                    // Most of these have intersection ID at operand 3 (except RayFlags)
                    if opcode != Op::RayQueryGetRayFlagsKHR {
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates ray query instructions that return vec3.
pub struct RayQueryVec3ResultRule;

impl ValidationRule for RayQueryVec3ResultRule {
    fn name(&self) -> &'static str {
        "ray-query-vec3-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::RayQueryGetIntersectionObjectRayDirectionKHR
                            | Op::RayQueryGetIntersectionObjectRayOriginKHR
                            | Op::RayQueryGetWorldRayDirectionKHR
                            | Op::RayQueryGetWorldRayOriginKHR
                    ) {
                        continue;
                    }

                    validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be 32-bit float vec3
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_float32_vec3(&ty) {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "32-bit float 3-component vector type",
                                }.into());
                            }
                        }
                    }

                    // Object ray direction/origin have intersection ID at operand 3
                    if opcode == Op::RayQueryGetIntersectionObjectRayDirectionKHR
                        || opcode == Op::RayQueryGetIntersectionObjectRayOriginKHR
                    {
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpRayQueryGetIntersectionBarycentricsKHR.
pub struct RayQueryBarycentricsRule;

impl ValidationRule for RayQueryBarycentricsRule {
    fn name(&self) -> &'static str {
        "ray-query-barycentrics"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::RayQueryGetIntersectionBarycentricsKHR {
                        continue;
                    }

                    validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;
                    validate_intersection_id(inst, 3, ctx, func_id, block_id)?;

                    // Result must be 32-bit float vec2
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_float32_vec2(&ty) {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::RayQueryGetIntersectionBarycentricsKHR,
                                    expected: "32-bit float 2-component vector type",
                                }.into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpRayQueryGenerateIntersectionKHR.
pub struct RayQueryGenerateIntersectionRule;

impl ValidationRule for RayQueryGenerateIntersectionRule {
    fn name(&self) -> &'static str {
        "ray-query-generate-intersection"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::RayQueryGenerateIntersectionKHR {
                        continue;
                    }

                    validate_ray_query_pointer(inst, 0, ctx, func_id, block_id)?;

                    // Hit T (operand 1) must be 32-bit float scalar
                    if let Some(ty) = get_operand_type(inst, 1, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::RayTracingInvalidHit {
                                function: func_id,
                                block: block_id,
                                opcode: Op::RayQueryGenerateIntersectionKHR,
                            }.into());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates ray query matrix result instructions (Object/World transformations).
pub struct RayQueryMatrixResultRule;

impl ValidationRule for RayQueryMatrixResultRule {
    fn name(&self) -> &'static str {
        "ray-query-matrix-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::RayQueryGetIntersectionObjectToWorldKHR
                            | Op::RayQueryGetIntersectionWorldToObjectKHR
                    ) {
                        continue;
                    }

                    validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;
                    validate_intersection_id(inst, 3, ctx, func_id, block_id)?;

                    // Result must be a 3x4 matrix of 32-bit floats
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            let is_valid_matrix = match ty {
                                TypeStructure::Matrix {
                                    component: ScalarKind::Float(w),
                                    rows,
                                    cols,
                                } => w.get() == 32 && rows == VectorSize::VEC3 && cols.get() == 4,
                                _ => false,
                            };
                            if !is_valid_matrix {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "matrix of 4 columns of 32-bit float 3-component vectors",
                                }.into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates NVIDIA ray query cluster ID instruction.
pub struct RayQueryClusterIdNVRule;

impl ValidationRule for RayQueryClusterIdNVRule {
    fn name(&self) -> &'static str {
        "ray-query-cluster-id-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::RayQueryGetClusterIdNV {
                        continue;
                    }

                    validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;
                    validate_intersection_id(inst, 3, ctx, func_id, block_id)?;

                    // Result must be 32-bit int scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_int32_scalar(&ty) {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::RayQueryGetClusterIdNV,
                                    expected: "32-bit int scalar type",
                                }.into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates NVIDIA sphere and LSS ray query instructions.
pub struct RayQuerySphereAndLSSNVRule;

impl ValidationRule for RayQuerySphereAndLSSNVRule {
    fn name(&self) -> &'static str {
        "ray-query-sphere-lss-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Handle sphere position (vec3 result)
                    if opcode == Op::RayQueryGetIntersectionSpherePositionNV {
                        validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;

                        if let Some(result_type_id) = inst.result_type {
                            if let Ok(type_id) = TypeId::try_from(result_type_id) {
                                let ty = get_type_structure(type_id, ctx.definitions);
                                if !is_float32_vec3(&ty) {
                                    return Err(ValidationError::RayQueryInvalidResultType {
                                        function: func_id,
                                        block: block_id,
                                        opcode,
                                        expected: "32-bit float 3-component vector type",
                                    }.into());
                                }
                            }
                        }
                    }

                    // Handle sphere radius and LSS hit value (float scalar result)
                    if matches!(
                        opcode,
                        Op::RayQueryGetIntersectionSphereRadiusNV
                            | Op::RayQueryGetIntersectionLSSHitValueNV
                    ) {
                        validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;

                        if let Some(result_type_id) = inst.result_type {
                            if let Ok(type_id) = TypeId::try_from(result_type_id) {
                                let ty = get_type_structure(type_id, ctx.definitions);
                                if !is_float32_scalar(&ty) {
                                    return Err(ValidationError::RayQueryInvalidResultType {
                                        function: func_id,
                                        block: block_id,
                                        opcode,
                                        expected: "32-bit float scalar type",
                                    }.into());
                                }
                            }
                        }
                    }

                    // Handle sphere/LSS hit tests (bool result)
                    if matches!(opcode, Op::RayQueryIsSphereHitNV | Op::RayQueryIsLSSHitNV) {
                        validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;

                        if let Some(result_type_id) = inst.result_type {
                            if let Ok(type_id) = TypeId::try_from(result_type_id) {
                                let ty = get_type_structure(type_id, ctx.definitions);
                                if !ty.is_bool_scalar() {
                                    return Err(ValidationError::RayQueryInvalidResultType {
                                        function: func_id,
                                        block: block_id,
                                        opcode,
                                        expected: "bool scalar type",
                                    }.into());
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates NVIDIA LSS array result ray query instructions.
///
/// - `OpRayQueryGetIntersectionLSSPositionsNV` - 2-element array of vec3
/// - `OpRayQueryGetIntersectionLSSRadiiNV` - 2-element array of float
pub struct RayQueryLSSArrayNVRule;

impl RayQueryLSSArrayNVRule {
    /// Helper to get array length from array type instruction.
    fn get_array_length(
        array_type_id: u32,
        definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
    ) -> Option<u32> {
        let type_result_id = ResultId::try_from(array_type_id).ok()?;
        let array_type_inst = definitions.get(&type_result_id)?;

        if array_type_inst.class.opcode != Op::TypeArray {
            return None;
        }

        // Length is the second operand (index 1)
        let length_id = array_type_inst.operands.get(1).and_then(id_ref)?;
        let length_result_id = ResultId::try_from(length_id).ok()?;
        let length_inst = definitions.get(&length_result_id)?;

        if length_inst.class.opcode != Op::Constant {
            return None;
        }

        match length_inst.operands.first() {
            Some(Operand::LiteralBit32(val)) => Some(*val),
            _ => None,
        }
    }

    /// Helper to get array element type from array type instruction.
    fn get_array_element_type(
        array_type_id: u32,
        definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
    ) -> Option<TypeId> {
        let type_result_id = ResultId::try_from(array_type_id).ok()?;
        let array_type_inst = definitions.get(&type_result_id)?;

        if array_type_inst.class.opcode != Op::TypeArray {
            return None;
        }

        // Element type is the first operand (index 0)
        let element_type_id = array_type_inst.operands.first().and_then(id_ref)?;
        TypeId::try_from(element_type_id).ok()
    }
}

impl ValidationRule for RayQueryLSSArrayNVRule {
    fn name(&self) -> &'static str {
        "ray-query-lss-array-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Handle LSS Positions (2-element array of vec3)
                    if opcode == Op::RayQueryGetIntersectionLSSPositionsNV {
                        validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;

                        if let Some(result_type_id) = inst.result_type {
                            // Check it's a 2-element array
                            let length = Self::get_array_length(result_type_id, ctx.definitions);
                            if length != Some(2) {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "2-element array of 32-bit float 3-component vectors",
                                }.into());
                            }

                            // Check element type is vec3 of float32
                            if let Some(element_type_id) =
                                Self::get_array_element_type(result_type_id, ctx.definitions)
                            {
                                let element_ty =
                                    get_type_structure(element_type_id, ctx.definitions);
                                if !is_float32_vec3(&element_ty) {
                                    return Err(ValidationError::RayQueryInvalidResultType {
                                        function: func_id,
                                        block: block_id,
                                        opcode,
                                        expected:
                                            "2-element array of 32-bit float 3-component vectors",
                                    }.into());
                                }
                            }
                        }
                    }

                    // Handle LSS Radii (2-element array of float)
                    if opcode == Op::RayQueryGetIntersectionLSSRadiiNV {
                        validate_ray_query_pointer(inst, 2, ctx, func_id, block_id)?;
                        validate_intersection_id(inst, 3, ctx, func_id, block_id)?;

                        if let Some(result_type_id) = inst.result_type {
                            // Check it's a 2-element array
                            let length = Self::get_array_length(result_type_id, ctx.definitions);
                            if length != Some(2) {
                                return Err(ValidationError::RayQueryInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "2-element array of 32-bit float scalars",
                                }.into());
                            }

                            // Check element type is float32 scalar
                            if let Some(element_type_id) =
                                Self::get_array_element_type(result_type_id, ctx.definitions)
                            {
                                let element_ty =
                                    get_type_structure(element_type_id, ctx.definitions);
                                if !is_float32_scalar(&element_ty) {
                                    return Err(ValidationError::RayQueryInvalidResultType {
                                        function: func_id,
                                        block: block_id,
                                        opcode,
                                        expected: "2-element array of 32-bit float scalars",
                                    }.into());
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Returns all ray tracing validation rules.
pub fn all_ray_tracing_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(TraceRayRule),
        Box::new(TraceRayMotionRule),
        Box::new(ReportIntersectionRule),
        Box::new(ExecuteCallableRule),
        Box::new(RayQueryInitializeRule),
        Box::new(RayQueryPointerOnlyRule),
        Box::new(RayQueryBoolResultRule),
        Box::new(RayQueryFloatResultRule),
        Box::new(RayQueryIntResultRule),
        Box::new(RayQueryVec3ResultRule),
        Box::new(RayQueryBarycentricsRule),
        Box::new(RayQueryGenerateIntersectionRule),
        Box::new(RayQueryMatrixResultRule),
        Box::new(RayQueryClusterIdNVRule),
        Box::new(RayQuerySphereAndLSSNVRule),
        Box::new(RayQueryLSSArrayNVRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::types::{BitWidth, ScalarKind, VectorSize};

    #[test]
    fn test_is_int32_scalar() {
        let signed = TypeStructure::Scalar(ScalarKind::SignedInt(BitWidth::BITS_32));
        assert!(is_int32_scalar(&signed));

        let unsigned = TypeStructure::Scalar(ScalarKind::UnsignedInt(BitWidth::BITS_32));
        assert!(is_int32_scalar(&unsigned));

        let int64 = TypeStructure::Scalar(ScalarKind::SignedInt(BitWidth::BITS_64));
        assert!(!is_int32_scalar(&int64));

        let float = TypeStructure::Scalar(ScalarKind::Float(BitWidth::BITS_32));
        assert!(!is_int32_scalar(&float));
    }

    #[test]
    fn test_is_uint32_scalar() {
        let unsigned = TypeStructure::Scalar(ScalarKind::UnsignedInt(BitWidth::BITS_32));
        assert!(is_uint32_scalar(&unsigned));

        let signed = TypeStructure::Scalar(ScalarKind::SignedInt(BitWidth::BITS_32));
        assert!(!is_uint32_scalar(&signed));
    }

    #[test]
    fn test_is_float32_scalar() {
        let f32_type = TypeStructure::Scalar(ScalarKind::Float(BitWidth::BITS_32));
        assert!(is_float32_scalar(&f32_type));

        let f64_type = TypeStructure::Scalar(ScalarKind::Float(BitWidth::BITS_64));
        assert!(!is_float32_scalar(&f64_type));

        let int_type = TypeStructure::Scalar(ScalarKind::SignedInt(BitWidth::BITS_32));
        assert!(!is_float32_scalar(&int_type));
    }

    #[test]
    fn test_is_float32_vec3() {
        let vec3 = TypeStructure::Vector {
            component: ScalarKind::Float(BitWidth::BITS_32),
            size: VectorSize::VEC3,
        };
        assert!(is_float32_vec3(&vec3));

        let vec4 = TypeStructure::Vector {
            component: ScalarKind::Float(BitWidth::BITS_32),
            size: VectorSize::VEC4,
        };
        assert!(!is_float32_vec3(&vec4));

        let int_vec3 = TypeStructure::Vector {
            component: ScalarKind::SignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC3,
        };
        assert!(!is_float32_vec3(&int_vec3));
    }

    #[test]
    fn test_is_float32_vec2() {
        let vec2 = TypeStructure::Vector {
            component: ScalarKind::Float(BitWidth::BITS_32),
            size: VectorSize::VEC2,
        };
        assert!(is_float32_vec2(&vec2));

        let vec3 = TypeStructure::Vector {
            component: ScalarKind::Float(BitWidth::BITS_32),
            size: VectorSize::VEC3,
        };
        assert!(!is_float32_vec2(&vec3));
    }
}
