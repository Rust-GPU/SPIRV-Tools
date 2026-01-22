//! Hit object instruction validation rules (SPV_NV_shader_execution_reorder).
//!
//! This module validates SPIR-V hit object instructions from the
//! SPV_NV_shader_execution_reorder extension:
//!
//! - `OpHitObjectIsMissNV`, `OpHitObjectIsHitNV`, `OpHitObjectIsEmptyNV`
//! - `OpHitObjectGet*NV` instructions for querying hit object properties
//! - `OpHitObjectRecordHitNV`, `OpHitObjectRecordMissNV`, etc.
//! - `OpHitObjectTraceRayNV`, `OpHitObjectTraceRayMotionNV`
//! - `OpReorderThreadWithHitObjectNV`, `OpReorderThreadWithHintNV`

use rspirv::dr::Operand;
use rspirv::spirv::{Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::{get_type_structure, id_ref};
use crate::validation::types::{Id, ResultId, ScalarKind, TypeId, TypeStructure, VectorSize};
use crate::validation::ValidationResult;

fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Helper to check if a type is a 32-bit int scalar (signed or unsigned).
fn is_int32_scalar(ty: &TypeStructure) -> bool {
    matches!(
        ty,
        TypeStructure::Scalar(ScalarKind::SignedInt(w) | ScalarKind::UnsignedInt(w))
            if w.get() == 32
    )
}

/// Helper to check if a type is a 32-bit unsigned int scalar.
fn is_uint32_scalar(ty: &TypeStructure) -> bool {
    matches!(
        ty,
        TypeStructure::Scalar(ScalarKind::UnsignedInt(w)) if w.get() == 32
    )
}

/// Helper to check if a type is a 32-bit float scalar.
fn is_float32_scalar(ty: &TypeStructure) -> bool {
    matches!(
        ty,
        TypeStructure::Scalar(ScalarKind::Float(w)) if w.get() == 32
    )
}

/// Helper to check if a type is a 32-bit float vec3.
fn is_float32_vec3(ty: &TypeStructure) -> bool {
    matches!(
        ty,
        TypeStructure::Vector { component: ScalarKind::Float(w), size }
            if w.get() == 32 && *size == VectorSize::VEC3
    )
}

/// Helper to check if a type is a 32-bit int vec2.
fn is_int32_vec2(ty: &TypeStructure) -> bool {
    matches!(
        ty,
        TypeStructure::Vector { component: ScalarKind::SignedInt(w) | ScalarKind::UnsignedInt(w), size }
            if w.get() == 32 && *size == VectorSize::VEC2
    )
}

/// Helper to get the type structure of an operand.
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

/// Validate that an operand is a hit object pointer.
fn validate_hit_object_pointer(
    inst: &rspirv::dr::Instruction,
    operand_idx: usize,
    ctx: &ValidationContext<'_>,
    func_id: Option<Id>,
    block_id: Option<Id>,
) -> ValidationResult {
    let hit_object_id = match inst.operands.get(operand_idx).and_then(id_ref) {
        Some(id) => id,
        None => return Ok(()),
    };

    let variable = match ResultId::try_from(hit_object_id)
        .ok()
        .and_then(|id| ctx.definitions.get(&id))
    {
        Some(v) => v,
        None => {
            return Err(ValidationError::HitObjectNotMemoryObject {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }
            .into());
        }
    };

    // Must be OpVariable, OpFunctionParameter, or OpAccessChain
    if !matches!(
        variable.class.opcode,
        Op::Variable | Op::FunctionParameter | Op::AccessChain
    ) {
        return Err(ValidationError::HitObjectNotMemoryObject {
            function: func_id,
            block: block_id,
            opcode: inst.class.opcode,
        }
        .into());
    }

    // Get pointer type
    let ptr_type_id = match variable.result_type {
        Some(id) => id,
        None => {
            return Err(ValidationError::HitObjectNotPointer {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }
            .into());
        }
    };

    let ptr_type_inst = match ResultId::try_from(ptr_type_id)
        .ok()
        .and_then(|id| ctx.definitions.get(&id))
    {
        Some(inst) => inst,
        None => {
            return Err(ValidationError::HitObjectNotPointer {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }
            .into());
        }
    };

    if ptr_type_inst.class.opcode != Op::TypePointer {
        return Err(ValidationError::HitObjectNotPointer {
            function: func_id,
            block: block_id,
            opcode: inst.class.opcode,
        }
        .into());
    }

    // Get pointee type (second operand of OpTypePointer after storage class)
    let pointee_id = match ptr_type_inst.operands.get(1).and_then(id_ref) {
        Some(id) => id,
        None => {
            return Err(ValidationError::HitObjectInvalidType {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }
            .into());
        }
    };

    let pointee_inst = match ResultId::try_from(pointee_id)
        .ok()
        .and_then(|id| ctx.definitions.get(&id))
    {
        Some(inst) => inst,
        None => {
            return Err(ValidationError::HitObjectInvalidType {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }
            .into());
        }
    };

    if pointee_inst.class.opcode != Op::TypeHitObjectNV {
        return Err(ValidationError::HitObjectInvalidType {
            function: func_id,
            block: block_id,
            opcode: inst.class.opcode,
        }
        .into());
    }

    Ok(())
}

/// Validate hit object attribute operand (must be HitObjectAttributeNV storage class).
fn validate_hit_object_attribute(
    inst: &rspirv::dr::Instruction,
    operand_idx: usize,
    ctx: &ValidationContext<'_>,
    func_id: Option<Id>,
    block_id: Option<Id>,
) -> ValidationResult {
    let attr_id = match inst.operands.get(operand_idx).and_then(id_ref) {
        Some(id) => id,
        None => return Ok(()),
    };

    let variable = match ResultId::try_from(attr_id)
        .ok()
        .and_then(|id| ctx.definitions.get(&id))
    {
        Some(v) => v,
        None => {
            return Err(ValidationError::HitObjectAttributeInvalid {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }
            .into());
        }
    };

    if variable.class.opcode != Op::Variable {
        return Err(ValidationError::HitObjectAttributeInvalid {
            function: func_id,
            block: block_id,
            opcode: inst.class.opcode,
        }
        .into());
    }

    // Check storage class
    if let Some(Operand::StorageClass(sc)) = variable.operands.first() {
        if *sc != StorageClass::HitObjectAttributeNV {
            return Err(ValidationError::HitObjectAttributeInvalid {
                function: func_id,
                block: block_id,
                opcode: inst.class.opcode,
            }
            .into());
        }
    } else {
        return Err(ValidationError::HitObjectAttributeInvalid {
            function: func_id,
            block: block_id,
            opcode: inst.class.opcode,
        }
        .into());
    }

    Ok(())
}

/// Get array length from array type instruction.
fn get_array_length(array_type_id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let type_result_id = ResultId::try_from(array_type_id).ok()?;
    let array_type_inst = ctx.definitions.get(&type_result_id)?;

    if array_type_inst.class.opcode != Op::TypeArray {
        return None;
    }

    // Length is the second operand (index 1)
    let length_id = array_type_inst.operands.get(1).and_then(id_ref)?;
    let length_result_id = ResultId::try_from(length_id).ok()?;
    let length_inst = ctx.definitions.get(&length_result_id)?;

    if length_inst.class.opcode != Op::Constant {
        return None;
    }

    match length_inst.operands.first() {
        Some(Operand::LiteralBit32(val)) => Some(*val),
        _ => None,
    }
}

/// Validates hit object boolean result instructions.
///
/// OpHitObjectIsMissNV, OpHitObjectIsHitNV, OpHitObjectIsEmptyNV,
/// OpHitObjectIsSphereHitNV, OpHitObjectIsLSSHitNV
pub struct HitObjectBoolResultRule;

impl ValidationRule for HitObjectBoolResultRule {
    fn name(&self) -> &'static str {
        "hit-object-bool-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::HitObjectIsMissNV
                            | Op::HitObjectIsHitNV
                            | Op::HitObjectIsEmptyNV
                            | Op::HitObjectIsSphereHitNV
                            | Op::HitObjectIsLSSHitNV
                    ) {
                        continue;
                    }

                    // Hit object pointer at operand 2
                    validate_hit_object_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be bool scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_bool_scalar() {
                                return Err(ValidationError::HitObjectInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "bool scalar",
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates hit object int scalar result instructions.
///
/// OpHitObjectGetHitKindNV, OpHitObjectGetPrimitiveIndexNV,
/// OpHitObjectGetGeometryIndexNV, OpHitObjectGetInstanceIdNV,
/// OpHitObjectGetInstanceCustomIndexNV, OpHitObjectGetShaderBindingTableRecordIndexNV,
/// OpHitObjectGetClusterIdNV
pub struct HitObjectIntResultRule;

impl ValidationRule for HitObjectIntResultRule {
    fn name(&self) -> &'static str {
        "hit-object-int-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::HitObjectGetHitKindNV
                            | Op::HitObjectGetPrimitiveIndexNV
                            | Op::HitObjectGetGeometryIndexNV
                            | Op::HitObjectGetInstanceIdNV
                            | Op::HitObjectGetInstanceCustomIndexNV
                            | Op::HitObjectGetShaderBindingTableRecordIndexNV
                            | Op::HitObjectGetClusterIdNV
                    ) {
                        continue;
                    }

                    // Hit object pointer at operand 2
                    validate_hit_object_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be 32-bit int scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_int32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "32-bit int scalar",
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates hit object float scalar result instructions.
///
/// OpHitObjectGetCurrentTimeNV, OpHitObjectGetRayTMaxNV, OpHitObjectGetRayTMinNV,
/// OpHitObjectGetSphereRadiusNV
pub struct HitObjectFloatResultRule;

impl ValidationRule for HitObjectFloatResultRule {
    fn name(&self) -> &'static str {
        "hit-object-float-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::HitObjectGetCurrentTimeNV
                            | Op::HitObjectGetRayTMaxNV
                            | Op::HitObjectGetRayTMinNV
                            | Op::HitObjectGetSphereRadiusNV
                    ) {
                        continue;
                    }

                    // Hit object pointer at operand 2
                    validate_hit_object_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be 32-bit float scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_float32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "32-bit float scalar",
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates hit object vec3 result instructions.
///
/// OpHitObjectGetObjectRayOriginNV, OpHitObjectGetObjectRayDirectionNV,
/// OpHitObjectGetWorldRayDirectionNV, OpHitObjectGetWorldRayOriginNV,
/// OpHitObjectGetSpherePositionNV
pub struct HitObjectVec3ResultRule;

impl ValidationRule for HitObjectVec3ResultRule {
    fn name(&self) -> &'static str {
        "hit-object-vec3-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::HitObjectGetObjectRayOriginNV
                            | Op::HitObjectGetObjectRayDirectionNV
                            | Op::HitObjectGetWorldRayDirectionNV
                            | Op::HitObjectGetWorldRayOriginNV
                            | Op::HitObjectGetSpherePositionNV
                    ) {
                        continue;
                    }

                    // Hit object pointer at operand 2
                    validate_hit_object_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be 32-bit float vec3
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_float32_vec3(&ty) {
                                return Err(ValidationError::HitObjectInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "32-bit float 3-component vector",
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates hit object matrix result instructions.
///
/// OpHitObjectGetObjectToWorldNV, OpHitObjectGetWorldToObjectNV
pub struct HitObjectMatrixResultRule;

impl ValidationRule for HitObjectMatrixResultRule {
    fn name(&self) -> &'static str {
        "hit-object-matrix-result"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !matches!(
                        opcode,
                        Op::HitObjectGetObjectToWorldNV | Op::HitObjectGetWorldToObjectNV
                    ) {
                        continue;
                    }

                    // Hit object pointer at operand 2
                    validate_hit_object_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be 4x3 matrix of 32-bit floats
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            let is_valid = match ty {
                                TypeStructure::Matrix {
                                    component,
                                    rows,
                                    cols,
                                } => {
                                    matches!(component, ScalarKind::Float(w) if w.get() == 32)
                                        && rows == VectorSize::VEC3
                                        && cols.get() == 4
                                }
                                _ => false,
                            };
                            if !is_valid {
                                return Err(ValidationError::HitObjectInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "4x3 matrix of 32-bit floats",
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpHitObjectGetShaderRecordBufferHandleNV.
pub struct HitObjectBufferHandleRule;

impl ValidationRule for HitObjectBufferHandleRule {
    fn name(&self) -> &'static str {
        "hit-object-buffer-handle"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::HitObjectGetShaderRecordBufferHandleNV {
                        continue;
                    }

                    // Hit object pointer at operand 2
                    validate_hit_object_pointer(inst, 2, ctx, func_id, block_id)?;

                    // Result must be 32-bit int vec2
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_int32_vec2(&ty) {
                                return Err(ValidationError::HitObjectInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                    expected: "32-bit int 2-component vector",
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpHitObjectGetAttributesNV and OpHitObjectExecuteShaderNV.
pub struct HitObjectAttributeAccessRule;

impl ValidationRule for HitObjectAttributeAccessRule {
    fn name(&self) -> &'static str {
        "hit-object-attribute-access"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if opcode == Op::HitObjectGetAttributesNV {
                        // Hit object pointer at operand 0
                        validate_hit_object_pointer(inst, 0, ctx, func_id, block_id)?;
                        // Hit object attribute at operand 1
                        validate_hit_object_attribute(inst, 1, ctx, func_id, block_id)?;
                    } else if opcode == Op::HitObjectExecuteShaderNV {
                        // Hit object pointer at operand 0
                        validate_hit_object_pointer(inst, 0, ctx, func_id, block_id)?;
                        // Payload at operand 1 (must be RayPayloadKHR)
                        let payload_id = match inst.operands.get(1).and_then(id_ref) {
                            Some(id) => id,
                            None => continue,
                        };
                        let variable = match ResultId::try_from(payload_id)
                            .ok()
                            .and_then(|id| ctx.definitions.get(&id))
                        {
                            Some(v) => v,
                            None => {
                                return Err(ValidationError::HitObjectPayloadInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        };
                        if variable.class.opcode != Op::Variable {
                            return Err(ValidationError::HitObjectPayloadInvalid {
                                function: func_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                        if let Some(Operand::StorageClass(sc)) = variable.operands.first() {
                            if *sc != StorageClass::RayPayloadKHR {
                                return Err(ValidationError::HitObjectPayloadInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpHitObjectRecordEmptyNV.
pub struct HitObjectRecordEmptyRule;

impl ValidationRule for HitObjectRecordEmptyRule {
    fn name(&self) -> &'static str {
        "hit-object-record-empty"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::HitObjectRecordEmptyNV {
                        continue;
                    }

                    // Hit object pointer at operand 0
                    validate_hit_object_pointer(inst, 0, ctx, func_id, block_id)?;
                }
            }
        }
        Ok(())
    }
}

/// Validates OpHitObjectRecordMissNV.
pub struct HitObjectRecordMissRule;

impl ValidationRule for HitObjectRecordMissRule {
    fn name(&self) -> &'static str {
        "hit-object-record-miss"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::HitObjectRecordMissNV {
                        continue;
                    }

                    // Hit object pointer at operand 0
                    validate_hit_object_pointer(inst, 0, ctx, func_id, block_id)?;

                    // Miss index (operand 1) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 1, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidMissIndex {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }
                    }

                    // Ray origin (operand 2) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 2, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayOrigin {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Ray TMin (operand 3) must be 32-bit float
                    if let Some(ty) = get_operand_type(inst, 3, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                                param_name: "TMin",
                            }
                            .into());
                        }
                    }

                    // Ray direction (operand 4) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 4, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayDirection {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Ray TMax (operand 5) must be 32-bit float
                    if let Some(ty) = get_operand_type(inst, 5, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                                param_name: "TMax",
                            }
                            .into());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpReorderThreadWithHintNV.
pub struct ReorderThreadWithHintRule;

impl ValidationRule for ReorderThreadWithHintRule {
    fn name(&self) -> &'static str {
        "reorder-thread-hint"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::ReorderThreadWithHintNV {
                        continue;
                    }

                    // Hint (operand 0) must be 32-bit int
                    if let Some(ty) = get_operand_type(inst, 0, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidHint {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }
                    }

                    // Bits (operand 1) must be 32-bit int
                    if let Some(ty) = get_operand_type(inst, 1, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidBits {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpReorderThreadWithHitObjectNV.
pub struct ReorderThreadWithHitObjectRule;

impl ValidationRule for ReorderThreadWithHitObjectRule {
    fn name(&self) -> &'static str {
        "reorder-thread-hit-object"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::ReorderThreadWithHitObjectNV {
                        continue;
                    }

                    // Hit object pointer at operand 0
                    validate_hit_object_pointer(inst, 0, ctx, func_id, block_id)?;

                    // Optional: Hint and Bits must both be present or neither
                    if inst.operands.len() > 1 {
                        if inst.operands.len() != 3 {
                            return Err(ValidationError::HitObjectOptionalOperandsMismatch {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }

                        // Hint (operand 1) must be 32-bit int
                        if let Some(ty) = get_operand_type(inst, 1, ctx) {
                            if !is_int32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidHint {
                                    function: func_id,
                                    block: block_id,
                                }
                                .into());
                            }
                        }

                        // Bits (operand 2) must be 32-bit int
                        if let Some(ty) = get_operand_type(inst, 2, ctx) {
                            if !is_int32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidBits {
                                    function: func_id,
                                    block: block_id,
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpHitObjectTraceRayNV.
pub struct HitObjectTraceRayRule;

impl ValidationRule for HitObjectTraceRayRule {
    fn name(&self) -> &'static str {
        "hit-object-trace-ray"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::HitObjectTraceRayNV {
                        continue;
                    }

                    // Hit object pointer at operand 0
                    validate_hit_object_pointer(inst, 0, ctx, func_id, block_id)?;

                    // Acceleration structure (operand 1) must be OpTypeAccelerationStructureKHR
                    if let Some(accel_id) = inst.operands.get(1).and_then(id_ref) {
                        if let Some(accel_inst) = ResultId::try_from(accel_id)
                            .ok()
                            .and_then(|id| ctx.definitions.get(&id))
                        {
                            if let Some(type_id) = accel_inst.result_type {
                                let type_opcode = ResultId::try_from(type_id)
                                    .ok()
                                    .and_then(|id| ctx.definitions.get(&id))
                                    .map(|i| i.class.opcode);
                                if type_opcode != Some(Op::TypeAccelerationStructureKHR) {
                                    return Err(ValidationError::HitObjectInvalidAccelStruct {
                                        function: func_id,
                                        block: block_id,
                                        opcode: inst.class.opcode,
                                    }
                                    .into());
                                }
                            }
                        }
                    }

                    // Ray flags (operand 2) must be 32-bit int
                    if let Some(ty) = get_operand_type(inst, 2, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayFlags {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Cull mask (operand 3) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 3, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidCullMask {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // SBT Offset (operand 4) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 4, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidSBTOffset {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // SBT Stride (operand 5) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 5, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidSBTStride {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Miss index (operand 6) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 6, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidMissIndex {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }
                    }

                    // Ray origin (operand 7) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 7, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayOrigin {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Ray TMin (operand 8) must be 32-bit float
                    if let Some(ty) = get_operand_type(inst, 8, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                                param_name: "TMin",
                            }
                            .into());
                        }
                    }

                    // Ray direction (operand 9) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 9, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayDirection {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Ray TMax (operand 10) must be 32-bit float
                    if let Some(ty) = get_operand_type(inst, 10, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                                param_name: "TMax",
                            }
                            .into());
                        }
                    }

                    // Payload (operand 11) must be RayPayloadKHR or IncomingRayPayloadKHR
                    if let Some(payload_id) = inst.operands.get(11).and_then(id_ref) {
                        if let Some(variable) = ResultId::try_from(payload_id)
                            .ok()
                            .and_then(|id| ctx.definitions.get(&id))
                        {
                            if variable.class.opcode != Op::Variable {
                                return Err(ValidationError::HitObjectPayloadInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                            if let Some(Operand::StorageClass(sc)) = variable.operands.first() {
                                if *sc != StorageClass::RayPayloadKHR
                                    && *sc != StorageClass::IncomingRayPayloadKHR
                                {
                                    return Err(ValidationError::HitObjectPayloadInvalid {
                                        function: func_id,
                                        block: block_id,
                                        opcode: inst.class.opcode,
                                    }
                                    .into());
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

/// Validates OpHitObjectTraceRayMotionNV.
pub struct HitObjectTraceRayMotionRule;

impl ValidationRule for HitObjectTraceRayMotionRule {
    fn name(&self) -> &'static str {
        "hit-object-trace-ray-motion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::HitObjectTraceRayMotionNV {
                        continue;
                    }

                    // Hit object pointer at operand 0
                    validate_hit_object_pointer(inst, 0, ctx, func_id, block_id)?;

                    // Acceleration structure (operand 1) must be OpTypeAccelerationStructureKHR
                    if let Some(accel_id) = inst.operands.get(1).and_then(id_ref) {
                        if let Some(accel_inst) = ResultId::try_from(accel_id)
                            .ok()
                            .and_then(|id| ctx.definitions.get(&id))
                        {
                            if let Some(type_id) = accel_inst.result_type {
                                let type_opcode = ResultId::try_from(type_id)
                                    .ok()
                                    .and_then(|id| ctx.definitions.get(&id))
                                    .map(|i| i.class.opcode);
                                if type_opcode != Some(Op::TypeAccelerationStructureKHR) {
                                    return Err(ValidationError::HitObjectInvalidAccelStruct {
                                        function: func_id,
                                        block: block_id,
                                        opcode: inst.class.opcode,
                                    }
                                    .into());
                                }
                            }
                        }
                    }

                    // Ray flags (operand 2) must be 32-bit int
                    if let Some(ty) = get_operand_type(inst, 2, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayFlags {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Cull mask (operand 3) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 3, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidCullMask {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // SBT Offset (operand 4) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 4, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidSBTOffset {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // SBT Stride (operand 5) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 5, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidSBTStride {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Miss index (operand 6) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 6, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidMissIndex {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }
                    }

                    // Ray origin (operand 7) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 7, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayOrigin {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Ray TMin (operand 8) must be 32-bit float
                    if let Some(ty) = get_operand_type(inst, 8, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                                param_name: "TMin",
                            }
                            .into());
                        }
                    }

                    // Ray direction (operand 9) must be 32-bit float vec3
                    if let Some(ty) = get_operand_type(inst, 9, ctx) {
                        if !is_float32_vec3(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayDirection {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Ray TMax (operand 10) must be 32-bit float
                    if let Some(ty) = get_operand_type(inst, 10, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidRayT {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                                param_name: "TMax",
                            }
                            .into());
                        }
                    }

                    // Current time (operand 11) must be 32-bit float
                    if let Some(ty) = get_operand_type(inst, 11, ctx) {
                        if !is_float32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidCurrentTime {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Payload (operand 12) must be RayPayloadKHR or IncomingRayPayloadKHR
                    if let Some(payload_id) = inst.operands.get(12).and_then(id_ref) {
                        if let Some(variable) = ResultId::try_from(payload_id)
                            .ok()
                            .and_then(|id| ctx.definitions.get(&id))
                        {
                            if variable.class.opcode != Op::Variable {
                                return Err(ValidationError::HitObjectPayloadInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into());
                            }
                            if let Some(Operand::StorageClass(sc)) = variable.operands.first() {
                                if *sc != StorageClass::RayPayloadKHR
                                    && *sc != StorageClass::IncomingRayPayloadKHR
                                {
                                    return Err(ValidationError::HitObjectPayloadInvalid {
                                        function: func_id,
                                        block: block_id,
                                        opcode: inst.class.opcode,
                                    }
                                    .into());
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

/// Validates OpHitObjectRecordHitNV and OpHitObjectRecordHitWithIndexNV.
pub struct HitObjectRecordHitRule;

impl ValidationRule for HitObjectRecordHitRule {
    fn name(&self) -> &'static str {
        "hit-object-record-hit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;
                    let is_with_index = opcode == Op::HitObjectRecordHitWithIndexNV;
                    let is_record_hit = opcode == Op::HitObjectRecordHitNV;

                    if !is_with_index && !is_record_hit {
                        continue;
                    }

                    // Hit object pointer at operand 0
                    validate_hit_object_pointer(inst, 0, ctx, func_id, block_id)?;

                    // Acceleration structure (operand 1) must be OpTypeAccelerationStructureKHR
                    if let Some(accel_id) = inst.operands.get(1).and_then(id_ref) {
                        if let Some(accel_inst) = ResultId::try_from(accel_id)
                            .ok()
                            .and_then(|id| ctx.definitions.get(&id))
                        {
                            if let Some(type_id) = accel_inst.result_type {
                                let type_opcode = ResultId::try_from(type_id)
                                    .ok()
                                    .and_then(|id| ctx.definitions.get(&id))
                                    .map(|i| i.class.opcode);
                                if type_opcode != Some(Op::TypeAccelerationStructureKHR) {
                                    return Err(ValidationError::HitObjectInvalidAccelStruct {
                                        function: func_id,
                                        block: block_id,
                                        opcode,
                                    }
                                    .into());
                                }
                            }
                        }
                    }

                    // Instance ID (operand 2) must be 32-bit int
                    if let Some(ty) = get_operand_type(inst, 2, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidInstanceId {
                                function: func_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }

                    // Primitive ID (operand 3) must be 32-bit int
                    if let Some(ty) = get_operand_type(inst, 3, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidPrimitiveId {
                                function: func_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }

                    // Geometry Index (operand 4) must be 32-bit int
                    if let Some(ty) = get_operand_type(inst, 4, ctx) {
                        if !is_int32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidGeometryIndex {
                                function: func_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }

                    // Hit Kind (operand 5) must be 32-bit unsigned int
                    if let Some(ty) = get_operand_type(inst, 5, ctx) {
                        if !is_uint32_scalar(&ty) {
                            return Err(ValidationError::HitObjectInvalidHitKind {
                                function: func_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }

                    // Different layout for WithIndex vs regular
                    if is_with_index {
                        // SBT Index (operand 6) must be 32-bit unsigned int
                        if let Some(ty) = get_operand_type(inst, 6, ctx) {
                            if !is_uint32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidSBTIndex {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }

                        // Ray origin (operand 7) must be 32-bit float vec3
                        if let Some(ty) = get_operand_type(inst, 7, ctx) {
                            if !is_float32_vec3(&ty) {
                                return Err(ValidationError::HitObjectInvalidRayOrigin {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }

                        // Ray TMin (operand 8) must be 32-bit float
                        if let Some(ty) = get_operand_type(inst, 8, ctx) {
                            if !is_float32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidRayT {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    param_name: "TMin",
                                }
                                .into());
                            }
                        }

                        // Ray direction (operand 9) must be 32-bit float vec3
                        if let Some(ty) = get_operand_type(inst, 9, ctx) {
                            if !is_float32_vec3(&ty) {
                                return Err(ValidationError::HitObjectInvalidRayDirection {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }

                        // Ray TMax (operand 10) must be 32-bit float
                        if let Some(ty) = get_operand_type(inst, 10, ctx) {
                            if !is_float32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidRayT {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    param_name: "TMax",
                                }
                                .into());
                            }
                        }

                        // Hit object attribute at operand 11
                        validate_hit_object_attribute(inst, 11, ctx, func_id, block_id)?;
                    } else {
                        // SBT Record Offset (operand 6) must be 32-bit unsigned int
                        if let Some(ty) = get_operand_type(inst, 6, ctx) {
                            if !is_uint32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidSBTRecordOffset {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }

                        // SBT Record Stride (operand 7) must be 32-bit unsigned int
                        if let Some(ty) = get_operand_type(inst, 7, ctx) {
                            if !is_uint32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidSBTRecordStride {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }

                        // Ray origin (operand 8) must be 32-bit float vec3
                        if let Some(ty) = get_operand_type(inst, 8, ctx) {
                            if !is_float32_vec3(&ty) {
                                return Err(ValidationError::HitObjectInvalidRayOrigin {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }

                        // Ray TMin (operand 9) must be 32-bit float
                        if let Some(ty) = get_operand_type(inst, 9, ctx) {
                            if !is_float32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidRayT {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    param_name: "TMin",
                                }
                                .into());
                            }
                        }

                        // Ray direction (operand 10) must be 32-bit float vec3
                        if let Some(ty) = get_operand_type(inst, 10, ctx) {
                            if !is_float32_vec3(&ty) {
                                return Err(ValidationError::HitObjectInvalidRayDirection {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }

                        // Ray TMax (operand 11) must be 32-bit float
                        if let Some(ty) = get_operand_type(inst, 11, ctx) {
                            if !is_float32_scalar(&ty) {
                                return Err(ValidationError::HitObjectInvalidRayT {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    param_name: "TMax",
                                }
                                .into());
                            }
                        }

                        // Hit object attribute at operand 12
                        validate_hit_object_attribute(inst, 12, ctx, func_id, block_id)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates hit object LSS array result instructions.
///
/// OpHitObjectGetLSSPositionsNV, OpHitObjectGetLSSRadiiNV
pub struct HitObjectLSSArrayRule;

impl ValidationRule for HitObjectLSSArrayRule {
    fn name(&self) -> &'static str {
        "hit-object-lss-array"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if opcode == Op::HitObjectGetLSSPositionsNV {
                        // Hit object pointer at operand 2
                        validate_hit_object_pointer(inst, 2, ctx, func_id, block_id)?;

                        // Result must be 2-element array of vec3 float
                        if let Some(result_type_id) = inst.result_type {
                            let length = get_array_length(result_type_id, ctx);
                            if length != Some(2) {
                                return Err(ValidationError::HitObjectInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "2-element array of 32-bit float 3-component vectors",
                                }
                                .into());
                            }

                            // Check element type
                            if let Ok(type_result_id) = ResultId::try_from(result_type_id) {
                                if let Some(array_type_inst) = ctx.definitions.get(&type_result_id)
                                {
                                    if let Some(element_id) =
                                        array_type_inst.operands.first().and_then(id_ref)
                                    {
                                        if let Ok(element_type_id) = TypeId::try_from(element_id) {
                                            let element_ty = get_type_structure(
                                                element_type_id,
                                                ctx.definitions,
                                            );
                                            if !is_float32_vec3(&element_ty) {
                                                return Err(ValidationError::HitObjectInvalidResultType {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode,
                                                    expected: "2-element array of 32-bit float 3-component vectors",
                                                }.into());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if opcode == Op::HitObjectGetLSSRadiiNV {
                        // Hit object pointer at operand 2
                        validate_hit_object_pointer(inst, 2, ctx, func_id, block_id)?;

                        // Result must be 2-element array of float
                        if let Some(result_type_id) = inst.result_type {
                            let length = get_array_length(result_type_id, ctx);
                            if length != Some(2) {
                                return Err(ValidationError::HitObjectInvalidResultType {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "2-element array of 32-bit float scalars",
                                }
                                .into());
                            }

                            // Check element type
                            if let Ok(type_result_id) = ResultId::try_from(result_type_id) {
                                if let Some(array_type_inst) = ctx.definitions.get(&type_result_id)
                                {
                                    if let Some(element_id) =
                                        array_type_inst.operands.first().and_then(id_ref)
                                    {
                                        if let Ok(element_type_id) = TypeId::try_from(element_id) {
                                            let element_ty = get_type_structure(
                                                element_type_id,
                                                ctx.definitions,
                                            );
                                            if !is_float32_scalar(&element_ty) {
                                                return Err(ValidationError::HitObjectInvalidResultType {
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
                }
            }
        }
        Ok(())
    }
}

/// Returns all hit object validation rules.
pub fn all_hit_object_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(HitObjectBoolResultRule),
        Box::new(HitObjectIntResultRule),
        Box::new(HitObjectFloatResultRule),
        Box::new(HitObjectVec3ResultRule),
        Box::new(HitObjectMatrixResultRule),
        Box::new(HitObjectBufferHandleRule),
        Box::new(HitObjectAttributeAccessRule),
        Box::new(HitObjectRecordEmptyRule),
        Box::new(HitObjectRecordMissRule),
        Box::new(HitObjectRecordHitRule),
        Box::new(HitObjectTraceRayRule),
        Box::new(HitObjectTraceRayMotionRule),
        Box::new(ReorderThreadWithHintRule),
        Box::new(ReorderThreadWithHitObjectRule),
        Box::new(HitObjectLSSArrayRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::types::BitWidth;

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
    }

    #[test]
    fn test_is_int32_vec2() {
        let vec2_signed = TypeStructure::Vector {
            component: ScalarKind::SignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC2,
        };
        assert!(is_int32_vec2(&vec2_signed));

        let vec2_unsigned = TypeStructure::Vector {
            component: ScalarKind::UnsignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC2,
        };
        assert!(is_int32_vec2(&vec2_unsigned));

        let vec3 = TypeStructure::Vector {
            component: ScalarKind::SignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC3,
        };
        assert!(!is_int32_vec2(&vec3));
    }
}
