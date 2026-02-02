//! Group operation validation rules.
//!
//! This module validates SPIR-V group operations including:
//!
//! - OpGroupAny, OpGroupAll (boolean predicate operations)
//! - OpGroupBroadcast (broadcast value to all invocations)
//! - OpGroupFAdd, OpGroupFMax, OpGroupFMin (float group operations)
//! - OpGroupIAdd, OpGroupUMin, OpGroupSMin, OpGroupUMax, OpGroupSMax (integer group operations)
//! - OpGroupAsyncCopy (async memory copy)
//! - OpGroupWaitEvents (wait for async events)

use std::collections::HashMap;

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{AddressingModel, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::{get_operand_type_structure, get_result_type_structure};
use crate::validation::type_ext::TypeInstructionExt;
use crate::validation::types::{ResultId, TypeId};
use crate::validation::ValidationResult;

// ============================================================================
// Helpers
// ============================================================================

/// Gets the type ID (result_type) of the value referenced by an operand.
fn get_operand_type_id(
    inst: &Instruction,
    operand_idx: usize,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<u32> {
    let operand_id = match inst.operands.get(operand_idx)? {
        Operand::IdRef(id) => *id,
        _ => return None,
    };
    let result_id = ResultId::try_from(operand_id).ok()?;
    definitions.get(&result_id)?.result_type
}

/// Gets the type instruction for the type of a value referenced by an operand.
fn get_operand_type_inst<'a>(
    inst: &Instruction,
    operand_idx: usize,
    definitions: &'a HashMap<ResultId, Instruction>,
) -> Option<&'a Instruction> {
    let type_id = get_operand_type_id(inst, operand_idx, definitions)?;
    let type_rid = ResultId::try_from(type_id).ok()?;
    definitions.get(&type_rid)
}

/// Gets the type instruction for a result type ID.
fn get_type_inst<'a>(
    type_id: u32,
    definitions: &'a HashMap<ResultId, Instruction>,
) -> Option<&'a Instruction> {
    let type_rid = ResultId::try_from(type_id).ok()?;
    definitions.get(&type_rid)
}

/// Gets the addressing model from the module.
fn get_addressing_model(ctx: &ValidationContext<'_>) -> Option<AddressingModel> {
    ctx.module
        .memory_model
        .as_ref()
        .and_then(|mm| mm.operands.first())
        .and_then(|op| match op {
            Operand::AddressingModel(am) => Some(*am),
            _ => None,
        })
}

// ============================================================================
// Group Operations Rule
// ============================================================================

/// Validates group operation instructions.
pub struct GroupRule;

impl ValidationRule for GroupRule {
    fn name(&self) -> &'static str {
        "group-operations"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for func in &ctx.module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    match inst.class.opcode {
                        Op::GroupAny | Op::GroupAll => {
                            validate_group_any_all(inst, ctx)?;
                        }
                        Op::GroupBroadcast => {
                            validate_group_broadcast(inst, ctx)?;
                        }
                        Op::GroupFAdd | Op::GroupFMax | Op::GroupFMin => {
                            validate_group_float(inst, ctx)?;
                        }
                        Op::GroupIAdd
                        | Op::GroupUMin
                        | Op::GroupSMin
                        | Op::GroupUMax
                        | Op::GroupSMax => {
                            validate_group_int(inst, ctx)?;
                        }
                        Op::GroupAsyncCopy => {
                            validate_group_async_copy(inst, ctx)?;
                        }
                        Op::GroupWaitEvents => {
                            validate_group_wait_events(inst, ctx)?;
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates OpGroupAny and OpGroupAll.
///
/// - Result must be a boolean scalar type
/// - Predicate must be a boolean scalar type
fn validate_group_any_all(inst: &Instruction, ctx: &ValidationContext<'_>) -> ValidationResult {
    let opcode = inst.class.opcode;

    // Result must be bool scalar
    if let Some(ts) = get_result_type_structure(inst, ctx.definitions) {
        if !ts.is_bool_scalar() {
            return Err(ValidationError::GroupResultMustBeBoolScalar { opcode }.into());
        }
    }

    // Predicate (rspirv operand 1: Execution=0, Predicate=1) must be bool scalar
    if let Some(ts) = get_operand_type_structure(inst, 1, ctx.definitions) {
        if !ts.is_bool_scalar() {
            return Err(ValidationError::GroupPredicateMustBeBoolScalar { opcode }.into());
        }
    }

    Ok(())
}

/// Validates OpGroupBroadcast.
///
/// - Result must be a scalar or vector of integer, floating-point, or boolean type
/// - The type of Value must match the Result type
fn validate_group_broadcast(inst: &Instruction, ctx: &ValidationContext<'_>) -> ValidationResult {
    let opcode = inst.class.opcode;

    // Result must be scalar or vector of int, float, or bool
    if let Some(ts) = get_result_type_structure(inst, ctx.definitions) {
        if !ts.is_float_scalar_or_vector()
            && !ts.is_int_scalar_or_vector()
            && !ts.is_bool_scalar_or_vector()
        {
            return Err(ValidationError::GroupBroadcastResultInvalidType { opcode }.into());
        }
    }

    // Value type (rspirv operand 1: Execution=0, Value=1) must match result type
    if let (Some(result_type), Some(value_type)) = (
        inst.result_type,
        get_operand_type_id(inst, 1, ctx.definitions),
    ) {
        if result_type != value_type {
            return Err(ValidationError::GroupValueTypeMismatch { opcode }.into());
        }
    }

    Ok(())
}

/// Validates OpGroupFAdd, OpGroupFMax, OpGroupFMin.
///
/// - Result must be a scalar or vector of float type
/// - The type of X must match the Result type
fn validate_group_float(inst: &Instruction, ctx: &ValidationContext<'_>) -> ValidationResult {
    let opcode = inst.class.opcode;

    // Result must be float scalar or vector
    if let Some(ts) = get_result_type_structure(inst, ctx.definitions) {
        if !ts.is_float_scalar_or_vector() {
            return Err(ValidationError::GroupResultMustBeFloatScalarOrVector { opcode }.into());
        }
    }

    // X type (rspirv operand 2: Execution=0, Operation=1, X=2) must match result type
    if let (Some(result_type), Some(x_type)) = (
        inst.result_type,
        get_operand_type_id(inst, 2, ctx.definitions),
    ) {
        if result_type != x_type {
            return Err(ValidationError::GroupXTypeMismatch { opcode }.into());
        }
    }

    Ok(())
}

/// Validates OpGroupIAdd, OpGroupUMin, OpGroupSMin, OpGroupUMax, OpGroupSMax.
///
/// - Result must be a scalar or vector of integer type
/// - The type of X must match the Result type
fn validate_group_int(inst: &Instruction, ctx: &ValidationContext<'_>) -> ValidationResult {
    let opcode = inst.class.opcode;

    // Result must be int scalar or vector
    if let Some(ts) = get_result_type_structure(inst, ctx.definitions) {
        if !ts.is_int_scalar_or_vector() {
            return Err(ValidationError::GroupResultMustBeIntScalarOrVector { opcode }.into());
        }
    }

    // X type (rspirv operand 2: Execution=0, Operation=1, X=2) must match result type
    if let (Some(result_type), Some(x_type)) = (
        inst.result_type,
        get_operand_type_id(inst, 2, ctx.definitions),
    ) {
        if result_type != x_type {
            return Err(ValidationError::GroupXTypeMismatch { opcode }.into());
        }
    }

    Ok(())
}

/// Validates OpGroupAsyncCopy.
///
/// - Result type must be OpTypeEvent
/// - Destination must be a pointer with storage class Workgroup or CrossWorkgroup
/// - Destination pointee must be scalar or vector of float or integer type
/// - Source and Destination must point to the same type
/// - Storage classes must be paired (Workgroup↔CrossWorkgroup)
/// - NumElements must be an int scalar with appropriate bit width
/// - Stride must be an int scalar with appropriate bit width
/// - Event must be OpTypeEvent
fn validate_group_async_copy(inst: &Instruction, ctx: &ValidationContext<'_>) -> ValidationResult {
    // Check result type is OpTypeEvent
    if let Some(result_type_id) = inst.result_type {
        if let Some(type_inst) = get_type_inst(result_type_id, ctx.definitions) {
            if type_inst.class.opcode != Op::TypeEvent {
                return Err(ValidationError::GroupAsyncCopyResultNotEvent.into());
            }
        }
    }

    // rspirv operands: Execution=0, Destination=1, Source=2, NumElements=3, Stride=4, Event=5

    // Validate Destination pointer
    let dest_type_inst = get_operand_type_inst(inst, 1, ctx.definitions);
    let dest_sc;
    let dest_pointee_type_id;

    if let Some(dest_type) = dest_type_inst {
        if !dest_type.is_pointer_type() {
            return Err(ValidationError::GroupAsyncCopyDestNotPointer.into());
        }
        dest_sc = dest_type.pointer_storage_class();
        if dest_sc != Some(StorageClass::Workgroup) && dest_sc != Some(StorageClass::CrossWorkgroup)
        {
            return Err(ValidationError::GroupAsyncCopyDestInvalidStorageClass.into());
        }
        dest_pointee_type_id = dest_type.pointer_pointee_type_id();

        // Check pointee type is scalar or vector of float or int
        if let Some(pointee_id) = dest_pointee_type_id {
            if let Ok(tid) = TypeId::try_from(pointee_id) {
                let ts = crate::validation::helpers::get_type_structure(tid, ctx.definitions);
                if !ts.is_int_scalar_or_vector() && !ts.is_float_scalar_or_vector() {
                    return Err(ValidationError::GroupAsyncCopyDestInvalidPointeeType.into());
                }
            }
        }
    } else {
        dest_sc = None;
        dest_pointee_type_id = None;
    }

    // Validate Source pointer - must have same pointee type as Destination
    let source_type_inst = get_operand_type_inst(inst, 2, ctx.definitions);
    if let Some(source_type) = source_type_inst {
        let source_pointee_type_id = source_type.pointer_pointee_type_id();
        let source_sc = source_type.pointer_storage_class();

        // Check pointee types match
        if let (Some(dest_pt), Some(src_pt)) = (dest_pointee_type_id, source_pointee_type_id) {
            if dest_pt != src_pt {
                return Err(ValidationError::GroupAsyncCopyTypeMismatch.into());
            }
        }

        // Check storage class pairing
        if let (Some(d_sc), Some(s_sc)) = (dest_sc, source_sc) {
            if d_sc == StorageClass::Workgroup && s_sc != StorageClass::CrossWorkgroup {
                return Err(ValidationError::GroupAsyncCopyStorageClassMismatch {
                    message: "If Destination storage class is Workgroup, then the Source storage class must be CrossWorkgroup.".to_string(),
                }
                .into());
            } else if d_sc == StorageClass::CrossWorkgroup && s_sc != StorageClass::Workgroup {
                return Err(ValidationError::GroupAsyncCopyStorageClassMismatch {
                    message: "If Destination storage class is CrossWorkgroup, then the Source storage class must be Workgroup.".to_string(),
                }
                .into());
            }
        }
    }

    // Check NumElements and Stride types based on addressing model
    let is_physical_64 = get_addressing_model(ctx) == Some(AddressingModel::Physical64);
    let bit_width = if is_physical_64 { 64 } else { 32 };
    let addressing_model_name = if is_physical_64 {
        "Physical64"
    } else {
        "Physical32"
    };

    // NumElements (rspirv operand 3) must be int scalar with appropriate width
    if let Some(num_elem_type_inst) = get_operand_type_inst(inst, 3, ctx.definitions) {
        if !num_elem_type_inst.is_int_type()
            || num_elem_type_inst.numeric_bit_width() != Some(bit_width)
        {
            return Err(ValidationError::GroupAsyncCopyNumElementsInvalidType {
                bit_width,
                addressing_model: addressing_model_name.to_string(),
            }
            .into());
        }
    }

    // Stride (rspirv operand 4) must be int scalar with appropriate width
    if let Some(stride_type_inst) = get_operand_type_inst(inst, 4, ctx.definitions) {
        if !stride_type_inst.is_int_type()
            || stride_type_inst.numeric_bit_width() != Some(bit_width)
        {
            return Err(ValidationError::GroupAsyncCopyStrideInvalidType {
                bit_width,
                addressing_model: addressing_model_name.to_string(),
            }
            .into());
        }
    }

    // Event (rspirv operand 5) must be OpTypeEvent
    if let Some(event_type_inst) = get_operand_type_inst(inst, 5, ctx.definitions) {
        if event_type_inst.class.opcode != Op::TypeEvent {
            return Err(ValidationError::GroupAsyncCopyEventNotEvent.into());
        }
    }

    Ok(())
}

/// Validates OpGroupWaitEvents.
///
/// - Num Events must be a 32-bit int scalar
/// - Events List must be a pointer to OpTypeEvent
fn validate_group_wait_events(inst: &Instruction, ctx: &ValidationContext<'_>) -> ValidationResult {
    // rspirv operands: Execution=0, NumEvents=1, EventsList=2

    // NumEvents (rspirv operand 1) must be 32-bit int scalar
    if let Some(num_events_type_inst) = get_operand_type_inst(inst, 1, ctx.definitions) {
        if !num_events_type_inst.is_int_type()
            || num_events_type_inst.numeric_bit_width() != Some(32)
        {
            return Err(ValidationError::GroupWaitEventsNumEventsInvalidType.into());
        }
    }

    // EventsList (rspirv operand 2) must be pointer to OpTypeEvent
    if let Some(events_type_inst) = get_operand_type_inst(inst, 2, ctx.definitions) {
        if !events_type_inst.is_pointer_type() {
            return Err(ValidationError::GroupWaitEventsEventsListNotPointer.into());
        }
        // Check pointee type is OpTypeEvent
        if let Some(pointee_id) = events_type_inst.pointer_pointee_type_id() {
            if let Some(pointee_inst) = get_type_inst(pointee_id, ctx.definitions) {
                if pointee_inst.class.opcode != Op::TypeEvent {
                    return Err(ValidationError::GroupWaitEventsEventsListNotEventPointer.into());
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// All group rules
// ============================================================================

static GROUP_RULE: GroupRule = GroupRule;

/// Returns all group operation validation rules.
pub fn all_group_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&GROUP_RULE]
}
