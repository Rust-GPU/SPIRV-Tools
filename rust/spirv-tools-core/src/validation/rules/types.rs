//! Type and ID validation rules.
//!
//! This module validates SPIR-V type and ID requirements including:
//!
//! - Result types must be type opcodes
//! - OpTypeFunction parameter validation
//! - Operand definitions

use std::collections::HashSet;

use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};

// ============================================================================
// Result Types Are Types Rule
// ============================================================================

/// Validates that result_type fields reference actual type instructions.
pub struct ResultTypesAreTypesRule;

impl ValidationRule for ResultTypesAreTypesRule {
    fn name(&self) -> &'static str {
        "result-types-are-types"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.definitions.values() {
            if let Some(result_type_raw) = inst.result_type {
                if let Ok(type_id) = ResultId::try_from(result_type_raw) {
                    if let Some(type_opcode) = ctx.opcodes.get(&type_id) {
                        if !is_type_opcode(*type_opcode) {
                            return Err(ValidationError::ResultTypeNotType {
                                instruction: inst.class.opcode,
                                result_type: Id::from(type_id),
                                found: *type_opcode,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Type Functions Rule
// ============================================================================

/// Validates OpTypeFunction requirements.
pub struct TypeFunctionsRule;

impl ValidationRule for TypeFunctionsRule {
    fn name(&self) -> &'static str {
        "type-functions"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeFunction {
                continue;
            }
            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .ok_or(ValidationError::ZeroId {
                    kind: crate::validation::types::IdKind::Result,
                    opcode: inst.class.opcode,
                })?;

            let mut operands = inst.operands.iter();
            let return_type = match operands.next() {
                Some(rspirv::dr::Operand::IdRef(raw)) => TypeId::try_from(*raw)
                    .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?,
                _ => {
                    return Err(ValidationError::InvalidTypeFunction { type_id });
                }
            };

            let return_id = ResultId::try_from(u32::from(return_type))
                .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?;
            let return_opcode = ctx
                .opcodes
                .get(&return_id)
                .copied()
                .ok_or(ValidationError::InvalidTypeFunction { type_id })?;
            if !is_type_opcode(return_opcode) {
                return Err(ValidationError::InvalidTypeFunction { type_id });
            }

            for op in operands {
                let param_type = match op {
                    rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw)
                        .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?,
                    _ => {
                        return Err(ValidationError::InvalidTypeFunction { type_id });
                    }
                };
                let param_id = ResultId::try_from(u32::from(param_type))
                    .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?;
                let param_opcode = ctx
                    .opcodes
                    .get(&param_id)
                    .copied()
                    .ok_or(ValidationError::InvalidTypeFunction { type_id })?;
                if param_opcode == Op::TypeVoid {
                    return Err(ValidationError::FunctionTypeParameterVoid {
                        type_id,
                        parameter: param_type,
                    });
                }
                if !is_type_opcode(param_opcode) {
                    return Err(ValidationError::InvalidTypeFunction { type_id });
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Operand Definitions Rule
// ============================================================================

/// Validates that all operand IDs are defined.
pub struct OperandDefinitionsRule;

impl ValidationRule for OperandDefinitionsRule {
    fn name(&self) -> &'static str {
        "operand-definitions"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            check_instruction_ids(inst, ctx.defined_ids, None)?;
        }
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|def| def.result_id)
                .and_then(|raw| Id::try_from(raw).ok());
            if let Some(def) = &function.def {
                check_instruction_ids(def, ctx.defined_ids, function_id)?;
            }
            for param in &function.parameters {
                check_instruction_ids(param, ctx.defined_ids, function_id)?;
            }
            for block in &function.blocks {
                for inst in &block.instructions {
                    check_instruction_ids(inst, ctx.defined_ids, function_id)?;
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn is_type_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::TypeVoid
            | Op::TypeBool
            | Op::TypeInt
            | Op::TypeFloat
            | Op::TypeVector
            | Op::TypeMatrix
            | Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
            | Op::TypeArray
            | Op::TypeRuntimeArray
            | Op::TypeStruct
            | Op::TypeOpaque
            | Op::TypePointer
            | Op::TypeUntypedPointerKHR
            | Op::TypeFunction
            | Op::TypeEvent
            | Op::TypeDeviceEvent
            | Op::TypeReserveId
            | Op::TypeQueue
            | Op::TypePipe
            | Op::TypeForwardPointer
            | Op::TypePipeStorage
            | Op::TypeNamedBarrier
            | Op::TypeAccelerationStructureKHR
            | Op::TypeCooperativeMatrixKHR
            | Op::TypeCooperativeMatrixNV
            | Op::TypeRayQueryKHR
            | Op::TypeHitObjectNV
    )
}

#[allow(clippy::manual_is_multiple_of)]
fn is_block_operand(opcode: Op, index: usize) -> bool {
    match opcode {
        Op::Branch => index == 0,
        Op::BranchConditional => index == 1 || index == 2,
        Op::Switch => index == 1 || (index > 1 && index % 2 == 0),
        Op::LoopMerge => index <= 1,
        Op::SelectionMerge => index == 0,
        Op::Phi => index % 2 == 1,
        _ => false,
    }
}

fn check_instruction_ids(
    inst: &rspirv::dr::Instruction,
    defined_ids: &HashSet<Id>,
    function: Option<Id>,
) -> Result<(), ValidationError> {
    if let Some(result_type) = inst.result_type {
        if let Ok(id) = Id::try_from(result_type) {
            if !defined_ids.contains(&id) {
                return Err(ValidationError::UndefinedId { function, id });
            }
        }
    }

    for (idx, operand) in inst.operands.iter().enumerate() {
        if is_block_operand(inst.class.opcode, idx) {
            continue;
        }
        if let rspirv::dr::Operand::IdRef(raw) = operand {
            if let Ok(id) = Id::try_from(*raw) {
                if !defined_ids.contains(&id) {
                    return Err(ValidationError::UndefinedId { function, id });
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// All type rules
// ============================================================================

/// Returns all type validation rules.
pub fn all_type_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &ResultTypesAreTypesRule,
        &TypeFunctionsRule,
        &OperandDefinitionsRule,
    ]
}
