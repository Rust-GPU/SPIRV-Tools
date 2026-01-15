//! Non-uniform group operation validation rules.
//!
//! This module validates SPIR-V non-uniform group operations:
//! - `OpGroupNonUniformElect`
//! - `OpGroupNonUniformAll/Any/AllEqual`
//! - `OpGroupNonUniformBroadcast/First/Shuffle/ShuffleXor/ShuffleUp/ShuffleDown`
//! - `OpGroupNonUniformQuadBroadcast/QuadSwap`
//! - `OpGroupNonUniformBallot/InverseBallot/BallotBitExtract/BallotBitCount/BallotFindLSB/BallotFindMSB`
//! - `OpGroupNonUniformArithmetic` operations (IAdd, FAdd, IMul, FMul, SMin, UMin, FMin, etc.)
//! - `OpGroupNonUniformRotateKHR`

use rspirv::dr::Operand;
use rspirv::spirv::{GroupOperation, Op};

use crate::validation::context::ValidationContext;
use crate::validation::error::ValidationError;
use crate::validation::helpers::{get_type_structure, id_ref, is_constant_opcode};
use crate::validation::types::{Id, ResultId, TypeId, TypeStructure, VectorSize};
use crate::version::SpirvVersion;

use super::super::context::ValidationRule;

/// Returns true if the opcode is a non-uniform group operation.
pub fn is_non_uniform_group_operation(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::GroupNonUniformElect
            | Op::GroupNonUniformAll
            | Op::GroupNonUniformAny
            | Op::GroupNonUniformAllEqual
            | Op::GroupNonUniformBroadcast
            | Op::GroupNonUniformBroadcastFirst
            | Op::GroupNonUniformBallot
            | Op::GroupNonUniformInverseBallot
            | Op::GroupNonUniformBallotBitExtract
            | Op::GroupNonUniformBallotBitCount
            | Op::GroupNonUniformBallotFindLSB
            | Op::GroupNonUniformBallotFindMSB
            | Op::GroupNonUniformShuffle
            | Op::GroupNonUniformShuffleXor
            | Op::GroupNonUniformShuffleUp
            | Op::GroupNonUniformShuffleDown
            | Op::GroupNonUniformIAdd
            | Op::GroupNonUniformFAdd
            | Op::GroupNonUniformIMul
            | Op::GroupNonUniformFMul
            | Op::GroupNonUniformSMin
            | Op::GroupNonUniformUMin
            | Op::GroupNonUniformFMin
            | Op::GroupNonUniformSMax
            | Op::GroupNonUniformUMax
            | Op::GroupNonUniformFMax
            | Op::GroupNonUniformBitwiseAnd
            | Op::GroupNonUniformBitwiseOr
            | Op::GroupNonUniformBitwiseXor
            | Op::GroupNonUniformLogicalAnd
            | Op::GroupNonUniformLogicalOr
            | Op::GroupNonUniformLogicalXor
            | Op::GroupNonUniformQuadBroadcast
            | Op::GroupNonUniformQuadSwap
            | Op::GroupNonUniformRotateKHR
            | Op::GroupNonUniformQuadAllKHR
            | Op::GroupNonUniformQuadAnyKHR
    )
}

/// Returns true if the type is a valid scalar or vector for non-uniform operations.
fn is_valid_value_type(ty: &TypeStructure) -> bool {
    ty.is_float_scalar_or_vector() || ty.is_int_scalar_or_vector() || ty.is_bool_scalar_or_vector()
}

/// Returns true if the type is a 4-component unsigned integer vector.
fn is_unsigned_int_vec4(ty: &TypeStructure) -> bool {
    match ty {
        TypeStructure::Vector { component, size } => {
            component.is_unsigned_int() && *size == VectorSize::VEC4
        }
        _ => false,
    }
}

/// Returns true if the type is a 4-component integer vector (signed or unsigned).
fn is_int_vec4(ty: &TypeStructure) -> bool {
    match ty {
        TypeStructure::Vector { component, size } => component.is_int() && *size == VectorSize::VEC4,
        _ => false,
    }
}

/// Returns the operand name for broadcast/shuffle instructions.
fn get_broadcast_shuffle_operand_name(opcode: Op) -> &'static str {
    match opcode {
        Op::GroupNonUniformBroadcast | Op::GroupNonUniformShuffle => "Id",
        Op::GroupNonUniformShuffleXor => "Mask",
        Op::GroupNonUniformQuadBroadcast => "Index",
        Op::GroupNonUniformQuadSwap => "Direction",
        Op::GroupNonUniformShuffleUp | Op::GroupNonUniformShuffleDown => "Delta",
        _ => "operand",
    }
}

/// Validates OpGroupNonUniformElect.
pub struct NonUniformElectRule;

impl ValidationRule for NonUniformElectRule {
    fn name(&self) -> &'static str {
        "non-uniform-elect"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::GroupNonUniformElect {
                        continue;
                    }

                    // Result must be boolean scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_bool_scalar() {
                                return Err(ValidationError::NonUniformResultMustBeBoolScalar {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::GroupNonUniformElect,
                                });
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpGroupNonUniformAll and OpGroupNonUniformAny.
pub struct NonUniformAnyAllRule;

impl ValidationRule for NonUniformAnyAllRule {
    fn name(&self) -> &'static str {
        "non-uniform-any-all"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                    if opcode != Op::GroupNonUniformAll
                        && opcode != Op::GroupNonUniformAny
                        && opcode != Op::GroupNonUniformQuadAllKHR
                        && opcode != Op::GroupNonUniformQuadAnyKHR
                    {
                        continue;
                    }

                    // Result must be boolean scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_bool_scalar() {
                                return Err(ValidationError::NonUniformResultMustBeBoolScalar {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                });
                            }
                        }
                    }

                    // Predicate (operand 3 for Any/All, operand 2 for QuadAllKHR/QuadAnyKHR) must be boolean scalar
                    let predicate_idx = if opcode == Op::GroupNonUniformQuadAllKHR
                        || opcode == Op::GroupNonUniformQuadAnyKHR
                    {
                        2 // QuadAllKHR/QuadAnyKHR don't have scope parameter
                    } else {
                        3
                    };

                    if let Some(pred_id) = inst.operands.get(predicate_idx).and_then(id_ref) {
                        if let Ok(pred_result_id) = ResultId::try_from(pred_id) {
                            if let Some(pred_inst) = ctx.definitions.get(&pred_result_id) {
                                if let Some(pred_type_id) = pred_inst.result_type {
                                    if let Ok(type_id) = TypeId::try_from(pred_type_id) {
                                        let ty = get_type_structure(type_id, ctx.definitions);
                                        if !ty.is_bool_scalar() {
                                            return Err(
                                                ValidationError::NonUniformPredicateMustBeBoolScalar {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode,
                                                },
                                            );
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

/// Validates OpGroupNonUniformAllEqual.
pub struct NonUniformAllEqualRule;

impl ValidationRule for NonUniformAllEqualRule {
    fn name(&self) -> &'static str {
        "non-uniform-all-equal"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::GroupNonUniformAllEqual {
                        continue;
                    }

                    // Result must be boolean scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_bool_scalar() {
                                return Err(ValidationError::NonUniformResultMustBeBoolScalar {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::GroupNonUniformAllEqual,
                                });
                            }
                        }
                    }

                    // Value (operand 3) must be scalar or vector of int/float/bool
                    if let Some(value_id) = inst.operands.get(3).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let Some(value_type_id) = value_inst.result_type {
                                    if let Ok(type_id) = TypeId::try_from(value_type_id) {
                                        let ty = get_type_structure(type_id, ctx.definitions);
                                        if !is_valid_value_type(&ty) {
                                            return Err(
                                                ValidationError::NonUniformValueInvalidType {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode: Op::GroupNonUniformAllEqual,
                                                },
                                            );
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

/// Validates OpGroupNonUniformBroadcast and shuffle operations.
pub struct NonUniformBroadcastShuffleRule;

impl ValidationRule for NonUniformBroadcastShuffleRule {
    fn name(&self) -> &'static str {
        "non-uniform-broadcast-shuffle"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        Op::GroupNonUniformBroadcast
                            | Op::GroupNonUniformShuffle
                            | Op::GroupNonUniformShuffleXor
                            | Op::GroupNonUniformShuffleUp
                            | Op::GroupNonUniformShuffleDown
                            | Op::GroupNonUniformQuadBroadcast
                            | Op::GroupNonUniformQuadSwap
                    ) {
                        continue;
                    }

                    // Result must be scalar or vector of int/float/bool
                    let result_type = inst.result_type.and_then(|id| TypeId::try_from(id).ok());
                    if let Some(type_id) = result_type {
                        let ty = get_type_structure(type_id, ctx.definitions);
                        if !is_valid_value_type(&ty) {
                            return Err(ValidationError::NonUniformResultTypeInvalid {
                                function: func_id,
                                block: block_id,
                                opcode,
                                expected: "scalar or vector of integer, floating-point, or boolean type",
                            });
                        }
                    }

                    // Value type must match result type
                    if let Some(value_id) = inst.operands.get(3).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let (Some(result_type_id), Some(value_type_id)) =
                                    (inst.result_type, value_inst.result_type)
                                {
                                    if result_type_id != value_type_id {
                                        return Err(ValidationError::NonUniformValueTypeMismatch {
                                            function: func_id,
                                            block: block_id,
                                            opcode,
                                        });
                                    }
                                }
                            }
                        }
                    }

                    // Id/Index/Mask/Delta/Direction must be unsigned integer scalar
                    if let Some(id_operand) = inst.operands.get(4).and_then(id_ref) {
                        if let Ok(id_result) = ResultId::try_from(id_operand) {
                            if let Some(id_inst) = ctx.definitions.get(&id_result) {
                                if let Some(id_type_raw) = id_inst.result_type {
                                    if let Ok(id_type_id) = TypeId::try_from(id_type_raw) {
                                        let id_ty = get_type_structure(id_type_id, ctx.definitions);
                                        if !id_ty.is_unsigned_int_scalar() {
                                            return Err(
                                                ValidationError::NonUniformIdMustBeUnsignedInt {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode,
                                                    operand_name: get_broadcast_shuffle_operand_name(
                                                        opcode,
                                                    ),
                                                },
                                            );
                                        }

                                        // Check if constant is required
                                        let should_be_constant = opcode
                                            == Op::GroupNonUniformQuadSwap
                                            || ((opcode == Op::GroupNonUniformBroadcast
                                                || opcode == Op::GroupNonUniformQuadBroadcast)
                                                && ctx.target_version < SpirvVersion::new(1, 5));

                                        if should_be_constant
                                            && !is_constant_opcode(id_inst.class.opcode)
                                        {
                                            return Err(
                                                ValidationError::NonUniformIdMustBeConstant {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode,
                                                    operand_name: get_broadcast_shuffle_operand_name(
                                                        opcode,
                                                    ),
                                                },
                                            );
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

/// Validates OpGroupNonUniformBroadcastFirst.
pub struct NonUniformBroadcastFirstRule;

impl ValidationRule for NonUniformBroadcastFirstRule {
    fn name(&self) -> &'static str {
        "non-uniform-broadcast-first"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::GroupNonUniformBroadcastFirst {
                        continue;
                    }

                    // Result must be scalar or vector of int/float/bool
                    let result_type = inst.result_type.and_then(|id| TypeId::try_from(id).ok());
                    if let Some(type_id) = result_type {
                        let ty = get_type_structure(type_id, ctx.definitions);
                        if !is_valid_value_type(&ty) {
                            return Err(ValidationError::NonUniformResultTypeInvalid {
                                function: func_id,
                                block: block_id,
                                opcode: Op::GroupNonUniformBroadcastFirst,
                                expected: "scalar or vector of integer, floating-point, or boolean type",
                            });
                        }
                    }

                    // Value type must match result type
                    if let Some(value_id) = inst.operands.get(3).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let (Some(result_type_id), Some(value_type_id)) =
                                    (inst.result_type, value_inst.result_type)
                                {
                                    if result_type_id != value_type_id {
                                        return Err(ValidationError::NonUniformValueTypeMismatch {
                                            function: func_id,
                                            block: block_id,
                                            opcode: Op::GroupNonUniformBroadcastFirst,
                                        });
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

/// Validates OpGroupNonUniformBallot.
pub struct NonUniformBallotRule;

impl ValidationRule for NonUniformBallotRule {
    fn name(&self) -> &'static str {
        "non-uniform-ballot"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::GroupNonUniformBallot {
                        continue;
                    }

                    // Result must be 4-component unsigned integer vector
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !is_unsigned_int_vec4(&ty) {
                                return Err(ValidationError::NonUniformBallotResultInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::GroupNonUniformBallot,
                                });
                            }
                        }
                    }

                    // Predicate must be boolean scalar
                    if let Some(pred_id) = inst.operands.get(3).and_then(id_ref) {
                        if let Ok(pred_result_id) = ResultId::try_from(pred_id) {
                            if let Some(pred_inst) = ctx.definitions.get(&pred_result_id) {
                                if let Some(pred_type_id) = pred_inst.result_type {
                                    if let Ok(type_id) = TypeId::try_from(pred_type_id) {
                                        let ty = get_type_structure(type_id, ctx.definitions);
                                        if !ty.is_bool_scalar() {
                                            return Err(
                                                ValidationError::NonUniformPredicateMustBeBoolScalar {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode: Op::GroupNonUniformBallot,
                                                },
                                            );
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

/// Validates OpGroupNonUniformInverseBallot.
pub struct NonUniformInverseBallotRule;

impl ValidationRule for NonUniformInverseBallotRule {
    fn name(&self) -> &'static str {
        "non-uniform-inverse-ballot"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::GroupNonUniformInverseBallot {
                        continue;
                    }

                    // Result must be boolean scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_bool_scalar() {
                                return Err(ValidationError::NonUniformResultMustBeBoolScalar {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::GroupNonUniformInverseBallot,
                                });
                            }
                        }
                    }

                    // Value must be 4-component unsigned integer vector
                    if let Some(value_id) = inst.operands.get(3).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let Some(value_type_id) = value_inst.result_type {
                                    if let Ok(type_id) = TypeId::try_from(value_type_id) {
                                        let ty = get_type_structure(type_id, ctx.definitions);
                                        if !is_unsigned_int_vec4(&ty) {
                                            return Err(
                                                ValidationError::NonUniformBallotValueInvalid {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode: Op::GroupNonUniformInverseBallot,
                                                },
                                            );
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

/// Validates OpGroupNonUniformBallotBitExtract.
pub struct NonUniformBallotBitExtractRule;

impl ValidationRule for NonUniformBallotBitExtractRule {
    fn name(&self) -> &'static str {
        "non-uniform-ballot-bit-extract"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::GroupNonUniformBallotBitExtract {
                        continue;
                    }

                    // Result must be boolean scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_bool_scalar() {
                                return Err(ValidationError::NonUniformResultMustBeBoolScalar {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::GroupNonUniformBallotBitExtract,
                                });
                            }
                        }
                    }

                    // Value must be 4-component unsigned integer vector
                    if let Some(value_id) = inst.operands.get(3).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let Some(value_type_id) = value_inst.result_type {
                                    if let Ok(type_id) = TypeId::try_from(value_type_id) {
                                        let ty = get_type_structure(type_id, ctx.definitions);
                                        if !is_unsigned_int_vec4(&ty) {
                                            return Err(
                                                ValidationError::NonUniformBallotValueInvalid {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode: Op::GroupNonUniformBallotBitExtract,
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Index must be unsigned integer scalar
                    if let Some(idx_id) = inst.operands.get(4).and_then(id_ref) {
                        if let Ok(idx_result) = ResultId::try_from(idx_id) {
                            if let Some(idx_inst) = ctx.definitions.get(&idx_result) {
                                if let Some(idx_type_raw) = idx_inst.result_type {
                                    if let Ok(idx_type_id) = TypeId::try_from(idx_type_raw) {
                                        let idx_ty = get_type_structure(idx_type_id, ctx.definitions);
                                        if !idx_ty.is_unsigned_int_scalar() {
                                            return Err(
                                                ValidationError::NonUniformIdMustBeUnsignedInt {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode: Op::GroupNonUniformBallotBitExtract,
                                                    operand_name: "Id",
                                                },
                                            );
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

/// Validates OpGroupNonUniformBallotBitCount.
pub struct NonUniformBallotBitCountRule;

impl ValidationRule for NonUniformBallotBitCountRule {
    fn name(&self) -> &'static str {
        "non-uniform-ballot-bit-count"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::GroupNonUniformBallotBitCount {
                        continue;
                    }

                    // Result must be unsigned integer scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_unsigned_int_scalar() {
                                return Err(ValidationError::NonUniformResultTypeInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode: Op::GroupNonUniformBallotBitCount,
                                    expected: "unsigned integer scalar",
                                });
                            }
                        }
                    }

                    // Value must be 4-component unsigned integer vector
                    if let Some(value_id) = inst.operands.get(4).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let Some(value_type_id) = value_inst.result_type {
                                    if let Ok(type_id) = TypeId::try_from(value_type_id) {
                                        let ty = get_type_structure(type_id, ctx.definitions);
                                        if !is_unsigned_int_vec4(&ty) {
                                            return Err(
                                                ValidationError::NonUniformBallotValueInvalid {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode: Op::GroupNonUniformBallotBitCount,
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // In Vulkan, group operation must be Reduce, InclusiveScan, or ExclusiveScan
                    if ctx.is_vulkan() {
                        if let Some(Operand::GroupOperation(group_op)) = inst.operands.get(3) {
                            if !matches!(
                                group_op,
                                GroupOperation::Reduce
                                    | GroupOperation::InclusiveScan
                                    | GroupOperation::ExclusiveScan
                            ) {
                                return Err(
                                    ValidationError::NonUniformBallotBitCountInvalidGroupOp {
                                        function: func_id,
                                        block: block_id,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Validates OpGroupNonUniformBallotFindLSB and OpGroupNonUniformBallotFindMSB.
pub struct NonUniformBallotFindRule;

impl ValidationRule for NonUniformBallotFindRule {
    fn name(&self) -> &'static str {
        "non-uniform-ballot-find"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                    if opcode != Op::GroupNonUniformBallotFindLSB
                        && opcode != Op::GroupNonUniformBallotFindMSB
                    {
                        continue;
                    }

                    // Result must be unsigned integer scalar
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);
                            if !ty.is_unsigned_int_scalar() {
                                return Err(ValidationError::NonUniformResultTypeInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "unsigned integer scalar",
                                });
                            }
                        }
                    }

                    // Value must be 4-component unsigned integer vector
                    if let Some(value_id) = inst.operands.get(3).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let Some(value_type_id) = value_inst.result_type {
                                    if let Ok(type_id) = TypeId::try_from(value_type_id) {
                                        let ty = get_type_structure(type_id, ctx.definitions);
                                        if !is_unsigned_int_vec4(&ty) {
                                            return Err(
                                                ValidationError::NonUniformBallotValueInvalid {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode,
                                                },
                                            );
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

/// Validates non-uniform arithmetic operations.
pub struct NonUniformArithmeticRule;

impl ValidationRule for NonUniformArithmeticRule {
    fn name(&self) -> &'static str {
        "non-uniform-arithmetic"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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

                    // Check if this is an arithmetic non-uniform op
                    let is_unsigned = matches!(
                        opcode,
                        Op::GroupNonUniformUMin | Op::GroupNonUniformUMax
                    );
                    let is_float = matches!(
                        opcode,
                        Op::GroupNonUniformFAdd
                            | Op::GroupNonUniformFMul
                            | Op::GroupNonUniformFMin
                            | Op::GroupNonUniformFMax
                    );
                    let is_bool = matches!(
                        opcode,
                        Op::GroupNonUniformLogicalAnd
                            | Op::GroupNonUniformLogicalOr
                            | Op::GroupNonUniformLogicalXor
                    );
                    let is_signed_or_bitwise = matches!(
                        opcode,
                        Op::GroupNonUniformIAdd
                            | Op::GroupNonUniformIMul
                            | Op::GroupNonUniformSMin
                            | Op::GroupNonUniformSMax
                            | Op::GroupNonUniformBitwiseAnd
                            | Op::GroupNonUniformBitwiseOr
                            | Op::GroupNonUniformBitwiseXor
                    );

                    if !is_unsigned && !is_float && !is_bool && !is_signed_or_bitwise {
                        continue;
                    }

                    // Validate result type
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(type_id) = TypeId::try_from(result_type_id) {
                            let ty = get_type_structure(type_id, ctx.definitions);

                            if is_float && !ty.is_float_scalar_or_vector() {
                                return Err(ValidationError::NonUniformResultTypeInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "floating-point scalar or vector",
                                });
                            } else if is_bool && !ty.is_bool_scalar_or_vector() {
                                return Err(ValidationError::NonUniformResultTypeInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "boolean scalar or vector",
                                });
                            } else if is_unsigned && !ty.is_unsigned_int_scalar_or_vector() {
                                return Err(ValidationError::NonUniformResultTypeInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "unsigned integer scalar or vector",
                                });
                            } else if is_signed_or_bitwise && !ty.is_int_scalar_or_vector() {
                                return Err(ValidationError::NonUniformResultTypeInvalid {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                    expected: "integer scalar or vector",
                                });
                            }
                        }
                    }

                    // Value type must match result type (operand index 4)
                    if let Some(value_id) = inst.operands.get(4).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let (Some(result_type_id), Some(value_type_id)) =
                                    (inst.result_type, value_inst.result_type)
                                {
                                    if result_type_id != value_type_id {
                                        return Err(ValidationError::NonUniformValueTypeMismatch {
                                            function: func_id,
                                            block: block_id,
                                            opcode,
                                        });
                                    }
                                }
                            }
                        }
                    }

                    // Check group operation and ClusterSize/Ballot operand
                    if let Some(Operand::GroupOperation(group_op)) = inst.operands.get(3) {
                        let is_clustered = *group_op == GroupOperation::ClusteredReduce;
                        let is_partitioned = matches!(
                            group_op,
                            GroupOperation::PartitionedReduceNV
                                | GroupOperation::PartitionedInclusiveScanNV
                                | GroupOperation::PartitionedExclusiveScanNV
                        );

                        // Check for required operand 5
                        if inst.operands.len() <= 5 {
                            if is_clustered {
                                return Err(ValidationError::NonUniformClusterSizeRequired {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                });
                            } else if is_partitioned {
                                return Err(ValidationError::NonUniformBallotRequired {
                                    function: func_id,
                                    block: block_id,
                                    opcode,
                                });
                            }
                        } else if let Some(operand_id) = inst.operands.get(5).and_then(id_ref) {
                            if let Ok(operand_result) = ResultId::try_from(operand_id) {
                                if let Some(operand_inst) = ctx.definitions.get(&operand_result) {
                                    if is_partitioned {
                                        // Ballot must be 4-component integer vector
                                        if let Some(operand_type_raw) = operand_inst.result_type {
                                            if let Ok(operand_type_id) =
                                                TypeId::try_from(operand_type_raw)
                                            {
                                                let operand_ty = get_type_structure(
                                                    operand_type_id,
                                                    ctx.definitions,
                                                );
                                                if !is_int_vec4(&operand_ty) {
                                                    return Err(
                                                        ValidationError::NonUniformPartitionedBallotInvalid {
                                                            function: func_id,
                                                            block: block_id,
                                                            opcode,
                                                        },
                                                    );
                                                }
                                            }
                                        }
                                    } else {
                                        // ClusterSize must be unsigned integer scalar constant
                                        if let Some(operand_type_raw) = operand_inst.result_type {
                                            if let Ok(operand_type_id) =
                                                TypeId::try_from(operand_type_raw)
                                            {
                                                let operand_ty = get_type_structure(
                                                    operand_type_id,
                                                    ctx.definitions,
                                                );
                                                if !operand_ty.is_unsigned_int_scalar() {
                                                    return Err(
                                                        ValidationError::NonUniformClusterSizeInvalid {
                                                            function: func_id,
                                                            block: block_id,
                                                            opcode,
                                                        },
                                                    );
                                                }
                                            }
                                        }

                                        if !is_constant_opcode(operand_inst.class.opcode) {
                                            return Err(
                                                ValidationError::NonUniformClusterSizeInvalid {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode,
                                                },
                                            );
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

/// Validates OpGroupNonUniformRotateKHR.
pub struct NonUniformRotateRule;

impl ValidationRule for NonUniformRotateRule {
    fn name(&self) -> &'static str {
        "non-uniform-rotate"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(|id| {
                Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
            });

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(|id| {
                    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
                });

                for inst in &block.instructions {
                    if inst.class.opcode != Op::GroupNonUniformRotateKHR {
                        continue;
                    }

                    // Result must be scalar or vector of int/float/bool
                    let result_type = inst.result_type.and_then(|id| TypeId::try_from(id).ok());
                    if let Some(type_id) = result_type {
                        let ty = get_type_structure(type_id, ctx.definitions);
                        if !is_valid_value_type(&ty) {
                            return Err(ValidationError::NonUniformResultTypeInvalid {
                                function: func_id,
                                block: block_id,
                                opcode: Op::GroupNonUniformRotateKHR,
                                expected: "scalar or vector of floating-point, integer or boolean type",
                            });
                        }
                    }

                    // Value type must match result type
                    if let Some(value_id) = inst.operands.get(3).and_then(id_ref) {
                        if let Ok(value_result_id) = ResultId::try_from(value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let (Some(result_type_id), Some(value_type_id)) =
                                    (inst.result_type, value_inst.result_type)
                                {
                                    if result_type_id != value_type_id {
                                        return Err(ValidationError::NonUniformValueTypeMismatch {
                                            function: func_id,
                                            block: block_id,
                                            opcode: Op::GroupNonUniformRotateKHR,
                                        });
                                    }
                                }
                            }
                        }
                    }

                    // Delta must be unsigned integer scalar
                    if let Some(delta_id) = inst.operands.get(4).and_then(id_ref) {
                        if let Ok(delta_result) = ResultId::try_from(delta_id) {
                            if let Some(delta_inst) = ctx.definitions.get(&delta_result) {
                                if let Some(delta_type_raw) = delta_inst.result_type {
                                    if let Ok(delta_type_id) = TypeId::try_from(delta_type_raw) {
                                        let delta_ty =
                                            get_type_structure(delta_type_id, ctx.definitions);
                                        if !delta_ty.is_unsigned_int_scalar() {
                                            return Err(
                                                ValidationError::NonUniformIdMustBeUnsignedInt {
                                                    function: func_id,
                                                    block: block_id,
                                                    opcode: Op::GroupNonUniformRotateKHR,
                                                    operand_name: "Delta",
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Optional ClusterSize must be unsigned integer scalar constant
                    if inst.operands.len() > 5 {
                        if let Some(cluster_id) = inst.operands.get(5).and_then(id_ref) {
                            if let Ok(cluster_result) = ResultId::try_from(cluster_id) {
                                if let Some(cluster_inst) = ctx.definitions.get(&cluster_result) {
                                    if let Some(cluster_type_raw) = cluster_inst.result_type {
                                        if let Ok(cluster_type_id) =
                                            TypeId::try_from(cluster_type_raw)
                                        {
                                            let cluster_ty =
                                                get_type_structure(cluster_type_id, ctx.definitions);
                                            if !cluster_ty.is_unsigned_int_scalar() {
                                                return Err(
                                                    ValidationError::NonUniformClusterSizeInvalid {
                                                        function: func_id,
                                                        block: block_id,
                                                        opcode: Op::GroupNonUniformRotateKHR,
                                                    },
                                                );
                                            }
                                        }
                                    }

                                    if !is_constant_opcode(cluster_inst.class.opcode) {
                                        return Err(
                                            ValidationError::NonUniformClusterSizeInvalid {
                                                function: func_id,
                                                block: block_id,
                                                opcode: Op::GroupNonUniformRotateKHR,
                                            },
                                        );
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

/// Returns all non-uniform validation rules.
pub fn all_non_uniform_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(NonUniformElectRule),
        Box::new(NonUniformAnyAllRule),
        Box::new(NonUniformAllEqualRule),
        Box::new(NonUniformBroadcastShuffleRule),
        Box::new(NonUniformBroadcastFirstRule),
        Box::new(NonUniformBallotRule),
        Box::new(NonUniformInverseBallotRule),
        Box::new(NonUniformBallotBitExtractRule),
        Box::new(NonUniformBallotBitCountRule),
        Box::new(NonUniformBallotFindRule),
        Box::new(NonUniformArithmeticRule),
        Box::new(NonUniformRotateRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::types::{BitWidth, ScalarKind, VectorSize};

    #[test]
    fn test_is_non_uniform_group_operation() {
        assert!(is_non_uniform_group_operation(Op::GroupNonUniformElect));
        assert!(is_non_uniform_group_operation(Op::GroupNonUniformBallot));
        assert!(is_non_uniform_group_operation(Op::GroupNonUniformFAdd));
        assert!(is_non_uniform_group_operation(Op::GroupNonUniformRotateKHR));
        assert!(is_non_uniform_group_operation(Op::GroupNonUniformQuadAllKHR));
        assert!(is_non_uniform_group_operation(Op::GroupNonUniformQuadAnyKHR));
        assert!(!is_non_uniform_group_operation(Op::IAdd));
        assert!(!is_non_uniform_group_operation(Op::Load));
    }

    #[test]
    fn test_is_valid_value_type() {
        // Float scalar
        let float_scalar = TypeStructure::Scalar(ScalarKind::Float(BitWidth::BITS_32));
        assert!(is_valid_value_type(&float_scalar));

        // Int scalar
        let int_scalar = TypeStructure::Scalar(ScalarKind::SignedInt(BitWidth::BITS_32));
        assert!(is_valid_value_type(&int_scalar));

        // Bool scalar
        let bool_scalar = TypeStructure::Scalar(ScalarKind::Bool);
        assert!(is_valid_value_type(&bool_scalar));

        // Float vector
        let float_vec = TypeStructure::Vector {
            component: ScalarKind::Float(BitWidth::BITS_32),
            size: VectorSize::VEC4,
        };
        assert!(is_valid_value_type(&float_vec));

        // Invalid: void
        let void_type = TypeStructure::Void;
        assert!(!is_valid_value_type(&void_type));

        // Invalid: struct
        let struct_type = TypeStructure::Struct { members: vec![] };
        assert!(!is_valid_value_type(&struct_type));
    }

    #[test]
    fn test_is_unsigned_int_vec4() {
        // Valid uvec4
        let uvec4 = TypeStructure::Vector {
            component: ScalarKind::UnsignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC4,
        };
        assert!(is_unsigned_int_vec4(&uvec4));

        // Invalid: signed ivec4
        let ivec4 = TypeStructure::Vector {
            component: ScalarKind::SignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC4,
        };
        assert!(!is_unsigned_int_vec4(&ivec4));

        // Invalid: uvec3
        let uvec3 = TypeStructure::Vector {
            component: ScalarKind::UnsignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC3,
        };
        assert!(!is_unsigned_int_vec4(&uvec3));

        // Invalid: scalar
        let uint_scalar = TypeStructure::Scalar(ScalarKind::UnsignedInt(BitWidth::BITS_32));
        assert!(!is_unsigned_int_vec4(&uint_scalar));
    }

    #[test]
    fn test_is_int_vec4() {
        // Valid uvec4
        let uvec4 = TypeStructure::Vector {
            component: ScalarKind::UnsignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC4,
        };
        assert!(is_int_vec4(&uvec4));

        // Valid ivec4
        let ivec4 = TypeStructure::Vector {
            component: ScalarKind::SignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC4,
        };
        assert!(is_int_vec4(&ivec4));

        // Invalid: fvec4
        let fvec4 = TypeStructure::Vector {
            component: ScalarKind::Float(BitWidth::BITS_32),
            size: VectorSize::VEC4,
        };
        assert!(!is_int_vec4(&fvec4));

        // Invalid: ivec3
        let ivec3 = TypeStructure::Vector {
            component: ScalarKind::SignedInt(BitWidth::BITS_32),
            size: VectorSize::VEC3,
        };
        assert!(!is_int_vec4(&ivec3));
    }

    #[test]
    fn test_get_broadcast_shuffle_operand_name() {
        assert_eq!(get_broadcast_shuffle_operand_name(Op::GroupNonUniformBroadcast), "Id");
        assert_eq!(get_broadcast_shuffle_operand_name(Op::GroupNonUniformShuffle), "Id");
        assert_eq!(get_broadcast_shuffle_operand_name(Op::GroupNonUniformShuffleXor), "Mask");
        assert_eq!(get_broadcast_shuffle_operand_name(Op::GroupNonUniformQuadBroadcast), "Index");
        assert_eq!(get_broadcast_shuffle_operand_name(Op::GroupNonUniformQuadSwap), "Direction");
        assert_eq!(get_broadcast_shuffle_operand_name(Op::GroupNonUniformShuffleUp), "Delta");
        assert_eq!(get_broadcast_shuffle_operand_name(Op::GroupNonUniformShuffleDown), "Delta");
    }

    #[test]
    fn test_all_non_uniform_rules() {
        let rules = all_non_uniform_rules();
        // Ensure we have all 12 rules
        assert_eq!(rules.len(), 12);

        // Check rule names
        let names: Vec<_> = rules.iter().map(|r| r.name()).collect();
        assert!(names.contains(&"non-uniform-elect"));
        assert!(names.contains(&"non-uniform-any-all"));
        assert!(names.contains(&"non-uniform-all-equal"));
        assert!(names.contains(&"non-uniform-broadcast-shuffle"));
        assert!(names.contains(&"non-uniform-broadcast-first"));
        assert!(names.contains(&"non-uniform-ballot"));
        assert!(names.contains(&"non-uniform-inverse-ballot"));
        assert!(names.contains(&"non-uniform-ballot-bit-extract"));
        assert!(names.contains(&"non-uniform-ballot-bit-count"));
        assert!(names.contains(&"non-uniform-ballot-find"));
        assert!(names.contains(&"non-uniform-arithmetic"));
        assert!(names.contains(&"non-uniform-rotate"));
    }
}
