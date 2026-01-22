//! Bitwise instruction validation rules.
//!
//! This module validates SPIR-V bitwise instructions including:
//!
//! - Shift operations (ShiftRightLogical, ShiftRightArithmetic, ShiftLeftLogical)
//! - Bitwise operations (BitwiseOr, BitwiseXor, BitwiseAnd, Not)
//! - Bit field operations (BitFieldInsert, BitFieldSExtract, BitFieldUExtract, BitReverse)
//! - Bit counting (BitCount)

use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::Id;
use crate::validation::ValidationResult;

// ============================================================================
// Shift Operations Rule
// ============================================================================

/// Validates shift operations.
///
/// Ensures that:
/// - Result type is an int scalar or vector
/// - Base operand has same dimension and bit width as result
/// - Shift operand has same dimension as result
pub struct ShiftOperationsRule;

impl ValidationRule for ShiftOperationsRule {
    fn name(&self) -> &'static str {
        "shift-operations"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let shift_ops = [
            Op::ShiftRightLogical,
            Op::ShiftRightArithmetic,
            Op::ShiftLeftLogical,
        ];

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !shift_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result type must be int scalar or vector
                    if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::BitwiseResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "int scalar or vector",
                            }
                            .into());
                        }
                    }

                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);
                    let result_width = resolver.get_bit_width(result_type_id, ctx.definitions);

                    // Check base operand (operand 0)
                    if let Some(rspirv::dr::Operand::IdRef(base_id)) = inst.operands.first() {
                        let base_inst = crate::validation::types::ResultId::try_from(*base_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(base_inst) = base_inst {
                            if let Some(base_type_id) = base_inst.result_type {
                                if !resolver.is_int_scalar_or_vector(base_type_id, ctx.definitions)
                                {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::BitwiseOperandTypeMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            operand_index: 0,
                                            result_type,
                                            expected: "int scalar or vector",
                                        }
                                        .into());
                                    }
                                }

                                let base_dim =
                                    resolver.get_dimension(base_type_id, ctx.definitions);
                                let base_width =
                                    resolver.get_bit_width(base_type_id, ctx.definitions);

                                if base_dim != result_dim {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::BitwiseDimensionMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            operand_name: "Base",
                                            result_type,
                                        }
                                        .into());
                                    }
                                }

                                if base_width != result_width {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::BitwiseBitWidthMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            operand_name: "Base",
                                            result_type,
                                        }
                                        .into());
                                    }
                                }
                            }
                        }
                    }

                    // Check shift operand (operand 1)
                    if let Some(rspirv::dr::Operand::IdRef(shift_id)) = inst.operands.get(1) {
                        let shift_inst = crate::validation::types::ResultId::try_from(*shift_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(shift_inst) = shift_inst {
                            if let Some(shift_type_id) = shift_inst.result_type {
                                if !resolver.is_int_scalar_or_vector(shift_type_id, ctx.definitions)
                                {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::BitwiseOperandTypeMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            operand_index: 1,
                                            result_type,
                                            expected: "int scalar or vector",
                                        }
                                        .into());
                                    }
                                }

                                let shift_dim =
                                    resolver.get_dimension(shift_type_id, ctx.definitions);

                                if shift_dim != result_dim {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::BitwiseDimensionMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            operand_name: "Shift",
                                            result_type,
                                        }
                                        .into());
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

// ============================================================================
// Bitwise Logic Operations Rule
// ============================================================================

/// Validates bitwise logic operations.
///
/// Ensures that:
/// - Result type is an int scalar or vector
/// - All operands have same dimension and bit width as result
pub struct BitwiseLogicRule;

impl ValidationRule for BitwiseLogicRule {
    fn name(&self) -> &'static str {
        "bitwise-logic"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let logic_ops = [Op::BitwiseOr, Op::BitwiseXor, Op::BitwiseAnd, Op::Not];

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !logic_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result type must be int scalar or vector
                    if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::BitwiseResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "int scalar or vector",
                            }
                            .into());
                        }
                    }

                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);
                    let result_width = resolver.get_bit_width(result_type_id, ctx.definitions);

                    // Check all operands
                    for (idx, operand) in inst.operands.iter().enumerate() {
                        let operand_id = match operand {
                            rspirv::dr::Operand::IdRef(id) => *id,
                            _ => continue,
                        };

                        let operand_inst = crate::validation::types::ResultId::try_from(operand_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        let Some(operand_inst) = operand_inst else {
                            continue;
                        };

                        let Some(operand_type_id) = operand_inst.result_type else {
                            continue;
                        };

                        if !resolver.is_int_scalar_or_vector(operand_type_id, ctx.definitions) {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::BitwiseOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: idx,
                                    result_type,
                                    expected: "int scalar or vector",
                                }
                                .into());
                            }
                        }

                        let operand_dim = resolver.get_dimension(operand_type_id, ctx.definitions);
                        let operand_width =
                            resolver.get_bit_width(operand_type_id, ctx.definitions);

                        if operand_dim != result_dim {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::BitwiseDimensionMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_name: "operand",
                                    result_type,
                                }
                                .into());
                            }
                        }

                        if operand_width != result_width {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::BitwiseBitWidthMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_name: "operand",
                                    result_type,
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

// ============================================================================
// Bit Field Operations Rule
// ============================================================================

/// Validates bit field operations.
///
/// Ensures that:
/// - Result type is an int scalar or vector
/// - Base operand matches result type
/// - Offset and Count are int scalars
pub struct BitFieldRule;

impl ValidationRule for BitFieldRule {
    fn name(&self) -> &'static str {
        "bit-field"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    match inst.class.opcode {
                        Op::BitFieldInsert => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result type must be int scalar or vector
                            if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::BitwiseResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "int scalar or vector",
                                    }
                                    .into());
                                }
                            }

                            // Check base (operand 0) matches result type
                            self.validate_base_matches_result(
                                inst,
                                0,
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;

                            // Check insert (operand 1) matches result type
                            self.validate_base_matches_result(
                                inst,
                                1,
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;

                            // Check offset (operand 2) is int scalar
                            self.validate_int_scalar_operand(
                                inst,
                                2,
                                "Offset",
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;

                            // Check count (operand 3) is int scalar
                            self.validate_int_scalar_operand(
                                inst,
                                3,
                                "Count",
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;
                        }
                        Op::BitFieldSExtract | Op::BitFieldUExtract => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result type must be int scalar or vector
                            if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::BitwiseResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "int scalar or vector",
                                    }
                                    .into());
                                }
                            }

                            // Check base (operand 0) matches result type
                            self.validate_base_matches_result(
                                inst,
                                0,
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;

                            // Check offset (operand 1) is int scalar
                            self.validate_int_scalar_operand(
                                inst,
                                1,
                                "Offset",
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;

                            // Check count (operand 2) is int scalar
                            self.validate_int_scalar_operand(
                                inst,
                                2,
                                "Count",
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;
                        }
                        Op::BitReverse => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result type must be int scalar or vector
                            if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::BitwiseResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "int scalar or vector",
                                    }
                                    .into());
                                }
                            }

                            // Check base (operand 0) matches result type
                            self.validate_base_matches_result(
                                inst,
                                0,
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;
                        }
                        _ => continue,
                    }
                }
            }
        }

        Ok(())
    }
}

impl BitFieldRule {
    fn validate_base_matches_result(
        &self,
        inst: &rspirv::dr::Instruction,
        operand_idx: usize,
        result_type_id: u32,
        resolver: &DefaultTypeResolver,
        ctx: &ValidationContext<'_>,
        function_id: Option<Id>,
        block_id: Option<Id>,
    ) -> ValidationResult {
        if let Some(rspirv::dr::Operand::IdRef(operand_id)) = inst.operands.get(operand_idx) {
            let operand_inst = crate::validation::types::ResultId::try_from(*operand_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            if let Some(operand_inst) = operand_inst {
                if let Some(operand_type_id) = operand_inst.result_type {
                    if operand_type_id != result_type_id {
                        // Check dimensions and bit widths match
                        let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);
                        let result_width = resolver.get_bit_width(result_type_id, ctx.definitions);
                        let operand_dim = resolver.get_dimension(operand_type_id, ctx.definitions);
                        let operand_width =
                            resolver.get_bit_width(operand_type_id, ctx.definitions);

                        if operand_dim != result_dim || operand_width != result_width {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::BitwiseOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: operand_idx,
                                    result_type,
                                    expected: "same type as result",
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

    fn validate_int_scalar_operand(
        &self,
        inst: &rspirv::dr::Instruction,
        operand_idx: usize,
        operand_name: &'static str,
        result_type_id: u32,
        resolver: &DefaultTypeResolver,
        ctx: &ValidationContext<'_>,
        function_id: Option<Id>,
        block_id: Option<Id>,
    ) -> ValidationResult {
        if let Some(rspirv::dr::Operand::IdRef(operand_id)) = inst.operands.get(operand_idx) {
            let operand_inst = crate::validation::types::ResultId::try_from(*operand_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            if let Some(operand_inst) = operand_inst {
                if let Some(operand_type_id) = operand_inst.result_type {
                    if !resolver.is_int_scalar(operand_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::BitwiseOperandTypeMismatch {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                operand_index: operand_idx,
                                result_type,
                                expected: operand_name,
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

// ============================================================================
// Bit Count Rule
// ============================================================================

/// Validates the OpBitCount instruction.
///
/// Ensures that:
/// - Result type is an int scalar or vector
/// - Base operand is an int scalar or vector
/// - Base and result have the same dimension (but can differ in bit width)
pub struct BitCountRule;

impl ValidationRule for BitCountRule {
    fn name(&self) -> &'static str {
        "bit-count"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if inst.class.opcode != Op::BitCount {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result type must be int scalar or vector
                    if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::BitwiseResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "int scalar or vector",
                            }
                            .into());
                        }
                    }

                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);

                    // Check base operand (operand 0)
                    if let Some(rspirv::dr::Operand::IdRef(base_id)) = inst.operands.first() {
                        let base_inst = crate::validation::types::ResultId::try_from(*base_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(base_inst) = base_inst {
                            if let Some(base_type_id) = base_inst.result_type {
                                if !resolver.is_int_scalar_or_vector(base_type_id, ctx.definitions)
                                {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::BitwiseOperandTypeMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            operand_index: 0,
                                            result_type,
                                            expected: "int scalar or vector",
                                        }
                                        .into());
                                    }
                                }

                                let base_dim =
                                    resolver.get_dimension(base_type_id, ctx.definitions);

                                // For BitCount, only dimension must match (bit width can differ)
                                if base_dim != result_dim {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::BitwiseDimensionMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            operand_name: "Base",
                                            result_type,
                                        }
                                        .into());
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

// ============================================================================
// All bitwise rules
// ============================================================================

/// Returns all bitwise validation rules.
pub fn all_bitwise_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &ShiftOperationsRule,
        &BitwiseLogicRule,
        &BitFieldRule,
        &BitCountRule,
    ]
}
