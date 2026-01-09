//! Conversion instruction validation rules.
//!
//! This module validates SPIR-V conversion instructions including:
//!
//! - Float to integer conversions (ConvertFToU, ConvertFToS)
//! - Integer to float conversions (ConvertSToF, ConvertUToF)
//! - Integer conversions (UConvert, SConvert)
//! - Float conversions (FConvert, QuantizeToF16)
//! - Bitcast (OpBitcast)
//! - Pointer conversions (ConvertPtrToU, ConvertUToPtr, etc.)

use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeInstructionExt, TypeResolver};
use crate::validation::types::Id;

// ============================================================================
// Float to Integer Conversion Rule
// ============================================================================

/// Validates float to integer conversion operations.
///
/// Ensures that:
/// - ConvertFToU: Result is unsigned int, input is float, same dimension
/// - ConvertFToS: Result is signed int, input is float, same dimension
pub struct FloatToIntConversionRule;

impl ValidationRule for FloatToIntConversionRule {
    fn name(&self) -> &'static str {
        "float-to-int-conversion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        Op::ConvertFToU => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be unsigned int scalar or vector
                            if !resolver
                                .is_unsigned_int_scalar_or_vector(result_type_id, ctx.definitions)
                            {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id)
                                        .ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "unsigned int scalar or vector",
                                    });
                                }
                            }

                            // Check input is float scalar or vector with same dimension
                            self.validate_float_input_same_dim(
                                inst,
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;
                        }
                        Op::ConvertFToS => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be int scalar or vector
                            if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id)
                                        .ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "int scalar or vector",
                                    });
                                }
                            }

                            // Check input is float scalar or vector with same dimension
                            self.validate_float_input_same_dim(
                                inst,
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

impl FloatToIntConversionRule {
    fn validate_float_input_same_dim(
        &self,
        inst: &rspirv::dr::Instruction,
        result_type_id: u32,
        resolver: &DefaultTypeResolver,
        ctx: &ValidationContext<'_>,
        function_id: Option<Id>,
        block_id: Option<Id>,
    ) -> Result<(), ValidationError> {
        if let Some(rspirv::dr::Operand::IdRef(input_id)) = inst.operands.first() {
            let input_inst = crate::validation::types::ResultId::try_from(*input_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            if let Some(input_inst) = input_inst {
                if let Some(input_type_id) = input_inst.result_type {
                    if !resolver.is_float_scalar_or_vector(input_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ConversionInputTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "float scalar or vector",
                            });
                        }
                    }

                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);
                    let input_dim = resolver.get_dimension(input_type_id, ctx.definitions);

                    if result_dim != input_dim {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ConversionDimensionMismatch {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
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
// Integer to Float Conversion Rule
// ============================================================================

/// Validates integer to float conversion operations.
///
/// Ensures that:
/// - ConvertSToF/ConvertUToF: Result is float, input is int, same dimension
pub struct IntToFloatConversionRule;

impl ValidationRule for IntToFloatConversionRule {
    fn name(&self) -> &'static str {
        "int-to-float-conversion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                    if inst.class.opcode != Op::ConvertSToF
                        && inst.class.opcode != Op::ConvertUToF
                    {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be float scalar or vector
                    if !resolver.is_float_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ConversionResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "float scalar or vector",
                            });
                        }
                    }

                    // Check input is int scalar or vector with same dimension
                    if let Some(rspirv::dr::Operand::IdRef(input_id)) = inst.operands.first() {
                        let input_inst = crate::validation::types::ResultId::try_from(*input_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(input_inst) = input_inst {
                            if let Some(input_type_id) = input_inst.result_type {
                                if !resolver.is_int_scalar_or_vector(input_type_id, ctx.definitions)
                                {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::ConversionInputTypeInvalid {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
                                            expected: "int scalar or vector",
                                        });
                                    }
                                }

                                let result_dim =
                                    resolver.get_dimension(result_type_id, ctx.definitions);
                                let input_dim =
                                    resolver.get_dimension(input_type_id, ctx.definitions);

                                if result_dim != input_dim {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::ConversionDimensionMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
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

// ============================================================================
// Integer Width Conversion Rule
// ============================================================================

/// Validates integer width conversion operations.
///
/// Ensures that:
/// - UConvert: Result is unsigned int, input is int, same dimension, different bit width
/// - SConvert: Result is int, input is int, same dimension, different bit width
pub struct IntWidthConversionRule;

impl ValidationRule for IntWidthConversionRule {
    fn name(&self) -> &'static str {
        "int-width-conversion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        Op::UConvert => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be unsigned int scalar or vector
                            if !resolver
                                .is_unsigned_int_scalar_or_vector(result_type_id, ctx.definitions)
                            {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id)
                                        .ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "unsigned int scalar or vector",
                                    });
                                }
                            }

                            self.validate_int_input_different_width(
                                inst,
                                result_type_id,
                                &resolver,
                                ctx,
                                function_id,
                                block_id,
                            )?;
                        }
                        Op::SConvert => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be int scalar or vector
                            if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id)
                                        .ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "int scalar or vector",
                                    });
                                }
                            }

                            self.validate_int_input_different_width(
                                inst,
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

impl IntWidthConversionRule {
    fn validate_int_input_different_width(
        &self,
        inst: &rspirv::dr::Instruction,
        result_type_id: u32,
        resolver: &DefaultTypeResolver,
        ctx: &ValidationContext<'_>,
        function_id: Option<Id>,
        block_id: Option<Id>,
    ) -> Result<(), ValidationError> {
        if let Some(rspirv::dr::Operand::IdRef(input_id)) = inst.operands.first() {
            let input_inst = crate::validation::types::ResultId::try_from(*input_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            if let Some(input_inst) = input_inst {
                if let Some(input_type_id) = input_inst.result_type {
                    if !resolver.is_int_scalar_or_vector(input_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ConversionInputTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "int scalar or vector",
                            });
                        }
                    }

                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);
                    let input_dim = resolver.get_dimension(input_type_id, ctx.definitions);

                    if result_dim != input_dim {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ConversionDimensionMismatch {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                            });
                        }
                    }

                    let result_width = resolver.get_bit_width(result_type_id, ctx.definitions);
                    let input_width = resolver.get_bit_width(input_type_id, ctx.definitions);

                    if result_width == input_width {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ConversionSameBitWidth {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
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
// Float Width Conversion Rule
// ============================================================================

/// Validates float width conversion operations.
///
/// Ensures that:
/// - FConvert: Result is float, input is float, same dimension, different type
/// - QuantizeToF16: Result is 32-bit float, input matches result
pub struct FloatWidthConversionRule;

impl ValidationRule for FloatWidthConversionRule {
    fn name(&self) -> &'static str {
        "float-width-conversion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        Op::FConvert => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be float scalar or vector
                            if !resolver.is_float_scalar_or_vector(result_type_id, ctx.definitions)
                            {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id)
                                        .ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "float scalar or vector",
                                    });
                                }
                            }

                            // Check input is float scalar or vector with same dimension
                            if let Some(rspirv::dr::Operand::IdRef(input_id)) =
                                inst.operands.first()
                            {
                                let input_inst =
                                    crate::validation::types::ResultId::try_from(*input_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(input_inst) = input_inst {
                                    if let Some(input_type_id) = input_inst.result_type {
                                        if !resolver.is_float_scalar_or_vector(
                                            input_type_id,
                                            ctx.definitions,
                                        ) {
                                            if let (Some(func), Some(block), Some(result_type)) = (
                                                function_id,
                                                block_id,
                                                crate::validation::types::TypeId::try_from(
                                                    result_type_id,
                                                )
                                                .ok(),
                                            ) {
                                                return Err(
                                                    ValidationError::ConversionInputTypeInvalid {
                                                        function: func,
                                                        block,
                                                        opcode: inst.class.opcode,
                                                        result_type,
                                                        expected: "float scalar or vector",
                                                    },
                                                );
                                            }
                                        }

                                        let result_dim =
                                            resolver.get_dimension(result_type_id, ctx.definitions);
                                        let input_dim =
                                            resolver.get_dimension(input_type_id, ctx.definitions);

                                        if result_dim != input_dim {
                                            if let (Some(func), Some(block), Some(result_type)) = (
                                                function_id,
                                                block_id,
                                                crate::validation::types::TypeId::try_from(
                                                    result_type_id,
                                                )
                                                .ok(),
                                            ) {
                                                return Err(
                                                    ValidationError::ConversionDimensionMismatch {
                                                        function: func,
                                                        block,
                                                        opcode: inst.class.opcode,
                                                        result_type,
                                                    },
                                                );
                                            }
                                        }

                                        // For FConvert, component types must differ
                                        // We check bit widths as a proxy for different types
                                        let result_width =
                                            resolver.get_bit_width(result_type_id, ctx.definitions);
                                        let input_width =
                                            resolver.get_bit_width(input_type_id, ctx.definitions);

                                        if result_width == input_width {
                                            if let (Some(func), Some(block), Some(result_type)) = (
                                                function_id,
                                                block_id,
                                                crate::validation::types::TypeId::try_from(
                                                    result_type_id,
                                                )
                                                .ok(),
                                            ) {
                                                return Err(
                                                    ValidationError::ConversionSameBitWidth {
                                                        function: func,
                                                        block,
                                                        opcode: inst.class.opcode,
                                                        result_type,
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Op::QuantizeToF16 => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be 32-bit float scalar or vector
                            if !resolver.is_float_scalar_or_vector(result_type_id, ctx.definitions)
                            {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id)
                                        .ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "32-bit float scalar or vector",
                                    });
                                }
                            }

                            let result_width =
                                resolver.get_bit_width(result_type_id, ctx.definitions);
                            if result_width != Some(32) {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id)
                                        .ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "32-bit float scalar or vector",
                                    });
                                }
                            }

                            // Input must match result type
                            if let Some(rspirv::dr::Operand::IdRef(input_id)) =
                                inst.operands.first()
                            {
                                let input_inst =
                                    crate::validation::types::ResultId::try_from(*input_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(input_inst) = input_inst {
                                    if let Some(input_type_id) = input_inst.result_type {
                                        if input_type_id != result_type_id {
                                            if let (Some(func), Some(block), Some(result_type)) = (
                                                function_id,
                                                block_id,
                                                crate::validation::types::TypeId::try_from(
                                                    result_type_id,
                                                )
                                                .ok(),
                                            ) {
                                                return Err(
                                                    ValidationError::ConversionInputTypeInvalid {
                                                        function: func,
                                                        block,
                                                        opcode: inst.class.opcode,
                                                        result_type,
                                                        expected: "same type as result",
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => continue,
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Bitcast Rule
// ============================================================================

/// Validates OpBitcast operations.
///
/// Ensures that:
/// - Result type and input are pointer/int/float scalar or vector
/// - Non-pointer operands have same total bit width
pub struct BitcastRule;

impl ValidationRule for BitcastRule {
    fn name(&self) -> &'static str {
        "bitcast"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                    if inst.class.opcode != Op::Bitcast {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    let result_type_inst =
                        crate::validation::types::ResultId::try_from(result_type_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                    let result_is_pointer = result_type_inst
                        .map(|i| i.is_pointer_type())
                        .unwrap_or(false);
                    let result_is_numeric = resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions)
                        || resolver.is_float_scalar_or_vector(result_type_id, ctx.definitions);

                    if !result_is_pointer && !result_is_numeric {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ConversionResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "pointer or int/float scalar or vector",
                            });
                        }
                    }

                    // Check input
                    if let Some(rspirv::dr::Operand::IdRef(input_id)) = inst.operands.first() {
                        let input_inst = crate::validation::types::ResultId::try_from(*input_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(input_inst) = input_inst {
                            if let Some(input_type_id) = input_inst.result_type {
                                let input_type_inst =
                                    crate::validation::types::ResultId::try_from(input_type_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                let input_is_pointer = input_type_inst
                                    .map(|i| i.is_pointer_type())
                                    .unwrap_or(false);
                                let input_is_numeric = resolver
                                    .is_int_scalar_or_vector(input_type_id, ctx.definitions)
                                    || resolver
                                        .is_float_scalar_or_vector(input_type_id, ctx.definitions);

                                if !input_is_pointer && !input_is_numeric {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::ConversionInputTypeInvalid {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
                                            expected: "pointer or int/float scalar or vector",
                                        });
                                    }
                                }

                                // Non-pointer operands must have same total bit width
                                if !result_is_pointer && !input_is_pointer {
                                    let result_width = resolver
                                        .get_bit_width(result_type_id, ctx.definitions)
                                        .unwrap_or(0);
                                    let result_dim =
                                        resolver.get_dimension(result_type_id, ctx.definitions);
                                    let result_total = result_width * result_dim;

                                    let input_width = resolver
                                        .get_bit_width(input_type_id, ctx.definitions)
                                        .unwrap_or(0);
                                    let input_dim =
                                        resolver.get_dimension(input_type_id, ctx.definitions);
                                    let input_total = input_width * input_dim;

                                    if result_total != input_total {
                                        if let (Some(func), Some(block), Some(result_type)) = (
                                            function_id,
                                            block_id,
                                            crate::validation::types::TypeId::try_from(
                                                result_type_id,
                                            )
                                            .ok(),
                                        ) {
                                            return Err(
                                                ValidationError::ConversionBitWidthMismatch {
                                                    function: func,
                                                    block,
                                                    opcode: inst.class.opcode,
                                                    result_type,
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

// ============================================================================
// Saturating Conversion Rule
// ============================================================================

/// Validates saturating conversion operations.
///
/// Ensures that:
/// - SatConvertSToU/SatConvertUToS: Result and input are int, same dimension
pub struct SaturatingConversionRule;

impl ValidationRule for SaturatingConversionRule {
    fn name(&self) -> &'static str {
        "saturating-conversion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                    if inst.class.opcode != Op::SatConvertSToU
                        && inst.class.opcode != Op::SatConvertUToS
                    {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be int scalar or vector
                    if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ConversionResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "int scalar or vector",
                            });
                        }
                    }

                    // Check input is int scalar or vector with same dimension
                    if let Some(rspirv::dr::Operand::IdRef(input_id)) = inst.operands.first() {
                        let input_inst = crate::validation::types::ResultId::try_from(*input_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(input_inst) = input_inst {
                            if let Some(input_type_id) = input_inst.result_type {
                                if !resolver.is_int_scalar_or_vector(input_type_id, ctx.definitions)
                                {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::ConversionInputTypeInvalid {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
                                            expected: "int scalar or vector",
                                        });
                                    }
                                }

                                let result_dim =
                                    resolver.get_dimension(result_type_id, ctx.definitions);
                                let input_dim =
                                    resolver.get_dimension(input_type_id, ctx.definitions);

                                if result_dim != input_dim {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::ConversionDimensionMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
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

// ============================================================================
// All conversion rules
// ============================================================================

/// Returns all conversion validation rules.
pub fn all_conversion_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &FloatToIntConversionRule,
        &IntToFloatConversionRule,
        &IntWidthConversionRule,
        &FloatWidthConversionRule,
        &BitcastRule,
        &SaturatingConversionRule,
    ]
}
