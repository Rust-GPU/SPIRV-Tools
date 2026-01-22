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
use crate::validation::ValidationResult;

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
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "unsigned int scalar or vector",
                                    }
                                    .into());
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
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "int scalar or vector",
                                    }
                                    .into());
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
    ) -> ValidationResult {
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
                            }
                            .into());
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
                    if inst.class.opcode != Op::ConvertSToF && inst.class.opcode != Op::ConvertUToF
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
                            }
                            .into());
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
                                        }
                                        .into());
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
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "unsigned int scalar or vector",
                                    }
                                    .into());
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
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "int scalar or vector",
                                    }
                                    .into());
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
    ) -> ValidationResult {
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
                            }
                            .into());
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
                            }
                            .into());
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
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "float scalar or vector",
                                    }
                                    .into());
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
                                                    }
                                                    .into(),
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
                                                    }
                                                    .into(),
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
                                                    }
                                                    .into(),
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
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::ConversionResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "32-bit float scalar or vector",
                                    }
                                    .into());
                                }
                            }

                            let result_width =
                                resolver.get_bit_width(result_type_id, ctx.definitions);
                            if result_width != Some(32) {
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
                                        expected: "32-bit float scalar or vector",
                                    }
                                    .into());
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
                                                    }
                                                    .into(),
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
                    let result_is_numeric = resolver
                        .is_int_scalar_or_vector(result_type_id, ctx.definitions)
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
                            }
                            .into());
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
                                        }
                                        .into());
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
                                                }
                                                .into(),
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
                            }
                            .into());
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
                                        }
                                        .into());
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
// Pointer Conversion Rule
// ============================================================================

/// Validates pointer conversion operations (ConvertPtrToU, ConvertUToPtr).
///
/// Ensures that:
/// - Logical addressing mode is not used (not supported)
/// - PhysicalStorageBuffer64 requires PhysicalStorageBuffer storage class
/// - Vulkan with PhysicalStorageBuffer64 requires 64-bit integers
pub struct PointerConversionRule;

impl ValidationRule for PointerConversionRule {
    fn name(&self) -> &'static str {
        "pointer-conversion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        use rspirv::spirv::{AddressingModel, StorageClass};

        let resolver = DefaultTypeResolver;

        // Get addressing model
        let addressing_model = ctx
            .module
            .memory_model
            .as_ref()
            .and_then(|inst| inst.operands.first())
            .and_then(|op| match op {
                rspirv::dr::Operand::AddressingModel(model) => Some(*model),
                _ => None,
            });

        let is_logical = matches!(addressing_model, Some(AddressingModel::Logical));
        let is_physical_storage_buffer_64 = matches!(
            addressing_model,
            Some(AddressingModel::PhysicalStorageBuffer64)
        );

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
                        Op::ConvertPtrToU => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be unsigned int scalar
                            if !resolver.is_unsigned_int_scalar(result_type_id, ctx.definitions) {
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
                                        expected: "unsigned int scalar",
                                    }
                                    .into());
                                }
                            }

                            // Logical addressing not supported
                            if is_logical {
                                if let (Some(func), Some(block)) = (function_id, block_id) {
                                    return Err(
                                        ValidationError::ConversionLogicalAddressingNotSupported {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                        }
                                        .into(),
                                    );
                                }
                            }

                            // Check input pointer
                            if let Some(rspirv::dr::Operand::IdRef(input_id)) =
                                inst.operands.first()
                            {
                                let input_inst =
                                    crate::validation::types::ResultId::try_from(*input_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(input_inst) = input_inst {
                                    if let Some(input_type_id) = input_inst.result_type {
                                        let input_type_inst =
                                            crate::validation::types::ResultId::try_from(
                                                input_type_id,
                                            )
                                            .ok()
                                            .and_then(|rid| ctx.definitions.get(&rid));

                                        // Input must be a pointer
                                        let is_pointer = input_type_inst
                                            .map(|i| i.is_pointer_type())
                                            .unwrap_or(false);

                                        if !is_pointer {
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
                                                        expected: "pointer",
                                                    }
                                                    .into(),
                                                );
                                            }
                                        }

                                        // PhysicalStorageBuffer64 requires PhysicalStorageBuffer storage class
                                        if is_physical_storage_buffer_64 {
                                            let storage_class = input_type_inst.and_then(|i| {
                                                i.operands.first().and_then(|op| match op {
                                                    rspirv::dr::Operand::StorageClass(sc) => {
                                                        Some(*sc)
                                                    }
                                                    _ => None,
                                                })
                                            });

                                            if storage_class
                                                != Some(StorageClass::PhysicalStorageBuffer)
                                            {
                                                if let (Some(func), Some(block)) =
                                                    (function_id, block_id)
                                                {
                                                    return Err(
                                                        ValidationError::ConversionInvalidStorageClass {
                                                            function: func,
                                                            block,
                                                            opcode: inst.class.opcode,
                                                            expected: "PhysicalStorageBuffer",
                                                        }.into(),
                        );
                                                }
                                            }

                                            // Vulkan requires 64-bit result
                                            if ctx.is_vulkan() {
                                                let result_width = resolver
                                                    .get_bit_width(result_type_id, ctx.definitions);
                                                if result_width != Some(64) {
                                                    if let (Some(func), Some(block)) =
                                                        (function_id, block_id)
                                                    {
                                                        return Err(
                                                            ValidationError::ConversionRequires64BitInteger {
                                                                function: func,
                                                                block,
                                                                opcode: inst.class.opcode,
                                                            }.into(),
                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Op::ConvertUToPtr => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be a pointer
                            let result_type_inst =
                                crate::validation::types::ResultId::try_from(result_type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid));

                            let is_pointer = result_type_inst
                                .map(|i| i.is_pointer_type())
                                .unwrap_or(false);

                            if !is_pointer {
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
                                        expected: "pointer",
                                    }
                                    .into());
                                }
                            }

                            // Logical addressing not supported
                            if is_logical {
                                if let (Some(func), Some(block)) = (function_id, block_id) {
                                    return Err(
                                        ValidationError::ConversionLogicalAddressingNotSupported {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                        }
                                        .into(),
                                    );
                                }
                            }

                            // Input must be int scalar
                            if let Some(rspirv::dr::Operand::IdRef(input_id)) =
                                inst.operands.first()
                            {
                                let input_inst =
                                    crate::validation::types::ResultId::try_from(*input_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(input_inst) = input_inst {
                                    if let Some(input_type_id) = input_inst.result_type {
                                        if !resolver.is_int_scalar(input_type_id, ctx.definitions) {
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
                                                        expected: "int scalar",
                                                    }
                                                    .into(),
                                                );
                                            }
                                        }

                                        // PhysicalStorageBuffer64 requires PhysicalStorageBuffer storage class
                                        if is_physical_storage_buffer_64 {
                                            let storage_class = result_type_inst.and_then(|i| {
                                                i.operands.first().and_then(|op| match op {
                                                    rspirv::dr::Operand::StorageClass(sc) => {
                                                        Some(*sc)
                                                    }
                                                    _ => None,
                                                })
                                            });

                                            if storage_class
                                                != Some(StorageClass::PhysicalStorageBuffer)
                                            {
                                                if let (Some(func), Some(block)) =
                                                    (function_id, block_id)
                                                {
                                                    return Err(
                                                        ValidationError::ConversionInvalidStorageClass {
                                                            function: func,
                                                            block,
                                                            opcode: inst.class.opcode,
                                                            expected: "PhysicalStorageBuffer",
                                                        }.into(),
                        );
                                                }
                                            }

                                            // Vulkan requires 64-bit input
                                            if ctx.is_vulkan() {
                                                let input_width = resolver
                                                    .get_bit_width(input_type_id, ctx.definitions);
                                                if input_width != Some(64) {
                                                    if let (Some(func), Some(block)) =
                                                        (function_id, block_id)
                                                    {
                                                        return Err(
                                                            ValidationError::ConversionRequires64BitInteger {
                                                                function: func,
                                                                block,
                                                                opcode: inst.class.opcode,
                                                            }.into(),
                        );
                                                    }
                                                }
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
// Limited Type Conversion Rule (8/16-bit restrictions)
// ============================================================================

/// Validates that 8/16-bit types are only used with width-only conversions in Shader capability.
///
/// Ensures that when Shader capability is present, general conversions (ConvertFToU, ConvertFToS,
/// ConvertSToF, ConvertUToF, Bitcast) cannot use 8- or 16-bit types.
pub struct LimitedTypeConversionRule;

impl ValidationRule for LimitedTypeConversionRule {
    fn name(&self) -> &'static str {
        "limited-type-conversion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        use rspirv::spirv::Capability;

        // Only applies when Shader capability is present
        if !ctx.has_capability(Capability::Shader) {
            return Ok(());
        }

        let resolver = DefaultTypeResolver;

        // Helper to check if a type contains "limited-use" 8/16-bit types
        // (types without their corresponding capability)
        let contains_limited_use_type = |type_id: u32| -> bool {
            let width = resolver.get_bit_width(type_id, ctx.definitions);
            match width {
                Some(8) => !ctx.has_capability(Capability::Int8),
                Some(16) => {
                    // 16-bit is limited if neither Int16 nor Float16 is present
                    !ctx.has_capability(Capability::Int16)
                        && !ctx.has_capability(Capability::Float16)
                }
                _ => false,
            }
        };

        // These ops cannot use limited-use 8/16-bit types in Shader capability
        let restricted_ops = [
            Op::ConvertFToU,
            Op::ConvertFToS,
            Op::ConvertSToF,
            Op::ConvertUToF,
            Op::Bitcast,
        ];

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
                    if !restricted_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Check result type for limited-use 8/16-bit
                    if contains_limited_use_type(result_type_id) {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::ConversionLimitedTypeNotAllowed {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Check input type for limited-use 8/16-bit
                    if let Some(rspirv::dr::Operand::IdRef(input_id)) = inst.operands.first() {
                        let input_inst = crate::validation::types::ResultId::try_from(*input_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(input_inst) = input_inst {
                            if let Some(input_type_id) = input_inst.result_type {
                                if contains_limited_use_type(input_type_id) {
                                    if let (Some(func), Some(block)) = (function_id, block_id) {
                                        return Err(
                                            ValidationError::ConversionLimitedTypeNotAllowed {
                                                function: func,
                                                block,
                                                opcode: inst.class.opcode,
                                            }
                                            .into(),
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

// ============================================================================
// Generic Pointer Cast Rule
// ============================================================================

/// Validates generic pointer cast operations (OpPtrCastToGeneric, OpGenericCastToPtr, OpGenericCastToPtrExplicit).
///
/// Ensures that:
/// - OpPtrCastToGeneric: Result must be pointer with Generic storage class, input must be pointer with valid source class
/// - OpGenericCastToPtr: Result must be pointer with Workgroup/CrossWorkgroup/Function, input must be Generic
/// - OpGenericCastToPtrExplicit: Same as OpGenericCastToPtr but with explicit storage class operand
pub struct GenericPointerCastRule;

impl ValidationRule for GenericPointerCastRule {
    fn name(&self) -> &'static str {
        "generic-pointer-cast"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        use rspirv::spirv::StorageClass;

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
                        Op::PtrCastToGeneric => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be a pointer with Generic storage class
                            let result_type_inst =
                                crate::validation::types::ResultId::try_from(result_type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid));

                            if let Some(type_inst) = result_type_inst {
                                if !type_inst.is_pointer_type() {
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
                                            expected: "pointer",
                                        }
                                        .into());
                                    }
                                }

                                // Check storage class is Generic
                                let storage_class =
                                    type_inst.operands.first().and_then(|op| match op {
                                        rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                                        _ => None,
                                    });

                                if storage_class != Some(StorageClass::Generic) {
                                    if let (Some(func), Some(block)) = (function_id, block_id) {
                                        return Err(
                                            ValidationError::ConversionInvalidStorageClass {
                                                function: func,
                                                block,
                                                opcode: inst.class.opcode,
                                                expected: "Generic",
                                            }
                                            .into(),
                                        );
                                    }
                                }
                            }

                            // Check input is a pointer with valid source storage class
                            if let Some(rspirv::dr::Operand::IdRef(input_id)) =
                                inst.operands.first()
                            {
                                let input_inst =
                                    crate::validation::types::ResultId::try_from(*input_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(input_inst) = input_inst {
                                    if let Some(input_type_id) = input_inst.result_type {
                                        let input_type_inst =
                                            crate::validation::types::ResultId::try_from(
                                                input_type_id,
                                            )
                                            .ok()
                                            .and_then(|rid| ctx.definitions.get(&rid));

                                        if let Some(input_type) = input_type_inst {
                                            if !input_type.is_pointer_type() {
                                                if let (
                                                    Some(func),
                                                    Some(block),
                                                    Some(result_type),
                                                ) = (
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
                                                            expected: "pointer",
                                                        }.into(),
                        );
                                                }
                                            }

                                            // Source must be Workgroup, CrossWorkgroup, or Function
                                            let input_sc = input_type.operands.first().and_then(
                                                |op| match op {
                                                    rspirv::dr::Operand::StorageClass(sc) => {
                                                        Some(*sc)
                                                    }
                                                    _ => None,
                                                },
                                            );

                                            if !matches!(
                                                input_sc,
                                                Some(StorageClass::Workgroup)
                                                    | Some(StorageClass::CrossWorkgroup)
                                                    | Some(StorageClass::Function)
                                            ) {
                                                if let (Some(func), Some(block)) =
                                                    (function_id, block_id)
                                                {
                                                    return Err(
                                                        ValidationError::ConversionInvalidStorageClass {
                                                            function: func,
                                                            block,
                                                            opcode: inst.class.opcode,
                                                            expected: "Workgroup, CrossWorkgroup, or Function",
                                                        }.into(),
                        );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Op::GenericCastToPtr => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be a pointer with Workgroup, CrossWorkgroup, or Function
                            let result_type_inst =
                                crate::validation::types::ResultId::try_from(result_type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid));

                            if let Some(type_inst) = result_type_inst {
                                if !type_inst.is_pointer_type() {
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
                                            expected: "pointer",
                                        }
                                        .into());
                                    }
                                }

                                let storage_class =
                                    type_inst.operands.first().and_then(|op| match op {
                                        rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                                        _ => None,
                                    });

                                if !matches!(
                                    storage_class,
                                    Some(StorageClass::Workgroup)
                                        | Some(StorageClass::CrossWorkgroup)
                                        | Some(StorageClass::Function)
                                ) {
                                    if let (Some(func), Some(block)) = (function_id, block_id) {
                                        return Err(
                                            ValidationError::ConversionInvalidStorageClass {
                                                function: func,
                                                block,
                                                opcode: inst.class.opcode,
                                                expected: "Workgroup, CrossWorkgroup, or Function",
                                            }
                                            .into(),
                                        );
                                    }
                                }
                            }

                            // Check input is a pointer with Generic storage class
                            if let Some(rspirv::dr::Operand::IdRef(input_id)) =
                                inst.operands.first()
                            {
                                let input_inst =
                                    crate::validation::types::ResultId::try_from(*input_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(input_inst) = input_inst {
                                    if let Some(input_type_id) = input_inst.result_type {
                                        let input_type_inst =
                                            crate::validation::types::ResultId::try_from(
                                                input_type_id,
                                            )
                                            .ok()
                                            .and_then(|rid| ctx.definitions.get(&rid));

                                        if let Some(input_type) = input_type_inst {
                                            if !input_type.is_pointer_type() {
                                                if let (
                                                    Some(func),
                                                    Some(block),
                                                    Some(result_type),
                                                ) = (
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
                                                            expected: "pointer",
                                                        }.into(),
                        );
                                                }
                                            }

                                            let input_sc = input_type.operands.first().and_then(
                                                |op| match op {
                                                    rspirv::dr::Operand::StorageClass(sc) => {
                                                        Some(*sc)
                                                    }
                                                    _ => None,
                                                },
                                            );

                                            if input_sc != Some(StorageClass::Generic) {
                                                if let (Some(func), Some(block)) =
                                                    (function_id, block_id)
                                                {
                                                    return Err(
                                                        ValidationError::ConversionInvalidStorageClass {
                                                            function: func,
                                                            block,
                                                            opcode: inst.class.opcode,
                                                            expected: "Generic (for input)",
                                                        }.into(),
                        );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Op::GenericCastToPtrExplicit => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be a pointer
                            let result_type_inst =
                                crate::validation::types::ResultId::try_from(result_type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid));

                            let result_storage_class = result_type_inst.and_then(|type_inst| {
                                if !type_inst.is_pointer_type() {
                                    return None;
                                }
                                type_inst.operands.first().and_then(|op| match op {
                                    rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                                    _ => None,
                                })
                            });

                            if result_storage_class.is_none() {
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
                                        expected: "pointer",
                                    }
                                    .into());
                                }
                            }

                            // Get target storage class from operand (operand 1)
                            let target_sc = inst.operands.get(1).and_then(|op| match op {
                                rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                                _ => None,
                            });

                            // Result storage class must match target
                            if let (Some(result_sc), Some(target)) =
                                (result_storage_class, target_sc)
                            {
                                if result_sc != target {
                                    if let (Some(func), Some(block)) = (function_id, block_id) {
                                        return Err(
                                            ValidationError::ConversionInvalidStorageClass {
                                                function: func,
                                                block,
                                                opcode: inst.class.opcode,
                                                expected: "same as target storage class operand",
                                            }
                                            .into(),
                                        );
                                    }
                                }
                            }

                            // Check input is a pointer with Generic storage class
                            if let Some(rspirv::dr::Operand::IdRef(input_id)) =
                                inst.operands.first()
                            {
                                let input_inst =
                                    crate::validation::types::ResultId::try_from(*input_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(input_inst) = input_inst {
                                    if let Some(input_type_id) = input_inst.result_type {
                                        let input_type_inst =
                                            crate::validation::types::ResultId::try_from(
                                                input_type_id,
                                            )
                                            .ok()
                                            .and_then(|rid| ctx.definitions.get(&rid));

                                        if let Some(input_type) = input_type_inst {
                                            if !input_type.is_pointer_type() {
                                                if let (
                                                    Some(func),
                                                    Some(block),
                                                    Some(result_type),
                                                ) = (
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
                                                            expected: "pointer",
                                                        }.into(),
                        );
                                                }
                                            }

                                            let input_sc = input_type.operands.first().and_then(
                                                |op| match op {
                                                    rspirv::dr::Operand::StorageClass(sc) => {
                                                        Some(*sc)
                                                    }
                                                    _ => None,
                                                },
                                            );

                                            if input_sc != Some(StorageClass::Generic) {
                                                if let (Some(func), Some(block)) =
                                                    (function_id, block_id)
                                                {
                                                    return Err(
                                                        ValidationError::ConversionInvalidStorageClass {
                                                            function: func,
                                                            block,
                                                            opcode: inst.class.opcode,
                                                            expected: "Generic (for input)",
                                                        }.into(),
                        );
                                                }
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
// Cooperative Matrix Conversion Rule
// ============================================================================

/// Validates that cooperative matrix conversions have matching shapes.
///
/// For conversions involving cooperative matrix types (OpTypeCooperativeMatrixKHR/NV),
/// the scope, rows, and columns must match between result and input types.
/// For KHR matrices, the Use parameter must also match (with special handling
/// for CooperativeMatrixConversionsNV capability).
pub struct CooperativeMatrixConversionRule;

impl ValidationRule for CooperativeMatrixConversionRule {
    fn name(&self) -> &'static str {
        "cooperative-matrix-conversion"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        use crate::validation::type_ext::TypeResolver;

        let resolver = DefaultTypeResolver;

        // Conversions that need cooperative matrix shape matching
        let conversion_ops = [
            Op::ConvertFToU,
            Op::ConvertFToS,
            Op::ConvertSToF,
            Op::ConvertUToF,
            Op::UConvert,
            Op::SConvert,
            Op::FConvert,
        ];

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
                    if !conversion_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Get input type
                    let input_type_id = inst
                        .operands
                        .first()
                        .and_then(|op| match op {
                            rspirv::dr::Operand::IdRef(id) => Some(*id),
                            _ => None,
                        })
                        .and_then(|id| {
                            crate::validation::types::ResultId::try_from(id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))
                        })
                        .and_then(|input_inst| input_inst.result_type);

                    let Some(input_type_id) = input_type_id else {
                        continue;
                    };

                    let is_result_coop_matrix =
                        resolver.is_cooperative_matrix(result_type_id, ctx.definitions);
                    let is_input_coop_matrix =
                        resolver.is_cooperative_matrix(input_type_id, ctx.definitions);

                    // If either is a cooperative matrix, validate shapes match
                    if is_result_coop_matrix || is_input_coop_matrix {
                        // Both must be cooperative matrices for a valid conversion
                        if is_result_coop_matrix != is_input_coop_matrix {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                return Err(
                                    ValidationError::CooperativeMatrixConversionTypeMismatch {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                    }
                                    .into(),
                                );
                            }
                        }

                        // Validate shapes match
                        self.validate_cooperative_matrix_shapes(
                            result_type_id,
                            input_type_id,
                            ctx,
                            inst.class.opcode,
                            function_id,
                            block_id,
                        )?;
                    }

                    // Check for cooperative vector NV types
                    let is_result_coop_vector =
                        resolver.is_cooperative_vector_nv(result_type_id, ctx.definitions);
                    let is_input_coop_vector =
                        resolver.is_cooperative_vector_nv(input_type_id, ctx.definitions);

                    if is_result_coop_vector || is_input_coop_vector {
                        // Both must be cooperative vectors for a valid conversion
                        if is_result_coop_vector != is_input_coop_vector {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                return Err(
                                    ValidationError::CooperativeMatrixConversionTypeMismatch {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                    }
                                    .into(),
                                );
                            }
                        }

                        // Validate dimensions match for cooperative vectors
                        self.validate_cooperative_vector_dimensions(
                            result_type_id,
                            input_type_id,
                            ctx,
                            inst.class.opcode,
                            function_id,
                            block_id,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl CooperativeMatrixConversionRule {
    /// Validate that two cooperative matrix types have matching shapes.
    fn validate_cooperative_matrix_shapes(
        &self,
        result_type_id: u32,
        input_type_id: u32,
        ctx: &ValidationContext<'_>,
        opcode: Op,
        function_id: Option<Id>,
        block_id: Option<Id>,
    ) -> ValidationResult {
        let result_type_inst = crate::validation::types::ResultId::try_from(result_type_id)
            .ok()
            .and_then(|rid| ctx.definitions.get(&rid));
        let input_type_inst = crate::validation::types::ResultId::try_from(input_type_id)
            .ok()
            .and_then(|rid| ctx.definitions.get(&rid));

        let (Some(result_type), Some(input_type)) = (result_type_inst, input_type_inst) else {
            return Ok(());
        };

        // Both must have the same opcode (KHR vs NV)
        if result_type.class.opcode != input_type.class.opcode {
            if let (Some(func), Some(block)) = (function_id, block_id) {
                return Err(ValidationError::CooperativeMatrixConversionTypeMismatch {
                    function: func,
                    block,
                    opcode,
                }
                .into());
            }
        }

        // OpTypeCooperativeMatrixKHR/NV layout:
        // operand 0: ComponentType (IdRef)
        // operand 1: Scope (IdRef)
        // operand 2: Rows (IdRef)
        // operand 3: Columns (IdRef)
        // For KHR only:
        // operand 4: Use (IdRef)

        // Get scope IDs
        let result_scope_id = result_type.operands.get(1).and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        });
        let input_scope_id = input_type.operands.get(1).and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        });

        // Validate scope matches (if both are constants)
        if let (Some(result_scope), Some(input_scope)) = (result_scope_id, input_scope_id) {
            if let (Some(result_val), Some(input_val)) = (
                self.eval_constant_u32(result_scope, ctx),
                self.eval_constant_u32(input_scope, ctx),
            ) {
                if result_val != input_val {
                    if let (Some(func), Some(block)) = (function_id, block_id) {
                        return Err(ValidationError::CooperativeMatrixShapeMismatch {
                            function: func,
                            block,
                            opcode,
                            dimension: "scope",
                        }
                        .into());
                    }
                }
            }
        }

        // Get rows IDs
        let result_rows_id = result_type.operands.get(2).and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        });
        let input_rows_id = input_type.operands.get(2).and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        });

        // Validate rows match
        if let (Some(result_rows), Some(input_rows)) = (result_rows_id, input_rows_id) {
            if let (Some(result_val), Some(input_val)) = (
                self.eval_constant_u32(result_rows, ctx),
                self.eval_constant_u32(input_rows, ctx),
            ) {
                if result_val != input_val {
                    if let (Some(func), Some(block)) = (function_id, block_id) {
                        return Err(ValidationError::CooperativeMatrixShapeMismatch {
                            function: func,
                            block,
                            opcode,
                            dimension: "rows",
                        }
                        .into());
                    }
                }
            }
        }

        // Get columns IDs
        let result_cols_id = result_type.operands.get(3).and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        });
        let input_cols_id = input_type.operands.get(3).and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        });

        // Validate columns match
        if let (Some(result_cols), Some(input_cols)) = (result_cols_id, input_cols_id) {
            if let (Some(result_val), Some(input_val)) = (
                self.eval_constant_u32(result_cols, ctx),
                self.eval_constant_u32(input_cols, ctx),
            ) {
                if result_val != input_val {
                    if let (Some(func), Some(block)) = (function_id, block_id) {
                        return Err(ValidationError::CooperativeMatrixShapeMismatch {
                            function: func,
                            block,
                            opcode,
                            dimension: "columns",
                        }
                        .into());
                    }
                }
            }
        }

        // For KHR matrices, also validate Use parameter
        if result_type.class.opcode == Op::TypeCooperativeMatrixKHR {
            let result_use_id = result_type.operands.get(4).and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let input_use_id = input_type.operands.get(4).and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });

            if let (Some(result_use), Some(input_use)) = (result_use_id, input_use_id) {
                if let (Some(result_val), Some(input_val)) = (
                    self.eval_constant_u32(result_use, ctx),
                    self.eval_constant_u32(input_use, ctx),
                ) {
                    // Check if CooperativeMatrixConversionsNV capability allows
                    // conversions from Accumulator to A/B
                    let has_conversions_nv = ctx
                        .has_capability(rspirv::spirv::Capability::CooperativeMatrixConversionsNV);

                    // MatrixAccumulatorKHR = 2
                    let is_acc_to_ab_conversion =
                        has_conversions_nv && input_val == 2 && result_val != input_val;

                    if result_val != input_val && !is_acc_to_ab_conversion {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::CooperativeMatrixShapeMismatch {
                                function: func,
                                block,
                                opcode,
                                dimension: "use",
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate that two cooperative vector NV types have matching dimensions.
    fn validate_cooperative_vector_dimensions(
        &self,
        result_type_id: u32,
        input_type_id: u32,
        ctx: &ValidationContext<'_>,
        opcode: Op,
        function_id: Option<Id>,
        block_id: Option<Id>,
    ) -> ValidationResult {
        let result_type_inst = crate::validation::types::ResultId::try_from(result_type_id)
            .ok()
            .and_then(|rid| ctx.definitions.get(&rid));
        let input_type_inst = crate::validation::types::ResultId::try_from(input_type_id)
            .ok()
            .and_then(|rid| ctx.definitions.get(&rid));

        let (Some(result_type), Some(input_type)) = (result_type_inst, input_type_inst) else {
            return Ok(());
        };

        // OpTypeCooperativeVectorNV layout:
        // operand 0: ComponentType (IdRef)
        // operand 1: Components (IdRef)

        let result_components_id = result_type.operands.get(1).and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        });
        let input_components_id = input_type.operands.get(1).and_then(|op| match op {
            rspirv::dr::Operand::IdRef(id) => Some(*id),
            _ => None,
        });

        if let (Some(result_comp), Some(input_comp)) = (result_components_id, input_components_id) {
            if let (Some(result_val), Some(input_val)) = (
                self.eval_constant_u32(result_comp, ctx),
                self.eval_constant_u32(input_comp, ctx),
            ) {
                if result_val != input_val {
                    if let (Some(func), Some(block)) = (function_id, block_id) {
                        return Err(ValidationError::CooperativeMatrixShapeMismatch {
                            function: func,
                            block,
                            opcode,
                            dimension: "components",
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Evaluate a constant instruction to get its u32 value.
    fn eval_constant_u32(&self, id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
        let result_id = crate::validation::types::ResultId::try_from(id).ok()?;
        let inst = ctx.definitions.get(&result_id)?;

        // Must be OpConstant
        if inst.class.opcode != Op::Constant {
            return None;
        }

        match inst.operands.first() {
            Some(rspirv::dr::Operand::LiteralBit32(v)) => Some(*v),
            Some(rspirv::dr::Operand::LiteralBit64(v)) => Some(*v as u32),
            _ => None,
        }
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
        &PointerConversionRule,
        &LimitedTypeConversionRule,
        &GenericPointerCastRule,
        &CooperativeMatrixConversionRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::context::ValidationRule;

    #[test]
    fn all_conversion_rules_has_expected_count() {
        let rules = all_conversion_rules();
        assert_eq!(rules.len(), 10);
    }

    #[test]
    fn conversion_rules_have_unique_names() {
        let rules = all_conversion_rules();
        let names: Vec<&str> = rules.iter().map(|r| r.name()).collect();

        let expected_names = [
            "float-to-int-conversion",
            "int-to-float-conversion",
            "int-width-conversion",
            "float-width-conversion",
            "bitcast",
            "saturating-conversion",
            "pointer-conversion",
            "limited-type-conversion",
            "generic-pointer-cast",
            "cooperative-matrix-conversion",
        ];

        for expected in expected_names {
            assert!(
                names.contains(&expected),
                "Missing expected conversion rule: {expected}"
            );
        }
    }

    #[test]
    fn generic_pointer_cast_rule_has_correct_name() {
        let rule = GenericPointerCastRule;
        assert_eq!(rule.name(), "generic-pointer-cast");
    }

    #[test]
    fn cooperative_matrix_conversion_rule_has_correct_name() {
        let rule = CooperativeMatrixConversionRule;
        assert_eq!(rule.name(), "cooperative-matrix-conversion");
    }
}
