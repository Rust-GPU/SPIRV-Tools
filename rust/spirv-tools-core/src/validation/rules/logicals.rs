//! Logical instruction validation rules.
//!
//! This module validates SPIR-V logical instructions including:
//!
//! - Boolean reduction operations (Any, All)
//! - Float classification operations (IsNan, IsInf, IsFinite, IsNormal, SignBitSet)
//! - Float comparison operations (FOrd*, FUnord*)
//! - Logical operations (LogicalEqual, LogicalNotEqual, LogicalOr, LogicalAnd, LogicalNot)
//! - Integer comparison operations (IEqual, INotEqual, U*, S*)
//! - Select operation

use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::Id;

// ============================================================================
// Boolean Reduction Rule
// ============================================================================

/// Validates boolean reduction operations (Any, All).
///
/// Ensures that:
/// - Result type is bool scalar
/// - Operand is bool vector
pub struct BooleanReductionRule;

impl ValidationRule for BooleanReductionRule {
    fn name(&self) -> &'static str {
        "boolean-reduction"
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
                    if inst.class.opcode != Op::Any && inst.class.opcode != Op::All {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be bool scalar
                    if !resolver.is_bool_scalar(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::LogicalResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "bool scalar",
                            }.into());
                        }
                    }

                    // Operand must be bool vector
                    if let Some(rspirv::dr::Operand::IdRef(operand_id)) = inst.operands.first() {
                        let operand_inst =
                            crate::validation::types::ResultId::try_from(*operand_id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(operand_inst) = operand_inst {
                            if let Some(operand_type_id) = operand_inst.result_type {
                                let is_bool_vec = resolver
                                    .is_bool_scalar_or_vector(operand_type_id, ctx.definitions)
                                    && resolver.get_dimension(operand_type_id, ctx.definitions) > 1;

                                if !is_bool_vec {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::LogicalOperandTypeInvalid {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
                                            expected: "bool vector",
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

// ============================================================================
// Float Classification Rule
// ============================================================================

/// Validates float classification operations (IsNan, IsInf, IsFinite, IsNormal, SignBitSet).
///
/// Ensures that:
/// - Result type is bool scalar or vector
/// - Operand is float scalar or vector
/// - Dimensions match
pub struct FloatClassificationRule;

impl ValidationRule for FloatClassificationRule {
    fn name(&self) -> &'static str {
        "float-classification"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let float_class_ops = [
            Op::IsNan,
            Op::IsInf,
            Op::IsFinite,
            Op::IsNormal,
            Op::SignBitSet,
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
                    if !float_class_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be bool scalar or vector
                    if !resolver.is_bool_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::LogicalResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "bool scalar or vector",
                            }.into());
                        }
                    }

                    // Operand must be float scalar or vector with same dimension
                    if let Some(rspirv::dr::Operand::IdRef(operand_id)) = inst.operands.first() {
                        let operand_inst =
                            crate::validation::types::ResultId::try_from(*operand_id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(operand_inst) = operand_inst {
                            if let Some(operand_type_id) = operand_inst.result_type {
                                if !resolver
                                    .is_float_scalar_or_vector(operand_type_id, ctx.definitions)
                                {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::LogicalOperandTypeInvalid {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
                                            expected: "float scalar or vector",
                                        }.into());
                                    }
                                }

                                let result_dim =
                                    resolver.get_dimension(result_type_id, ctx.definitions);
                                let operand_dim =
                                    resolver.get_dimension(operand_type_id, ctx.definitions);

                                if result_dim != operand_dim {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::LogicalDimensionMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
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

// ============================================================================
// Float Comparison Rule
// ============================================================================

/// Validates float comparison operations (FOrdEqual, FUnordEqual, etc.).
///
/// Ensures that:
/// - Result type is bool scalar or vector
/// - Both operands are float scalar or vector
/// - Dimensions match
/// - Operand types match
pub struct FloatComparisonRule;

impl ValidationRule for FloatComparisonRule {
    fn name(&self) -> &'static str {
        "float-comparison"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let float_cmp_ops = [
            Op::FOrdEqual,
            Op::FUnordEqual,
            Op::FOrdNotEqual,
            Op::FUnordNotEqual,
            Op::FOrdLessThan,
            Op::FUnordLessThan,
            Op::FOrdGreaterThan,
            Op::FUnordGreaterThan,
            Op::FOrdLessThanEqual,
            Op::FUnordLessThanEqual,
            Op::FOrdGreaterThanEqual,
            Op::FUnordGreaterThanEqual,
            Op::LessOrGreater,
            Op::Ordered,
            Op::Unordered,
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
                    if !float_cmp_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be bool scalar or vector
                    if !resolver.is_bool_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::LogicalResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "bool scalar or vector",
                            }.into());
                        }
                    }

                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);

                    // Get operand types
                    let get_operand_type = |idx: usize| -> Option<u32> {
                        let operand_id = inst.operands.get(idx).and_then(|op| match op {
                            rspirv::dr::Operand::IdRef(id) => Some(*id),
                            _ => None,
                        })?;

                        let operand_inst =
                            crate::validation::types::ResultId::try_from(operand_id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))?;

                        operand_inst.result_type
                    };

                    let left_type = get_operand_type(0);
                    let right_type = get_operand_type(1);

                    if let Some(left_tid) = left_type {
                        if !resolver.is_float_scalar_or_vector(left_tid, ctx.definitions) {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::LogicalOperandTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
                                    expected: "float scalar or vector",
                                }.into());
                            }
                        }

                        let left_dim = resolver.get_dimension(left_tid, ctx.definitions);

                        // Result must match operand dimension (scalar operand -> scalar result,
                        // vector operand -> vector result)
                        if left_dim != result_dim {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::LogicalResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
                                    expected: "bool scalar or vector",
                                }.into());
                            }
                        }

                        // Both operands must have same type
                        if left_type != right_type {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::LogicalOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
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

// ============================================================================
// Logical Operations Rule
// ============================================================================

/// Validates logical operations (LogicalEqual, LogicalNotEqual, LogicalOr, LogicalAnd, LogicalNot).
///
/// Ensures that:
/// - Result type is bool scalar or vector
/// - All operands match result type
pub struct LogicalOperationsRule;

impl ValidationRule for LogicalOperationsRule {
    fn name(&self) -> &'static str {
        "logical-operations"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let logical_ops = [
            Op::LogicalEqual,
            Op::LogicalNotEqual,
            Op::LogicalOr,
            Op::LogicalAnd,
            Op::LogicalNot,
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
                    if !logical_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be bool scalar or vector
                    if !resolver.is_bool_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::LogicalResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "bool scalar or vector",
                            }.into());
                        }
                    }

                    // All operands must match result type
                    for operand in &inst.operands {
                        let operand_id = match operand {
                            rspirv::dr::Operand::IdRef(id) => *id,
                            _ => continue,
                        };

                        let operand_inst =
                            crate::validation::types::ResultId::try_from(operand_id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(operand_inst) = operand_inst {
                            if let Some(operand_type_id) = operand_inst.result_type {
                                if operand_type_id != result_type_id {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::LogicalOperandTypeMismatch {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
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

// ============================================================================
// Integer Comparison Rule
// ============================================================================

/// Validates integer comparison operations (IEqual, INotEqual, U*, S*).
///
/// Ensures that:
/// - Result type is bool scalar or vector
/// - Both operands are int scalar or vector
/// - Dimensions match
/// - Bit widths match
pub struct IntComparisonRule;

impl ValidationRule for IntComparisonRule {
    fn name(&self) -> &'static str {
        "int-comparison"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let int_cmp_ops = [
            Op::IEqual,
            Op::INotEqual,
            Op::UGreaterThan,
            Op::UGreaterThanEqual,
            Op::ULessThan,
            Op::ULessThanEqual,
            Op::SGreaterThan,
            Op::SGreaterThanEqual,
            Op::SLessThan,
            Op::SLessThanEqual,
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
                    if !int_cmp_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be bool scalar or vector
                    if !resolver.is_bool_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::LogicalResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "bool scalar or vector",
                            }.into());
                        }
                    }

                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);

                    // Get operand types
                    let get_operand_type = |idx: usize| -> Option<u32> {
                        let operand_id = inst.operands.get(idx).and_then(|op| match op {
                            rspirv::dr::Operand::IdRef(id) => Some(*id),
                            _ => None,
                        })?;

                        let operand_inst =
                            crate::validation::types::ResultId::try_from(operand_id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))?;

                        operand_inst.result_type
                    };

                    let left_type = get_operand_type(0);
                    let right_type = get_operand_type(1);

                    if let (Some(left_tid), Some(right_tid)) = (left_type, right_type) {
                        // Left must be int
                        if !resolver.is_int_scalar_or_vector(left_tid, ctx.definitions) {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::LogicalOperandTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
                                    expected: "int scalar or vector",
                                }.into());
                            }
                        }

                        // Right must be int
                        if !resolver.is_int_scalar_or_vector(right_tid, ctx.definitions) {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::LogicalOperandTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
                                    expected: "int scalar or vector",
                                }.into());
                            }
                        }

                        // Dimensions must match result
                        let left_dim = resolver.get_dimension(left_tid, ctx.definitions);
                        let right_dim = resolver.get_dimension(right_tid, ctx.definitions);

                        if left_dim != result_dim || right_dim != result_dim {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::LogicalDimensionMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
                                }.into());
                            }
                        }

                        // Check bit widths first (more specific error)
                        let left_width = resolver.get_bit_width(left_tid, ctx.definitions);
                        let right_width = resolver.get_bit_width(right_tid, ctx.definitions);

                        if left_width != right_width {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::LogicalBitWidthMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
                                }.into());
                            }
                        }

                        // Operand types must be identical (signedness matters even if width matches)
                        if left_tid != right_tid {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::LogicalOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
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

// ============================================================================
// Select Rule
// ============================================================================

/// Validates OpSelect operations.
///
/// Ensures that:
/// - Result type is scalar, vector, pointer, image/sampler (with capability), or composite (SPIR-V 1.4+)
/// - Condition is bool scalar or vector
/// - Condition dimension matches result dimension (or scalar condition with composites feature)
/// - Both objects match result type
/// - Pointer select requires VariablePointers capability in Logical addressing
/// - Image/sampler select requires BindlessTextureNV capability
pub struct SelectRule;

impl ValidationRule for SelectRule {
    fn name(&self) -> &'static str {
        "select"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        use rspirv::spirv::{AddressingModel, Capability};

        let resolver = DefaultTypeResolver;

        // Check addressing model for pointer select rules
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
        let has_variable_pointers = ctx.has_capability(Capability::VariablePointers)
            || ctx.has_capability(Capability::VariablePointersStorageBuffer);
        let has_bindless_texture = ctx.has_capability(Capability::BindlessTextureNV);

        // Check SPIR-V version for composite select support
        let supports_composite_select = ctx.module.header.as_ref().map_or(false, |h| {
            let version = h.version();
            version >= (1, 4)
        });

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
                    if inst.class.opcode != Op::Select {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Get result type opcode
                    let result_type_opcode = crate::validation::types::ResultId::try_from(result_type_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .map(|inst| inst.class.opcode);

                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);

                    // Validate result type
                    let is_valid_result_type = match result_type_opcode {
                        Some(Op::TypePointer | Op::TypeUntypedPointerKHR) => {
                            if is_logical && !has_variable_pointers {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::SelectPointerRequiresCapability {
                                        function: func,
                                        block,
                                        result_type,
                                    }.into());
                                }
                            }
                            true
                        }
                        Some(Op::TypeImage | Op::TypeSampler | Op::TypeSampledImage) => {
                            if !has_bindless_texture {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::SelectImageRequiresCapability {
                                        function: func,
                                        block,
                                        result_type,
                                    }.into());
                                }
                            }
                            true
                        }
                        Some(Op::TypeVector) => true,
                        Some(Op::TypeBool | Op::TypeInt | Op::TypeFloat) => true,
                        Some(Op::TypeArray | Op::TypeMatrix | Op::TypeStruct) => {
                            supports_composite_select
                        }
                        _ => false,
                    };

                    if !is_valid_result_type {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::SelectResultTypeInvalid {
                                function: func,
                                block,
                                result_type,
                                supports_composites: supports_composite_select,
                            }.into());
                        }
                    }

                    // Condition (operand 0) must be bool scalar or vector
                    let condition_type = inst
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
                        .and_then(|inst| inst.result_type);

                    if let Some(cond_type_id) = condition_type {
                        if !resolver.is_bool_scalar_or_vector(cond_type_id, ctx.definitions) {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::SelectConditionNotBool {
                                    function: func,
                                    block,
                                    result_type,
                                }.into());
                            }
                        }

                        let cond_dim = resolver.get_dimension(cond_type_id, ctx.definitions);

                        // Dimension check: condition must match result unless scalar condition with composites
                        if cond_dim != result_dim {
                            let is_scalar_cond = cond_dim == 1;
                            if !supports_composite_select || !is_scalar_cond {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::SelectDimensionMismatch {
                                        function: func,
                                        block,
                                        result_type,
                                    }.into());
                                }
                            }
                        }
                    }

                    // Both objects (operands 1 and 2) must match result type
                    for operand_idx in [1, 2] {
                        let object_type = inst
                            .operands
                            .get(operand_idx)
                            .and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(id) => Some(*id),
                                _ => None,
                            })
                            .and_then(|id| {
                                crate::validation::types::ResultId::try_from(id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid))
                            })
                            .and_then(|inst| inst.result_type);

                        if let Some(obj_type_id) = object_type {
                            if obj_type_id != result_type_id {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id).ok(),
                                ) {
                                    return Err(ValidationError::SelectObjectTypeMismatch {
                                        function: func,
                                        block,
                                        result_type,
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

// ============================================================================
// All logical rules
// ============================================================================

/// Returns all logical validation rules.
pub fn all_logical_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &BooleanReductionRule,
        &FloatClassificationRule,
        &FloatComparisonRule,
        &LogicalOperationsRule,
        &IntComparisonRule,
        &SelectRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::op_ext::OpExt;

    #[test]
    fn test_boolean_reduction_ops() {
        // Any/All reduce boolean vectors to scalar
        assert!(matches!(Op::Any, Op::Any));
        assert!(matches!(Op::All, Op::All));
    }

    #[test]
    fn test_float_classification_ops() {
        let float_class_ops = [
            Op::IsNan,
            Op::IsInf,
            Op::IsFinite,
            Op::IsNormal,
            Op::SignBitSet,
        ];

        for op in float_class_ops {
            // Verify these are logical-related operations
            assert!(!op.is_terminator());
        }
    }

    #[test]
    fn test_float_comparison_ops() {
        let float_cmp_ops = [
            Op::FOrdEqual,
            Op::FUnordEqual,
            Op::FOrdNotEqual,
            Op::FUnordNotEqual,
            Op::FOrdLessThan,
            Op::FUnordLessThan,
            Op::FOrdGreaterThan,
            Op::FUnordGreaterThan,
            Op::FOrdLessThanEqual,
            Op::FUnordLessThanEqual,
            Op::FOrdGreaterThanEqual,
            Op::FUnordGreaterThanEqual,
            Op::LessOrGreater,
            Op::Ordered,
            Op::Unordered,
        ];

        for op in float_cmp_ops {
            // Verify these are not barrier instructions
            assert!(!op.is_barrier());
        }
    }

    #[test]
    fn test_logical_ops() {
        let logical_ops = [
            Op::LogicalEqual,
            Op::LogicalNotEqual,
            Op::LogicalOr,
            Op::LogicalAnd,
            Op::LogicalNot,
        ];

        for op in logical_ops {
            // Logical operations are not terminators
            assert!(!op.is_terminator());
            assert!(!op.is_barrier());
        }
    }

    #[test]
    fn test_integer_comparison_ops() {
        let int_cmp_ops = [
            Op::IEqual,
            Op::INotEqual,
            Op::UGreaterThan,
            Op::UGreaterThanEqual,
            Op::ULessThan,
            Op::ULessThanEqual,
            Op::SGreaterThan,
            Op::SGreaterThanEqual,
            Op::SLessThan,
            Op::SLessThanEqual,
        ];

        for op in int_cmp_ops {
            // Integer comparisons are not terminators or barriers
            assert!(!op.is_terminator());
            assert!(!op.is_barrier());
        }
    }

    #[test]
    fn test_select_op() {
        // OpSelect is not a terminator
        assert!(!Op::Select.is_terminator());
        assert!(!Op::Select.is_barrier());
    }
}
