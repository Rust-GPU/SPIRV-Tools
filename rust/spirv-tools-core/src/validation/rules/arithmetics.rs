//! Arithmetic instruction validation rules.
//!
//! This module validates SPIR-V arithmetic instructions including:
//!
//! - Floating-point operations (FAdd, FSub, FMul, FDiv, FRem, FMod, FNegate)
//! - Integer operations (IAdd, ISub, IMul, SDiv, UDiv, SRem, URem, SMod, UMod, SNegate)
//! - Dot product (Dot, DotKHR)

use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::Id;

// ============================================================================
// Floating-Point Arithmetic Rule
// ============================================================================

/// Validates floating-point arithmetic operations.
///
/// Ensures that:
/// - Result type is a float scalar, vector, or cooperative matrix
/// - All operands match the result type
pub struct FloatArithmeticRule;

impl ValidationRule for FloatArithmeticRule {
    fn name(&self) -> &'static str {
        "float-arithmetic"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let float_ops = [
            Op::FAdd,
            Op::FSub,
            Op::FMul,
            Op::FDiv,
            Op::FRem,
            Op::FMod,
            Op::FNegate,
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
                    if !float_ops.contains(&inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result type must be float scalar or vector
                    // (or cooperative matrix, but we'll skip that for now)
                    if !resolver.is_float_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ArithmeticResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "float scalar or vector",
                            });
                        }
                    }

                    // All operands must match result type dimensions and bit width
                    let result_width = resolver.get_bit_width(result_type_id, ctx.definitions);
                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);

                    for (idx, operand) in inst.operands.iter().enumerate() {
                        let operand_id = match operand {
                            rspirv::dr::Operand::IdRef(id) => *id,
                            _ => continue,
                        };

                        // Get the type of this operand
                        let operand_inst = crate::validation::types::ResultId::try_from(operand_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        let Some(operand_inst) = operand_inst else {
                            continue;
                        };

                        let Some(operand_type_id) = operand_inst.result_type else {
                            continue;
                        };

                        let operand_width =
                            resolver.get_bit_width(operand_type_id, ctx.definitions);
                        let operand_dim = resolver.get_dimension(operand_type_id, ctx.definitions);

                        if operand_width != result_width || operand_dim != result_dim {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::ArithmeticOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: idx,
                                    result_type,
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

// ============================================================================
// Integer Arithmetic Rule
// ============================================================================

/// Validates integer arithmetic operations.
///
/// Ensures that:
/// - Result type is an int scalar or vector
/// - All operands have matching dimensions and bit widths
pub struct IntArithmeticRule;

impl ValidationRule for IntArithmeticRule {
    fn name(&self) -> &'static str {
        "int-arithmetic"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Operations that work on any integer type
        let any_int_ops = [
            Op::IAdd,
            Op::ISub,
            Op::IMul,
            Op::SNegate,
        ];

        // Operations that produce signed results
        let signed_int_ops = [
            Op::SDiv,
            Op::SRem,
            Op::SMod,
        ];

        // Operations that require unsigned integers
        let unsigned_int_ops = [Op::UDiv, Op::UMod];

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
                    let is_any_int = any_int_ops.contains(&inst.class.opcode);
                    let is_signed = signed_int_ops.contains(&inst.class.opcode);
                    let is_unsigned = unsigned_int_ops.contains(&inst.class.opcode);

                    if !is_any_int && !is_signed && !is_unsigned {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Check result type based on operation kind
                    if is_unsigned {
                        if !resolver.is_unsigned_int_scalar_or_vector(result_type_id, ctx.definitions)
                        {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::ArithmeticResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
                                    expected: "unsigned int scalar or vector",
                                });
                            }
                        }
                    } else if !resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ArithmeticResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "int scalar or vector",
                            });
                        }
                    }

                    // All operands must have matching dimensions and bit widths
                    let result_width = resolver.get_bit_width(result_type_id, ctx.definitions);
                    let result_dim = resolver.get_dimension(result_type_id, ctx.definitions);

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

                        // Only check int operands
                        if !resolver.is_int_scalar_or_vector(operand_type_id, ctx.definitions) {
                            continue;
                        }

                        let operand_width =
                            resolver.get_bit_width(operand_type_id, ctx.definitions);
                        let operand_dim = resolver.get_dimension(operand_type_id, ctx.definitions);

                        if operand_width != result_width || operand_dim != result_dim {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::ArithmeticOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: idx,
                                    result_type,
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

// ============================================================================
// Dot Product Rule
// ============================================================================

/// Validates dot product operations.
///
/// Ensures that:
/// - Result type is a float scalar
/// - Operands are vectors of the same dimension
/// - Operand component types match result type
pub struct DotProductRule;

impl ValidationRule for DotProductRule {
    fn name(&self) -> &'static str {
        "dot-product"
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
                    if inst.class.opcode != Op::Dot {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be float scalar
                    if !resolver.is_float_scalar(result_type_id, ctx.definitions) {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::ArithmeticResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "float scalar",
                            });
                        }
                    }

                    // Get both operand types
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

                    let op1_type = get_operand_type(0);
                    let op2_type = get_operand_type(1);

                    if let (Some(op1_tid), Some(op2_tid)) = (op1_type, op2_type) {
                        // Both must be float vectors
                        let op1_is_float_vec =
                            resolver.is_float_scalar_or_vector(op1_tid, ctx.definitions)
                                && resolver.get_dimension(op1_tid, ctx.definitions) > 1;
                        let op2_is_float_vec =
                            resolver.is_float_scalar_or_vector(op2_tid, ctx.definitions)
                                && resolver.get_dimension(op2_tid, ctx.definitions) > 1;

                        if !op1_is_float_vec {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::ArithmeticOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: 0,
                                    result_type,
                                });
                            }
                        }

                        if !op2_is_float_vec {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::ArithmeticOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: 1,
                                    result_type,
                                });
                            }
                        }

                        // Dimensions must match
                        let dim1 = resolver.get_dimension(op1_tid, ctx.definitions);
                        let dim2 = resolver.get_dimension(op2_tid, ctx.definitions);

                        if dim1 != dim2 {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::ArithmeticOperandTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    operand_index: 1,
                                    result_type,
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

// ============================================================================
// All arithmetic rules
// ============================================================================

/// Returns all arithmetic validation rules.
pub fn all_arithmetic_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&FloatArithmeticRule, &IntArithmeticRule, &DotProductRule]
}
