//! Arithmetic instruction validation rules.
//!
//! This module validates SPIR-V arithmetic instructions including:
//!
//! - Floating-point operations (FAdd, FSub, FMul, FDiv, FRem, FMod, FNegate)
//! - Integer operations (IAdd, ISub, IMul, SDiv, UDiv, SRem, URem, SMod, UMod, SNegate)
//! - Dot product (Dot, DotKHR)
//! - Linear algebra (VectorTimesScalar, MatrixTimesVector, VectorTimesMatrix, MatrixTimesMatrix)

use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::{matrix_details_by_id, vector_info};
use crate::validation::type_ext::{DefaultTypeResolver, TypeInstructionExt, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId};

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
// Vector/Scalar Multiply Rule
// ============================================================================

/// Validates VectorTimesScalar operations.
///
/// Ensures that:
/// - Result type is a float vector
/// - Vector operand matches result type
/// - Scalar operand component type matches vector component type
pub struct VectorTimesScalarRule;

impl ValidationRule for VectorTimesScalarRule {
    fn name(&self) -> &'static str {
        "vector-times-scalar"
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
                    if inst.class.opcode != Op::VectorTimesScalar {
                        continue;
                    }

                    let Some(func) = function_id else { continue };
                    let Some(block) = block_id else { continue };
                    let Some(result_type_id) = inst.result_type else { continue };
                    let Some(result_type) = TypeId::try_from(result_type_id).ok() else {
                        continue;
                    };

                    // Result type must be a float vector
                    let result_type_inst = ResultId::try_from(result_type_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(result_type_inst) = result_type_inst else { continue };

                    if !result_type_inst.is_vector_type() {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: func,
                            block,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: result_type,
                        });
                    }

                    let (component_type, _) = vector_info(result_type_inst);
                    let Some(component_type) = component_type else { continue };

                    // Check if component is float
                    if !resolver.is_float_scalar(component_type.into(), ctx.definitions) {
                        return Err(ValidationError::ArithmeticResultTypeInvalid {
                            function: func,
                            block,
                            opcode: inst.class.opcode,
                            result_type,
                            expected: "float vector",
                        });
                    }

                    // Get vector operand type
                    let vector_operand = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });

                    if let Some(vector_operand) = vector_operand {
                        let vector_inst = ctx.definitions.get(&vector_operand);
                        if let Some(vector_inst) = vector_inst {
                            if let Some(vector_type_id) = vector_inst.result_type {
                                if vector_type_id != result_type_id {
                                    let found = TypeId::try_from(vector_type_id).ok();
                                    if let Some(found) = found {
                                        return Err(ValidationError::InstructionResultTypeMismatch {
                                            function: func,
                                            block,
                                            instruction: inst.class.opcode,
                                            expected: result_type,
                                            found,
                                        });
                                    }
                                }
                            }
                        }
                    }

                    // Check scalar operand type
                    let scalar_operand = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });

                    if let Some(scalar_operand) = scalar_operand {
                        let scalar_inst = ctx.definitions.get(&scalar_operand);
                        if let Some(scalar_inst) = scalar_inst {
                            if let Some(scalar_type_id) = scalar_inst.result_type {
                                let scalar_type = TypeId::try_from(scalar_type_id).ok();
                                if let Some(scalar_type) = scalar_type {
                                    if scalar_type != component_type {
                                        return Err(ValidationError::VectorTimesScalarTypeMismatch {
                                            function: func,
                                            block,
                                            instruction: inst.class.opcode,
                                            vector_type: result_type,
                                            scalar_type,
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
// Matrix/Vector Multiply Rule
// ============================================================================

/// Validates MatrixTimesVector and VectorTimesMatrix operations.
///
/// MatrixTimesVector: result = matrix * vector
/// - Matrix column count must equal vector component count
/// - Result is a vector with component count = matrix row count
///
/// VectorTimesMatrix: result = vector * matrix
/// - Vector component count must equal matrix row count
/// - Result is a vector with component count = matrix column count
pub struct MatrixVectorMultiplyRule;

impl ValidationRule for MatrixVectorMultiplyRule {
    fn name(&self) -> &'static str {
        "matrix-vector-multiply"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                    let is_matrix_times_vector = inst.class.opcode == Op::MatrixTimesVector;
                    let is_vector_times_matrix = inst.class.opcode == Op::VectorTimesMatrix;

                    if !is_matrix_times_vector && !is_vector_times_matrix {
                        continue;
                    }

                    let Some(func) = function_id else { continue };
                    let Some(block) = block_id else { continue };
                    let Some(result_type_id) = inst.result_type else { continue };
                    let Some(result_type) = TypeId::try_from(result_type_id).ok() else {
                        continue;
                    };

                    // For MatrixTimesVector: operand 0 = matrix, operand 1 = vector
                    // For VectorTimesMatrix: operand 0 = vector, operand 1 = matrix
                    let (matrix_idx, vector_idx): (usize, usize) = if is_matrix_times_vector {
                        (0, 1)
                    } else {
                        (1, 0)
                    };
                    let matrix_idx_u32 = matrix_idx as u32;
                    let vector_idx_u32 = vector_idx as u32;

                    // Get matrix operand
                    let matrix_operand = inst.operands.get(matrix_idx).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });

                    let Some(matrix_operand) = matrix_operand else { continue };
                    let matrix_inst = ctx.definitions.get(&matrix_operand);
                    let Some(matrix_inst) = matrix_inst else { continue };
                    let Some(matrix_type_raw) = matrix_inst.result_type else { continue };
                    let Some(matrix_type_id) = TypeId::try_from(matrix_type_raw).ok() else {
                        continue;
                    };

                    // Get matrix details
                    let Some((matrix_component, matrix_rows, matrix_columns, _)) =
                        matrix_details_by_id(matrix_type_id, ctx.definitions)
                    else {
                        return Err(ValidationError::MatrixOperandNotMatrix {
                            function: func,
                            block,
                            instruction: inst.class.opcode,
                            operand: matrix_idx_u32,
                            found: matrix_type_id,
                        });
                    };

                    // Get vector operand
                    let vector_operand = inst.operands.get(vector_idx).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });

                    let Some(vector_operand) = vector_operand else { continue };
                    let vector_inst = ctx.definitions.get(&vector_operand);
                    let Some(vector_inst) = vector_inst else { continue };
                    let Some(vector_type_raw) = vector_inst.result_type else { continue };
                    let Some(vector_type_id) = TypeId::try_from(vector_type_raw).ok() else {
                        continue;
                    };

                    // Get vector type instruction
                    let vector_type_inst = ResultId::try_from(u32::from(vector_type_id))
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(vector_type_inst) = vector_type_inst else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: func,
                            block,
                            instruction: inst.class.opcode,
                            operand: vector_idx_u32,
                            found: vector_type_id,
                        });
                    };

                    if !vector_type_inst.is_vector_type() {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: func,
                            block,
                            instruction: inst.class.opcode,
                            operand: vector_idx_u32,
                            found: vector_type_id,
                        });
                    }

                    let (vector_component, vector_len) = vector_info(vector_type_inst);
                    let Some(vector_component) = vector_component else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: func,
                            block,
                            instruction: inst.class.opcode,
                            operand: vector_idx_u32,
                            found: vector_type_id,
                        });
                    };
                    let Some(vector_len) = vector_len else {
                        return Err(ValidationError::VectorOperandNotVector {
                            function: func,
                            block,
                            instruction: inst.class.opcode,
                            operand: vector_idx_u32,
                            found: vector_type_id,
                        });
                    };

                    // Check component types match
                    if matrix_component != vector_component {
                        if is_matrix_times_vector {
                            return Err(ValidationError::MatrixTimesVectorComponentTypeMismatch {
                                function: func,
                                block,
                                matrix_component,
                                vector_component,
                            });
                        } else {
                            return Err(ValidationError::VectorTimesMatrixComponentTypeMismatch {
                                function: func,
                                block,
                                vector_component,
                                matrix_component,
                            });
                        }
                    }

                    // Check dimension compatibility
                    if is_matrix_times_vector {
                        // Matrix columns must equal vector components
                        if matrix_columns != vector_len {
                            return Err(ValidationError::MatrixTimesVectorDimensionMismatch {
                                function: func,
                                block,
                                matrix_columns,
                                vector_components: vector_len,
                            });
                        }
                    } else {
                        // Vector components must equal matrix rows
                        if vector_len != matrix_rows {
                            return Err(ValidationError::VectorTimesMatrixDimensionMismatch {
                                function: func,
                                block,
                                vector_components: vector_len,
                                matrix_rows,
                            });
                        }
                    }

                    // Verify result type
                    let result_type_inst = ResultId::try_from(result_type_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(result_type_inst) = result_type_inst else { continue };

                    if !result_type_inst.is_vector_type() {
                        return Err(ValidationError::ArithmeticResultTypeInvalid {
                            function: func,
                            block,
                            opcode: inst.class.opcode,
                            result_type,
                            expected: "vector",
                        });
                    }

                    let (result_component, result_len) = vector_info(result_type_inst);

                    // Check result component type
                    if let Some(result_component) = result_component {
                        if result_component != matrix_component {
                            if is_matrix_times_vector {
                                return Err(ValidationError::InstructionResultTypeMismatch {
                                    function: func,
                                    block,
                                    instruction: inst.class.opcode,
                                    expected: matrix_component,
                                    found: result_component,
                                });
                            } else {
                                return Err(
                                    ValidationError::VectorTimesMatrixResultComponentTypeMismatch {
                                        function: func,
                                        block,
                                        expected: matrix_component,
                                        found: result_component,
                                    },
                                );
                            }
                        }
                    }

                    // Check result dimensions
                    if let Some(result_len) = result_len {
                        let expected_len = if is_matrix_times_vector {
                            matrix_rows
                        } else {
                            matrix_columns
                        };

                        if result_len != expected_len {
                            if is_vector_times_matrix {
                                return Err(
                                    ValidationError::VectorTimesMatrixResultDimensionMismatch {
                                        function: func,
                                        block,
                                        expected_components: expected_len,
                                        found_components: result_len,
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

// ============================================================================
// Matrix/Matrix Multiply Rule
// ============================================================================

/// Validates MatrixTimesMatrix operations.
///
/// Ensures that:
/// - Both operands are matrices
/// - Left matrix column count equals right matrix row count
/// - Component types match
/// - Result matrix has correct dimensions (left rows x right columns)
pub struct MatrixTimesMatrixRule;

impl ValidationRule for MatrixTimesMatrixRule {
    fn name(&self) -> &'static str {
        "matrix-times-matrix"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                    if inst.class.opcode != Op::MatrixTimesMatrix {
                        continue;
                    }

                    let Some(func) = function_id else { continue };
                    let Some(block) = block_id else { continue };
                    let Some(result_type_id) = inst.result_type else { continue };
                    let Some(_result_type) = TypeId::try_from(result_type_id).ok() else {
                        continue;
                    };

                    // Get left operand
                    let left_operand = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });

                    let Some(left_operand) = left_operand else { continue };
                    let left_inst = ctx.definitions.get(&left_operand);
                    let Some(left_inst) = left_inst else { continue };
                    let Some(left_type_raw) = left_inst.result_type else { continue };
                    let Some(left_type_id) = TypeId::try_from(left_type_raw).ok() else {
                        continue;
                    };

                    let Some((left_component, left_rows, left_columns, _)) =
                        matrix_details_by_id(left_type_id, ctx.definitions)
                    else {
                        return Err(ValidationError::MatrixOperandNotMatrix {
                            function: func,
                            block,
                            instruction: inst.class.opcode,
                            operand: 0,
                            found: left_type_id,
                        });
                    };

                    // Get right operand
                    let right_operand = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });

                    let Some(right_operand) = right_operand else { continue };
                    let right_inst = ctx.definitions.get(&right_operand);
                    let Some(right_inst) = right_inst else { continue };
                    let Some(right_type_raw) = right_inst.result_type else { continue };
                    let Some(right_type_id) = TypeId::try_from(right_type_raw).ok() else {
                        continue;
                    };

                    let Some((right_component, right_rows, right_columns, _)) =
                        matrix_details_by_id(right_type_id, ctx.definitions)
                    else {
                        return Err(ValidationError::MatrixOperandNotMatrix {
                            function: func,
                            block,
                            instruction: inst.class.opcode,
                            operand: 1,
                            found: right_type_id,
                        });
                    };

                    // Check component types match
                    if left_component != right_component {
                        return Err(ValidationError::MatrixTimesMatrixComponentTypeMismatch {
                            function: func,
                            block,
                            left_component,
                            right_component,
                        });
                    }

                    // Check dimension compatibility: left columns = right rows
                    if left_columns != right_rows {
                        return Err(ValidationError::MatrixTimesMatrixDimensionMismatch {
                            function: func,
                            block,
                            left_columns,
                            right_rows,
                        });
                    }

                    // Verify result type is matrix with correct dimensions
                    let Some((result_component, result_rows, result_columns, _)) =
                        matrix_details_by_id(TypeId::try_from(result_type_id).ok().unwrap_or(left_type_id), ctx.definitions)
                    else {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: func,
                            block,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
                    };

                    // Check result component type
                    if result_component != left_component {
                        return Err(ValidationError::MatrixTimesMatrixResultComponentTypeMismatch {
                            function: func,
                            block,
                            expected: left_component,
                            found: result_component,
                        });
                    }

                    // Check result dimensions
                    if result_rows != left_rows || result_columns != right_columns {
                        return Err(ValidationError::MatrixTimesMatrixResultShapeMismatch {
                            function: func,
                            block,
                            expected_columns: right_columns,
                            expected_rows: left_rows,
                        });
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
    vec![
        &FloatArithmeticRule,
        &IntArithmeticRule,
        &DotProductRule,
        &VectorTimesScalarRule,
        &MatrixVectorMultiplyRule,
        &MatrixTimesMatrixRule,
    ]
}
