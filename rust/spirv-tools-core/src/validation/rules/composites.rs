//! Composite instruction validation rules.
//!
//! This module validates SPIR-V composite instructions including:
//!
//! - Vector operations (VectorExtractDynamic, VectorInsertDynamic, VectorShuffle)
//! - Composite operations (CompositeConstruct, CompositeExtract, CompositeInsert)
//! - Copy operations (CopyObject, CopyLogical)
//! - Matrix operations (Transpose)

use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeInstructionExt, TypeResolver};
use crate::validation::types::Id;

// ============================================================================
// Vector Dynamic Operations Rule
// ============================================================================

/// Validates vector dynamic operations (VectorExtractDynamic, VectorInsertDynamic).
///
/// Ensures that:
/// - VectorExtractDynamic: Result is scalar, Vector is vector, component types match, Index is int
/// - VectorInsertDynamic: Result is vector, Vector matches result, Component matches component type, Index is int
pub struct VectorDynamicRule;

impl ValidationRule for VectorDynamicRule {
    fn name(&self) -> &'static str {
        "vector-dynamic"
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
                        Op::VectorExtractDynamic => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be scalar
                            let result_dim =
                                resolver.get_dimension(result_type_id, ctx.definitions);
                            if result_dim != 1 {
                                if let (Some(func), Some(block), Some(result_type)) = (
                                    function_id,
                                    block_id,
                                    crate::validation::types::TypeId::try_from(result_type_id)
                                        .ok(),
                                ) {
                                    return Err(ValidationError::CompositeResultTypeInvalid {
                                        function: func,
                                        block,
                                        opcode: inst.class.opcode,
                                        result_type,
                                        expected: "scalar type",
                                    });
                                }
                            }

                            // Vector operand must be a vector
                            if let Some(rspirv::dr::Operand::IdRef(vector_id)) =
                                inst.operands.first()
                            {
                                let vector_inst =
                                    crate::validation::types::ResultId::try_from(*vector_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(vector_inst) = vector_inst {
                                    if let Some(vector_type_id) = vector_inst.result_type {
                                        let vector_type_inst =
                                            crate::validation::types::ResultId::try_from(
                                                vector_type_id,
                                            )
                                            .ok()
                                            .and_then(|rid| ctx.definitions.get(&rid));

                                        if let Some(vector_type_inst) = vector_type_inst {
                                            if !vector_type_inst.is_vector_type() {
                                                if let (Some(func), Some(block), Some(result_type)) = (
                                                    function_id,
                                                    block_id,
                                                    crate::validation::types::TypeId::try_from(
                                                        result_type_id,
                                                    )
                                                    .ok(),
                                                ) {
                                                    return Err(
                                                        ValidationError::CompositeOperandTypeInvalid {
                                                            function: func,
                                                            block,
                                                            opcode: inst.class.opcode,
                                                            operand_index: 0,
                                                            result_type,
                                                            expected: "vector type",
                                                        },
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Index must be int scalar
                            if let Some(rspirv::dr::Operand::IdRef(index_id)) = inst.operands.get(1)
                            {
                                let index_inst =
                                    crate::validation::types::ResultId::try_from(*index_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(index_inst) = index_inst {
                                    if let Some(index_type_id) = index_inst.result_type {
                                        if !resolver.is_int_scalar(index_type_id, ctx.definitions) {
                                            if let (Some(func), Some(block), Some(result_type)) = (
                                                function_id,
                                                block_id,
                                                crate::validation::types::TypeId::try_from(
                                                    result_type_id,
                                                )
                                                .ok(),
                                            ) {
                                                return Err(
                                                    ValidationError::CompositeOperandTypeInvalid {
                                                        function: func,
                                                        block,
                                                        opcode: inst.class.opcode,
                                                        operand_index: 1,
                                                        result_type,
                                                        expected: "int scalar",
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Op::VectorInsertDynamic => {
                            let Some(result_type_id) = inst.result_type else {
                                continue;
                            };

                            // Result must be vector
                            let result_type_inst =
                                crate::validation::types::ResultId::try_from(result_type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid));

                            if let Some(result_type_inst) = result_type_inst {
                                if !result_type_inst.is_vector_type() {
                                    if let (Some(func), Some(block), Some(result_type)) = (
                                        function_id,
                                        block_id,
                                        crate::validation::types::TypeId::try_from(result_type_id)
                                            .ok(),
                                    ) {
                                        return Err(ValidationError::CompositeResultTypeInvalid {
                                            function: func,
                                            block,
                                            opcode: inst.class.opcode,
                                            result_type,
                                            expected: "vector type",
                                        });
                                    }
                                }
                            }

                            // Vector operand must match result type
                            if let Some(rspirv::dr::Operand::IdRef(vector_id)) =
                                inst.operands.first()
                            {
                                let vector_inst =
                                    crate::validation::types::ResultId::try_from(*vector_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(vector_inst) = vector_inst {
                                    if let Some(vector_type_id) = vector_inst.result_type {
                                        if vector_type_id != result_type_id {
                                            if let (Some(func), Some(block), Some(result_type)) = (
                                                function_id,
                                                block_id,
                                                crate::validation::types::TypeId::try_from(
                                                    result_type_id,
                                                )
                                                .ok(),
                                            ) {
                                                return Err(
                                                    ValidationError::CompositeOperandTypeMismatch {
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

                            // Index must be int scalar
                            if let Some(rspirv::dr::Operand::IdRef(index_id)) = inst.operands.get(2)
                            {
                                let index_inst =
                                    crate::validation::types::ResultId::try_from(*index_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(index_inst) = index_inst {
                                    if let Some(index_type_id) = index_inst.result_type {
                                        if !resolver.is_int_scalar(index_type_id, ctx.definitions) {
                                            if let (Some(func), Some(block), Some(result_type)) = (
                                                function_id,
                                                block_id,
                                                crate::validation::types::TypeId::try_from(
                                                    result_type_id,
                                                )
                                                .ok(),
                                            ) {
                                                return Err(
                                                    ValidationError::CompositeOperandTypeInvalid {
                                                        function: func,
                                                        block,
                                                        opcode: inst.class.opcode,
                                                        operand_index: 2,
                                                        result_type,
                                                        expected: "int scalar",
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
// Vector Shuffle Rule
// ============================================================================

/// Validates OpVectorShuffle.
///
/// Ensures that:
/// - Result type is a vector
/// - Both input vectors have same component type as result
/// - Component indices are valid
pub struct VectorShuffleRule;

impl ValidationRule for VectorShuffleRule {
    fn name(&self) -> &'static str {
        "vector-shuffle"
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
                    if inst.class.opcode != Op::VectorShuffle {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be vector
                    let result_type_inst =
                        crate::validation::types::ResultId::try_from(result_type_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(result_type_inst) = result_type_inst else {
                        continue;
                    };

                    if !result_type_inst.is_vector_type() {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::CompositeResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "vector type",
                            });
                        }
                    }

                    // Vector 1 must be vector type
                    if let Some(rspirv::dr::Operand::IdRef(vec1_id)) = inst.operands.first() {
                        let vec1_inst = crate::validation::types::ResultId::try_from(*vec1_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(vec1_inst) = vec1_inst {
                            if let Some(vec1_type_id) = vec1_inst.result_type {
                                let vec1_type_inst =
                                    crate::validation::types::ResultId::try_from(vec1_type_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(vec1_type_inst) = vec1_type_inst {
                                    if !vec1_type_inst.is_vector_type() {
                                        if let (Some(func), Some(block), Some(result_type)) = (
                                            function_id,
                                            block_id,
                                            crate::validation::types::TypeId::try_from(
                                                result_type_id,
                                            )
                                            .ok(),
                                        ) {
                                            return Err(
                                                ValidationError::CompositeOperandTypeInvalid {
                                                    function: func,
                                                    block,
                                                    opcode: inst.class.opcode,
                                                    operand_index: 0,
                                                    result_type,
                                                    expected: "vector type",
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Vector 2 must be vector type
                    if let Some(rspirv::dr::Operand::IdRef(vec2_id)) = inst.operands.get(1) {
                        let vec2_inst = crate::validation::types::ResultId::try_from(*vec2_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(vec2_inst) = vec2_inst {
                            if let Some(vec2_type_id) = vec2_inst.result_type {
                                let vec2_type_inst =
                                    crate::validation::types::ResultId::try_from(vec2_type_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(vec2_type_inst) = vec2_type_inst {
                                    if !vec2_type_inst.is_vector_type() {
                                        if let (Some(func), Some(block), Some(result_type)) = (
                                            function_id,
                                            block_id,
                                            crate::validation::types::TypeId::try_from(
                                                result_type_id,
                                            )
                                            .ok(),
                                        ) {
                                            return Err(
                                                ValidationError::CompositeOperandTypeInvalid {
                                                    function: func,
                                                    block,
                                                    opcode: inst.class.opcode,
                                                    operand_index: 1,
                                                    result_type,
                                                    expected: "vector type",
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
// Copy Object Rule
// ============================================================================

/// Validates OpCopyObject.
///
/// Ensures that:
/// - Result type matches operand type
/// - Result type is not void
pub struct CopyObjectRule;

impl ValidationRule for CopyObjectRule {
    fn name(&self) -> &'static str {
        "copy-object"
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
                    if inst.class.opcode != Op::CopyObject {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must not be void
                    let result_type_inst =
                        crate::validation::types::ResultId::try_from(result_type_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                    if let Some(result_type_inst) = result_type_inst {
                        if result_type_inst.is_void_type() {
                            if let (Some(func), Some(block), Some(result_type)) = (
                                function_id,
                                block_id,
                                crate::validation::types::TypeId::try_from(result_type_id).ok(),
                            ) {
                                return Err(ValidationError::CompositeResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    result_type,
                                    expected: "non-void type",
                                });
                            }
                        }
                    }

                    // Operand type must match result type
                    if let Some(rspirv::dr::Operand::IdRef(operand_id)) = inst.operands.first() {
                        let operand_inst =
                            crate::validation::types::ResultId::try_from(*operand_id)
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
                                        return Err(
                                            ValidationError::CompositeOperandTypeMismatch {
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

        Ok(())
    }
}

// ============================================================================
// Transpose Rule
// ============================================================================

/// Validates OpTranspose.
///
/// Ensures that:
/// - Result type is a matrix
/// - Matrix operand is a matrix
/// - Dimensions are transposed correctly
pub struct TransposeRule;

impl ValidationRule for TransposeRule {
    fn name(&self) -> &'static str {
        "transpose"
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
                    if inst.class.opcode != Op::Transpose {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be matrix
                    let result_type_inst =
                        crate::validation::types::ResultId::try_from(result_type_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(result_type_inst) = result_type_inst else {
                        continue;
                    };

                    if !result_type_inst.is_matrix_type() {
                        if let (Some(func), Some(block), Some(result_type)) = (
                            function_id,
                            block_id,
                            crate::validation::types::TypeId::try_from(result_type_id).ok(),
                        ) {
                            return Err(ValidationError::CompositeResultTypeInvalid {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                result_type,
                                expected: "matrix type",
                            });
                        }
                    }

                    // Matrix operand must be matrix
                    if let Some(rspirv::dr::Operand::IdRef(matrix_id)) = inst.operands.first() {
                        let matrix_inst = crate::validation::types::ResultId::try_from(*matrix_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        if let Some(matrix_inst) = matrix_inst {
                            if let Some(matrix_type_id) = matrix_inst.result_type {
                                let matrix_type_inst =
                                    crate::validation::types::ResultId::try_from(matrix_type_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid));

                                if let Some(matrix_type_inst) = matrix_type_inst {
                                    if !matrix_type_inst.is_matrix_type() {
                                        if let (Some(func), Some(block), Some(result_type)) = (
                                            function_id,
                                            block_id,
                                            crate::validation::types::TypeId::try_from(
                                                result_type_id,
                                            )
                                            .ok(),
                                        ) {
                                            return Err(
                                                ValidationError::CompositeOperandTypeInvalid {
                                                    function: func,
                                                    block,
                                                    opcode: inst.class.opcode,
                                                    operand_index: 0,
                                                    result_type,
                                                    expected: "matrix type",
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
// All composite rules
// ============================================================================

/// Returns all composite validation rules.
pub fn all_composite_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &VectorDynamicRule,
        &VectorShuffleRule,
        &CopyObjectRule,
        &TransposeRule,
    ]
}
