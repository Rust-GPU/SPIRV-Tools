//! Composite instruction validation rules.
//!
//! This module validates SPIR-V composite instructions including:
//!
//! - Vector operations (VectorExtractDynamic, VectorInsertDynamic, VectorShuffle)
//! - Composite operations (CompositeConstruct, CompositeExtract, CompositeInsert)
//! - Copy operations (CopyObject, CopyLogical)
//! - Matrix operations (Transpose)

use rspirv::spirv::{Capability, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::vector_info;
use crate::validation::type_ext::TypeInstructionExt;
use crate::validation::types::{Id, ResultId, TypeId};
use crate::validation::ValidationResult;

// ============================================================================
// Vector Dynamic Operations Rule
// ============================================================================

/// Validates vector dynamic operations (VectorExtractDynamic, VectorInsertDynamic).
///
/// Ensures that:
/// - VectorExtractDynamic: Result matches vector component type, Vector is vector, Index is int
/// - VectorInsertDynamic: Result is vector matching input, Component matches vector element type, Index is int
pub struct VectorDynamicRule;

impl ValidationRule for VectorDynamicRule {
    fn name(&self) -> &'static str {
        "vector-dynamic"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                    let (Some(func), Some(blk)) = (function_id, block_id) else {
                        continue;
                    };

                    match inst.class.opcode {
                        Op::VectorExtractDynamic => {
                            let Some(result_type_id) =
                                inst.result_type.and_then(|r| TypeId::try_from(r).ok())
                            else {
                                continue;
                            };

                            // Vector operand must be a vector
                            let vector_operand = inst.operands.first().and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                                _ => None,
                            });
                            let Some(vector_operand) = vector_operand else {
                                continue;
                            };
                            let vector_type_id = ctx
                                .definitions
                                .get(&vector_operand)
                                .and_then(|inst| inst.result_type)
                                .and_then(|t| TypeId::try_from(t).ok());
                            let Some(vector_type_id) = vector_type_id else {
                                continue;
                            };

                            let vector_type_inst = ResultId::try_from(u32::from(vector_type_id))
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid));
                            let Some(vector_type_inst) = vector_type_inst else {
                                return Err(ValidationError::VectorOperandNotVector {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    operand: 0,
                                    found: vector_type_id,
                                }
                                .into());
                            };
                            if vector_type_inst.class.opcode != Op::TypeVector {
                                return Err(ValidationError::VectorOperandNotVector {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    operand: 0,
                                    found: vector_type_id,
                                }
                                .into());
                            }

                            // Component type must match result type
                            let (component_type, _) = vector_info(vector_type_inst);
                            if let Some(component_type) = component_type {
                                if component_type != result_type_id {
                                    return Err(ValidationError::InstructionResultTypeMismatch {
                                        function: func,
                                        block: blk,
                                        instruction: inst.class.opcode,
                                        expected: component_type,
                                        found: result_type_id,
                                    }
                                    .into());
                                }
                            }

                            // Index must be int scalar
                            let index_operand = inst.operands.get(1).and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                                _ => None,
                            });
                            if let Some(index_operand) = index_operand {
                                let index_type_id = ctx
                                    .definitions
                                    .get(&index_operand)
                                    .and_then(|inst| inst.result_type)
                                    .and_then(|t| TypeId::try_from(t).ok());
                                if let Some(index_type_id) = index_type_id {
                                    let index_type_inst =
                                        ResultId::try_from(u32::from(index_type_id))
                                            .ok()
                                            .and_then(|rid| ctx.definitions.get(&rid));
                                    if let Some(index_type_inst) = index_type_inst {
                                        if index_type_inst.class.opcode != Op::TypeInt {
                                            return Err(ValidationError::VectorIndexTypeInvalid {
                                                function: func,
                                                block: blk,
                                                instruction: inst.class.opcode,
                                                operand_index: 1,
                                                found: index_type_id,
                                            }
                                            .into());
                                        }
                                    }
                                }
                            }

                            // Shader capability restricts 8/16-bit types
                            if ctx.has_capability(Capability::Shader)
                                && contains_limited_type(u32::from(result_type_id), ctx.definitions)
                            {
                                return Err(ValidationError::VectorDynamicLimitedType {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    operation: "extract from",
                                }
                                .into());
                            }
                        }
                        Op::VectorInsertDynamic => {
                            let Some(result_type_id) =
                                inst.result_type.and_then(|r| TypeId::try_from(r).ok())
                            else {
                                continue;
                            };

                            // Vector operand must be a vector
                            let vector_operand = inst.operands.first().and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                                _ => None,
                            });
                            let Some(vector_operand) = vector_operand else {
                                continue;
                            };
                            let vector_type_id = ctx
                                .definitions
                                .get(&vector_operand)
                                .and_then(|inst| inst.result_type)
                                .and_then(|t| TypeId::try_from(t).ok());
                            let Some(vector_type_id) = vector_type_id else {
                                continue;
                            };

                            let vector_type_inst = ResultId::try_from(u32::from(vector_type_id))
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid));
                            let Some(vector_type_inst) = vector_type_inst else {
                                return Err(ValidationError::VectorOperandNotVector {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    operand: 0,
                                    found: vector_type_id,
                                }
                                .into());
                            };
                            if vector_type_inst.class.opcode != Op::TypeVector {
                                return Err(ValidationError::VectorOperandNotVector {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    operand: 0,
                                    found: vector_type_id,
                                }
                                .into());
                            }

                            // Result type must match vector type
                            if result_type_id != vector_type_id {
                                return Err(ValidationError::InstructionResultTypeMismatch {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    expected: vector_type_id,
                                    found: result_type_id,
                                }
                                .into());
                            }

                            // Component operand must match vector element type
                            let (component_type, _) = vector_info(vector_type_inst);
                            if let Some(component_type) = component_type {
                                let component_operand =
                                    inst.operands.get(1).and_then(|op| match op {
                                        rspirv::dr::Operand::IdRef(id) => {
                                            ResultId::try_from(*id).ok()
                                        }
                                        _ => None,
                                    });
                                if let Some(component_operand) = component_operand {
                                    let component_operand_type = ctx
                                        .definitions
                                        .get(&component_operand)
                                        .and_then(|inst| inst.result_type)
                                        .and_then(|t| TypeId::try_from(t).ok());
                                    if let Some(component_operand_type) = component_operand_type {
                                        if component_operand_type != component_type {
                                            return Err(ValidationError::OperandTypeMismatch {
                                                function: func,
                                                block: blk,
                                                instruction: inst.class.opcode,
                                                operand_index: 1,
                                                expected: component_type,
                                                found: component_operand_type,
                                            }
                                            .into());
                                        }
                                    }
                                }
                            }

                            // Index must be int scalar
                            let index_operand = inst.operands.get(2).and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                                _ => None,
                            });
                            if let Some(index_operand) = index_operand {
                                let index_type_id = ctx
                                    .definitions
                                    .get(&index_operand)
                                    .and_then(|inst| inst.result_type)
                                    .and_then(|t| TypeId::try_from(t).ok());
                                if let Some(index_type_id) = index_type_id {
                                    let index_type_inst =
                                        ResultId::try_from(u32::from(index_type_id))
                                            .ok()
                                            .and_then(|rid| ctx.definitions.get(&rid));
                                    if let Some(index_type_inst) = index_type_inst {
                                        if index_type_inst.class.opcode != Op::TypeInt {
                                            return Err(ValidationError::VectorIndexTypeInvalid {
                                                function: func,
                                                block: blk,
                                                instruction: inst.class.opcode,
                                                operand_index: 2,
                                                found: index_type_id,
                                            }
                                            .into());
                                        }
                                    }
                                }
                            }

                            // Shader capability restricts 8/16-bit types
                            if ctx.has_capability(Capability::Shader)
                                && contains_limited_type(u32::from(result_type_id), ctx.definitions)
                            {
                                return Err(ValidationError::VectorDynamicLimitedType {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    operation: "insert into",
                                }
                                .into());
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
/// - Component indices are valid (0..N+M-1, or 0xFFFFFFFF for undefined)
pub struct VectorShuffleRule;

impl ValidationRule for VectorShuffleRule {
    fn name(&self) -> &'static str {
        "vector-shuffle"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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

                    let (Some(func), Some(blk)) = (function_id, block_id) else {
                        continue;
                    };

                    let Some(result_type_id) =
                        inst.result_type.and_then(|r| TypeId::try_from(r).ok())
                    else {
                        continue;
                    };

                    // Helper to get vector type info
                    let get_vector_type_info =
                        |ty: TypeId| -> Result<(TypeId, u32), ValidationError> {
                            let type_inst = ResultId::try_from(u32::from(ty))
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid));
                            let Some(type_inst) = type_inst else {
                                return Err(ValidationError::VectorShuffleOperandNotVector {
                                    function: func,
                                    block: blk,
                                    operand: 0,
                                    found: ty,
                                });
                            };
                            if type_inst.class.opcode != Op::TypeVector {
                                return Err(ValidationError::VectorShuffleOperandNotVector {
                                    function: func,
                                    block: blk,
                                    operand: 0,
                                    found: ty,
                                });
                            }
                            let (elem, count) = vector_info(type_inst);
                            match (elem, count) {
                                (Some(elem), Some(count)) => Ok((elem, count)),
                                _ => Err(ValidationError::VectorShuffleOperandNotVector {
                                    function: func,
                                    block: blk,
                                    operand: 0,
                                    found: ty,
                                }),
                            }
                        };

                    // Get vector 1 type
                    let vec1_id = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(vec1_id) = vec1_id else {
                        continue;
                    };
                    let vec1_type_id = ctx
                        .definitions
                        .get(&vec1_id)
                        .and_then(|inst| inst.result_type)
                        .and_then(|t| TypeId::try_from(t).ok());
                    let Some(vec1_type_id) = vec1_type_id else {
                        continue;
                    };

                    // Get vector 2 type
                    let vec2_id = inst.operands.get(1).and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                        _ => None,
                    });
                    let Some(vec2_id) = vec2_id else {
                        continue;
                    };
                    let vec2_type_id = ctx
                        .definitions
                        .get(&vec2_id)
                        .and_then(|inst| inst.result_type)
                        .and_then(|t| TypeId::try_from(t).ok());
                    let Some(vec2_type_id) = vec2_type_id else {
                        continue;
                    };

                    // Validate vector 1
                    let (vec1_component, vec1_len) = match get_vector_type_info(vec1_type_id) {
                        Ok(info) => info,
                        Err(mut err) => {
                            if let ValidationError::VectorShuffleOperandNotVector {
                                ref mut operand,
                                ..
                            } = err
                            {
                                *operand = 0;
                            }
                            return Err(err.into());
                        }
                    };

                    // Validate vector 2
                    let (vec2_component, vec2_len) = match get_vector_type_info(vec2_type_id) {
                        Ok(info) => info,
                        Err(mut err) => {
                            if let ValidationError::VectorShuffleOperandNotVector {
                                ref mut operand,
                                ..
                            } = err
                            {
                                *operand = 1;
                            }
                            return Err(err.into());
                        }
                    };

                    // Component types must match
                    if vec1_component != vec2_component {
                        return Err(ValidationError::VectorShuffleComponentTypeMismatch {
                            function: func,
                            block: blk,
                            first: vec1_component,
                            second: vec2_component,
                        }
                        .into());
                    }

                    // Result type must be a vector with matching component type
                    let result_vector_inst = ResultId::try_from(u32::from(result_type_id))
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid));
                    let Some(result_vector_inst) = result_vector_inst else {
                        continue;
                    };
                    if result_vector_inst.class.opcode != Op::TypeVector {
                        return Err(ValidationError::VectorShuffleResultTypeMismatch {
                            function: func,
                            block: blk,
                            result_type: result_type_id,
                            component_type: vec1_component,
                        }
                        .into());
                    }
                    let (result_component, result_len) = vector_info(result_vector_inst);
                    let Some(result_component) = result_component else {
                        return Err(ValidationError::VectorShuffleResultTypeMismatch {
                            function: func,
                            block: blk,
                            result_type: result_type_id,
                            component_type: vec1_component,
                        }
                        .into());
                    };
                    if result_component != vec1_component {
                        return Err(ValidationError::VectorShuffleResultTypeMismatch {
                            function: func,
                            block: blk,
                            result_type: result_type_id,
                            component_type: vec1_component,
                        }
                        .into());
                    }

                    // Validate component count matches result type
                    let literal_components: Vec<u32> = inst
                        .operands
                        .iter()
                        .skip(2)
                        .filter_map(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            rspirv::dr::Operand::LiteralBit64(v) => u32::try_from(*v).ok(),
                            _ => None,
                        })
                        .collect();
                    let operand_component_count = literal_components.len() as u32;
                    let Some(result_component_len) = result_len else {
                        return Err(ValidationError::VectorShuffleComponentCountMismatch {
                            function: func,
                            block: blk,
                            operand_components: operand_component_count,
                            result_components: 0,
                        }
                        .into());
                    };
                    if operand_component_count != result_component_len {
                        return Err(ValidationError::VectorShuffleComponentCountMismatch {
                            function: func,
                            block: blk,
                            operand_components: operand_component_count,
                            result_components: result_component_len,
                        }
                        .into());
                    }

                    // Validate component indices are in range
                    let max_index = vec1_len + vec2_len;
                    for value in literal_components {
                        // 0xFFFFFFFF is special "undefined" value
                        if value == u32::MAX {
                            continue;
                        }
                        if value >= max_index {
                            return Err(ValidationError::VectorShuffleComponentOutOfRange {
                                function: func,
                                block: blk,
                                value,
                                max: max_index.saturating_sub(1),
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                                }
                                .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                            }
                            .into());
                        }
                    }

                    // Matrix operand must be matrix and dimensions must be transposed
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
                                                }
                                                .into(),
                                            );
                                        }
                                    }

                                    // Validate transpose dimensions:
                                    // Input MxN (M columns of N-row vectors) -> Result NxM
                                    if matrix_type_inst.is_matrix_type()
                                        && result_type_inst.is_matrix_type()
                                    {
                                        let input_col_count =
                                            matrix_type_inst.matrix_column_count();
                                        let input_col_type_id =
                                            matrix_type_inst.matrix_column_type_id();
                                        let result_col_count =
                                            result_type_inst.matrix_column_count();
                                        let result_col_type_id =
                                            result_type_inst.matrix_column_type_id();

                                        // Get row counts from column vector types
                                        let input_row_count = input_col_type_id
                                            .and_then(|id| ResultId::try_from(id).ok())
                                            .and_then(|rid| ctx.definitions.get(&rid))
                                            .and_then(|inst| inst.vector_component_count());
                                        let result_row_count = result_col_type_id
                                            .and_then(|id| ResultId::try_from(id).ok())
                                            .and_then(|rid| ctx.definitions.get(&rid))
                                            .and_then(|inst| inst.vector_component_count());

                                        // Check that result columns = input rows
                                        // and result rows = input columns
                                        if let (
                                            Some(in_cols),
                                            Some(in_rows),
                                            Some(res_cols),
                                            Some(res_rows),
                                        ) = (
                                            input_col_count,
                                            input_row_count,
                                            result_col_count,
                                            result_row_count,
                                        ) {
                                            if res_cols != in_rows || res_rows != in_cols {
                                                return Err(
                                                    ValidationError::TransposeDimensionMismatch {
                                                        function: function_id,
                                                        block: block_id,
                                                    }
                                                    .into(),
                                                );
                                            }
                                        }

                                        // Also check that component types match
                                        let input_component_type = input_col_type_id
                                            .and_then(|id| ResultId::try_from(id).ok())
                                            .and_then(|rid| ctx.definitions.get(&rid))
                                            .and_then(|inst| inst.vector_component_type_id());
                                        let result_component_type = result_col_type_id
                                            .and_then(|id| ResultId::try_from(id).ok())
                                            .and_then(|rid| ctx.definitions.get(&rid))
                                            .and_then(|inst| inst.vector_component_type_id());

                                        if let (Some(in_comp), Some(res_comp)) =
                                            (input_component_type, result_component_type)
                                        {
                                            if in_comp != res_comp {
                                                return Err(
                                                    ValidationError::TransposeDimensionMismatch {
                                                        function: function_id,
                                                        block: block_id,
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
        }

        Ok(())
    }
}

// ============================================================================
// Composite Extract/Insert Rule
// ============================================================================

/// Validates OpCompositeExtract and OpCompositeInsert.
///
/// Ensures that:
/// - CompositeExtract: Result type matches the component type at the given indices
/// - CompositeInsert: Object type matches the component type at indices, result type matches composite
/// - Index values are in bounds for the composite type
pub struct CompositeExtractInsertRule;

impl ValidationRule for CompositeExtractInsertRule {
    fn name(&self) -> &'static str {
        "composite-extract-insert"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                    let (Some(func), Some(blk)) = (function_id, block_id) else {
                        continue;
                    };

                    match inst.class.opcode {
                        Op::CompositeExtract => {
                            let Some(result_type_id) =
                                inst.result_type.and_then(|r| TypeId::try_from(r).ok())
                            else {
                                continue;
                            };

                            // Get composite operand type
                            let composite_operand = inst.operands.first().and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                                _ => None,
                            });
                            let Some(composite_operand) = composite_operand else {
                                continue;
                            };
                            let composite_type_id = ctx
                                .definitions
                                .get(&composite_operand)
                                .and_then(|inst| inst.result_type)
                                .and_then(|t| TypeId::try_from(t).ok());
                            let Some(composite_type_id) = composite_type_id else {
                                continue;
                            };

                            // Collect literal indices
                            let indices: Vec<u32> = inst
                                .operands
                                .iter()
                                .skip(1)
                                .filter_map(|op| match op {
                                    rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                                    _ => None,
                                })
                                .collect();

                            // Validate at least one index is present
                            if indices.is_empty() {
                                return Err(ValidationError::CompositeExtractInsertNoIndices {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                }
                                .into());
                            }

                            // Validate maximum of 255 indices
                            const MAX_COMPOSITE_INDICES: usize = 255;
                            if indices.len() > MAX_COMPOSITE_INDICES {
                                return Err(
                                    ValidationError::CompositeExtractInsertTooManyIndices {
                                        function: func,
                                        block: blk,
                                        instruction: inst.class.opcode,
                                        count: indices.len(),
                                    }
                                    .into(),
                                );
                            }

                            // Walk the composite type to find the component type
                            match walk_composite_type(composite_type_id, &indices, ctx.definitions)
                            {
                                Ok(component_type) => {
                                    if component_type != result_type_id {
                                        return Err(
                                            ValidationError::CompositeOperandTypeMismatch {
                                                function: func,
                                                block: blk,
                                                opcode: inst.class.opcode,
                                                result_type: result_type_id,
                                            }
                                            .into(),
                                        );
                                    }
                                }
                                Err(CompositeWalkError::OutOfBounds {
                                    composite_type,
                                    index_position,
                                    index,
                                    bound,
                                }) => {
                                    return Err(ValidationError::CompositeIndexOutOfBounds {
                                        function: func,
                                        block: blk,
                                        instruction: inst.class.opcode,
                                        composite_type,
                                        index_position,
                                        index,
                                        bound,
                                    }
                                    .into());
                                }
                                Err(CompositeWalkError::NotComposite) => {
                                    // Type not found or not a composite - skip
                                    continue;
                                }
                            }
                        }
                        Op::CompositeInsert => {
                            let Some(result_type_id) =
                                inst.result_type.and_then(|r| TypeId::try_from(r).ok())
                            else {
                                continue;
                            };

                            // Get object operand (what we're inserting)
                            let object_operand = inst.operands.first().and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                                _ => None,
                            });
                            let Some(object_operand) = object_operand else {
                                continue;
                            };
                            let object_type_id = ctx
                                .definitions
                                .get(&object_operand)
                                .and_then(|inst| inst.result_type)
                                .and_then(|t| TypeId::try_from(t).ok());
                            let Some(object_type_id) = object_type_id else {
                                continue;
                            };

                            // Get composite operand (where we're inserting)
                            let composite_operand = inst.operands.get(1).and_then(|op| match op {
                                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                                _ => None,
                            });
                            let Some(composite_operand) = composite_operand else {
                                continue;
                            };
                            let composite_type_id = ctx
                                .definitions
                                .get(&composite_operand)
                                .and_then(|inst| inst.result_type)
                                .and_then(|t| TypeId::try_from(t).ok());
                            let Some(composite_type_id) = composite_type_id else {
                                continue;
                            };

                            // Result type must match composite type
                            if result_type_id != composite_type_id {
                                return Err(ValidationError::CompositeResultTypeInvalid {
                                    function: func,
                                    block: blk,
                                    opcode: inst.class.opcode,
                                    result_type: result_type_id,
                                    expected: "same type as composite operand",
                                }
                                .into());
                            }

                            // Collect literal indices
                            let indices: Vec<u32> = inst
                                .operands
                                .iter()
                                .skip(2)
                                .filter_map(|op| match op {
                                    rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                                    _ => None,
                                })
                                .collect();

                            // Validate at least one index is present
                            if indices.is_empty() {
                                return Err(ValidationError::CompositeExtractInsertNoIndices {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                }
                                .into());
                            }

                            // Validate maximum of 255 indices
                            const MAX_COMPOSITE_INDICES: usize = 255;
                            if indices.len() > MAX_COMPOSITE_INDICES {
                                return Err(
                                    ValidationError::CompositeExtractInsertTooManyIndices {
                                        function: func,
                                        block: blk,
                                        instruction: inst.class.opcode,
                                        count: indices.len(),
                                    }
                                    .into(),
                                );
                            }

                            // Walk the composite type to find the component type
                            match walk_composite_type(composite_type_id, &indices, ctx.definitions)
                            {
                                Ok(component_type) => {
                                    if component_type != object_type_id {
                                        return Err(ValidationError::CompositeOperandTypeInvalid {
                                            function: func,
                                            block: blk,
                                            opcode: inst.class.opcode,
                                            operand_index: 0,
                                            result_type: result_type_id,
                                            expected: "matching component type",
                                        }
                                        .into());
                                    }
                                }
                                Err(CompositeWalkError::OutOfBounds {
                                    composite_type,
                                    index_position,
                                    index,
                                    bound,
                                }) => {
                                    return Err(ValidationError::CompositeIndexOutOfBounds {
                                        function: func,
                                        block: blk,
                                        instruction: inst.class.opcode,
                                        composite_type,
                                        index_position,
                                        index,
                                        bound,
                                    }
                                    .into());
                                }
                                Err(CompositeWalkError::NotComposite) => {
                                    // Type not found or not a composite - skip
                                    continue;
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

/// Error when walking a composite type hierarchy.
#[derive(Debug)]
enum CompositeWalkError {
    NotComposite,
    OutOfBounds {
        composite_type: TypeId,
        index_position: usize,
        index: u32,
        bound: u32,
    },
}

/// Walks a composite type hierarchy to find the component type at given indices.
fn walk_composite_type(
    composite_type: TypeId,
    indices: &[u32],
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
) -> Result<TypeId, CompositeWalkError> {
    if indices.is_empty() {
        return Ok(composite_type);
    }

    let mut current_type = composite_type;
    for (position, &index) in indices.iter().enumerate() {
        let rid = ResultId::try_from(u32::from(current_type))
            .map_err(|_| CompositeWalkError::NotComposite)?;
        let inst = definitions
            .get(&rid)
            .ok_or(CompositeWalkError::NotComposite)?;

        match inst.class.opcode {
            Op::TypeVector | Op::TypeMatrix => {
                let element_type = inst
                    .operands
                    .first()
                    .and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw).ok(),
                        _ => None,
                    })
                    .ok_or(CompositeWalkError::NotComposite)?;
                let bound = inst
                    .operands
                    .get(1)
                    .and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .unwrap_or(0);
                if bound != 0 && index >= bound {
                    return Err(CompositeWalkError::OutOfBounds {
                        composite_type: current_type,
                        index_position: position,
                        index,
                        bound,
                    });
                }
                current_type = element_type;
            }
            Op::TypeArray | Op::TypeRuntimeArray => {
                let element_type = inst
                    .operands
                    .first()
                    .and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw).ok(),
                        _ => None,
                    })
                    .ok_or(CompositeWalkError::NotComposite)?;
                if inst.class.opcode == Op::TypeArray {
                    if let Some(bound) = get_array_length(inst, definitions) {
                        if index >= bound {
                            return Err(CompositeWalkError::OutOfBounds {
                                composite_type: current_type,
                                index_position: position,
                                index,
                                bound,
                            });
                        }
                    }
                }
                current_type = element_type;
            }
            Op::TypeStruct => {
                let bound = inst.operands.len() as u32;
                if index >= bound {
                    return Err(CompositeWalkError::OutOfBounds {
                        composite_type: current_type,
                        index_position: position,
                        index,
                        bound,
                    });
                }
                let member_type = inst.operands.get(index as usize).and_then(|op| match op {
                    rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw).ok(),
                    _ => None,
                });
                current_type = member_type.ok_or(CompositeWalkError::NotComposite)?;
            }
            Op::TypeCooperativeMatrixKHR | Op::TypeCooperativeMatrixNV => {
                // Cooperative matrices: first operand is component type
                let element_type = inst
                    .operands
                    .first()
                    .and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw).ok(),
                        _ => None,
                    })
                    .ok_or(CompositeWalkError::NotComposite)?;
                // No bounds checking for cooperative matrices - index is dynamic
                current_type = element_type;
            }
            _ => return Err(CompositeWalkError::NotComposite),
        }
    }
    Ok(current_type)
}

/// Gets the length of an array type by looking up the length constant.
fn get_array_length(
    array_type_inst: &rspirv::dr::Instruction,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let length_id = array_type_inst.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
        _ => None,
    })?;
    let length_inst = definitions.get(&length_id)?;
    // Must be a constant or spec constant
    if !matches!(
        length_inst.class.opcode,
        Op::Constant | Op::ConstantComposite | Op::SpecConstant | Op::SpecConstantComposite
    ) {
        return None;
    }
    // Get the literal value
    length_inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
        rspirv::dr::Operand::LiteralBit64(v) => u32::try_from(*v).ok(),
        _ => None,
    })
}

// ============================================================================
// CompositeConstruct Rule
// ============================================================================

/// Validates OpCompositeConstruct instructions.
///
/// Validates that:
/// - Result type is a valid composite type (vector, matrix, array, struct, cooperative matrix)
/// - For vectors: at least 2 constituents, component types match, total count matches vector size
/// - For matrices: constituent count equals column count, types match column type
/// - For arrays: constituent count equals array size, element types match
/// - For structs: constituent count equals member count, member types match
/// - For cooperative matrices: exactly one constituent of component type
pub struct CompositeConstructRule;

impl ValidationRule for CompositeConstructRule {
    fn name(&self) -> &'static str {
        "composite-construct"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                    if inst.class.opcode != Op::CompositeConstruct {
                        continue;
                    }

                    let Some(result_type_raw) = inst.result_type else {
                        continue;
                    };
                    let Ok(result_type_id) = ResultId::try_from(result_type_raw) else {
                        continue;
                    };
                    let Some(result_type_inst) = ctx.definitions.get(&result_type_id) else {
                        continue;
                    };

                    let num_constituents = inst.operands.len();

                    match result_type_inst.class.opcode {
                        Op::TypeVector | Op::TypeCooperativeVectorNV => {
                            // Get vector component type and size
                            let component_type_id =
                                result_type_inst.operands.first().and_then(|op| {
                                    if let rspirv::dr::Operand::IdRef(id) = op {
                                        Some(*id)
                                    } else {
                                        None
                                    }
                                });

                            // For regular vectors, require at least 2 constituents
                            if result_type_inst.class.opcode == Op::TypeVector
                                && num_constituents < 2
                            {
                                return Err(
                                    ValidationError::CompositeConstructVectorTooFewConstituents {
                                        function: function_id,
                                        block: block_id,
                                    }
                                    .into(),
                                );
                            }

                            // Get vector dimension if it's a regular vector
                            let expected_count = if result_type_inst.class.opcode == Op::TypeVector
                            {
                                result_type_inst.operands.get(1).and_then(|op| {
                                    if let rspirv::dr::Operand::LiteralBit32(count) = op {
                                        Some(*count)
                                    } else {
                                        None
                                    }
                                })
                            } else {
                                None // Cooperative vectors may have dynamic count
                            };

                            // Validate constituents
                            let mut given_count: u32 = 0;
                            for operand in &inst.operands {
                                if let rspirv::dr::Operand::IdRef(id) = operand {
                                    if let Ok(rid) = ResultId::try_from(*id) {
                                        if let Some(value_inst) = ctx.definitions.get(&rid) {
                                            if let Some(value_type) = value_inst.result_type {
                                                if let Ok(value_type_id) =
                                                    ResultId::try_from(value_type)
                                                {
                                                    if let Some(value_type_inst) =
                                                        ctx.definitions.get(&value_type_id)
                                                    {
                                                        // Check if scalar (matching component type) or vector of same component type
                                                        match value_type_inst.class.opcode {
                                                            Op::TypeInt
                                                            | Op::TypeFloat
                                                            | Op::TypeBool => {
                                                                // Scalar: check it matches component type
                                                                if component_type_id
                                                                    != Some(value_type)
                                                                {
                                                                    return Err(ValidationError::CompositeConstructVectorConstituentTypeMismatch {
                                                                        function: function_id,
                                                                        block: block_id,
                                                                    }.into());
                                                                }
                                                                given_count += 1;
                                                            }
                                                            Op::TypeVector => {
                                                                // Vector: check component type matches
                                                                let vec_component = value_type_inst
                                                                    .operands
                                                                    .first()
                                                                    .and_then(|op| {
                                                                        if let rspirv::dr::Operand::IdRef(id) = op {
                                                                            Some(*id)
                                                                        } else {
                                                                            None
                                                                        }
                                                                    });
                                                                if component_type_id
                                                                    != vec_component
                                                                {
                                                                    return Err(ValidationError::CompositeConstructVectorConstituentTypeMismatch {
                                                                        function: function_id,
                                                                        block: block_id,
                                                                    }.into());
                                                                }
                                                                // Add vector size to count
                                                                if let Some(rspirv::dr::Operand::LiteralBit32(size)) = value_type_inst.operands.get(1) {
                                                                    given_count += size;
                                                                }
                                                            }
                                                            _ => {
                                                                return Err(ValidationError::CompositeConstructVectorConstituentTypeMismatch {
                                                                    function: function_id,
                                                                    block: block_id,
                                                                }.into());
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Check total count for regular vectors
                            if let Some(expected) = expected_count {
                                if expected != given_count {
                                    return Err(
                                        ValidationError::CompositeConstructVectorComponentCountMismatch {
                                            expected,
                                            given: given_count,
                                            function: function_id,
                                            block: block_id,
                                        }.into(),
                        );
                                }
                            }
                        }
                        Op::TypeMatrix => {
                            // Get matrix column type and count
                            let col_type_id = result_type_inst.operands.first().and_then(|op| {
                                if let rspirv::dr::Operand::IdRef(id) = op {
                                    Some(*id)
                                } else {
                                    None
                                }
                            });
                            let col_count = result_type_inst.operands.get(1).and_then(|op| {
                                if let rspirv::dr::Operand::LiteralBit32(count) = op {
                                    Some(*count)
                                } else {
                                    None
                                }
                            });

                            if let Some(expected) = col_count {
                                if num_constituents != expected as usize {
                                    return Err(
                                        ValidationError::CompositeConstructMatrixColumnCountMismatch {
                                            expected,
                                            given: num_constituents as u32,
                                            function: function_id,
                                            block: block_id,
                                        }.into(),
                        );
                                }
                            }

                            // Validate each constituent matches column type
                            for operand in &inst.operands {
                                if let rspirv::dr::Operand::IdRef(id) = operand {
                                    if let Ok(rid) = ResultId::try_from(*id) {
                                        if let Some(value_inst) = ctx.definitions.get(&rid) {
                                            if value_inst.result_type != col_type_id {
                                                return Err(
                                                    ValidationError::CompositeConstructMatrixConstituentTypeMismatch {
                                                        function: function_id,
                                                        block: block_id,
                                                    }.into(),
                        );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Op::TypeArray => {
                            // Get array element type and size
                            let elem_type_id = result_type_inst.operands.first().and_then(|op| {
                                if let rspirv::dr::Operand::IdRef(id) = op {
                                    Some(*id)
                                } else {
                                    None
                                }
                            });
                            // Array size is an id reference to a constant
                            let size_id = result_type_inst.operands.get(1).and_then(|op| {
                                if let rspirv::dr::Operand::IdRef(id) = op {
                                    ResultId::try_from(*id).ok()
                                } else {
                                    None
                                }
                            });
                            let array_size = size_id
                                .and_then(|sid| ctx.definitions.get(&sid))
                                .filter(|size_inst| {
                                    // Only use constant (non-spec-constant) sizes for validation
                                    matches!(size_inst.class.opcode, Op::Constant)
                                })
                                .and_then(|size_inst| {
                                    size_inst.operands.first().and_then(|op| match op {
                                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v as u64),
                                        rspirv::dr::Operand::LiteralBit64(v) => Some(*v),
                                        _ => None,
                                    })
                                });

                            if let Some(expected) = array_size {
                                if num_constituents as u64 != expected {
                                    return Err(
                                        ValidationError::CompositeConstructArrayElementCountMismatch {
                                            expected,
                                            given: num_constituents as u64,
                                            function: function_id,
                                            block: block_id,
                                        }.into(),
                        );
                                }
                            }

                            // Validate each constituent matches element type
                            for operand in &inst.operands {
                                if let rspirv::dr::Operand::IdRef(id) = operand {
                                    if let Ok(rid) = ResultId::try_from(*id) {
                                        if let Some(value_inst) = ctx.definitions.get(&rid) {
                                            if value_inst.result_type != elem_type_id {
                                                return Err(
                                                    ValidationError::CompositeConstructArrayConstituentTypeMismatch {
                                                        function: function_id,
                                                        block: block_id,
                                                    }.into(),
                        );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Op::TypeStruct => {
                            // Struct: constituent count must equal member count, types must match
                            let member_count = result_type_inst.operands.len();
                            if num_constituents != member_count {
                                return Err(
                                    ValidationError::CompositeConstructStructMemberCountMismatch {
                                        expected: member_count as u32,
                                        given: num_constituents as u32,
                                        function: function_id,
                                        block: block_id,
                                    }
                                    .into(),
                                );
                            }

                            // Validate each constituent matches member type
                            for (idx, (member_op, constituent_op)) in result_type_inst
                                .operands
                                .iter()
                                .zip(inst.operands.iter())
                                .enumerate()
                            {
                                let member_type = match member_op {
                                    rspirv::dr::Operand::IdRef(id) => Some(*id),
                                    _ => None,
                                };
                                let constituent_type = match constituent_op {
                                    rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid))
                                        .and_then(|inst| inst.result_type),
                                    _ => None,
                                };
                                if member_type != constituent_type {
                                    return Err(
                                        ValidationError::CompositeConstructStructConstituentTypeMismatch {
                                            index: idx as u32,
                                            function: function_id,
                                            block: block_id,
                                        }.into(),
                        );
                                }
                            }
                        }
                        Op::TypeCooperativeMatrixKHR | Op::TypeCooperativeMatrixNV => {
                            // Cooperative matrices require exactly one constituent
                            if num_constituents != 1 {
                                return Err(
                                    ValidationError::CompositeConstructCoopMatrixSingleConstituent {
                                        function: function_id,
                                        block: block_id,
                                    }.into(),
                        );
                            }
                            // Get component type and validate
                            let component_type_id =
                                result_type_inst.operands.first().and_then(|op| {
                                    if let rspirv::dr::Operand::IdRef(id) = op {
                                        Some(*id)
                                    } else {
                                        None
                                    }
                                });
                            if let Some(rspirv::dr::Operand::IdRef(constituent_id)) =
                                inst.operands.first()
                            {
                                if let Ok(rid) = ResultId::try_from(*constituent_id) {
                                    if let Some(constituent_inst) = ctx.definitions.get(&rid) {
                                        if constituent_inst.result_type != component_type_id {
                                            return Err(
                                                ValidationError::CompositeConstructCoopMatrixConstituentTypeMismatch {
                                                    function: function_id,
                                                    block: block_id,
                                                }.into(),
                        );
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            return Err(ValidationError::CompositeConstructResultTypeInvalid {
                                function: function_id,
                                block: block_id,
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
// CopyLogical Rule
// ============================================================================

/// Validates OpCopyLogical instructions.
///
/// Validates that:
/// - Result type does not equal operand type (they must be different)
/// - Result type logically matches operand type (same structure)
/// - With Shader capability, cannot copy composites of 8/16-bit types
pub struct CopyLogicalRule;

impl ValidationRule for CopyLogicalRule {
    fn name(&self) -> &'static str {
        "copy-logical"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        use rspirv::spirv::Capability;

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
                    if inst.class.opcode != Op::CopyLogical {
                        continue;
                    }

                    let Some(result_type_raw) = inst.result_type else {
                        continue;
                    };

                    // Get source operand's type
                    let source_type_raw = inst.operands.first().and_then(|op| {
                        if let rspirv::dr::Operand::IdRef(id) = op {
                            ResultId::try_from(*id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))
                                .and_then(|inst| inst.result_type)
                        } else {
                            None
                        }
                    });

                    let Some(source_type) = source_type_raw else {
                        continue;
                    };

                    // Result type must not equal source type
                    if result_type_raw == source_type {
                        return Err(ValidationError::CopyLogicalTypesEqual {
                            function: function_id,
                            block: block_id,
                        }
                        .into());
                    }

                    // Result type must logically match operand type
                    if !types_logically_match(result_type_raw, source_type, ctx.definitions) {
                        return Err(
                            ValidationError::CopyLogicalTypesNotLogicallyMatching {
                                function: function_id,
                                block: block_id,
                            }
                            .into(),
                        );
                    }

                    // Check Shader capability restriction for 8/16-bit types
                    if ctx.has_capability(Capability::Shader) {
                        // Check if result type contains limited use int/float types
                        if let Ok(result_type_id) = ResultId::try_from(result_type_raw) {
                            if contains_small_type(result_type_id, ctx) {
                                return Err(ValidationError::CopyLogicalSmallTypeRestriction {
                                    function: function_id,
                                    block: block_id,
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

/// Check if two types are "logically matching" for OpCopyLogical.
///
/// Two types logically match if they are the same type, or if they are composites
/// with the same structure (same opcode, same number of members/elements, and
/// recursively logically-matching member types).
fn types_logically_match(
    type_a: u32,
    type_b: u32,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    if type_a == type_b {
        return true;
    }
    let Ok(rid_a) = ResultId::try_from(type_a) else {
        return false;
    };
    let Ok(rid_b) = ResultId::try_from(type_b) else {
        return false;
    };
    let Some(inst_a) = definitions.get(&rid_a) else {
        return false;
    };
    let Some(inst_b) = definitions.get(&rid_b) else {
        return false;
    };

    // Must be same opcode to logically match
    if inst_a.class.opcode != inst_b.class.opcode {
        return false;
    }

    match inst_a.class.opcode {
        Op::TypeStruct => {
            // Same number of members
            if inst_a.operands.len() != inst_b.operands.len() {
                return false;
            }
            // Each member must logically match
            for (op_a, op_b) in inst_a.operands.iter().zip(inst_b.operands.iter()) {
                let (Some(member_a), Some(member_b)) = (
                    match op_a {
                        rspirv::dr::Operand::IdRef(id) => Some(*id),
                        _ => None,
                    },
                    match op_b {
                        rspirv::dr::Operand::IdRef(id) => Some(*id),
                        _ => None,
                    },
                ) else {
                    return false;
                };
                if !types_logically_match(member_a, member_b, definitions) {
                    return false;
                }
            }
            true
        }
        Op::TypeArray => {
            // Element types must logically match, and lengths must be identical
            let elem_a = inst_a.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let elem_b = inst_b.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let len_a = inst_a.operands.get(1).and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let len_b = inst_b.operands.get(1).and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            // Array lengths must be the same ID (same constant)
            if len_a != len_b {
                return false;
            }
            match (elem_a, elem_b) {
                (Some(a), Some(b)) => types_logically_match(a, b, definitions),
                _ => false,
            }
        }
        Op::TypeRuntimeArray => {
            // Element types must logically match
            let elem_a = inst_a.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let elem_b = inst_b.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            match (elem_a, elem_b) {
                (Some(a), Some(b)) => types_logically_match(a, b, definitions),
                _ => false,
            }
        }
        Op::TypeVector | Op::TypeMatrix => {
            // Element types must logically match and counts must be the same
            let elem_a = inst_a.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let elem_b = inst_b.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let count_a = inst_a.operands.get(1).and_then(|op| match op {
                rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                _ => None,
            });
            let count_b = inst_b.operands.get(1).and_then(|op| match op {
                rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                _ => None,
            });
            if count_a != count_b {
                return false;
            }
            match (elem_a, elem_b) {
                (Some(a), Some(b)) => types_logically_match(a, b, definitions),
                _ => false,
            }
        }
        // For non-composite types, they must be identical (already checked type_a == type_b above)
        _ => false,
    }
}

/// Check if a type contains 8 or 16-bit int/float types (for limited use type restrictions).
/// This version takes definitions directly for use in VectorDynamicRule.
fn contains_limited_type(
    type_id: u32,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    let Ok(rid) = ResultId::try_from(type_id) else {
        return false;
    };
    let Some(type_inst) = definitions.get(&rid) else {
        return false;
    };

    match type_inst.class.opcode {
        Op::TypeInt | Op::TypeFloat => {
            // Check width - 8 or 16-bit types are limited
            type_inst.operands.first().is_some_and(|op| {
                if let rspirv::dr::Operand::LiteralBit32(width) = op {
                    *width == 8 || *width == 16
                } else {
                    false
                }
            })
        }
        Op::TypeVector | Op::TypeMatrix | Op::TypeArray | Op::TypeRuntimeArray => {
            // Check element type
            type_inst
                .operands
                .first()
                .and_then(|op| {
                    if let rspirv::dr::Operand::IdRef(id) = op {
                        Some(*id)
                    } else {
                        None
                    }
                })
                .is_some_and(|elem_type_id| contains_limited_type(elem_type_id, definitions))
        }
        Op::TypeStruct => {
            // Check all member types
            type_inst.operands.iter().any(|op| {
                if let rspirv::dr::Operand::IdRef(id) = op {
                    contains_limited_type(*id, definitions)
                } else {
                    false
                }
            })
        }
        _ => false,
    }
}

/// Check if a type contains 8 or 16-bit int/float types (recursively).
fn contains_small_type(type_id: ResultId, ctx: &ValidationContext<'_>) -> bool {
    let Some(type_inst) = ctx.definitions.get(&type_id) else {
        return false;
    };

    match type_inst.class.opcode {
        Op::TypeInt | Op::TypeFloat => {
            // Check width
            type_inst.operands.first().is_some_and(|op| {
                if let rspirv::dr::Operand::LiteralBit32(width) = op {
                    *width == 8 || *width == 16
                } else {
                    false
                }
            })
        }
        Op::TypeVector | Op::TypeMatrix | Op::TypeArray | Op::TypeRuntimeArray => {
            // Check element type
            type_inst
                .operands
                .first()
                .and_then(|op| {
                    if let rspirv::dr::Operand::IdRef(id) = op {
                        ResultId::try_from(*id).ok()
                    } else {
                        None
                    }
                })
                .is_some_and(|elem_type_id| contains_small_type(elem_type_id, ctx))
        }
        Op::TypeStruct => {
            // Check all member types
            type_inst.operands.iter().any(|op| {
                if let rspirv::dr::Operand::IdRef(id) = op {
                    ResultId::try_from(*id)
                        .ok()
                        .is_some_and(|member_type_id| contains_small_type(member_type_id, ctx))
                } else {
                    false
                }
            })
        }
        _ => false,
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
        &CompositeExtractInsertRule,
        &CompositeConstructRule,
        &CopyLogicalRule,
    ]
}
