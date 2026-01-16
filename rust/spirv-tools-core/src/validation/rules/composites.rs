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
use crate::validation::helpers::vector_info;
use crate::validation::type_ext::TypeInstructionExt;
use crate::validation::types::{Id, ResultId, TypeId};

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
                                });
                            };
                            if vector_type_inst.class.opcode != Op::TypeVector {
                                return Err(ValidationError::VectorOperandNotVector {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    operand: 0,
                                    found: vector_type_id,
                                });
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
                                    });
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
                                            });
                                        }
                                    }
                                }
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
                                });
                            };
                            if vector_type_inst.class.opcode != Op::TypeVector {
                                return Err(ValidationError::VectorOperandNotVector {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    operand: 0,
                                    found: vector_type_id,
                                });
                            }

                            // Result type must match vector type
                            if result_type_id != vector_type_id {
                                return Err(ValidationError::InstructionResultTypeMismatch {
                                    function: func,
                                    block: blk,
                                    instruction: inst.class.opcode,
                                    expected: vector_type_id,
                                    found: result_type_id,
                                });
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
                                            });
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
                                            });
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
/// - Component indices are valid (0..N+M-1, or 0xFFFFFFFF for undefined)
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
                            return Err(err);
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
                            return Err(err);
                        }
                    };

                    // Component types must match
                    if vec1_component != vec2_component {
                        return Err(ValidationError::VectorShuffleComponentTypeMismatch {
                            function: func,
                            block: blk,
                            first: vec1_component,
                            second: vec2_component,
                        });
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
                        });
                    }
                    let (result_component, result_len) = vector_info(result_vector_inst);
                    let Some(result_component) = result_component else {
                        return Err(ValidationError::VectorShuffleResultTypeMismatch {
                            function: func,
                            block: blk,
                            result_type: result_type_id,
                            component_type: vec1_component,
                        });
                    };
                    if result_component != vec1_component {
                        return Err(ValidationError::VectorShuffleResultTypeMismatch {
                            function: func,
                            block: blk,
                            result_type: result_type_id,
                            component_type: vec1_component,
                        });
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
                        });
                    };
                    if operand_component_count != result_component_len {
                        return Err(ValidationError::VectorShuffleComponentCountMismatch {
                            function: func,
                            block: blk,
                            operand_components: operand_component_count,
                            result_components: result_component_len,
                        });
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
