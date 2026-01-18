//! Cooperative matrix and vector operation validation rules.
//!
//! This module validates SPIR-V cooperative matrix and cooperative vector operations including:
//!
//! - OpCooperativeMatrixLengthNV/KHR: Matrix length instruction validation
//! - OpCooperativeMatrixLoadNV: NVIDIA cooperative matrix load validation
//! - OpCooperativeMatrixStoreNV: NVIDIA cooperative matrix store validation
//! - OpCooperativeMatrixLoadKHR: Khronos cooperative matrix load validation
//! - OpCooperativeMatrixStoreKHR: Khronos cooperative matrix store validation
//! - OpCooperativeVectorLoadNV: NVIDIA cooperative vector load validation
//! - OpCooperativeVectorStoreNV: NVIDIA cooperative vector store validation

use rspirv::dr::Operand;
use rspirv::spirv::{Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::ResultId;

use super::helpers::id_from_u32;

/// Helper to check if an instruction is a logical pointer producer.
fn is_logical_pointer_producer_coop(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Variable
            | Op::AccessChain
            | Op::InBoundsAccessChain
            | Op::FunctionParameter
            | Op::ImageTexelPointer
            | Op::CopyObject
            | Op::Select
            | Op::Phi
            | Op::PtrAccessChain
            | Op::InBoundsPtrAccessChain
            | Op::Load
            | Op::VectorExtractDynamic
            | Op::CompositeExtract
    )
}

/// KHR cooperative matrix layout values that require a stride.
mod cooperative_matrix_layout {
    pub const ROW_MAJOR_KHR: u64 = 0;
    pub const COLUMN_MAJOR_KHR: u64 = 1;
}

// ============================================================================
// Cooperative Matrix Length Validation
// ============================================================================

/// Validates OpCooperativeMatrixLengthNV and OpCooperativeMatrixLengthKHR instructions.
pub struct CooperativeMatrixLengthRule;

impl ValidationRule for CooperativeMatrixLengthRule {
    fn name(&self) -> &'static str {
        "cooperative-matrix-length"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let function_id = function.def.as_ref().and_then(|d| d.result_id).map(id_from_u32);

            for block in &function.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(id_from_u32);

                for inst in &block.instructions {
                    let (is_khr, op_name) = match inst.class.opcode {
                        Op::CooperativeMatrixLengthKHR => (true, "OpCooperativeMatrixLengthKHR"),
                        Op::CooperativeMatrixLengthNV => (false, "OpCooperativeMatrixLengthNV"),
                        _ => continue,
                    };

                    // Result type must be OpTypeInt with width 32 and signedness 0
                    if let Some(result_type) = inst.result_type {
                        if let Ok(result_type_id) = ResultId::try_from(result_type) {
                            if let Some(result_type_inst) = ctx.definitions.get(&result_type_id) {
                                if result_type_inst.class.opcode != Op::TypeInt {
                                    return Err(
                                        ValidationError::CooperativeMatrixLengthResultTypeMismatch {
                                            op_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                                // Check width is 32
                                if let Some(Operand::LiteralBit32(width)) =
                                    result_type_inst.operands.first()
                                {
                                    if *width != 32 {
                                        return Err(
                                            ValidationError::CooperativeMatrixLengthResultTypeMismatch {
                                                op_name,
                                                function: function_id,
                                                block: block_id,
                                            },
                                        );
                                    }
                                }
                                // Check signedness is 0
                                if let Some(Operand::LiteralBit32(signedness)) =
                                    result_type_inst.operands.get(1)
                                {
                                    if *signedness != 0 {
                                        return Err(
                                            ValidationError::CooperativeMatrixLengthResultTypeMismatch {
                                                op_name,
                                                function: function_id,
                                                block: block_id,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Validate the type operand
                    if let Some(Operand::IdRef(type_id)) = inst.operands.first() {
                        if let Ok(type_result_id) = ResultId::try_from(*type_id) {
                            if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                                if is_khr {
                                    if type_inst.class.opcode != Op::TypeCooperativeMatrixKHR {
                                        return Err(
                                            ValidationError::CooperativeMatrixLengthKhrTypeMismatch {
                                                function: function_id,
                                                block: block_id,
                                            },
                                        );
                                    }
                                } else if type_inst.class.opcode != Op::TypeCooperativeMatrixNV {
                                    return Err(
                                        ValidationError::CooperativeMatrixLengthNvTypeMismatch {
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
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
// Cooperative Matrix Load/Store NV Validation
// ============================================================================

/// Validates OpCooperativeMatrixLoadNV and OpCooperativeMatrixStoreNV instructions.
pub struct CooperativeMatrixLoadStoreNVRule;

impl ValidationRule for CooperativeMatrixLoadStoreNVRule {
    fn name(&self) -> &'static str {
        "cooperative-matrix-load-store-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let function_id = function.def.as_ref().and_then(|d| d.result_id).map(id_from_u32);

            for block in &function.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(id_from_u32);

                for inst in &block.instructions {
                    let (is_load, op_name) = match inst.class.opcode {
                        Op::CooperativeMatrixLoadNV => (true, "OpCooperativeMatrixLoadNV"),
                        Op::CooperativeMatrixStoreNV => (false, "OpCooperativeMatrixStoreNV"),
                        _ => continue,
                    };

                    // Get the matrix type
                    let type_id = if is_load {
                        inst.result_type
                    } else {
                        // For store, get Object operand's type (operand index 1)
                        inst.operands.get(1).and_then(|op| {
                            if let Operand::IdRef(id) = op {
                                ResultId::try_from(*id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid))
                                    .and_then(|inst| inst.result_type)
                            } else {
                                None
                            }
                        })
                    };

                    if let Some(matrix_type_id) = type_id {
                        if let Ok(matrix_type_result_id) = ResultId::try_from(matrix_type_id) {
                            if let Some(matrix_type_inst) =
                                ctx.definitions.get(&matrix_type_result_id)
                            {
                                if matrix_type_inst.class.opcode != Op::TypeCooperativeMatrixNV {
                                    let operand_name = if is_load {
                                        "Result Type"
                                    } else {
                                        "Object type"
                                    };
                                    return Err(
                                        ValidationError::CooperativeMatrixLoadStoreTypeMismatch {
                                            op_name,
                                            operand_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                            }
                        }
                    }

                    // Validate pointer operand
                    // For Load: operands are Pointer, Stride, ColumnMajor
                    // For Store: operands are Pointer, Object, Stride, ColumnMajor
                    let pointer_index = 0; // Both use operand 0 for pointer in NV
                    if let Some(Operand::IdRef(pointer_id)) = inst.operands.get(pointer_index) {
                        if let Ok(pointer_result_id) = ResultId::try_from(*pointer_id) {
                            if let Some(pointer_inst) = ctx.definitions.get(&pointer_result_id) {
                                // Check if it's a logical pointer producer
                                if !is_logical_pointer_producer_coop(pointer_inst.class.opcode) {
                                    return Err(
                                        ValidationError::CooperativeMatrixLoadStorePointerNotLogical {
                                            op_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }

                                // Validate pointer type
                                if let Some(pointer_type) = pointer_inst.result_type {
                                    if let Ok(pointer_type_id) = ResultId::try_from(pointer_type) {
                                        if let Some(pointer_type_inst) =
                                            ctx.definitions.get(&pointer_type_id)
                                        {
                                            if pointer_type_inst.class.opcode != Op::TypePointer {
                                                return Err(
                                                    ValidationError::CooperativeMatrixLoadStorePointerTypeInvalid {
                                                        op_name,
                                                        function: function_id,
                                                        block: block_id,
                                                    },
                                                );
                                            }

                                            // Check storage class
                                            if let Some(Operand::StorageClass(sc)) =
                                                pointer_type_inst.operands.first()
                                            {
                                                if !matches!(
                                                    sc,
                                                    StorageClass::Workgroup
                                                        | StorageClass::StorageBuffer
                                                        | StorageClass::PhysicalStorageBuffer
                                                ) {
                                                    return Err(
                                                        ValidationError::CooperativeMatrixLoadStoreInvalidStorageClass {
                                                            op_name,
                                                            function: function_id,
                                                            block: block_id,
                                                        },
                                                    );
                                                }
                                            }

                                            // Check pointee type is scalar or vector
                                            if let Some(Operand::IdRef(pointee_id)) =
                                                pointer_type_inst.operands.get(1)
                                            {
                                                if let Ok(pointee_result_id) =
                                                    ResultId::try_from(*pointee_id)
                                                {
                                                    if let Some(pointee_type_inst) =
                                                        ctx.definitions.get(&pointee_result_id)
                                                    {
                                                        let is_scalar_or_vector = matches!(
                                                            pointee_type_inst.class.opcode,
                                                            Op::TypeInt
                                                                | Op::TypeFloat
                                                                | Op::TypeVector
                                                        );
                                                        if !is_scalar_or_vector {
                                                            return Err(
                                                                ValidationError::CooperativeMatrixLoadStorePointeeTypeMismatch {
                                                                    op_name,
                                                                    function: function_id,
                                                                    block: block_id,
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
                    }

                    // Validate Stride operand
                    let stride_index = if is_load { 1 } else { 2 };
                    if let Some(Operand::IdRef(stride_id)) = inst.operands.get(stride_index) {
                        if let Ok(stride_result_id) = ResultId::try_from(*stride_id) {
                            if let Some(stride_inst) = ctx.definitions.get(&stride_result_id) {
                                if let Some(stride_type) = stride_inst.result_type {
                                    if let Ok(stride_type_id) = ResultId::try_from(stride_type) {
                                        if let Some(stride_type_inst) =
                                            ctx.definitions.get(&stride_type_id)
                                        {
                                            if stride_type_inst.class.opcode != Op::TypeInt {
                                                return Err(
                                                    ValidationError::CooperativeMatrixLoadStoreStrideTypeMismatch {
                                                        op_name,
                                                        function: function_id,
                                                        block: block_id,
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Validate ColumnMajor operand (must be boolean constant)
                    let colmajor_index = if is_load { 2 } else { 3 };
                    if let Some(Operand::IdRef(colmajor_id)) = inst.operands.get(colmajor_index) {
                        if let Ok(colmajor_result_id) = ResultId::try_from(*colmajor_id) {
                            if let Some(colmajor_inst) = ctx.definitions.get(&colmajor_result_id) {
                                // Check it's a constant
                                let is_constant = matches!(
                                    colmajor_inst.class.opcode,
                                    Op::ConstantTrue
                                        | Op::ConstantFalse
                                        | Op::Constant
                                        | Op::SpecConstantTrue
                                        | Op::SpecConstantFalse
                                        | Op::SpecConstant
                                );
                                if !is_constant {
                                    return Err(
                                        ValidationError::CooperativeMatrixLoadStoreColumnMajorMismatch {
                                            op_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                                // Check type is bool
                                if let Some(colmajor_type) = colmajor_inst.result_type {
                                    if let Ok(colmajor_type_id) = ResultId::try_from(colmajor_type)
                                    {
                                        if let Some(colmajor_type_inst) =
                                            ctx.definitions.get(&colmajor_type_id)
                                        {
                                            if colmajor_type_inst.class.opcode != Op::TypeBool {
                                                return Err(
                                                    ValidationError::CooperativeMatrixLoadStoreColumnMajorMismatch {
                                                        op_name,
                                                        function: function_id,
                                                        block: block_id,
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
        }

        Ok(())
    }
}

// ============================================================================
// Cooperative Matrix Load/Store KHR Validation
// ============================================================================

/// Validates OpCooperativeMatrixLoadKHR and OpCooperativeMatrixStoreKHR instructions.
pub struct CooperativeMatrixLoadStoreKHRRule;

impl ValidationRule for CooperativeMatrixLoadStoreKHRRule {
    fn name(&self) -> &'static str {
        "cooperative-matrix-load-store-khr"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let function_id = function.def.as_ref().and_then(|d| d.result_id).map(id_from_u32);

            for block in &function.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(id_from_u32);

                for inst in &block.instructions {
                    let (is_load, op_name) = match inst.class.opcode {
                        Op::CooperativeMatrixLoadKHR => (true, "OpCooperativeMatrixLoadKHR"),
                        Op::CooperativeMatrixStoreKHR => (false, "OpCooperativeMatrixStoreKHR"),
                        _ => continue,
                    };

                    // Get the matrix type
                    let type_id = if is_load {
                        inst.result_type
                    } else {
                        // For store, get Object operand's type (operand index 1)
                        inst.operands.get(1).and_then(|op| {
                            if let Operand::IdRef(id) = op {
                                ResultId::try_from(*id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid))
                                    .and_then(|inst| inst.result_type)
                            } else {
                                None
                            }
                        })
                    };

                    if let Some(matrix_type_id) = type_id {
                        if let Ok(matrix_type_result_id) = ResultId::try_from(matrix_type_id) {
                            if let Some(matrix_type_inst) =
                                ctx.definitions.get(&matrix_type_result_id)
                            {
                                if matrix_type_inst.class.opcode != Op::TypeCooperativeMatrixKHR {
                                    let operand_name = if is_load {
                                        "Result Type"
                                    } else {
                                        "Object type"
                                    };
                                    return Err(
                                        ValidationError::CooperativeMatrixLoadStoreTypeMismatch {
                                            op_name,
                                            operand_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                            }
                        }
                    }

                    // Validate pointer operand
                    let pointer_index = 0;
                    if let Some(Operand::IdRef(pointer_id)) = inst.operands.get(pointer_index) {
                        if let Ok(pointer_result_id) = ResultId::try_from(*pointer_id) {
                            if let Some(pointer_inst) = ctx.definitions.get(&pointer_result_id) {
                                // Check if it's a logical pointer producer
                                if !is_logical_pointer_producer_coop(pointer_inst.class.opcode) {
                                    return Err(
                                        ValidationError::CooperativeMatrixLoadStorePointerNotLogical {
                                            op_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }

                                // Validate pointer type
                                if let Some(pointer_type) = pointer_inst.result_type {
                                    if let Ok(pointer_type_id) = ResultId::try_from(pointer_type) {
                                        if let Some(pointer_type_inst) =
                                            ctx.definitions.get(&pointer_type_id)
                                        {
                                            let is_pointer_type = matches!(
                                                pointer_type_inst.class.opcode,
                                                Op::TypePointer | Op::TypeUntypedPointerKHR
                                            );
                                            if !is_pointer_type {
                                                return Err(
                                                    ValidationError::CooperativeMatrixLoadStorePointerTypeInvalid {
                                                        op_name,
                                                        function: function_id,
                                                        block: block_id,
                                                    },
                                                );
                                            }

                                            // Check storage class for Vulkan
                                            if ctx.is_vulkan_env() {
                                                if let Some(Operand::StorageClass(sc)) =
                                                    pointer_type_inst.operands.first()
                                                {
                                                    if !matches!(
                                                        sc,
                                                        StorageClass::Workgroup
                                                            | StorageClass::StorageBuffer
                                                            | StorageClass::PhysicalStorageBuffer
                                                    ) {
                                                        return Err(
                                                            ValidationError::CooperativeMatrixLoadStoreInvalidStorageClass {
                                                                op_name,
                                                                function: function_id,
                                                                block: block_id,
                                                            },
                                                        );
                                                    }
                                                }
                                            }

                                            // Check pointee type for typed pointers
                                            if pointer_type_inst.class.opcode == Op::TypePointer {
                                                if let Some(Operand::IdRef(pointee_id)) =
                                                    pointer_type_inst.operands.get(1)
                                                {
                                                    if let Ok(pointee_result_id) =
                                                        ResultId::try_from(*pointee_id)
                                                    {
                                                        if let Some(pointee_type_inst) =
                                                            ctx.definitions.get(&pointee_result_id)
                                                        {
                                                            let is_scalar_or_vector = matches!(
                                                                pointee_type_inst.class.opcode,
                                                                Op::TypeInt
                                                                    | Op::TypeFloat
                                                                    | Op::TypeVector
                                                            );
                                                            if !is_scalar_or_vector {
                                                                return Err(
                                                                    ValidationError::CooperativeMatrixLoadStorePointeeTypeMismatch {
                                                                        op_name,
                                                                        function: function_id,
                                                                        block: block_id,
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
                        }
                    }

                    // Validate MemoryLayout operand (must be 32-bit integer constant)
                    let layout_index = if is_load { 1 } else { 2 };
                    let mut layout_value: Option<u64> = None;
                    if let Some(Operand::IdRef(layout_id)) = inst.operands.get(layout_index) {
                        if let Ok(layout_result_id) = ResultId::try_from(*layout_id) {
                            if let Some(layout_inst) = ctx.definitions.get(&layout_result_id) {
                                // Check it's a constant
                                let is_constant = matches!(
                                    layout_inst.class.opcode,
                                    Op::Constant | Op::SpecConstant
                                );
                                if !is_constant {
                                    return Err(
                                        ValidationError::CooperativeMatrixLoadStoreLayoutMismatch {
                                            op_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                                // Check type is int
                                if let Some(layout_type) = layout_inst.result_type {
                                    if let Ok(layout_type_id) = ResultId::try_from(layout_type) {
                                        if let Some(layout_type_inst) =
                                            ctx.definitions.get(&layout_type_id)
                                        {
                                            if layout_type_inst.class.opcode != Op::TypeInt {
                                                return Err(
                                                    ValidationError::CooperativeMatrixLoadStoreLayoutMismatch {
                                                        op_name,
                                                        function: function_id,
                                                        block: block_id,
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                                // Get the actual value for stride checking
                                if let Some(Operand::LiteralBit32(val)) = layout_inst.operands.first()
                                {
                                    layout_value = Some(*val as u64);
                                }
                            }
                        }
                    }

                    // Check if stride is required based on layout
                    let stride_required = matches!(
                        layout_value,
                        Some(cooperative_matrix_layout::ROW_MAJOR_KHR)
                            | Some(cooperative_matrix_layout::COLUMN_MAJOR_KHR)
                    );

                    // Validate Stride operand if present
                    let stride_index = if is_load { 2 } else { 3 };
                    if inst.operands.len() > stride_index {
                        if let Some(Operand::IdRef(stride_id)) = inst.operands.get(stride_index) {
                            if let Ok(stride_result_id) = ResultId::try_from(*stride_id) {
                                if let Some(stride_inst) = ctx.definitions.get(&stride_result_id) {
                                    if let Some(stride_type) = stride_inst.result_type {
                                        if let Ok(stride_type_id) = ResultId::try_from(stride_type)
                                        {
                                            if let Some(stride_type_inst) =
                                                ctx.definitions.get(&stride_type_id)
                                            {
                                                if stride_type_inst.class.opcode != Op::TypeInt {
                                                    return Err(
                                                        ValidationError::CooperativeMatrixLoadStoreStrideTypeMismatch {
                                                            op_name,
                                                            function: function_id,
                                                            block: block_id,
                                                        },
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if stride_required {
                        if let Some(layout) = layout_value {
                            return Err(
                                ValidationError::CooperativeMatrixLoadStoreLayoutRequiresStride {
                                    op_name,
                                    layout,
                                    function: function_id,
                                    block: block_id,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Cooperative Vector Load/Store NV Validation
// ============================================================================

/// Validates OpCooperativeVectorLoadNV and OpCooperativeVectorStoreNV instructions.
pub struct CooperativeVectorLoadStoreNVRule;

impl ValidationRule for CooperativeVectorLoadStoreNVRule {
    fn name(&self) -> &'static str {
        "cooperative-vector-load-store-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let function_id = function.def.as_ref().and_then(|d| d.result_id).map(id_from_u32);

            for block in &function.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(id_from_u32);

                for inst in &block.instructions {
                    let (is_load, op_name) = match inst.class.opcode {
                        Op::CooperativeVectorLoadNV => (true, "OpCooperativeVectorLoadNV"),
                        Op::CooperativeVectorStoreNV => (false, "OpCooperativeVectorStoreNV"),
                        _ => continue,
                    };

                    // Get the vector type
                    let type_id = if is_load {
                        inst.result_type
                    } else {
                        // For store, get Object operand's type (operand index 2)
                        inst.operands.get(2).and_then(|op| {
                            if let Operand::IdRef(id) = op {
                                ResultId::try_from(*id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid))
                                    .and_then(|inst| inst.result_type)
                            } else {
                                None
                            }
                        })
                    };

                    if let Some(vector_type_id) = type_id {
                        if let Ok(vector_type_result_id) = ResultId::try_from(vector_type_id) {
                            if let Some(vector_type_inst) =
                                ctx.definitions.get(&vector_type_result_id)
                            {
                                if vector_type_inst.class.opcode != Op::TypeCooperativeVectorNV {
                                    let operand_name = if is_load {
                                        "Result Type"
                                    } else {
                                        "Object type"
                                    };
                                    return Err(
                                        ValidationError::CooperativeVectorLoadStoreTypeMismatch {
                                            op_name,
                                            operand_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                            }
                        }
                    }

                    // Validate pointer operand
                    let pointer_index = 0;
                    if let Some(Operand::IdRef(pointer_id)) = inst.operands.get(pointer_index) {
                        if let Ok(pointer_result_id) = ResultId::try_from(*pointer_id) {
                            if let Some(pointer_inst) = ctx.definitions.get(&pointer_result_id) {
                                // Check if it's a logical pointer producer
                                if !is_logical_pointer_producer_coop(pointer_inst.class.opcode) {
                                    return Err(
                                        ValidationError::CooperativeVectorPointerNotLogical {
                                            op_name,
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }

                                // Validate pointer type
                                if let Some(pointer_type) = pointer_inst.result_type {
                                    if let Ok(pointer_type_id) = ResultId::try_from(pointer_type) {
                                        if let Some(pointer_type_inst) =
                                            ctx.definitions.get(&pointer_type_id)
                                        {
                                            if pointer_type_inst.class.opcode != Op::TypePointer {
                                                return Err(
                                                    ValidationError::CooperativeVectorPointerTypeInvalid {
                                                        op_name,
                                                        function: function_id,
                                                        block: block_id,
                                                    },
                                                );
                                            }

                                            // Check storage class
                                            if let Some(Operand::StorageClass(sc)) =
                                                pointer_type_inst.operands.first()
                                            {
                                                if !matches!(
                                                    sc,
                                                    StorageClass::Workgroup
                                                        | StorageClass::StorageBuffer
                                                        | StorageClass::PhysicalStorageBuffer
                                                ) {
                                                    return Err(
                                                        ValidationError::CooperativeVectorInvalidStorageClass {
                                                            op_name,
                                                            function: function_id,
                                                            block: block_id,
                                                        },
                                                    );
                                                }
                                            }

                                            // Check pointee type is array
                                            if let Some(Operand::IdRef(pointee_id)) =
                                                pointer_type_inst.operands.get(1)
                                            {
                                                if let Ok(pointee_result_id) =
                                                    ResultId::try_from(*pointee_id)
                                                {
                                                    if let Some(pointee_type_inst) =
                                                        ctx.definitions.get(&pointee_result_id)
                                                    {
                                                        let is_array = matches!(
                                                            pointee_type_inst.class.opcode,
                                                            Op::TypeArray | Op::TypeRuntimeArray
                                                        );
                                                        if !is_array {
                                                            return Err(
                                                                ValidationError::CooperativeVectorPointeeTypeNotArray {
                                                                    op_name,
                                                                    function: function_id,
                                                                    block: block_id,
                                                                },
                                                            );
                                                        }

                                                        // Check array element type is scalar or vector
                                                        if let Some(Operand::IdRef(
                                                            array_elem_id,
                                                        )) = pointee_type_inst.operands.first()
                                                        {
                                                            if let Ok(array_elem_result_id) =
                                                                ResultId::try_from(*array_elem_id)
                                                            {
                                                                if let Some(array_elem_type_inst) =
                                                                    ctx.definitions
                                                                        .get(&array_elem_result_id)
                                                                {
                                                                    let is_scalar_or_vector =
                                                                        matches!(
                                                                            array_elem_type_inst
                                                                                .class
                                                                                .opcode,
                                                                            Op::TypeInt
                                                                                | Op::TypeFloat
                                                                                | Op::TypeVector
                                                                        );
                                                                    if !is_scalar_or_vector {
                                                                        return Err(
                                                                            ValidationError::CooperativeVectorArrayElementTypeMismatch {
                                                                                op_name,
                                                                                function: function_id,
                                                                                block: block_id,
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
                                }
                            }
                        }
                    }

                    // Validate Offset operand (must be 32 or 64-bit integer)
                    let offset_index = 1;
                    if let Some(Operand::IdRef(offset_id)) = inst.operands.get(offset_index) {
                        if let Ok(offset_result_id) = ResultId::try_from(*offset_id) {
                            if let Some(offset_inst) = ctx.definitions.get(&offset_result_id) {
                                if let Some(offset_type) = offset_inst.result_type {
                                    if let Ok(offset_type_id) = ResultId::try_from(offset_type) {
                                        if let Some(offset_type_inst) =
                                            ctx.definitions.get(&offset_type_id)
                                        {
                                            if offset_type_inst.class.opcode != Op::TypeInt {
                                                return Err(
                                                    ValidationError::CooperativeVectorOffsetTypeMismatch {
                                                        op_name,
                                                        function: function_id,
                                                        block: block_id,
                                                    },
                                                );
                                            }
                                            // Check width is 32 or 64
                                            if let Some(Operand::LiteralBit32(width)) =
                                                offset_type_inst.operands.first()
                                            {
                                                if *width != 32 && *width != 64 {
                                                    return Err(
                                                        ValidationError::CooperativeVectorOffsetTypeMismatch {
                                                            op_name,
                                                            function: function_id,
                                                            block: block_id,
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
            }
        }

        Ok(())
    }
}

// ============================================================================
// Cooperative Matrix MulAdd Validation
// ============================================================================

/// Helper to extract constant u32 value from an ID.
fn get_constant_u32(id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let result_id = ResultId::try_from(id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;
    if !matches!(
        inst.class.opcode,
        Op::Constant | Op::SpecConstant
    ) {
        return None;
    }
    match inst.operands.first() {
        Some(Operand::LiteralBit32(val)) => Some(*val),
        _ => None,
    }
}

/// Helper to check if a type is OpTypeCooperativeMatrixNV.
fn is_cooperative_matrix_nv_type(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    ResultId::try_from(type_id)
        .ok()
        .and_then(|id| ctx.definitions.get(&id))
        .map(|inst| inst.class.opcode == Op::TypeCooperativeMatrixNV)
        .unwrap_or(false)
}

/// Helper to check if a type is OpTypeCooperativeMatrixKHR.
fn is_cooperative_matrix_khr_type(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    ResultId::try_from(type_id)
        .ok()
        .and_then(|id| ctx.definitions.get(&id))
        .map(|inst| inst.class.opcode == Op::TypeCooperativeMatrixKHR)
        .unwrap_or(false)
}

/// Extracts cooperative matrix type parameters (Scope, Rows, Cols).
fn get_matrix_params(
    type_id: u32,
    ctx: &ValidationContext<'_>,
) -> Option<(Option<u32>, Option<u32>, Option<u32>)> {
    let result_id = ResultId::try_from(type_id).ok()?;
    let type_inst = ctx.definitions.get(&result_id)?;

    // For both NV and KHR types:
    // Operand 0: Component Type
    // Operand 1 (NV) / 1 (KHR): Scope
    // Operand 2 (NV) / 2 (KHR): Rows
    // Operand 3 (NV) / 3 (KHR): Columns
    // KHR also has Use at operand 4

    let scope_idx = match type_inst.class.opcode {
        Op::TypeCooperativeMatrixNV | Op::TypeCooperativeMatrixKHR => 1,
        _ => return None,
    };

    let scope_id = type_inst.operands.get(scope_idx).and_then(|op| {
        if let Operand::IdRef(id) = op {
            Some(*id)
        } else {
            None
        }
    });
    let rows_id = type_inst.operands.get(scope_idx + 1).and_then(|op| {
        if let Operand::IdRef(id) = op {
            Some(*id)
        } else {
            None
        }
    });
    let cols_id = type_inst.operands.get(scope_idx + 2).and_then(|op| {
        if let Operand::IdRef(id) = op {
            Some(*id)
        } else {
            None
        }
    });

    Some((
        scope_id.and_then(|id| get_constant_u32(id, ctx)),
        rows_id.and_then(|id| get_constant_u32(id, ctx)),
        cols_id.and_then(|id| get_constant_u32(id, ctx)),
    ))
}

/// Validates OpCooperativeMatrixMulAddNV instructions.
pub struct CooperativeMatrixMulAddNVRule;

impl ValidationRule for CooperativeMatrixMulAddNVRule {
    fn name(&self) -> &'static str {
        "cooperative-matrix-muladd-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let function_id = function.def.as_ref().and_then(|d| d.result_id).map(id_from_u32);

            for block in &function.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(id_from_u32);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::CooperativeMatrixMulAddNV {
                        continue;
                    }

                    let op_name = "OpCooperativeMatrixMulAddNV";

                    // Get types for D (result), A, B, C operands
                    let d_type_id = inst.result_type.unwrap_or(0);

                    // Operands: A, B, C
                    let a_id = inst.operands.first().and_then(|op| {
                        if let Operand::IdRef(id) = op { Some(*id) } else { None }
                    });
                    let b_id = inst.operands.get(1).and_then(|op| {
                        if let Operand::IdRef(id) = op { Some(*id) } else { None }
                    });
                    let c_id = inst.operands.get(2).and_then(|op| {
                        if let Operand::IdRef(id) = op { Some(*id) } else { None }
                    });

                    // Get operand types
                    let a_type_id = a_id
                        .and_then(|id| ResultId::try_from(id).ok())
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type)
                        .unwrap_or(0);
                    let b_type_id = b_id
                        .and_then(|id| ResultId::try_from(id).ok())
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type)
                        .unwrap_or(0);
                    let c_type_id = c_id
                        .and_then(|id| ResultId::try_from(id).ok())
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type)
                        .unwrap_or(0);

                    // Check all are cooperative matrix NV types
                    if !is_cooperative_matrix_nv_type(a_type_id, ctx) {
                        return Err(ValidationError::CooperativeMatrixMulAddTypeMismatch {
                            op_name,
                            operand_name: "A",
                            function: function_id,
                            block: block_id,
                        });
                    }
                    if !is_cooperative_matrix_nv_type(b_type_id, ctx) {
                        return Err(ValidationError::CooperativeMatrixMulAddTypeMismatch {
                            op_name,
                            operand_name: "B",
                            function: function_id,
                            block: block_id,
                        });
                    }
                    if !is_cooperative_matrix_nv_type(c_type_id, ctx) {
                        return Err(ValidationError::CooperativeMatrixMulAddTypeMismatch {
                            op_name,
                            operand_name: "C",
                            function: function_id,
                            block: block_id,
                        });
                    }
                    if !is_cooperative_matrix_nv_type(d_type_id, ctx) {
                        return Err(ValidationError::CooperativeMatrixMulAddTypeMismatch {
                            op_name,
                            operand_name: "Result Type",
                            function: function_id,
                            block: block_id,
                        });
                    }

                    // Get matrix parameters
                    let a_params = get_matrix_params(a_type_id, ctx);
                    let b_params = get_matrix_params(b_type_id, ctx);
                    let c_params = get_matrix_params(c_type_id, ctx);
                    let d_params = get_matrix_params(d_type_id, ctx);

                    if let (Some(a), Some(b), Some(c), Some(d)) = (a_params, b_params, c_params, d_params) {
                        // Check scopes match
                        let scopes = [a.0, b.0, c.0, d.0];
                        for i in 0..scopes.len() {
                            for j in (i + 1)..scopes.len() {
                                if let (Some(s1), Some(s2)) = (scopes[i], scopes[j]) {
                                    if s1 != s2 {
                                        return Err(ValidationError::CooperativeMatrixMulAddScopeMismatch {
                                            op_name,
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }
                                }
                            }
                        }

                        // Check M dimension: A_rows == C_rows == D_rows
                        if let (Some(a_rows), Some(c_rows)) = (a.1, c.1) {
                            if a_rows != c_rows {
                                return Err(ValidationError::CooperativeMatrixMulAddMDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
                                });
                            }
                        }
                        if let (Some(a_rows), Some(d_rows)) = (a.1, d.1) {
                            if a_rows != d_rows {
                                return Err(ValidationError::CooperativeMatrixMulAddMDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
                                });
                            }
                        }

                        // Check N dimension: B_cols == C_cols == D_cols
                        if let (Some(b_cols), Some(c_cols)) = (b.2, c.2) {
                            if b_cols != c_cols {
                                return Err(ValidationError::CooperativeMatrixMulAddNDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
                                });
                            }
                        }
                        if let (Some(b_cols), Some(d_cols)) = (b.2, d.2) {
                            if b_cols != d_cols {
                                return Err(ValidationError::CooperativeMatrixMulAddNDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
                                });
                            }
                        }

                        // Check K dimension: A_cols == B_rows
                        if let (Some(a_cols), Some(b_rows)) = (a.2, b.1) {
                            if a_cols != b_rows {
                                return Err(ValidationError::CooperativeMatrixMulAddKDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
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

/// Validates OpCooperativeMatrixMulAddKHR instructions.
pub struct CooperativeMatrixMulAddKHRRule;

impl ValidationRule for CooperativeMatrixMulAddKHRRule {
    fn name(&self) -> &'static str {
        "cooperative-matrix-muladd-khr"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let function_id = function.def.as_ref().and_then(|d| d.result_id).map(id_from_u32);

            for block in &function.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(id_from_u32);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::CooperativeMatrixMulAddKHR {
                        continue;
                    }

                    let op_name = "OpCooperativeMatrixMulAddKHR";

                    // Get types for D (result), A, B, C operands
                    let d_type_id = inst.result_type.unwrap_or(0);

                    // Operands: A, B, C, (optional operands)
                    let a_id = inst.operands.first().and_then(|op| {
                        if let Operand::IdRef(id) = op { Some(*id) } else { None }
                    });
                    let b_id = inst.operands.get(1).and_then(|op| {
                        if let Operand::IdRef(id) = op { Some(*id) } else { None }
                    });
                    let c_id = inst.operands.get(2).and_then(|op| {
                        if let Operand::IdRef(id) = op { Some(*id) } else { None }
                    });

                    // Get operand types
                    let a_type_id = a_id
                        .and_then(|id| ResultId::try_from(id).ok())
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type)
                        .unwrap_or(0);
                    let b_type_id = b_id
                        .and_then(|id| ResultId::try_from(id).ok())
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type)
                        .unwrap_or(0);
                    let c_type_id = c_id
                        .and_then(|id| ResultId::try_from(id).ok())
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type)
                        .unwrap_or(0);

                    // For KHR version, A, B, C, D must all be OpTypeCooperativeMatrixKHR
                    if !is_cooperative_matrix_khr_type(a_type_id, ctx) {
                        return Err(ValidationError::CooperativeMatrixMulAddTypeMismatch {
                            op_name,
                            operand_name: "A",
                            function: function_id,
                            block: block_id,
                        });
                    }
                    if !is_cooperative_matrix_khr_type(b_type_id, ctx) {
                        return Err(ValidationError::CooperativeMatrixMulAddTypeMismatch {
                            op_name,
                            operand_name: "B",
                            function: function_id,
                            block: block_id,
                        });
                    }
                    if !is_cooperative_matrix_khr_type(c_type_id, ctx) {
                        return Err(ValidationError::CooperativeMatrixMulAddTypeMismatch {
                            op_name,
                            operand_name: "C",
                            function: function_id,
                            block: block_id,
                        });
                    }
                    if !is_cooperative_matrix_khr_type(d_type_id, ctx) {
                        return Err(ValidationError::CooperativeMatrixMulAddTypeMismatch {
                            op_name,
                            operand_name: "Result Type",
                            function: function_id,
                            block: block_id,
                        });
                    }

                    // Get matrix parameters and check dimensions
                    let a_params = get_matrix_params(a_type_id, ctx);
                    let b_params = get_matrix_params(b_type_id, ctx);
                    let c_params = get_matrix_params(c_type_id, ctx);
                    let d_params = get_matrix_params(d_type_id, ctx);

                    if let (Some(a), Some(b), Some(c), Some(d)) = (a_params, b_params, c_params, d_params) {
                        // Check scopes match
                        let scopes = [a.0, b.0, c.0, d.0];
                        for i in 0..scopes.len() {
                            for j in (i + 1)..scopes.len() {
                                if let (Some(s1), Some(s2)) = (scopes[i], scopes[j]) {
                                    if s1 != s2 {
                                        return Err(ValidationError::CooperativeMatrixMulAddScopeMismatch {
                                            op_name,
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }
                                }
                            }
                        }

                        // Check M dimension: A_rows == C_rows == D_rows
                        if let (Some(a_rows), Some(c_rows)) = (a.1, c.1) {
                            if a_rows != c_rows {
                                return Err(ValidationError::CooperativeMatrixMulAddMDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
                                });
                            }
                        }
                        if let (Some(a_rows), Some(d_rows)) = (a.1, d.1) {
                            if a_rows != d_rows {
                                return Err(ValidationError::CooperativeMatrixMulAddMDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
                                });
                            }
                        }

                        // Check N dimension: B_cols == C_cols == D_cols
                        if let (Some(b_cols), Some(c_cols)) = (b.2, c.2) {
                            if b_cols != c_cols {
                                return Err(ValidationError::CooperativeMatrixMulAddNDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
                                });
                            }
                        }
                        if let (Some(b_cols), Some(d_cols)) = (b.2, d.2) {
                            if b_cols != d_cols {
                                return Err(ValidationError::CooperativeMatrixMulAddNDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
                                });
                            }
                        }

                        // Check K dimension: A_cols == B_rows
                        if let (Some(a_cols), Some(b_rows)) = (a.2, b.1) {
                            if a_cols != b_rows {
                                return Err(ValidationError::CooperativeMatrixMulAddKDimensionMismatch {
                                    op_name,
                                    function: function_id,
                                    block: block_id,
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
