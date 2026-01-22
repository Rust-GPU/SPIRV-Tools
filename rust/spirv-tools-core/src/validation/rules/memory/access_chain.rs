//! Access chain instruction validation rules.
//!
//! This module validates SPIR-V access chain instructions including:
//!
//! - OpAccessChain/OpInBoundsAccessChain: Standard access chain validation
//! - OpPtrAccessChain/OpInBoundsPtrAccessChain: Pointer access chain validation
//! - OpRawAccessChainNV: Raw access chain validation (NVIDIA extension)

use rspirv::dr::Operand;
use rspirv::spirv::{Decoration, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId};
use crate::validation::ValidationResult;

use super::helpers::{get_pointee_type, get_pointer_storage_class, id_from_u32, type_id_from_u32};

// ============================================================================
// Access Chain Validation Rule
// ============================================================================

/// Validates OpAccessChain, OpInBoundsAccessChain, OpPtrAccessChain, and OpInBoundsPtrAccessChain instructions.
pub struct AccessChainRule;

impl ValidationRule for AccessChainRule {
    fn name(&self) -> &'static str {
        "access-chain"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(id_from_u32)
                .unwrap_or_else(|| id_from_u32(0));

            for block in &func.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .map(id_from_u32)
                    .unwrap_or_else(|| id_from_u32(0));

                for inst in &block.instructions {
                    if !matches!(
                        inst.class.opcode,
                        Op::AccessChain
                            | Op::InBoundsAccessChain
                            | Op::PtrAccessChain
                            | Op::InBoundsPtrAccessChain
                    ) {
                        continue;
                    }

                    // For PtrAccessChain variants, the first operand is the element operand (not an index)
                    let is_ptr_access_chain = matches!(
                        inst.class.opcode,
                        Op::PtrAccessChain | Op::InBoundsPtrAccessChain
                    );

                    let opcode = inst.class.opcode;

                    // Get result type (must be a pointer type)
                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };
                    let Some(result_type_rid) = ResultId::try_from(result_type_id).ok() else {
                        continue;
                    };
                    let Some(result_type) = ctx.definitions.get(&result_type_rid) else {
                        continue;
                    };

                    if result_type.class.opcode != Op::TypePointer {
                        return Err(ValidationError::AccessChainResultTypeNotPointer {
                            function: func_id,
                            block: block_id,
                            instruction: opcode,
                            result_type: type_id_from_u32(result_type_id),
                        }
                        .into());
                    }

                    // Get base operand
                    let Some(Operand::IdRef(base_id)) = inst.operands.first() else {
                        continue;
                    };

                    // Look up the base instruction
                    let Some(base_rid) = ResultId::try_from(*base_id).ok() else {
                        continue;
                    };
                    let Some(base_inst) = ctx.definitions.get(&base_rid) else {
                        continue;
                    };

                    // Get base type (must be a pointer)
                    let Some(base_type_id) = base_inst.result_type else {
                        continue;
                    };
                    let Some(base_type_rid) = ResultId::try_from(base_type_id).ok() else {
                        continue;
                    };
                    let Some(base_type) = ctx.definitions.get(&base_type_rid) else {
                        continue;
                    };

                    if base_type.class.opcode != Op::TypePointer
                        && base_type.class.opcode != Op::TypeUntypedPointerKHR
                    {
                        return Err(ValidationError::AccessChainBaseNotPointer {
                            function: func_id,
                            block: block_id,
                            instruction: opcode,
                            base_type: type_id_from_u32(base_type_id),
                        }
                        .into());
                    }

                    // Storage classes must match
                    let result_sc = get_pointer_storage_class(result_type);
                    let base_sc = get_pointer_storage_class(base_type);
                    if let (Some(r_sc), Some(b_sc)) = (result_sc, base_sc) {
                        if r_sc != b_sc {
                            return Err(ValidationError::AccessChainStorageClassMismatch {
                                function: func_id,
                                block: block_id,
                                instruction: opcode,
                                base_storage_class: b_sc,
                                result_storage_class: r_sc,
                            }
                            .into());
                        }
                    }

                    // Validate indices
                    // For PtrAccessChain variants:
                    // - operand 0: base (pointer)
                    // - operand 1: element (integer for pointer arithmetic, does not index into composite)
                    // - operand 2+: indexes into the composite type
                    // For regular AccessChain:
                    // - operand 0: base (pointer)
                    // - operand 1+: indexes into the composite type

                    let base_pointee_id = get_pointee_type(base_type);
                    if let Some(mut current_type_id) = base_pointee_id {
                        // For PtrAccessChain, first validate the Element operand (operand 1)
                        // which must be an integer but doesn't traverse into composite types
                        let indices_start = if is_ptr_access_chain {
                            // Validate the Element operand
                            if let Some(Operand::IdRef(element_id)) = inst.operands.get(1) {
                                let Some(element_rid) = ResultId::try_from(*element_id).ok() else {
                                    continue;
                                };
                                let Some(element_inst) = ctx.definitions.get(&element_rid) else {
                                    continue;
                                };
                                let Some(element_type_id) = element_inst.result_type else {
                                    continue;
                                };
                                let Some(element_type_rid) =
                                    ResultId::try_from(element_type_id).ok()
                                else {
                                    continue;
                                };
                                let Some(element_type) = ctx.definitions.get(&element_type_rid)
                                else {
                                    continue;
                                };

                                if element_type.class.opcode != Op::TypeInt {
                                    return Err(ValidationError::AccessChainIndexTypeInvalid {
                                        function: func_id,
                                        block: block_id,
                                        instruction: opcode,
                                        operand_index: 1,
                                        found: type_id_from_u32(element_type_id),
                                    }
                                    .into());
                                }
                            }
                            2 // Start composite indices at operand 2
                        } else {
                            1 // Start composite indices at operand 1
                        };

                        for (idx, operand) in inst.operands.iter().skip(indices_start).enumerate() {
                            let Operand::IdRef(index_id) = operand else {
                                continue;
                            };

                            // Index must be integer type
                            let Some(index_rid) = ResultId::try_from(*index_id).ok() else {
                                continue;
                            };
                            let Some(index_inst) = ctx.definitions.get(&index_rid) else {
                                continue;
                            };
                            let Some(index_type_id) = index_inst.result_type else {
                                continue;
                            };
                            let Some(index_type_rid) = ResultId::try_from(index_type_id).ok()
                            else {
                                continue;
                            };
                            let Some(index_type) = ctx.definitions.get(&index_type_rid) else {
                                continue;
                            };

                            // Report the operand index relative to the full operand list
                            let actual_operand_index = idx + indices_start;

                            if index_type.class.opcode != Op::TypeInt {
                                return Err(ValidationError::AccessChainIndexTypeInvalid {
                                    function: func_id,
                                    block: block_id,
                                    instruction: opcode,
                                    operand_index: actual_operand_index,
                                    found: type_id_from_u32(index_type_id),
                                }
                                .into());
                            }

                            // Check for negative signed integer constants in logical addressing
                            // (SPIR-V spec restriction for logical addressing mode)
                            if ctx.is_logical_addressing() {
                                // Check if index is a signed integer type (signedness operand == 1)
                                let is_signed = index_type
                                    .operands
                                    .get(1)
                                    .map(|op| matches!(op, Operand::LiteralBit32(1)))
                                    .unwrap_or(false);

                                if is_signed
                                    && matches!(
                                        index_inst.class.opcode,
                                        Op::Constant | Op::SpecConstant
                                    )
                                {
                                    // Get the constant value and check if negative
                                    if let Some(Operand::LiteralBit32(val)) =
                                        index_inst.operands.first()
                                    {
                                        // Interpret as signed 32-bit
                                        let signed_val = *val as i32;
                                        if signed_val < 0 {
                                            return Err(
                                                ValidationError::AccessChainNegativeIndex {
                                                    function: func_id,
                                                    block: block_id,
                                                    instruction: opcode,
                                                    operand_index: actual_operand_index,
                                                    value: signed_val as i64,
                                                }
                                                .into(),
                                            );
                                        }
                                    }
                                }
                            }

                            // Get current type instruction
                            let Some(current_type_rid) = ResultId::try_from(current_type_id).ok()
                            else {
                                break;
                            };
                            let Some(current_type) = ctx.definitions.get(&current_type_rid) else {
                                break;
                            };

                            // Traverse into the type based on index
                            match current_type.class.opcode {
                                Op::TypeArray
                                | Op::TypeRuntimeArray
                                | Op::TypeVector
                                | Op::TypeMatrix => {
                                    // Element type is first operand
                                    if let Some(Operand::IdRef(elem_id)) =
                                        current_type.operands.first()
                                    {
                                        current_type_id = *elem_id;
                                    } else {
                                        break;
                                    }
                                }
                                Op::TypeStruct => {
                                    // Index must be a constant for structs
                                    let is_constant = matches!(
                                        index_inst.class.opcode,
                                        Op::Constant | Op::ConstantNull | Op::SpecConstant
                                    );
                                    if !is_constant {
                                        return Err(
                                            ValidationError::AccessChainStructIndexNotLiteral {
                                                function: func_id,
                                                block: block_id,
                                                instruction: opcode,
                                                composite_type: type_id_from_u32(current_type_id),
                                            }
                                            .into(),
                                        );
                                    }

                                    // Get member type by index
                                    if let Some(Operand::LiteralBit32(member_idx)) =
                                        index_inst.operands.first()
                                    {
                                        if let Some(Operand::IdRef(member_type_id)) =
                                            current_type.operands.get(*member_idx as usize)
                                        {
                                            current_type_id = *member_type_id;
                                        } else {
                                            return Err(
                                                ValidationError::AccessChainStructIndexOutOfBounds {
                                                    function: func_id,
                                                    block: block_id,
                                                    instruction: opcode,
                                                    composite_type: type_id_from_u32(
                                                        current_type_id,
                                                    ),
                                                    index: *member_idx,
                                                    bound: current_type.operands.len() as u32,
                                                }.into(),
                        );
                                        }
                                    } else {
                                        break;
                                    }
                                }
                                _ => {
                                    return Err(ValidationError::AccessChainNonCompositeTarget {
                                        function: func_id,
                                        block: block_id,
                                        instruction: opcode,
                                        composite_type: type_id_from_u32(current_type_id),
                                    }
                                    .into());
                                }
                            }
                        }

                        // Check final type matches result pointee type
                        // For PtrAccessChain with no composite indices, result must match base pointee
                        // For access chains with composite indices, result must match traversed type
                        if let Some(result_pointee_id) = get_pointee_type(result_type) {
                            let has_composite_indices = inst.operands.len() > indices_start;
                            if has_composite_indices {
                                // Result must match the type we traversed to
                                if current_type_id != result_pointee_id {
                                    return Err(ValidationError::AccessChainResultTypeMismatch {
                                        function: func_id,
                                        block: block_id,
                                        instruction: opcode,
                                        expected: type_id_from_u32(current_type_id),
                                        found: type_id_from_u32(result_pointee_id),
                                    }
                                    .into());
                                }
                            } else if is_ptr_access_chain {
                                // PtrAccessChain with just element operand: result must match base pointee
                                if current_type_id != result_pointee_id {
                                    return Err(ValidationError::AccessChainResultTypeMismatch {
                                        function: func_id,
                                        block: block_id,
                                        instruction: opcode,
                                        expected: type_id_from_u32(current_type_id),
                                        found: type_id_from_u32(result_pointee_id),
                                    }
                                    .into());
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
// Raw Access Chain Rule
// ============================================================================

/// Validates OpRawAccessChainNV instructions.
pub struct RawAccessChainRule;

impl ValidationRule for RawAccessChainRule {
    fn name(&self) -> &'static str {
        "raw-access-chain"
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
                    if inst.class.opcode != Op::RawAccessChainNV {
                        continue;
                    }

                    // Validate result type is OpTypePointer
                    let result_type_id = match inst.result_type {
                        Some(id) => id,
                        None => continue,
                    };

                    let result_type_inst = match ResultId::try_from(result_type_id)
                        .ok()
                        .and_then(|id| ctx.definitions.get(&id))
                    {
                        Some(inst) => inst,
                        None => continue,
                    };

                    if result_type_inst.class.opcode != Op::TypePointer {
                        return Err(ValidationError::RawAccessChainResultNotPointer {
                            function: function_id,
                            block: block_id,
                        }
                        .into());
                    }

                    // Validate storage class
                    let storage_class = match result_type_inst.operands.first() {
                        Some(Operand::StorageClass(sc)) => *sc,
                        _ => continue,
                    };

                    if storage_class != StorageClass::StorageBuffer
                        && storage_class != StorageClass::PhysicalStorageBuffer
                        && storage_class != StorageClass::Uniform
                    {
                        return Err(ValidationError::RawAccessChainInvalidStorageClass {
                            function: function_id,
                            block: block_id,
                        }
                        .into());
                    }

                    // Validate pointed type is not Array, Matrix, or Struct
                    if let Some(Operand::IdRef(pointee_id)) = result_type_inst.operands.get(1) {
                        if let Ok(pointee_result_id) = ResultId::try_from(*pointee_id) {
                            if let Some(pointee_inst) = ctx.definitions.get(&pointee_result_id) {
                                let pointee_op = pointee_inst.class.opcode;
                                if pointee_op == Op::TypeArray
                                    || pointee_op == Op::TypeMatrix
                                    || pointee_op == Op::TypeStruct
                                {
                                    return Err(
                                        ValidationError::RawAccessChainInvalidPointedType {
                                            function: function_id,
                                            block: block_id,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                    }

                    // Validate Stride (operand 1) is OpConstant with OpTypeInt
                    if let Some(Operand::IdRef(stride_id)) = inst.operands.get(1) {
                        if let Ok(stride_result_id) = ResultId::try_from(*stride_id) {
                            if let Some(stride_inst) = ctx.definitions.get(&stride_result_id) {
                                if stride_inst.class.opcode != Op::Constant {
                                    return Err(ValidationError::RawAccessChainStrideNotConstant {
                                        function: function_id,
                                        block: block_id,
                                    }
                                    .into());
                                }

                                // Check stride type is OpTypeInt
                                if let Some(stride_type) = stride_inst.result_type {
                                    if let Ok(stride_type_id) = ResultId::try_from(stride_type) {
                                        if let Some(stride_type_inst) =
                                            ctx.definitions.get(&stride_type_id)
                                        {
                                            if stride_type_inst.class.opcode != Op::TypeInt {
                                                return Err(
                                                    ValidationError::RawAccessChainStrideNotInt {
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

                    // Validate Index (operand 2) is 32-bit int
                    validate_32bit_int_operand(inst, 2, "Index", ctx, function_id, block_id)?;

                    // Validate Offset (operand 3) is 32-bit int
                    validate_32bit_int_operand(inst, 3, "Offset", ctx, function_id, block_id)?
                }
            }
        }

        Ok(())
    }
}

/// Helper to validate an operand is a 32-bit integer type.
fn validate_32bit_int_operand(
    inst: &rspirv::dr::Instruction,
    operand_idx: usize,
    operand_name: &'static str,
    ctx: &ValidationContext<'_>,
    function_id: Option<Id>,
    block_id: Option<Id>,
) -> ValidationResult {
    if let Some(Operand::IdRef(operand_id)) = inst.operands.get(operand_idx) {
        if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
            if let Some(operand_inst) = ctx.definitions.get(&operand_result_id) {
                if let Some(operand_type) = operand_inst.result_type {
                    if let Ok(operand_type_id) = ResultId::try_from(operand_type) {
                        if let Some(operand_type_inst) = ctx.definitions.get(&operand_type_id) {
                            if operand_type_inst.class.opcode != Op::TypeInt {
                                return Err(ValidationError::RawAccessChainOperandNot32BitInt {
                                    function: function_id,
                                    block: block_id,
                                    operand_name,
                                }
                                .into());
                            }
                            // Check width is 32
                            if let Some(Operand::LiteralBit32(width)) =
                                operand_type_inst.operands.first()
                            {
                                if *width != 32 {
                                    return Err(
                                        ValidationError::RawAccessChainOperandNot32BitInt {
                                            function: function_id,
                                            block: block_id,
                                            operand_name,
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

// ============================================================================
// Ptr Access Chain Rule
// ============================================================================

/// Validates OpPtrAccessChain and OpInBoundsPtrAccessChain instructions.
pub struct PtrAccessChainRule;

impl ValidationRule for PtrAccessChainRule {
    fn name(&self) -> &'static str {
        "ptr-access-chain"
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
                    if inst.class.opcode != Op::PtrAccessChain
                        && inst.class.opcode != Op::InBoundsPtrAccessChain
                    {
                        continue;
                    }

                    // Get Base operand (operand 0)
                    let base_id = match inst.operands.first() {
                        Some(Operand::IdRef(id)) => *id,
                        _ => continue,
                    };

                    let base_inst = match ResultId::try_from(base_id)
                        .ok()
                        .and_then(|id| ctx.definitions.get(&id))
                    {
                        Some(inst) => inst,
                        None => continue,
                    };

                    let base_type_id = match base_inst.result_type {
                        Some(id) => id,
                        None => continue,
                    };

                    let base_type_inst = match ResultId::try_from(base_type_id)
                        .ok()
                        .and_then(|id| ctx.definitions.get(&id))
                    {
                        Some(inst) => inst,
                        None => continue,
                    };

                    // Get storage class from base type
                    let storage_class = match base_type_inst.operands.first() {
                        Some(Operand::StorageClass(sc)) => *sc,
                        _ => continue,
                    };

                    // Validate Element operand (operand 1) is integer
                    if let Some(Operand::IdRef(element_id)) = inst.operands.get(1) {
                        if let Ok(element_result_id) = ResultId::try_from(*element_id) {
                            if let Some(element_inst) = ctx.definitions.get(&element_result_id) {
                                if let Some(element_type) = element_inst.result_type {
                                    if let Ok(element_type_id) = ResultId::try_from(element_type) {
                                        if let Some(element_type_inst) =
                                            ctx.definitions.get(&element_type_id)
                                        {
                                            if element_type_inst.class.opcode != Op::TypeInt {
                                                return Err(
                                                    ValidationError::PtrAccessChainElementNotInt {
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

                    // For Vulkan, validate storage class restrictions
                    if ctx.is_vulkan_env() {
                        let has_variable_pointers =
                            ctx.has_capability(Capability::VariablePointers);
                        let has_variable_pointers_storage_buffer =
                            ctx.has_capability(Capability::VariablePointersStorageBuffer);

                        match storage_class {
                            StorageClass::Workgroup => {
                                if !has_variable_pointers {
                                    return Err(
                                        ValidationError::PtrAccessChainWorkgroupRequiresVariablePointers {
                                            function: function_id,
                                            block: block_id,
                                        }.into(),
                        );
                                }
                            }
                            StorageClass::StorageBuffer => {
                                if !has_variable_pointers && !has_variable_pointers_storage_buffer {
                                    return Err(
                                        ValidationError::PtrAccessChainStorageBufferRequiresVariablePointers {
                                            function: function_id,
                                            block: block_id,
                                        }.into(),
                        );
                                }
                            }
                            StorageClass::PhysicalStorageBuffer => {
                                // PhysicalStorageBuffer is allowed
                            }
                            _ => {
                                return Err(
                                    ValidationError::PtrAccessChainInvalidVulkanStorageClass {
                                        function: function_id,
                                        block: block_id,
                                    }
                                    .into(),
                                );
                            }
                        }
                    }

                    // Check ArrayStride decoration for Shader capability with certain storage classes
                    if ctx.has_capability(Capability::Shader) {
                        let needs_array_stride = matches!(
                            storage_class,
                            StorageClass::Uniform
                                | StorageClass::StorageBuffer
                                | StorageClass::PhysicalStorageBuffer
                                | StorageClass::PushConstant
                        );

                        if needs_array_stride {
                            // Check if base type has ArrayStride decoration
                            let has_array_stride =
                                has_decoration_on_type(base_type_id, Decoration::ArrayStride, ctx);

                            if !has_array_stride {
                                return Err(ValidationError::PtrAccessChainMissingArrayStride {
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

/// Check if a type has a specific decoration.
fn has_decoration_on_type(
    type_id: u32,
    target_decoration: Decoration,
    ctx: &ValidationContext<'_>,
) -> bool {
    for inst in &ctx.module.annotations {
        if inst.class.opcode == Op::Decorate {
            if let Some(Operand::IdRef(decorated_id)) = inst.operands.first() {
                if *decorated_id == type_id {
                    if let Some(Operand::Decoration(dec)) = inst.operands.get(1) {
                        if *dec == target_decoration {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}
