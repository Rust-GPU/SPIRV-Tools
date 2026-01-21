//! Load and store instruction validation rules.
//!
//! This module validates OpLoad and OpStore instructions.

use rspirv::dr::Operand;
use rspirv::spirv::{MemoryAccess, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::types::ResultId;

use super::helpers::{
    allows_non_private_pointer, contains_runtime_array, get_largest_scalar_type, get_pointee_type,
    get_pointer_storage_class, id_from_u32, is_logical_pointer_producer, is_readonly_storage_class,
    type_id_from_u32,
};

// ============================================================================
// Load Validation Rule
// ============================================================================

/// Validates OpLoad instructions.
pub struct LoadRule;

impl ValidationRule for LoadRule {
    fn name(&self) -> &'static str {
        "load"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::Load {
                continue;
            }

            // Get result type
            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            // Get pointer operand
            let Some(Operand::IdRef(pointer_id)) = inst.operands.first() else {
                continue;
            };

            // Look up the pointer instruction
            let Some(pointer_rid) = ResultId::try_from(*pointer_id).ok() else {
                continue;
            };
            let Some(pointer_inst) = ctx.definitions.get(&pointer_rid) else {
                continue;
            };

            // Check if pointer comes from a logical pointer producer
            if !is_logical_pointer_producer(pointer_inst.class.opcode) {
                return Err(ValidationError::NotALogicalPointer {
                    instruction: Op::Load,
                    pointer: id_from_u32(*pointer_id),
                    source_opcode: pointer_inst.class.opcode,
                }.into());
            }

            // Get pointer type
            let Some(pointer_type_id) = pointer_inst.result_type else {
                continue;
            };
            let Some(pointer_type_rid) = ResultId::try_from(pointer_type_id).ok() else {
                continue;
            };
            let Some(pointer_type) = ctx.definitions.get(&pointer_type_rid) else {
                continue;
            };

            if pointer_type.class.opcode != Op::TypePointer
                && pointer_type.class.opcode != Op::TypeUntypedPointerKHR
            {
                return Err(ValidationError::LoadPointerNotPointerType {
                    pointer: id_from_u32(*pointer_id),
                }.into());
            }

            // For typed pointers, check pointee type matches result type
            if pointer_type.class.opcode == Op::TypePointer {
                if let Some(pointee_id) = get_pointee_type(pointer_type) {
                    if pointee_id != result_type_id {
                        return Err(ValidationError::LoadResultTypeMismatch {
                            result_type: type_id_from_u32(result_type_id),
                            pointee_type: type_id_from_u32(pointee_id),
                        }.into());
                    }
                }
            }

            // Cannot load a runtime array
            if contains_runtime_array(
                result_type_id,
                ctx.definitions,
                &mut std::collections::HashSet::new(),
            ) {
                return Err(ValidationError::LoadRuntimeArray.into());
            }

            // Check memory access operands if present
            if let Some(Operand::MemoryAccess(access)) = inst.operands.get(1) {
                // For Load, use the result type for largest scalar calculation
                validate_memory_access_for_load(
                    ctx,
                    inst,
                    *pointer_id,
                    result_type_id,
                    *access,
                    1,
                )?;
            } else {
                // No memory access operand - check if PhysicalStorageBuffer
                if let Some(sc) = get_pointer_storage_class_for_value(ctx, *pointer_id) {
                    if sc == StorageClass::PhysicalStorageBuffer {
                        return Err(ValidationError::PhysicalStorageBufferRequiresAligned.into());
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates memory access flags for load operations.
fn validate_memory_access_for_load(
    ctx: &ValidationContext<'_>,
    inst: &rspirv::dr::Instruction,
    pointer_id: u32,
    accessed_type_id: u32,
    access: MemoryAccess,
    memory_access_operand_index: usize,
) -> ValidationResult {
    // MakePointerAvailable cannot be used with OpLoad
    if access.contains(MemoryAccess::MAKE_POINTER_AVAILABLE) {
        return Err(ValidationError::LoadMakePointerAvailable.into());
    }

    // Get storage class
    let storage_class = get_pointer_storage_class_for_value(ctx, pointer_id);

    // Check NonPrivatePointer storage class requirements
    if access.contains(MemoryAccess::NON_PRIVATE_POINTER) {
        if let Some(sc) = storage_class {
            if !allows_non_private_pointer(sc) {
                return Err(ValidationError::NonPrivatePointerInvalidStorageClass {
                    storage_class: sc,
                }.into());
            }
        }
    }

    // MakePointerVisible requires NonPrivatePointer
    if access.contains(MemoryAccess::MAKE_POINTER_VISIBLE)
        && !access.contains(MemoryAccess::NON_PRIVATE_POINTER)
    {
        return Err(ValidationError::MakeVisibleRequiresNonPrivate.into());
    }

    // PhysicalStorageBuffer requires Aligned
    if let Some(sc) = storage_class {
        if sc == StorageClass::PhysicalStorageBuffer {
            if !access.contains(MemoryAccess::ALIGNED) {
                return Err(ValidationError::PhysicalStorageBufferRequiresAligned.into());
            }
        }
    }

    // Validate Aligned operand if present
    if access.contains(MemoryAccess::ALIGNED) {
        // Aligned operand value follows the memory access mask
        if let Some(Operand::LiteralBit32(aligned_value)) =
            inst.operands.get(memory_access_operand_index + 1)
        {
            // Must be a power of two
            if *aligned_value == 0 || (*aligned_value & (*aligned_value - 1)) != 0 {
                return Err(ValidationError::AlignedValueNotPowerOfTwo {
                    value: *aligned_value,
                }.into());
            }

            // For PhysicalStorageBuffer, alignment must be >= largest scalar type
            if let Some(sc) = storage_class {
                if sc == StorageClass::PhysicalStorageBuffer {
                    let largest_scalar = get_largest_scalar_type(
                        accessed_type_id,
                        ctx.definitions,
                        &mut std::collections::HashSet::new(),
                    );
                    if largest_scalar > 0 && *aligned_value < largest_scalar {
                        return Err(ValidationError::AlignedValueTooSmall {
                            alignment: *aligned_value,
                            largest_scalar,
                        }.into());
                    }
                }
            }
        }
    }

    Ok(())
}

/// Gets the storage class for a pointer value.
fn get_pointer_storage_class_for_value(
    ctx: &ValidationContext<'_>,
    pointer_id: u32,
) -> Option<StorageClass> {
    let result_id = ResultId::try_from(pointer_id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;
    let type_id = inst.result_type?;
    let type_rid = ResultId::try_from(type_id).ok()?;
    let type_inst = ctx.definitions.get(&type_rid)?;
    get_pointer_storage_class(type_inst)
}

// ============================================================================
// Store Validation Rule
// ============================================================================

/// Validates OpStore instructions.
pub struct StoreRule;

impl ValidationRule for StoreRule {
    fn name(&self) -> &'static str {
        "store"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::Store {
                continue;
            }

            // Get pointer operand
            let Some(Operand::IdRef(pointer_id)) = inst.operands.first() else {
                continue;
            };

            // Get object operand (presence check only - value not needed yet)
            let Some(Operand::IdRef(_object_id)) = inst.operands.get(1) else {
                continue;
            };

            // Look up the pointer instruction
            let Some(pointer_rid) = ResultId::try_from(*pointer_id).ok() else {
                continue;
            };
            let Some(pointer_inst) = ctx.definitions.get(&pointer_rid) else {
                continue;
            };

            // Check if pointer comes from a logical pointer producer
            if !is_logical_pointer_producer(pointer_inst.class.opcode) {
                return Err(ValidationError::NotALogicalPointer {
                    instruction: Op::Store,
                    pointer: id_from_u32(*pointer_id),
                    source_opcode: pointer_inst.class.opcode,
                }.into());
            }

            // Get pointer type
            let Some(pointer_type_id) = pointer_inst.result_type else {
                continue;
            };
            let Some(pointer_type_rid) = ResultId::try_from(pointer_type_id).ok() else {
                continue;
            };
            let Some(pointer_type) = ctx.definitions.get(&pointer_type_rid) else {
                continue;
            };

            if pointer_type.class.opcode != Op::TypePointer
                && pointer_type.class.opcode != Op::TypeUntypedPointerKHR
            {
                return Err(ValidationError::StorePointerNotPointerType {
                    pointer: id_from_u32(*pointer_id),
                }.into());
            }

            // Check storage class is not read-only
            if let Some(sc) = get_pointer_storage_class(pointer_type) {
                if is_readonly_storage_class(sc) {
                    return Err(ValidationError::StoreToReadOnlyStorageClass {
                        pointer: id_from_u32(*pointer_id),
                        storage_class: sc,
                    }.into());
                }

                // ShaderRecordBufferKHR is also read-only
                if sc == StorageClass::ShaderRecordBufferKHR {
                    return Err(ValidationError::StoreToReadOnlyStorageClass {
                        pointer: id_from_u32(*pointer_id),
                        storage_class: sc,
                    }.into());
                }
            }

            // Note: Store type compatibility (pointer vs object type) is checked by
            // StoreTypeCompatibilityRule in pointers.rs, which properly handles
            // the relax_struct_store option.

            // Get pointee type for largest scalar calculation (for typed pointers)
            let accessed_type_id = if pointer_type.class.opcode == Op::TypePointer {
                get_pointee_type(pointer_type)
            } else {
                None // Untyped pointers don't have a pointee type to check
            };

            // Check memory access operands if present
            if let Some(Operand::MemoryAccess(access)) = inst.operands.get(2) {
                validate_memory_access_for_store(
                    ctx,
                    inst,
                    *pointer_id,
                    accessed_type_id,
                    *access,
                    2,
                )?;
            } else {
                // No memory access operand - check if PhysicalStorageBuffer
                if let Some(sc) = get_pointer_storage_class_for_value(ctx, *pointer_id) {
                    if sc == StorageClass::PhysicalStorageBuffer {
                        return Err(ValidationError::PhysicalStorageBufferRequiresAligned.into());
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates memory access flags for store operations.
fn validate_memory_access_for_store(
    ctx: &ValidationContext<'_>,
    inst: &rspirv::dr::Instruction,
    pointer_id: u32,
    accessed_type_id: Option<u32>,
    access: MemoryAccess,
    memory_access_operand_index: usize,
) -> ValidationResult {
    // MakePointerVisible cannot be used with OpStore
    if access.contains(MemoryAccess::MAKE_POINTER_VISIBLE) {
        return Err(ValidationError::StoreMakePointerVisible.into());
    }

    // Get storage class
    let storage_class = get_pointer_storage_class_for_value(ctx, pointer_id);

    // Check NonPrivatePointer storage class requirements
    if access.contains(MemoryAccess::NON_PRIVATE_POINTER) {
        if let Some(sc) = storage_class {
            if !allows_non_private_pointer(sc) {
                return Err(ValidationError::NonPrivatePointerInvalidStorageClass {
                    storage_class: sc,
                }.into());
            }
        }
    }

    // MakePointerAvailable requires NonPrivatePointer
    if access.contains(MemoryAccess::MAKE_POINTER_AVAILABLE)
        && !access.contains(MemoryAccess::NON_PRIVATE_POINTER)
    {
        return Err(ValidationError::MakeAvailableRequiresNonPrivate.into());
    }

    // PhysicalStorageBuffer requires Aligned
    if let Some(sc) = storage_class {
        if sc == StorageClass::PhysicalStorageBuffer {
            if !access.contains(MemoryAccess::ALIGNED) {
                return Err(ValidationError::PhysicalStorageBufferRequiresAligned.into());
            }
        }
    }

    // Validate Aligned operand if present
    if access.contains(MemoryAccess::ALIGNED) {
        // Aligned operand value follows the memory access mask
        if let Some(Operand::LiteralBit32(aligned_value)) =
            inst.operands.get(memory_access_operand_index + 1)
        {
            // Must be a power of two
            if *aligned_value == 0 || (*aligned_value & (*aligned_value - 1)) != 0 {
                return Err(ValidationError::AlignedValueNotPowerOfTwo {
                    value: *aligned_value,
                }.into());
            }

            // For PhysicalStorageBuffer, alignment must be >= largest scalar type
            if let Some(sc) = storage_class {
                if sc == StorageClass::PhysicalStorageBuffer {
                    if let Some(type_id) = accessed_type_id {
                        let largest_scalar = get_largest_scalar_type(
                            type_id,
                            ctx.definitions,
                            &mut std::collections::HashSet::new(),
                        );
                        if largest_scalar > 0 && *aligned_value < largest_scalar {
                            return Err(ValidationError::AlignedValueTooSmall {
                                alignment: *aligned_value,
                                largest_scalar,
                            }.into());
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
