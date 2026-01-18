//! Miscellaneous memory operation validation rules.
//!
//! This module validates miscellaneous SPIR-V memory operations including:
//!
//! - OpArrayLength: Array length validation
//! - OpCopyMemory/OpCopyMemorySized: Memory copy validation
//! - OpMemoryModel: Memory model validation
//! - OpPtrEqual/OpPtrNotEqual/OpPtrDiff: Pointer comparison validation

use rspirv::dr::Operand;
use rspirv::spirv::{AddressingModel, Capability, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId};

use super::helpers::{
    get_pointer_storage_class, get_pointee_type, id_from_u32, result_id_from_u32,
    type_id_from_u32,
};

// ============================================================================
// Array Length Validation Rule
// ============================================================================

/// Validates OpArrayLength instructions.
pub struct ArrayLengthRule;

impl ValidationRule for ArrayLengthRule {
    fn name(&self) -> &'static str {
        "array-length"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::ArrayLength {
                continue;
            }

            // Result type must be 32-bit or 64-bit unsigned integer
            let Some(result_type_id) = inst.result_type else {
                continue;
            };
            let Some(result_type_rid) = ResultId::try_from(result_type_id).ok() else {
                continue;
            };
            let Some(result_type) = ctx.definitions.get(&result_type_rid) else {
                continue;
            };

            if result_type.class.opcode != Op::TypeInt {
                return Err(ValidationError::ArrayLengthResultTypeNotInt {
                    instruction: id_from_u32(inst.result_id.unwrap_or(0)),
                });
            }

            // Check width is 32 or 64 and signedness is 0
            let width = result_type.operands.first().and_then(|op| match op {
                Operand::LiteralBit32(w) => Some(*w),
                _ => None,
            });
            let signedness = result_type.operands.get(1).and_then(|op| match op {
                Operand::LiteralBit32(s) => Some(*s),
                _ => None,
            });

            if !matches!(width, Some(32) | Some(64)) {
                return Err(ValidationError::ArrayLengthResultTypeInvalidWidth {
                    instruction: id_from_u32(inst.result_id.unwrap_or(0)),
                    width: width.unwrap_or(0),
                });
            }

            if signedness != Some(0) {
                return Err(ValidationError::ArrayLengthResultTypeSigned {
                    instruction: id_from_u32(inst.result_id.unwrap_or(0)),
                });
            }

            // Structure operand must be a pointer to a struct
            let Some(Operand::IdRef(structure_id)) = inst.operands.first() else {
                continue;
            };

            let Some(structure_rid) = ResultId::try_from(*structure_id).ok() else {
                continue;
            };
            let Some(structure_inst) = ctx.definitions.get(&structure_rid) else {
                continue;
            };
            let Some(structure_type_id) = structure_inst.result_type else {
                continue;
            };
            let Some(structure_type_rid) = ResultId::try_from(structure_type_id).ok() else {
                continue;
            };
            let Some(structure_type) = ctx.definitions.get(&structure_type_rid) else {
                continue;
            };

            if structure_type.class.opcode != Op::TypePointer {
                return Err(ValidationError::ArrayLengthStructureNotPointer {
                    instruction: id_from_u32(inst.result_id.unwrap_or(0)),
                });
            }

            // Pointee must be a struct
            let Some(pointee_id) = get_pointee_type(structure_type) else {
                continue;
            };
            let Some(pointee_rid) = ResultId::try_from(pointee_id).ok() else {
                continue;
            };
            let Some(pointee_type) = ctx.definitions.get(&pointee_rid) else {
                continue;
            };

            if pointee_type.class.opcode != Op::TypeStruct {
                return Err(ValidationError::ArrayLengthPointeeNotStruct {
                    instruction: id_from_u32(inst.result_id.unwrap_or(0)),
                });
            }

            // Array member index must be last member
            let Some(Operand::LiteralBit32(member_index)) = inst.operands.get(1) else {
                continue;
            };

            let num_members = pointee_type.operands.len();
            if *member_index as usize != num_members - 1 {
                return Err(ValidationError::ArrayLengthMemberNotLast {
                    instruction: id_from_u32(inst.result_id.unwrap_or(0)),
                    member_index: *member_index as usize,
                    last_member: num_members - 1,
                });
            }

            // Last member must be a runtime array
            if let Some(Operand::IdRef(last_member_type_id)) =
                pointee_type.operands.get(num_members - 1)
            {
                let Some(last_member_rid) = ResultId::try_from(*last_member_type_id).ok() else {
                    continue;
                };
                let Some(last_member_type) = ctx.definitions.get(&last_member_rid) else {
                    continue;
                };

                if last_member_type.class.opcode != Op::TypeRuntimeArray {
                    return Err(ValidationError::ArrayLengthMemberNotRuntimeArray {
                        instruction: id_from_u32(inst.result_id.unwrap_or(0)),
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Copy Memory Validation Rule
// ============================================================================

/// Validates OpCopyMemory and OpCopyMemorySized instructions.
pub struct CopyMemoryRule;

impl ValidationRule for CopyMemoryRule {
    fn name(&self) -> &'static str {
        "copy-memory"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            if !matches!(inst.class.opcode, Op::CopyMemory | Op::CopyMemorySized) {
                continue;
            }

            // Get target and source operands
            let Some(Operand::IdRef(target_id)) = inst.operands.first() else {
                continue;
            };
            let Some(Operand::IdRef(source_id)) = inst.operands.get(1) else {
                continue;
            };

            // Both must be pointers
            for (ptr_id, name) in [(*target_id, "target"), (*source_id, "source")] {
                let Some(ptr_rid) = ResultId::try_from(ptr_id).ok() else {
                    continue;
                };
                let Some(ptr_inst) = ctx.definitions.get(&ptr_rid) else {
                    continue;
                };
                let Some(ptr_type_id) = ptr_inst.result_type else {
                    continue;
                };
                let Some(ptr_type_rid) = ResultId::try_from(ptr_type_id).ok() else {
                    continue;
                };
                let Some(ptr_type) = ctx.definitions.get(&ptr_type_rid) else {
                    continue;
                };

                if ptr_type.class.opcode != Op::TypePointer
                    && ptr_type.class.opcode != Op::TypeUntypedPointerKHR
                {
                    return Err(ValidationError::CopyMemoryOperandNotPointer {
                        operand: id_from_u32(ptr_id),
                        operand_name: name,
                    });
                }
            }

            // For OpCopyMemory, check types match
            if inst.class.opcode == Op::CopyMemory {
                let target_pointee = get_pointee_type_for_value(ctx, *target_id);
                let source_pointee = get_pointee_type_for_value(ctx, *source_id);

                if let (Some(t), Some(s)) = (target_pointee, source_pointee) {
                    if t != s {
                        return Err(ValidationError::CopyMemoryTypeMismatch {
                            target_type: type_id_from_u32(t),
                            source_type: type_id_from_u32(s),
                        });
                    }
                }
            }

            // For OpCopyMemorySized, check size operand
            if inst.class.opcode == Op::CopyMemorySized {
                let Some(Operand::IdRef(size_id)) = inst.operands.get(2) else {
                    continue;
                };

                let Some(size_rid) = ResultId::try_from(*size_id).ok() else {
                    continue;
                };
                let Some(size_inst) = ctx.definitions.get(&size_rid) else {
                    continue;
                };
                let Some(size_type_id) = size_inst.result_type else {
                    continue;
                };
                let Some(size_type_rid) = ResultId::try_from(size_type_id).ok() else {
                    continue;
                };
                let Some(size_type) = ctx.definitions.get(&size_type_rid) else {
                    continue;
                };

                if size_type.class.opcode != Op::TypeInt {
                    return Err(ValidationError::CopyMemorySizeNotInteger {
                        size: id_from_u32(*size_id),
                    });
                }

                // Check for constant zero
                if size_inst.class.opcode == Op::ConstantNull {
                    return Err(ValidationError::CopyMemorySizeZero {
                        size: id_from_u32(*size_id),
                    });
                }

                if size_inst.class.opcode == Op::Constant {
                    let is_zero = size_inst.operands.iter().all(|op| {
                        matches!(op, Operand::LiteralBit32(0) | Operand::LiteralBit64(0))
                    });
                    if is_zero {
                        return Err(ValidationError::CopyMemorySizeZero {
                            size: id_from_u32(*size_id),
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

/// Gets the pointee type ID for a pointer value.
fn get_pointee_type_for_value(ctx: &ValidationContext<'_>, pointer_id: u32) -> Option<u32> {
    let result_id = ResultId::try_from(pointer_id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;
    let type_id = inst.result_type?;
    let type_rid = ResultId::try_from(type_id).ok()?;
    let type_inst = ctx.definitions.get(&type_rid)?;
    get_pointee_type(type_inst)
}

// ============================================================================
// Memory Model Rule
// ============================================================================

/// Validates that the module contains a memory model instruction.
pub struct MemoryModelRule;

impl ValidationRule for MemoryModelRule {
    fn name(&self) -> &'static str {
        "memory-model"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        if ctx.module.memory_model.is_none() {
            return Err(ValidationError::MissingMemoryModel);
        }
        Ok(())
    }
}

// ============================================================================
// Pointer Comparison Rule (PtrEqual, PtrNotEqual, PtrDiff)
// ============================================================================

/// Validates pointer comparison instructions.
pub struct PointerComparisonRule;

impl ValidationRule for PointerComparisonRule {
    fn name(&self) -> &'static str {
        "pointer-comparison"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let func_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(id_from_u32)
                .unwrap_or_else(|| id_from_u32(0));

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .map(id_from_u32)
                    .unwrap_or_else(|| id_from_u32(0));

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Only handle pointer comparison instructions
                    if !matches!(opcode, Op::PtrEqual | Op::PtrNotEqual | Op::PtrDiff) {
                        continue;
                    }

                    let result_type = inst.result_type.map(type_id_from_u32);

                    // Check result type
                    if let Some(result_type_id) = result_type {
                        let result_type_u32: u32 = result_type_id.into();
                        let result_type_def = ResultId::try_from(result_type_u32)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        match opcode {
                            Op::PtrEqual | Op::PtrNotEqual => {
                                // Result type must be bool
                                if let Some(type_inst) = result_type_def {
                                    if type_inst.class.opcode != Op::TypeBool {
                                        // For bool result type validation, we expect a TypeBool
                                        // Find the TypeBool in the module
                                        let expected_bool = ctx
                                            .module
                                            .types_global_values
                                            .iter()
                                            .find(|i| i.class.opcode == Op::TypeBool)
                                            .and_then(|i| i.result_id)
                                            .map(type_id_from_u32)
                                            .unwrap_or(result_type_id);

                                        return Err(
                                            ValidationError::InstructionResultTypeMismatch {
                                                function: func_id,
                                                block: block_id,
                                                instruction: opcode,
                                                expected: expected_bool,
                                                found: result_type_id,
                                            },
                                        );
                                    }
                                }
                            }
                            Op::PtrDiff => {
                                // Result type must be integer
                                if let Some(type_inst) = result_type_def {
                                    if type_inst.class.opcode != Op::TypeInt {
                                        // Find an integer type for expected
                                        let expected_int = ctx
                                            .module
                                            .types_global_values
                                            .iter()
                                            .find(|i| i.class.opcode == Op::TypeInt)
                                            .and_then(|i| i.result_id)
                                            .map(type_id_from_u32)
                                            .unwrap_or(result_type_id);

                                        return Err(
                                            ValidationError::InstructionResultTypeMismatch {
                                                function: func_id,
                                                block: block_id,
                                                instruction: opcode,
                                                expected: expected_int,
                                                found: result_type_id,
                                            },
                                        );
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    // Check operands
                    let operands: Vec<u32> = inst
                        .operands
                        .iter()
                        .filter_map(|op| {
                            if let Operand::IdRef(id) = op {
                                Some(*id)
                            } else {
                                None
                            }
                        })
                        .collect();

                    if operands.len() < 2 {
                        continue;
                    }

                    // Get operand types
                    let op1_id = operands[0];
                    let op2_id = operands[1];

                    let op1_type = ctx.result_types.get(&result_id_from_u32(op1_id));
                    let op2_type = ctx.result_types.get(&result_id_from_u32(op2_id));

                    // Both operands must be pointers
                    if let (Some(&op1_type_id), Some(&op2_type_id)) = (op1_type, op2_type) {
                        let op1_type_u32: u32 = op1_type_id.into();
                        let op2_type_u32: u32 = op2_type_id.into();
                        let op1_type_def = ResultId::try_from(op1_type_u32)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));
                        let op2_type_def = ResultId::try_from(op2_type_u32)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid));

                        // Check if first operand is a pointer
                        let op1_is_pointer = op1_type_def
                            .map(|i| {
                                matches!(
                                    i.class.opcode,
                                    Op::TypePointer | Op::TypeUntypedPointerKHR
                                )
                            })
                            .unwrap_or(false);

                        if !op1_is_pointer {
                            return Err(ValidationError::PointerComparisonOperandNotPointer {
                                function: func_id,
                                block: block_id,
                                instruction: opcode,
                                operand_index: 0,
                                found: op1_type_id,
                            });
                        }

                        // Check if second operand is a pointer
                        let op2_is_pointer = op2_type_def
                            .map(|i| {
                                matches!(
                                    i.class.opcode,
                                    Op::TypePointer | Op::TypeUntypedPointerKHR
                                )
                            })
                            .unwrap_or(false);

                        if !op2_is_pointer {
                            return Err(ValidationError::PointerComparisonOperandNotPointer {
                                function: func_id,
                                block: block_id,
                                instruction: opcode,
                                operand_index: 1,
                                found: op2_type_id,
                            });
                        }

                        // Check that operand types match
                        if op1_type_id != op2_type_id {
                            // For untyped pointers, check storage classes match
                            let op1_storage = op1_type_def.and_then(get_pointer_storage_class);
                            let op2_storage = op2_type_def.and_then(get_pointer_storage_class);

                            let op1_is_untyped = op1_type_def
                                .map(|i| i.class.opcode == Op::TypeUntypedPointerKHR)
                                .unwrap_or(false);
                            let op2_is_untyped = op2_type_def
                                .map(|i| i.class.opcode == Op::TypeUntypedPointerKHR)
                                .unwrap_or(false);

                            if op1_is_untyped && op2_is_untyped {
                                // For untyped pointers, storage classes must match
                                if op1_storage != op2_storage {
                                    return Err(ValidationError::OperandTypeMismatch {
                                        function: func_id,
                                        block: block_id,
                                        instruction: opcode,
                                        operand_index: 1,
                                        expected: op1_type_id,
                                        found: op2_type_id,
                                    });
                                }
                            } else {
                                // For typed pointers, types must match exactly
                                return Err(ValidationError::OperandTypeMismatch {
                                    function: func_id,
                                    block: block_id,
                                    instruction: opcode,
                                    operand_index: 1,
                                    expected: op1_type_id,
                                    found: op2_type_id,
                                });
                            }
                        }

                        // Check capability requirements based on storage class
                        let storage_class = op1_type_def.and_then(get_pointer_storage_class);
                        if let Some(sc) = storage_class {
                            self.check_storage_class_capability(
                                ctx, func_id, block_id, opcode, sc,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl PointerComparisonRule {
    /// Check capability requirements for pointer comparisons based on storage class.
    fn check_storage_class_capability(
        &self,
        ctx: &ValidationContext<'_>,
        func_id: Id,
        block_id: Id,
        opcode: Op,
        storage_class: StorageClass,
    ) -> Result<(), ValidationError> {
        // Check if we're in a physical addressing model by checking the memory model
        let is_physical_addressing = ctx
            .module
            .memory_model
            .as_ref()
            .map(|mm| {
                mm.operands
                    .first()
                    .map(|op| {
                        matches!(
                            op,
                            Operand::AddressingModel(
                                AddressingModel::Physical32
                                    | AddressingModel::Physical64
                                    | AddressingModel::PhysicalStorageBuffer64
                            )
                        )
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        let has_physical_storage_buffer =
            ctx.has_capability(Capability::PhysicalStorageBufferAddresses);

        if is_physical_addressing {
            // Physical addressing models allow all pointer comparisons
            return Ok(());
        }

        // For logical addressing with pointer comparison support
        // VariablePointers/VariablePointersStorageBuffer only allow specific storage classes
        match storage_class {
            StorageClass::StorageBuffer => {
                // Requires VariablePointersStorageBuffer or VariablePointers
                if !ctx.has_capability(Capability::VariablePointersStorageBuffer)
                    && !ctx.has_capability(Capability::VariablePointers)
                {
                    return Err(ValidationError::PointerComparisonMissingCapability {
                        function: func_id,
                        block: block_id,
                        instruction: opcode,
                        storage_class,
                        required_capability: Capability::VariablePointersStorageBuffer,
                    });
                }
            }
            StorageClass::Workgroup => {
                // Requires VariablePointers
                if !ctx.has_capability(Capability::VariablePointers) {
                    return Err(ValidationError::PointerComparisonMissingCapability {
                        function: func_id,
                        block: block_id,
                        instruction: opcode,
                        storage_class,
                        required_capability: Capability::VariablePointers,
                    });
                }
            }
            StorageClass::PhysicalStorageBuffer => {
                // PhysicalStorageBuffer with PhysicalStorageBufferAddresses is allowed
                if !has_physical_storage_buffer {
                    return Err(ValidationError::PointerComparisonMissingCapability {
                        function: func_id,
                        block: block_id,
                        instruction: opcode,
                        storage_class,
                        required_capability: Capability::PhysicalStorageBufferAddresses,
                    });
                }
            }
            StorageClass::Function | StorageClass::Private => {
                // Function and Private storage classes are NOT allowed in logical addressing
                // even with VariablePointers capability (per SPIR-V spec)
                return Err(ValidationError::PointerComparisonInvalidStorageClass {
                    function: func_id,
                    block: block_id,
                    instruction: opcode,
                    storage_class,
                });
            }
            _ => {
                // Other storage classes are not allowed for pointer comparisons
                return Err(ValidationError::PointerComparisonInvalidStorageClass {
                    function: func_id,
                    block: block_id,
                    instruction: opcode,
                    storage_class,
                });
            }
        }

        Ok(())
    }
}
