//! Memory operation validation rules.
//!
//! This module validates SPIR-V memory operations including:
//!
//! - OpVariable: Variable declaration validation
//! - OpLoad: Load instruction validation
//! - OpStore: Store instruction validation
//! - OpAccessChain/OpInBoundsAccessChain: Access chain validation
//! - OpPtrAccessChain/OpInBoundsPtrAccessChain: Pointer access chain validation
//! - OpCopyMemory/OpCopyMemorySized: Memory copy validation
//! - OpArrayLength: Array length validation
//! - OpPtrEqual/OpPtrNotEqual/OpPtrDiff: Pointer comparison validation

use std::collections::HashMap;

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Decoration, MemoryAccess, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};

/// Helper to convert a u32 to Id, returning a default Id(1) for zero values.
/// This is used for error reporting where we need an Id even if the actual
/// instruction had an invalid zero id.
fn id_from_u32(value: u32) -> Id {
    Id::try_from(value).unwrap_or_else(|_| Id::try_from(1).unwrap())
}

/// Helper to convert a u32 to TypeId, returning a default TypeId(1) for zero values.
fn type_id_from_u32(value: u32) -> TypeId {
    TypeId::try_from(value).unwrap_or_else(|_| TypeId::try_from(1).unwrap())
}

/// Helper to convert a u32 to ResultId, returning a default ResultId(1) for zero values.
fn result_id_from_u32(value: u32) -> ResultId {
    ResultId::try_from(value).unwrap_or_else(|_| ResultId::try_from(1).unwrap())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Gets the storage class from a pointer type instruction.
fn get_pointer_storage_class(
    type_inst: &Instruction,
) -> Option<StorageClass> {
    if type_inst.class.opcode != Op::TypePointer && type_inst.class.opcode != Op::TypeUntypedPointerKHR {
        return None;
    }
    match type_inst.operands.first() {
        Some(Operand::StorageClass(sc)) => Some(*sc),
        _ => None,
    }
}

/// Gets the pointee type ID from a pointer type instruction.
fn get_pointee_type(type_inst: &Instruction) -> Option<u32> {
    if type_inst.class.opcode != Op::TypePointer {
        return None;
    }
    match type_inst.operands.get(1) {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

/// Checks if a type contains OpTypeBool anywhere in it (recursively).
fn contains_bool(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
    visited: &mut std::collections::HashSet<u32>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }

    let Some(result_id) = ResultId::try_from(type_id).ok() else {
        return false;
    };
    let Some(inst) = definitions.get(&result_id) else {
        return false;
    };

    match inst.class.opcode {
        Op::TypeBool => true,
        Op::TypeVector | Op::TypeMatrix | Op::TypeArray | Op::TypeRuntimeArray => {
            if let Some(Operand::IdRef(elem_id)) = inst.operands.first() {
                contains_bool(*elem_id, definitions, visited)
            } else {
                false
            }
        }
        Op::TypeStruct => {
            for op in &inst.operands {
                if let Operand::IdRef(member_id) = op {
                    if contains_bool(*member_id, definitions, visited) {
                        return true;
                    }
                }
            }
            false
        }
        Op::TypePointer => {
            if let Some(Operand::IdRef(pointee_id)) = inst.operands.get(1) {
                contains_bool(*pointee_id, definitions, visited)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Checks if a type contains a runtime array anywhere in it.
fn contains_runtime_array(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
    visited: &mut std::collections::HashSet<u32>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }

    let Some(result_id) = ResultId::try_from(type_id).ok() else {
        return false;
    };
    let Some(inst) = definitions.get(&result_id) else {
        return false;
    };

    match inst.class.opcode {
        Op::TypeRuntimeArray => true,
        Op::TypeArray => {
            if let Some(Operand::IdRef(elem_id)) = inst.operands.first() {
                contains_runtime_array(*elem_id, definitions, visited)
            } else {
                false
            }
        }
        Op::TypeStruct => {
            for op in &inst.operands {
                if let Operand::IdRef(member_id) = op {
                    if contains_runtime_array(*member_id, definitions, visited) {
                        return true;
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Checks if the instruction produces a logical pointer.
fn is_logical_pointer_producer(opcode: Op) -> bool {
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
            | Op::FunctionCall
            | Op::PtrAccessChain
            | Op::InBoundsPtrAccessChain
            | Op::Load
            | Op::ConstantNull
            | Op::Bitcast
            | Op::UntypedVariableKHR
            | Op::UntypedAccessChainKHR
            | Op::UntypedInBoundsAccessChainKHR
            | Op::UntypedPtrAccessChainKHR
            | Op::UntypedInBoundsPtrAccessChainKHR
    )
}

/// Checks if a storage class is read-only.
fn is_readonly_storage_class(sc: StorageClass) -> bool {
    matches!(
        sc,
        StorageClass::UniformConstant | StorageClass::Input | StorageClass::PushConstant
    )
}

/// Checks if the storage class allows NonPrivatePointer memory access.
fn allows_non_private_pointer(sc: StorageClass) -> bool {
    matches!(
        sc,
        StorageClass::Uniform
            | StorageClass::Workgroup
            | StorageClass::CrossWorkgroup
            | StorageClass::Generic
            | StorageClass::Image
            | StorageClass::StorageBuffer
            | StorageClass::PhysicalStorageBuffer
    )
}

/// Check if the instruction has a specific decoration.
fn has_decoration(module: &Module, id: u32, dec: Decoration) -> bool {
    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate {
            if let (Some(Operand::IdRef(target)), Some(Operand::Decoration(d))) =
                (inst.operands.first(), inst.operands.get(1))
            {
                if *target == id && *d == dec {
                    return true;
                }
            }
        }
    }
    false
}

// ============================================================================
// Variable Validation Rule
// ============================================================================

/// Validates OpVariable instructions.
pub struct VariableRule;

impl ValidationRule for VariableRule {
    fn name(&self) -> &'static str {
        "variable"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::Variable {
                continue;
            }

            // Get the result type (must be a pointer type)
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
                return Err(ValidationError::VariableResultTypeNotPointer {
                    variable: id_from_u32(inst.result_id.unwrap_or(0)),
                });
            }

            // Get storage class from operand
            let Some(Operand::StorageClass(storage_class)) = inst.operands.first() else {
                continue;
            };

            // Get storage class from result type
            let result_sc = get_pointer_storage_class(result_type);
            if let Some(rsc) = result_sc {
                if rsc != *storage_class {
                    return Err(ValidationError::VariableStorageClassMismatch {
                        variable: id_from_u32(inst.result_id.unwrap_or(0)),
                        operand_class: *storage_class,
                        type_class: rsc,
                    });
                }
            }

            // Generic storage class is not allowed for variables
            if *storage_class == StorageClass::Generic {
                return Err(ValidationError::VariableGenericStorageClass {
                    variable: id_from_u32(inst.result_id.unwrap_or(0)),
                });
            }

            // PhysicalStorageBuffer is not allowed for OpVariable
            if *storage_class == StorageClass::PhysicalStorageBuffer {
                return Err(ValidationError::VariablePhysicalStorageBuffer {
                    variable: id_from_u32(inst.result_id.unwrap_or(0)),
                });
            }

            // Get pointee type
            let pointee_type_id = get_pointee_type(result_type);
            if let Some(pt_id) = pointee_type_id {
                // Check for bool in non-allowed storage classes
                let allows_bool = matches!(
                    *storage_class,
                    StorageClass::Workgroup
                        | StorageClass::CrossWorkgroup
                        | StorageClass::Private
                        | StorageClass::Function
                        | StorageClass::UniformConstant
                        | StorageClass::RayPayloadKHR
                        | StorageClass::IncomingRayPayloadKHR
                        | StorageClass::HitAttributeKHR
                        | StorageClass::CallableDataKHR
                        | StorageClass::IncomingCallableDataKHR
                        | StorageClass::Input
                        | StorageClass::Output
                );

                if !allows_bool && contains_bool(pt_id, ctx.definitions, &mut std::collections::HashSet::new()) {
                    // Input/Output with BuiltIn is allowed
                    let is_builtin = inst.result_id.map_or(false, |id| {
                        has_decoration(ctx.module, id, Decoration::BuiltIn)
                    });

                    if !is_builtin {
                        return Err(ValidationError::VariableContainsBool {
                            variable: id_from_u32(inst.result_id.unwrap_or(0)),
                            storage_class: *storage_class,
                        });
                    }
                }
            }

            // Check initializer validity
            if inst.operands.len() > 1 {
                let Some(Operand::IdRef(init_id)) = inst.operands.get(1) else {
                    continue;
                };

                let Some(init_rid) = ResultId::try_from(*init_id).ok() else {
                    continue;
                };
                let Some(init_inst) = ctx.definitions.get(&init_rid) else {
                    return Err(ValidationError::VariableInitializerNotFound {
                        variable: id_from_u32(inst.result_id.unwrap_or(0)),
                        initializer: id_from_u32(*init_id),
                    });
                };

                // Initializer must be a constant or module-scope variable
                let is_constant = matches!(
                    init_inst.class.opcode,
                    Op::Constant
                        | Op::ConstantNull
                        | Op::ConstantTrue
                        | Op::ConstantFalse
                        | Op::ConstantComposite
                        | Op::ConstantSampler
                        | Op::SpecConstant
                        | Op::SpecConstantTrue
                        | Op::SpecConstantFalse
                        | Op::SpecConstantComposite
                        | Op::SpecConstantOp
                        | Op::Undef
                );

                let is_module_scope_var = init_inst.class.opcode == Op::Variable
                    && init_inst.operands.first().map_or(false, |op| {
                        matches!(op, Operand::StorageClass(sc) if *sc != StorageClass::Function)
                    });

                if !is_constant && !is_module_scope_var {
                    return Err(ValidationError::VariableInitializerNotConstant {
                        variable: id_from_u32(inst.result_id.unwrap_or(0)),
                        initializer: id_from_u32(*init_id),
                    });
                }

                // Input storage class cannot have initializer
                if *storage_class == StorageClass::Input {
                    return Err(ValidationError::VariableInputHasInitializer {
                        variable: id_from_u32(inst.result_id.unwrap_or(0)),
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Load Validation Rule
// ============================================================================

/// Validates OpLoad instructions.
pub struct LoadRule;

impl ValidationRule for LoadRule {
    fn name(&self) -> &'static str {
        "load"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                });
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
                });
            }

            // For typed pointers, check pointee type matches result type
            if pointer_type.class.opcode == Op::TypePointer {
                if let Some(pointee_id) = get_pointee_type(pointer_type) {
                    if pointee_id != result_type_id {
                        return Err(ValidationError::LoadResultTypeMismatch {
                            result_type: type_id_from_u32(result_type_id),
                            pointee_type: type_id_from_u32(pointee_id),
                        });
                    }
                }
            }

            // Cannot load a runtime array
            if contains_runtime_array(result_type_id, ctx.definitions, &mut std::collections::HashSet::new()) {
                return Err(ValidationError::LoadRuntimeArray);
            }

            // Check memory access operands if present
            if let Some(Operand::MemoryAccess(access)) = inst.operands.get(1) {
                validate_memory_access_for_load(ctx, *pointer_id, *access)?;
            }
        }

        Ok(())
    }
}

/// Validates memory access flags for load operations.
fn validate_memory_access_for_load(
    ctx: &ValidationContext<'_>,
    pointer_id: u32,
    access: MemoryAccess,
) -> Result<(), ValidationError> {
    // MakePointerAvailable cannot be used with OpLoad
    if access.contains(MemoryAccess::MAKE_POINTER_AVAILABLE) {
        return Err(ValidationError::LoadMakePointerAvailable);
    }

    // Check NonPrivatePointer storage class requirements
    if access.contains(MemoryAccess::NON_PRIVATE_POINTER) {
        if let Some(sc) = get_pointer_storage_class_for_value(ctx, pointer_id) {
            if !allows_non_private_pointer(sc) {
                return Err(ValidationError::NonPrivatePointerInvalidStorageClass {
                    storage_class: sc,
                });
            }
        }
    }

    // MakePointerVisible requires NonPrivatePointer
    if access.contains(MemoryAccess::MAKE_POINTER_VISIBLE)
        && !access.contains(MemoryAccess::NON_PRIVATE_POINTER)
    {
        return Err(ValidationError::MakeVisibleRequiresNonPrivate);
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::Store {
                continue;
            }

            // Get pointer operand
            let Some(Operand::IdRef(pointer_id)) = inst.operands.first() else {
                continue;
            };

            // Get object operand
            let Some(Operand::IdRef(object_id)) = inst.operands.get(1) else {
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
                });
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
                });
            }

            // Check storage class is not read-only
            if let Some(sc) = get_pointer_storage_class(pointer_type) {
                if is_readonly_storage_class(sc) {
                    return Err(ValidationError::StoreToReadOnlyStorageClass {
                        pointer: id_from_u32(*pointer_id),
                        storage_class: sc,
                    });
                }

                // ShaderRecordBufferKHR is also read-only
                if sc == StorageClass::ShaderRecordBufferKHR {
                    return Err(ValidationError::StoreToReadOnlyStorageClass {
                        pointer: id_from_u32(*pointer_id),
                        storage_class: sc,
                    });
                }
            }

            // Get object type
            let Some(object_rid) = ResultId::try_from(*object_id).ok() else {
                continue;
            };
            let Some(object_inst) = ctx.definitions.get(&object_rid) else {
                continue;
            };
            let Some(object_type_id) = object_inst.result_type else {
                continue;
            };

            // For typed pointers, check pointee type matches object type
            if pointer_type.class.opcode == Op::TypePointer {
                if let Some(pointee_id) = get_pointee_type(pointer_type) {
                    if pointee_id != object_type_id {
                        return Err(ValidationError::StoreTypeMismatch {
                            pointer: result_id_from_u32(*pointer_id),
                            pointer_type: type_id_from_u32(pointee_id),
                            object_type: type_id_from_u32(object_type_id),
                        });
                    }
                }
            }

            // Check memory access operands if present
            if let Some(Operand::MemoryAccess(access)) = inst.operands.get(2) {
                validate_memory_access_for_store(ctx, *pointer_id, *access)?;
            }
        }

        Ok(())
    }
}

/// Validates memory access flags for store operations.
fn validate_memory_access_for_store(
    ctx: &ValidationContext<'_>,
    pointer_id: u32,
    access: MemoryAccess,
) -> Result<(), ValidationError> {
    // MakePointerVisible cannot be used with OpStore
    if access.contains(MemoryAccess::MAKE_POINTER_VISIBLE) {
        return Err(ValidationError::StoreMakePointerVisible);
    }

    // Check NonPrivatePointer storage class requirements
    if access.contains(MemoryAccess::NON_PRIVATE_POINTER) {
        if let Some(sc) = get_pointer_storage_class_for_value(ctx, pointer_id) {
            if !allows_non_private_pointer(sc) {
                return Err(ValidationError::NonPrivatePointerInvalidStorageClass {
                    storage_class: sc,
                });
            }
        }
    }

    // MakePointerAvailable requires NonPrivatePointer
    if access.contains(MemoryAccess::MAKE_POINTER_AVAILABLE)
        && !access.contains(MemoryAccess::NON_PRIVATE_POINTER)
    {
        return Err(ValidationError::MakeAvailableRequiresNonPrivate);
    }

    Ok(())
}

// ============================================================================
// Access Chain Validation Rule
// ============================================================================

/// Validates OpAccessChain and OpInBoundsAccessChain instructions.
pub struct AccessChainRule;

impl ValidationRule for AccessChainRule {
    fn name(&self) -> &'static str {
        "access-chain"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        Op::AccessChain | Op::InBoundsAccessChain
                    ) {
                        continue;
                    }

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
                        });
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
                        });
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
                            });
                        }
                    }

                    // Validate indices
                    let base_pointee_id = get_pointee_type(base_type);
                    if let Some(mut current_type_id) = base_pointee_id {
                        for (idx, operand) in inst.operands.iter().skip(1).enumerate() {
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

                            if index_type.class.opcode != Op::TypeInt {
                                return Err(ValidationError::AccessChainIndexTypeInvalid {
                                    function: func_id,
                                    block: block_id,
                                    instruction: opcode,
                                    operand_index: idx,
                                    found: type_id_from_u32(index_type_id),
                                });
                            }

                            // Get current type instruction
                            let Some(current_type_rid) =
                                ResultId::try_from(current_type_id).ok()
                            else {
                                break;
                            };
                            let Some(current_type) = ctx.definitions.get(&current_type_rid)
                            else {
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
                                            },
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
                                                },
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
                                    });
                                }
                            }
                        }

                        // Check final type matches result pointee type
                        if let Some(result_pointee_id) = get_pointee_type(result_type) {
                            if inst.operands.len() > 1 && current_type_id != result_pointee_id {
                                return Err(ValidationError::AccessChainResultTypeMismatch {
                                    function: func_id,
                                    block: block_id,
                                    instruction: opcode,
                                    expected: type_id_from_u32(current_type_id),
                                    found: type_id_from_u32(result_pointee_id),
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
// All memory rules
// ============================================================================

/// Returns all memory validation rules.
pub fn all_memory_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &MemoryModelRule,
        &VariableRule,
        &LoadRule,
        &StoreRule,
        &AccessChainRule,
        &ArrayLengthRule,
        &CopyMemoryRule,
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::context::TestContextData;

    #[test]
    fn test_memory_model_present() {
        let mut data = TestContextData::default();
        data.module.memory_model = Some(rspirv::dr::Instruction::new(
            Op::MemoryModel,
            None,
            None,
            vec![
                Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical),
                Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450),
            ],
        ));
        let ctx = data.as_context();
        assert!(MemoryModelRule.validate(&ctx).is_ok());
    }

    #[test]
    fn test_memory_model_missing() {
        let data = TestContextData::default();
        let ctx = data.as_context();
        assert!(matches!(
            MemoryModelRule.validate(&ctx),
            Err(ValidationError::MissingMemoryModel)
        ));
    }

    #[test]
    fn test_is_logical_pointer_producer() {
        assert!(is_logical_pointer_producer(Op::Variable));
        assert!(is_logical_pointer_producer(Op::AccessChain));
        assert!(is_logical_pointer_producer(Op::FunctionParameter));
        assert!(!is_logical_pointer_producer(Op::IAdd));
        assert!(!is_logical_pointer_producer(Op::FAdd));
    }

    #[test]
    fn test_is_readonly_storage_class() {
        assert!(is_readonly_storage_class(StorageClass::UniformConstant));
        assert!(is_readonly_storage_class(StorageClass::Input));
        assert!(is_readonly_storage_class(StorageClass::PushConstant));
        assert!(!is_readonly_storage_class(StorageClass::Private));
        assert!(!is_readonly_storage_class(StorageClass::Function));
    }

    #[test]
    fn test_allows_non_private_pointer() {
        assert!(allows_non_private_pointer(StorageClass::Uniform));
        assert!(allows_non_private_pointer(StorageClass::Workgroup));
        assert!(allows_non_private_pointer(StorageClass::StorageBuffer));
        assert!(!allows_non_private_pointer(StorageClass::Private));
        assert!(!allows_non_private_pointer(StorageClass::Function));
    }
}
