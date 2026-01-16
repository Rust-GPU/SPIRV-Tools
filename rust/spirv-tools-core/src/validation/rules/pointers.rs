//! Pointer validation rules.
//!
//! This module validates SPIR-V pointer requirements including:
//!
//! - Logical pointer storage class restrictions
//! - Store type compatibility
//! - Variable pointer constraints (matrix, block array, same-buffer)

use std::collections::{HashMap, HashSet};

use rspirv::dr::{Module, Operand};
use rspirv::spirv::{AddressingModel, Capability, Decoration, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};

// ============================================================================
// Logical Pointer Rule
// ============================================================================

/// Validates logical pointer storage class restrictions.
pub struct LogicalPointerRule;

impl ValidationRule for LogicalPointerRule {
    fn name(&self) -> &'static str {
        "logical-pointers"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        if ctx.options.relax_logical_pointer {
            return Ok(());
        }

        let addressing_model = ctx
            .module
            .memory_model
            .as_ref()
            .and_then(|inst| inst.operands.first())
            .and_then(|op| match op {
                rspirv::dr::Operand::AddressingModel(model) => Some(*model),
                _ => None,
            });

        let is_logical = matches!(
            addressing_model,
            Some(AddressingModel::Logical | AddressingModel::PhysicalStorageBuffer64)
        );
        if !is_logical {
            return Ok(());
        }

        for inst in ctx
            .module
            .types_global_values
            .iter()
            .chain(ctx.module.functions.iter().flat_map(|f| f.all_inst_iter()))
        {
            if inst.class.opcode != Op::Variable {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            let Ok(type_id) = TypeId::try_from(result_type) else {
                continue;
            };
            let Some(type_result_id) = ResultId::try_from(u32::from(type_id)).ok() else {
                continue;
            };
            let Some(type_inst) = ctx.definitions.get(&type_result_id) else {
                continue;
            };
            if type_inst.class.opcode != Op::TypePointer
                && type_inst.class.opcode != Op::TypeUntypedPointerKHR
            {
                continue;
            }
            let pointee_type_id = match type_inst.operands.get(1) {
                Some(rspirv::dr::Operand::IdRef(raw)) => ResultId::try_from(*raw).ok(),
                _ => None,
            };
            let Some(pointee_inst) = pointee_type_id.and_then(|id| ctx.definitions.get(&id)) else {
                continue;
            };
            if pointee_inst.class.opcode != Op::TypePointer
                && pointee_inst.class.opcode != Op::TypeUntypedPointerKHR
            {
                continue;
            }
            let pointee_storage_class = match pointee_inst.operands.first() {
                Some(rspirv::dr::Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };
            if pointee_storage_class == StorageClass::PhysicalStorageBuffer {
                continue;
            }

            let variable = inst
                .result_id
                .and_then(|id| Id::try_from(id).ok())
                .unwrap_or_else(|| Id::try_from(1).unwrap());

            match pointee_storage_class {
                StorageClass::StorageBuffer => {
                    if !ctx
                        .declared_capabilities
                        .contains(&Capability::VariablePointersStorageBuffer)
                    {
                        return Err(ValidationError::LogicalPointerMissingCapability {
                            variable,
                            pointee_storage_class,
                            required_capability: Capability::VariablePointersStorageBuffer,
                        });
                    }
                }
                StorageClass::Workgroup => {
                    if !ctx
                        .declared_capabilities
                        .contains(&Capability::VariablePointers)
                    {
                        return Err(ValidationError::LogicalPointerMissingCapability {
                            variable,
                            pointee_storage_class,
                            required_capability: Capability::VariablePointers,
                        });
                    }
                }
                _ => {
                    return Err(ValidationError::LogicalPointerPointeeStorageClassInvalid {
                        variable,
                        pointee_storage_class,
                    });
                }
            }

            let var_storage_class = inst
                .operands
                .first()
                .and_then(|op| match op {
                    rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                    _ => None,
                })
                .unwrap_or(StorageClass::Function);
            if var_storage_class != StorageClass::Function
                && var_storage_class != StorageClass::Private
            {
                return Err(ValidationError::LogicalPointerInvalidStorageClass {
                    variable,
                    storage_class: var_storage_class,
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Load/Store Logical Pointer Rule
// ============================================================================

/// Returns true if the given opcode returns a logical pointer.
/// These are the only opcodes that can produce pointers valid for
/// OpLoad/OpStore in Logical addressing mode (without VariablePointers).
fn opcode_returns_logical_pointer(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Variable
            | Op::UntypedVariableKHR
            | Op::AccessChain
            | Op::InBoundsAccessChain
            | Op::UntypedAccessChainKHR
            | Op::UntypedInBoundsAccessChainKHR
            | Op::FunctionParameter
            | Op::ImageTexelPointer
            | Op::CopyObject
            | Op::RawAccessChainNV
            | Op::AllocateNodePayloadsAMDX
    )
}

/// Returns true if the given opcode returns a logical variable pointer.
/// These are the opcodes that can produce pointers valid for
/// OpLoad/OpStore in Logical addressing mode WITH VariablePointers capability.
fn opcode_returns_logical_variable_pointer(opcode: Op) -> bool {
    if opcode_returns_logical_pointer(opcode) {
        return true;
    }
    matches!(
        opcode,
        Op::PtrAccessChain
            | Op::UntypedPtrAccessChainKHR
            | Op::UntypedInBoundsPtrAccessChainKHR
            | Op::Load
            | Op::Select
            | Op::Phi
            | Op::FunctionCall
            | Op::ConstantNull
    )
}

/// Validates that OpLoad and OpStore use valid logical pointers.
/// In Logical addressing mode, the pointer operand must come from a
/// logical pointer-producing instruction.
pub struct LoadStoreLogicalPointerRule;

impl ValidationRule for LoadStoreLogicalPointerRule {
    fn name(&self) -> &'static str {
        "load-store-logical-pointers"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        if ctx.options.relax_logical_pointer {
            return Ok(());
        }

        let addressing_model = ctx
            .module
            .memory_model
            .as_ref()
            .and_then(|inst| inst.operands.first())
            .and_then(|op| match op {
                rspirv::dr::Operand::AddressingModel(model) => Some(*model),
                _ => None,
            });

        // Only validate in Logical addressing mode
        let is_logical = matches!(addressing_model, Some(AddressingModel::Logical));
        if !is_logical {
            return Ok(());
        }

        let has_variable_pointers =
            ctx.declared_capabilities
                .contains(&Capability::VariablePointers)
                || ctx
                    .declared_capabilities
                    .contains(&Capability::VariablePointersStorageBuffer);

        for inst in ctx.module.all_inst_iter() {
            let (opcode, ptr_operand_index) = match inst.class.opcode {
                Op::Load => (Op::Load, 0),
                Op::Store => (Op::Store, 0),
                Op::CopyMemory => (Op::CopyMemory, 0),
                Op::CopyMemorySized => (Op::CopyMemorySized, 0),
                _ => continue,
            };

            let Some(rspirv::dr::Operand::IdRef(ptr_id_raw)) = inst.operands.get(ptr_operand_index)
            else {
                continue;
            };
            let Ok(ptr_id) = ResultId::try_from(*ptr_id_raw) else {
                continue;
            };
            let Some(ptr_inst) = ctx.definitions.get(&ptr_id) else {
                continue;
            };

            // Check if the pointer type has PhysicalStorageBuffer storage class
            // If so, it's not subject to logical pointer restrictions
            if let Some(ptr_type_raw) = ptr_inst.result_type {
                if let Ok(ptr_type_id) = ResultId::try_from(ptr_type_raw) {
                    if let Some(ptr_type_inst) = ctx.definitions.get(&ptr_type_id) {
                        if ptr_type_inst.class.opcode == Op::TypePointer {
                            if let Some(rspirv::dr::Operand::StorageClass(sc)) =
                                ptr_type_inst.operands.first()
                            {
                                if *sc == StorageClass::PhysicalStorageBuffer {
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            let source_opcode = ptr_inst.class.opcode;

            let is_valid = if has_variable_pointers {
                opcode_returns_logical_variable_pointer(source_opcode)
            } else {
                opcode_returns_logical_pointer(source_opcode)
            };

            if !is_valid {
                let pointer = Id::try_from(*ptr_id_raw).unwrap_or_else(|_| Id::try_from(1).unwrap());
                return Err(ValidationError::NotALogicalPointer {
                    instruction: opcode,
                    pointer,
                    source_opcode,
                });
            }

            // For CopyMemory/CopyMemorySized, also check the source pointer
            if opcode == Op::CopyMemory || opcode == Op::CopyMemorySized {
                let Some(rspirv::dr::Operand::IdRef(src_ptr_id_raw)) = inst.operands.get(1) else {
                    continue;
                };
                let Ok(src_ptr_id) = ResultId::try_from(*src_ptr_id_raw) else {
                    continue;
                };
                let Some(src_ptr_inst) = ctx.definitions.get(&src_ptr_id) else {
                    continue;
                };

                // Check PhysicalStorageBuffer for source pointer too
                if let Some(ptr_type_raw) = src_ptr_inst.result_type {
                    if let Ok(ptr_type_id) = ResultId::try_from(ptr_type_raw) {
                        if let Some(ptr_type_inst) = ctx.definitions.get(&ptr_type_id) {
                            if ptr_type_inst.class.opcode == Op::TypePointer {
                                if let Some(rspirv::dr::Operand::StorageClass(sc)) =
                                    ptr_type_inst.operands.first()
                                {
                                    if *sc == StorageClass::PhysicalStorageBuffer {
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                let src_source_opcode = src_ptr_inst.class.opcode;
                let src_is_valid = if has_variable_pointers {
                    opcode_returns_logical_variable_pointer(src_source_opcode)
                } else {
                    opcode_returns_logical_pointer(src_source_opcode)
                };

                if !src_is_valid {
                    let pointer =
                        Id::try_from(*src_ptr_id_raw).unwrap_or_else(|_| Id::try_from(1).unwrap());
                    return Err(ValidationError::NotALogicalPointer {
                        instruction: opcode,
                        pointer,
                        source_opcode: src_source_opcode,
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Store Type Compatibility Rule
// ============================================================================

/// Validates that OpStore pointer and object types are compatible.
pub struct StoreTypeCompatibilityRule;

impl ValidationRule for StoreTypeCompatibilityRule {
    fn name(&self) -> &'static str {
        "store-type-compatibility"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::Store {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(ptr_id_raw)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::IdRef(obj_id_raw)) = inst.operands.get(1) else {
                continue;
            };
            let Ok(ptr_id) = ResultId::try_from(*ptr_id_raw) else {
                continue;
            };
            let Some(ptr_inst) = ctx.definitions.get(&ptr_id) else {
                continue;
            };
            let Some(ptr_type_raw) = ptr_inst.result_type else {
                continue;
            };
            let Ok(ptr_type_id) = TypeId::try_from(ptr_type_raw) else {
                continue;
            };
            let Some(ptr_type_result) = ResultId::try_from(u32::from(ptr_type_id)).ok() else {
                continue;
            };
            let Some(ptr_type_inst) = ctx.definitions.get(&ptr_type_result) else {
                continue;
            };
            if ptr_type_inst.class.opcode != Op::TypePointer {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(pointee_raw)) = ptr_type_inst.operands.get(1)
            else {
                continue;
            };
            let Ok(pointee_id) = TypeId::try_from(*pointee_raw) else {
                continue;
            };

            let Ok(obj_id) = ResultId::try_from(*obj_id_raw) else {
                continue;
            };
            let Some(obj_inst) = ctx.definitions.get(&obj_id) else {
                continue;
            };
            let Some(obj_type_raw) = obj_inst.result_type else {
                continue;
            };
            let Ok(obj_type_id) = TypeId::try_from(obj_type_raw) else {
                continue;
            };

            if pointee_id == obj_type_id {
                continue;
            }

            if ctx.options.relax_struct_store {
                let layout_relaxed = ctx.options.relax_block_layout
                    || ctx.options.uniform_buffer_standard_layout
                    || ctx.options.scalar_block_layout
                    || ctx.options.workgroup_scalar_block_layout;
                if layout_relaxed {
                    continue;
                }
                if layout_compatible_types(
                    pointee_id,
                    obj_type_id,
                    ctx.module,
                    ctx.definitions,
                    &mut HashSet::new(),
                ) {
                    continue;
                }
            }

            return Err(ValidationError::StoreTypeMismatch {
                pointer: ptr_id,
                pointer_type: pointee_id,
                object_type: obj_type_id,
            });
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn layout_compatible_types(
    a: TypeId,
    b: TypeId,
    module: &Module,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
    visiting: &mut HashSet<TypeId>,
) -> bool {
    if a == b {
        return true;
    }
    if !visiting.insert(a) {
        return false;
    }
    let Some(result_a) = ResultId::try_from(u32::from(a)).ok() else {
        visiting.remove(&a);
        return false;
    };
    let Some(result_b) = ResultId::try_from(u32::from(b)).ok() else {
        visiting.remove(&a);
        return false;
    };
    let Some(inst_a) = definitions.get(&result_a) else {
        visiting.remove(&a);
        return false;
    };
    let Some(inst_b) = definitions.get(&result_b) else {
        visiting.remove(&a);
        return false;
    };
    let compatible = match (inst_a.class.opcode, inst_b.class.opcode) {
        (Op::TypeStruct, Op::TypeStruct) => {
            if inst_a.operands.len() != inst_b.operands.len() {
                false
            } else {
                inst_a
                    .operands
                    .iter()
                    .zip(&inst_b.operands)
                    .all(|(op_a, op_b)| match (op_a, op_b) {
                        (rspirv::dr::Operand::IdRef(id_a), rspirv::dr::Operand::IdRef(id_b)) => {
                            let Ok(type_a) = TypeId::try_from(*id_a) else {
                                return false;
                            };
                            let Ok(type_b) = TypeId::try_from(*id_b) else {
                                return false;
                            };
                            layout_compatible_types(type_a, type_b, module, definitions, visiting)
                        }
                        _ => false,
                    })
            }
        }
        (Op::TypeArray, Op::TypeArray) => inst_a
            .operands
            .first()
            .and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id_a) => TypeId::try_from(*id_a).ok(),
                _ => None,
            })
            .and_then(|elem_a| {
                let elem_b = inst_b.operands.first().and_then(|op| match op {
                    rspirv::dr::Operand::IdRef(id_b) => TypeId::try_from(*id_b).ok(),
                    _ => None,
                })?;
                let len_a = array_length(inst_a, definitions);
                let len_b = array_length(inst_b, definitions);
                Some((elem_a, elem_b, len_a, len_b))
            })
            .is_some_and(|(elem_a, elem_b, len_a, len_b)| {
                let stride_a = array_stride(module, result_a);
                let stride_b = array_stride(module, result_b);
                len_a == len_b
                    && stride_a == stride_b
                    && layout_compatible_types(elem_a, elem_b, module, definitions, visiting)
            }),
        (Op::TypeRuntimeArray, Op::TypeRuntimeArray) => inst_a
            .operands
            .first()
            .and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id_a) => TypeId::try_from(*id_a).ok(),
                _ => None,
            })
            .and_then(|elem_a| {
                inst_b
                    .operands
                    .first()
                    .and_then(|op| match op {
                        rspirv::dr::Operand::IdRef(id_b) => TypeId::try_from(*id_b).ok(),
                        _ => None,
                    })
                    .map(|elem_b| (elem_a, elem_b))
            })
            .is_some_and(|(elem_a, elem_b)| {
                layout_compatible_types(elem_a, elem_b, module, definitions, visiting)
            }),
        (Op::TypeVector, Op::TypeVector) => {
            let (elem_a, count_a) = vector_info(inst_a);
            let (elem_b, count_b) = vector_info(inst_b);
            elem_a
                .zip(elem_b)
                .zip(count_a.zip(count_b))
                .is_some_and(|((a, b), (ca, cb))| {
                    ca == cb && layout_compatible_types(a, b, module, definitions, visiting)
                })
        }
        (Op::TypeMatrix, Op::TypeMatrix) => {
            let (col_a, count_a) = matrix_info(inst_a);
            let (col_b, count_b) = matrix_info(inst_b);
            col_a
                .zip(col_b)
                .zip(count_a.zip(count_b))
                .is_some_and(|((a, b), (ca, cb))| {
                    ca == cb && layout_compatible_types(a, b, module, definitions, visiting)
                })
        }
        _ => false,
    };
    visiting.remove(&a);
    compatible
}

fn array_length(
    inst: &rspirv::dr::Instruction,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let len_id = match inst.operands.get(1) {
        Some(rspirv::dr::Operand::IdRef(id)) => ResultId::try_from(*id).ok()?,
        _ => return None,
    };
    let len_inst = definitions.get(&len_id)?;
    if len_inst.class.opcode != Op::Constant {
        return None;
    }
    match len_inst.operands.first() {
        Some(rspirv::dr::Operand::LiteralBit32(v)) => Some(*v),
        Some(rspirv::dr::Operand::LiteralBit64(v)) => u32::try_from(*v).ok(),
        _ => None,
    }
}

fn array_stride(module: &Module, array_type: ResultId) -> Option<u32> {
    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate {
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
                Some(rspirv::dr::Operand::LiteralBit32(stride)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            ) {
                if *decoration == rspirv::spirv::Decoration::ArrayStride {
                    if let Ok(target_id) = ResultId::try_from(*target) {
                        if target_id == array_type {
                            return Some(*stride);
                        }
                    }
                }
            }
        }
    }
    None
}

fn vector_info(inst: &rspirv::dr::Instruction) -> (Option<TypeId>, Option<u32>) {
    let elem = inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    });
    let count = inst.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(c) => Some(*c),
        _ => None,
    });
    (elem, count)
}

fn matrix_info(inst: &rspirv::dr::Instruction) -> (Option<TypeId>, Option<u32>) {
    let column = inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    });
    let count = inst.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(c) => Some(*c),
        _ => None,
    });
    (column, count)
}

// ============================================================================
// Variable Pointer Validation
// ============================================================================

/// Returns true if inst is a logical pointer (not PhysicalStorageBuffer).
fn is_logical_pointer(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    let Some(type_id) = inst.result_type else {
        return false;
    };
    let Ok(type_result_id) = ResultId::try_from(type_id) else {
        return false;
    };
    let Some(type_inst) = definitions.get(&type_result_id) else {
        return false;
    };

    match type_inst.class.opcode {
        Op::TypePointer => {
            if let Some(Operand::StorageClass(sc)) = type_inst.operands.first() {
                *sc != StorageClass::PhysicalStorageBuffer
            } else {
                false
            }
        }
        Op::TypeUntypedPointerKHR => {
            if let Some(Operand::StorageClass(sc)) = type_inst.operands.first() {
                *sc != StorageClass::PhysicalStorageBuffer
            } else {
                true
            }
        }
        _ => false,
    }
}

/// Returns the storage class of a pointer instruction's type.
fn get_pointer_storage_class(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<StorageClass> {
    let type_id = inst.result_type?;
    let type_result_id = ResultId::try_from(type_id).ok()?;
    let type_inst = definitions.get(&type_result_id)?;

    match type_inst.class.opcode {
        Op::TypePointer | Op::TypeUntypedPointerKHR => {
            if let Some(Operand::StorageClass(sc)) = type_inst.operands.first() {
                Some(*sc)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Returns true if the instruction is a variable pointer.
/// Variable pointers are pointers that can have multiple possible values at runtime.
fn is_variable_pointer(
    inst_id: ResultId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    cache: &mut HashMap<ResultId, bool>,
) -> bool {
    if let Some(&cached) = cache.get(&inst_id) {
        return cached;
    }

    let Some(inst) = definitions.get(&inst_id) else {
        return false;
    };

    if !is_logical_pointer(inst, definitions) {
        cache.insert(inst_id, false);
        return false;
    }

    let is_var_ptr = match inst.class.opcode {
        // These opcodes always produce variable pointers
        Op::PtrAccessChain
        | Op::UntypedPtrAccessChainKHR
        | Op::UntypedInBoundsPtrAccessChainKHR
        | Op::Load
        | Op::Select
        | Op::Phi
        | Op::FunctionCall
        | Op::ConstantNull => true,

        // Function parameters may be variable pointers depending on call sites
        Op::FunctionParameter => true,

        // For other instructions, check if any operand is a variable pointer
        _ => {
            let mut result = false;
            for operand in &inst.operands {
                if let Operand::IdRef(op_id) = operand {
                    if let Ok(op_result_id) = ResultId::try_from(*op_id) {
                        if let Some(op_inst) = definitions.get(&op_result_id) {
                            if is_logical_pointer(op_inst, definitions)
                                && is_variable_pointer(op_result_id, definitions, cache)
                            {
                                result = true;
                                break;
                            }
                        }
                    }
                }
            }
            result
        }
    };

    cache.insert(inst_id, is_var_ptr);
    is_var_ptr
}

/// Check if a type contains a matrix type.
fn type_contains_matrix(
    type_id: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visited: &mut HashSet<TypeId>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }

    let Ok(result_id) = ResultId::try_from(u32::from(type_id)) else {
        return false;
    };
    let Some(type_inst) = definitions.get(&result_id) else {
        return false;
    };

    match type_inst.class.opcode {
        Op::TypeMatrix => true,
        Op::TypeArray | Op::TypeRuntimeArray => {
            if let Some(Operand::IdRef(elem_id)) = type_inst.operands.first() {
                if let Ok(elem_type_id) = TypeId::try_from(*elem_id) {
                    return type_contains_matrix(elem_type_id, definitions, visited);
                }
            }
            false
        }
        Op::TypeStruct => {
            for operand in &type_inst.operands {
                if let Operand::IdRef(member_id) = operand {
                    if let Ok(member_type_id) = TypeId::try_from(*member_id) {
                        if type_contains_matrix(member_type_id, definitions, visited) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Check if an array type points to Block or BufferBlock decorated structs.
fn is_block_array(
    type_inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    module: &Module,
) -> bool {
    if type_inst.class.opcode != Op::TypeArray
        && type_inst.class.opcode != Op::TypeRuntimeArray
    {
        return false;
    }

    let Some(Operand::IdRef(elem_id)) = type_inst.operands.first() else {
        return false;
    };
    let Ok(elem_result_id) = ResultId::try_from(*elem_id) else {
        return false;
    };
    let Some(elem_inst) = definitions.get(&elem_result_id) else {
        return false;
    };

    if elem_inst.class.opcode != Op::TypeStruct {
        return false;
    }

    // Check if the struct has Block or BufferBlock decoration
    for annotation in &module.annotations {
        if annotation.class.opcode != Op::Decorate {
            continue;
        }
        if let (Some(Operand::IdRef(target)), Some(Operand::Decoration(dec))) =
            (annotation.operands.first(), annotation.operands.get(1))
        {
            if *target == *elem_id
                && (*dec == Decoration::Block || *dec == Decoration::BufferBlock)
            {
                return true;
            }
        }
    }

    false
}

/// Trace variable pointer back through access chains to check for matrix access.
fn traces_through_matrix(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visited: &mut HashSet<ResultId>,
) -> bool {
    let Some(inst_id) = inst.result_id.and_then(|id| ResultId::try_from(id).ok()) else {
        return false;
    };
    if !visited.insert(inst_id) {
        return false;
    }

    match inst.class.opcode {
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain => {
            // Get the base pointer operand
            let base_idx = if inst.class.opcode == Op::PtrAccessChain {
                2
            } else {
                2
            };

            // First check if base type is a matrix
            if let Some(Operand::IdRef(base_id)) = inst.operands.get(base_idx - 2) {
                if let Ok(base_result_id) = ResultId::try_from(*base_id) {
                    if let Some(base_inst) = definitions.get(&base_result_id) {
                        if let Some(base_type_raw) = base_inst.result_type {
                            if let Ok(base_type_id) = ResultId::try_from(base_type_raw) {
                                if let Some(base_type_inst) = definitions.get(&base_type_id) {
                                    if base_type_inst.class.opcode == Op::TypePointer {
                                        if let Some(Operand::IdRef(pointee_id)) =
                                            base_type_inst.operands.get(1)
                                        {
                                            if let Ok(pointee_result_id) =
                                                ResultId::try_from(*pointee_id)
                                            {
                                                if let Some(pointee_inst) =
                                                    definitions.get(&pointee_result_id)
                                                {
                                                    if pointee_inst.class.opcode == Op::TypeMatrix
                                                    {
                                                        return true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Recursively check base
                        if traces_through_matrix(base_inst, definitions, visited) {
                            return true;
                        }
                    }
                }
            }

            // Check if any index accesses a matrix
            let start_idx = if inst.class.opcode == Op::PtrAccessChain {
                3
            } else {
                3
            };
            let mut current_type: Option<ResultId> = None;

            // Get base pointee type
            if let Some(Operand::IdRef(base_id)) = inst.operands.first() {
                if let Ok(base_result_id) = ResultId::try_from(*base_id) {
                    if let Some(base_inst) = definitions.get(&base_result_id) {
                        if let Some(base_type_raw) = base_inst.result_type {
                            if let Ok(base_type_id) = ResultId::try_from(base_type_raw) {
                                if let Some(base_type_inst) = definitions.get(&base_type_id) {
                                    if base_type_inst.class.opcode == Op::TypePointer {
                                        if let Some(Operand::IdRef(pointee_id)) =
                                            base_type_inst.operands.get(1)
                                        {
                                            current_type = ResultId::try_from(*pointee_id).ok();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Walk through indices
            for i in (start_idx - 1)..inst.operands.len() {
                let Some(type_id) = current_type else {
                    break;
                };
                let Some(type_inst) = definitions.get(&type_id) else {
                    break;
                };

                if type_inst.class.opcode == Op::TypeMatrix {
                    return true;
                }

                // Get next type
                current_type = match type_inst.class.opcode {
                    Op::TypeStruct => {
                        // Need to get the constant index value
                        if let Some(Operand::IdRef(idx_id)) = inst.operands.get(i) {
                            if let Ok(idx_result_id) = ResultId::try_from(*idx_id) {
                                if let Some(idx_inst) = definitions.get(&idx_result_id) {
                                    if idx_inst.class.opcode == Op::Constant {
                                        if let Some(Operand::LiteralBit32(val)) =
                                            idx_inst.operands.first()
                                        {
                                            if let Some(Operand::IdRef(member_id)) =
                                                type_inst.operands.get(*val as usize)
                                            {
                                                ResultId::try_from(*member_id).ok()
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                        if let Some(Operand::IdRef(elem_id)) = type_inst.operands.first() {
                            ResultId::try_from(*elem_id).ok()
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
            }

            false
        }
        Op::Phi => {
            for i in (0..inst.operands.len()).step_by(2) {
                if let Some(Operand::IdRef(val_id)) = inst.operands.get(i) {
                    if let Ok(val_result_id) = ResultId::try_from(*val_id) {
                        if let Some(val_inst) = definitions.get(&val_result_id) {
                            if traces_through_matrix(val_inst, definitions, visited) {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        }
        Op::Select => {
            for i in [1, 2] {
                if let Some(Operand::IdRef(val_id)) = inst.operands.get(i) {
                    if let Ok(val_result_id) = ResultId::try_from(*val_id) {
                        if let Some(val_inst) = definitions.get(&val_result_id) {
                            if traces_through_matrix(val_inst, definitions, visited) {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        }
        Op::CopyObject => {
            if let Some(Operand::IdRef(src_id)) = inst.operands.first() {
                if let Ok(src_result_id) = ResultId::try_from(*src_id) {
                    if let Some(src_inst) = definitions.get(&src_result_id) {
                        return traces_through_matrix(src_inst, definitions, visited);
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Validates variable pointer constraints.
///
/// Variable pointers must not:
/// - Point to arrays of Block/BufferBlock decorated structs
/// - Point to objects containing matrices
/// - Point to columns or components of matrices
/// - (Without VariablePointers capability) be selected from different buffers
pub struct VariablePointerRule;

impl ValidationRule for VariablePointerRule {
    fn name(&self) -> &'static str {
        "variable-pointers"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        if ctx.options.relax_logical_pointer {
            return Ok(());
        }

        let addressing_model = ctx
            .module
            .memory_model
            .as_ref()
            .and_then(|inst| inst.operands.first())
            .and_then(|op| match op {
                Operand::AddressingModel(model) => Some(*model),
                _ => None,
            });

        // Only validate in Logical or PhysicalStorageBuffer64 addressing mode
        if !matches!(
            addressing_model,
            Some(AddressingModel::Logical | AddressingModel::PhysicalStorageBuffer64)
        ) {
            return Ok(());
        }

        let has_variable_pointers = ctx
            .declared_capabilities
            .contains(&Capability::VariablePointers);
        let has_variable_pointers_storage_buffer = ctx
            .declared_capabilities
            .contains(&Capability::VariablePointersStorageBuffer);

        // Build variable pointer cache
        let mut var_ptr_cache: HashMap<ResultId, bool> = HashMap::new();

        // First pass: identify all variable pointers
        for inst in ctx.module.all_inst_iter() {
            if let Some(result_id) = inst.result_id {
                if let Ok(id) = ResultId::try_from(result_id) {
                    if is_logical_pointer(inst, ctx.definitions) {
                        is_variable_pointer(id, ctx.definitions, &mut var_ptr_cache);
                    }
                }
            }
        }

        // Second pass: validate variable pointer constraints
        for inst in ctx.module.all_inst_iter() {
            let Some(result_id_raw) = inst.result_id else {
                continue;
            };
            let Ok(result_id) = ResultId::try_from(result_id_raw) else {
                continue;
            };

            // Skip if not a variable pointer
            if !var_ptr_cache.get(&result_id).copied().unwrap_or(false) {
                continue;
            }

            let inst_id = Id::try_from(result_id_raw).unwrap_or_else(|_| Id::try_from(1).unwrap());

            // Get the storage class
            let storage_class = get_pointer_storage_class(inst, ctx.definitions);

            // Check if this is a typed pointer with pointee type
            if let Some(type_id) = inst.result_type {
                if let Ok(type_result_id) = ResultId::try_from(type_id) {
                    if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                        if type_inst.class.opcode == Op::TypePointer {
                            if let Some(Operand::IdRef(pointee_id)) = type_inst.operands.get(1) {
                                if let Ok(pointee_type_id) = TypeId::try_from(*pointee_id) {
                                    // Check: variable pointer must not point to block array
                                    if let Ok(pointee_result_id) =
                                        ResultId::try_from(*pointee_id)
                                    {
                                        if let Some(pointee_inst) =
                                            ctx.definitions.get(&pointee_result_id)
                                        {
                                            if is_block_array(
                                                pointee_inst,
                                                ctx.definitions,
                                                ctx.module,
                                            ) {
                                                return Err(
                                                    ValidationError::VariablePointerToBlockArray {
                                                        pointer: inst_id,
                                                    },
                                                );
                                            }
                                        }
                                    }

                                    // Check: variable pointer must not point to matrix-containing type
                                    let mut visited = HashSet::new();
                                    if type_contains_matrix(
                                        pointee_type_id,
                                        ctx.definitions,
                                        &mut visited,
                                    ) {
                                        return Err(
                                            ValidationError::VariablePointerToMatrixType {
                                                pointer: inst_id,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Check: variable pointer must not point to matrix column/component
            let mut trace_visited = HashSet::new();
            if traces_through_matrix(inst, ctx.definitions, &mut trace_visited) {
                return Err(ValidationError::VariablePointerToMatrixElement {
                    pointer: inst_id,
                });
            }

            // Check same-buffer constraint for OpSelect/OpPhi without VariablePointers capability
            if !has_variable_pointers
                && matches!(inst.class.opcode, Op::Select | Op::Phi)
                && matches!(
                    storage_class,
                    Some(StorageClass::StorageBuffer | StorageClass::Workgroup)
                )
            {
                // For StorageBuffer, need VariablePointersStorageBuffer
                // For Workgroup, need VariablePointers
                let needs_full_capability = storage_class == Some(StorageClass::Workgroup);

                if needs_full_capability
                    || (storage_class == Some(StorageClass::StorageBuffer)
                        && !has_variable_pointers_storage_buffer)
                {
                    // Collect source variables
                    let mut source_vars: HashSet<ResultId> = HashSet::new();
                    let operand_indices: Vec<usize> = match inst.class.opcode {
                        Op::Select => vec![1, 2],
                        Op::Phi => (0..inst.operands.len()).step_by(2).collect(),
                        _ => vec![],
                    };

                    for idx in operand_indices {
                        if let Some(Operand::IdRef(val_id)) = inst.operands.get(idx) {
                            if let Some(var_id) =
                                trace_to_variable(*val_id, ctx.definitions, &mut HashSet::new())
                            {
                                source_vars.insert(var_id);
                            }
                        }
                    }

                    // Without full VariablePointers, must point to same structure
                    if source_vars.len() > 1 {
                        return Err(ValidationError::VariablePointerDifferentBuffers {
                            pointer: inst_id,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

/// Trace a pointer value back to its source variable.
fn trace_to_variable(
    id: u32,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visited: &mut HashSet<u32>,
) -> Option<ResultId> {
    if !visited.insert(id) {
        return None;
    }

    let result_id = ResultId::try_from(id).ok()?;
    let inst = definitions.get(&result_id)?;

    match inst.class.opcode {
        Op::Variable | Op::UntypedVariableKHR => {
            // Check if it's StorageBuffer or Workgroup
            if let Some(Operand::StorageClass(sc)) = inst.operands.first() {
                if *sc == StorageClass::StorageBuffer || *sc == StorageClass::Workgroup {
                    return Some(result_id);
                }
            }
            None
        }
        Op::AccessChain
        | Op::InBoundsAccessChain
        | Op::PtrAccessChain
        | Op::UntypedAccessChainKHR
        | Op::UntypedInBoundsAccessChainKHR
        | Op::UntypedPtrAccessChainKHR => {
            // Trace to base pointer
            if let Some(Operand::IdRef(base_id)) = inst.operands.first() {
                trace_to_variable(*base_id, definitions, visited)
            } else {
                None
            }
        }
        Op::CopyObject => {
            if let Some(Operand::IdRef(src_id)) = inst.operands.first() {
                trace_to_variable(*src_id, definitions, visited)
            } else {
                None
            }
        }
        Op::ConstantNull => None, // Null is allowed
        _ => None,
    }
}

// ============================================================================
// All pointer rules
// ============================================================================

/// Returns all pointer validation rules.
pub fn all_pointer_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &LogicalPointerRule,
        &LoadStoreLogicalPointerRule,
        &StoreTypeCompatibilityRule,
        &VariablePointerRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_pointer_rules() {
        let rules = all_pointer_rules();
        assert_eq!(rules.len(), 4);

        let names: Vec<_> = rules.iter().map(|r| r.name()).collect();
        assert!(names.contains(&"logical-pointers"));
        assert!(names.contains(&"load-store-logical-pointers"));
        assert!(names.contains(&"store-type-compatibility"));
        assert!(names.contains(&"variable-pointers"));
    }
}
