//! Pointer validation rules.
//!
//! This module validates SPIR-V pointer requirements including:
//!
//! - Logical pointer storage class restrictions
//! - Store type compatibility

use std::collections::HashSet;

use rspirv::dr::Module;
use rspirv::spirv::{AddressingModel, Capability, Op, StorageClass};

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
// All pointer rules
// ============================================================================

/// Returns all pointer validation rules.
pub fn all_pointer_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&LogicalPointerRule, &StoreTypeCompatibilityRule]
}
