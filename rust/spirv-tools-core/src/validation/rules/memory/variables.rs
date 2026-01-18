//! Variable declaration validation rules.
//!
//! This module validates OpVariable instructions.

use rspirv::dr::Operand;
use rspirv::spirv::{Decoration, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::ResultId;

use super::helpers::{
    contains_bool, get_pointee_type, get_pointer_storage_class, has_decoration, id_from_u32,
};

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

                if !allows_bool
                    && contains_bool(pt_id, ctx.definitions, &mut std::collections::HashSet::new())
                {
                    // Input/Output with BuiltIn is allowed
                    let is_builtin = inst
                        .result_id
                        .map_or(false, |id| has_decoration(ctx.module, id, Decoration::BuiltIn));

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
