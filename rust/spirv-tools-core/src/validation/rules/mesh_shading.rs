//! Mesh shading instruction validation rules.
//!
//! This module implements validation for mesh shading instructions from the
//! SPV_EXT_mesh_shader extension, including:
//! - OpEmitMeshTasksEXT (task shader emit)
//! - OpSetMeshOutputsEXT (mesh shader outputs)
//! - PerPrimitiveEXT decoration validation

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Decoration, ExecutionModel, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::get_type_structure;
use crate::validation::types::{Id, ResultId, ScalarKind, TypeId, TypeStructure};

#[cfg(test)]
use crate::validation::types::BitWidth;

/// Helper to check if a type is a 32-bit unsigned integer scalar.
fn is_uint32_scalar(ty: &TypeStructure) -> bool {
    match ty {
        TypeStructure::Scalar(ScalarKind::UnsignedInt(w)) => w.get() == 32,
        _ => false,
    }
}

/// Helper to convert a u32 to Id (with fallback to id 1).
fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Helper to get the type of a value operand (result of an instruction).
fn get_operand_type(id: u32, ctx: &ValidationContext<'_>) -> Option<TypeStructure> {
    let result_id = ResultId::try_from(id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;
    let type_id = TypeId::try_from(inst.result_type?).ok()?;
    Some(get_type_structure(type_id, ctx.definitions))
}

/// Validates OpEmitMeshTasksEXT instruction.
///
/// Validation rules:
/// - Must be used in TaskEXT execution model (enforced via execution model limitations)
/// - Group Count X, Y, Z must be 32-bit unsigned int scalars
/// - Optional Payload must be an OpVariable with TaskPayloadWorkgroupEXT storage class
pub struct EmitMeshTasksRule;

impl ValidationRule for EmitMeshTasksRule {
    fn name(&self) -> &'static str {
        "emit-mesh-tasks"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode == Op::EmitMeshTasksEXT {
                        // Validate group counts (operands 0, 1, 2)
                        for (idx, name) in [(0usize, "X"), (1, "Y"), (2, "Z")] {
                            if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                                if let Some(ty) = get_operand_type(*id, ctx) {
                                    if !is_uint32_scalar(&ty) {
                                        return Err(
                                            ValidationError::MeshShadingInvalidGroupCount {
                                                function: func_id,
                                                block: block_id,
                                                component: name,
                                            },
                                        );
                                    }
                                }
                            }
                        }

                        // Validate optional payload (operand 3)
                        if let Some(Operand::IdRef(payload_id)) = inst.operands.get(3) {
                            if let Ok(result_id) = ResultId::try_from(*payload_id) {
                                if let Some(payload_inst) = ctx.definitions.get(&result_id) {
                                    // Check that payload is an OpVariable
                                    if payload_inst.class.opcode != Op::Variable {
                                        return Err(
                                            ValidationError::MeshShadingPayloadMustBeVariable {
                                                function: func_id,
                                                block: block_id,
                                            },
                                        );
                                    }

                                    // Check storage class is TaskPayloadWorkgroupEXT
                                    if let Some(Operand::StorageClass(sc)) =
                                        payload_inst.operands.first()
                                    {
                                        if *sc != StorageClass::TaskPayloadWorkgroupEXT {
                                            return Err(
                                                ValidationError::MeshShadingPayloadWrongStorageClass {
                                                    function: func_id,
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
        Ok(())
    }
}

/// Validates OpSetMeshOutputsEXT instruction.
///
/// Validation rules:
/// - Must be used in MeshEXT execution model (enforced via execution model limitations)
/// - Vertex Count must be a 32-bit unsigned int scalar
/// - Primitive Count must be a 32-bit unsigned int scalar
pub struct SetMeshOutputsRule;

impl ValidationRule for SetMeshOutputsRule {
    fn name(&self) -> &'static str {
        "set-mesh-outputs"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module();

        for func in &module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode == Op::SetMeshOutputsEXT {
                        // Validate vertex count (operand 0)
                        if let Some(Operand::IdRef(id)) = inst.operands.first() {
                            if let Some(ty) = get_operand_type(*id, ctx) {
                                if !is_uint32_scalar(&ty) {
                                    return Err(
                                        ValidationError::MeshShadingInvalidOutputCount {
                                            function: func_id,
                                            block: block_id,
                                            count_name: "Vertex Count",
                                        },
                                    );
                                }
                            }
                        }

                        // Validate primitive count (operand 1)
                        if let Some(Operand::IdRef(id)) = inst.operands.get(1) {
                            if let Some(ty) = get_operand_type(*id, ctx) {
                                if !is_uint32_scalar(&ty) {
                                    return Err(
                                        ValidationError::MeshShadingInvalidOutputCount {
                                            function: func_id,
                                            block: block_id,
                                            count_name: "Primitive Count",
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

/// Validates PerPrimitiveEXT decoration on OpVariable instructions.
///
/// Validation rules:
/// - In Fragment execution model: PerPrimitiveEXT can only be on Input storage class
/// - In MeshEXT execution model: PerPrimitiveEXT can only be on Output storage class
pub struct PerPrimitiveDecorationRule;

impl ValidationRule for PerPrimitiveDecorationRule {
    fn name(&self) -> &'static str {
        "per-primitive-decoration"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Only applies when MeshShadingEXT capability is present
        if !ctx
            .module()
            .capabilities
            .iter()
            .any(|c| c.operands.first() == Some(&Operand::Capability(Capability::MeshShadingEXT)))
        {
            return Ok(());
        }

        let module = ctx.module();

        // Build map of variable ID -> storage class from global variables
        let mut var_storage_class: std::collections::HashMap<u32, StorageClass> =
            std::collections::HashMap::new();
        for inst in module.types_global_values.iter() {
            if inst.class.opcode == Op::Variable {
                if let (Some(result_id), Some(Operand::StorageClass(sc))) =
                    (inst.result_id, inst.operands.first())
                {
                    var_storage_class.insert(result_id, *sc);
                }
            }
        }

        // Build set of variables with PerPrimitiveEXT decoration
        let mut per_primitive_vars: std::collections::HashSet<u32> =
            std::collections::HashSet::new();
        for inst in &module.annotations {
            if inst.class.opcode == Op::Decorate {
                if let (Some(Operand::IdRef(target)), Some(Operand::Decoration(dec))) =
                    (inst.operands.first(), inst.operands.get(1))
                {
                    if *dec == Decoration::PerPrimitiveEXT {
                        per_primitive_vars.insert(*target);
                    }
                }
            }
        }

        // Build map of execution model -> interface variables
        let mut mesh_interfaces: std::collections::HashSet<u32> =
            std::collections::HashSet::new();
        let mut fragment_interfaces: std::collections::HashSet<u32> =
            std::collections::HashSet::new();

        for entry in &module.entry_points {
            if entry.class.opcode == Op::EntryPoint {
                if let Some(Operand::ExecutionModel(model)) = entry.operands.first() {
                    // Interface variables start at operand index 2 (after execution model and function id)
                    for operand in entry.operands.iter().skip(2) {
                        if let Operand::IdRef(id) = operand {
                            match model {
                                ExecutionModel::MeshEXT => {
                                    mesh_interfaces.insert(*id);
                                }
                                ExecutionModel::Fragment => {
                                    fragment_interfaces.insert(*id);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Validate each variable with PerPrimitiveEXT decoration
        for var_id in &per_primitive_vars {
            if let Some(storage_class) = var_storage_class.get(var_id) {
                let is_fragment_interface = fragment_interfaces.contains(var_id);
                let is_mesh_interface = mesh_interfaces.contains(var_id);

                // Fragment: PerPrimitiveEXT requires Input storage class
                if is_fragment_interface && *storage_class != StorageClass::Input {
                    return Err(
                        ValidationError::MeshShadingPerPrimitiveFragmentWrongStorageClass {
                            variable_id: to_id(*var_id),
                        },
                    );
                }

                // MeshEXT: PerPrimitiveEXT requires Output storage class
                if is_mesh_interface && *storage_class != StorageClass::Output {
                    return Err(
                        ValidationError::MeshShadingPerPrimitiveMeshWrongStorageClass {
                            variable_id: to_id(*var_id),
                        },
                    );
                }
            }
        }

        Ok(())
    }
}

/// Returns all mesh shading validation rules.
pub fn all_mesh_shading_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(EmitMeshTasksRule),
        Box::new(SetMeshOutputsRule),
        Box::new(PerPrimitiveDecorationRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_uint32_scalar() {
        assert!(is_uint32_scalar(&TypeStructure::Scalar(
            ScalarKind::UnsignedInt(BitWidth::BITS_32)
        )));

        assert!(!is_uint32_scalar(&TypeStructure::Scalar(
            ScalarKind::SignedInt(BitWidth::BITS_32)
        )));

        assert!(!is_uint32_scalar(&TypeStructure::Scalar(
            ScalarKind::UnsignedInt(BitWidth::BITS_64)
        )));

        assert!(!is_uint32_scalar(&TypeStructure::Scalar(
            ScalarKind::Float(BitWidth::BITS_32)
        )));

        assert!(!is_uint32_scalar(&TypeStructure::Scalar(ScalarKind::Bool)));
    }

    #[test]
    fn test_all_mesh_shading_rules() {
        let rules = all_mesh_shading_rules();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[0].name(), "emit-mesh-tasks");
        assert_eq!(rules[1].name(), "set-mesh-outputs");
        assert_eq!(rules[2].name(), "per-primitive-decoration");
    }
}
