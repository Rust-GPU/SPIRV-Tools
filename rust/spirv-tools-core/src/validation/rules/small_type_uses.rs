//! Small type usage validation rules.
//!
//! This module validates that 8-bit and 16-bit types are used correctly
//! when the corresponding capability (Int8, Int16, Float16) is not present.
//!
//! When these capabilities are missing, uses of 8/16-bit types are restricted to:
//! - OpDecorate / OpDecorateId
//! - OpCopyObject
//! - OpStore
//! - OpFConvert / OpUConvert / OpSConvert (width conversions)

use std::collections::HashMap;

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::helpers::get_type_structure;
use crate::validation::types::{Id, TypeId, TypeStructure};

/// Helper to convert a u32 to Id (with fallback to id 1).
fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Check if a type contains an 8-bit integer without Int8 capability.
fn contains_limited_int8(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    if ctx.has_capability(Capability::Int8) {
        return false;
    }
    contains_sized_int(type_id, 8, ctx)
}

/// Check if a type contains a 16-bit integer without Int16 capability.
fn contains_limited_int16(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    if ctx.has_capability(Capability::Int16) {
        return false;
    }
    contains_sized_int(type_id, 16, ctx)
}

/// Check if a type contains a 16-bit float without Float16 capability.
fn contains_limited_float16(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    if ctx.has_capability(Capability::Float16) {
        return false;
    }
    contains_sized_float(type_id, 16, ctx)
}

/// Check if a type is or contains a limited-use 8/16-bit int or float.
fn contains_limited_use_type(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    contains_limited_int8(type_id, ctx)
        || contains_limited_int16(type_id, ctx)
        || contains_limited_float16(type_id, ctx)
}

/// Recursively check if a type contains an integer of the given bit width.
fn contains_sized_int(type_id: TypeId, bit_width: u32, ctx: &ValidationContext<'_>) -> bool {
    let ty = get_type_structure(type_id, ctx.definitions);
    match ty {
        TypeStructure::Scalar(crate::validation::types::ScalarKind::SignedInt(w))
        | TypeStructure::Scalar(crate::validation::types::ScalarKind::UnsignedInt(w)) => {
            w.get() == bit_width
        }
        TypeStructure::Vector { component, .. } => {
            let comp_ty = match component {
                crate::validation::types::ScalarKind::SignedInt(w)
                | crate::validation::types::ScalarKind::UnsignedInt(w) => w.get() == bit_width,
                _ => false,
            };
            comp_ty
        }
        TypeStructure::Matrix { component, .. } => {
            let comp_ty = match component {
                crate::validation::types::ScalarKind::SignedInt(w)
                | crate::validation::types::ScalarKind::UnsignedInt(w) => w.get() == bit_width,
                _ => false,
            };
            comp_ty
        }
        TypeStructure::Array { element, .. } => contains_sized_int(element, bit_width, ctx),
        TypeStructure::RuntimeArray { element } => contains_sized_int(element, bit_width, ctx),
        TypeStructure::Struct { members } => members
            .iter()
            .any(|m| contains_sized_int(*m, bit_width, ctx)),
        TypeStructure::Pointer { pointee, .. } => {
            if let Some(p) = pointee {
                contains_sized_int(p, bit_width, ctx)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Recursively check if a type contains a float of the given bit width.
fn contains_sized_float(type_id: TypeId, bit_width: u32, ctx: &ValidationContext<'_>) -> bool {
    let ty = get_type_structure(type_id, ctx.definitions);
    match ty {
        TypeStructure::Scalar(crate::validation::types::ScalarKind::Float(w)) => {
            w.get() == bit_width
        }
        TypeStructure::Vector { component, .. } => {
            let comp_ty = match component {
                crate::validation::types::ScalarKind::Float(w) => w.get() == bit_width,
                _ => false,
            };
            comp_ty
        }
        TypeStructure::Matrix { component, .. } => {
            let comp_ty = match component {
                crate::validation::types::ScalarKind::Float(w) => w.get() == bit_width,
                _ => false,
            };
            comp_ty
        }
        TypeStructure::Array { element, .. } => contains_sized_float(element, bit_width, ctx),
        TypeStructure::RuntimeArray { element } => contains_sized_float(element, bit_width, ctx),
        TypeStructure::Struct { members } => members
            .iter()
            .any(|m| contains_sized_float(*m, bit_width, ctx)),
        TypeStructure::Pointer { pointee, .. } => {
            if let Some(p) = pointee {
                contains_sized_float(p, bit_width, ctx)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if a type is a pointer type.
fn is_pointer_type(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    let ty = get_type_structure(type_id, ctx.definitions);
    matches!(ty, TypeStructure::Pointer { .. } | TypeStructure::ForwardPointer { .. })
}

/// Check if a type is scalar, vector, or matrix (numeric types allowed for limited-use Load/Store).
fn is_numeric_type(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    let ty = get_type_structure(type_id, ctx.definitions);
    matches!(
        ty,
        TypeStructure::Scalar(_) | TypeStructure::Vector { .. } | TypeStructure::Matrix { .. }
    )
}

/// Check if an opcode is allowed for limited-use small types.
fn is_allowed_use_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Decorate
            | Op::DecorateId
            | Op::CopyObject
            | Op::Store
            | Op::FConvert
            | Op::UConvert
            | Op::SConvert
    )
}

/// Build a map from result ID to all instructions that use it.
fn build_use_map(ctx: &ValidationContext<'_>) -> HashMap<u32, Vec<(Op, Option<u32>)>> {
    let mut uses: HashMap<u32, Vec<(Op, Option<u32>)>> = HashMap::new();

    // Collect uses from all instructions in the module
    for inst in ctx.module.all_inst_iter() {
        let inst_result_id = inst.result_id;
        for operand in &inst.operands {
            if let Operand::IdRef(id) = operand {
                uses.entry(*id)
                    .or_default()
                    .push((inst.class.opcode, inst_result_id));
            }
        }
    }

    uses
}

/// Validates small type usage restrictions.
///
/// When 8- or 16-bit types are used without the corresponding capability,
/// their uses are restricted to specific operations like stores and conversions.
pub struct SmallTypeUsesRule;

impl ValidationRule for SmallTypeUsesRule {
    fn name(&self) -> &'static str {
        "small-type-uses"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Only applies when Shader capability is present
        if !ctx.has_capability(Capability::Shader) {
            return Ok(());
        }

        // Build the use map once
        let use_map = build_use_map(ctx);

        // Check all instructions with a result type
        for func in &ctx.module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    // Skip instructions without a type
                    let Some(result_type_raw) = inst.result_type else {
                        continue;
                    };
                    let Ok(type_id) = TypeId::try_from(result_type_raw) else {
                        continue;
                    };

                    // Skip if not a limited-use type
                    if !contains_limited_use_type(type_id, ctx) {
                        continue;
                    }

                    // Skip pointer types - they're allowed
                    if is_pointer_type(type_id, ctx) {
                        continue;
                    }

                    // Check all uses of this instruction's result
                    let Some(result_id) = inst.result_id else {
                        continue;
                    };

                    if let Some(users) = use_map.get(&result_id) {
                        for (user_opcode, user_result_id) in users {
                            if !is_allowed_use_opcode(*user_opcode) {
                                return Err(ValidationError::InvalidSmallTypeUse {
                                    instruction_id: user_result_id.map(to_id),
                                    opcode: *user_opcode,
                                }.into());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates that OpLoad with limited-use 8/16-bit types has a numeric result type.
///
/// When Shader capability is present and a type contains 8/16-bit integers or floats
/// without the corresponding capability (Int8, Int16, Float16), the Load/Store operand
/// type must be scalar, vector, or matrix - not an array or struct.
pub struct SmallTypeLoadStoreRule;

impl ValidationRule for SmallTypeLoadStoreRule {
    fn name(&self) -> &'static str {
        "small-type-load-store"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Only applies when Shader capability is present
        if !ctx.has_capability(Capability::Shader) {
            return Ok(());
        }

        for func in &ctx.module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id)
                .unwrap_or_else(|| to_id(0));

            for block in &func.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .map(to_id)
                    .unwrap_or_else(|| to_id(0));

                for inst in &block.instructions {
                    // Check OpLoad - result type is the loaded type
                    if inst.class.opcode == Op::Load {
                        let Some(result_type_raw) = inst.result_type else {
                            continue;
                        };
                        let Ok(type_id) = TypeId::try_from(result_type_raw) else {
                            continue;
                        };

                        // Only check if type contains limited-use small types
                        if !contains_limited_use_type(type_id, ctx) {
                            continue;
                        }

                        // Type must be scalar, vector, or matrix
                        if !is_numeric_type(type_id, ctx) && !is_pointer_type(type_id, ctx) {
                            return Err(ValidationError::LoadStoreSmallTypeComposite {
                                function: func_id,
                                block: block_id,
                                opcode: Op::Load,
                            }.into());
                        }
                    }

                    // Check OpStore - check the object's type
                    if inst.class.opcode == Op::Store {
                        // Get the object operand (second operand)
                        let Some(Operand::IdRef(object_id)) = inst.operands.get(1) else {
                            continue;
                        };

                        // Look up the object's definition to get its type
                        let Some(obj_rid) = crate::validation::types::ResultId::try_from(*object_id).ok() else {
                            continue;
                        };
                        let Some(obj_inst) = ctx.definitions.get(&obj_rid) else {
                            continue;
                        };
                        let Some(obj_type_raw) = obj_inst.result_type else {
                            continue;
                        };
                        let Ok(type_id) = TypeId::try_from(obj_type_raw) else {
                            continue;
                        };

                        // Only check if type contains limited-use small types
                        if !contains_limited_use_type(type_id, ctx) {
                            continue;
                        }

                        // Type must be scalar, vector, or matrix
                        if !is_numeric_type(type_id, ctx) && !is_pointer_type(type_id, ctx) {
                            return Err(ValidationError::LoadStoreSmallTypeComposite {
                                function: func_id,
                                block: block_id,
                                opcode: Op::Store,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Returns all small type usage validation rules.
pub fn all_small_type_uses_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![Box::new(SmallTypeUsesRule), Box::new(SmallTypeLoadStoreRule)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_allowed_use_opcode() {
        assert!(is_allowed_use_opcode(Op::Decorate));
        assert!(is_allowed_use_opcode(Op::DecorateId));
        assert!(is_allowed_use_opcode(Op::CopyObject));
        assert!(is_allowed_use_opcode(Op::Store));
        assert!(is_allowed_use_opcode(Op::FConvert));
        assert!(is_allowed_use_opcode(Op::UConvert));
        assert!(is_allowed_use_opcode(Op::SConvert));

        // Not allowed
        assert!(!is_allowed_use_opcode(Op::IAdd));
        assert!(!is_allowed_use_opcode(Op::FAdd));
        assert!(!is_allowed_use_opcode(Op::IMul));
        assert!(!is_allowed_use_opcode(Op::Load));
    }
}
