//! Composite type validation rules.
//!
//! This module validates SPIR-V composite type requirements:
//! - OpTypeVector component count and capability requirements
//! - OpTypeMatrix column type and count requirements
//! - OpTypeArray/OpTypeRuntimeArray element type requirements
//! - OpTypeStruct member requirements

use std::collections::HashSet;

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Decoration, Op};

use crate::target_env::TargetEnv;
use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::helpers::has_decoration;
use crate::validation::types::{ResultId, TypeId};

use super::helpers::{get_constant_int_value, is_constant_opcode, is_scalar_type, is_type_opcode};

// ============================================================================
// OpTypeVector Validation Rule
// ============================================================================

/// Validates OpTypeVector requirements.
///
/// Checks:
/// - Component type must be a scalar type
/// - Component count must be 2, 3, or 4 (or 8, 16 with Vector16 capability)
pub struct TypeVectorRule;

impl ValidationRule for TypeVectorRule {
    fn name(&self) -> &'static str {
        "type-vector"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeVector {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get component type (operand 0)
            let component_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate component type is a scalar (or pointer with MaskedGatherScatterINTEL)
            if let Ok(component_result_id) = ResultId::try_from(component_type_raw) {
                if let Some(component_opcode) = ctx.opcodes.get(&component_result_id) {
                    let is_scalar = is_scalar_type(*component_opcode);
                    let is_pointer = *component_opcode == Op::TypePointer;
                    let has_masked_gather_scatter =
                        ctx.has_capability(Capability::MaskedGatherScatterINTEL);

                    // With MaskedGatherScatterINTEL, allow scalar or pointer
                    // Without it, only allow scalar
                    if has_masked_gather_scatter {
                        if !is_scalar && !is_pointer {
                            let component_type = TypeId::try_from(component_type_raw)
                                .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                            return Err(ValidationError::TypeVectorComponentNotScalarOrPointer {
                                type_id,
                                component_type,
                            }.into());
                        }
                    } else if !is_scalar {
                        let component_type = TypeId::try_from(component_type_raw)
                            .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                        return Err(ValidationError::TypeVectorComponentNotScalar {
                            type_id,
                            component_type,
                        }.into());
                    }
                }
            }

            // Get component count (operand 1)
            let component_count = match inst.operands.get(1) {
                Some(Operand::LiteralBit32(c)) => *c,
                _ => continue,
            };

            // Validate component count
            match component_count {
                2 | 3 | 4 => {
                    // Always valid
                }
                8 | 16 => {
                    if !ctx.declared_capabilities.contains(&Capability::Vector16) {
                        return Err(ValidationError::TypeVectorRequiresVector16Capability {
                            type_id,
                            component_count,
                        }.into());
                    }
                }
                _ => {
                    return Err(ValidationError::TypeVectorInvalidComponentCount {
                        type_id,
                        component_count,
                    }.into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeMatrix Validation Rule
// ============================================================================

/// Validates OpTypeMatrix requirements.
///
/// Checks:
/// - Column type must be a vector type
/// - Vector component type must be a float type
/// - Column count must be 2, 3, or 4
pub struct TypeMatrixRule;

impl ValidationRule for TypeMatrixRule {
    fn name(&self) -> &'static str {
        "type-matrix"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeMatrix {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get column type (operand 0)
            let column_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate column type is a vector
            if let Ok(column_result_id) = ResultId::try_from(column_type_raw) {
                if let Some(column_inst) = ctx.definitions.get(&column_result_id) {
                    if column_inst.class.opcode != Op::TypeVector {
                        return Err(ValidationError::TypeMatrixColumnNotVector { type_id }.into());
                    }

                    // Check that the vector component type is float
                    if let Some(Operand::IdRef(component_type_raw)) = column_inst.operands.first() {
                        if let Ok(component_result_id) = ResultId::try_from(*component_type_raw) {
                            if let Some(component_opcode) = ctx.opcodes.get(&component_result_id) {
                                if *component_opcode != Op::TypeFloat {
                                    return Err(ValidationError::TypeMatrixComponentNotFloat {
                                        type_id,
                                    }.into());
                                }
                            }
                        }
                    }
                }
            }

            // Get column count (operand 1)
            let column_count = match inst.operands.get(1) {
                Some(Operand::LiteralBit32(c)) => *c,
                _ => continue,
            };

            // Validate column count
            if column_count < 2 || column_count > 4 {
                return Err(ValidationError::TypeMatrixInvalidColumnCount {
                    type_id,
                    column_count,
                }.into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeArray Validation Rule
// ============================================================================

/// Validates OpTypeArray requirements.
///
/// Checks:
/// - Element type must not be void
/// - Length must be a constant integer >= 1
pub struct TypeArrayRule;

impl ValidationRule for TypeArrayRule {
    fn name(&self) -> &'static str {
        "type-array"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeArray {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get element type (operand 0)
            let element_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate element type is not void
            if let Ok(element_result_id) = ResultId::try_from(element_type_raw) {
                if let Some(element_opcode) = ctx.opcodes.get(&element_result_id) {
                    if *element_opcode == Op::TypeVoid {
                        return Err(ValidationError::TypeArrayElementVoid { type_id }.into());
                    }

                    // In Vulkan, array element cannot be RuntimeArray
                    if ctx.is_vulkan_env() && *element_opcode == Op::TypeRuntimeArray {
                        return Err(ValidationError::TypeArrayElementCannotBeRuntimeArray {
                            type_id,
                        }.into());
                    }
                }
            }

            // Get length (operand 1)
            let length_id_raw = match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate length is a constant
            if let Ok(length_result_id) = ResultId::try_from(length_id_raw) {
                if let Some(length_inst) = ctx.definitions.get(&length_result_id) {
                    if !is_constant_opcode(length_inst.class.opcode) {
                        return Err(ValidationError::TypeArrayLengthNotConstant { type_id }.into());
                    }

                    // Check that the constant type is integer
                    if let Some(const_type_raw) = length_inst.result_type {
                        if let Ok(const_type_result_id) = ResultId::try_from(const_type_raw) {
                            if let Some(const_type_opcode) = ctx.opcodes.get(&const_type_result_id) {
                                if *const_type_opcode != Op::TypeInt {
                                    return Err(ValidationError::TypeArrayLengthNotInteger {
                                        type_id,
                                    }.into());
                                }
                            }
                        }
                    }

                    // Try to evaluate the constant value
                    if let Some(length_value) = get_constant_int_value(length_inst, ctx) {
                        if length_value <= 0 {
                            return Err(ValidationError::TypeArrayLengthInvalid {
                                type_id,
                                length: length_value,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeRuntimeArray Validation Rule
// ============================================================================

/// Validates OpTypeRuntimeArray requirements.
///
/// Checks:
/// - Element type must not be void
pub struct TypeRuntimeArrayRule;

impl ValidationRule for TypeRuntimeArrayRule {
    fn name(&self) -> &'static str {
        "type-runtime-array"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeRuntimeArray {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get element type (operand 0)
            let element_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate element type is not void
            if let Ok(element_result_id) = ResultId::try_from(element_type_raw) {
                if let Some(element_opcode) = ctx.opcodes.get(&element_result_id) {
                    if *element_opcode == Op::TypeVoid {
                        return Err(ValidationError::TypeRuntimeArrayElementVoid { type_id }.into());
                    }

                    // In Vulkan, runtime array element cannot be RuntimeArray
                    if ctx.is_vulkan_env() && *element_opcode == Op::TypeRuntimeArray {
                        return Err(ValidationError::TypeArrayElementCannotBeRuntimeArray {
                            type_id,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeStruct Validation Rule
// ============================================================================

/// Validates OpTypeStruct requirements.
///
/// Checks:
/// - Members cannot be self-references (referring to the struct being defined)
/// - Members must be type instructions
/// - Members cannot be void types
/// - Cannot contain struct members with BuiltIn decoration
/// - (Vulkan) RuntimeArray must be last member and struct must have Block/BufferBlock
/// - Cannot nest Block/BufferBlock decorated structs
/// - BuiltIn decoration must be all-or-nothing for members
/// - (Vulkan) Cannot contain opaque types
pub struct TypeStructRule;

impl ValidationRule for TypeStructRule {
    fn name(&self) -> &'static str {
        "type-struct"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let is_vulkan = matches!(
            ctx.env,
            TargetEnv::Vulkan1_0
                | TargetEnv::Vulkan1_1
                | TargetEnv::Vulkan1_1Spirv1_4
                | TargetEnv::Vulkan1_2
                | TargetEnv::Vulkan1_3
                | TargetEnv::Vulkan1_4
        );

        // Collect struct types with BuiltIn member decorations
        let structs_with_builtin_members = collect_structs_with_builtin_members(ctx);

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeStruct {
                continue;
            }

            let struct_id = inst.result_id.unwrap_or(0);
            let type_id = TypeId::try_from(struct_id)
                .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());

            let member_count = inst.operands.len();

            // Validate each member
            for (member_idx, operand) in inst.operands.iter().enumerate() {
                let member_type_raw = match operand {
                    Operand::IdRef(id) => *id,
                    _ => continue,
                };

                // Check for self-reference
                if member_type_raw == struct_id {
                    return Err(ValidationError::TypeStructMemberSelfReference { type_id }.into());
                }

                // Get the member type instruction
                let member_result_id = match ResultId::try_from(member_type_raw) {
                    Ok(id) => id,
                    Err(_) => continue,
                };

                let member_inst = match ctx.definitions.get(&member_result_id) {
                    Some(inst) => inst,
                    None => continue,
                };

                // Check that member is a type instruction
                if !is_type_opcode(member_inst.class.opcode) {
                    let member_type = TypeId::try_from(member_type_raw)
                        .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                    return Err(ValidationError::TypeStructMemberNotType {
                        type_id,
                        member_type,
                    }.into());
                }

                // Check for void type
                if member_inst.class.opcode == Op::TypeVoid {
                    return Err(ValidationError::TypeStructMemberVoid { type_id }.into());
                }

                // Check for nested struct with BuiltIn members
                if member_inst.class.opcode == Op::TypeStruct {
                    if structs_with_builtin_members.contains(&member_result_id) {
                        let member_type = TypeId::try_from(member_type_raw)
                            .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                        return Err(ValidationError::TypeStructContainsBuiltInStruct {
                            type_id,
                            member_type,
                        }.into());
                    }
                }

                // Vulkan: RuntimeArray validation
                if is_vulkan && member_inst.class.opcode == Op::TypeRuntimeArray {
                    let is_last_member = member_idx == member_count - 1;
                    if !is_last_member {
                        return Err(ValidationError::TypeStructRuntimeArrayNotLast { type_id }.into());
                    }

                    // Struct must have Block or BufferBlock decoration
                    let has_block =
                        has_decoration(ctx.module, struct_id, Decoration::Block);
                    let has_buffer_block =
                        has_decoration(ctx.module, struct_id, Decoration::BufferBlock);
                    if !has_block && !has_buffer_block {
                        return Err(ValidationError::TypeStructRuntimeArrayNoBlockDecoration {
                            type_id,
                        }.into());
                    }
                }

                // Vulkan: Check for opaque types
                if is_vulkan && !ctx.options.before_hlsl_legalization {
                    if contains_opaque_type(member_type_raw, ctx) {
                        return Err(ValidationError::TypeStructContainsOpaqueType { type_id }.into());
                    }
                }
            }

            // Check for nested Block/BufferBlock
            let this_has_block = has_decoration(ctx.module, struct_id, Decoration::Block);
            let this_has_buffer_block =
                has_decoration(ctx.module, struct_id, Decoration::BufferBlock);

            if this_has_block || this_has_buffer_block {
                if has_nested_block_or_buffer_block(inst, ctx) {
                    return Err(ValidationError::TypeStructNestedBlockOrBufferBlock { type_id }.into());
                }
            }

            // Check BuiltIn all-or-nothing rule
            let builtin_member_count =
                count_builtin_decorated_members(struct_id, ctx);
            if builtin_member_count > 0 && builtin_member_count != member_count {
                return Err(ValidationError::TypeStructBuiltInNotAllMembers {
                    type_id,
                    builtin_count: builtin_member_count,
                    total_count: member_count,
                }.into());
            }
        }

        Ok(())
    }
}

/// Collects all struct type IDs that have at least one member with BuiltIn decoration.
fn collect_structs_with_builtin_members(ctx: &ValidationContext<'_>) -> HashSet<ResultId> {
    let mut result = HashSet::new();

    for inst in &ctx.module.annotations {
        if inst.class.opcode == Op::MemberDecorate {
            if let (
                Some(Operand::IdRef(struct_id)),
                Some(Operand::LiteralBit32(_)),
                Some(Operand::Decoration(Decoration::BuiltIn)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            ) {
                if let Ok(result_id) = ResultId::try_from(*struct_id) {
                    result.insert(result_id);
                }
            }
        }
    }

    result
}

/// Counts members of a struct that have BuiltIn decoration.
fn count_builtin_decorated_members(struct_id: u32, ctx: &ValidationContext<'_>) -> usize {
    let mut builtin_members = HashSet::new();

    for inst in &ctx.module.annotations {
        if inst.class.opcode == Op::MemberDecorate {
            if let (
                Some(Operand::IdRef(target_id)),
                Some(Operand::LiteralBit32(member_idx)),
                Some(Operand::Decoration(Decoration::BuiltIn)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            ) {
                if *target_id == struct_id {
                    builtin_members.insert(*member_idx);
                }
            }
        }
    }

    builtin_members.len()
}

/// Checks if a struct has any nested Block or BufferBlock decorated structs.
fn has_nested_block_or_buffer_block(
    struct_inst: &rspirv::dr::Instruction,
    ctx: &ValidationContext<'_>,
) -> bool {
    for operand in &struct_inst.operands {
        let member_type_raw = match operand {
            Operand::IdRef(id) => *id,
            _ => continue,
        };

        if contains_block_or_buffer_block(member_type_raw, ctx, &mut HashSet::new()) {
            return true;
        }
    }
    false
}

/// Recursively checks if a type contains a Block or BufferBlock decorated struct.
fn contains_block_or_buffer_block(
    type_id: u32,
    ctx: &ValidationContext<'_>,
    visited: &mut HashSet<u32>,
) -> bool {
    if !visited.insert(type_id) {
        return false; // Already visited, prevent infinite recursion
    }

    let result_id = match ResultId::try_from(type_id) {
        Ok(id) => id,
        Err(_) => return false,
    };

    let type_inst = match ctx.definitions.get(&result_id) {
        Some(inst) => inst,
        None => return false,
    };

    match type_inst.class.opcode {
        Op::TypeStruct => {
            // Check if this struct has Block or BufferBlock decoration
            let has_block = has_decoration(ctx.module, type_id, Decoration::Block);
            let has_buffer_block = has_decoration(ctx.module, type_id, Decoration::BufferBlock);
            if has_block || has_buffer_block {
                return true;
            }
            // Check members recursively
            for operand in &type_inst.operands {
                if let Operand::IdRef(member_type_id) = operand {
                    if contains_block_or_buffer_block(*member_type_id, ctx, visited) {
                        return true;
                    }
                }
            }
            false
        }
        Op::TypeArray | Op::TypeRuntimeArray => {
            // Check element type
            if let Some(Operand::IdRef(element_type_id)) = type_inst.operands.first() {
                contains_block_or_buffer_block(*element_type_id, ctx, visited)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Checks if a type is or contains an opaque type.
fn contains_opaque_type(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    contains_opaque_type_impl(type_id, ctx, &mut HashSet::new())
}

fn contains_opaque_type_impl(
    type_id: u32,
    ctx: &ValidationContext<'_>,
    visited: &mut HashSet<u32>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }

    let result_id = match ResultId::try_from(type_id) {
        Ok(id) => id,
        Err(_) => return false,
    };

    let type_inst = match ctx.definitions.get(&result_id) {
        Some(inst) => inst,
        None => return false,
    };

    // Check if this is an opaque type
    if is_base_opaque_type(type_inst.class.opcode) {
        // Exception: BindlessTextureNV capability allows Image/Sampler/SampledImage
        if ctx.has_capability(Capability::BindlessTextureNV) {
            if matches!(
                type_inst.class.opcode,
                Op::TypeImage | Op::TypeSampler | Op::TypeSampledImage
            ) {
                return false;
            }
        }
        return true;
    }

    // Check nested types
    match type_inst.class.opcode {
        Op::TypeStruct => {
            for operand in &type_inst.operands {
                if let Operand::IdRef(member_type_id) = operand {
                    if contains_opaque_type_impl(*member_type_id, ctx, visited) {
                        return true;
                    }
                }
            }
            false
        }
        Op::TypeArray | Op::TypeRuntimeArray => {
            if let Some(Operand::IdRef(element_type_id)) = type_inst.operands.first() {
                contains_opaque_type_impl(*element_type_id, ctx, visited)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Checks if an opcode is a base opaque type.
fn is_base_opaque_type(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
            | Op::TypeOpaque
            | Op::TypeEvent
            | Op::TypeDeviceEvent
            | Op::TypeReserveId
            | Op::TypeQueue
            | Op::TypePipe
    )
}

/// Returns all composite type validation rules.
pub fn all_composite_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &TypeVectorRule,
        &TypeMatrixRule,
        &TypeArrayRule,
        &TypeRuntimeArrayRule,
        &TypeStructRule,
    ]
}
