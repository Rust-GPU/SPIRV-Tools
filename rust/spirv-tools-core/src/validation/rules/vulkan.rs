//! Vulkan-specific validation rules.
//!
//! This module validates Vulkan-specific SPIR-V requirements including:
//!
//! - Offset texture operand restrictions
//! - Bitwise operation width requirements
//! - Small type storage class capabilities

use std::collections::HashSet;

use rspirv::spirv::{Capability, Decoration, ImageOperands, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::helpers::is_vulkan_env;
use crate::validation::types::{ResultId, TypeId};

// ============================================================================
// Offset Texture Operand Rule
// ============================================================================

/// Validates that Offset texture operand is only used with gather operations in Vulkan.
pub struct OffsetTextureOperandRule;

impl ValidationRule for OffsetTextureOperandRule {
    fn name(&self) -> &'static str {
        "offset-texture-operand"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        ctx.options.allow_offset_texture_operand
            || ctx.options.before_hlsl_legalization
            || !is_vulkan_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let gather_opcodes = [
            Op::ImageGather,
            Op::ImageDrefGather,
            Op::ImageSparseGather,
            Op::ImageSparseDrefGather,
        ];

        for inst in ctx.module.all_inst_iter() {
            let has_offset = inst.operands.iter().any(|op| {
                matches!(
                    op,
                    rspirv::dr::Operand::ImageOperands(mask)
                        if mask.contains(ImageOperands::OFFSET)
                )
            });
            if has_offset && !gather_opcodes.contains(&inst.class.opcode) {
                return Err(ValidationError::OffsetTextureOperandDisallowed {
                    opcode: inst.class.opcode,
                }.into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// Vulkan Bitwise Widths Rule
// ============================================================================

/// Validates that bit field and bit count operations use 32-bit integers in Vulkan.
///
/// Per the Vulkan spec, the Base operand of OpBitFieldInsert, OpBitFieldSExtract,
/// OpBitFieldUExtract, OpBitReverse, and OpBitCount must be 32-bit integers
/// unless the maintenance9 feature is enabled (allow_vulkan_32_bit_bitwise option).
///
/// NOTE: This restriction does NOT apply to shift operations (OpShiftRightLogical,
/// OpShiftLeftLogical, OpShiftRightArithmetic) or basic bitwise operations
/// (OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd, OpNot).
pub struct VulkanBitwiseWidthsRule;

impl ValidationRule for VulkanBitwiseWidthsRule {
    fn name(&self) -> &'static str {
        "vulkan-bitwise-widths"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        ctx.options.allow_vulkan_32_bit_bitwise || !is_vulkan_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Only bit field and bit count operations have the 32-bit restriction in Vulkan
        let restricted_opcodes = [
            Op::BitFieldInsert,
            Op::BitFieldSExtract,
            Op::BitFieldUExtract,
            Op::BitReverse,
            Op::BitCount,
        ];

        for inst in ctx.module.all_inst_iter() {
            if !restricted_opcodes.contains(&inst.class.opcode) {
                continue;
            }
            let Some(raw_type) = inst.result_type else {
                continue;
            };
            let Ok(type_id) = TypeId::try_from(raw_type) else {
                continue;
            };
            if let Some(bit_width) = int_bit_width(type_id, ctx.definitions) {
                if bit_width != 32 {
                    return Err(ValidationError::VulkanBitwiseRequires32Bit {
                        opcode: inst.class.opcode,
                        bit_width,
                    }.into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Small Type Storage Capabilities Rule
// ============================================================================

/// Validates that 8-bit and 16-bit types have required capabilities for their storage classes.
pub struct SmallTypeStorageCapabilitiesRule;

impl ValidationRule for SmallTypeStorageCapabilitiesRule {
    fn name(&self) -> &'static str {
        "small-type-storage-capabilities"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        !ctx.declared_capabilities.contains(&Capability::Shader)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::Variable && inst.class.opcode != Op::UntypedVariableKHR {
                continue;
            }
            let Some(raw_type) = inst.result_type else {
                continue;
            };
            let Ok(ptr_type_id) = TypeId::try_from(raw_type) else {
                continue;
            };
            let Some(ptr_type_inst) = ResultId::try_from(u32::from(ptr_type_id))
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid))
            else {
                continue;
            };
            if ptr_type_inst.class.opcode != Op::TypePointer {
                continue;
            }
            let storage_class = match ptr_type_inst.operands.first() {
                Some(rspirv::dr::Operand::StorageClass(class)) => *class,
                _ => continue,
            };
            let pointee = match ptr_type_inst.operands.get(1) {
                Some(rspirv::dr::Operand::IdRef(raw)) => match TypeId::try_from(*raw) {
                    Ok(id) => id,
                    Err(_) => continue,
                },
                _ => continue,
            };

            let contains_int = |width: u32| -> bool {
                let mut visited = HashSet::new();
                contains_sized_int_or_float(
                    pointee,
                    Op::TypeInt,
                    width,
                    ctx.definitions,
                    &mut visited,
                )
            };
            let contains_float = |width: u32| -> bool {
                let mut visited = HashSet::new();
                contains_sized_int_or_float(
                    pointee,
                    Op::TypeFloat,
                    width,
                    ctx.definitions,
                    &mut visited,
                )
            };

            for bit_width in [8u32, 16u32] {
                let has_width =
                    contains_int(bit_width) || (bit_width == 16 && contains_float(bit_width));
                if !has_width {
                    continue;
                }

                // Some storage classes never allow 8-bit or 16-bit types, regardless of Int8/Int16/Float16 capability.
                // These are always rejected first.
                let never_allows_small = matches!(
                    storage_class,
                    StorageClass::Input | StorageClass::Output | StorageClass::UniformConstant
                );

                if bit_width == 8 && never_allows_small {
                    return Err(ValidationError::SmallTypeDisallowedInStorageClass {
                        bit_width,
                        storage_class,
                    }.into());
                }
                if bit_width == 16 && storage_class == StorageClass::UniformConstant {
                    return Err(ValidationError::SmallTypeDisallowedInStorageClass {
                        bit_width,
                        storage_class,
                    }.into());
                }

                let require_capability = |cap: Capability| -> ValidationResult {
                    if ctx.declared_capabilities.contains(&cap) {
                        Ok(())
                    } else {
                        Err(ValidationError::SmallTypeMissingCapability {
                            bit_width,
                            storage_class,
                            required_capability: cap,
                        }.into())
                    }
                };

                match storage_class {
                    StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer => {
                        let required = if bit_width == 8 {
                            Capability::StorageBuffer8BitAccess
                        } else {
                            Capability::StorageBuffer16BitAccess
                        };
                        require_capability(required)?
                    }
                    StorageClass::Uniform => {
                        let (primary, fallback) = if bit_width == 8 {
                            (
                                Capability::UniformAndStorageBuffer8BitAccess,
                                Capability::StorageBuffer8BitAccess,
                            )
                        } else {
                            (
                                Capability::UniformAndStorageBuffer16BitAccess,
                                Capability::StorageBuffer16BitAccess,
                            )
                        };
                        if ctx.declared_capabilities.contains(&primary) {
                            continue;
                        }
                        if ctx.declared_capabilities.contains(&fallback)
                            && has_decoration(ctx, u32::from(pointee), Decoration::BufferBlock)
                        {
                            continue;
                        }
                        return Err(ValidationError::SmallTypeMissingCapability {
                            bit_width,
                            storage_class,
                            required_capability: primary,
                        }.into());
                    }
                    StorageClass::PushConstant => {
                        let required = if bit_width == 8 {
                            Capability::StoragePushConstant8
                        } else {
                            Capability::StoragePushConstant16
                        };
                        require_capability(required)?
                    }
                    StorageClass::Input | StorageClass::Output => {
                        // 8-bit already rejected above; only 16-bit can reach here
                        require_capability(Capability::StorageInputOutput16)?
                    }
                    StorageClass::Workgroup => {
                        let required = if bit_width == 8 {
                            Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR
                        } else {
                            Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR
                        };
                        require_capability(required)?
                    }
                    // For other storage classes (Function, Private, etc.), having the base
                    // capability (Int8/Int16/Float16) is sufficient - no additional storage
                    // capability is required.
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn int_bit_width(
    type_id: TypeId,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let Ok(result_id) = ResultId::try_from(u32::from(type_id)) else {
        return None;
    };
    let inst = definitions.get(&result_id)?;
    match inst.class.opcode {
        Op::TypeInt => inst.operands.first().and_then(|op| match op {
            rspirv::dr::Operand::LiteralBit32(width) => Some(*width),
            rspirv::dr::Operand::LiteralBit64(width) => Some(*width as u32),
            _ => None,
        }),
        Op::TypeVector => {
            let component = match inst.operands.first() {
                Some(rspirv::dr::Operand::IdRef(raw)) => TypeId::try_from(*raw).ok()?,
                _ => return None,
            };
            int_bit_width(component, definitions)
        }
        _ => None,
    }
}

fn has_decoration(ctx: &ValidationContext<'_>, target: u32, decoration: Decoration) -> bool {
    ctx.module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && matches!(
                (inst.operands.first(), inst.operands.get(1)),
                (
                    Some(rspirv::dr::Operand::IdRef(id)),
                    Some(rspirv::dr::Operand::Decoration(dec))
                ) if *id == target && *dec == decoration
            )
    })
}

fn contains_sized_int_or_float(
    type_id: TypeId,
    target_opcode: Op,
    width: u32,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
    visited: &mut HashSet<TypeId>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }
    let Ok(result_id) = ResultId::try_from(u32::from(type_id)) else {
        return false;
    };
    let Some(inst) = definitions.get(&result_id) else {
        return false;
    };
    match inst.class.opcode {
        Op::TypeInt if target_opcode == Op::TypeInt => inst.operands.iter().any(|op| match op {
            rspirv::dr::Operand::LiteralBit32(bits) => *bits == width,
            rspirv::dr::Operand::LiteralBit64(bits) => *bits as u32 == width,
            _ => false,
        }),
        Op::TypeFloat if target_opcode == Op::TypeFloat => {
            inst.operands.iter().any(|op| match op {
                rspirv::dr::Operand::LiteralBit32(bits) => *bits == width,
                rspirv::dr::Operand::LiteralBit64(bits) => *bits as u32 == width,
                _ => false,
            })
        }
        Op::TypeVector
        | Op::TypeMatrix
        | Op::TypeArray
        | Op::TypeRuntimeArray
        | Op::TypeStruct
        | Op::TypePointer => inst.operands.iter().any(|op| {
            if let rspirv::dr::Operand::IdRef(raw) = op {
                if let Ok(child) = TypeId::try_from(*raw) {
                    return contains_sized_int_or_float(
                        child,
                        target_opcode,
                        width,
                        definitions,
                        visited,
                    );
                }
            }
            false
        }),
        _ => false,
    }
}

// ============================================================================
// Vulkan Descriptor Binding Rule
// ============================================================================

/// Validates that variables in Uniform, UniformConstant, and StorageBuffer storage classes
/// have DescriptorSet and Binding decorations in Vulkan.
///
/// From Vulkan spec (VUID-06677):
/// These variables must have DescriptorSet and Binding decorations specified.
pub struct VulkanDescriptorBindingRule;

impl ValidationRule for VulkanDescriptorBindingRule {
    fn name(&self) -> &'static str {
        "vulkan-descriptor-binding"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        !is_vulkan_env(ctx.env) || ctx.options.before_hlsl_legalization
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        use crate::validation::types::Id;

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::Variable && inst.class.opcode != Op::UntypedVariableKHR {
                continue;
            }

            let Some(var_id) = inst.result_id else {
                continue;
            };

            let storage_class = match inst.operands.first() {
                Some(rspirv::dr::Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };

            // Check if this storage class requires DescriptorSet/Binding
            let requires_descriptor = matches!(
                storage_class,
                StorageClass::Uniform | StorageClass::UniformConstant | StorageClass::StorageBuffer
            );

            if !requires_descriptor {
                continue;
            }

            // Skip if variable is not used by any entry point (HLSL legalization case)
            // We check if the variable appears in any entry point interface
            let is_referenced = ctx.module.entry_points.iter().any(|ep| {
                ep.operands.iter().skip(2).any(|op| {
                    matches!(op, rspirv::dr::Operand::IdRef(id) if *id == var_id)
                })
            });

            if !is_referenced && ctx.options.before_hlsl_legalization {
                continue;
            }

            // Check for DescriptorSet decoration
            let has_descriptor_set = has_decoration(ctx, var_id, Decoration::DescriptorSet);
            if !has_descriptor_set {
                return Err(ValidationError::MissingDescriptorSetDecoration {
                    variable: Id::try_from(var_id).unwrap_or_else(|_| Id::try_from(1u32).unwrap()),
                }.into());
            }

            // Check for Binding decoration
            let has_binding = has_decoration(ctx, var_id, Decoration::Binding);
            if !has_binding {
                return Err(ValidationError::MissingBindingDecoration {
                    variable: Id::try_from(var_id).unwrap_or_else(|_| Id::try_from(1u32).unwrap()),
                }.into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// Vulkan Push Constant Block Rule
// ============================================================================

/// Validates that there is at most one PushConstant interface per entry point in Vulkan.
///
/// From Vulkan spec (VUID-06674):
/// There must be no more than one push constant block statically used per shader entry point.
pub struct VulkanPushConstantBlockRule;

impl ValidationRule for VulkanPushConstantBlockRule {
    fn name(&self) -> &'static str {
        "vulkan-push-constant-block"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        !is_vulkan_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        use crate::validation::types::Id;

        // Collect all PushConstant variables
        let push_constant_vars: HashSet<u32> = ctx
            .module
            .types_global_values
            .iter()
            .filter(|inst| {
                (inst.class.opcode == Op::Variable || inst.class.opcode == Op::UntypedVariableKHR)
                    && matches!(
                        inst.operands.first(),
                        Some(rspirv::dr::Operand::StorageClass(StorageClass::PushConstant))
                    )
            })
            .filter_map(|inst| inst.result_id)
            .collect();

        if push_constant_vars.len() <= 1 {
            return Ok(());
        }

        // Check each entry point for multiple push constants
        for ep in &ctx.module.entry_points {
            let ep_id = ep.operands.get(1).and_then(|op| {
                if let rspirv::dr::Operand::IdRef(id) = op {
                    Some(*id)
                } else {
                    None
                }
            });

            let interface_push_constants: Vec<u32> = ep
                .operands
                .iter()
                .skip(3) // Skip execution model, entry point id, name
                .filter_map(|op| {
                    if let rspirv::dr::Operand::IdRef(id) = op {
                        if push_constant_vars.contains(id) {
                            return Some(*id);
                        }
                    }
                    None
                })
                .collect();

            if interface_push_constants.len() > 1 {
                return Err(ValidationError::EntryPointInterfaceStorageClassDuplicate {
                    entry_point: ep_id
                        .and_then(|id| Id::try_from(id).ok())
                        .unwrap_or_else(|| Id::try_from(1u32).unwrap()),
                    storage_class: StorageClass::PushConstant,
                }.into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// Vulkan Buffer Block Decorations Rule
// ============================================================================

/// Validates that buffer variables have required Block/BufferBlock decorations in Vulkan.
///
/// From Vulkan spec (VUID-06675, VUID-06676):
/// - PushConstant and StorageBuffer must be decorated with Block
/// - Uniform must be decorated with Block or BufferBlock
/// - StorageBuffer must NOT be decorated with BufferBlock
pub struct VulkanBufferBlockDecorationsRule;

impl ValidationRule for VulkanBufferBlockDecorationsRule {
    fn name(&self) -> &'static str {
        "vulkan-buffer-block-decorations"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        !is_vulkan_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::Variable && inst.class.opcode != Op::UntypedVariableKHR {
                continue;
            }

            let storage_class = match inst.operands.first() {
                Some(rspirv::dr::Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };

            // Get the pointee struct type
            let pointee_type_id = get_variable_pointee_type(inst, ctx);
            let Some(struct_id) = pointee_type_id else {
                continue;
            };

            // Skip if pointee is not a struct (arrays of structs handled separately)
            let pointee_opcode = ResultId::try_from(struct_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            // If it's an array, get the element type
            let final_struct_id = if matches!(pointee_opcode, Some(Op::TypeArray) | Some(Op::TypeRuntimeArray)) {
                get_array_element_struct(struct_id, ctx)
            } else if pointee_opcode == Some(Op::TypeStruct) {
                Some(struct_id)
            } else {
                None
            };

            let Some(struct_id) = final_struct_id else {
                continue;
            };

            let has_block = has_decoration(ctx, struct_id, Decoration::Block);
            let has_buffer_block = has_decoration(ctx, struct_id, Decoration::BufferBlock);

            match storage_class {
                StorageClass::PushConstant => {
                    if !has_block {
                        return Err(ValidationError::VulkanBufferMissingBlockDecoration {
                            storage_class,
                            struct_id,
                        }.into());
                    }
                }
                StorageClass::StorageBuffer => {
                    if has_buffer_block {
                        return Err(ValidationError::VulkanStorageBufferHasBufferBlock {
                            struct_id,
                        }.into());
                    }
                    if !has_block {
                        return Err(ValidationError::VulkanBufferMissingBlockDecoration {
                            storage_class,
                            struct_id,
                        }.into());
                    }
                }
                StorageClass::Uniform => {
                    if !has_block && !has_buffer_block {
                        return Err(ValidationError::VulkanUniformMissingBlockDecoration {
                            struct_id,
                        }.into());
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Gets the pointee type ID from a variable instruction.
fn get_variable_pointee_type(inst: &rspirv::dr::Instruction, ctx: &ValidationContext<'_>) -> Option<u32> {
    // For untyped variables, the data type is in operand 1
    if inst.class.opcode == Op::UntypedVariableKHR {
        return inst.operands.get(1).and_then(|op| {
            if let rspirv::dr::Operand::IdRef(id) = op {
                Some(*id)
            } else {
                None
            }
        });
    }

    // For typed variables, get from pointer type
    let ptr_type_id = inst.result_type?;
    let ptr_type_inst = ResultId::try_from(ptr_type_id)
        .ok()
        .and_then(|rid| ctx.definitions.get(&rid))?;

    if ptr_type_inst.class.opcode != Op::TypePointer {
        return None;
    }

    ptr_type_inst.operands.get(1).and_then(|op| {
        if let rspirv::dr::Operand::IdRef(id) = op {
            Some(*id)
        } else {
            None
        }
    })
}

/// Gets the struct element type from an array type.
fn get_array_element_struct(array_type_id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let array_inst = ResultId::try_from(array_type_id)
        .ok()
        .and_then(|rid| ctx.definitions.get(&rid))?;

    let elem_id = array_inst.operands.first().and_then(|op| {
        if let rspirv::dr::Operand::IdRef(id) = op {
            Some(*id)
        } else {
            None
        }
    })?;

    let elem_opcode = ResultId::try_from(elem_id)
        .ok()
        .and_then(|rid| ctx.opcodes.get(&rid))
        .copied()?;

    if elem_opcode == Op::TypeStruct {
        Some(elem_id)
    } else {
        None
    }
}

// ============================================================================
// Block Layout Decorations Rule
// ============================================================================

/// Validates that Block/BufferBlock structs have required layout decorations.
///
/// Block and BufferBlock decorated structs must have:
/// - ArrayStride on all OpTypeArray members
/// - MatrixStride on all OpTypeMatrix members
/// - RowMajor or ColMajor on all OpTypeMatrix members
pub struct BlockLayoutDecorationsRule;

impl ValidationRule for BlockLayoutDecorationsRule {
    fn name(&self) -> &'static str {
        "block-layout-decorations"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Find all Block and BufferBlock decorated structs
        for inst in &ctx.module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }

            let struct_id = match inst.operands.first() {
                Some(rspirv::dr::Operand::IdRef(id)) => *id,
                _ => continue,
            };

            let decoration = match inst.operands.get(1) {
                Some(rspirv::dr::Operand::Decoration(dec)) => *dec,
                _ => continue,
            };

            let decoration_type = match decoration {
                Decoration::Block => "Block",
                Decoration::BufferBlock => "BufferBlock",
                _ => continue,
            };

            // Verify this is actually a struct
            let struct_opcode = ResultId::try_from(struct_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            if struct_opcode != Some(Op::TypeStruct) {
                continue;
            }

            // Check for required decorations recursively
            check_struct_layout_decorations(struct_id, decoration_type, ctx)?;
        }

        Ok(())
    }
}

/// Recursively checks a struct for required layout decorations.
fn check_struct_layout_decorations(
    struct_id: u32,
    decoration_type: &'static str,
    ctx: &ValidationContext<'_>,
) -> ValidationResult {
    let struct_inst = ResultId::try_from(struct_id)
        .ok()
        .and_then(|rid| ctx.definitions.get(&rid));

    let Some(struct_inst) = struct_inst else {
        return Ok(());
    };

    if struct_inst.class.opcode != Op::TypeStruct {
        return Ok(());
    }

    // Check each member
    for (member_idx, operand) in struct_inst.operands.iter().enumerate() {
        let member_type_id = match operand {
            rspirv::dr::Operand::IdRef(id) => *id,
            _ => continue,
        };

        // Get the actual type, unwrapping arrays for matrix checks
        let member_opcode = ResultId::try_from(member_type_id)
            .ok()
            .and_then(|rid| ctx.opcodes.get(&rid))
            .copied();

        match member_opcode {
            Some(Op::TypeArray) | Some(Op::TypeRuntimeArray) => {
                // Check ArrayStride decoration on the array type
                if !has_decoration(ctx, member_type_id, Decoration::ArrayStride) {
                    return Err(ValidationError::BlockMissingArrayStride {
                        struct_id,
                        decoration_type,
                    }.into());
                }

                // For matrix members inside arrays, check the element type
                let elem_type_id = get_array_element_type(member_type_id, ctx);
                if let Some(elem_id) = elem_type_id {
                    check_matrix_decorations(struct_id, member_idx, elem_id, decoration_type, ctx)?;
                    // Recursively check nested structs
                    let elem_opcode = ResultId::try_from(elem_id)
                        .ok()
                        .and_then(|rid| ctx.opcodes.get(&rid))
                        .copied();
                    if elem_opcode == Some(Op::TypeStruct) {
                        check_struct_layout_decorations(elem_id, decoration_type, ctx)?;
                    }
                }
            }
            Some(Op::TypeMatrix) => {
                check_matrix_decorations(struct_id, member_idx, member_type_id, decoration_type, ctx)?;
            }
            Some(Op::TypeStruct) => {
                // Recursively check nested structs
                check_struct_layout_decorations(member_type_id, decoration_type, ctx)?;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Gets the element type of an array (unwrapping nested arrays).
fn get_array_element_type(array_type_id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let mut current_id = array_type_id;

    loop {
        let inst = ResultId::try_from(current_id)
            .ok()
            .and_then(|rid| ctx.definitions.get(&rid))?;

        match inst.class.opcode {
            Op::TypeArray | Op::TypeRuntimeArray => {
                current_id = inst.operands.first().and_then(|op| {
                    if let rspirv::dr::Operand::IdRef(id) = op {
                        Some(*id)
                    } else {
                        None
                    }
                })?;
            }
            _ => return Some(current_id),
        }
    }
}

/// Checks matrix decorations (MatrixStride and RowMajor/ColMajor).
fn check_matrix_decorations(
    struct_id: u32,
    member_idx: usize,
    type_id: u32,
    decoration_type: &'static str,
    ctx: &ValidationContext<'_>,
) -> ValidationResult {
    let type_opcode = ResultId::try_from(type_id)
        .ok()
        .and_then(|rid| ctx.opcodes.get(&rid))
        .copied();

    if type_opcode != Some(Op::TypeMatrix) {
        return Ok(());
    }

    // Check MatrixStride on the type itself or as a member decoration
    let has_matrix_stride = has_decoration(ctx, type_id, Decoration::MatrixStride)
        || has_member_decoration(ctx, struct_id, member_idx as u32, Decoration::MatrixStride);

    if !has_matrix_stride {
        return Err(ValidationError::BlockMissingMatrixStride {
            struct_id,
            decoration_type,
        }.into());
    }

    // Check RowMajor or ColMajor on the type itself or as a member decoration
    let has_row_major = has_decoration(ctx, type_id, Decoration::RowMajor)
        || has_member_decoration(ctx, struct_id, member_idx as u32, Decoration::RowMajor);
    let has_col_major = has_decoration(ctx, type_id, Decoration::ColMajor)
        || has_member_decoration(ctx, struct_id, member_idx as u32, Decoration::ColMajor);

    if !has_row_major && !has_col_major {
        return Err(ValidationError::BlockMissingMatrixOrder {
            struct_id,
            decoration_type,
        }.into());
    }

    Ok(())
}

/// Checks if a struct member has a specific decoration.
fn has_member_decoration(
    ctx: &ValidationContext<'_>,
    struct_id: u32,
    member_idx: u32,
    decoration: Decoration,
) -> bool {
    ctx.module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::MemberDecorate
            && matches!(
                (&inst.operands[..], inst.operands.get(2)),
                (
                    [rspirv::dr::Operand::IdRef(id), rspirv::dr::Operand::LiteralBit32(idx), ..],
                    Some(rspirv::dr::Operand::Decoration(dec))
                ) if *id == struct_id && *idx == member_idx && *dec == decoration
            )
    })
}

// ============================================================================
// All Vulkan rules
// ============================================================================

/// Returns all Vulkan-specific validation rules.
pub fn all_vulkan_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &OffsetTextureOperandRule,
        &VulkanBitwiseWidthsRule,
        &SmallTypeStorageCapabilitiesRule,
        &VulkanDescriptorBindingRule,
        &VulkanPushConstantBlockRule,
        &VulkanBufferBlockDecorationsRule,
        &BlockLayoutDecorationsRule,
    ]
}
