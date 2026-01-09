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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Vulkan Bitwise Widths Rule
// ============================================================================

/// Validates that bitwise operations use 32-bit integers in Vulkan.
pub struct VulkanBitwiseWidthsRule;

impl ValidationRule for VulkanBitwiseWidthsRule {
    fn name(&self) -> &'static str {
        "vulkan-bitwise-widths"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        ctx.options.allow_vulkan_32_bit_bitwise || !is_vulkan_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let bitwise_opcodes = [
            Op::ShiftRightLogical,
            Op::ShiftRightArithmetic,
            Op::ShiftLeftLogical,
            Op::BitwiseOr,
            Op::BitwiseXor,
            Op::BitwiseAnd,
            Op::Not,
        ];

        for inst in ctx.module.all_inst_iter() {
            if !bitwise_opcodes.contains(&inst.class.opcode) {
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
                    });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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

                let require_capability = |cap: Capability| -> Result<(), ValidationError> {
                    if ctx.declared_capabilities.contains(&cap) {
                        Ok(())
                    } else {
                        Err(ValidationError::SmallTypeMissingCapability {
                            bit_width,
                            storage_class,
                            required_capability: cap,
                        })
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
                        });
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
                        if bit_width == 16 {
                            require_capability(Capability::StorageInputOutput16)?
                        } else {
                            return Err(ValidationError::SmallTypeDisallowedInStorageClass {
                                bit_width,
                                storage_class,
                            });
                        }
                    }
                    StorageClass::Workgroup => {
                        let required = if bit_width == 8 {
                            Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR
                        } else {
                            Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR
                        };
                        require_capability(required)?
                    }
                    _ => {
                        return Err(ValidationError::SmallTypeDisallowedInStorageClass {
                            bit_width,
                            storage_class,
                        })
                    }
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
// All Vulkan rules
// ============================================================================

/// Returns all Vulkan-specific validation rules.
pub fn all_vulkan_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &OffsetTextureOperandRule,
        &VulkanBitwiseWidthsRule,
        &SmallTypeStorageCapabilitiesRule,
    ]
}
