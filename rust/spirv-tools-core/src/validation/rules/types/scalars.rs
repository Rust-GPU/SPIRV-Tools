//! Scalar type validation rules.
//!
//! This module validates SPIR-V scalar type requirements:
//! - OpTypeInt capability requirements (Int8, Int16, Int64)
//! - OpTypeFloat capability requirements (Float16, Float64)

use std::collections::HashSet;

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, FPEncoding, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::types::TypeId;

// ============================================================================
// OpTypeInt Validation Rule
// ============================================================================

/// Validates OpTypeInt capability and bit width requirements.
///
/// Checks:
/// - 8-bit integers require Int8 capability (or storage buffer access capabilities)
/// - 16-bit integers require Int16 capability (or storage buffer access capabilities)
/// - 64-bit integers require Int64 capability
/// - Valid bit widths are 8, 16, 32, 64
/// - Signedness must be 0 or 1
/// - Kernel capability requires signedness 0
pub struct TypeIntRule;

/// Checks if any capability that enables 8-bit integers is declared.
fn has_8bit_capability(caps: &HashSet<Capability>) -> bool {
    caps.contains(&Capability::Int8)
        || caps.contains(&Capability::StorageBuffer8BitAccess)
        || caps.contains(&Capability::UniformAndStorageBuffer8BitAccess)
        || caps.contains(&Capability::StoragePushConstant8)
}

/// Checks if any capability that enables 16-bit integers is declared.
fn has_16bit_int_capability(caps: &HashSet<Capability>) -> bool {
    caps.contains(&Capability::Int16)
        || caps.contains(&Capability::StorageBuffer16BitAccess)
        || caps.contains(&Capability::UniformAndStorageBuffer16BitAccess)
        || caps.contains(&Capability::StoragePushConstant16)
        || caps.contains(&Capability::StorageInputOutput16)
}

impl ValidationRule for TypeIntRule {
    fn name(&self) -> &'static str {
        "type-int"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeInt {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get bit width (operand 0)
            let width = match inst.operands.first() {
                Some(Operand::LiteralBit32(w)) => *w,
                _ => continue,
            };

            // Validate bit width and capability requirements
            match width {
                8 => {
                    if !has_8bit_capability(ctx.declared_capabilities) {
                        return Err(ValidationError::TypeIntRequiresInt8Capability { type_id }.into());
                    }
                }
                16 => {
                    if !has_16bit_int_capability(ctx.declared_capabilities) {
                        return Err(ValidationError::TypeIntRequiresInt16Capability { type_id }.into());
                    }
                }
                32 => {
                    // 32-bit is always valid
                }
                64 => {
                    if !ctx.declared_capabilities.contains(&Capability::Int64) {
                        return Err(ValidationError::TypeIntRequiresInt64Capability { type_id }.into());
                    }
                }
                _ => {
                    return Err(ValidationError::TypeIntInvalidBitWidth { type_id, width }.into());
                }
            }

            // Get signedness (operand 1)
            let signedness = match inst.operands.get(1) {
                Some(Operand::LiteralBit32(s)) => *s,
                _ => continue,
            };

            // Validate signedness value
            if signedness > 1 {
                return Err(ValidationError::TypeIntInvalidSignedness { type_id, signedness }.into());
            }

            // Kernel capability requires signedness 0
            if ctx.declared_capabilities.contains(&Capability::Kernel) && signedness != 0 {
                return Err(ValidationError::TypeIntKernelRequiresUnsigned { type_id }.into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeFloat Validation Rule
// ============================================================================

/// Checks if any capability that enables 16-bit floats is declared.
fn has_16bit_float_capability(caps: &HashSet<Capability>) -> bool {
    caps.contains(&Capability::Float16)
        || caps.contains(&Capability::Float16Buffer)
        || caps.contains(&Capability::StorageBuffer16BitAccess)
        || caps.contains(&Capability::UniformAndStorageBuffer16BitAccess)
        || caps.contains(&Capability::StoragePushConstant16)
        || caps.contains(&Capability::StorageInputOutput16)
}

/// Validates OpTypeFloat capability and bit width requirements.
///
/// Checks:
/// - 8-bit floats require Float8EXT capability and FPEncoding operand
/// - 16-bit floats require Float16, Float16Buffer, or storage access capabilities
/// - 64-bit floats require Float64 capability
/// - Valid bit widths are 8, 16, 32, 64
pub struct TypeFloatRule;

impl ValidationRule for TypeFloatRule {
    fn name(&self) -> &'static str {
        "type-float"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeFloat {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get bit width (operand 0)
            let width = match inst.operands.first() {
                Some(Operand::LiteralBit32(w)) => *w,
                _ => continue,
            };

            // Check for encoding operand (optional second operand)
            let encoding = inst.operands.get(1).and_then(|op| {
                if let Operand::FPEncoding(enc) = op {
                    Some(*enc)
                } else {
                    None
                }
            });

            // Validate bit width and capability requirements
            match width {
                8 => {
                    // Float8EXT capability required
                    if !ctx.declared_capabilities.contains(&Capability::Float8EXT) {
                        return Err(ValidationError::TypeFloatRequiresFloat8Capability {
                            type_id,
                        }.into());
                    }
                    // 8-bit float requires encoding
                    let Some(enc) = encoding else {
                        return Err(ValidationError::TypeFloat8RequiresEncoding { type_id }.into());
                    };
                    // Only Float8E4M3EXT and Float8E5M2EXT are supported
                    if !matches!(
                        enc,
                        FPEncoding::Float8E4M3EXT | FPEncoding::Float8E5M2EXT
                    ) {
                        return Err(ValidationError::TypeFloat8UnsupportedEncoding {
                            type_id,
                            encoding: enc,
                        }.into());
                    }
                }
                16 => {
                    // If there's an encoding, it's valid (e.g., BFloat16)
                    // Otherwise, Float16, Float16Buffer, or storage access capability required
                    if encoding.is_none()
                        && !has_16bit_float_capability(ctx.declared_capabilities)
                    {
                        return Err(ValidationError::TypeFloatRequiresFloat16Capability {
                            type_id,
                        }.into());
                    }
                }
                32 => {
                    // 32-bit is always valid
                }
                64 => {
                    if !ctx.declared_capabilities.contains(&Capability::Float64) {
                        return Err(ValidationError::TypeFloatRequiresFloat64Capability {
                            type_id,
                        }.into());
                    }
                }
                _ => {
                    return Err(ValidationError::TypeFloatInvalidBitWidth { type_id, width }.into());
                }
            }
        }

        Ok(())
    }
}

/// Returns all scalar type validation rules.
pub fn all_scalar_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&TypeIntRule, &TypeFloatRule]
}
